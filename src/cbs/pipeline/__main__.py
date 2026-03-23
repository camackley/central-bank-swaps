"""CLI entry point — ``python -m cbs.pipeline``."""

from __future__ import annotations

import argparse
import logging
import sys
import uuid
from pathlib import Path
from uuid import UUID

import duckdb
from dotenv import load_dotenv

from cbs.config.banks import load_bank_config
from cbs.config.tracing import configure_tracing
from cbs.db.run_manager import RunManager
from cbs.db.schema import init_db, init_main
from cbs.llm.claude_code_model import ClaudeRateLimitError
from cbs.llm.provider import get_llm
from cbs.pipeline.backfill import BackfillOrchestrator
from cbs.pipeline.bank_processor import DefaultBankProcessor
from cbs.pipeline.incremental import IncrementalOrchestrator
from cbs.pipeline.orchestrator import Orchestrator
from cbs.scraper.browser import BrowserAdapter

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the pipeline CLI."""
    parser = argparse.ArgumentParser(
        prog="python -m cbs.pipeline",
        description="Run the central bank swaps scraping pipeline.",
    )
    parser.add_argument(
        "--config",
        default="config/banks.yaml",
        help="Path to banks.yaml config file (default: config/banks.yaml)",
    )
    parser.add_argument(
        "--banks",
        nargs="+",
        default=None,
        metavar="BANK",
        help=(
            "Filter to specific banks by name or country "
            "(case-insensitive partial match). "
            "E.g.: --banks banrep australia japan"
        ),
    )
    parser.add_argument(
        "--db",
        default="central_bank_swaps.duckdb",
        help="Path to DuckDB database file (default: central_bank_swaps.duckdb)",
    )
    parser.add_argument(
        "--provider",
        default="anthropic",
        help=(
            "LLM provider: claude-code, anthropic, openai, google-genai "
            "(default: anthropic)"
        ),
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-6",
        help="LLM model name (default: claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=5,
        help="Max listing pages to paginate per bank (default: 5)",
    )
    parser.add_argument(
        "--classify-model",
        default=None,
        help="Override LLM model for classification stage (e.g. claude-haiku-3-5)",
    )
    parser.add_argument(
        "--extract-model",
        default=None,
        help="Override LLM model for extraction stage (e.g. o3)",
    )
    parser.add_argument(
        "--translate-model",
        default=None,
        help="Override LLM model for translation stage (e.g. gpt-4o)",
    )
    parser.add_argument(
        "--mode",
        choices=["backfill", "incremental"],
        default="backfill",
        help="Run mode: backfill (full history) or incremental (new only)",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume an existing run by UUID",
    )
    parser.add_argument(
        "--schedule",
        type=int,
        default=None,
        metavar="DAYS",
        help="Run on a recurring schedule every N days (blocks forever)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of banks to process in parallel (default: 1)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the pipeline in backfill or incremental mode."""
    load_dotenv()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    configure_tracing()

    parser = build_parser()
    args = parser.parse_args(argv)

    # Load config
    banks_config = load_bank_config(Path(args.config))
    logger.info("Loaded %d banks from %s", len(banks_config.banks), args.config)

    # Optional bank filter
    if args.banks:
        search_terms = [t.lower() for t in args.banks]
        banks_config.banks = [
            b
            for b in banks_config.banks
            if any(
                term in b.name.lower() or term in b.country.lower()
                for term in search_terms
            )
        ]
        if not banks_config.banks:
            logger.error("No banks matched filter: %s", args.banks)
            sys.exit(1)
        logger.info(
            "Filtered to %d banks: %s",
            len(banks_config.banks),
            [b.name for b in banks_config.banks],
        )

    # Master DB: single file that catalogs all runs
    master_db_path = Path(args.db)
    master_conn = duckdb.connect(str(master_db_path))
    init_main(master_conn)  # ensure main.runs catalog exists

    # Resolve or create the per-run DB file
    runs_dir = master_db_path.parent / "runs"
    runs_dir.mkdir(exist_ok=True)

    resume_id = UUID(args.resume) if args.resume else None
    if resume_id is not None:
        row = master_conn.execute(
            "SELECT schema_name FROM main.runs WHERE id = ?", [str(resume_id)]
        ).fetchone()
        if row is None:
            logger.error("Resume run %s not found in catalog", resume_id)
            sys.exit(1)
        run_file = row[0]  # stored as the file path
        run_id: UUID = resume_id
        logger.info("Resuming run %s → %s", run_id, run_file)
    else:
        run_id = uuid.uuid4()
        run_file = str(runs_dir / f"run_{run_id.hex[:8]}.duckdb")
        master_conn.execute(
            "INSERT INTO main.runs (id, schema_name, run_type) VALUES (?, ?, ?)",
            [str(run_id), run_file, args.mode],
        )
        logger.info("Starting new run %s → %s", run_id, run_file)

    master_conn.close()

    # Per-run connection — plain main schema, no schema switching needed
    conn = duckdb.connect(run_file)
    init_db(conn)

    # Set up LLMs
    llm = get_llm(args.provider, args.model)
    classify_llm = (
        get_llm(args.provider, args.classify_model) if args.classify_model else None
    )
    extract_llm = (
        get_llm(args.provider, args.extract_model) if args.extract_model else None
    )
    translate_llm = (
        get_llm(args.provider, args.translate_model) if args.translate_model else None
    )

    # Wire pipeline
    run_manager = RunManager(conn)
    bank_names = [b.name for b in banks_config.banks]
    if resume_id is None:
        run_manager.create_run(args.mode, bank_names, run_id=run_id)
    concurrency = args.concurrency

    try:
        if concurrency > 1:
            from cbs.pipeline.worker_factory import create_worker
            from cbs.scraper.instance_pool import PinchTabInstancePool

            with PinchTabInstancePool(size=concurrency) as pool:
                workers = [
                    create_worker(
                        db_path=run_file,
                        instance_port=inst.port,
                        llm=llm,
                        classify_llm=classify_llm,
                        extract_llm=extract_llm,
                        translate_llm=translate_llm,
                        max_pages=args.max_pages,
                    )
                    for inst in pool.instances
                ]
                try:
                    processors = [w.processor for w in workers]

                    runner: BackfillOrchestrator | IncrementalOrchestrator
                    if args.mode == "incremental":
                        runner = IncrementalOrchestrator(
                            conn=conn,
                            run_manager=run_manager,
                            bank_processor=processors,
                            banks_config=banks_config,
                        )
                    else:
                        runner = BackfillOrchestrator(
                            conn=conn,
                            run_manager=run_manager,
                            bank_processor=processors,
                            banks_config=banks_config,
                        )

                    summary = runner.run(resume_run_id=run_id)
                finally:
                    for w in workers:
                        w.close()
        else:
            with BrowserAdapter() as browser:
                orchestrator = Orchestrator(
                    conn=conn,
                    llm=llm,
                    browser=browser,
                    classify_llm=classify_llm,
                    extract_llm=extract_llm,
                    translate_llm=translate_llm,
                )
                processor = DefaultBankProcessor(
                    orchestrator=orchestrator,
                    browser=browser,
                    llm=llm,
                    max_pages=args.max_pages,
                )

                runner = (
                    IncrementalOrchestrator(
                        conn=conn,
                        run_manager=run_manager,
                        bank_processor=processor,
                        banks_config=banks_config,
                    )
                    if args.mode == "incremental"
                    else BackfillOrchestrator(
                        conn=conn,
                        run_manager=run_manager,
                        bank_processor=processor,
                        banks_config=banks_config,
                    )
                )

                if args.schedule is not None:
                    from cbs.scheduler import PipelineScheduler

                    scheduler = PipelineScheduler(
                        run_fn=lambda: runner.run(),
                        interval_days=args.schedule,
                    )
                    scheduler.start()  # blocks forever
                    return

                summary = runner.run(resume_run_id=run_id)

    except ClaudeRateLimitError as exc:
        conn.close()
        logger.error("Claude Code usage limit reached: %s", exc)
        logger.info(
            "Progress saved. Resume when limits reset with:\n"
            "  uv run python -m cbs.pipeline --resume %s",
            run_id,
        )
        sys.exit(2)

    conn.close()

    logger.info(
        "Pipeline complete: %d/%d banks, %d PRs, %d swaps, %d errors",
        summary.banks_succeeded,
        summary.banks_attempted,
        summary.press_releases_found,
        summary.swaps_extracted,
        len(summary.errors),
    )

    if summary.errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
