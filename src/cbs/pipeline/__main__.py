"""CLI entry point — ``python -m cbs.pipeline``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from uuid import UUID

import duckdb
from dotenv import load_dotenv

from cbs.config.banks import load_bank_config
from cbs.config.tracing import configure_tracing
from cbs.db.run_manager import RunManager
from cbs.db.schema import init_db
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
        "--db",
        default="central_bank_swaps.duckdb",
        help="Path to DuckDB database file (default: central_bank_swaps.duckdb)",
    )
    parser.add_argument(
        "--provider",
        default="anthropic",
        help="LLM provider: anthropic, openai, google-genai (default: anthropic)",
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-20250514",
        help="LLM model name (default: claude-sonnet-4-20250514)",
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
        help="Override LLM model for extraction stage (e.g. claude-sonnet-4-20250514)",
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

    # Connect to DuckDB
    conn = duckdb.connect(args.db)
    init_db(conn)

    # Set up LLMs
    llm = get_llm(args.provider, args.model)
    classify_llm = (
        get_llm(args.provider, args.classify_model) if args.classify_model else None
    )
    extract_llm = (
        get_llm(args.provider, args.extract_model) if args.extract_model else None
    )

    # Wire pipeline
    with BrowserAdapter() as browser:
        orchestrator = Orchestrator(
            conn=conn,
            llm=llm,
            browser=browser,
            classify_llm=classify_llm,
            extract_llm=extract_llm,
        )
        processor = DefaultBankProcessor(
            orchestrator=orchestrator,
            browser=browser,
            llm=llm,
            max_pages=args.max_pages,
        )
        run_manager = RunManager(conn)
        resume_id = UUID(args.resume) if args.resume else None

        runner: BackfillOrchestrator | IncrementalOrchestrator
        if args.mode == "incremental":
            runner = IncrementalOrchestrator(
                conn=conn,
                run_manager=run_manager,
                bank_processor=processor,
                banks_config=banks_config,
            )
        else:
            runner = BackfillOrchestrator(
                conn=conn,
                run_manager=run_manager,
                bank_processor=processor,
                banks_config=banks_config,
            )

        if args.schedule is not None:
            from cbs.scheduler import PipelineScheduler

            scheduler = PipelineScheduler(
                run_fn=lambda: runner.run(),
                interval_days=args.schedule,
            )
            scheduler.start()  # blocks forever
            return

        summary = runner.run(resume_run_id=resume_id)

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
