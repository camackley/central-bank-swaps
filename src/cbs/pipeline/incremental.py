"""Incremental orchestrator — process only new press releases (Slice 2.1)."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING
from uuid import UUID

from langsmith import traceable

from cbs.config.banks import BankConfig, BanksConfig
from cbs.db.run_manager import BankStatus, RunManager
from cbs.db.schema import TABLE_SCRAPING_RUNS
from cbs.pipeline.models import BankProcessingResult, RunSummary

if TYPE_CHECKING:
    import duckdb

    from cbs.pipeline.protocols import BankProcessor

logger = logging.getLogger(__name__)


class IncrementalOrchestrator:
    """Orchestrate an incremental update across all configured banks.

    Structurally similar to BackfillOrchestrator but uses run_type='incremental'
    and scans fewer listing pages (only recent press releases).
    """

    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        run_manager: RunManager,
        bank_processor: BankProcessor | Sequence[BankProcessor],
        banks_config: BanksConfig,
    ) -> None:
        self._conn = conn
        self._run_manager = run_manager
        if isinstance(bank_processor, Sequence):
            self._processors = list(bank_processor)
        else:
            self._processors = [bank_processor]
        self._banks_config = banks_config

    @traceable(name="incremental_run", run_type="chain")
    def run(self, resume_run_id: UUID | None = None) -> RunSummary:
        """Execute an incremental update.

        Args:
            resume_run_id: If provided, resume an existing run instead of
                creating a new one. Completed banks are skipped.

        Returns:
            RunSummary with aggregated counts and errors.
        """
        bank_names = [b.name for b in self._banks_config.banks]

        if resume_run_id is not None:
            run_id = resume_run_id
            logger.info("Resuming incremental run %s", run_id)
        else:
            scraping_run = self._run_manager.create_run("incremental", bank_names)
            run_id = scraping_run.id
            logger.info("Starting new incremental run %s", run_id)

        bank_lookup: dict[str, BankConfig] = {
            b.name: b for b in self._banks_config.banks
        }

        banks_to_process = self._run_manager.get_banks_to_process(run_id)

        summary = RunSummary(banks_attempted=len(bank_names))

        if len(self._processors) == 1:
            self._run_sequential(run_id, banks_to_process, bank_lookup, summary)
        else:
            self._run_parallel(run_id, banks_to_process, bank_lookup, summary)

        # Finalize the run record
        errors_json = json.dumps(summary.errors) if summary.errors else None
        self._conn.execute(
            f"UPDATE {TABLE_SCRAPING_RUNS} SET "
            "completed_at = current_timestamp, "
            "banks_succeeded = ?, press_releases_found = ?, "
            "swaps_extracted = ?, errors = ? "
            "WHERE id = ?",
            [
                summary.banks_succeeded,
                summary.press_releases_found,
                summary.swaps_extracted,
                errors_json,
                str(run_id),
            ],
        )

        logger.info(
            "Incremental run %s complete: %d/%d banks succeeded, "
            "%d press releases, %d swaps",
            run_id,
            summary.banks_succeeded,
            summary.banks_attempted,
            summary.press_releases_found,
            summary.swaps_extracted,
        )

        return summary

    def _run_sequential(
        self,
        run_id: UUID,
        banks_to_process: list[BankStatus],
        bank_lookup: dict[str, BankConfig],
        summary: RunSummary,
    ) -> None:
        processor = self._processors[0]
        for bank_status in banks_to_process:
            bank_config = bank_lookup.get(bank_status.central_bank_name)
            if bank_config is None:
                logger.warning(
                    "Bank '%s' in run but not in config — skipping",
                    bank_status.central_bank_name,
                )
                continue

            self._run_manager.set_bank_status(run_id, bank_config.name, "in_progress")
            logger.info("Processing bank: %s", bank_config.name)

            result = processor.process_bank(bank_config)

            self._handle_result(run_id, bank_config, result, summary)

    def _run_parallel(
        self,
        run_id: UUID,
        banks_to_process: list[BankStatus],
        bank_lookup: dict[str, BankConfig],
        summary: RunSummary,
    ) -> None:
        with ThreadPoolExecutor(max_workers=len(self._processors)) as executor:
            futures = {}
            for i, bank_status in enumerate(banks_to_process):
                bank_config = bank_lookup.get(bank_status.central_bank_name)
                if bank_config is None:
                    logger.warning(
                        "Bank '%s' in run but not in config — skipping",
                        bank_status.central_bank_name,
                    )
                    continue

                self._run_manager.set_bank_status(
                    run_id, bank_config.name, "in_progress"
                )
                logger.info("Processing bank (parallel): %s", bank_config.name)

                worker_idx = i % len(self._processors)
                future = executor.submit(
                    self._processors[worker_idx].process_bank, bank_config
                )
                futures[future] = bank_config

            for future in as_completed(futures):
                bank_config = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    error_msg = f"Unhandled error processing {bank_config.name}: {exc}"
                    logger.exception(error_msg)
                    self._run_manager.set_bank_status(
                        run_id, bank_config.name, "failed", error_message=error_msg
                    )
                    summary.errors.append(error_msg)
                    continue

                self._handle_result(run_id, bank_config, result, summary)

    def _handle_result(
        self,
        run_id: UUID,
        bank_config: BankConfig,
        result: BankProcessingResult,
        summary: RunSummary,
    ) -> None:
        if result.errors:
            error_msg = "; ".join(result.errors)
            self._run_manager.set_bank_status(
                run_id,
                bank_config.name,
                "failed",
                error_message=error_msg,
            )
            summary.errors.extend(result.errors)
            logger.error("Bank %s failed: %s", bank_config.name, error_msg)
        else:
            self._run_manager.set_bank_status(
                run_id,
                bank_config.name,
                "completed",
                press_releases_found=result.press_releases_found,
            )
            summary.banks_succeeded += 1
            logger.info(
                "Bank %s completed: %d new, %d skipped, %d swaps",
                bank_config.name,
                result.press_releases_found,
                result.skipped_duplicates,
                result.swaps_extracted,
            )

        summary.press_releases_found += result.press_releases_found
        summary.swaps_extracted += result.swaps_extracted
