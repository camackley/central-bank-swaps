"""Backfill orchestrator — full pipeline for all configured banks (FR-006)."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING
from uuid import UUID

from cbs.config.banks import BankConfig, BanksConfig
from cbs.db.run_manager import RunManager
from cbs.db.schema import TABLE_SCRAPING_RUNS
from cbs.pipeline.models import RunSummary

if TYPE_CHECKING:
    import duckdb

    from cbs.pipeline.protocols import BankProcessor

logger = logging.getLogger(__name__)


class BackfillOrchestrator:
    """Orchestrate a historical backfill across all configured banks.

    Supports per-bank resumability: on re-run, completed banks are skipped
    and only failed/pending banks are retried.
    """

    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        run_manager: RunManager,
        bank_processor: BankProcessor,
        banks_config: BanksConfig,
    ) -> None:
        self._conn = conn
        self._run_manager = run_manager
        self._processor = bank_processor
        self._banks_config = banks_config

    def run(self, resume_run_id: UUID | None = None) -> RunSummary:
        """Execute the backfill.

        Args:
            resume_run_id: If provided, resume an existing run instead of
                creating a new one. Completed banks are skipped.

        Returns:
            RunSummary with aggregated counts and errors.
        """
        bank_names = [b.name for b in self._banks_config.banks]

        if resume_run_id is not None:
            run_id = resume_run_id
            logger.info("Resuming backfill run %s", run_id)
        else:
            scraping_run = self._run_manager.create_run("backfill", bank_names)
            run_id = scraping_run.id
            logger.info("Starting new backfill run %s", run_id)

        bank_lookup: dict[str, BankConfig] = {
            b.name: b for b in self._banks_config.banks
        }

        banks_to_process = self._run_manager.get_banks_to_process(run_id)

        summary = RunSummary(banks_attempted=len(bank_names))

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

            try:
                result = self._processor.process_bank(bank_config)
            except Exception as exc:
                error_msg = f"Unhandled error processing {bank_config.name}: {exc}"
                logger.exception(error_msg)
                self._run_manager.set_bank_status(
                    run_id, bank_config.name, "failed", error_message=error_msg
                )
                summary.errors.append(error_msg)
                continue

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
                    "Bank %s completed: %d press releases, %d swaps",
                    bank_config.name,
                    result.press_releases_found,
                    result.swaps_extracted,
                )

            summary.press_releases_found += result.press_releases_found
            summary.swaps_extracted += result.swaps_extracted

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
            "Backfill run %s complete: %d/%d banks succeeded, "
            "%d press releases, %d swaps",
            run_id,
            summary.banks_succeeded,
            summary.banks_attempted,
            summary.press_releases_found,
            summary.swaps_extracted,
        )

        return summary
