"""Tests for BackfillOrchestrator — Slice 1.12 (FR-006)."""

from __future__ import annotations

import duckdb

from cbs.config.banks import BankConfig, BanksConfig
from cbs.db.run_manager import RunManager
from cbs.db.schema import init_db
from cbs.pipeline.backfill import BackfillOrchestrator
from cbs.pipeline.models import BankProcessingResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeBankProcessor:
    """A fake BankProcessor that returns preconfigured results per bank name."""

    def __init__(
        self,
        results: dict[str, BankProcessingResult] | None = None,
        *,
        default_press_releases: int = 5,
        default_swaps: int = 2,
    ) -> None:
        self._results = results or {}
        self._default_pr = default_press_releases
        self._default_swaps = default_swaps
        self.processed_banks: list[str] = []

    def process_bank(self, bank: BankConfig) -> BankProcessingResult:
        self.processed_banks.append(bank.name)
        if bank.name in self._results:
            return self._results[bank.name]
        return BankProcessingResult(
            bank_name=bank.name,
            press_releases_found=self._default_pr,
            swaps_extracted=self._default_swaps,
        )


def _make_banks_config(names: list[str]) -> BanksConfig:
    """Create a BanksConfig with minimal test banks."""
    return BanksConfig(
        banks=[
            BankConfig(
                name=name,
                country="Testland",
                homepage_url=f"https://{name.lower().replace(' ', '-')}.example.com",
            )
            for name in names
        ]
    )


def _make_orchestrator(
    db: duckdb.DuckDBPyConnection,
    processor: FakeBankProcessor,
    banks: BanksConfig,
) -> BackfillOrchestrator:
    return BackfillOrchestrator(db, RunManager(db), processor, banks)


# ---------------------------------------------------------------------------
# test_backfill_processes_all_configured_banks
# ---------------------------------------------------------------------------


class TestBackfillProcessesAllConfiguredBanks:
    def test_all_banks_processed(self, db: duckdb.DuckDBPyConnection) -> None:
        init_db(db)
        processor = FakeBankProcessor()
        banks = _make_banks_config(["Bank A", "Bank B", "Bank C"])
        orch = _make_orchestrator(db, processor, banks)

        summary = orch.run()

        assert set(processor.processed_banks) == {"Bank A", "Bank B", "Bank C"}
        assert summary.banks_attempted == 3
        assert summary.banks_succeeded == 3

    def test_all_banks_marked_completed_in_db(
        self, db: duckdb.DuckDBPyConnection
    ) -> None:
        init_db(db)
        processor = FakeBankProcessor()
        banks = _make_banks_config(["Bank A", "Bank B"])
        orch = _make_orchestrator(db, processor, banks)

        orch.run()

        rows = db.execute("SELECT status FROM bank_scraping_status").fetchall()
        statuses = {r[0] for r in rows}
        assert statuses == {"completed"}


# ---------------------------------------------------------------------------
# test_backfill_stops_at_historical_cutoff
# ---------------------------------------------------------------------------


class TestBackfillStopsAtHistoricalCutoff:
    def test_cutoff_bank_still_completed(self, db: duckdb.DuckDBPyConnection) -> None:
        init_db(db)
        results = {
            "Fed": BankProcessingResult(
                bank_name="Fed",
                press_releases_found=15,
                swaps_extracted=3,
                hit_cutoff=True,
            )
        }
        processor = FakeBankProcessor(results=results)
        banks = _make_banks_config(["Fed"])
        orch = _make_orchestrator(db, processor, banks)

        summary = orch.run()

        assert summary.banks_succeeded == 1
        assert summary.press_releases_found == 15

    def test_cutoff_year_passed_via_bank_config(
        self, db: duckdb.DuckDBPyConnection
    ) -> None:
        """Verify the BankConfig (with historical_cutoff_year) is passed through."""
        init_db(db)

        class CutoffCapturingProcessor:
            def __init__(self) -> None:
                self.received_cutoffs: list[int] = []

            def process_bank(self, bank: BankConfig) -> BankProcessingResult:
                self.received_cutoffs.append(bank.historical_cutoff_year)
                return BankProcessingResult(bank_name=bank.name)

        processor = CutoffCapturingProcessor()
        banks = BanksConfig(
            banks=[
                BankConfig(
                    name="Fed",
                    country="US",
                    homepage_url="https://fed.gov",
                    historical_cutoff_year=2000,
                ),
                BankConfig(
                    name="ECB",
                    country="EU",
                    homepage_url="https://ecb.eu",
                ),
            ]
        )
        orch = BackfillOrchestrator(db, RunManager(db), processor, banks)

        orch.run()

        assert set(processor.received_cutoffs) == {2000, 2008}


# ---------------------------------------------------------------------------
# test_backfill_resume_after_failure
# ---------------------------------------------------------------------------


class TestBackfillResumeAfterFailure:
    def test_resume_skips_completed_banks(self, db: duckdb.DuckDBPyConnection) -> None:
        init_db(db)
        mgr = RunManager(db)

        # Run 1: Bank A succeeds, Bank B fails
        results_run1 = {
            "Bank A": BankProcessingResult(
                bank_name="Bank A", press_releases_found=10, swaps_extracted=4
            ),
            "Bank B": BankProcessingResult(
                bank_name="Bank B",
                press_releases_found=0,
                errors=["Connection refused"],
            ),
        }
        processor1 = FakeBankProcessor(results=results_run1)
        banks = _make_banks_config(["Bank A", "Bank B"])
        orch1 = BackfillOrchestrator(db, mgr, processor1, banks)
        orch1.run()

        # Find the run_id from DB
        row = db.execute(
            "SELECT id FROM scraping_runs ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        assert row is not None
        run_id = row[0]

        # Run 2 (resume): Bank B now succeeds
        results_run2 = {
            "Bank B": BankProcessingResult(
                bank_name="Bank B", press_releases_found=8, swaps_extracted=2
            ),
        }
        processor2 = FakeBankProcessor(results=results_run2)
        orch2 = BackfillOrchestrator(db, mgr, processor2, banks)
        summary = orch2.run(resume_run_id=run_id)

        assert "Bank A" not in processor2.processed_banks
        assert "Bank B" in processor2.processed_banks
        assert summary.banks_succeeded == 1

    def test_resume_retries_pending_banks(self, db: duckdb.DuckDBPyConnection) -> None:
        init_db(db)
        mgr = RunManager(db)

        # Simulate interrupted run: Bank A completed, Bank B still pending
        run = mgr.create_run("backfill", ["Bank A", "Bank B"])
        mgr.set_bank_status(run.id, "Bank A", "in_progress")
        mgr.set_bank_status(run.id, "Bank A", "completed", press_releases_found=10)
        # Bank B stays pending

        processor = FakeBankProcessor()
        banks = _make_banks_config(["Bank A", "Bank B"])
        orch = BackfillOrchestrator(db, mgr, processor, banks)
        summary = orch.run(resume_run_id=run.id)

        assert "Bank A" not in processor.processed_banks
        assert "Bank B" in processor.processed_banks
        assert summary.banks_succeeded == 1


# ---------------------------------------------------------------------------
# test_backfill_records_run_summary
# ---------------------------------------------------------------------------


class TestBackfillRecordsRunSummary:
    def test_summary_counts(self, db: duckdb.DuckDBPyConnection) -> None:
        init_db(db)
        processor = FakeBankProcessor(default_press_releases=10, default_swaps=3)
        banks = _make_banks_config(["Bank A", "Bank B"])
        orch = _make_orchestrator(db, processor, banks)

        summary = orch.run()

        assert summary.banks_attempted == 2
        assert summary.banks_succeeded == 2
        assert summary.press_releases_found == 20
        assert summary.swaps_extracted == 6
        assert summary.errors == []

    def test_summary_persisted_to_db(self, db: duckdb.DuckDBPyConnection) -> None:
        init_db(db)
        processor = FakeBankProcessor(default_press_releases=7, default_swaps=1)
        banks = _make_banks_config(["Bank A"])
        orch = _make_orchestrator(db, processor, banks)

        orch.run()

        row = db.execute(
            "SELECT banks_attempted, banks_succeeded, press_releases_found, "
            "swaps_extracted, completed_at FROM scraping_runs"
        ).fetchone()
        assert row is not None
        assert row[0] == 1  # banks_attempted
        assert row[1] == 1  # banks_succeeded
        assert row[2] == 7  # press_releases_found
        assert row[3] == 1  # swaps_extracted
        assert row[4] is not None  # completed_at set

    def test_failed_bank_errors_in_summary(self, db: duckdb.DuckDBPyConnection) -> None:
        init_db(db)
        results = {
            "Bad Bank": BankProcessingResult(
                bank_name="Bad Bank",
                press_releases_found=0,
                errors=["SSL certificate error"],
            ),
        }
        processor = FakeBankProcessor(results=results)
        banks = _make_banks_config(["Bad Bank", "Good Bank"])
        orch = _make_orchestrator(db, processor, banks)

        summary = orch.run()

        assert summary.banks_succeeded == 1
        assert len(summary.errors) == 1
        assert "SSL certificate error" in summary.errors[0]


# ---------------------------------------------------------------------------
# test_backfill_parallel_mode
# ---------------------------------------------------------------------------


class TestBackfillParallelMode:
    def test_parallel_processes_all_banks(self, db: duckdb.DuckDBPyConnection) -> None:
        init_db(db)
        proc1 = FakeBankProcessor()
        proc2 = FakeBankProcessor()
        banks = _make_banks_config(["Bank A", "Bank B", "Bank C"])
        orch = BackfillOrchestrator(db, RunManager(db), [proc1, proc2], banks)

        summary = orch.run()

        all_processed = set(proc1.processed_banks + proc2.processed_banks)
        assert all_processed == {"Bank A", "Bank B", "Bank C"}
        assert summary.banks_succeeded == 3
        assert summary.press_releases_found == 15  # 3 banks * 5 default

    def test_parallel_handles_errors(self, db: duckdb.DuckDBPyConnection) -> None:
        init_db(db)
        results = {
            "Bad Bank": BankProcessingResult(
                bank_name="Bad Bank",
                press_releases_found=0,
                errors=["Connection refused"],
            ),
        }
        proc1 = FakeBankProcessor(results=results)
        proc2 = FakeBankProcessor()
        banks = _make_banks_config(["Bad Bank", "Good Bank"])
        orch = BackfillOrchestrator(db, RunManager(db), [proc1, proc2], banks)

        summary = orch.run()

        assert summary.banks_succeeded == 1
        assert len(summary.errors) == 1

    def test_parallel_handles_exceptions(self, db: duckdb.DuckDBPyConnection) -> None:
        init_db(db)

        class ExplodingProcessor:
            def process_bank(self, bank: BankConfig) -> BankProcessingResult:
                raise RuntimeError("boom")

        banks = _make_banks_config(["Bank A"])
        orch = BackfillOrchestrator(db, RunManager(db), [ExplodingProcessor()], banks)

        summary = orch.run()

        assert summary.banks_succeeded == 0
        assert len(summary.errors) == 1
        assert "boom" in summary.errors[0]

    def test_single_processor_uses_sequential(
        self, db: duckdb.DuckDBPyConnection
    ) -> None:
        """A single-element list still works (uses sequential path)."""
        init_db(db)
        processor = FakeBankProcessor()
        banks = _make_banks_config(["Bank A"])
        orch = BackfillOrchestrator(db, RunManager(db), [processor], banks)

        summary = orch.run()

        assert summary.banks_succeeded == 1
        assert processor.processed_banks == ["Bank A"]
