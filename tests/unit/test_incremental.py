"""Tests for IncrementalOrchestrator — Slice 2.1."""

from __future__ import annotations

import duckdb
import pytest

from cbs.config.banks import BankConfig, BanksConfig
from cbs.db.run_manager import RunManager
from cbs.db.schema import init_db
from cbs.pipeline.incremental import IncrementalOrchestrator
from cbs.pipeline.models import BankProcessingResult

# ---------------------------------------------------------------------------
# Helpers (reused patterns from test_backfill.py)
# ---------------------------------------------------------------------------


class FakeBankProcessor:
    """A fake BankProcessor that returns preconfigured results per bank name."""

    def __init__(
        self,
        results: dict[str, BankProcessingResult] | None = None,
        *,
        default_press_releases: int = 3,
        default_swaps: int = 1,
        default_skipped: int = 0,
    ) -> None:
        self._results = results or {}
        self._default_pr = default_press_releases
        self._default_swaps = default_swaps
        self._default_skipped = default_skipped
        self.processed_banks: list[str] = []

    def process_bank(self, bank: BankConfig) -> BankProcessingResult:
        self.processed_banks.append(bank.name)
        if bank.name in self._results:
            return self._results[bank.name]
        return BankProcessingResult(
            bank_name=bank.name,
            press_releases_found=self._default_pr,
            skipped_duplicates=self._default_skipped,
            swaps_extracted=self._default_swaps,
        )


def _make_banks_config(names: list[str]) -> BanksConfig:
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
) -> IncrementalOrchestrator:
    return IncrementalOrchestrator(db, RunManager(db), processor, banks)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestIncrementalCreatesRunWithCorrectType:
    def test_run_type_is_incremental(self, db: duckdb.DuckDBPyConnection) -> None:
        init_db(db)
        processor = FakeBankProcessor()
        banks = _make_banks_config(["Bank A"])
        orch = _make_orchestrator(db, processor, banks)

        orch.run()

        row = db.execute(
            "SELECT run_type FROM scraping_runs ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        assert row is not None
        assert row[0] == "incremental"


class TestIncrementalSkipsExistingUrls:
    def test_skipped_duplicates_in_summary(self, db: duckdb.DuckDBPyConnection) -> None:
        init_db(db)
        results = {
            "Bank A": BankProcessingResult(
                bank_name="Bank A",
                press_releases_found=2,
                skipped_duplicates=5,
                swaps_extracted=1,
            ),
        }
        processor = FakeBankProcessor(results=results)
        banks = _make_banks_config(["Bank A"])
        orch = _make_orchestrator(db, processor, banks)

        summary = orch.run()

        assert summary.press_releases_found == 2
        assert summary.banks_succeeded == 1


class TestIncrementalProcessesNewUrls:
    def test_new_press_releases_counted(self, db: duckdb.DuckDBPyConnection) -> None:
        init_db(db)
        processor = FakeBankProcessor(default_press_releases=4, default_swaps=2)
        banks = _make_banks_config(["Bank A", "Bank B"])
        orch = _make_orchestrator(db, processor, banks)

        summary = orch.run()

        assert summary.press_releases_found == 8
        assert summary.swaps_extracted == 4
        assert summary.banks_succeeded == 2


class TestIncrementalLogsSummary:
    def test_logs_new_and_skipped_counts(
        self,
        db: duckdb.DuckDBPyConnection,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        init_db(db)
        results = {
            "Bank A": BankProcessingResult(
                bank_name="Bank A",
                press_releases_found=3,
                skipped_duplicates=7,
                swaps_extracted=2,
            ),
        }
        processor = FakeBankProcessor(results=results)
        banks = _make_banks_config(["Bank A"])
        orch = _make_orchestrator(db, processor, banks)

        with caplog.at_level("INFO"):
            orch.run()

        assert any("3 new" in msg and "7 skipped" in msg for msg in caplog.messages)


class TestIncrementalResumeSupport:
    def test_resume_skips_completed_banks(self, db: duckdb.DuckDBPyConnection) -> None:
        init_db(db)
        mgr = RunManager(db)

        # Simulate interrupted run: Bank A completed, Bank B pending
        run = mgr.create_run("incremental", ["Bank A", "Bank B"])
        mgr.set_bank_status(run.id, "Bank A", "in_progress")
        mgr.set_bank_status(run.id, "Bank A", "completed", press_releases_found=3)

        processor = FakeBankProcessor()
        banks = _make_banks_config(["Bank A", "Bank B"])
        orch = IncrementalOrchestrator(db, mgr, processor, banks)
        summary = orch.run(resume_run_id=run.id)

        assert "Bank A" not in processor.processed_banks
        assert "Bank B" in processor.processed_banks
        assert summary.banks_succeeded == 1
