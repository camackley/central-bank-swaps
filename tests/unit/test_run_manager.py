"""Tests for run manager & bank status tracking (Slice 1.3)."""

import uuid

import duckdb

from cbs.db.run_manager import RunManager, ScrapingRun
from cbs.db.schema import init_db


class TestCreateScrapingRun:
    """Creating a new scraping run inserts a row and returns a model."""

    def test_create_scraping_run(self, db: duckdb.DuckDBPyConnection) -> None:
        init_db(db)
        mgr = RunManager(db)

        run = mgr.create_run(
            run_type="backfill",
            bank_names=["Federal Reserve", "ECB", "Bank of Japan"],
        )

        assert isinstance(run, ScrapingRun)
        assert run.run_type == "backfill"
        assert run.banks_attempted == 3

        # Row persisted in DB
        row = db.execute(
            "SELECT run_type, banks_attempted FROM scraping_runs WHERE id = ?",
            [str(run.id)],
        ).fetchone()
        assert row is not None
        assert row[0] == "backfill"
        assert row[1] == 3

    def test_create_run_initializes_bank_statuses(
        self, db: duckdb.DuckDBPyConnection
    ) -> None:
        """Creating a run also inserts a 'pending' status row per bank."""
        init_db(db)
        mgr = RunManager(db)

        run = mgr.create_run(
            run_type="backfill",
            bank_names=["Federal Reserve", "ECB"],
        )

        rows = db.execute(
            "SELECT central_bank_name, status FROM bank_scraping_status "
            "WHERE run_id = ? ORDER BY central_bank_name",
            [str(run.id)],
        ).fetchall()
        assert len(rows) == 2
        assert rows[0] == ("ECB", "pending")
        assert rows[1] == ("Federal Reserve", "pending")

    def test_create_run_with_explicit_run_id(
        self, db: duckdb.DuckDBPyConnection
    ) -> None:
        """Pre-generated run_id is preserved in the DB."""
        init_db(db)
        mgr = RunManager(db)
        explicit_id = uuid.uuid4()

        run = mgr.create_run(
            run_type="backfill",
            bank_names=["Federal Reserve"],
            run_id=explicit_id,
        )

        assert run.id == explicit_id
        row = db.execute(
            "SELECT id FROM scraping_runs WHERE id = ?", [str(explicit_id)]
        ).fetchone()
        assert row is not None
        assert row[0] == explicit_id


class TestBankStatusTransitions:
    """Bank statuses transition: pending → in_progress → completed/failed."""

    def test_transition_pending_to_in_progress(
        self, db: duckdb.DuckDBPyConnection
    ) -> None:
        init_db(db)
        mgr = RunManager(db)
        run = mgr.create_run(run_type="backfill", bank_names=["Federal Reserve"])

        mgr.set_bank_status(run.id, "Federal Reserve", "in_progress")

        status = mgr.get_bank_status(run.id, "Federal Reserve")
        assert status is not None
        assert status.status == "in_progress"
        assert status.started_at is not None

    def test_transition_in_progress_to_completed(
        self, db: duckdb.DuckDBPyConnection
    ) -> None:
        init_db(db)
        mgr = RunManager(db)
        run = mgr.create_run(run_type="backfill", bank_names=["Federal Reserve"])

        mgr.set_bank_status(run.id, "Federal Reserve", "in_progress")
        mgr.set_bank_status(
            run.id, "Federal Reserve", "completed", press_releases_found=15
        )

        status = mgr.get_bank_status(run.id, "Federal Reserve")
        assert status is not None
        assert status.status == "completed"
        assert status.press_releases_found == 15
        assert status.completed_at is not None

    def test_transition_in_progress_to_failed(
        self, db: duckdb.DuckDBPyConnection
    ) -> None:
        init_db(db)
        mgr = RunManager(db)
        run = mgr.create_run(run_type="backfill", bank_names=["Federal Reserve"])

        mgr.set_bank_status(run.id, "Federal Reserve", "in_progress")
        mgr.set_bank_status(
            run.id,
            "Federal Reserve",
            "failed",
            error_message="Connection timeout",
        )

        status = mgr.get_bank_status(run.id, "Federal Reserve")
        assert status is not None
        assert status.status == "failed"
        assert status.error_message == "Connection timeout"


class TestResumeSkipsCompletedBanks:
    """Resuming a run skips banks already marked completed."""

    def test_resume_skips_completed_banks(self, db: duckdb.DuckDBPyConnection) -> None:
        init_db(db)
        mgr = RunManager(db)
        run = mgr.create_run(
            run_type="backfill",
            bank_names=["Federal Reserve", "ECB", "Bank of Japan"],
        )

        # Mark Fed as completed
        mgr.set_bank_status(run.id, "Federal Reserve", "in_progress")
        mgr.set_bank_status(
            run.id, "Federal Reserve", "completed", press_releases_found=10
        )

        banks_to_process = mgr.get_banks_to_process(run.id)
        bank_names = [b.central_bank_name for b in banks_to_process]

        assert "Federal Reserve" not in bank_names
        assert "ECB" in bank_names
        assert "Bank of Japan" in bank_names


class TestResumeRetriesFailedBanks:
    """Resuming a run retries banks that previously failed."""

    def test_resume_retries_failed_banks(self, db: duckdb.DuckDBPyConnection) -> None:
        init_db(db)
        mgr = RunManager(db)
        run = mgr.create_run(
            run_type="backfill",
            bank_names=["Federal Reserve", "ECB"],
        )

        # Mark ECB as failed
        mgr.set_bank_status(run.id, "ECB", "in_progress")
        mgr.set_bank_status(run.id, "ECB", "failed", error_message="Site unavailable")

        banks_to_process = mgr.get_banks_to_process(run.id)
        bank_names = [b.central_bank_name for b in banks_to_process]

        assert "ECB" in bank_names
        assert "Federal Reserve" in bank_names


class TestResumeRetriesPendingBanks:
    """Resuming a run processes banks still in pending state."""

    def test_resume_retries_pending_banks(self, db: duckdb.DuckDBPyConnection) -> None:
        init_db(db)
        mgr = RunManager(db)
        run = mgr.create_run(
            run_type="backfill",
            bank_names=["Federal Reserve", "ECB", "Bank of Japan"],
        )

        # Mark Fed as completed, ECB as failed, BoJ stays pending
        mgr.set_bank_status(run.id, "Federal Reserve", "in_progress")
        mgr.set_bank_status(
            run.id, "Federal Reserve", "completed", press_releases_found=5
        )
        mgr.set_bank_status(run.id, "ECB", "in_progress")
        mgr.set_bank_status(run.id, "ECB", "failed", error_message="Timeout")

        banks_to_process = mgr.get_banks_to_process(run.id)
        bank_names = [b.central_bank_name for b in banks_to_process]

        # Pending and failed should both be retried
        assert "Bank of Japan" in bank_names
        assert "ECB" in bank_names
        # Completed should be skipped
        assert "Federal Reserve" not in bank_names
        assert len(banks_to_process) == 2
