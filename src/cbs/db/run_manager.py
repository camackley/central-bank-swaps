"""Run manager & bank status tracking for scraping resumability (Slice 1.3)."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

import duckdb
from pydantic import BaseModel

from cbs.db.schema import TABLE_BANK_SCRAPING_STATUS, TABLE_SCRAPING_RUNS


class ScrapingRun(BaseModel):
    """Row representation for the scraping_runs table."""

    id: UUID
    run_type: str
    started_at: datetime
    banks_attempted: int


class BankStatus(BaseModel):
    """Row representation for the bank_scraping_status table."""

    id: UUID
    run_id: UUID
    central_bank_name: str
    status: str
    press_releases_found: int | None = None
    error_message: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


class RunManager:
    """Manages scraping runs and per-bank status tracking."""

    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self._conn = conn

    def create_run(
        self,
        run_type: str,
        bank_names: list[str],
    ) -> ScrapingRun:
        """Create a new scraping run with pending status rows for each bank."""
        row = self._conn.execute(
            f"INSERT INTO {TABLE_SCRAPING_RUNS} "
            "(id, run_type, started_at, banks_attempted) "
            "VALUES (uuid(), ?, current_timestamp, ?) "
            "RETURNING id, run_type, started_at, banks_attempted",
            [run_type, len(bank_names)],
        ).fetchone()
        assert row is not None

        run = ScrapingRun(
            id=row[0],
            run_type=row[1],
            started_at=row[2],
            banks_attempted=row[3],
        )

        for bank_name in bank_names:
            self._conn.execute(
                f"INSERT INTO {TABLE_BANK_SCRAPING_STATUS} "
                "(id, run_id, central_bank_name, status) "
                "VALUES (uuid(), ?, ?, 'pending')",
                [str(run.id), bank_name],
            )

        return run

    def set_bank_status(
        self,
        run_id: UUID,
        bank_name: str,
        status: str,
        *,
        press_releases_found: int | None = None,
        error_message: str | None = None,
    ) -> None:
        """Update the status of a bank within a run."""
        if status == "in_progress":
            self._conn.execute(
                f"UPDATE {TABLE_BANK_SCRAPING_STATUS} "
                "SET status = ?, started_at = current_timestamp "
                "WHERE run_id = ? AND central_bank_name = ?",
                [status, str(run_id), bank_name],
            )
        elif status in ("completed", "failed"):
            self._conn.execute(
                f"UPDATE {TABLE_BANK_SCRAPING_STATUS} "
                "SET status = ?, completed_at = current_timestamp, "
                "press_releases_found = ?, error_message = ? "
                "WHERE run_id = ? AND central_bank_name = ?",
                [status, press_releases_found, error_message, str(run_id), bank_name],
            )
        else:
            self._conn.execute(
                f"UPDATE {TABLE_BANK_SCRAPING_STATUS} "
                "SET status = ? "
                "WHERE run_id = ? AND central_bank_name = ?",
                [status, str(run_id), bank_name],
            )

    def get_bank_status(
        self,
        run_id: UUID,
        bank_name: str,
    ) -> BankStatus | None:
        """Get the status of a specific bank within a run."""
        row = self._conn.execute(
            f"SELECT id, run_id, central_bank_name, status, "
            "press_releases_found, error_message, started_at, completed_at "
            f"FROM {TABLE_BANK_SCRAPING_STATUS} "
            "WHERE run_id = ? AND central_bank_name = ?",
            [str(run_id), bank_name],
        ).fetchone()
        if row is None:
            return None
        return BankStatus(
            id=row[0],
            run_id=row[1],
            central_bank_name=row[2],
            status=row[3],
            press_releases_found=row[4],
            error_message=row[5],
            started_at=row[6],
            completed_at=row[7],
        )

    def get_banks_to_process(self, run_id: UUID) -> list[BankStatus]:
        """Get banks that still need processing (pending or failed)."""
        rows = self._conn.execute(
            f"SELECT id, run_id, central_bank_name, status, "
            "press_releases_found, error_message, started_at, completed_at "
            f"FROM {TABLE_BANK_SCRAPING_STATUS} "
            "WHERE run_id = ? AND status != 'completed' "
            "ORDER BY central_bank_name",
            [str(run_id)],
        ).fetchall()
        return [
            BankStatus(
                id=r[0],
                run_id=r[1],
                central_bank_name=r[2],
                status=r[3],
                press_releases_found=r[4],
                error_message=r[5],
                started_at=r[6],
                completed_at=r[7],
            )
            for r in rows
        ]
