"""Tests for DuckDB schema creation (Slice 0.2)."""

import duckdb
import pytest

from cbs.db.schema import init_db


def _get_columns(conn: duckdb.DuckDBPyConnection, table: str) -> dict[str, str]:
    """Return {column_name: column_type} for a table."""
    rows = conn.execute(
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_name = ? ORDER BY ordinal_position",
        [table],
    ).fetchall()
    return {name: dtype for name, dtype in rows}


def _has_unique_constraint(
    conn: duckdb.DuckDBPyConnection, table: str, column: str
) -> bool:
    """Check if a column has a UNIQUE constraint (including via unique index)."""
    # Try inserting two rows with the same value — if it raises, unique is enforced
    try:
        conn.execute("BEGIN TRANSACTION")
        # Build a minimal INSERT with NULLs for other required cols
        cols = _get_columns(conn, table)
        values = {c: "NULL" for c in cols}
        values[column] = "'https://duplicate-test.example.com'"
        # We need a valid UUID for id columns
        values["id"] = "uuid()"
        col_list = ", ".join(values.keys())
        val_list = ", ".join(values.values())
        conn.execute(f"INSERT INTO {table} ({col_list}) VALUES ({val_list})")
        # Second insert with a different id but same url
        values["id"] = "uuid()"
        conn.execute(f"INSERT INTO {table} ({col_list}) VALUES ({val_list})")
        conn.execute("ROLLBACK")
        return False
    except duckdb.ConstraintException:
        conn.execute("ROLLBACK")
        return True


class TestPressReleasesTable:
    """press_releases table: columns, types, UNIQUE on url."""

    def test_schema_creates_press_releases_table(
        self, db: duckdb.DuckDBPyConnection
    ) -> None:
        init_db(db)
        columns = _get_columns(db, "press_releases")
        assert columns, "press_releases table should exist with columns"

    def test_press_releases_columns_and_types(
        self, db: duckdb.DuckDBPyConnection
    ) -> None:
        init_db(db)
        columns = _get_columns(db, "press_releases")

        expected = {
            "id": "UUID",
            "central_bank_name": "VARCHAR",
            "country": "VARCHAR",
            "url": "VARCHAR",
            "title": "VARCHAR",
            "publication_date": "DATE",
            "original_language": "VARCHAR",
            "original_body": "VARCHAR",
            "body_en": "VARCHAR",
            "is_swap_related": "BOOLEAN",
            "classification_reason": "VARCHAR",
            "processed": "BOOLEAN",
            "source_type": "VARCHAR",
            "created_at": "TIMESTAMP",
            "processed_at": "TIMESTAMP",
        }
        for col_name, col_type in expected.items():
            assert col_name in columns, f"Missing column: {col_name}"
            assert columns[col_name] == col_type, (
                f"Column {col_name}: expected {col_type}, got {columns[col_name]}"
            )

    def test_press_releases_url_is_unique(self, db: duckdb.DuckDBPyConnection) -> None:
        init_db(db)
        assert _has_unique_constraint(db, "press_releases", "url")


class TestSwapsTable:
    """swaps table: columns, FK to press_releases."""

    def test_schema_creates_swaps_table(self, db: duckdb.DuckDBPyConnection) -> None:
        init_db(db)
        columns = _get_columns(db, "swaps")
        assert columns, "swaps table should exist with columns"

    def test_swaps_columns_and_types(self, db: duckdb.DuckDBPyConnection) -> None:
        init_db(db)
        columns = _get_columns(db, "swaps")

        expected = {
            "id": "UUID",
            "press_release_id": "UUID",
            "provider_central_bank": "VARCHAR",
            "provider_country": "VARCHAR",
            "receiver_central_bank": "VARCHAR",
            "receiver_country": "VARCHAR",
            "currency": "VARCHAR",
            "swap_amount": "DECIMAL(18,2)",
            "swap_type": "VARCHAR",
            "announcement_type": "VARCHAR",
            "type_of_change": "VARCHAR",
            "conditions": "VARCHAR",
            "reasons_for_swap": "VARCHAR",
            "announcement_date": "DATE",
            "effective_date": "DATE",
            "maturity_date": "DATE",
            "maturity_text": "VARCHAR",
            "duration_description": "VARCHAR",
            "raw_extract": "VARCHAR",
            "created_at": "TIMESTAMP",
        }
        for col_name, col_type in expected.items():
            assert col_name in columns, f"Missing column: {col_name}"
            assert columns[col_name] == col_type, (
                f"Column {col_name}: expected {col_type}, got {columns[col_name]}"
            )

    def test_swaps_fk_to_press_releases(self, db: duckdb.DuckDBPyConnection) -> None:
        """Inserting a swap with a non-existent press_release_id should fail."""
        init_db(db)
        with pytest.raises(duckdb.ConstraintException):
            db.execute(
                "INSERT INTO swaps (id, press_release_id) VALUES (uuid(), uuid())"
            )


class TestScrapingRunsTable:
    """scraping_runs table: columns and types."""

    def test_schema_creates_scraping_runs_table(
        self, db: duckdb.DuckDBPyConnection
    ) -> None:
        init_db(db)
        columns = _get_columns(db, "scraping_runs")
        assert columns, "scraping_runs table should exist with columns"

    def test_scraping_runs_columns_and_types(
        self, db: duckdb.DuckDBPyConnection
    ) -> None:
        init_db(db)
        columns = _get_columns(db, "scraping_runs")

        expected = {
            "id": "UUID",
            "run_type": "VARCHAR",
            "started_at": "TIMESTAMP",
            "completed_at": "TIMESTAMP",
            "banks_attempted": "INTEGER",
            "banks_succeeded": "INTEGER",
            "press_releases_found": "INTEGER",
            "swaps_extracted": "INTEGER",
            "total_llm_cost_usd": "DECIMAL(10,4)",
            "errors": "VARCHAR",
        }
        for col_name, col_type in expected.items():
            assert col_name in columns, f"Missing column: {col_name}"
            assert columns[col_name] == col_type, (
                f"Column {col_name}: expected {col_type}, got {columns[col_name]}"
            )


class TestBankScrapingStatusTable:
    """bank_scraping_status table: columns and FK to scraping_runs."""

    def test_schema_creates_bank_scraping_status_table(
        self, db: duckdb.DuckDBPyConnection
    ) -> None:
        init_db(db)
        columns = _get_columns(db, "bank_scraping_status")
        assert columns, "bank_scraping_status table should exist with columns"

    def test_bank_scraping_status_columns_and_types(
        self, db: duckdb.DuckDBPyConnection
    ) -> None:
        init_db(db)
        columns = _get_columns(db, "bank_scraping_status")

        expected = {
            "id": "UUID",
            "run_id": "UUID",
            "central_bank_name": "VARCHAR",
            "homepage_url": "VARCHAR",
            "status": "VARCHAR",
            "press_releases_found": "INTEGER",
            "error_message": "VARCHAR",
            "started_at": "TIMESTAMP",
            "completed_at": "TIMESTAMP",
        }
        for col_name, col_type in expected.items():
            assert col_name in columns, f"Missing column: {col_name}"
            assert columns[col_name] == col_type, (
                f"Column {col_name}: expected {col_type}, got {columns[col_name]}"
            )

    def test_bank_scraping_status_fk_to_scraping_runs(
        self, db: duckdb.DuckDBPyConnection
    ) -> None:
        """Inserting a status with a non-existent run_id should fail."""
        init_db(db)
        with pytest.raises(duckdb.ConstraintException):
            db.execute(
                "INSERT INTO bank_scraping_status "
                "(id, run_id, central_bank_name, status) "
                "VALUES (uuid(), uuid(), 'Test Bank', 'pending')"
            )


class TestSchemaIdempotency:
    """Calling init_db() twice should not error."""

    def test_schema_is_idempotent(self, db: duckdb.DuckDBPyConnection) -> None:
        init_db(db)
        init_db(db)  # Should not raise
        # Verify tables still exist
        tables = {
            row[0]
            for row in db.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'main'"
            ).fetchall()
        }
        assert "press_releases" in tables
        assert "swaps" in tables
        assert "scraping_runs" in tables
        assert "bank_scraping_status" in tables
