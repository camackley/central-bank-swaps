import duckdb
import pytest


@pytest.fixture()
def db() -> duckdb.DuckDBPyConnection:
    """Fresh in-memory DuckDB connection per test."""
    conn = duckdb.connect(":memory:")
    yield conn
    conn.close()
