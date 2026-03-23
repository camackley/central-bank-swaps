"""DuckDB schema initialization for central-bank-swaps."""

import duckdb

# Table names — use these constants in repos / queries to avoid typos.
TABLE_PRESS_RELEASES = "press_releases"
TABLE_SWAPS = "swaps"
TABLE_SCRAPING_RUNS = "scraping_runs"
TABLE_BANK_SCRAPING_STATUS = "bank_scraping_status"

_PRESS_RELEASES_DDL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_PRESS_RELEASES} (
    id UUID DEFAULT uuid() PRIMARY KEY,
    central_bank_name VARCHAR,
    country VARCHAR,
    url VARCHAR UNIQUE,
    title VARCHAR,
    publication_date DATE,
    original_language VARCHAR,
    original_body VARCHAR,
    body_en VARCHAR,
    is_swap_related BOOLEAN,
    classification_reason VARCHAR,
    processed BOOLEAN DEFAULT FALSE,
    source_type VARCHAR,
    created_at TIMESTAMP DEFAULT current_timestamp,
    processed_at TIMESTAMP
);
"""

_SWAPS_DDL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_SWAPS} (
    id UUID DEFAULT uuid() PRIMARY KEY,
    press_release_id UUID REFERENCES {TABLE_PRESS_RELEASES}(id),
    provider_central_bank VARCHAR,
    provider_country VARCHAR,
    receiver_central_bank VARCHAR,
    receiver_country VARCHAR,
    currency VARCHAR,
    swap_amount DECIMAL(18, 2),
    swap_type VARCHAR,
    announcement_type VARCHAR,
    type_of_change VARCHAR,
    conditions VARCHAR,
    reasons_for_swap VARCHAR,
    announcement_date DATE,
    effective_date DATE,
    maturity_date DATE,
    maturity_text VARCHAR,
    duration_description VARCHAR,
    raw_extract VARCHAR,
    created_at TIMESTAMP DEFAULT current_timestamp
);
"""

_SCRAPING_RUNS_DDL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_SCRAPING_RUNS} (
    id UUID DEFAULT uuid() PRIMARY KEY,
    run_type VARCHAR,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    banks_attempted INTEGER,
    banks_succeeded INTEGER,
    press_releases_found INTEGER,
    swaps_extracted INTEGER,
    total_llm_cost_usd DECIMAL(10, 4),
    errors VARCHAR
);
"""

_BANK_SCRAPING_STATUS_DDL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_BANK_SCRAPING_STATUS} (
    id UUID DEFAULT uuid() PRIMARY KEY,
    run_id UUID REFERENCES {TABLE_SCRAPING_RUNS}(id),
    central_bank_name VARCHAR,
    homepage_url VARCHAR,
    status VARCHAR,
    press_releases_found INTEGER,
    error_message VARCHAR,
    started_at TIMESTAMP,
    completed_at TIMESTAMP
);
"""

_ALL_DDL = [
    _PRESS_RELEASES_DDL,
    _SWAPS_DDL,
    _SCRAPING_RUNS_DDL,
    _BANK_SCRAPING_STATUS_DDL,
]


def init_db(conn: duckdb.DuckDBPyConnection) -> None:
    """Create all tables in the active schema. Safe to call multiple times."""
    for ddl in _ALL_DDL:
        conn.execute(ddl)


_RUNS_CATALOG_DDL = """
CREATE TABLE IF NOT EXISTS main.runs (
    id UUID PRIMARY KEY,
    schema_name VARCHAR UNIQUE,
    run_type VARCHAR,
    created_at TIMESTAMP DEFAULT current_timestamp
);
"""


def init_main(conn: duckdb.DuckDBPyConnection) -> None:
    """Create the global runs catalog in the main schema. Idempotent."""
    conn.execute(_RUNS_CATALOG_DDL)
