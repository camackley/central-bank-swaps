"""Swap repository — CRUD layer for the swaps table (Slice 1.2)."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from uuid import UUID

import duckdb
from pydantic import BaseModel


class SwapCreate(BaseModel):
    """Data required to insert a new swap row."""

    press_release_id: UUID
    provider_central_bank: str
    provider_country: str
    receiver_central_bank: str | None = None
    receiver_country: str | None = None
    currency: str
    swap_amount: Decimal | None = None
    swap_type: str
    announcement_type: str
    type_of_change: str | None = None
    conditions: str | None = None
    reasons_for_swap: str | None = None
    announcement_date: date | None = None
    effective_date: date | None = None
    maturity_date: date | None = None
    maturity_text: str | None = None
    duration_description: str | None = None
    raw_extract: str | None = None


class SwapRow(BaseModel):
    """A swap row as read from the database."""

    id: UUID
    press_release_id: UUID
    provider_central_bank: str | None = None
    provider_country: str | None = None
    receiver_central_bank: str | None = None
    receiver_country: str | None = None
    currency: str | None = None
    swap_amount: Decimal | None = None
    swap_type: str | None = None
    announcement_type: str | None = None
    type_of_change: str | None = None
    conditions: str | None = None
    reasons_for_swap: str | None = None
    announcement_date: date | None = None
    effective_date: date | None = None
    maturity_date: date | None = None
    maturity_text: str | None = None
    duration_description: str | None = None
    raw_extract: str | None = None
    created_at: datetime | None = None


_INSERT_SQL = """
INSERT INTO swaps (
    id, press_release_id,
    provider_central_bank, provider_country,
    receiver_central_bank, receiver_country,
    currency, swap_amount,
    swap_type, announcement_type, type_of_change,
    conditions, reasons_for_swap,
    announcement_date, effective_date, maturity_date,
    maturity_text, duration_description,
    raw_extract
) VALUES (
    uuid(), ?,
    ?, ?,
    ?, ?,
    ?, ?,
    ?, ?, ?,
    ?, ?,
    ?, ?, ?,
    ?, ?,
    ?
) RETURNING *
"""

_QUERY_BY_PR_SQL = """
SELECT * FROM swaps WHERE press_release_id = ? ORDER BY created_at
"""


def _row_to_swap(row: tuple[object, ...], columns: list[str]) -> SwapRow:
    """Convert a raw DuckDB row tuple into a SwapRow."""
    data = dict(zip(columns, row, strict=True))
    return SwapRow.model_validate(data)


def insert_swap(conn: duckdb.DuckDBPyConnection, swap: SwapCreate) -> SwapRow:
    """Insert a single swap row and return the created SwapRow."""
    params = [
        str(swap.press_release_id),
        swap.provider_central_bank,
        swap.provider_country,
        swap.receiver_central_bank,
        swap.receiver_country,
        swap.currency,
        float(swap.swap_amount) if swap.swap_amount is not None else None,
        swap.swap_type,
        swap.announcement_type,
        swap.type_of_change,
        swap.conditions,
        swap.reasons_for_swap,
        swap.announcement_date,
        swap.effective_date,
        swap.maturity_date,
        swap.maturity_text,
        swap.duration_description,
        swap.raw_extract,
    ]
    result = conn.execute(_INSERT_SQL, params)
    columns = [desc[0] for desc in result.description]
    row = result.fetchone()
    assert row is not None, "INSERT RETURNING should always return a row"
    return _row_to_swap(row, columns)


def query_swaps_by_press_release(
    conn: duckdb.DuckDBPyConnection, press_release_id: UUID
) -> list[SwapRow]:
    """Return all swaps linked to a given press release."""
    result = conn.execute(_QUERY_BY_PR_SQL, [str(press_release_id)])
    columns = [desc[0] for desc in result.description]
    return [_row_to_swap(row, columns) for row in result.fetchall()]
