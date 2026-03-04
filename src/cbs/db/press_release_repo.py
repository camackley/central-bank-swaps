"""Data access layer for the press_releases table (Slice 1.1)."""

from __future__ import annotations

import uuid
from datetime import date, datetime

import duckdb
from pydantic import BaseModel

from cbs.db.schema import TABLE_PRESS_RELEASES


class PressRelease(BaseModel):
    """Pydantic model representing a row in the press_releases table."""

    id: uuid.UUID | None = None
    central_bank_name: str
    country: str
    url: str
    title: str
    publication_date: date | None = None
    original_language: str | None = None
    original_body: str | None = None
    body_en: str | None = None
    is_swap_related: bool | None = None
    classification_reason: str | None = None
    processed: bool = False
    source_type: str | None = None
    created_at: datetime | None = None
    processed_at: datetime | None = None


def insert_press_release(
    conn: duckdb.DuckDBPyConnection, pr: PressRelease
) -> uuid.UUID:
    """Insert a press release and return its generated UUID.

    Raises ``duckdb.ConstraintException`` if the URL already exists.
    """
    row_id = uuid.uuid4()
    conn.execute(
        f"INSERT INTO {TABLE_PRESS_RELEASES} "  # noqa: S608
        "(id, central_bank_name, country, url, title, publication_date, "
        "original_language, original_body, body_en, is_swap_related, "
        "classification_reason, processed, source_type) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            str(row_id),
            pr.central_bank_name,
            pr.country,
            pr.url,
            pr.title,
            pr.publication_date,
            pr.original_language,
            pr.original_body,
            pr.body_en,
            pr.is_swap_related,
            pr.classification_reason,
            pr.processed,
            pr.source_type,
        ],
    )
    return row_id


def query_unprocessed(
    conn: duckdb.DuckDBPyConnection,
    *,
    central_bank_name: str | None = None,
) -> list[PressRelease]:
    """Return press releases where ``processed`` is FALSE.

    Optionally filter by ``central_bank_name``.
    """
    query = f"SELECT * FROM {TABLE_PRESS_RELEASES} WHERE processed = FALSE"  # noqa: S608
    params: list[object] = []
    if central_bank_name is not None:
        query += " AND central_bank_name = ?"
        params.append(central_bank_name)
    query += " ORDER BY created_at"

    rows = conn.execute(query, params).fetchall()
    assert conn.description is not None
    columns = [desc[0] for desc in conn.description]
    return [PressRelease(**dict(zip(columns, row, strict=True))) for row in rows]


def mark_as_processed(
    conn: duckdb.DuckDBPyConnection, press_release_id: uuid.UUID
) -> None:
    """Set ``processed = TRUE`` and ``processed_at`` to now."""
    conn.execute(
        f"UPDATE {TABLE_PRESS_RELEASES} "  # noqa: S608
        "SET processed = TRUE, processed_at = current_timestamp "
        "WHERE id = ?",
        [str(press_release_id)],
    )
