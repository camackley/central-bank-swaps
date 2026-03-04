"""Tests for press release repository CRUD layer (Slice 1.1)."""

from datetime import date

import duckdb
import pytest

from cbs.db.press_release_repo import (
    PressRelease,
    insert_press_release,
    mark_as_processed,
    query_unprocessed,
)
from cbs.db.schema import init_db


def _make_press_release(**overrides: object) -> PressRelease:
    """Build a PressRelease with sensible defaults, overridable per test."""
    defaults: dict[str, object] = {
        "central_bank_name": "Federal Reserve",
        "country": "US",
        "url": "https://www.federalreserve.gov/newsevents/pressreleases/swap-2024.htm",
        "title": "Federal Reserve announces swap arrangement",
        "publication_date": date(2024, 3, 15),
        "original_language": "en",
        "original_body": "The Federal Reserve announced a bilateral swap...",
        "source_type": "html",
    }
    defaults.update(overrides)
    return PressRelease(**defaults)  # type: ignore[arg-type]


class TestInsertPressRelease:
    """Inserting a press release stores it and returns its id."""

    def test_insert_press_release(self, db: duckdb.DuckDBPyConnection) -> None:
        init_db(db)
        pr = _make_press_release()

        row_id = insert_press_release(db, pr)

        assert row_id is not None
        row = db.execute(
            "SELECT central_bank_name, country, url, title, processed "
            "FROM press_releases WHERE id = ?",
            [row_id],
        ).fetchone()
        assert row is not None
        assert row[0] == "Federal Reserve"
        assert row[1] == "US"
        assert row[2] == pr.url
        assert row[3] == pr.title
        assert row[4] is False  # default unprocessed


class TestDuplicateUrlRejected:
    """URL-based deduplication: inserting the same URL twice raises."""

    def test_duplicate_url_rejected(self, db: duckdb.DuckDBPyConnection) -> None:
        init_db(db)
        pr = _make_press_release()

        insert_press_release(db, pr)

        with pytest.raises(duckdb.ConstraintException):
            insert_press_release(db, pr)


class TestQueryUnprocessed:
    """Query for press releases that have not been processed yet."""

    def test_query_unprocessed_press_releases(
        self, db: duckdb.DuckDBPyConnection
    ) -> None:
        init_db(db)
        pr1 = _make_press_release(url="https://fed.gov/pr1", title="First release")
        pr2 = _make_press_release(url="https://fed.gov/pr2", title="Second release")
        pr3 = _make_press_release(
            url="https://ecb.eu/pr1",
            title="ECB release",
            central_bank_name="European Central Bank",
            country="Eurozone",
        )

        id1 = insert_press_release(db, pr1)
        insert_press_release(db, pr2)
        insert_press_release(db, pr3)

        # Mark first as processed
        mark_as_processed(db, id1)

        # Query unprocessed for Fed — should get only pr2
        results = query_unprocessed(db, central_bank_name="Federal Reserve")
        assert len(results) == 1
        assert results[0].url == "https://fed.gov/pr2"
        assert results[0].title == "Second release"
        assert results[0].id is not None

    def test_query_unprocessed_all_banks(self, db: duckdb.DuckDBPyConnection) -> None:
        init_db(db)
        pr_fed = _make_press_release(url="https://fed.gov/pr1")
        pr_ecb = _make_press_release(
            url="https://ecb.eu/pr1",
            central_bank_name="European Central Bank",
            country="Eurozone",
        )
        insert_press_release(db, pr_fed)
        insert_press_release(db, pr_ecb)

        # No bank filter — should return all unprocessed
        results = query_unprocessed(db)
        assert len(results) == 2

    def test_query_unprocessed_returns_empty_when_all_processed(
        self, db: duckdb.DuckDBPyConnection
    ) -> None:
        init_db(db)
        pr = _make_press_release()
        row_id = insert_press_release(db, pr)
        mark_as_processed(db, row_id)

        results = query_unprocessed(db)
        assert results == []


class TestMarkAsProcessed:
    """Marking a press release as processed updates the flag and timestamp."""

    def test_mark_as_processed(self, db: duckdb.DuckDBPyConnection) -> None:
        init_db(db)
        pr = _make_press_release()
        row_id = insert_press_release(db, pr)

        mark_as_processed(db, row_id)

        row = db.execute(
            "SELECT processed, processed_at FROM press_releases WHERE id = ?",
            [row_id],
        ).fetchone()
        assert row is not None
        assert row[0] is True
        assert row[1] is not None  # timestamp should be set
