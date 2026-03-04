"""Tests for swap repository CRUD layer (Slice 1.2)."""

from datetime import date
from decimal import Decimal
from uuid import UUID

import duckdb
import pytest

from cbs.db.schema import init_db
from cbs.db.swap_repo import (
    SwapCreate,
    SwapRow,
    insert_swap,
    query_swaps_by_press_release,
)


def _insert_press_release(
    conn: duckdb.DuckDBPyConnection, url: str = "https://example.com/pr1"
) -> UUID:
    """Insert a minimal press release and return its id."""
    result = conn.execute(
        "INSERT INTO press_releases (id, central_bank_name, country, url) "
        "VALUES (uuid(), 'Federal Reserve', 'United States', ?) RETURNING id",
        [url],
    ).fetchone()
    assert result is not None
    return UUID(str(result[0]))


class TestInsertSwapPair:
    """test_insert_swap_pair — two rows per bilateral agreement."""

    def test_insert_swap_pair(self, db: duckdb.DuckDBPyConnection) -> None:
        """A bilateral swap should produce two directional rows."""
        init_db(db)
        pr_id = _insert_press_release(db)

        # Direction 1: Fed provides USD to ECB
        swap_usd = SwapCreate(
            press_release_id=pr_id,
            provider_central_bank="Federal Reserve",
            provider_country="United States",
            receiver_central_bank="European Central Bank",
            receiver_country="Eurozone",
            currency="USD",
            swap_amount=Decimal("50000000000.00"),
            swap_type="bilateral",
            announcement_type="new",
            announcement_date=date(2024, 1, 15),
            effective_date=date(2024, 2, 1),
            maturity_date=date(2024, 8, 1),
            maturity_text="for six months",
            duration_description="6 months",
            raw_extract="The Fed will provide up to $50B in USD to the ECB.",
        )
        # Direction 2: ECB provides EUR to Fed
        swap_eur = SwapCreate(
            press_release_id=pr_id,
            provider_central_bank="European Central Bank",
            provider_country="Eurozone",
            receiver_central_bank="Federal Reserve",
            receiver_country="United States",
            currency="EUR",
            swap_amount=Decimal("45000000000.00"),
            swap_type="bilateral",
            announcement_type="new",
            announcement_date=date(2024, 1, 15),
            effective_date=date(2024, 2, 1),
            maturity_date=date(2024, 8, 1),
            maturity_text="for six months",
            duration_description="6 months",
            raw_extract="The ECB will provide up to EUR 45B to the Fed.",
        )

        row_usd = insert_swap(db, swap_usd)
        row_eur = insert_swap(db, swap_eur)

        assert isinstance(row_usd.id, UUID)
        assert isinstance(row_eur.id, UUID)
        assert row_usd.id != row_eur.id

        rows = db.execute(
            "SELECT COUNT(*) FROM swaps WHERE press_release_id = ?", [str(pr_id)]
        ).fetchone()
        assert rows is not None
        assert rows[0] == 2


class TestSwapLinksToPressRelease:
    """test_swap_links_to_press_release — FK integrity."""

    def test_swap_links_to_press_release(self, db: duckdb.DuckDBPyConnection) -> None:
        """Inserted swap must reference an existing press release."""
        init_db(db)
        pr_id = _insert_press_release(db)

        swap = SwapCreate(
            press_release_id=pr_id,
            provider_central_bank="Bank of Japan",
            provider_country="Japan",
            receiver_central_bank="Federal Reserve",
            receiver_country="United States",
            currency="JPY",
            swap_type="standing",
            announcement_type="renewal",
            announcement_date=date(2023, 6, 1),
        )
        row = insert_swap(db, swap)
        assert row.press_release_id == pr_id

    def test_swap_rejects_invalid_press_release_id(
        self, db: duckdb.DuckDBPyConnection
    ) -> None:
        """Swap with non-existent press_release_id should fail."""
        init_db(db)
        from uuid import uuid4

        fake_pr_id = uuid4()
        swap = SwapCreate(
            press_release_id=fake_pr_id,
            provider_central_bank="Bank of England",
            provider_country="United Kingdom",
            receiver_central_bank="Federal Reserve",
            receiver_country="United States",
            currency="GBP",
            swap_type="bilateral",
            announcement_type="new",
            announcement_date=date(2024, 3, 1),
        )
        with pytest.raises(duckdb.ConstraintException):
            insert_swap(db, swap)


class TestQuerySwapsByPressReleaseId:
    """test_query_swaps_by_press_release_id."""

    def test_query_swaps_by_press_release_id(
        self, db: duckdb.DuckDBPyConnection
    ) -> None:
        """Query returns all swaps linked to a given press release."""
        init_db(db)
        pr_id = _insert_press_release(db)

        swap1 = SwapCreate(
            press_release_id=pr_id,
            provider_central_bank="Federal Reserve",
            provider_country="United States",
            receiver_central_bank="European Central Bank",
            receiver_country="Eurozone",
            currency="USD",
            swap_type="bilateral",
            announcement_type="new",
            announcement_date=date(2024, 1, 15),
        )
        swap2 = SwapCreate(
            press_release_id=pr_id,
            provider_central_bank="European Central Bank",
            provider_country="Eurozone",
            receiver_central_bank="Federal Reserve",
            receiver_country="United States",
            currency="EUR",
            swap_type="bilateral",
            announcement_type="new",
            announcement_date=date(2024, 1, 15),
        )
        insert_swap(db, swap1)
        insert_swap(db, swap2)

        results = query_swaps_by_press_release(db, pr_id)
        assert len(results) == 2
        assert all(isinstance(r, SwapRow) for r in results)
        assert all(r.press_release_id == pr_id for r in results)

    def test_query_returns_empty_for_no_swaps(
        self, db: duckdb.DuckDBPyConnection
    ) -> None:
        """Query returns empty list when press release has no swaps."""
        init_db(db)
        pr_id = _insert_press_release(db)

        results = query_swaps_by_press_release(db, pr_id)
        assert results == []

    def test_query_isolates_by_press_release(
        self, db: duckdb.DuckDBPyConnection
    ) -> None:
        """Swaps from different press releases are not mixed."""
        init_db(db)
        pr1_id = _insert_press_release(db, url="https://example.com/pr1")
        pr2_id = _insert_press_release(db, url="https://example.com/pr2")

        swap_for_pr1 = SwapCreate(
            press_release_id=pr1_id,
            provider_central_bank="Federal Reserve",
            provider_country="United States",
            receiver_central_bank="Bank of Japan",
            receiver_country="Japan",
            currency="USD",
            swap_type="bilateral",
            announcement_type="new",
            announcement_date=date(2024, 1, 1),
        )
        swap_for_pr2 = SwapCreate(
            press_release_id=pr2_id,
            provider_central_bank="Bank of Canada",
            provider_country="Canada",
            receiver_central_bank="Swiss National Bank",
            receiver_country="Switzerland",
            currency="CAD",
            swap_type="standing",
            announcement_type="extension",
            announcement_date=date(2024, 2, 1),
        )
        insert_swap(db, swap_for_pr1)
        insert_swap(db, swap_for_pr2)

        results_pr1 = query_swaps_by_press_release(db, pr1_id)
        results_pr2 = query_swaps_by_press_release(db, pr2_id)

        assert len(results_pr1) == 1
        assert results_pr1[0].provider_central_bank == "Federal Reserve"
        assert len(results_pr2) == 1
        assert results_pr2[0].provider_central_bank == "Bank of Canada"


class TestSingleDirectionSwapAllowed:
    """test_single_direction_swap_allowed — one row when only one side known."""

    def test_single_direction_swap_allowed(self, db: duckdb.DuckDBPyConnection) -> None:
        """A single directional row is valid when only one side is known."""
        init_db(db)
        pr_id = _insert_press_release(db)

        swap = SwapCreate(
            press_release_id=pr_id,
            provider_central_bank="People's Bank of China",
            provider_country="China",
            receiver_central_bank=None,
            receiver_country=None,
            currency="CNY",
            swap_amount=None,
            swap_type="bilateral",
            announcement_type="new",
            announcement_date=date(2024, 5, 1),
            raw_extract="PBoC announced a new swap arrangement in CNY.",
        )
        row = insert_swap(db, swap)

        assert isinstance(row.id, UUID)
        assert row.receiver_central_bank is None
        assert row.receiver_country is None
        assert row.swap_amount is None

        results = query_swaps_by_press_release(db, pr_id)
        assert len(results) == 1
        assert results[0].receiver_central_bank is None

    def test_nullable_fields_stored_correctly(
        self, db: duckdb.DuckDBPyConnection
    ) -> None:
        """None optional fields stored as NULL, retrieved as None."""
        init_db(db)
        pr_id = _insert_press_release(db)

        swap = SwapCreate(
            press_release_id=pr_id,
            provider_central_bank="Reserve Bank of India",
            provider_country="India",
            receiver_central_bank="Bank of Japan",
            receiver_country="Japan",
            currency="INR",
            swap_type="bilateral",
            announcement_type="new",
            announcement_date=date(2024, 4, 1),
            # All other optional fields left as None
        )
        row = insert_swap(db, swap)

        assert row.swap_amount is None
        assert row.effective_date is None
        assert row.maturity_date is None
        assert row.maturity_text is None
        assert row.duration_description is None
        assert row.type_of_change is None
        assert row.conditions is None
        assert row.reasons_for_swap is None
        assert row.raw_extract is None
