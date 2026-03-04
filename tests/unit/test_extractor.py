"""Tests for the swap data extractor (Slice 1.6 — FR-005).

All tests mock the LLM — no network calls.
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from cbs.pipeline.extractor import (
    ExtractionResult,
    SwapDirection,
    SwapRecord,
    extract_swaps,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_llm_response(result: ExtractionResult) -> MagicMock:
    """Return a mock LLM with structured output returning *result*."""
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = result
    mock_llm.with_structured_output.return_value = mock_structured
    return mock_llm


# ---------------------------------------------------------------------------
# Fixture: sample press-release text (Fed ↔ ECB bilateral swap)
# ---------------------------------------------------------------------------

FED_ECB_TEXT = """\
The Federal Reserve announced today the establishment of a bilateral \
currency swap arrangement with the European Central Bank. Under the \
agreement, the Federal Reserve will provide up to $50 billion in U.S. \
dollars to the ECB, while the ECB will provide up to €45 billion in euros \
to the Federal Reserve. The arrangement is effective December 12, 2008, \
and will remain in place until February 1, 2010. The swap lines are being \
established to improve liquidity conditions in global financial markets. \
The interest rate applied will be the overnight index swap rate plus 100 \
basis points.
"""

FED_ECB_PARAGRAPH = (
    "The Federal Reserve announced today the establishment of a bilateral "
    "currency swap arrangement with the European Central Bank."
)


# ---------------------------------------------------------------------------
# test_bilateral_swap_produces_two_rows
# ---------------------------------------------------------------------------


def test_bilateral_swap_produces_two_rows() -> None:
    """A bilateral swap press release should yield two SwapDirection rows."""
    extraction = ExtractionResult(
        swaps=[
            SwapRecord(
                directions=[
                    SwapDirection(
                        provider_central_bank="Federal Reserve",
                        provider_country="United States",
                        receiver_central_bank="European Central Bank",
                        receiver_country="Eurozone",
                        currency="USD",
                        swap_amount=Decimal("50000000000"),
                    ),
                    SwapDirection(
                        provider_central_bank="European Central Bank",
                        provider_country="Eurozone",
                        receiver_central_bank="Federal Reserve",
                        receiver_country="United States",
                        currency="EUR",
                        swap_amount=Decimal("45000000000"),
                    ),
                ],
                swap_type="bilateral",
                announcement_type="new",
                announcement_date="2008-12-12",
                effective_date="2008-12-12",
                maturity_date="2010-02-01",
                maturity_text="until February 1, 2010",
                duration_description="approximately 14 months",
                conditions="overnight index swap rate plus 100 basis points",
                reasons_for_swap=(
                    "improve liquidity conditions in global financial markets"
                ),
                raw_extract=FED_ECB_TEXT.strip(),
            ),
        ],
    )

    llm = _make_llm_response(extraction)
    result = extract_swaps(llm, FED_ECB_TEXT)

    assert len(result.swaps) == 1
    assert len(result.swaps[0].directions) == 2

    usd_side = result.swaps[0].directions[0]
    eur_side = result.swaps[0].directions[1]

    assert usd_side.provider_central_bank == "Federal Reserve"
    assert usd_side.currency == "USD"
    assert usd_side.swap_amount == Decimal("50000000000")

    assert eur_side.provider_central_bank == "European Central Bank"
    assert eur_side.currency == "EUR"
    assert eur_side.swap_amount == Decimal("45000000000")


# ---------------------------------------------------------------------------
# test_missing_amount_stored_as_null
# ---------------------------------------------------------------------------


def test_missing_amount_stored_as_null() -> None:
    """When the press release doesn't state an amount, swap_amount must be None."""
    extraction = ExtractionResult(
        swaps=[
            SwapRecord(
                directions=[
                    SwapDirection(
                        provider_central_bank="Federal Reserve",
                        provider_country="United States",
                        receiver_central_bank="Bank of Japan",
                        receiver_country="Japan",
                        currency="USD",
                        swap_amount=None,
                    ),
                    SwapDirection(
                        provider_central_bank="Bank of Japan",
                        provider_country="Japan",
                        receiver_central_bank="Federal Reserve",
                        receiver_country="United States",
                        currency="JPY",
                        swap_amount=None,
                    ),
                ],
                swap_type="bilateral",
                announcement_type="new",
                announcement_date="2008-09-18",
                raw_extract="The Federal Reserve and Bank of Japan agreed...",
            ),
        ],
    )

    llm = _make_llm_response(extraction)
    result = extract_swaps(llm, "dummy text")

    for direction in result.swaps[0].directions:
        assert direction.swap_amount is None


# ---------------------------------------------------------------------------
# test_announcement_type_extracted
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ann_type",
    ["new", "extension", "renewal", "modification"],
)
def test_announcement_type_extracted(ann_type: str) -> None:
    """announcement_type must be one of new / extension / renewal / modification."""
    extraction = ExtractionResult(
        swaps=[
            SwapRecord(
                directions=[
                    SwapDirection(
                        provider_central_bank="Federal Reserve",
                        provider_country="United States",
                        receiver_central_bank="European Central Bank",
                        receiver_country="Eurozone",
                        currency="USD",
                    ),
                ],
                swap_type="bilateral",
                announcement_type=ann_type,
                announcement_date="2020-03-15",
                raw_extract="some text",
            ),
        ],
    )

    llm = _make_llm_response(extraction)
    result = extract_swaps(llm, "dummy text")

    assert result.swaps[0].announcement_type == ann_type


# ---------------------------------------------------------------------------
# test_conditions_and_reasons_extracted
# ---------------------------------------------------------------------------


def test_conditions_and_reasons_extracted() -> None:
    """conditions and reasons_for_swap should be captured from the LLM output."""
    extraction = ExtractionResult(
        swaps=[
            SwapRecord(
                directions=[
                    SwapDirection(
                        provider_central_bank="Swiss National Bank",
                        provider_country="Switzerland",
                        receiver_central_bank="Federal Reserve",
                        receiver_country="United States",
                        currency="CHF",
                    ),
                ],
                swap_type="bilateral",
                announcement_type="new",
                announcement_date="2020-03-19",
                conditions="OIS rate plus 25 basis points; minimum bid $5 million",
                reasons_for_swap="to ease strains in global USD funding markets",
                raw_extract="some text",
            ),
        ],
    )

    llm = _make_llm_response(extraction)
    result = extract_swaps(llm, "dummy text")

    swap = result.swaps[0]
    assert swap.conditions == "OIS rate plus 25 basis points; minimum bid $5 million"
    assert swap.reasons_for_swap == "to ease strains in global USD funding markets"


# ---------------------------------------------------------------------------
# test_maturity_text_verbatim
# ---------------------------------------------------------------------------


def test_maturity_text_verbatim() -> None:
    """maturity_text should preserve the original phrasing from the press release."""
    extraction = ExtractionResult(
        swaps=[
            SwapRecord(
                directions=[
                    SwapDirection(
                        provider_central_bank="Bank of England",
                        provider_country="United Kingdom",
                        receiver_central_bank="European Central Bank",
                        receiver_country="Eurozone",
                        currency="GBP",
                    ),
                ],
                swap_type="standing",
                announcement_type="extension",
                announcement_date="2019-11-06",
                maturity_text="until further notice",
                duration_description="indefinite",
                raw_extract="some text",
            ),
        ],
    )

    llm = _make_llm_response(extraction)
    result = extract_swaps(llm, "dummy text")

    assert result.swaps[0].maturity_text == "until further notice"
    assert result.swaps[0].duration_description == "indefinite"


# ---------------------------------------------------------------------------
# test_multiple_swaps_from_single_release
# ---------------------------------------------------------------------------


def test_multiple_swaps_from_single_release() -> None:
    """A single press release can announce multiple swap agreements."""
    extraction = ExtractionResult(
        swaps=[
            SwapRecord(
                directions=[
                    SwapDirection(
                        provider_central_bank="Federal Reserve",
                        provider_country="United States",
                        receiver_central_bank="European Central Bank",
                        receiver_country="Eurozone",
                        currency="USD",
                        swap_amount=Decimal("30000000000"),
                    ),
                    SwapDirection(
                        provider_central_bank="European Central Bank",
                        provider_country="Eurozone",
                        receiver_central_bank="Federal Reserve",
                        receiver_country="United States",
                        currency="EUR",
                        swap_amount=Decimal("25000000000"),
                    ),
                ],
                swap_type="bilateral",
                announcement_type="new",
                announcement_date="2008-12-12",
                raw_extract="paragraph about Fed-ECB swap",
            ),
            SwapRecord(
                directions=[
                    SwapDirection(
                        provider_central_bank="Federal Reserve",
                        provider_country="United States",
                        receiver_central_bank="Bank of Japan",
                        receiver_country="Japan",
                        currency="USD",
                        swap_amount=Decimal("60000000000"),
                    ),
                    SwapDirection(
                        provider_central_bank="Bank of Japan",
                        provider_country="Japan",
                        receiver_central_bank="Federal Reserve",
                        receiver_country="United States",
                        currency="JPY",
                    ),
                ],
                swap_type="bilateral",
                announcement_type="new",
                announcement_date="2008-12-12",
                raw_extract="paragraph about Fed-BoJ swap",
            ),
        ],
    )

    llm = _make_llm_response(extraction)
    result = extract_swaps(llm, "multi-swap press release text")

    assert len(result.swaps) == 2
    ecb = result.swaps[0].directions[0].receiver_central_bank
    boj = result.swaps[1].directions[0].receiver_central_bank
    assert ecb == "European Central Bank"
    assert boj == "Bank of Japan"


# ---------------------------------------------------------------------------
# test_single_direction_when_one_side_missing
# ---------------------------------------------------------------------------


def test_single_direction_when_one_side_missing() -> None:
    """When only one side is stated, a single SwapDirection row is acceptable."""
    extraction = ExtractionResult(
        swaps=[
            SwapRecord(
                directions=[
                    SwapDirection(
                        provider_central_bank="People's Bank of China",
                        provider_country="China",
                        receiver_central_bank="Bank of Korea",
                        receiver_country="South Korea",
                        currency="CNY",
                        swap_amount=Decimal("180000000000"),
                    ),
                ],
                swap_type="bilateral",
                announcement_type="new",
                announcement_date="2020-10-22",
                maturity_text="for three years",
                duration_description="3 years",
                raw_extract="The PBoC and Bank of Korea signed a bilateral...",
            ),
        ],
    )

    llm = _make_llm_response(extraction)
    result = extract_swaps(llm, "one-sided text")

    assert len(result.swaps) == 1
    assert len(result.swaps[0].directions) == 1
    assert result.swaps[0].directions[0].currency == "CNY"


# ---------------------------------------------------------------------------
# test_modification_includes_type_of_change
# ---------------------------------------------------------------------------


def test_modification_includes_type_of_change() -> None:
    """A 'modification' announcement should include type_of_change."""
    extraction = ExtractionResult(
        swaps=[
            SwapRecord(
                directions=[
                    SwapDirection(
                        provider_central_bank="Federal Reserve",
                        provider_country="United States",
                        receiver_central_bank="European Central Bank",
                        receiver_country="Eurozone",
                        currency="USD",
                        swap_amount=Decimal("120000000000"),
                    ),
                ],
                swap_type="bilateral",
                announcement_type="modification",
                type_of_change="increased swap amount from $80 billion to $120 billion",
                announcement_date="2020-03-20",
                raw_extract="The Federal Reserve increased the swap line...",
            ),
        ],
    )

    llm = _make_llm_response(extraction)
    result = extract_swaps(llm, "modification text")

    assert result.swaps[0].announcement_type == "modification"
    expected = "increased swap amount from $80 billion to $120 billion"
    assert result.swaps[0].type_of_change == expected


# ---------------------------------------------------------------------------
# test_extract_swaps_passes_text_to_llm
# ---------------------------------------------------------------------------


def test_extract_swaps_passes_text_to_llm() -> None:
    """extract_swaps must forward the press release text to the LLM."""
    extraction = ExtractionResult(swaps=[])
    llm = _make_llm_response(extraction)

    extract_swaps(llm, "The press release body goes here.")

    # The structured-output chain should have been invoked with the text
    structured = llm.with_structured_output.return_value
    structured.invoke.assert_called_once()
    call_arg = structured.invoke.call_args[0][0]
    assert "The press release body goes here." in str(call_arg)


# ---------------------------------------------------------------------------
# test_empty_extraction_returns_no_swaps
# ---------------------------------------------------------------------------


def test_empty_extraction_returns_no_swaps() -> None:
    """If the LLM finds no swaps, result.swaps should be an empty list."""
    extraction = ExtractionResult(swaps=[])
    llm = _make_llm_response(extraction)

    result = extract_swaps(llm, "This press release is about interest rates.")

    assert result.swaps == []
