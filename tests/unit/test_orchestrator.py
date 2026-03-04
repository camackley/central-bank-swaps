"""Tests for the pipeline orchestrator — Slice 1.11 (FR-001 → FR-005).

Wires all stages: extract → translate → classify → extract swaps → store.
All LLM-dependent functions are patched at the module level.
"""

from __future__ import annotations

import datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

import duckdb
import pytest

from cbs.db.schema import init_db
from cbs.pipeline.classifier import ClassificationResult
from cbs.pipeline.extractor import (
    ExtractionResult,
    SwapDirection,
    SwapRecord,
)
from cbs.pipeline.orchestrator import Orchestrator
from cbs.pipeline.translator import TranslationResult
from cbs.scraper.html_extractor import HtmlExtractResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db() -> duckdb.DuckDBPyConnection:
    """Fresh in-memory DuckDB with schema initialised."""
    conn = duckdb.connect(":memory:")
    init_db(conn)
    yield conn
    conn.close()


# ---------------------------------------------------------------------------
# Canned LLM responses
# ---------------------------------------------------------------------------

_SWAP_CLASSIFICATION = ClassificationResult(
    is_swap_related=True,
    reason="Announces a bilateral currency swap arrangement between central banks.",
)

_NON_SWAP_CLASSIFICATION = ClassificationResult(
    is_swap_related=False,
    reason="Discusses interest rate policy, not a swap agreement.",
)

_TRANSLATION_EN = TranslationResult(
    body_en="The Federal Reserve and ECB announced a swap line...",
    original_language="en",
    was_translated=False,
)

_BILATERAL_EXTRACTION = ExtractionResult(
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
            announcement_date="2020-03-19",
            effective_date="2020-03-19",
            maturity_text="at least six months",
            duration_description="6 months",
            conditions="overnight index swap rate plus 25 basis points",
            reasons_for_swap="lessen strains in global U.S. dollar funding markets",
            raw_extract="The Federal Reserve announced the establishment...",
        ),
    ],
)

_HTML_RESULT_SWAP = HtmlExtractResult(
    url="https://www.federalreserve.gov/newsevents/pressreleases/monetary20200319a.htm",
    title="Federal Reserve announces temporary USD liquidity arrangements",
    body=(
        "The Federal Reserve on Thursday announced the establishment of "
        "temporary U.S. dollar liquidity arrangements (swap lines) with "
        "the Reserve Bank of Australia, the Banco Central do Brasil..."
    ),
    publication_date=datetime.date(2020, 3, 19),
    language="en",
)

_HTML_RESULT_NON_SWAP = HtmlExtractResult(
    url="https://www.federalreserve.gov/newsevents/pressreleases/monetary20240320a.htm",
    title="Federal Reserve issues FOMC statement",
    body=(
        "The Federal Open Market Committee decided to maintain the target "
        "range for the federal funds rate at 5-1/4 to 5-1/2 percent."
    ),
    publication_date=datetime.date(2024, 3, 20),
    language="en",
)

_HTML_RESULT_PDF = HtmlExtractResult(
    url="https://www.snb.ch/en/press/swap-2024-01.pdf",
    title="SNB-BoJ Swap Line Agreement",
    body=(
        "The Swiss National Bank and Bank of Japan announced a bilateral "
        "currency swap arrangement to improve liquidity conditions."
    ),
    publication_date=datetime.date(2024, 2, 10),
    language="en",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PATCH_BASE = "cbs.pipeline.orchestrator"


def _make_orchestrator(db: duckdb.DuckDBPyConnection) -> Orchestrator:
    """Build an orchestrator with a dummy LLM (all calls are patched)."""
    mock_llm = MagicMock()
    mock_browser = MagicMock()
    return Orchestrator(conn=db, llm=mock_llm, browser=mock_browser)


# ---------------------------------------------------------------------------
# test_full_pipeline_for_single_press_release
# ---------------------------------------------------------------------------


class TestFullPipelineForSinglePressRelease:
    """Fixture HTML in → swap rows out."""

    @patch(f"{_PATCH_BASE}.extract_swaps", return_value=_BILATERAL_EXTRACTION)
    @patch(f"{_PATCH_BASE}.classify_press_release", return_value=_SWAP_CLASSIFICATION)
    @patch(f"{_PATCH_BASE}.translate_text", return_value=_TRANSLATION_EN)
    @patch(f"{_PATCH_BASE}.detect_language", return_value="en")
    def test_swap_press_release_produces_two_swap_rows(
        self,
        _mock_detect: MagicMock,
        _mock_translate: MagicMock,
        _mock_classify: MagicMock,
        _mock_extract: MagicMock,
        db: duckdb.DuckDBPyConnection,
    ) -> None:
        orch = _make_orchestrator(db)
        result = orch.process_press_release(
            _HTML_RESULT_SWAP,
            bank_name="Federal Reserve",
            country="US",
        )

        # Pipeline result
        assert result.press_release_id is not None
        assert len(result.swap_ids) == 2
        assert result.skipped_duplicate is False
        assert result.skipped_not_swap is False

        # DB: press release inserted and marked processed
        pr = db.execute(
            "SELECT processed, is_swap_related, body_en, original_language "
            "FROM press_releases WHERE url = ?",
            [_HTML_RESULT_SWAP.url],
        ).fetchone()
        assert pr is not None
        assert pr[0] is True  # processed
        assert pr[1] is True  # is_swap_related
        assert pr[2] is not None  # body_en populated
        assert pr[3] is not None  # original_language populated

        # DB: two swap rows
        swaps = db.execute("SELECT * FROM swaps").fetchall()
        assert len(swaps) == 2


# ---------------------------------------------------------------------------
# test_pipeline_skips_non_swap_releases
# ---------------------------------------------------------------------------


class TestPipelineSkipsNonSwapReleases:
    """Non-swap press release → zero swap rows, still marked processed."""

    @patch(
        f"{_PATCH_BASE}.classify_press_release",
        return_value=_NON_SWAP_CLASSIFICATION,
    )
    @patch(f"{_PATCH_BASE}.translate_text", return_value=_TRANSLATION_EN)
    @patch(f"{_PATCH_BASE}.detect_language", return_value="en")
    def test_non_swap_produces_zero_swap_rows(
        self,
        _mock_detect: MagicMock,
        _mock_translate: MagicMock,
        _mock_classify: MagicMock,
        db: duckdb.DuckDBPyConnection,
    ) -> None:
        orch = _make_orchestrator(db)
        result = orch.process_press_release(
            _HTML_RESULT_NON_SWAP,
            bank_name="Federal Reserve",
            country="US",
        )

        assert result.press_release_id is not None
        assert len(result.swap_ids) == 0
        assert result.skipped_not_swap is True

        # DB: no swaps
        swap_count = db.execute("SELECT COUNT(*) FROM swaps").fetchone()
        assert swap_count is not None
        assert swap_count[0] == 0

        # DB: PR is processed with is_swap_related=False
        pr = db.execute(
            "SELECT processed, is_swap_related, classification_reason "
            "FROM press_releases WHERE url = ?",
            [_HTML_RESULT_NON_SWAP.url],
        ).fetchone()
        assert pr is not None
        assert pr[0] is True  # processed
        assert pr[1] is False  # is_swap_related
        assert pr[2] is not None  # classification_reason populated


# ---------------------------------------------------------------------------
# test_pipeline_deduplicates_on_rerun
# ---------------------------------------------------------------------------


class TestPipelineDeduplicatesOnRerun:
    """Same URL processed twice → second run is a no-op."""

    @patch(f"{_PATCH_BASE}.extract_swaps", return_value=_BILATERAL_EXTRACTION)
    @patch(f"{_PATCH_BASE}.classify_press_release", return_value=_SWAP_CLASSIFICATION)
    @patch(f"{_PATCH_BASE}.translate_text", return_value=_TRANSLATION_EN)
    @patch(f"{_PATCH_BASE}.detect_language", return_value="en")
    def test_duplicate_url_skipped_on_rerun(
        self,
        _mock_detect: MagicMock,
        _mock_translate: MagicMock,
        _mock_classify: MagicMock,
        _mock_extract: MagicMock,
        db: duckdb.DuckDBPyConnection,
    ) -> None:
        orch = _make_orchestrator(db)

        # First run
        result1 = orch.process_press_release(
            _HTML_RESULT_SWAP,
            bank_name="Federal Reserve",
            country="US",
        )
        assert result1.press_release_id is not None
        assert len(result1.swap_ids) == 2

        # Second run — same URL
        result2 = orch.process_press_release(
            _HTML_RESULT_SWAP,
            bank_name="Federal Reserve",
            country="US",
        )
        assert result2.skipped_duplicate is True
        assert result2.press_release_id is None
        assert len(result2.swap_ids) == 0

        # DB: exactly 1 press release and 2 swaps
        pr_count = db.execute("SELECT COUNT(*) FROM press_releases").fetchone()
        swap_count = db.execute("SELECT COUNT(*) FROM swaps").fetchone()
        assert pr_count is not None and pr_count[0] == 1
        assert swap_count is not None and swap_count[0] == 2


# ---------------------------------------------------------------------------
# test_pipeline_handles_pdf_press_release
# ---------------------------------------------------------------------------


class TestPipelineHandlesPdfPressRelease:
    """PDF press release flows through the full pipeline with source_type='pdf'."""

    @patch(f"{_PATCH_BASE}.extract_swaps", return_value=_BILATERAL_EXTRACTION)
    @patch(f"{_PATCH_BASE}.classify_press_release", return_value=_SWAP_CLASSIFICATION)
    @patch(f"{_PATCH_BASE}.translate_text", return_value=_TRANSLATION_EN)
    @patch(f"{_PATCH_BASE}.detect_language", return_value="en")
    def test_pdf_source_type_stored_correctly(
        self,
        _mock_detect: MagicMock,
        _mock_translate: MagicMock,
        _mock_classify: MagicMock,
        _mock_extract: MagicMock,
        db: duckdb.DuckDBPyConnection,
    ) -> None:
        orch = _make_orchestrator(db)

        # Build an HtmlExtractResult that represents PDF-extracted content
        # (the orchestrator receives already-extracted content regardless of source)
        result = orch.process_press_release(
            _HTML_RESULT_PDF,
            bank_name="Swiss National Bank",
            country="Switzerland",
            source_type="pdf",
        )

        assert result.press_release_id is not None
        assert len(result.swap_ids) == 2

        # DB: source_type should be "pdf"
        pr = db.execute(
            "SELECT source_type FROM press_releases WHERE url = ?",
            [_HTML_RESULT_PDF.url],
        ).fetchone()
        assert pr is not None
        assert pr[0] == "pdf"


# ---------------------------------------------------------------------------
# test_per_stage_model_tiering (Slice 1.5.2)
# ---------------------------------------------------------------------------


class TestPerStageModelTiering:
    """Per-stage LLMs are forwarded to the correct pipeline functions."""

    @patch(f"{_PATCH_BASE}.extract_swaps", return_value=_BILATERAL_EXTRACTION)
    @patch(f"{_PATCH_BASE}.classify_press_release", return_value=_SWAP_CLASSIFICATION)
    @patch(f"{_PATCH_BASE}.translate_text", return_value=_TRANSLATION_EN)
    @patch(f"{_PATCH_BASE}.detect_language", return_value="en")
    def test_classifier_uses_classify_llm(
        self,
        mock_detect: MagicMock,
        mock_translate: MagicMock,
        mock_classify: MagicMock,
        _mock_extract: MagicMock,
        db: duckdb.DuckDBPyConnection,
    ) -> None:
        default_llm = MagicMock(name="default")
        classify_llm = MagicMock(name="classify")
        mock_browser = MagicMock()
        orch = Orchestrator(
            conn=db,
            llm=default_llm,
            browser=mock_browser,
            classify_llm=classify_llm,
        )

        orch.process_press_release(_HTML_RESULT_SWAP, bank_name="Fed", country="US")

        # Classifier should receive the classify LLM
        mock_classify.assert_called_once()
        assert mock_classify.call_args[0][0] is classify_llm

        # Translate should use default LLM (no translate_llm override)
        mock_detect.assert_called_once()
        assert mock_detect.call_args[0][0] is default_llm
        mock_translate.assert_called_once()
        assert mock_translate.call_args[0][0] is default_llm

    @patch(f"{_PATCH_BASE}.extract_swaps", return_value=_BILATERAL_EXTRACTION)
    @patch(f"{_PATCH_BASE}.classify_press_release", return_value=_SWAP_CLASSIFICATION)
    @patch(f"{_PATCH_BASE}.translate_text", return_value=_TRANSLATION_EN)
    @patch(f"{_PATCH_BASE}.detect_language", return_value="en")
    def test_extractor_uses_extract_llm(
        self,
        _mock_detect: MagicMock,
        _mock_translate: MagicMock,
        _mock_classify: MagicMock,
        mock_extract: MagicMock,
        db: duckdb.DuckDBPyConnection,
    ) -> None:
        default_llm = MagicMock(name="default")
        extract_llm = MagicMock(name="extract")
        mock_browser = MagicMock()
        orch = Orchestrator(
            conn=db,
            llm=default_llm,
            browser=mock_browser,
            extract_llm=extract_llm,
        )

        orch.process_press_release(_HTML_RESULT_SWAP, bank_name="Fed", country="US")

        # Extractor should receive the extract LLM
        mock_extract.assert_called_once()
        assert mock_extract.call_args[0][0] is extract_llm

    @patch(f"{_PATCH_BASE}.extract_swaps", return_value=_BILATERAL_EXTRACTION)
    @patch(f"{_PATCH_BASE}.classify_press_release", return_value=_SWAP_CLASSIFICATION)
    @patch(f"{_PATCH_BASE}.translate_text", return_value=_TRANSLATION_EN)
    @patch(f"{_PATCH_BASE}.detect_language", return_value="en")
    def test_default_llm_used_when_no_overrides(
        self,
        mock_detect: MagicMock,
        mock_translate: MagicMock,
        mock_classify: MagicMock,
        mock_extract: MagicMock,
        db: duckdb.DuckDBPyConnection,
    ) -> None:
        default_llm = MagicMock(name="default")
        mock_browser = MagicMock()
        orch = Orchestrator(conn=db, llm=default_llm, browser=mock_browser)

        orch.process_press_release(_HTML_RESULT_SWAP, bank_name="Fed", country="US")

        # All stages should use the default LLM
        assert mock_detect.call_args[0][0] is default_llm
        assert mock_translate.call_args[0][0] is default_llm
        assert mock_classify.call_args[0][0] is default_llm
        assert mock_extract.call_args[0][0] is default_llm
