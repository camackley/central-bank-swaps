"""Tests for DefaultBankProcessor — Slice 1.13."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

from cbs.config.banks import BankConfig
from cbs.pipeline.bank_processor import DefaultBankProcessor
from cbs.pipeline.models import BankProcessingResult
from cbs.pipeline.orchestrator import Orchestrator, PipelineResult
from cbs.scraper.browser import BrowserAdapter, PageLink, PageSnapshot
from cbs.scraper.models import DiscoveredPressRelease, NavigationResult
from cbs.scraper.pdf_extractor import PDFChunk, PDFExtractResult

_PDF_BODY = "PDF body text " + "x" * 60

_PR_BODY = (
    "Body text of the press release about central"
    " bank swap agreements and monetary policy."
)

_MOCK_HTML = (
    f"<html><body><article><h1>Press Release</h1>"
    f"<p>{_PR_BODY}</p></article></body></html>"
)
_EMPTY_HTML = "<html><body></body></html>"
_NOT_FOUND_HTML = (
    "<html><body><h1>Page not found</h1>"
    "<p>The page you are looking for does not exist. Error 404.</p></body></html>"
)


def _make_bank() -> BankConfig:
    return BankConfig(
        name="Federal Reserve",
        country="US",
        homepage_url="https://www.federalreserve.gov",  # type: ignore[arg-type]
    )


def _make_snapshot(url: str = "https://example.com/pr1") -> PageSnapshot:
    return PageSnapshot(
        url=url,
        title="Press Release",
        text_content=(
            "Body text of the press release about central"
            " bank swap agreements and monetary policy."
        ),
        links=[PageLink(text="Link", url="https://example.com", element_ref="e0")],
    )


def _make_nav_result(urls: list[str]) -> NavigationResult:
    return NavigationResult(
        bank_name="Federal Reserve",
        press_releases=[
            DiscoveredPressRelease(url=u, title=f"PR {i}") for i, u in enumerate(urls)
        ],
        navigation_steps=[],
        listing_page_url=None,
        pages_visited=1,
        used_direct_url=True,
    )


class TestProcessBankCallsFindPressReleases:
    @patch("cbs.pipeline.bank_processor.find_press_releases")
    def test_calls_find_press_releases(self, mock_find: MagicMock) -> None:
        mock_find.return_value = _make_nav_result([])
        browser = MagicMock(spec=BrowserAdapter)
        orchestrator = MagicMock(spec=Orchestrator)
        bank = _make_bank()

        processor = DefaultBankProcessor(
            orchestrator=orchestrator, browser=browser, llm=MagicMock()
        )
        processor.process_bank(bank)

        mock_find.assert_called_once()
        call_kwargs = mock_find.call_args
        assert call_kwargs[0][0] == bank
        assert call_kwargs[0][1] is browser


class TestProcessBankFetchesEachUrl:
    @patch("cbs.pipeline.bank_processor.find_press_releases")
    def test_fetches_each_discovered_url(self, mock_find: MagicMock) -> None:
        urls = [
            "https://example.com/pr1",
            "https://example.com/pr2",
            "https://example.com/pr3",
        ]
        mock_find.return_value = _make_nav_result(urls)
        browser = MagicMock(spec=BrowserAdapter)
        browser.navigate.return_value = _make_snapshot()
        browser.get_page_html.return_value = _MOCK_HTML
        orchestrator = MagicMock(spec=Orchestrator)
        orchestrator.process_press_release.return_value = PipelineResult()

        processor = DefaultBankProcessor(
            orchestrator=orchestrator, browser=browser, llm=MagicMock()
        )
        processor.process_bank(_make_bank())

        assert browser.navigate.call_count == 3


class TestProcessBankPassesResultToOrchestrator:
    @patch("cbs.pipeline.bank_processor.find_press_releases")
    def test_passes_result_to_orchestrator(self, mock_find: MagicMock) -> None:
        mock_find.return_value = _make_nav_result(["https://example.com/pr1"])
        browser = MagicMock(spec=BrowserAdapter)
        browser.navigate.return_value = _make_snapshot()
        browser.get_page_html.return_value = _MOCK_HTML
        orchestrator = MagicMock(spec=Orchestrator)
        orchestrator.process_press_release.return_value = PipelineResult()

        processor = DefaultBankProcessor(
            orchestrator=orchestrator, browser=browser, llm=MagicMock()
        )
        processor.process_bank(_make_bank())

        orchestrator.process_press_release.assert_called_once()
        call_kwargs = orchestrator.process_press_release.call_args
        assert call_kwargs.kwargs["bank_name"] == "Federal Reserve"
        assert call_kwargs.kwargs["country"] == "US"


class TestProcessBankReturnsCorrectCounts:
    @patch("cbs.pipeline.bank_processor.find_press_releases")
    def test_returns_correct_counts(self, mock_find: MagicMock) -> None:
        urls = [
            "https://example.com/pr1",
            "https://example.com/pr2",
            "https://example.com/pr3",
        ]
        mock_find.return_value = _make_nav_result(urls)
        browser = MagicMock(spec=BrowserAdapter)
        browser.navigate.return_value = _make_snapshot()
        browser.get_page_html.return_value = _MOCK_HTML
        orchestrator = MagicMock(spec=Orchestrator)

        # 2 PRs yield swaps, 1 is not swap-related
        swap_id = uuid.uuid4()
        orchestrator.process_press_release.side_effect = [
            PipelineResult(press_release_id=uuid.uuid4(), swap_ids=[swap_id, swap_id]),
            PipelineResult(press_release_id=uuid.uuid4()),
            PipelineResult(press_release_id=uuid.uuid4(), swap_ids=[swap_id]),
        ]

        processor = DefaultBankProcessor(
            orchestrator=orchestrator, browser=browser, llm=MagicMock()
        )
        result = processor.process_bank(_make_bank())

        assert isinstance(result, BankProcessingResult)
        assert result.press_releases_found == 3
        assert result.swaps_extracted == 3
        assert result.bank_name == "Federal Reserve"
        assert not result.errors


class TestBotChallengeSkipped:
    @patch("cbs.pipeline.bank_processor.find_press_releases")
    def test_bot_challenge_skipped(self, mock_find: MagicMock) -> None:
        urls = [
            "https://example.com/pr1",
            "https://example.com/pr2",
        ]
        mock_find.return_value = _make_nav_result(urls)
        browser = MagicMock(spec=BrowserAdapter)
        # First PR redirects to Radware bot challenge
        radware_snapshot = PageSnapshot(
            url="https://validate.perfdrive.com/abc123/?ssa=xxx",
            title="",
            text_content="",
        )
        browser.navigate.side_effect = [
            radware_snapshot,
            _make_snapshot("https://example.com/pr2"),
        ]
        browser.get_page_html.return_value = _MOCK_HTML
        orchestrator = MagicMock(spec=Orchestrator)
        orchestrator.process_press_release.return_value = PipelineResult(
            press_release_id=uuid.uuid4()
        )

        processor = DefaultBankProcessor(
            orchestrator=orchestrator, browser=browser, llm=MagicMock()
        )
        result = processor.process_bank(_make_bank())

        assert len(result.errors) == 1
        assert "Bot challenge" in result.errors[0]
        assert result.press_releases_found == 1


class TestNavigationErrorCapturedContinues:
    @patch("cbs.pipeline.bank_processor.find_press_releases")
    def test_navigation_error_captured_continues(self, mock_find: MagicMock) -> None:
        urls = [
            "https://example.com/pr1",
            "https://example.com/pr2",
        ]
        mock_find.return_value = _make_nav_result(urls)
        browser = MagicMock(spec=BrowserAdapter)
        browser.navigate.side_effect = [
            RuntimeError("connection lost"),
            _make_snapshot("https://example.com/pr2"),
        ]
        browser.get_page_html.return_value = _MOCK_HTML
        orchestrator = MagicMock(spec=Orchestrator)
        orchestrator.process_press_release.return_value = PipelineResult(
            press_release_id=uuid.uuid4()
        )

        processor = DefaultBankProcessor(
            orchestrator=orchestrator, browser=browser, llm=MagicMock()
        )
        result = processor.process_bank(_make_bank())

        assert len(result.errors) == 1
        assert "connection lost" in result.errors[0]
        assert result.press_releases_found == 1


class TestOrchestratorErrorCapturedContinues:
    @patch("cbs.pipeline.bank_processor.find_press_releases")
    def test_orchestrator_error_captured_continues(self, mock_find: MagicMock) -> None:
        urls = [
            "https://example.com/pr1",
            "https://example.com/pr2",
        ]
        mock_find.return_value = _make_nav_result(urls)
        browser = MagicMock(spec=BrowserAdapter)
        browser.navigate.return_value = _make_snapshot()
        browser.get_page_html.return_value = _MOCK_HTML
        orchestrator = MagicMock(spec=Orchestrator)
        orchestrator.process_press_release.side_effect = [
            ValueError("LLM failed"),
            PipelineResult(press_release_id=uuid.uuid4()),
        ]

        processor = DefaultBankProcessor(
            orchestrator=orchestrator, browser=browser, llm=MagicMock()
        )
        result = processor.process_bank(_make_bank())

        assert len(result.errors) == 1
        assert "LLM failed" in result.errors[0]
        assert result.press_releases_found == 1


class TestBankNameAndCountryForwarded:
    @patch("cbs.pipeline.bank_processor.find_press_releases")
    def test_bank_name_and_country_forwarded(self, mock_find: MagicMock) -> None:
        mock_find.return_value = _make_nav_result(["https://example.com/pr1"])
        browser = MagicMock(spec=BrowserAdapter)
        browser.navigate.return_value = _make_snapshot()
        browser.get_page_html.return_value = _MOCK_HTML
        orchestrator = MagicMock(spec=Orchestrator)
        orchestrator.process_press_release.return_value = PipelineResult()

        bank = BankConfig(
            name="ECB",
            country="Eurozone",
            homepage_url="https://www.ecb.europa.eu",  # type: ignore[arg-type]
        )
        processor = DefaultBankProcessor(
            orchestrator=orchestrator, browser=browser, llm=MagicMock()
        )
        processor.process_bank(bank)

        call_kwargs = orchestrator.process_press_release.call_args
        assert call_kwargs.kwargs["bank_name"] == "ECB"
        assert call_kwargs.kwargs["country"] == "Eurozone"


# ---------------------------------------------------------------------------
# PDF routing tests
# ---------------------------------------------------------------------------


class TestPdfUrlDownloadsAndExtracts:
    """PDF URLs should be downloaded and extracted, not navigated via PinchTab."""

    @patch("cbs.pipeline.bank_processor._download_and_extract_pdf")
    @patch("cbs.pipeline.bank_processor.find_press_releases")
    def test_pdf_url_downloads_and_extracts(
        self, mock_find: MagicMock, mock_pdf: MagicMock
    ) -> None:
        urls = ["https://example.com/docs/report.pdf"]
        mock_find.return_value = _make_nav_result(urls)
        mock_pdf.return_value = PDFExtractResult(
            text=_PDF_BODY,
            page_count=1,
            chunks=[
                PDFChunk(
                    text=_PDF_BODY,
                    start_page=1,
                    end_page=1,
                )
            ],
            extractor="pymupdf",
        )
        browser = MagicMock(spec=BrowserAdapter)
        orchestrator = MagicMock(spec=Orchestrator)
        orchestrator.process_press_release.return_value = PipelineResult(
            press_release_id=uuid.uuid4()
        )

        processor = DefaultBankProcessor(
            orchestrator=orchestrator, browser=browser, llm=MagicMock()
        )
        processor.process_bank(_make_bank())

        # Browser.navigate should NOT be called for PDF URLs
        browser.navigate.assert_not_called()
        mock_pdf.assert_called_once_with("https://example.com/docs/report.pdf")

    @patch("cbs.pipeline.bank_processor._download_and_extract_pdf")
    @patch("cbs.pipeline.bank_processor.find_press_releases")
    def test_pdf_body_sent_to_orchestrator(
        self, mock_find: MagicMock, mock_pdf: MagicMock
    ) -> None:
        urls = ["https://example.com/docs/report.pdf"]
        mock_find.return_value = _make_nav_result(urls)
        mock_pdf.return_value = PDFExtractResult(
            text=_PDF_BODY,
            page_count=1,
            chunks=[
                PDFChunk(
                    text=_PDF_BODY,
                    start_page=1,
                    end_page=1,
                )
            ],
            extractor="pymupdf",
        )
        browser = MagicMock(spec=BrowserAdapter)
        orchestrator = MagicMock(spec=Orchestrator)
        orchestrator.process_press_release.return_value = PipelineResult(
            press_release_id=uuid.uuid4()
        )

        processor = DefaultBankProcessor(
            orchestrator=orchestrator, browser=browser, llm=MagicMock()
        )
        processor.process_bank(_make_bank())

        call_args = orchestrator.process_press_release.call_args[0][0]
        assert "PDF body text" in call_args.body

    @patch("cbs.pipeline.bank_processor._download_and_extract_pdf")
    @patch("cbs.pipeline.bank_processor.find_press_releases")
    def test_pdf_extraction_error_captured(
        self, mock_find: MagicMock, mock_pdf: MagicMock
    ) -> None:
        urls = ["https://example.com/docs/broken.pdf", "https://example.com/pr1"]
        mock_find.return_value = _make_nav_result(urls)
        mock_pdf.side_effect = RuntimeError("download failed")
        browser = MagicMock(spec=BrowserAdapter)
        browser.navigate.return_value = _make_snapshot("https://example.com/pr1")
        browser.get_page_html.return_value = _MOCK_HTML
        orchestrator = MagicMock(spec=Orchestrator)
        orchestrator.process_press_release.return_value = PipelineResult(
            press_release_id=uuid.uuid4()
        )

        processor = DefaultBankProcessor(
            orchestrator=orchestrator, browser=browser, llm=MagicMock()
        )
        result = processor.process_bank(_make_bank())

        assert any("PDF extraction failed" in e for e in result.errors)
        # Second (non-PDF) URL should still be processed
        assert result.press_releases_found == 1


# ---------------------------------------------------------------------------
# Guard tests: empty body, 404, listing page
# ---------------------------------------------------------------------------


class TestEmptyBodySkipped:
    @patch("cbs.pipeline.bank_processor.find_press_releases")
    def test_empty_body_skipped(self, mock_find: MagicMock) -> None:
        mock_find.return_value = _make_nav_result(["https://example.com/pr1"])
        browser = MagicMock(spec=BrowserAdapter)
        browser.navigate.return_value = _make_snapshot()
        browser.get_page_html.return_value = _EMPTY_HTML
        orchestrator = MagicMock(spec=Orchestrator)

        processor = DefaultBankProcessor(
            orchestrator=orchestrator, browser=browser, llm=MagicMock()
        )
        result = processor.process_bank(_make_bank())

        orchestrator.process_press_release.assert_not_called()
        assert any("Empty/short body" in e for e in result.errors)


class TestErrorPageSkipped:
    @patch("cbs.pipeline.bank_processor.find_press_releases")
    def test_404_page_skipped(self, mock_find: MagicMock) -> None:
        mock_find.return_value = _make_nav_result(["https://example.com/missing"])
        browser = MagicMock(spec=BrowserAdapter)
        browser.navigate.return_value = _make_snapshot("https://example.com/missing")
        browser.get_page_html.return_value = _NOT_FOUND_HTML
        orchestrator = MagicMock(spec=Orchestrator)

        processor = DefaultBankProcessor(
            orchestrator=orchestrator, browser=browser, llm=MagicMock()
        )
        result = processor.process_bank(_make_bank())

        orchestrator.process_press_release.assert_not_called()
        assert any("Error page" in e for e in result.errors)


class TestListingPageUrlSkipped:
    @patch("cbs.pipeline.bank_processor.find_press_releases")
    def test_listing_page_url_skipped(self, mock_find: MagicMock) -> None:
        listing_url = "https://example.com/press-releases"
        nav = _make_nav_result([listing_url, "https://example.com/pr1"])
        nav.listing_page_url = listing_url
        mock_find.return_value = nav

        browser = MagicMock(spec=BrowserAdapter)
        browser.navigate.return_value = _make_snapshot("https://example.com/pr1")
        browser.get_page_html.return_value = _MOCK_HTML
        orchestrator = MagicMock(spec=Orchestrator)
        orchestrator.process_press_release.return_value = PipelineResult(
            press_release_id=uuid.uuid4()
        )

        processor = DefaultBankProcessor(
            orchestrator=orchestrator, browser=browser, llm=MagicMock()
        )
        result = processor.process_bank(_make_bank())

        # Only the non-listing URL should be processed
        assert result.press_releases_found == 1
        assert browser.navigate.call_count == 1
