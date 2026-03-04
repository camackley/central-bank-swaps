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
        text_content="Body text of the press release.",
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
        orchestrator = MagicMock(spec=Orchestrator)

        # 2 PRs yield swaps, 1 is not swap-related
        swap_id = uuid.uuid4()
        orchestrator.process_press_release.side_effect = [
            PipelineResult(press_release_id=uuid.uuid4(), swap_ids=[swap_id, swap_id]),
            PipelineResult(press_release_id=uuid.uuid4(), skipped_not_swap=True),
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
