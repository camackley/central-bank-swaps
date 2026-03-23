"""Tests for the agentic navigator."""

from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from cbs.config.banks import BankConfig
from cbs.scraper.browser import BrowserAdapter, PageLink, PageSnapshot
from cbs.scraper.models import NavigationResult
from cbs.scraper.navigator import (
    _extract_urls_from_html,
    _filter_off_domain,
    find_press_releases,
)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_bank(
    *,
    press_releases_url: str | None = None,
    wait_strategy: str = "networkidle",
    wait_for_selector: str | None = None,
) -> BankConfig:
    return BankConfig(
        name="Test Bank",
        country="Testland",
        homepage_url="https://www.testbank.org",
        press_releases_url=press_releases_url,
        page_load_timeout=10,
        wait_strategy=wait_strategy,  # type: ignore[arg-type]
        wait_for_selector=wait_for_selector,
    )


def _make_snapshot(
    url: str = "https://www.testbank.org",
    title: str = "Test Page",
    text_content: str = "Page content",
    links: list[PageLink] | None = None,
) -> PageSnapshot:
    return PageSnapshot(
        url=url,
        title=title,
        text_content=text_content,
        links=links or [],
    )


def _make_browser(
    snapshot: PageSnapshot | None = None,
    html: str = "<html><body>Test</body></html>",
) -> MagicMock:
    """Build a mock BrowserAdapter with sensible defaults."""
    browser = MagicMock(spec=BrowserAdapter)
    browser.navigate.return_value = snapshot or _make_snapshot()
    browser.get_snapshot.return_value = snapshot or _make_snapshot()
    browser.get_page_html.return_value = html
    browser.click.return_value = snapshot or _make_snapshot()
    return browser


def _llm_returning(urls: list[str]) -> MagicMock:
    """LLM mock that returns the given URLs as a JSON array."""
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=json.dumps(urls))
    return llm


# ---------------------------------------------------------------------------
# 1. Direct URL mode
# ---------------------------------------------------------------------------


class TestDirectUrlMode:
    """When press_releases_url is set, navigate directly — no LLM agent."""

    def test_navigates_to_configured_url(self) -> None:
        bank = _make_bank(press_releases_url="https://www.testbank.org/press")
        browser = _make_browser()
        llm = _llm_returning(["https://www.testbank.org/pr/1"])

        find_press_releases(bank, browser, llm, max_pages=1)

        browser.navigate.assert_called_once_with(
            "https://www.testbank.org/press",
            timeout=10,
            wait_strategy="networkidle",
            wait_for_selector=None,
        )

    def test_passes_wait_strategy_to_navigate(self) -> None:
        bank = _make_bank(
            press_releases_url="https://www.testbank.org/press",
            wait_strategy="domcontentloaded",
        )
        browser = _make_browser()
        llm = _llm_returning([])

        find_press_releases(bank, browser, llm, max_pages=1)

        browser.navigate.assert_called_once_with(
            "https://www.testbank.org/press",
            timeout=10,
            wait_strategy="domcontentloaded",
            wait_for_selector=None,
        )

    def test_passes_wait_for_selector_to_navigate(self) -> None:
        bank = _make_bank(
            press_releases_url="https://www.testbank.org/press",
            wait_for_selector="article",
        )
        browser = _make_browser()
        llm = _llm_returning([])

        find_press_releases(bank, browser, llm, max_pages=1)

        browser.navigate.assert_called_once_with(
            "https://www.testbank.org/press",
            timeout=10,
            wait_strategy="networkidle",
            wait_for_selector="article",
        )

    def test_calls_get_page_html(self) -> None:
        bank = _make_bank(press_releases_url="https://www.testbank.org/press")
        browser = _make_browser()
        llm = _llm_returning([])

        find_press_releases(bank, browser, llm, max_pages=1)

        browser.get_page_html.assert_called()

    def test_returns_urls_from_llm_extraction(self) -> None:
        bank = _make_bank(press_releases_url="https://www.testbank.org/press")
        browser = _make_browser()
        pr_url = "https://www.testbank.org/press/swap-ecb-2024"
        llm = _llm_returning([pr_url])

        result = find_press_releases(bank, browser, llm, max_pages=1)

        assert isinstance(result, NavigationResult)
        assert result.used_direct_url is True
        assert any(pr.url == pr_url for pr in result.press_releases)

    def test_fallback_to_snapshot_links_when_llm_returns_empty(self) -> None:
        """When LLM extraction returns [], fall back to links from snapshot."""
        bank = _make_bank(press_releases_url="https://www.testbank.org/press")
        pr_link = PageLink(
            text="Swap agreement",
            url="https://www.testbank.org/pr/1",
            element_ref="https://www.testbank.org/pr/1",
        )
        browser = _make_browser(snapshot=_make_snapshot(links=[pr_link]))
        llm = _llm_returning([])

        result = find_press_releases(bank, browser, llm, max_pages=1)

        assert any(
            pr.url == "https://www.testbank.org/pr/1" for pr in result.press_releases
        )

    def test_agent_not_invoked_for_direct_url(self) -> None:
        bank = _make_bank(press_releases_url="https://www.testbank.org/press")
        browser = _make_browser()
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content="[]")

        find_press_releases(bank, browser, llm, max_pages=1)

        llm.bind_tools.assert_not_called()

    def test_marks_result_as_direct_url(self) -> None:
        bank = _make_bank(press_releases_url="https://www.testbank.org/press")
        browser = _make_browser()
        llm = _llm_returning([])

        result = find_press_releases(bank, browser, llm, max_pages=1)

        assert result.used_direct_url is True
        assert result.bank_name == "Test Bank"
        assert result.listing_page_url == "https://www.testbank.org/press"


# ---------------------------------------------------------------------------
# 2. HTML URL extraction
# ---------------------------------------------------------------------------


class TestHtmlUrlExtraction:
    """_extract_urls_from_html() asks the LLM to identify press release URLs."""

    def test_returns_press_release_urls(self) -> None:
        html = "<html><body><a href='/news/2024/swap'>Swap agreement</a></body></html>"
        llm = _llm_returning(["https://example.com/news/2024/swap"])

        result = _extract_urls_from_html(
            html, llm, bank_name="Test Bank", page_url="https://example.com/press"
        )

        assert len(result) == 1
        assert result[0].url == "https://example.com/news/2024/swap"

    def test_prompt_includes_bank_name_and_page_url(self) -> None:
        html = "<html><body></body></html>"
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content="[]")

        _extract_urls_from_html(
            html,
            llm,
            bank_name="Federal Reserve",
            page_url="https://fed.gov/press",
        )

        prompt_text = llm.invoke.call_args[0][0][0].content
        assert "Federal Reserve" in prompt_text
        assert "https://fed.gov/press" in prompt_text

    def test_handles_json_with_preamble(self) -> None:
        """Claude sometimes adds text before the JSON array — we extract it."""
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(
            content='Here are the URLs:\n["https://example.com/pr/1"]\nDone.'
        )

        result = _extract_urls_from_html(
            "<html/>", llm, bank_name="Bank", page_url="https://example.com"
        )

        assert len(result) == 1
        assert result[0].url == "https://example.com/pr/1"

    def test_falls_back_to_empty_on_invalid_json(self) -> None:
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content="not valid json")

        result = _extract_urls_from_html(
            "<html/>", llm, bank_name="Bank", page_url="https://example.com"
        )

        assert result == []

    def test_filters_non_http_urls(self) -> None:
        llm = _llm_returning(
            ["https://example.com/pr/1", "ftp://files.example.com/doc.pdf", ""]
        )

        result = _extract_urls_from_html(
            "<html/>", llm, bank_name="Bank", page_url="https://example.com"
        )

        assert len(result) == 1
        assert result[0].url == "https://example.com/pr/1"

    def test_truncates_large_html(self) -> None:
        """HTML larger than _HTML_CHAR_LIMIT is truncated before sending."""
        large_html = "x" * 500_000
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content="[]")

        _extract_urls_from_html(
            large_html, llm, bank_name="Bank", page_url="https://example.com"
        )

        prompt_text = llm.invoke.call_args[0][0][0].content
        # Prompt overhead + 150K char HTML limit
        assert len(prompt_text) < 160_000


# ---------------------------------------------------------------------------
# 3. Pagination
# ---------------------------------------------------------------------------


class TestPagination:
    """Pagination follows 'next page' links to discover more press releases."""

    def test_pagination_navigates_to_next_page(self) -> None:
        bank = _make_bank(press_releases_url="https://www.testbank.org/press")

        page1_links = [
            PageLink(
                text="Next",
                url="https://www.testbank.org/press?page=2",
                element_ref="https://www.testbank.org/press?page=2",
            ),
        ]
        page1 = _make_snapshot(links=page1_links)
        page2 = _make_snapshot(url="https://www.testbank.org/press?page=2", links=[])

        browser = MagicMock(spec=BrowserAdapter)
        browser.navigate.return_value = page1
        browser.get_snapshot.return_value = page2
        browser.click.return_value = page2
        browser.get_page_html.return_value = "<html/>"

        llm = FakeMessagesListChatModel(
            responses=[
                # page 1 extraction
                AIMessage(content='["https://www.testbank.org/pr/1"]'),
                # pagination: LLM finds next link
                AIMessage(
                    content='{"element_ref": "https://www.testbank.org/press?page=2"}'
                ),
                # page 2 extraction
                AIMessage(content='["https://www.testbank.org/pr/3"]'),
                # no more pages
                AIMessage(content="null"),
            ]
        )

        result = find_press_releases(bank, browser, llm, max_pages=3)

        assert result.pages_visited >= 2
        urls = [pr.url for pr in result.press_releases]
        assert "https://www.testbank.org/pr/1" in urls
        assert "https://www.testbank.org/pr/3" in urls

    def test_pagination_stops_at_max_pages(self) -> None:
        bank = _make_bank(press_releases_url="https://www.testbank.org/press")
        page = _make_snapshot(
            links=[
                PageLink(
                    text="Next",
                    url="https://www.testbank.org/press?page=2",
                    element_ref="https://www.testbank.org/press?page=2",
                )
            ]
        )
        browser = _make_browser(snapshot=page)

        llm = FakeMessagesListChatModel(
            responses=[
                AIMessage(content="[]"),
                AIMessage(
                    content='{"element_ref": "https://www.testbank.org/press?page=2"}'
                ),
                AIMessage(content="[]"),
            ]
        )

        result = find_press_releases(bank, browser, llm, max_pages=2)

        assert result.pages_visited <= 2

    def test_pagination_stops_when_no_next_link(self) -> None:
        bank = _make_bank(press_releases_url="https://www.testbank.org/press")
        page = _make_snapshot(links=[])
        browser = _make_browser(snapshot=page)

        llm = FakeMessagesListChatModel(
            responses=[
                AIMessage(content='["https://www.testbank.org/pr/1"]'),
                AIMessage(content="null"),
            ]
        )

        result = find_press_releases(bank, browser, llm, max_pages=5)

        assert result.pages_visited == 1


# ---------------------------------------------------------------------------
# 4. Navigation steps logged
# ---------------------------------------------------------------------------


class TestNavigationStepsLogged:
    """Every navigation step is recorded in the result and emitted via logging."""

    def test_direct_url_logs_single_step(self) -> None:
        bank = _make_bank(press_releases_url="https://www.testbank.org/press")
        browser = _make_browser()
        llm = _llm_returning([])

        result = find_press_releases(bank, browser, llm, max_pages=1)

        assert len(result.navigation_steps) >= 1
        step = result.navigation_steps[0]
        assert step.action == "direct_url"
        assert step.url == "https://www.testbank.org/press"

    def test_navigation_steps_have_required_fields(self) -> None:
        bank = _make_bank(press_releases_url="https://www.testbank.org/press")
        browser = _make_browser()
        llm = _llm_returning([])

        result = find_press_releases(bank, browser, llm, max_pages=1)

        for step in result.navigation_steps:
            assert step.step_number > 0
            assert step.action in ("direct_url", "navigate", "click", "paginate")
            assert step.url
            assert step.reasoning

    def test_steps_include_python_logging(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        bank = _make_bank(press_releases_url="https://www.testbank.org/press")
        browser = _make_browser()
        llm = _llm_returning([])

        with caplog.at_level(logging.INFO, logger="cbs.scraper.navigator"):
            find_press_releases(bank, browser, llm, max_pages=1)

        assert any("direct_url" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# 5. Agent discovery mode
# ---------------------------------------------------------------------------


class TestAgentDiscovery:
    """LLM agent navigates from homepage to discover press releases section."""

    def test_agent_discovery_used_when_no_press_releases_url(self) -> None:
        bank = _make_bank()  # no press_releases_url

        pr_url = "https://www.testbank.org/media/press/swap-2024"
        browser = MagicMock(spec=BrowserAdapter)
        browser.navigate.return_value = _make_snapshot(
            url="https://www.testbank.org",
            links=[
                PageLink(
                    text="Media",
                    url="https://www.testbank.org/media",
                    element_ref="https://www.testbank.org/media",
                )
            ],
        )
        listing = _make_snapshot(
            url="https://www.testbank.org/media/press",
            links=[
                PageLink(
                    text="Swap 2024",
                    url=pr_url,
                    element_ref=pr_url,
                )
            ],
        )
        browser.click.return_value = listing
        browser.get_snapshot.return_value = listing
        browser.get_page_html.return_value = f"<html><a href='{pr_url}'>Swap</a></html>"

        fake_llm = FakeMessagesListChatModel(
            responses=[
                AIMessage(
                    content="Clicking Media link",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "name": "click_link",
                            "args": {"element_ref": "https://www.testbank.org/media"},
                        }
                    ],
                ),
                AIMessage(
                    content="Extracting press release URLs",
                    tool_calls=[
                        {
                            "id": "call_2",
                            "name": "extract_press_release_urls",
                            "args": {},
                        }
                    ],
                ),
                AIMessage(content="Done."),
                # _extract_urls_from_html calls llm.invoke once more
                AIMessage(content=f'["{pr_url}"]'),
            ]
        )

        result = find_press_releases(bank, browser, fake_llm, max_pages=1)

        assert result.used_direct_url is False
        assert result.pages_visited >= 1

    def test_result_is_not_used_direct_url_for_discovery(self) -> None:
        bank = _make_bank()
        browser = MagicMock(spec=BrowserAdapter)
        browser.navigate.return_value = _make_snapshot(links=[])
        browser.get_snapshot.return_value = _make_snapshot(links=[])
        browser.get_page_html.return_value = "<html/>"

        fake_llm = FakeMessagesListChatModel(
            responses=[
                AIMessage(content="No press releases found."),
                AIMessage(content="[]"),
            ]
        )

        result = find_press_releases(bank, browser, fake_llm, max_pages=1)

        assert result.used_direct_url is False


# ---------------------------------------------------------------------------
# 6. Off-domain URL filtering
# ---------------------------------------------------------------------------


class TestOffDomainUrlsFiltered:
    """Off-domain URLs (social media, etc.) are removed from results."""

    def test_off_domain_urls_filtered(self) -> None:
        from cbs.scraper.models import DiscoveredPressRelease

        press_releases = [
            DiscoveredPressRelease(
                url="https://www.federalreserve.gov/pr/2024-01",
                title="Swap agreement",
            ),
            DiscoveredPressRelease(
                url="https://www.instagram.com/federalreserve",
                title="Instagram",
            ),
            DiscoveredPressRelease(
                url="https://twitter.com/federalreserve",
                title="Twitter",
            ),
            DiscoveredPressRelease(
                url="https://www.federalreserve.gov/pr/2024-02",
                title="Rate decision",
            ),
        ]

        filtered = _filter_off_domain(press_releases, "www.federalreserve.gov")

        assert len(filtered) == 2
        urls = [pr.url for pr in filtered]
        assert "https://www.federalreserve.gov/pr/2024-01" in urls
        assert "https://www.federalreserve.gov/pr/2024-02" in urls

    def test_subdomain_kept(self) -> None:
        from cbs.scraper.models import DiscoveredPressRelease

        press_releases = [
            DiscoveredPressRelease(
                url="https://press.testbank.org/release/1",
                title="PR 1",
            ),
        ]

        filtered = _filter_off_domain(press_releases, "www.testbank.org")

        assert len(filtered) == 1
