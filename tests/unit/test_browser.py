"""Tests for the Playwright browser adapter."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from cbs.scraper.browser import (
    BrowserAdapter,
    BrowserError,
    BrowserNavigationError,
    BrowserTimeoutError,
    PageLink,
    PageSnapshot,
    PlaywrightBrowserAdapter,
)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

_PAGE_URL = "https://example.com"
_PAGE_TITLE = "Example Domain"
_PAGE_TEXT = "This is example page text content."
_PAGE_LINKS = [
    {"text": "Page 1", "href": "https://example.com/page1"},
    {"text": "Page 2", "href": "https://example.com/page2"},
]
_PAGE_HTML = "<html><body><a href='https://example.com/page1'>Page 1</a></body></html>"


def _make_mock_page(
    url: str = _PAGE_URL,
    title: str = _PAGE_TITLE,
    text: str = _PAGE_TEXT,
    links: list[dict[str, str]] | None = None,
    html: str = _PAGE_HTML,
) -> MagicMock:
    """Build a mock Playwright Page with realistic default responses."""
    page = MagicMock()
    page.url = url
    page.title.return_value = title
    page.content.return_value = html
    page.goto.return_value = None
    page.wait_for_selector.return_value = None
    page.route.return_value = None
    page.close.return_value = None

    _links = links if links is not None else _PAGE_LINKS

    def mock_evaluate(expr: Any, *args: Any) -> Any:
        if "innerText" in str(expr):
            return text
        # _EXTRACT_LINKS_JS callable
        return _links

    page.evaluate.side_effect = mock_evaluate
    return page


@pytest.fixture()
def mock_page() -> MagicMock:
    """A mock Playwright Page with default content."""
    return _make_mock_page()


@pytest.fixture()
def adapter(mock_page: MagicMock) -> PlaywrightBrowserAdapter:
    """PlaywrightBrowserAdapter with an injected mock page, already navigated."""
    a = PlaywrightBrowserAdapter(_page=mock_page)
    a._current_url = _PAGE_URL
    a._current_title = _PAGE_TITLE
    return a


# ===========================================================================
# Navigate
# ===========================================================================


class TestNavigate:
    """navigate() returns PageSnapshot with url, title, text, and links."""

    def test_navigate_returns_page_snapshot(
        self, adapter: PlaywrightBrowserAdapter, mock_page: MagicMock
    ) -> None:
        mock_page.url = _PAGE_URL
        result = adapter.navigate(_PAGE_URL)

        assert isinstance(result, PageSnapshot)
        assert result.url == _PAGE_URL
        assert result.title == _PAGE_TITLE
        assert result.text_content == _PAGE_TEXT
        assert len(result.links) == 2

    def test_navigate_calls_page_goto(
        self, adapter: PlaywrightBrowserAdapter, mock_page: MagicMock
    ) -> None:
        adapter.navigate(_PAGE_URL, timeout=45)

        mock_page.goto.assert_called_once_with(
            _PAGE_URL, wait_until="networkidle", timeout=45_000
        )

    def test_navigate_respects_wait_strategy(
        self, adapter: PlaywrightBrowserAdapter, mock_page: MagicMock
    ) -> None:
        adapter.navigate(_PAGE_URL, wait_strategy="domcontentloaded")

        mock_page.goto.assert_called_once_with(
            _PAGE_URL, wait_until="domcontentloaded", timeout=30_000
        )

    def test_navigate_calls_wait_for_selector_when_configured(
        self, adapter: PlaywrightBrowserAdapter, mock_page: MagicMock
    ) -> None:
        adapter.navigate(_PAGE_URL, wait_for_selector="article")

        mock_page.wait_for_selector.assert_called_once_with("article", timeout=30_000)

    def test_navigate_skips_wait_for_selector_when_none(
        self, adapter: PlaywrightBrowserAdapter, mock_page: MagicMock
    ) -> None:
        adapter.navigate(_PAGE_URL)

        mock_page.wait_for_selector.assert_not_called()

    def test_navigate_timeout_raises_browser_timeout_error(
        self, adapter: PlaywrightBrowserAdapter, mock_page: MagicMock
    ) -> None:
        mock_page.goto.side_effect = PlaywrightTimeoutError("timed out")

        with pytest.raises(BrowserTimeoutError, match="timed out"):
            adapter.navigate(_PAGE_URL)

    def test_navigate_playwright_error_raises_navigation_error(
        self, adapter: PlaywrightBrowserAdapter, mock_page: MagicMock
    ) -> None:
        mock_page.goto.side_effect = PlaywrightError("net::ERR_NAME_NOT_RESOLVED")

        with pytest.raises(BrowserNavigationError):
            adapter.navigate("https://not-a-real-site.example")

    def test_navigate_without_start_raises(self) -> None:
        adapter = PlaywrightBrowserAdapter()  # no _page injected, no __enter__

        with pytest.raises(BrowserError, match="not started"):
            adapter.navigate(_PAGE_URL)

    def test_links_have_url_as_element_ref(
        self, adapter: PlaywrightBrowserAdapter
    ) -> None:
        """element_ref must equal url (used by click() for navigation)."""
        result = adapter.navigate(_PAGE_URL)

        for link in result.links:
            assert link.element_ref == link.url


# ===========================================================================
# Click
# ===========================================================================


class TestClick:
    """click() navigates to the given URL and returns a snapshot."""

    def test_click_navigates_to_url(
        self, adapter: PlaywrightBrowserAdapter, mock_page: MagicMock
    ) -> None:
        target = "https://example.com/page1"
        adapter.click(target)

        mock_page.goto.assert_called_once_with(
            target, wait_until="networkidle", timeout=30_000
        )

    def test_click_returns_page_snapshot(
        self, adapter: PlaywrightBrowserAdapter
    ) -> None:
        result = adapter.click("https://example.com/page1")

        assert isinstance(result, PageSnapshot)

    def test_click_without_prior_navigate_raises(self) -> None:
        page = _make_mock_page()
        adapter = PlaywrightBrowserAdapter(_page=page)
        # _current_url not set

        with pytest.raises(BrowserError, match="No active page"):
            adapter.click("https://example.com/page1")

    def test_click_timeout_raises_browser_timeout_error(
        self, adapter: PlaywrightBrowserAdapter, mock_page: MagicMock
    ) -> None:
        mock_page.goto.side_effect = PlaywrightTimeoutError("timed out")

        with pytest.raises(BrowserTimeoutError):
            adapter.click("https://example.com/page1")


# ===========================================================================
# get_snapshot
# ===========================================================================


class TestGetSnapshot:
    """get_snapshot() returns current page state without navigating."""

    def test_get_snapshot_returns_page_snapshot(
        self, adapter: PlaywrightBrowserAdapter
    ) -> None:
        result = adapter.get_snapshot()

        assert isinstance(result, PageSnapshot)
        assert result.text_content == _PAGE_TEXT
        assert len(result.links) == 2

    def test_get_snapshot_without_navigate_raises(self) -> None:
        page = _make_mock_page()
        adapter = PlaywrightBrowserAdapter(_page=page)

        with pytest.raises(BrowserError, match="No active page"):
            adapter.get_snapshot()


# ===========================================================================
# get_page_html
# ===========================================================================


class TestGetPageHtml:
    """get_page_html() returns the full rendered DOM HTML."""

    def test_returns_page_content(
        self, adapter: PlaywrightBrowserAdapter, mock_page: MagicMock
    ) -> None:
        result = adapter.get_page_html()

        mock_page.content.assert_called_once()
        assert result == _PAGE_HTML

    def test_get_page_html_without_navigate_raises(self) -> None:
        page = _make_mock_page()
        adapter = PlaywrightBrowserAdapter(_page=page)

        with pytest.raises(BrowserError, match="No active page"):
            adapter.get_page_html()

    def test_playwright_error_raises_navigation_error(
        self, adapter: PlaywrightBrowserAdapter, mock_page: MagicMock
    ) -> None:
        mock_page.content.side_effect = PlaywrightError("frame detached")

        with pytest.raises(BrowserNavigationError):
            adapter.get_page_html()


# ===========================================================================
# Lifecycle
# ===========================================================================


class TestLifecycle:
    """Adapter starts and stops Playwright cleanly."""

    def test_close_session_resets_state(
        self, adapter: PlaywrightBrowserAdapter
    ) -> None:
        adapter.close_session()

        assert adapter._current_url == ""
        assert adapter._current_title == ""

    def test_close_session_idempotent(self, adapter: PlaywrightBrowserAdapter) -> None:
        adapter.close_session()
        adapter.close_session()  # must not raise

    def test_context_manager_with_injected_page(self, mock_page: MagicMock) -> None:
        """With an injected page, __enter__/__exit__ don't touch Playwright."""
        with PlaywrightBrowserAdapter(_page=mock_page) as a:
            a._current_url = _PAGE_URL
            result = a.get_snapshot()

        assert isinstance(result, PageSnapshot)
        # The injected page itself is not closed (caller owns it)
        mock_page.close.assert_not_called()

    def test_browser_adapter_alias(self) -> None:
        """BrowserAdapter is an alias for PlaywrightBrowserAdapter."""
        assert BrowserAdapter is PlaywrightBrowserAdapter


# ===========================================================================
# Link extraction
# ===========================================================================


class TestExtractLinks:
    """_extract_links() extracts unique absolute links from the DOM."""

    def test_extracts_all_links(
        self, adapter: PlaywrightBrowserAdapter, mock_page: MagicMock
    ) -> None:
        links = adapter._extract_links(mock_page)

        assert len(links) == 2
        assert links[0] == PageLink(
            text="Page 1",
            url="https://example.com/page1",
            element_ref="https://example.com/page1",
        )

    def test_empty_when_js_fails(
        self, adapter: PlaywrightBrowserAdapter, mock_page: MagicMock
    ) -> None:
        mock_page.evaluate.side_effect = PlaywrightError("context destroyed")

        links = adapter._extract_links(mock_page)

        assert links == []

    def test_element_ref_equals_url(
        self, adapter: PlaywrightBrowserAdapter, mock_page: MagicMock
    ) -> None:
        links = adapter._extract_links(mock_page)

        for link in links:
            assert link.element_ref == link.url
