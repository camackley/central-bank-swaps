"""Tests for the PinchTab browser adapter — Slice 1.9."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import Mock

import httpx
import pytest

from cbs.scraper.browser import (
    BrowserAdapter,
    BrowserClient,
    BrowserConnectionError,
    BrowserError,
    BrowserNavigationError,
    BrowserTimeoutError,
    PageContent,
    PageLink,
    PageSnapshot,
)

# ---------------------------------------------------------------------------
# Helpers — build canned PinchTab HTTP responses
# ---------------------------------------------------------------------------

_PAGE_URL = "https://example.com"
_PAGE_TITLE = "Example Domain"
_PAGE_TEXT = "This is example page text content."
_PAGE_SNAPSHOT_NODES = [
    {
        "ref": "e0",
        "role": "RootWebArea",
        "name": "Example Domain",
        "depth": 0,
        "nodeId": 1,
    },
    {"ref": "e1", "role": "link", "name": "Page 1", "depth": 1, "nodeId": 2},
    {"ref": "e2", "role": "link", "name": "Page 2", "depth": 1, "nodeId": 3},
]
_DOM_LINKS = [
    {"t": "Page 1", "h": "https://example.com/page1"},
    {"t": "Page 2", "h": "https://example.com/page2"},
]


_FAKE_REQUEST = httpx.Request("GET", "http://localhost:9867")


def _json_response(
    data: dict[str, Any] | list[Any], status_code: int = 200
) -> httpx.Response:
    """Build an httpx.Response with JSON body."""
    return httpx.Response(status_code=status_code, json=data, request=_FAKE_REQUEST)


def _text_response(text: str, status_code: int = 200) -> httpx.Response:
    """Build an httpx.Response with plain text body."""
    return httpx.Response(status_code=status_code, text=text, request=_FAKE_REQUEST)


def _error_response(status_code: int = 500) -> httpx.Response:
    """Build an httpx.Response that will raise on raise_for_status()."""
    return httpx.Response(status_code=status_code, request=_FAKE_REQUEST)


# ---------------------------------------------------------------------------
# BrowserClient fixtures and routes
# ---------------------------------------------------------------------------


def _route_default(method: str, url: str, **kwargs: Any) -> httpx.Response:
    """Route mock HTTP calls to canned PinchTab responses."""
    path = url.replace("http://localhost:9867", "")

    if method == "get" and path == "/health":
        return _json_response({"status": "ok", "cdp": "", "tabs": 1})
    if method == "post" and path == "/navigate":
        return _json_response({"url": _PAGE_URL, "title": _PAGE_TITLE})
    if method == "get" and path == "/text":
        return _json_response(
            {"text": _PAGE_TEXT, "title": _PAGE_TITLE, "url": _PAGE_URL}
        )
    if method == "get" and path == "/snapshot":
        return _json_response(
            {
                "count": len(_PAGE_SNAPSHOT_NODES),
                "nodes": _PAGE_SNAPSHOT_NODES,
                "title": _PAGE_TITLE,
                "url": _PAGE_URL,
            }
        )

    msg = f"Unexpected {method.upper()} {path}"
    raise AssertionError(msg)


def _make_mock_client() -> Mock:
    """Build a mock httpx.Client that dispatches via _route_default."""
    client = Mock(spec=httpx.Client)

    def mock_post(url: str, **kwargs: Any) -> httpx.Response:
        return _route_default("post", url, **kwargs)

    def mock_get(url: str, **kwargs: Any) -> httpx.Response:
        return _route_default("get", url, **kwargs)

    client.post = Mock(side_effect=mock_post)
    client.get = Mock(side_effect=mock_get)
    return client


@pytest.fixture()
def mock_http() -> Mock:
    """Pre-configured mock httpx.Client with valid PinchTab responses."""
    return _make_mock_client()


@pytest.fixture()
def browser(mock_http: Mock) -> BrowserClient:
    """BrowserClient backed by the mock HTTP client, already started."""
    b = BrowserClient(_http_client=mock_http)
    b.start()
    return b


# ===========================================================================
# Test classes
# ===========================================================================


class TestNavigateToUrlReturnsContent:
    """navigate() returns PageContent with url, title, text, and snapshot."""

    def test_navigate_returns_page_content(self, browser: BrowserClient) -> None:
        result = browser.navigate(_PAGE_URL, timeout=30)

        assert isinstance(result, PageContent)
        assert result.url == _PAGE_URL
        assert result.title == _PAGE_TITLE
        assert result.text == _PAGE_TEXT
        assert "nodes" in result.snapshot

    def test_navigate_passes_block_images(
        self, browser: BrowserClient, mock_http: Mock
    ) -> None:
        browser.navigate(_PAGE_URL)

        navigate_calls = [
            c for c in mock_http.post.call_args_list if "/navigate" in str(c)
        ]
        assert len(navigate_calls) == 1
        payload = navigate_calls[0].kwargs.get("json", {})
        assert payload.get("blockImages") is True


class TestTimeoutRaisesAfterConfiguredSeconds:
    """navigate() raises BrowserTimeoutError when page load exceeds timeout."""

    def test_timeout_raises_browser_timeout_error(self, mock_http: Mock) -> None:
        def timeout_post(url: str, **kwargs: Any) -> httpx.Response:
            if "/navigate" in url:
                raise httpx.ReadTimeout("Timed out")
            return _route_default("post", url, **kwargs)

        mock_http.post = Mock(side_effect=timeout_post)

        b = BrowserClient(_http_client=mock_http)
        b.start()

        with pytest.raises(BrowserTimeoutError, match="timed out"):
            b.navigate(_PAGE_URL, timeout=5)

    def test_timeout_uses_provided_value(
        self, browser: BrowserClient, mock_http: Mock
    ) -> None:
        browser.navigate(_PAGE_URL, timeout=60)

        navigate_calls = [
            c for c in mock_http.post.call_args_list if "/navigate" in str(c)
        ]
        payload = navigate_calls[0].kwargs.get("json", {})
        assert payload["timeout"] == 60

    def test_default_timeout_is_30(
        self, browser: BrowserClient, mock_http: Mock
    ) -> None:
        browser.navigate(_PAGE_URL)

        navigate_calls = [
            c for c in mock_http.post.call_args_list if "/navigate" in str(c)
        ]
        payload = navigate_calls[0].kwargs.get("json", {})
        assert payload["timeout"] == 30


class TestPageLoadWaitsForRender:
    """navigate() fetches content only after PinchTab navigation completes."""

    def test_text_retrieved_after_navigation_completes(self, mock_http: Mock) -> None:
        """Call order: navigate -> text -> snapshot."""
        call_log: list[str] = []

        def tracking_post(url: str, **kwargs: Any) -> httpx.Response:
            path = url.replace("http://localhost:9867", "")
            call_log.append(f"POST {path}")
            return _route_default("post", url, **kwargs)

        def tracking_get(url: str, **kwargs: Any) -> httpx.Response:
            path = url.replace("http://localhost:9867", "")
            call_log.append(f"GET {path}")
            return _route_default("get", url, **kwargs)

        mock_http.post = Mock(side_effect=tracking_post)
        mock_http.get = Mock(side_effect=tracking_get)

        b = BrowserClient(_http_client=mock_http)
        b.start()
        b.navigate(_PAGE_URL)

        # Filter to only the navigate-related calls (skip health check)
        nav_calls = [c for c in call_log if "/health" not in c]

        # navigate must come before text and snapshot
        navigate_idx = next(i for i, c in enumerate(nav_calls) if "/navigate" in c)
        text_idx = next(i for i, c in enumerate(nav_calls) if "/text" in c)
        snapshot_idx = next(i for i, c in enumerate(nav_calls) if "/snapshot" in c)

        assert navigate_idx < text_idx
        assert navigate_idx < snapshot_idx

    def test_navigate_timeout_passed_to_pinchtab(
        self, browser: BrowserClient, mock_http: Mock
    ) -> None:
        browser.navigate(_PAGE_URL, timeout=45)

        navigate_calls = [
            c for c in mock_http.post.call_args_list if "/navigate" in str(c)
        ]
        payload = navigate_calls[0].kwargs.get("json", {})
        assert payload["timeout"] == 45


class TestBrowserClientLifecycle:
    """BrowserClient manages lifecycle correctly."""

    def test_context_manager_starts_and_stops(self, mock_http: Mock) -> None:
        with BrowserClient(_http_client=mock_http) as b:
            assert b._started is True

        assert b._started is False

    def test_stop_called_on_exception(self, mock_http: Mock) -> None:
        with pytest.raises(RuntimeError), BrowserClient(_http_client=mock_http):
            raise RuntimeError("something broke")

    def test_navigate_without_start_raises(self, mock_http: Mock) -> None:
        b = BrowserClient(_http_client=mock_http)

        with pytest.raises(BrowserError, match="not started"):
            b.navigate(_PAGE_URL)

    def test_stop_is_idempotent(self, mock_http: Mock) -> None:
        b = BrowserClient(_http_client=mock_http)
        b.start()
        b.stop()
        b.stop()  # Second call should not raise


class TestBrowserConnectionErrors:
    """Connection failures raise the correct BrowserError subclasses."""

    def test_unreachable_server_raises_connection_error(self, mock_http: Mock) -> None:
        mock_http.get = Mock(side_effect=httpx.ConnectError("Connection refused"))

        b = BrowserClient(_http_client=mock_http)

        with pytest.raises(BrowserConnectionError, match="Cannot connect"):
            b.start()

    def test_navigation_http_error_raises_browser_navigation_error(
        self, mock_http: Mock
    ) -> None:
        def error_post(url: str, **kwargs: Any) -> httpx.Response:
            if "/navigate" in url:
                resp = _error_response(500)
                raise httpx.HTTPStatusError(
                    "Server Error", request=resp.request, response=resp
                )
            return _route_default("post", url, **kwargs)

        mock_http.post = Mock(side_effect=error_post)

        b = BrowserClient(_http_client=mock_http)
        b.start()

        with pytest.raises(BrowserNavigationError, match="500"):
            b.navigate(_PAGE_URL)


# ===========================================================================
# BrowserAdapter tests (Slice 1.13)
# ===========================================================================

_ADAPTER_SNAPSHOT_NODES = [
    {"ref": "e0", "role": "link", "name": "Page 1", "depth": 1, "nodeId": 2},
    {"ref": "e1", "role": "link", "name": "Page 2", "depth": 1, "nodeId": 3},
    {"ref": "e3", "role": "heading", "name": "Not a link", "depth": 1, "nodeId": 4},
]
_ADAPTER_DOM_LINKS = [
    {"t": "Page 1", "h": "https://example.com/page1"},
    {"t": "Page 2", "h": "https://example.com/page2"},
    {"t": "No Ref Link", "h": "https://example.com/no-ref"},
]


def _adapter_route(method: str, url: str, **kwargs: Any) -> httpx.Response:
    """Route mock HTTP calls for BrowserAdapter tests."""
    path = url.replace("http://localhost:9867", "")

    if method == "get" and path == "/health":
        return _json_response({"status": "ok", "cdp": "", "tabs": 1})
    if method == "post" and path == "/navigate":
        return _json_response({"url": _PAGE_URL, "title": _PAGE_TITLE})
    if method == "get" and path == "/text":
        return _json_response(
            {"text": _PAGE_TEXT, "title": _PAGE_TITLE, "url": _PAGE_URL}
        )
    if method == "get" and path == "/snapshot":
        return _json_response(
            {
                "count": len(_ADAPTER_SNAPSHOT_NODES),
                "nodes": _ADAPTER_SNAPSHOT_NODES,
                "title": _PAGE_TITLE,
                "url": _PAGE_URL,
            }
        )
    if method == "post" and path == "/evaluate":
        return _json_response({"result": json.dumps(_ADAPTER_DOM_LINKS)})
    if method == "post" and path == "/action":
        return _json_response({"clicked": True})

    msg = f"Unexpected {method.upper()} {path}"
    raise AssertionError(msg)


def _make_adapter_mock() -> Mock:
    """Build a mock httpx.Client for BrowserAdapter tests."""
    client = Mock(spec=httpx.Client)
    client.post = Mock(side_effect=lambda url, **kw: _adapter_route("post", url, **kw))
    client.get = Mock(side_effect=lambda url, **kw: _adapter_route("get", url, **kw))
    return client


@pytest.fixture()
def adapter_http() -> Mock:
    return _make_adapter_mock()


@pytest.fixture()
def adapter(adapter_http: Mock) -> BrowserAdapter:
    return BrowserAdapter(_http_client=adapter_http)


class TestBrowserAdapterNavigate:
    """BrowserAdapter.navigate() returns PageSnapshot with links."""

    def test_navigate_returns_page_snapshot(self, adapter: BrowserAdapter) -> None:
        result = adapter.navigate(_PAGE_URL)

        assert isinstance(result, PageSnapshot)
        assert result.url == _PAGE_URL
        assert result.title == _PAGE_TITLE
        assert result.text_content == _PAGE_TEXT
        assert len(result.links) == 2
        assert result.links[0] == PageLink(
            text="Page 1", url="https://example.com/page1", element_ref="e0"
        )

    def test_navigate_checks_health_on_first_call(
        self, adapter: BrowserAdapter, adapter_http: Mock
    ) -> None:
        adapter.navigate(_PAGE_URL)

        health_calls = [
            c for c in adapter_http.get.call_args_list if "/health" in str(c)
        ]
        assert len(health_calls) == 1

    def test_navigate_reuses_connection_on_second_call(
        self, adapter: BrowserAdapter, adapter_http: Mock
    ) -> None:
        adapter.navigate(_PAGE_URL)
        adapter.navigate(_PAGE_URL)

        health_calls = [
            c for c in adapter_http.get.call_args_list if "/health" in str(c)
        ]
        assert len(health_calls) == 1

    def test_navigate_connection_error(self, adapter_http: Mock) -> None:
        adapter_http.get = Mock(side_effect=httpx.ConnectError("Connection refused"))
        adapter = BrowserAdapter(_http_client=adapter_http)

        with pytest.raises(BrowserConnectionError):
            adapter.navigate(_PAGE_URL)

    def test_navigate_timeout_error(self, adapter_http: Mock) -> None:
        original_post = adapter_http.post.side_effect

        def timeout_on_navigate(url: str, **kw: Any) -> httpx.Response:
            if "/navigate" in url:
                raise httpx.ReadTimeout("Timed out")
            return original_post(url, **kw)

        adapter_http.post = Mock(side_effect=timeout_on_navigate)
        adapter = BrowserAdapter(_http_client=adapter_http)

        with pytest.raises(BrowserTimeoutError):
            adapter.navigate(_PAGE_URL)


class TestBrowserAdapterClick:
    """BrowserAdapter.click() posts action and returns snapshot."""

    def test_click_posts_action_and_returns_snapshot(
        self, adapter: BrowserAdapter, adapter_http: Mock
    ) -> None:
        adapter.navigate(_PAGE_URL)
        result = adapter.click("e0")

        assert isinstance(result, PageSnapshot)
        action_calls = [
            c for c in adapter_http.post.call_args_list if "/action" in str(c)
        ]
        assert len(action_calls) == 1
        payload = action_calls[0].kwargs.get("json", {})
        assert payload == {"kind": "click", "ref": "e0"}

    def test_click_without_session_raises(self) -> None:
        adapter = BrowserAdapter(_http_client=_make_adapter_mock())

        with pytest.raises(BrowserError, match="No active tab"):
            adapter.click("e0")


class TestBrowserAdapterGetSnapshot:
    """BrowserAdapter.get_snapshot() returns current page state."""

    def test_get_snapshot_returns_current_state(self, adapter: BrowserAdapter) -> None:
        adapter.navigate(_PAGE_URL)
        result = adapter.get_snapshot()

        assert isinstance(result, PageSnapshot)
        assert result.text_content == _PAGE_TEXT
        assert len(result.links) == 2

    def test_get_snapshot_without_session_raises(self) -> None:
        adapter = BrowserAdapter(_http_client=_make_adapter_mock())

        with pytest.raises(BrowserError, match="No active tab"):
            adapter.get_snapshot()


class TestBrowserAdapterLifecycle:
    """BrowserAdapter lifecycle management."""

    def test_close_session_resets_state(
        self, adapter: BrowserAdapter, adapter_http: Mock
    ) -> None:
        adapter.navigate(_PAGE_URL)
        adapter.close_session()

        assert adapter._active is False
        assert adapter._current_url == ""

    def test_close_session_idempotent(self, adapter: BrowserAdapter) -> None:
        adapter.close_session()
        adapter.close_session()  # Should not raise

    def test_context_manager_calls_close_session(self, adapter_http: Mock) -> None:
        with BrowserAdapter(_http_client=adapter_http) as a:
            a.navigate(_PAGE_URL)

        assert a._active is False


class TestExtractLinks:
    """BrowserAdapter._extract_links() extracts links from snapshot + evaluate."""

    def test_extract_links_matches_by_text(self) -> None:
        adapter = BrowserAdapter()
        adapter._http_client = _make_adapter_mock()
        snap_data = {
            "nodes": _ADAPTER_SNAPSHOT_NODES,
            "title": _PAGE_TITLE,
            "url": _PAGE_URL,
        }
        links = adapter._extract_links(snap_data)

        assert len(links) == 2
        assert links[0] == PageLink(
            text="Page 1", url="https://example.com/page1", element_ref="e0"
        )
        assert links[1] == PageLink(
            text="Page 2", url="https://example.com/page2", element_ref="e1"
        )

    def test_extract_links_empty_when_no_link_nodes(self) -> None:
        adapter = BrowserAdapter()
        snap_data = {"nodes": [{"ref": "e0", "role": "heading", "name": "Title"}]}
        links = adapter._extract_links(snap_data)

        assert links == []
