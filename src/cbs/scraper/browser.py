"""PinchTab browser adapter — thin wrapper around PinchTab's HTTP API.

Provides two layers:
- ``BrowserClient``: low-level HTTP adapter (Slice 1.9) with ``navigate()``
- ``BrowserAdapter``: higher-level adapter (Slice 1.10) adding ``click()``
  and ``get_snapshot()`` for the agentic navigator

Targets PinchTab standalone mode (v0.7+) where Chrome is managed by the
server.  Uses flat top-level endpoints: ``/navigate``, ``/text``,
``/snapshot``, ``/action``, ``/evaluate``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import httpx
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class BrowserError(Exception):
    """Base exception for all PinchTab browser errors."""


class BrowserConnectionError(BrowserError):
    """Raised when PinchTab server is unreachable."""


class BrowserTimeoutError(BrowserError):
    """Raised when page navigation exceeds the configured timeout."""


class BrowserNavigationError(BrowserError):
    """Raised when navigation fails (non-2xx from PinchTab)."""


# ---------------------------------------------------------------------------
# Public data classes — navigator's view of a browser page
# ---------------------------------------------------------------------------


class PageContent(BaseModel):
    """Content retrieved from a navigated page (low-level)."""

    url: str
    title: str
    text: str
    snapshot: str


@dataclass(frozen=True)
class PageLink:
    """A link discovered on a page via the accessibility snapshot."""

    text: str
    url: str
    element_ref: str  # PinchTab element reference (e.g., "e0", "e1")


@dataclass(frozen=True)
class PageSnapshot:
    """Snapshot of the current browser page state (high-level)."""

    url: str
    title: str
    text_content: str
    links: list[PageLink] = field(default_factory=list)


# ---------------------------------------------------------------------------
# BrowserClient — low-level PinchTab HTTP adapter (Slice 1.9)
# ---------------------------------------------------------------------------

_DEFAULT_BASE_URL = "http://localhost:9867"
_DEFAULT_TIMEOUT = 30
_HTTPX_BUFFER_SECONDS = 10

# JS expression that extracts all link texts and hrefs in DOM order.
_EXTRACT_LINKS_JS = (
    "JSON.stringify(Array.from(document.querySelectorAll('a[href]'))"
    ".map(a=>({t:a.textContent.trim().slice(0,200),h:a.href})))"
)


class BrowserClient:
    """Thin adapter around PinchTab's standalone HTTP API.

    In standalone mode PinchTab manages Chrome directly — no instance
    lifecycle is needed.  Use as a context manager for consistency::

        with BrowserClient() as browser:
            page = browser.navigate("https://example.com", timeout=30)
            print(page.text)
    """

    def __init__(
        self,
        base_url: str = _DEFAULT_BASE_URL,
        headless: bool = True,
        _http_client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url
        self._headless = headless
        self._http_client = _http_client or httpx.Client()
        self._started = False

    def __enter__(self) -> BrowserClient:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        self.stop()

    # -- Instance lifecycle -------------------------------------------------

    def start(self) -> None:
        """Verify PinchTab is reachable.

        Raises:
            BrowserConnectionError: If PinchTab server is unreachable.
        """
        try:
            resp = self._http_client.get(f"{self._base_url}/health")
            resp.raise_for_status()
            self._started = True
        except httpx.ConnectError as exc:
            msg = f"Cannot connect to PinchTab at {self._base_url}"
            raise BrowserConnectionError(msg) from exc
        except httpx.HTTPStatusError as exc:
            msg = f"PinchTab health check failed: {exc.response.status_code}"
            raise BrowserError(msg) from exc

    def stop(self) -> None:
        """No-op — PinchTab manages Chrome in standalone mode."""
        self._started = False

    # -- Navigation ---------------------------------------------------------

    def navigate(self, url: str, timeout: int = _DEFAULT_TIMEOUT) -> PageContent:
        """Navigate to *url*, wait for render, and return page content.

        Args:
            url: The URL to navigate to.
            timeout: Maximum seconds to wait for page load.

        Returns:
            PageContent with url, title, text, and snapshot.

        Raises:
            BrowserTimeoutError: If page load exceeds *timeout*.
            BrowserNavigationError: If PinchTab returns a non-2xx status.
            BrowserConnectionError: If PinchTab server is unreachable.
            BrowserError: If the client has not been started.
        """
        if not self._started:
            msg = (
                "Browser instance not started — call start() or use as context manager"
            )
            raise BrowserError(msg)

        try:
            # 1. Navigate (PinchTab waits for render completion)
            resp = self._http_client.post(
                f"{self._base_url}/navigate",
                json={"url": url, "timeout": timeout, "blockImages": True},
                timeout=timeout + _HTTPX_BUFFER_SECONDS,
            )
            resp.raise_for_status()
            nav_data = resp.json()

            # 2. Get text content
            text_resp = self._http_client.get(
                f"{self._base_url}/text",
            )
            text_resp.raise_for_status()
            text_data = text_resp.json()
            text = text_data.get("text", "")

            # 3. Get accessibility snapshot
            snap_resp = self._http_client.get(
                f"{self._base_url}/snapshot",
            )
            snap_resp.raise_for_status()
            snapshot = snap_resp.text

            return PageContent(
                url=nav_data.get("url", url),
                title=nav_data.get("title", ""),
                text=text,
                snapshot=snapshot,
            )

        except httpx.ConnectError as exc:
            msg = f"Lost connection to PinchTab at {self._base_url}"
            raise BrowserConnectionError(msg) from exc
        except httpx.TimeoutException as exc:
            msg = f"Page load timed out after {timeout}s for {url}"
            raise BrowserTimeoutError(msg) from exc
        except httpx.HTTPStatusError as exc:
            msg = (
                f"Navigation failed for {url}: "
                f"PinchTab returned {exc.response.status_code}"
            )
            raise BrowserNavigationError(msg) from exc


# ---------------------------------------------------------------------------
# BrowserAdapter — high-level adapter for agentic navigation (Slice 1.10)
# ---------------------------------------------------------------------------


class BrowserAdapter:
    """Higher-level adapter for the agentic navigator.

    Adds ``click()`` and ``get_snapshot()`` on top of basic navigation.
    Uses PinchTab standalone flat API::

        with BrowserAdapter() as browser:
            page = browser.navigate("https://example.com")
            page2 = browser.click("e3")
    """

    def __init__(
        self,
        base_url: str = _DEFAULT_BASE_URL,
        headless: bool = True,
        _http_client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url
        self._headless = headless
        self._http_client = _http_client or httpx.Client()
        self._active = False
        self._current_url = ""
        self._current_title = ""

    def __enter__(self) -> BrowserAdapter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        self.close_session()

    # -- Internal helpers ---------------------------------------------------

    def _ensure_ready(self) -> None:
        """Verify PinchTab is reachable on first use."""
        if self._active:
            return
        try:
            resp = self._http_client.get(f"{self._base_url}/health")
            resp.raise_for_status()
            self._active = True
        except httpx.ConnectError as exc:
            msg = f"Cannot connect to PinchTab at {self._base_url}"
            raise BrowserConnectionError(msg) from exc
        except httpx.HTTPStatusError as exc:
            msg = f"PinchTab health check failed: {exc.response.status_code}"
            raise BrowserError(msg) from exc

    def _require_navigated(self) -> None:
        """Ensure we have navigated to a page."""
        if not self._current_url:
            msg = "No active tab — call navigate() first"
            raise BrowserError(msg)

    def _fetch_snapshot(self) -> PageSnapshot:
        """Get text + accessibility snapshot for the current page."""
        self._require_navigated()
        try:
            text_resp = self._http_client.get(f"{self._base_url}/text")
            text_resp.raise_for_status()
            text_data = text_resp.json()
            text = text_data.get("text", "")
            # Update current URL/title from text response (may have changed
            # after click-induced navigation).
            self._current_url = text_data.get("url", self._current_url)
            self._current_title = text_data.get("title", self._current_title)

            snap_resp = self._http_client.get(f"{self._base_url}/snapshot")
            snap_resp.raise_for_status()
            snap_data = snap_resp.json()

            links = self._extract_links(snap_data)

            return PageSnapshot(
                url=self._current_url,
                title=self._current_title,
                text_content=text,
                links=links,
            )
        except httpx.ConnectError as exc:
            msg = f"Lost connection to PinchTab at {self._base_url}"
            raise BrowserConnectionError(msg) from exc
        except httpx.HTTPStatusError as exc:
            msg = f"Snapshot failed: PinchTab returned {exc.response.status_code}"
            raise BrowserNavigationError(msg) from exc

    def _extract_links(self, snap_data: dict[str, Any]) -> list[PageLink]:
        """Extract links by combining snapshot nodes with JS-evaluated URLs.

        The snapshot gives us link nodes (role="link") with their element
        refs and accessible names, but no hrefs.  We run a JS expression
        via ``/evaluate`` to get all ``<a>`` hrefs in DOM order, then match
        them to snapshot link nodes by text.
        """
        nodes = snap_data.get("nodes", [])
        link_nodes = [n for n in nodes if n.get("role") == "link"]
        if not link_nodes:
            return []

        # Get link URLs via JS evaluation
        try:
            eval_resp = self._http_client.post(
                f"{self._base_url}/evaluate",
                json={"expression": _EXTRACT_LINKS_JS},
            )
            eval_resp.raise_for_status()
            eval_data = eval_resp.json()
            dom_links: list[dict[str, str]] = json.loads(eval_data.get("result", "[]"))
        except (httpx.HTTPError, json.JSONDecodeError, KeyError):
            logger.warning("Failed to extract link URLs via evaluate, using refs only")
            return [
                PageLink(
                    text=n.get("name", ""),
                    url="",
                    element_ref=n.get("ref", ""),
                )
                for n in link_nodes
            ]

        # Build a lookup: text → list of hrefs (for duplicate text handling)
        text_to_hrefs: dict[str, list[str]] = {}
        for dl in dom_links:
            t = dl.get("t", "")
            h = dl.get("h", "")
            if t not in text_to_hrefs:
                text_to_hrefs[t] = []
            text_to_hrefs[t].append(h)

        links: list[PageLink] = []
        for node in link_nodes:
            name = node.get("name", "")
            ref = node.get("ref", "")
            href = ""
            # Try exact match first
            if name in text_to_hrefs and text_to_hrefs[name]:
                href = text_to_hrefs[name].pop(0)
            else:
                # Try substring match (accessible name may be truncated)
                for t, hrefs in text_to_hrefs.items():
                    if hrefs and (name in t or t in name):
                        href = hrefs.pop(0)
                        break
            if href:
                links.append(PageLink(text=name, url=href, element_ref=ref))

        return links

    # -- Public API ---------------------------------------------------------

    def navigate(self, url: str, timeout: int = 30) -> PageSnapshot:
        """Navigate to *url* and return the resulting page snapshot.

        Raises:
            BrowserTimeoutError: If page load exceeds *timeout*.
            BrowserNavigationError: If PinchTab returns a non-2xx status.
            BrowserConnectionError: If PinchTab server is unreachable.
        """
        try:
            self._ensure_ready()

            resp = self._http_client.post(
                f"{self._base_url}/navigate",
                json={"url": url, "timeout": timeout, "blockImages": True},
                timeout=timeout + _HTTPX_BUFFER_SECONDS,
            )
            resp.raise_for_status()
            nav_data = resp.json()
            self._current_url = nav_data.get("url", url)
            self._current_title = nav_data.get("title", "")

            return self._fetch_snapshot()

        except httpx.ConnectError as exc:
            msg = f"Cannot connect to PinchTab at {self._base_url}"
            raise BrowserConnectionError(msg) from exc
        except httpx.TimeoutException as exc:
            msg = f"Page load timed out after {timeout}s for {url}"
            raise BrowserTimeoutError(msg) from exc
        except httpx.HTTPStatusError as exc:
            msg = (
                f"Navigation failed for {url}: "
                f"PinchTab returned {exc.response.status_code}"
            )
            raise BrowserNavigationError(msg) from exc

    def click(self, element_ref: str, timeout: int = 30) -> PageSnapshot:
        """Click an element by its PinchTab ref and return the resulting page.

        Raises:
            BrowserError: If the click or subsequent page load fails.
        """
        self._require_navigated()
        try:
            resp = self._http_client.post(
                f"{self._base_url}/action",
                json={"kind": "click", "ref": element_ref},
                timeout=timeout + _HTTPX_BUFFER_SECONDS,
            )
            resp.raise_for_status()
            return self._fetch_snapshot()
        except httpx.ConnectError as exc:
            msg = f"Lost connection to PinchTab at {self._base_url}"
            raise BrowserConnectionError(msg) from exc
        except httpx.TimeoutException as exc:
            msg = f"Click timed out after {timeout}s"
            raise BrowserTimeoutError(msg) from exc
        except httpx.HTTPStatusError as exc:
            msg = f"Click failed: PinchTab returned {exc.response.status_code}"
            raise BrowserNavigationError(msg) from exc

    def get_snapshot(self) -> PageSnapshot:
        """Get a snapshot of the current page without navigating."""
        return self._fetch_snapshot()

    def close_session(self) -> None:
        """Reset adapter state.  Safe to call multiple times."""
        self._active = False
        self._current_url = ""
        self._current_title = ""
