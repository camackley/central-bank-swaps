"""PinchTab browser adapter — thin wrapper around PinchTab's HTTP API."""

from __future__ import annotations

import contextlib
from typing import Any

import httpx
from pydantic import BaseModel

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
# Internal Pydantic models (PinchTab response parsing)
# ---------------------------------------------------------------------------


class _InstanceResponse(BaseModel):
    id: str


class _TabResponse(BaseModel):
    tabId: str  # noqa: N815
    url: str
    title: str


# ---------------------------------------------------------------------------
# Public Pydantic model
# ---------------------------------------------------------------------------


class PageContent(BaseModel):
    """Content retrieved from a navigated page."""

    url: str
    title: str
    text: str
    snapshot: str


# ---------------------------------------------------------------------------
# Browser client
# ---------------------------------------------------------------------------

_DEFAULT_BASE_URL = "http://localhost:9867"
_DEFAULT_TIMEOUT = 30
_HTTPX_BUFFER_SECONDS = 10


class BrowserClient:
    """Thin adapter around PinchTab's HTTP API.

    Manages a single Chrome instance lifecycle.  Use as a context manager
    to ensure the Chrome instance is stopped on exit.

    Usage::

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
        self._instance_id: str | None = None

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
        """Start a Chrome instance via PinchTab.

        Raises:
            BrowserConnectionError: If PinchTab server is unreachable.
            BrowserError: If instance creation fails.
        """
        try:
            mode = "headless" if self._headless else "headed"
            resp = self._http_client.post(
                f"{self._base_url}/instances/start",
                json={"profileId": "", "mode": mode},
            )
            resp.raise_for_status()
            data = _InstanceResponse.model_validate(resp.json())
            self._instance_id = data.id
        except httpx.ConnectError as exc:
            msg = f"Cannot connect to PinchTab at {self._base_url}"
            raise BrowserConnectionError(msg) from exc
        except httpx.HTTPStatusError as exc:
            msg = f"PinchTab instance start failed: {exc.response.status_code}"
            raise BrowserError(msg) from exc

    def stop(self) -> None:
        """Stop the Chrome instance.  Safe to call multiple times."""
        if self._instance_id is None:
            return
        try:
            self._http_client.post(
                f"{self._base_url}/instances/{self._instance_id}/stop",
            )
        except httpx.HTTPError:
            pass  # Best-effort cleanup
        finally:
            self._instance_id = None

    # -- Navigation ---------------------------------------------------------

    def navigate(self, url: str, timeout: int = _DEFAULT_TIMEOUT) -> PageContent:
        """Navigate to *url*, wait for render, and return page content.

        Opens a new tab, navigates with the given timeout, retrieves text
        and accessibility snapshot, then closes the tab.

        Args:
            url: The URL to navigate to.
            timeout: Maximum seconds to wait for page load.

        Returns:
            PageContent with url, title, text, and snapshot.

        Raises:
            BrowserTimeoutError: If page load exceeds *timeout*.
            BrowserNavigationError: If PinchTab returns a non-2xx status.
            BrowserConnectionError: If PinchTab server is unreachable.
            BrowserError: If the instance has not been started.
        """
        if self._instance_id is None:
            msg = (
                "Browser instance not started — call start() or use as context manager"
            )
            raise BrowserError(msg)

        tab_id: str | None = None
        try:
            # 1. Open tab
            resp = self._http_client.post(
                f"{self._base_url}/instances/{self._instance_id}/tabs/open",
                json={"url": url},
            )
            resp.raise_for_status()
            tab_data = _TabResponse.model_validate(resp.json())
            tab_id = tab_data.tabId

            # 2. Navigate (PinchTab waits for render completion)
            resp = self._http_client.post(
                f"{self._base_url}/tabs/{tab_id}/navigate",
                json={"url": url, "timeout": timeout, "blockImages": True},
                timeout=timeout + _HTTPX_BUFFER_SECONDS,
            )
            resp.raise_for_status()

            # 3. Get text content
            text_resp = self._http_client.get(
                f"{self._base_url}/tabs/{tab_id}/text",
            )
            text_resp.raise_for_status()
            text = text_resp.text

            # 4. Get accessibility snapshot
            snap_resp = self._http_client.get(
                f"{self._base_url}/tabs/{tab_id}/snapshot",
            )
            snap_resp.raise_for_status()
            snapshot = snap_resp.text

            return PageContent(
                url=url,
                title=tab_data.title,
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
        finally:
            if tab_id is not None:
                with contextlib.suppress(httpx.HTTPError):
                    self._http_client.post(
                        f"{self._base_url}/tabs/{tab_id}/close",
                    )
