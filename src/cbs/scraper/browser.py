"""PinchTab browser adapter — thin wrapper around PinchTab's HTTP API.

Provides the interface that the agentic navigator programs against.
Unit tests mock this adapter — never PinchTab directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
# Data classes — the navigator's view of a browser page
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PageLink:
    """A link discovered on a page via the accessibility snapshot."""

    text: str
    url: str
    element_ref: str  # PinchTab element reference (e.g., "e0", "e1")


@dataclass(frozen=True)
class PageSnapshot:
    """Snapshot of the current browser page state."""

    url: str
    title: str
    text_content: str
    links: list[PageLink] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Browser adapter
# ---------------------------------------------------------------------------

_DEFAULT_BASE_URL = "http://localhost:9867"


class BrowserAdapter:
    """Thin adapter around PinchTab's HTTP API for agentic navigation.

    Manages a Chrome instance with a persistent tab for multi-step
    navigation (clicking links, paginating).  Use as a context manager::

        with BrowserAdapter() as browser:
            page = browser.navigate("https://example.com")
            page2 = browser.click("e3")
    """

    def __init__(self, base_url: str = _DEFAULT_BASE_URL) -> None:
        self._base_url = base_url
        self._instance_id: str | None = None
        self._tab_id: str | None = None

    def __enter__(self) -> BrowserAdapter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        self.close_session()

    def navigate(self, url: str, timeout: int = 30) -> PageSnapshot:
        """Navigate to *url* and return the resulting page snapshot.

        Raises:
            BrowserTimeoutError: If page load exceeds *timeout*.
            BrowserNavigationError: If PinchTab returns a non-2xx status.
            BrowserConnectionError: If PinchTab server is unreachable.
        """
        raise NotImplementedError("Real PinchTab HTTP calls — see Slice 1.9")

    def click(self, element_ref: str, timeout: int = 30) -> PageSnapshot:
        """Click an element by its PinchTab ref and return the resulting page.

        Raises:
            BrowserError: If the click or subsequent page load fails.
        """
        raise NotImplementedError("Real PinchTab HTTP calls — see Slice 1.9")

    def get_snapshot(self) -> PageSnapshot:
        """Get a snapshot of the current page without navigating."""
        raise NotImplementedError("Real PinchTab HTTP calls — see Slice 1.9")

    def close_session(self) -> None:
        """Stop the Chrome instance.  Safe to call multiple times."""
