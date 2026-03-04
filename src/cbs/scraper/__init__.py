"""PinchTab browser automation — thin adapter for agentic scraping."""

from cbs.scraper.browser import (
    BrowserClient,
    BrowserConnectionError,
    BrowserError,
    BrowserNavigationError,
    BrowserTimeoutError,
    PageContent,
)

__all__ = [
    "BrowserClient",
    "BrowserConnectionError",
    "BrowserError",
    "BrowserNavigationError",
    "BrowserTimeoutError",
    "PageContent",
]
