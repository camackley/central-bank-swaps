"""Playwright browser adapter — direct JS-rendering browser automation.

Replaces the former PinchTab HTTP adapter with a self-contained Playwright
backend.  Uses ``networkidle`` wait strategy by default so that React/JS
content is fully rendered before snapshotting.
"""

from __future__ import annotations

import contextlib
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Literal

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Page, sync_playwright
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class BrowserError(Exception):
    """Base exception for all browser errors."""


class BrowserConnectionError(BrowserError):
    """Raised when the browser cannot be launched."""


class BrowserTimeoutError(BrowserError):
    """Raised when page navigation exceeds the configured timeout."""


class BrowserNavigationError(BrowserError):
    """Raised when navigation fails."""


# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------


class PageContent(BaseModel):
    """Legacy data class kept for backward compatibility."""

    url: str
    title: str
    text: str
    snapshot: str


@dataclass(frozen=True)
class PageLink:
    """A link discovered on a page.

    ``element_ref`` is the absolute URL of the link — used as the argument
    to ``click()`` for multi-step navigation.
    """

    text: str
    url: str
    element_ref: str


@dataclass(frozen=True)
class PageSnapshot:
    """Snapshot of the current browser page state."""

    url: str
    title: str
    text_content: str
    links: list[PageLink] = field(default_factory=list)


# ---------------------------------------------------------------------------
# PlaywrightBrowserAdapter
# ---------------------------------------------------------------------------

_DEFAULT_TIMEOUT = 30
_MAX_LINKS = 500  # guard against link-farm pages

# Extracts all unique absolute <a href> links from the fully-rendered DOM.
_EXTRACT_LINKS_JS = """
() => {
    const seen = new Set();
    const results = [];
    for (const a of document.querySelectorAll('a[href]')) {
        const href = a.href;
        if (href && href.startsWith('http') && !seen.has(href)) {
            seen.add(href);
            results.push({
                text: a.textContent.trim().slice(0, 200),
                href: href,
            });
        }
    }
    return results;
}
"""

WaitStrategy = Literal["networkidle", "domcontentloaded", "load"]


_BOT_CHALLENGE_MARKERS = ("perfdrive.com", "shieldsquare.com", "radware")


class PlaywrightBrowserAdapter:
    """Browser adapter backed by Playwright.

    Uses ``networkidle`` wait strategy by default so that React/JS-rendered
    content is present in the DOM before link extraction.

    Use as a context manager::

        with PlaywrightBrowserAdapter() as browser:
            snapshot = browser.navigate("https://example.com")
            html = browser.get_page_html()

    Args:
        headless: Run Chromium in headless mode (default True).
        profile_dir: Path to a Chromium user-data directory for persistent
            cookies and session storage.  When set, the adapter uses
            ``launch_persistent_context()`` so that bot-manager session
            cookies (Radware, Cloudflare) survive across runs.  Defaults
            to the ``CBS_BROWSER_PROFILE`` environment variable, or None.
        _page: Inject a Playwright ``Page`` for unit testing.  When set,
            the adapter skips Playwright startup/shutdown.
    """

    def __init__(
        self,
        headless: bool = True,
        profile_dir: str | None = None,
        _page: Page | None = None,
    ) -> None:
        self._headless = headless
        self._profile_dir: str | None = profile_dir or os.environ.get(
            "CBS_BROWSER_PROFILE"
        )
        self._page: Page | None = _page
        self._owned: bool = _page is None
        self._current_url: str = ""
        self._current_title: str = ""
        self._pw_ctx: Any = None
        self._browser_instance: Any = None  # None when using persistent context
        self._context: Any = None  # BrowserContext — set in both modes

    def __enter__(self) -> PlaywrightBrowserAdapter:
        if self._owned:
            self._start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        self.close_session()

    # -- Internal helpers ---------------------------------------------------

    def _start(self) -> None:
        """Launch Playwright + Chromium and open a new page.

        Applies anti-detection measures so that sites with bot protection
        (Radware, Cloudflare, etc.) do not block headless Chromium:
        - ``--disable-blink-features=AutomationControlled`` removes the
          ``window.navigator.webdriver`` property that marks automated browsers.
        - A realistic Windows/Chrome user agent replaces the ``HeadlessChrome``
          string that triggers most bot-detection heuristics.
        - ``navigator.webdriver`` is deleted via an init script as a second layer.
        """
        _launch_args = [
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-blink-features=AutomationControlled",
        ]
        _context_kwargs: dict[str, Any] = {
            "user_agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
            "viewport": {"width": 1920, "height": 1080},
            "locale": "en-US",
        }
        try:
            self._pw_ctx = sync_playwright().start()
            if self._profile_dir:
                # Persistent context — cookies/session survive across runs.
                # launch_persistent_context returns a BrowserContext directly.
                os.makedirs(self._profile_dir, exist_ok=True)
                self._context = self._pw_ctx.chromium.launch_persistent_context(
                    self._profile_dir,
                    headless=self._headless,
                    args=_launch_args,
                    **_context_kwargs,
                )
                # No separate browser instance when using persistent context.
                self._browser_instance = None
            else:
                self._browser_instance = self._pw_ctx.chromium.launch(
                    headless=self._headless,
                    args=_launch_args,
                )
                # Create a browser context with a realistic user agent and viewport.
                self._context = self._browser_instance.new_context(**_context_kwargs)
            self._page = self._context.new_page()
            # Remove the webdriver property that signals browser automation.
            self._page.add_init_script(
                "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
            )
            # Block images, fonts, and media to speed up page loads.
            self._page.route(
                "**/*.{png,jpg,jpeg,gif,svg,ico,woff,woff2,ttf,mp4,webm}",
                lambda route: route.abort(),
            )
        except PlaywrightError as exc:
            raise BrowserConnectionError(f"Failed to launch Chromium: {exc}") from exc

    def _require_page(self) -> Page:
        if self._page is None:
            raise BrowserError("Browser not started — use as context manager")
        return self._page

    def _open_fresh_page(self) -> None:
        """Open a fresh page, clearing cross-bank session state.

        - Non-persistent mode: closes the current context and creates a new one
          (clears all cookies/session for bot-detection isolation per bank).
        - Persistent mode: opens a new page in the same context (cookies are
          preserved intentionally across banks to maintain Radware trust score).
        """
        if self._context is None:
            raise BrowserError("Browser not started — use as context manager")
        if self._page is not None:
            with contextlib.suppress(PlaywrightError):
                self._page.close()
            self._page = None

        if self._browser_instance is not None:
            # Non-persistent: create a fresh context (clears cookies)
            with contextlib.suppress(PlaywrightError):
                self._context.close()
            self._context = self._browser_instance.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/131.0.0.0 Safari/537.36"
                ),
                viewport={"width": 1920, "height": 1080},
                locale="en-US",
            )
            logger.debug("Opened fresh browser context (cleared cookies/session)")
        # else: persistent context — reuse same context, cookies intentionally kept

        self._page = self._context.new_page()
        self._page.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )
        self._page.route(
            "**/*.{png,jpg,jpeg,gif,svg,ico,woff,woff2,ttf,mp4,webm}",
            lambda route: route.abort(),
        )
        self._current_url = ""
        self._current_title = ""

    def _extract_links(self, page: Page) -> list[PageLink]:
        """Extract all unique absolute links from the rendered DOM."""
        try:
            raw: list[dict[str, str]] = page.evaluate(_EXTRACT_LINKS_JS)
            links = []
            for item in raw[:_MAX_LINKS]:
                url = item.get("href", "")
                text = item.get("text", "")
                if url:
                    links.append(PageLink(text=text, url=url, element_ref=url))
            return links
        except PlaywrightError as exc:
            logger.warning("Failed to extract links via JS: %s", exc)
            return []

    def _build_snapshot(self, page: Page) -> PageSnapshot:
        """Build a PageSnapshot from the current page state."""
        self._current_url = page.url
        self._current_title = page.title()
        try:
            text: str = page.evaluate("() => document.body.innerText") or ""
        except PlaywrightError:
            text = ""
        links = self._extract_links(page)
        return PageSnapshot(
            url=self._current_url,
            title=self._current_title,
            text_content=text,
            links=links,
        )

    # -- Public API ---------------------------------------------------------

    def navigate(
        self,
        url: str,
        timeout: int = _DEFAULT_TIMEOUT,
        wait_strategy: WaitStrategy = "networkidle",
        wait_for_selector: str | None = None,
    ) -> PageSnapshot:
        """Navigate to *url* and wait for the page to finish rendering.

        Args:
            url: The URL to navigate to.
            timeout: Maximum seconds to wait for page load.
            wait_strategy: Playwright load-state condition.
                ``"networkidle"`` (default) waits until no network requests
                are in flight for 500 ms — ensures React/fetch content loads.
            wait_for_selector: Optional CSS selector to wait for after the
                load state is reached.  Useful for SPAs that render content
                asynchronously (e.g. ``"article"`` for Bank of England).

        Raises:
            BrowserTimeoutError: If the page load exceeds *timeout* seconds.
            BrowserNavigationError: If Playwright reports a navigation error.
            BrowserError: If the browser has not been started.
        """
        page = self._require_page()
        try:
            page.goto(url, wait_until=wait_strategy, timeout=timeout * 1000)
            if wait_for_selector:
                page.wait_for_selector(wait_for_selector, timeout=timeout * 1000)
        except PlaywrightTimeoutError as exc:
            raise BrowserTimeoutError(
                f"Page load timed out after {timeout}s for {url}"
            ) from exc
        except PlaywrightError as exc:
            raise BrowserNavigationError(f"Navigation failed for {url}: {exc}") from exc
        snapshot = self._build_snapshot(page)
        if any(m in snapshot.url.lower() for m in _BOT_CHALLENGE_MARKERS):
            logger.warning(
                "Bot challenge detected navigating to %s — landed on %s. "
                "Set CBS_BROWSER_PROFILE to a directory path for persistent "
                "cookies across runs.",
                url,
                snapshot.url[:120],
            )
        return snapshot

    def click(self, element_ref: str, timeout: int = _DEFAULT_TIMEOUT) -> PageSnapshot:
        """Navigate to *element_ref* (absolute URL) and return the snapshot.

        In the Playwright adapter, ``element_ref`` is the absolute URL of the
        link to follow.  The browser navigates directly to it rather than
        clicking a DOM element by ref.

        Raises:
            BrowserError: If no page has been navigated yet.
            BrowserTimeoutError: If the navigation exceeds *timeout* seconds.
            BrowserNavigationError: If the navigation fails.
        """
        page = self._require_page()
        if not self._current_url:
            raise BrowserError("No active page — call navigate() first")
        try:
            page.goto(element_ref, wait_until="networkidle", timeout=timeout * 1000)
        except PlaywrightTimeoutError as exc:
            raise BrowserTimeoutError(f"Navigation timed out after {timeout}s") from exc
        except PlaywrightError as exc:
            raise BrowserNavigationError(
                f"Navigation failed for {element_ref}: {exc}"
            ) from exc
        return self._build_snapshot(page)

    def get_snapshot(self) -> PageSnapshot:
        """Return a snapshot of the current page without navigating."""
        page = self._require_page()
        if not self._current_url:
            raise BrowserError("No active page — call navigate() first")
        return self._build_snapshot(page)

    def get_page_html(self) -> str:
        """Return the full rendered DOM HTML (after JS execution).

        Use this when you want Claude to read the complete page content
        and extract press release URLs directly — more reliable than
        accessibility-tree link extraction for complex React/SPA pages.

        Raises:
            BrowserError: If no page has been navigated yet.
            BrowserNavigationError: If the HTML cannot be retrieved.
        """
        page = self._require_page()
        if not self._current_url:
            raise BrowserError("No active page — call navigate() first")
        try:
            return page.content()
        except PlaywrightError as exc:
            raise BrowserNavigationError(f"Failed to get page HTML: {exc}") from exc

    def new_session(self) -> None:
        """Reset browser state for a new bank visit (clears cookies/session).

        Closes the current browser context and opens a fresh one on the same
        running Chromium instance.  Call this between banks to avoid cross-bank
        session fingerprinting by bot-detection systems (Radware, Cloudflare).

        No-op when the adapter was created with an injected ``_page`` (test mode).
        """
        if not self._owned:
            return
        self._open_fresh_page()

    def close_session(self) -> None:
        """Shut down Playwright and release all browser resources."""
        if not self._owned:
            self._current_url = ""
            self._current_title = ""
            return
        if self._page is not None:
            with contextlib.suppress(PlaywrightError):
                self._page.close()
            self._page = None
        if self._context is not None:
            with contextlib.suppress(PlaywrightError):
                self._context.close()
            self._context = None
        if self._browser_instance is not None:
            with contextlib.suppress(PlaywrightError):
                self._browser_instance.close()
            self._browser_instance = None
        if self._pw_ctx is not None:
            with contextlib.suppress(PlaywrightError):
                self._pw_ctx.stop()
            self._pw_ctx = None
        self._current_url = ""
        self._current_title = ""


# Backward-compatible alias — the rest of the codebase imports BrowserAdapter.
BrowserAdapter = PlaywrightBrowserAdapter
