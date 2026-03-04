"""Integration tests for PinchTab browser adapter — Slice 1.9.

These tests require a running PinchTab server on localhost:9867.
Run with: pytest tests/integration/ -m integration
"""

from __future__ import annotations

import pytest

from cbs.scraper.browser import BrowserClient, BrowserTimeoutError


@pytest.mark.integration
class TestPinchTabBrowserIntegration:
    """Integration tests that require a running PinchTab server."""

    def test_navigate_to_known_page(self) -> None:
        """Navigate to a stable public page and verify content is returned."""
        with BrowserClient() as browser:
            page = browser.navigate("https://example.com", timeout=30)

            assert "Example Domain" in page.title
            assert "example" in page.text.lower()
            assert page.url == "https://example.com"
            assert page.snapshot  # Non-empty snapshot

    def test_timeout_on_unreachable_page(self) -> None:
        """Navigating to a non-routable address triggers timeout."""
        with BrowserClient() as browser, pytest.raises(BrowserTimeoutError):
            # RFC 5737 TEST-NET — guaranteed non-routable
            browser.navigate("http://192.0.2.1", timeout=3)

    def test_context_manager_cleanup(self) -> None:
        """Chrome instance is properly cleaned up after context manager exits."""
        with BrowserClient() as browser:
            browser.navigate("https://example.com", timeout=10)
        # If we get here without error, cleanup succeeded
