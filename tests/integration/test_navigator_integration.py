"""Integration tests for the agentic navigator — requires LLM API keys."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cbs.config.banks import BankConfig
from cbs.llm import get_llm
from cbs.scraper.browser import BrowserAdapter, PageLink, PageSnapshot
from cbs.scraper.navigator import find_press_releases


@pytest.mark.integration
class TestAgentWithRealLLMMockedBrowser:
    """Real LLM reasons over mocked browser snapshots to find press releases."""

    def test_agent_finds_press_releases_from_homepage(self) -> None:
        bank = BankConfig(
            name="Test Central Bank",
            country="Testland",
            homepage_url="https://www.testcentralbank.org",
            page_load_timeout=10,
        )
        browser = MagicMock(spec=BrowserAdapter)

        homepage = PageSnapshot(
            url="https://www.testcentralbank.org",
            title="Test Central Bank - Home",
            text_content="Welcome to the Test Central Bank.",
            links=[
                PageLink(text="About Us", url="/about", element_ref="e0"),
                PageLink(text="Publications", url="/publications", element_ref="e1"),
                PageLink(
                    text="Press Releases",
                    url="/press-releases",
                    element_ref="e2",
                ),
                PageLink(text="Careers", url="/careers", element_ref="e3"),
            ],
        )
        listing = PageSnapshot(
            url="https://www.testcentralbank.org/press-releases",
            title="Press Releases - Test Central Bank",
            text_content="Recent press releases from the bank.",
            links=[
                PageLink(
                    text="Central bank swap line with Fed - 2024-01-15",
                    url="https://www.testcentralbank.org/pr/swap-fed-2024",
                    element_ref="e0",
                ),
                PageLink(
                    text="Monetary policy decision - 2024-01-10",
                    url="https://www.testcentralbank.org/pr/policy-2024",
                    element_ref="e1",
                ),
            ],
        )

        browser.navigate.return_value = homepage
        browser.click.return_value = listing
        browser.get_snapshot.return_value = listing

        llm = get_llm("anthropic", "claude-sonnet-4-20250514")

        result = find_press_releases(bank, browser, llm, max_pages=1)

        assert result.used_direct_url is False
        assert len(result.press_releases) >= 1
        assert result.pages_visited >= 1
