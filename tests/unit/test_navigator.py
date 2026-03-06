"""Tests for the agentic navigator — Slice 1.10 (FR-001)."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from cbs.config.banks import BankConfig
from cbs.scraper.browser import BrowserAdapter, PageLink, PageSnapshot
from cbs.scraper.models import NavigationResult
from cbs.scraper.navigator import (
    _extract_press_releases_from_snapshot,
    _filter_off_domain,
    find_press_releases,
)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_bank(*, press_releases_url: str | None = None) -> BankConfig:
    return BankConfig(
        name="Test Bank",
        country="Testland",
        homepage_url="https://www.testbank.org",
        press_releases_url=press_releases_url,
        page_load_timeout=10,
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


# ---------------------------------------------------------------------------
# 1. test_direct_url_skips_navigation
# ---------------------------------------------------------------------------


class TestDirectUrlSkipsNavigation:
    """When press_releases_url is configured, navigate directly — no LLM agent."""

    def test_navigates_to_configured_url(self) -> None:
        bank = _make_bank(press_releases_url="https://www.testbank.org/press")
        browser = MagicMock(spec=BrowserAdapter)
        browser.navigate.return_value = _make_snapshot(
            url="https://www.testbank.org/press",
            title="Press Releases",
            links=[
                PageLink(
                    text="Swap agreement with ECB",
                    url="https://www.testbank.org/pr/1",
                    element_ref="e0",
                ),
            ],
        )
        llm = MagicMock()

        result = find_press_releases(bank, browser, llm, max_pages=1)

        browser.navigate.assert_called_once_with(
            "https://www.testbank.org/press",
            timeout=10,
        )
        assert result.used_direct_url is True
        assert result.bank_name == "Test Bank"

    def test_llm_not_invoked_for_direct_url(self) -> None:
        bank = _make_bank(press_releases_url="https://www.testbank.org/press")
        browser = MagicMock(spec=BrowserAdapter)
        browser.navigate.return_value = _make_snapshot(links=[])
        llm = MagicMock()

        find_press_releases(bank, browser, llm, max_pages=1)

        llm.invoke.assert_not_called()
        llm.bind_tools.assert_not_called()

    def test_returns_navigation_result_with_press_releases(self) -> None:
        bank = _make_bank(press_releases_url="https://www.testbank.org/press")
        browser = MagicMock(spec=BrowserAdapter)
        browser.navigate.return_value = _make_snapshot(
            url="https://www.testbank.org/press",
            links=[
                PageLink(
                    text="Swap agreement with ECB",
                    url="https://www.testbank.org/pr/1",
                    element_ref="e0",
                ),
                PageLink(
                    text="Rate decision Dec 2023",
                    url="https://www.testbank.org/pr/2",
                    element_ref="e1",
                ),
            ],
        )
        llm = MagicMock()

        result = find_press_releases(bank, browser, llm, max_pages=1)

        assert isinstance(result, NavigationResult)
        assert result.used_direct_url is True
        assert result.listing_page_url == "https://www.testbank.org/press"
        assert len(result.press_releases) == 2
        assert result.press_releases[0].url == "https://www.testbank.org/pr/1"


# ---------------------------------------------------------------------------
# 2. test_agent_finds_press_releases_from_homepage
# ---------------------------------------------------------------------------


class TestAgentFindsPressReleasesFromHomepage:
    """LLM agent navigates from homepage to discover press releases section."""

    def test_agent_discovers_press_releases_section(self) -> None:
        """Agent navigates: homepage → clicks 'Media' → clicks 'Press Releases'."""
        bank = _make_bank()  # No press_releases_url

        browser = MagicMock(spec=BrowserAdapter)

        homepage = _make_snapshot(
            url="https://www.testbank.org",
            title="Test Bank - Home",
            links=[
                PageLink(text="About", url="/about", element_ref="e0"),
                PageLink(text="Media & Press", url="/media", element_ref="e1"),
                PageLink(text="Research", url="/research", element_ref="e2"),
            ],
        )
        media_page = _make_snapshot(
            url="https://www.testbank.org/media",
            title="Media",
            links=[
                PageLink(
                    text="Press Releases",
                    url="/media/press",
                    element_ref="e0",
                ),
                PageLink(text="Speeches", url="/media/speeches", element_ref="e1"),
            ],
        )
        listing_page = _make_snapshot(
            url="https://www.testbank.org/media/press",
            title="Press Releases",
            links=[
                PageLink(
                    text="Swap agreement with ECB - Jan 2024",
                    url="https://www.testbank.org/pr/swap-ecb-2024",
                    element_ref="e0",
                ),
                PageLink(
                    text="Interest rate decision - Dec 2023",
                    url="https://www.testbank.org/pr/rate-dec-2023",
                    element_ref="e1",
                ),
            ],
        )

        # navigate(homepage_url) → homepage; then clicks happen
        browser.navigate.return_value = homepage
        browser.click.side_effect = [media_page, listing_page]
        browser.get_snapshot.return_value = listing_page

        # Fake LLM that returns predetermined tool-calling messages:
        # 1) click "Media & Press" (e1)
        # 2) click "Press Releases" (e0)
        # 3) extract URLs
        # 4) final answer
        fake_llm = FakeMessagesListChatModel(
            responses=[
                AIMessage(
                    content="I see a 'Media & Press' link. Let me click it.",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "name": "click_link",
                            "args": {"element_ref": "e1"},
                        }
                    ],
                ),
                AIMessage(
                    content="Found 'Press Releases' link. Clicking it.",
                    tool_calls=[
                        {
                            "id": "call_2",
                            "name": "click_link",
                            "args": {"element_ref": "e0"},
                        }
                    ],
                ),
                AIMessage(
                    content="I'm on the press releases listing. Extracting URLs.",
                    tool_calls=[
                        {
                            "id": "call_3",
                            "name": "extract_press_release_urls",
                            "args": {},
                        }
                    ],
                ),
                AIMessage(content="Found 2 press releases on the listing page."),
            ],
        )

        result = find_press_releases(bank, browser, fake_llm, max_pages=1)

        assert result.used_direct_url is False
        assert len(result.press_releases) >= 1
        assert result.pages_visited >= 1


# ---------------------------------------------------------------------------
# 3. test_pagination_discovers_older_releases
# ---------------------------------------------------------------------------


class TestPaginationDiscoversOlderReleases:
    """Pagination follows 'next page' links to discover more press releases."""

    def test_pagination_follows_next_page_link(self) -> None:
        bank = _make_bank(press_releases_url="https://www.testbank.org/press")
        browser = MagicMock(spec=BrowserAdapter)

        page1 = _make_snapshot(
            url="https://www.testbank.org/press",
            links=[
                PageLink(
                    text="PR 1",
                    url="https://www.testbank.org/pr/1",
                    element_ref="e0",
                ),
                PageLink(
                    text="PR 2",
                    url="https://www.testbank.org/pr/2",
                    element_ref="e1",
                ),
                PageLink(
                    text="Next Page",
                    url="/press?page=2",
                    element_ref="e10",
                ),
            ],
        )
        page2 = _make_snapshot(
            url="https://www.testbank.org/press?page=2",
            links=[
                PageLink(
                    text="PR 3",
                    url="https://www.testbank.org/pr/3",
                    element_ref="e0",
                ),
                PageLink(
                    text="PR 4",
                    url="https://www.testbank.org/pr/4",
                    element_ref="e1",
                ),
            ],
        )

        browser.navigate.return_value = page1
        browser.click.return_value = page2

        # LLM responses: filter page1 links, pagination ref, filter page2, no more pages
        fake_llm = FakeMessagesListChatModel(
            responses=[
                AIMessage(content='["e0", "e1"]'),  # filter links page 1
                AIMessage(content='{"element_ref": "e10"}'),  # pagination
                AIMessage(content='["e0", "e1"]'),  # filter links page 2
                AIMessage(content="null"),  # no more pages
            ],
        )

        result = find_press_releases(bank, browser, fake_llm, max_pages=3)

        assert result.pages_visited >= 2
        all_urls = [pr.url for pr in result.press_releases]
        assert "https://www.testbank.org/pr/1" in all_urls
        assert "https://www.testbank.org/pr/3" in all_urls

    def test_pagination_stops_at_max_pages(self) -> None:
        bank = _make_bank(press_releases_url="https://www.testbank.org/press")
        browser = MagicMock(spec=BrowserAdapter)

        page = _make_snapshot(
            links=[
                PageLink(
                    text="PR",
                    url="https://www.testbank.org/pr/1",
                    element_ref="e0",
                ),
                PageLink(
                    text="Next",
                    url="/press?page=2",
                    element_ref="e10",
                ),
            ],
        )
        browser.navigate.return_value = page
        browser.click.return_value = page

        fake_llm = FakeMessagesListChatModel(
            responses=[
                AIMessage(content='["e0"]'),  # filter links page 1
                AIMessage(content='{"element_ref": "e10"}'),  # pagination
                AIMessage(content='["e0"]'),  # filter links page 2
            ],
        )

        result = find_press_releases(bank, browser, fake_llm, max_pages=2)

        # max_pages=2 means at most 2 listing pages total (initial + 1 pagination)
        assert result.pages_visited <= 2

    def test_pagination_stops_when_no_next_link(self) -> None:
        bank = _make_bank(press_releases_url="https://www.testbank.org/press")
        browser = MagicMock(spec=BrowserAdapter)

        page = _make_snapshot(
            links=[
                PageLink(
                    text="PR 1",
                    url="https://www.testbank.org/pr/1",
                    element_ref="e0",
                ),
            ],
        )
        browser.navigate.return_value = page
        fake_llm = FakeMessagesListChatModel(
            responses=[
                AIMessage(content='["e0"]'),  # filter links page 1
                AIMessage(content="null"),  # no pagination link
            ],
        )

        result = find_press_releases(bank, browser, fake_llm, max_pages=5)

        assert result.pages_visited == 1


# ---------------------------------------------------------------------------
# 4. test_navigation_steps_logged
# ---------------------------------------------------------------------------


class TestNavigationStepsLogged:
    """Every navigation step is recorded in the result and emitted via logging."""

    def test_direct_url_logs_single_step(self) -> None:
        bank = _make_bank(press_releases_url="https://www.testbank.org/press")
        browser = MagicMock(spec=BrowserAdapter)
        browser.navigate.return_value = _make_snapshot(links=[])
        llm = MagicMock()

        result = find_press_releases(bank, browser, llm, max_pages=1)

        assert len(result.navigation_steps) >= 1
        step = result.navigation_steps[0]
        assert step.action == "direct_url"
        assert step.url == "https://www.testbank.org/press"

    def test_navigation_steps_have_required_fields(self) -> None:
        bank = _make_bank(press_releases_url="https://www.testbank.org/press")
        browser = MagicMock(spec=BrowserAdapter)
        browser.navigate.return_value = _make_snapshot(links=[])
        llm = MagicMock()

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
        browser = MagicMock(spec=BrowserAdapter)
        browser.navigate.return_value = _make_snapshot(links=[])
        llm = MagicMock()

        with caplog.at_level(logging.INFO, logger="cbs.scraper.navigator"):
            find_press_releases(bank, browser, llm, max_pages=1)

        assert any("direct_url" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# 5. test_filter_prompt_and_safety_net
# ---------------------------------------------------------------------------


class TestFilterPromptAndSafetyNet:
    """Filter prompt includes bank context; safety net catches empty results."""

    def test_filter_prompt_includes_bank_name(self) -> None:
        """The filter prompt sent to the LLM contains the bank name."""
        snapshot = _make_snapshot(
            links=[
                PageLink(text="PR 1", url="https://example.com/pr/1", element_ref="e0"),
            ],
        )
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content='["e0"]')

        _extract_press_releases_from_snapshot(
            snapshot, llm, bank_name="Federal Reserve", page_url="https://fed.gov/press"
        )

        prompt_text = llm.invoke.call_args[0][0][0].content
        assert "Federal Reserve" in prompt_text
        assert "https://fed.gov/press" in prompt_text

    def test_empty_result_safety_net_returns_all_links(self) -> None:
        """When LLM returns [] but page has >= 5 links, fall back to all."""
        links = [
            PageLink(
                text=f"PR {i}",
                url=f"https://example.com/pr/{i}",
                element_ref=f"e{i}",
            )
            for i in range(10)
        ]
        snapshot = _make_snapshot(links=links)
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content="[]")

        result = _extract_press_releases_from_snapshot(
            snapshot, llm, bank_name="Test Bank", page_url="https://example.com/press"
        )

        assert len(result) == 10

    def test_empty_result_safety_net_skipped_for_few_links(self) -> None:
        """When LLM returns [] and page has < 5 links, respect the empty result."""
        links = [
            PageLink(
                text=f"PR {i}",
                url=f"https://example.com/pr/{i}",
                element_ref=f"e{i}",
            )
            for i in range(3)
        ]
        snapshot = _make_snapshot(links=links)
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content="[]")

        result = _extract_press_releases_from_snapshot(
            snapshot, llm, bank_name="Test Bank", page_url="https://example.com/press"
        )

        assert len(result) == 0


# ---------------------------------------------------------------------------
# 6. test_off_domain_urls_filtered
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
