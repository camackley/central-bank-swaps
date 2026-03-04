"""Tests for HTML press release extraction (Slice 1.7 — FR-002)."""

from __future__ import annotations

import datetime
from pathlib import Path

from cbs.scraper.html_extractor import HtmlExtractResult, extract_press_release

FIXTURES = Path(__file__).parent.parent / "fixtures" / "html"


def _load_fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Fed fixture
# ---------------------------------------------------------------------------
class TestExtractTitleFromHtml:
    """FR-002: Extracts press release title."""

    def test_fed_title(self) -> None:
        html = _load_fixture("fed_swap_press_release.html")
        result = extract_press_release(
            html,
            url="https://www.federalreserve.gov/newsevents/pressreleases/monetary20200319a.htm",
        )
        assert (
            "Federal Reserve announces the establishment of temporary" in result.title
        )

    def test_ecb_title(self) -> None:
        html = _load_fixture("ecb_swap_press_release.html")
        result = extract_press_release(
            html,
            url="https://www.ecb.europa.eu/press/pr/date/2020/html/ecb.pr200315~1fab6a9b89.en.html",
        )
        assert "Coordinated central bank action" in result.title

    def test_boj_title(self) -> None:
        html = _load_fixture("boj_swap_press_release.html")
        result = extract_press_release(
            html,
            url="https://www.boj.or.jp/announcements/release_2020/rel200315b.htm",
        )
        assert "米ドル資金供給オペレーション" in result.title


class TestExtractCleanBodyNoNavNoBoilerplate:
    """FR-002: Extracts cleaned plain text body — no HTML tags, nav, or boilerplate."""

    def test_body_contains_article_text(self) -> None:
        html = _load_fixture("fed_swap_press_release.html")
        result = extract_press_release(html, url="https://example.com/press/1")
        assert "swap lines" in result.body
        assert "Reserve Bank of Australia" in result.body

    def test_body_excludes_nav_elements(self) -> None:
        html = _load_fixture("fed_swap_press_release.html")
        result = extract_press_release(html, url="https://example.com/press/1")
        # Nav links should not appear in the cleaned body
        assert "Supervision & Regulation" not in result.body
        assert "Financial Stability" not in result.body

    def test_body_excludes_footer(self) -> None:
        html = _load_fixture("fed_swap_press_release.html")
        result = extract_press_release(html, url="https://example.com/press/1")
        assert "20th Street and Constitution Avenue" not in result.body

    def test_body_excludes_breadcrumbs(self) -> None:
        html = _load_fixture("fed_swap_press_release.html")
        result = extract_press_release(html, url="https://example.com/press/1")
        # Breadcrumb text should not be in the body
        assert "Home\nNews" not in result.body

    def test_body_excludes_sidebar(self) -> None:
        html = _load_fixture("fed_swap_press_release.html")
        result = extract_press_release(html, url="https://example.com/press/1")
        assert "Enhanced swap lines" not in result.body

    def test_body_has_no_html_tags(self) -> None:
        html = _load_fixture("fed_swap_press_release.html")
        result = extract_press_release(html, url="https://example.com/press/1")
        assert "<p>" not in result.body
        assert "<div>" not in result.body
        assert "<a " not in result.body

    def test_ecb_body_content(self) -> None:
        html = _load_fixture("ecb_swap_press_release.html")
        result = extract_press_release(html, url="https://example.com/press/2")
        assert "25 basis points" in result.body
        # Should exclude footer
        assert "Reproduction is permitted" not in result.body


class TestExtractPublicationDate:
    """FR-002: Extracts publication date."""

    def test_fed_date_from_meta(self) -> None:
        html = _load_fixture("fed_swap_press_release.html")
        result = extract_press_release(html, url="https://example.com/press/1")
        assert result.publication_date == datetime.date(2020, 3, 19)

    def test_ecb_date_from_meta(self) -> None:
        html = _load_fixture("ecb_swap_press_release.html")
        result = extract_press_release(html, url="https://example.com/press/2")
        assert result.publication_date == datetime.date(2020, 3, 15)

    def test_boj_date_from_meta(self) -> None:
        html = _load_fixture("boj_swap_press_release.html")
        result = extract_press_release(html, url="https://example.com/press/3")
        assert result.publication_date == datetime.date(2020, 3, 15)


class TestExtractCanonicalUrl:
    """FR-002: Extracts canonical/permalink URL."""

    def test_canonical_url_from_link_tag(self) -> None:
        html = _load_fixture("fed_swap_press_release.html")
        result = extract_press_release(html, url="https://example.com/press/1")
        assert (
            result.url
            == "https://www.federalreserve.gov/newsevents/pressreleases/monetary20200319a.htm"
        )

    def test_ecb_canonical_url(self) -> None:
        html = _load_fixture("ecb_swap_press_release.html")
        result = extract_press_release(html, url="https://example.com/press/2")
        assert (
            result.url
            == "https://www.ecb.europa.eu/press/pr/date/2020/html/ecb.pr200315~1fab6a9b89.en.html"
        )

    def test_falls_back_to_provided_url(self) -> None:
        """When no canonical link exists, use the URL passed in."""
        html = (
            "<html><head><title>No canonical</title>"
            "</head><body><p>Text</p></body></html>"
        )
        result = extract_press_release(html, url="https://example.com/fallback")
        assert result.url == "https://example.com/fallback"


class TestDetectLanguage:
    """FR-002: Detects and stores the original language."""

    def test_english_detected(self) -> None:
        html = _load_fixture("fed_swap_press_release.html")
        result = extract_press_release(html, url="https://example.com/press/1")
        assert result.language == "en"

    def test_japanese_detected(self) -> None:
        html = _load_fixture("boj_swap_press_release.html")
        result = extract_press_release(html, url="https://example.com/press/3")
        assert result.language == "ja"

    def test_ecb_english(self) -> None:
        html = _load_fixture("ecb_swap_press_release.html")
        result = extract_press_release(html, url="https://example.com/press/2")
        assert result.language == "en"


class TestHtmlExtractResult:
    """Verify the result model has all required fields."""

    def test_result_has_all_fields(self) -> None:
        html = _load_fixture("fed_swap_press_release.html")
        result = extract_press_release(html, url="https://example.com/press/1")
        assert isinstance(result, HtmlExtractResult)
        assert isinstance(result.url, str)
        assert isinstance(result.title, str)
        assert isinstance(result.body, str)
        assert isinstance(result.publication_date, datetime.date | None)
        assert isinstance(result.language, str)

    def test_body_is_not_empty(self) -> None:
        html = _load_fixture("fed_swap_press_release.html")
        result = extract_press_release(html, url="https://example.com/press/1")
        assert len(result.body.strip()) > 100
