"""HTML press release extractor (Slice 1.7 — FR-002).

Given raw HTML of a press release page, extracts:
- URL (canonical or provided fallback)
- Title
- Cleaned plain-text body (no nav, footer, sidebar, boilerplate)
- Publication date
- Original language (ISO 639-1)
"""

from __future__ import annotations

import datetime
import re

from bs4 import BeautifulSoup, Tag
from pydantic import BaseModel


class HtmlExtractResult(BaseModel):
    """Structured result of HTML press release extraction."""

    url: str
    title: str
    body: str
    publication_date: datetime.date | None
    language: str


# Tags / selectors that are always boilerplate.
_BOILERPLATE_TAGS = {"nav", "header", "footer", "aside", "script", "style", "noscript"}
_BOILERPLATE_IDS = {
    "header",
    "footer",
    "main-nav",
    "global-nav",
    "breadcrumbs",
    "breadcrumb",
}
_BOILERPLATE_ROLES = {"navigation"}


def _attr_str(tag: Tag, attr: str) -> str:
    """Get a tag attribute as a lowercase string (handles list attrs)."""
    val = tag.get(attr)
    if isinstance(val, list):
        return " ".join(val).lower()
    if isinstance(val, str):
        return val.lower()
    return ""


def _is_boilerplate(tag: Tag) -> bool:
    """Check if a tag is boilerplate (nav, footer, sidebar, etc.)."""
    if tag.name in _BOILERPLATE_TAGS:
        return True
    if _attr_str(tag, "id") in _BOILERPLATE_IDS:
        return True
    if _attr_str(tag, "role") in _BOILERPLATE_ROLES:
        return True
    classes = _attr_str(tag, "class")
    return any(kw in classes for kw in ("breadcrumb", "sidebar", "footer", "nav"))


def _remove_boilerplate(soup: BeautifulSoup) -> None:
    """Strip nav, header, footer, aside, script, style, and breadcrumb elements."""
    to_remove = [
        tag
        for tag in soup.find_all(True)
        if isinstance(tag, Tag) and _is_boilerplate(tag)
    ]
    for tag in to_remove:
        tag.decompose()


def _extract_canonical_url(soup: BeautifulSoup, fallback_url: str) -> str:
    """Return canonical URL from <link rel='canonical'> or fall back."""
    link = soup.find("link", rel="canonical")
    if isinstance(link, Tag) and link.get("href"):
        return str(link["href"])
    return fallback_url


def _extract_title(soup: BeautifulSoup) -> str:
    """Extract the best title: prefer <h1>/<h3> in article, fall back to <title>."""
    # Try article headings first
    for selector in ("h1", "h3.title", "h2"):
        heading = soup.find(selector)
        if isinstance(heading, Tag):
            text = heading.get_text(strip=True)
            if text:
                return text

    # Fall back to <title> tag (strip site suffix)
    title_tag = soup.find("title")
    if isinstance(title_tag, Tag):
        raw = title_tag.get_text(strip=True)
        # Remove common suffixes like " -- Federal Reserve Board"
        for sep in (" -- ", " | ", " - ", " — "):
            if sep in raw:
                return raw.split(sep)[0].strip()
        return raw

    return ""


_DATE_META_NAMES = ("DC.date", "date", "article:published_time", "dcterms.date")

# Patterns: YYYY-MM-DD, YYYY/MM/DD
_ISO_DATE_RE = re.compile(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})")


def _extract_date(soup: BeautifulSoup) -> datetime.date | None:
    """Extract publication date from meta tags or <time> elements."""
    # Try meta tags
    for name in _DATE_META_NAMES:
        meta = soup.find("meta", attrs={"name": name})
        if isinstance(meta, Tag) and meta.get("content"):
            m = _ISO_DATE_RE.search(str(meta["content"]))
            if m:
                return datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))

    # Try <time datetime="...">
    time_tag = soup.find("time")
    if isinstance(time_tag, Tag) and time_tag.get("datetime"):
        m = _ISO_DATE_RE.search(str(time_tag["datetime"]))
        if m:
            return datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))

    return None


def _detect_language(soup: BeautifulSoup) -> str:
    """Detect language from <html lang=''>, meta tags, or default to 'en'."""
    # <html lang="...">
    html_tag = soup.find("html")
    if isinstance(html_tag, Tag) and html_tag.get("lang"):
        lang = str(html_tag["lang"]).strip().lower()
        # Normalize "en-US" → "en"
        return lang.split("-")[0]

    # <meta name="DC.language" content="...">
    for name in ("DC.language", "content-language", "language"):
        meta = soup.find("meta", attrs={"name": name})
        if isinstance(meta, Tag) and meta.get("content"):
            return str(meta["content"]).strip().lower().split("-")[0]

    # <meta http-equiv="Content-Language" content="...">
    meta_http = soup.find("meta", attrs={"http-equiv": "Content-Language"})
    if isinstance(meta_http, Tag) and meta_http.get("content"):
        return str(meta_http["content"]).strip().lower().split("-")[0]

    return "en"


def _extract_body(soup: BeautifulSoup) -> str:
    """Extract clean text from the main content area after boilerplate removal."""
    # Look for <article> or <main> or #article
    content = (
        soup.find("article")
        or soup.find("main")
        or soup.find(id="article")
        or soup.find(id="main-content")
        or soup.body
    )
    if content is None:
        return ""

    text = content.get_text(separator="\n", strip=True)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_press_release(html: str, *, url: str) -> HtmlExtractResult:
    """Extract structured data from a press release HTML page.

    Args:
        html: Raw HTML source of the press release page.
        url: The URL the page was fetched from (used as fallback for canonical URL).

    Returns:
        HtmlExtractResult with url, title, body, publication_date, language.
    """
    soup = BeautifulSoup(html, "lxml")

    # Extract canonical URL and date BEFORE removing boilerplate
    canonical_url = _extract_canonical_url(soup, url)
    publication_date = _extract_date(soup)
    language = _detect_language(soup)
    title = _extract_title(soup)

    # Remove boilerplate, then extract body text
    _remove_boilerplate(soup)
    body = _extract_body(soup)

    return HtmlExtractResult(
        url=canonical_url,
        title=title,
        body=body,
        publication_date=publication_date,
        language=language,
    )
