"""Default bank processor — processes all press releases for a single bank."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langsmith import traceable

from cbs.config.banks import BankConfig
from cbs.pipeline.models import BankProcessingResult
from cbs.scraper.html_extractor import HtmlExtractResult
from cbs.scraper.navigator import find_press_releases
from cbs.scraper.pdf_extractor import PDFExtractResult, extract_pdf

if TYPE_CHECKING:
    from cbs.pipeline.orchestrator import Orchestrator
    from cbs.scraper.browser import BrowserAdapter

logger = logging.getLogger(__name__)

_MIN_BODY_LENGTH = 50

_ERROR_PATTERNS = ("404", "page not found", "not found", "error 404")


def _is_pdf_url(url: str) -> bool:
    """Check if a URL points to a PDF file."""
    return urlparse(url).path.lower().endswith(".pdf")


def _download_and_extract_pdf(url: str) -> PDFExtractResult:
    """Download a PDF from *url* and extract its text."""
    response = httpx.get(url, follow_redirects=True, timeout=60)
    response.raise_for_status()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(response.content)
        tmp_path = Path(tmp.name)

    try:
        return extract_pdf(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)


class DefaultBankProcessor:
    """Process all press releases for a single bank.

    Discovers press release URLs via the agentic navigator, then fetches
    and processes each one through the pipeline orchestrator.
    """

    def __init__(
        self,
        orchestrator: Orchestrator,
        browser: BrowserAdapter,
        llm: BaseChatModel,
        *,
        max_pages: int = 5,
    ) -> None:
        self._orchestrator = orchestrator
        self._browser = browser
        self._llm = llm
        self._max_pages = max_pages

    @traceable(name="process_bank", run_type="chain")
    def process_bank(self, bank: BankConfig) -> BankProcessingResult:
        """Discover and process all press releases for a bank."""
        result = BankProcessingResult(bank_name=bank.name)

        # 1. Find press release URLs
        try:
            nav_result = find_press_releases(
                bank, self._browser, self._llm, max_pages=self._max_pages
            )
        except Exception as exc:
            error_msg = f"Navigation/discovery failed for {bank.name}: {exc}"
            logger.error(error_msg)
            result.errors.append(error_msg)
            return result

        listing_url = nav_result.listing_page_url

        # 2. Process each discovered press release
        for pr in nav_result.press_releases:
            # Skip listing page URL
            if listing_url and pr.url == listing_url:
                logger.warning("Skipping listing page URL: %s", pr.url)
                continue

            # Route PDF vs HTML
            if _is_pdf_url(pr.url):
                try:
                    pdf_result = _download_and_extract_pdf(pr.url)
                except Exception as exc:
                    logger.error("PDF extraction failed for %s: %s", pr.url, exc)
                    result.errors.append(f"PDF extraction failed: {pr.url}: {exc}")
                    continue

                extract_result = HtmlExtractResult(
                    url=pr.url,
                    title=pr.title or "",
                    body=pdf_result.text,
                    publication_date=None,
                    language="en",
                )
                title = pr.title or ""
            else:
                try:
                    snapshot = self._browser.navigate(pr.url)
                except Exception as exc:
                    error_msg = f"Navigation error for {pr.url}: {exc}"
                    logger.error(error_msg)
                    result.errors.append(error_msg)
                    continue

                extract_result = HtmlExtractResult(
                    url=snapshot.url,
                    title=pr.title or snapshot.title,
                    body=snapshot.text_content,
                    publication_date=None,
                    language="en",
                )
                title = pr.title or snapshot.title

            # Guard: skip empty/short bodies
            body = extract_result.body.strip()
            if len(body) < _MIN_BODY_LENGTH:
                logger.warning(
                    "Skipping %s: body too short (%d chars)", pr.url, len(body)
                )
                result.errors.append(
                    f"Empty/short body for {pr.url} ({len(body)} chars)"
                )
                continue

            # Guard: skip 404/error pages
            lower_title = title.lower()
            lower_body_start = body[:500].lower()
            if any(p in lower_title or p in lower_body_start for p in _ERROR_PATTERNS):
                logger.warning("Skipping error page: %s", pr.url)
                result.errors.append(f"Error page: {pr.url}")
                continue

            try:
                pipeline_result = self._orchestrator.process_press_release(
                    extract_result,
                    bank_name=bank.name,
                    country=bank.country,
                )
            except Exception as exc:
                error_msg = f"Processing error for {pr.url}: {exc}"
                logger.error(error_msg)
                result.errors.append(error_msg)
                continue

            if pipeline_result.skipped_duplicate:
                result.skipped_duplicates += 1
            else:
                result.press_releases_found += 1
                result.swaps_extracted += len(pipeline_result.swap_ids)

        return result
