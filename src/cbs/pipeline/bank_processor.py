"""Default bank processor — processes all press releases for a single bank."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.language_models.chat_models import BaseChatModel

from cbs.config.banks import BankConfig
from cbs.pipeline.models import BankProcessingResult
from cbs.scraper.html_extractor import HtmlExtractResult
from cbs.scraper.navigator import find_press_releases

if TYPE_CHECKING:
    from cbs.pipeline.orchestrator import Orchestrator
    from cbs.scraper.browser import BrowserAdapter

logger = logging.getLogger(__name__)


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

    def process_bank(self, bank: BankConfig) -> BankProcessingResult:
        """Discover and process all press releases for a bank."""
        result = BankProcessingResult(bank_name=bank.name)

        # 1. Find press release URLs
        nav_result = find_press_releases(
            bank, self._browser, self._llm, max_pages=self._max_pages
        )

        # 2. Process each discovered press release
        for pr in nav_result.press_releases:
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
