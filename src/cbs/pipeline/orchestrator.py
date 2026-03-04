"""Pipeline orchestrator — wires all stages together (Slice 1.11).

Flow: extract → translate → classify → extract swaps → store.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import date

import duckdb
from langchain_core.language_models.chat_models import BaseChatModel

from cbs.db.press_release_repo import PressRelease, insert_press_release
from cbs.db.swap_repo import SwapCreate, insert_swap
from cbs.pipeline.classifier import classify_press_release
from cbs.pipeline.extractor import extract_swaps
from cbs.pipeline.translator import detect_language, translate_text
from cbs.scraper.browser import BrowserAdapter
from cbs.scraper.html_extractor import HtmlExtractResult

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result of processing a single press release through the pipeline."""

    press_release_id: uuid.UUID | None = None
    swap_ids: list[uuid.UUID] = field(default_factory=list)
    skipped_duplicate: bool = False
    skipped_not_swap: bool = False


class Orchestrator:
    """Chains all pipeline stages for processing press releases.

    Stages:
    1. Deduplication (URL-based)
    2. Insert press release
    3. Detect language + translate to English
    4. Classify as swap-related or not
    5. If swap-related: extract structured swap data
    6. Store swap rows + mark press release as processed
    """

    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        llm: BaseChatModel,
        browser: BrowserAdapter,
    ) -> None:
        self._conn = conn
        self._llm = llm
        self._browser = browser

    def process_press_release(
        self,
        extract_result: HtmlExtractResult,
        *,
        bank_name: str,
        country: str,
        source_type: str = "html",
    ) -> PipelineResult:
        """Process a single press release through the full pipeline.

        Args:
            extract_result: Already-extracted content (from HTML or PDF).
            bank_name: Central bank name for the DB record.
            country: Country for the DB record.
            source_type: ``"html"`` or ``"pdf"``.

        Returns:
            PipelineResult describing what happened.
        """
        url = extract_result.url

        # 1. Deduplication — check if URL already exists
        existing = self._conn.execute(
            "SELECT 1 FROM press_releases WHERE url = ?", [url]
        ).fetchone()
        if existing is not None:
            logger.info("Skipping duplicate URL: %s", url)
            return PipelineResult(skipped_duplicate=True)

        # 2. Insert press release
        pr = PressRelease(
            central_bank_name=bank_name,
            country=country,
            url=url,
            title=extract_result.title,
            publication_date=extract_result.publication_date,
            original_language=extract_result.language,
            original_body=extract_result.body,
            source_type=source_type,
        )
        pr_id = insert_press_release(self._conn, pr)

        # 3. Detect language + translate
        lang_code = detect_language(self._llm, extract_result.body)
        translation = translate_text(
            self._llm, extract_result.body, original_language=lang_code
        )
        body_en = translation.body_en

        # 4. Classify
        classification = classify_press_release(self._llm, body_en)

        # 5. Update press release with classification + translation results
        self._conn.execute(
            "UPDATE press_releases SET "
            "body_en = ?, original_language = ?, "
            "is_swap_related = ?, classification_reason = ?, "
            "processed = TRUE, processed_at = current_timestamp "
            "WHERE id = ?",
            [
                body_en,
                lang_code,
                classification.is_swap_related,
                classification.reason,
                str(pr_id),
            ],
        )

        if not classification.is_swap_related:
            logger.info("Not swap-related: %s", url)
            return PipelineResult(
                press_release_id=pr_id,
                skipped_not_swap=True,
            )

        # 6. Extract swaps and insert each direction
        extraction = extract_swaps(self._llm, body_en)
        swap_ids: list[uuid.UUID] = []

        for swap_record in extraction.swaps:
            for direction in swap_record.directions:
                swap_create = SwapCreate(
                    press_release_id=pr_id,
                    provider_central_bank=direction.provider_central_bank,
                    provider_country=direction.provider_country,
                    receiver_central_bank=direction.receiver_central_bank,
                    receiver_country=direction.receiver_country,
                    currency=direction.currency,
                    swap_amount=direction.swap_amount,
                    swap_type=swap_record.swap_type,
                    announcement_type=swap_record.announcement_type,
                    type_of_change=swap_record.type_of_change,
                    conditions=swap_record.conditions,
                    reasons_for_swap=swap_record.reasons_for_swap,
                    announcement_date=_parse_date(swap_record.announcement_date),
                    effective_date=_parse_date(swap_record.effective_date),
                    maturity_date=_parse_date(swap_record.maturity_date),
                    maturity_text=swap_record.maturity_text,
                    duration_description=swap_record.duration_description,
                    raw_extract=swap_record.raw_extract,
                )
                swap_row = insert_swap(self._conn, swap_create)
                swap_ids.append(swap_row.id)

        logger.info("Processed %s: %d swap rows created", url, len(swap_ids))
        return PipelineResult(press_release_id=pr_id, swap_ids=swap_ids)


def _parse_date(value: str | None) -> date | None:
    """Parse an ISO 8601 date string, returning None if empty or invalid."""
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        logger.warning("Could not parse date: %s", value)
        return None
