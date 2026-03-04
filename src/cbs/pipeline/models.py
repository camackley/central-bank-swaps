"""Pydantic models for pipeline results and run summaries."""

from __future__ import annotations

from pydantic import BaseModel


class BankProcessingResult(BaseModel):
    """Result of processing a single bank's press releases."""

    bank_name: str
    press_releases_found: int = 0
    skipped_duplicates: int = 0
    swaps_extracted: int = 0
    errors: list[str] = []
    hit_cutoff: bool = False


class RunSummary(BaseModel):
    """Aggregated summary of a complete backfill run."""

    banks_attempted: int = 0
    banks_succeeded: int = 0
    press_releases_found: int = 0
    swaps_extracted: int = 0
    errors: list[str] = []
