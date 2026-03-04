"""Protocols for pipeline components — implemented by later slices."""

from __future__ import annotations

from typing import Protocol

from cbs.config.banks import BankConfig
from cbs.pipeline.models import BankProcessingResult


class BankProcessor(Protocol):
    """Process all press releases for a single bank.

    Implementations (Slice 1.11) will:
    1. Navigate to the bank's press releases page
    2. Paginate through press releases, stopping at historical_cutoff_year
    3. For each press release: extract, translate, classify, extract swaps
    4. Return a BankProcessingResult with counts and any errors
    """

    def process_bank(self, bank: BankConfig) -> BankProcessingResult: ...
