"""Pydantic models for navigation results."""

from __future__ import annotations

from pydantic import BaseModel


class NavigationStep(BaseModel):
    """A single step in the agent's navigation journey."""

    step_number: int
    action: str  # "direct_url", "navigate", "click", "paginate"
    url: str
    reasoning: str
    links_found: int


class DiscoveredPressRelease(BaseModel):
    """A press release URL discovered by the navigator."""

    url: str
    title: str | None = None


class NavigationResult(BaseModel):
    """Complete result of navigating a bank's website."""

    bank_name: str
    press_releases: list[DiscoveredPressRelease]
    navigation_steps: list[NavigationStep]
    listing_page_url: str | None = None
    pages_visited: int
    used_direct_url: bool
