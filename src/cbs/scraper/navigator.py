"""Agentic navigator — LLM-driven agent that finds press releases."""

from __future__ import annotations

import json
import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langsmith import traceable

from cbs.config.banks import BankConfig
from cbs.scraper.browser import BrowserAdapter, BrowserNavigationError, PageSnapshot
from cbs.scraper.models import (
    DiscoveredPressRelease,
    NavigationResult,
    NavigationStep,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class NavigationError(Exception):
    """Raised when the navigator cannot find press releases."""


# ---------------------------------------------------------------------------
# Agent prompt
# ---------------------------------------------------------------------------

_NAVIGATOR_SYSTEM_PROMPT = """\
You are a web navigation agent. Your task is to find the press releases \
section on a central bank website and collect links to individual press releases.

## Your Goal
Find the page that lists press releases for {bank_name} ({country}). \
Then extract all press release URLs from that listing page.

## Strategy
1. Look at the current page's links for sections like: \
"Press Releases", "News", "Media", "Publications", "Announcements", \
"Press Room", "Newsroom", "Communications"
2. If you see a promising section, click it using click_link.
3. You may need multiple clicks (e.g., "Media" → "Press Releases").
4. Once on the listing page, use extract_press_release_urls to collect links.
5. After extracting, respond with a final message (no tool call) confirming completion.

## Rules
- Do NOT click on individual press releases — only collect their URLs.
- Do NOT navigate away from the central bank's domain.
- If you cannot find press releases after 10 steps, stop and report failure.
"""

_PAGINATION_PROMPT = """\
Given these links on a press releases listing page, identify the "next page" \
link (for pagination). Look for links with text like "Next", "Older", \
"Load More", page numbers (2, 3, ...), "»", or ">".

Links:
{links_text}

Respond with ONLY a JSON object: {{"element_ref": "<ref>"}} if you find \
a pagination link, or "null" if there is no pagination link.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_snapshot_for_agent(snapshot: PageSnapshot) -> str:
    """Format a page snapshot as a concise string for the LLM agent."""
    lines = [f"Page: {snapshot.title}", f"URL: {snapshot.url}", "", "Links:"]
    for link in snapshot.links:
        lines.append(f"  [{link.element_ref}] {link.text} → {link.url}")
    return "\n".join(lines)


def _format_links_text(snapshot: PageSnapshot) -> str:
    """Format links for the pagination prompt."""
    lines = []
    for link in snapshot.links:
        lines.append(f"[{link.element_ref}] {link.text} → {link.url}")
    return "\n".join(lines)


_FILTER_LINKS_PROMPT = """\
You are filtering links from the {bank_name} press releases listing page ({page_url}).
Identify links that point to individual press releases, \
news articles, announcements, or statements.

Heuristics for press release links:
- Often contain dates in the URL or text (e.g., 2024, 2023-01-15)
- Often have descriptive titles about policy decisions, agreements, rates
- URL paths often include: /press/, /pr/, /news/, /release/, /announcement/
- They are the MAIN CONTENT links on a listing page

Exclude ONLY links that are clearly:
- Site navigation (Home, About, Contact, Login)
- Footer links (Privacy, Terms, Accessibility)
- Social media links (Twitter, Facebook, LinkedIn)
- Category/filter links (e.g., "Filter by year", "All categories")
- Pagination links (Next, Previous, page numbers)

Err on the side of inclusion. \
A press releases listing page typically has many press release links.

Links:
{links_text}

Respond with ONLY a JSON array of element_ref strings. Example: ["e5", "e7", "e12"]
"""


def _extract_press_releases_from_snapshot(
    snapshot: PageSnapshot,
    llm: BaseChatModel | None = None,
    *,
    bank_name: str = "",
    page_url: str = "",
) -> list[DiscoveredPressRelease]:
    """Extract press release links from a snapshot.

    If *llm* is provided, uses the LLM to filter out navigation/footer
    links.  Otherwise falls back to returning all links.
    """
    if not snapshot.links:
        return []

    if llm is None:
        return [
            DiscoveredPressRelease(url=link.url, title=link.text)
            for link in snapshot.links
        ]

    links_text = "\n".join(
        f"[{link.element_ref}] {link.text} → {link.url}" for link in snapshot.links
    )
    prompt = _FILTER_LINKS_PROMPT.format(
        bank_name=bank_name,
        page_url=page_url,
        links_text=links_text,
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content if isinstance(response.content, str) else ""
        content = content.strip()
        if not content:
            raise ValueError("Empty LLM response")
        # Strip markdown code fences if present
        if content.startswith("```"):
            content = content.split("\n", 1)[1] if "\n" in content else content
        if content.endswith("```"):
            content = content.rsplit("```", 1)[0].strip()
        pr_refs: list[str] = json.loads(content)
        if not isinstance(pr_refs, list):
            raise ValueError("Expected list")
    except Exception:
        logger.warning("Failed to filter links via LLM, using all links")
        return [
            DiscoveredPressRelease(url=link.url, title=link.text)
            for link in snapshot.links
        ]

    # Safety net: a listing page with many links but 0 press releases
    # is almost certainly a filtering failure — return all links and let
    # the downstream classifier handle false positives.
    if len(pr_refs) == 0 and len(snapshot.links) >= 5:
        logger.warning(
            "LLM returned 0 press releases from %d links on %s — "
            "falling back to all links",
            len(snapshot.links),
            page_url,
        )
        return [
            DiscoveredPressRelease(url=link.url, title=link.text)
            for link in snapshot.links
        ]

    ref_set = set(pr_refs)
    return [
        DiscoveredPressRelease(url=link.url, title=link.text)
        for link in snapshot.links
        if link.element_ref in ref_set
    ]


def _log_step(step: NavigationStep) -> None:
    """Emit a log message for a navigation step."""
    logger.info(
        "Step %d [%s]: %s — %s (%d links)",
        step.step_number,
        step.action,
        step.url,
        step.reasoning,
        step.links_found,
    )


# ---------------------------------------------------------------------------
# Pagination (uses focused LLM call, not full agent)
# ---------------------------------------------------------------------------


def _find_next_page_ref(
    llm: BaseChatModel,
    snapshot: PageSnapshot,
) -> str | None:
    """Ask the LLM to identify a pagination link on the current page.

    Returns the element_ref string if found, or None.
    """
    if not snapshot.links:
        return None

    prompt = _PAGINATION_PROMPT.format(links_text=_format_links_text(snapshot))
    response = llm.invoke([HumanMessage(content=prompt)])

    content = response.content if isinstance(response.content, str) else ""
    content = content.strip()

    if content == "null" or not content:
        return None

    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict) and "element_ref" in parsed:
            return str(parsed["element_ref"])
    except (json.JSONDecodeError, KeyError):
        pass

    return None


@traceable(name="paginate", run_type="chain")
def _paginate(
    browser: BrowserAdapter,
    llm: BaseChatModel,
    bank: BankConfig,
    current_snapshot: PageSnapshot,
    remaining_pages: int,
    steps: list[NavigationStep],
) -> list[DiscoveredPressRelease]:
    """Paginate through listing pages to discover more press releases."""
    all_releases: list[DiscoveredPressRelease] = []

    for _ in range(remaining_pages):
        next_ref = _find_next_page_ref(llm, current_snapshot)
        if next_ref is None:
            break

        try:
            current_snapshot = browser.click(next_ref, timeout=bank.page_load_timeout)
        except BrowserNavigationError:
            logger.warning("Pagination click failed for ref %s — stopping", next_ref)
            break

        step = NavigationStep(
            step_number=len(steps) + 1,
            action="paginate",
            url=current_snapshot.url,
            reasoning=f"Clicked pagination link: {next_ref}",
            links_found=len(current_snapshot.links),
        )
        steps.append(step)
        _log_step(step)

        new_prs = _extract_press_releases_from_snapshot(
            current_snapshot,
            llm,
            bank_name=bank.name,
            page_url=current_snapshot.url,
        )
        all_releases.extend(new_prs)

    return all_releases


# ---------------------------------------------------------------------------
# Tool execution (manual ReAct loop)
# ---------------------------------------------------------------------------

_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "click_link",
            "description": (
                "Click a link on the current page by its element reference. "
                "The element_ref is the identifier (e.g., 'e1') shown next "
                "to each link in the page snapshot."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "element_ref": {
                        "type": "string",
                        "description": "The element reference of the link.",
                    }
                },
                "required": ["element_ref"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_press_release_urls",
            "description": (
                "Extract all links from the current page as press release "
                "candidates. Use once you've navigated to the listing page."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]

_MAX_AGENT_STEPS = 10


def _execute_tool_call(
    tool_name: str,
    tool_args: dict[str, str],
    browser: BrowserAdapter,
    timeout: int,
    steps: list[NavigationStep],
) -> str:
    """Execute a single tool call and return the result string."""
    if tool_name == "click_link":
        element_ref = tool_args.get("element_ref", "")
        snapshot = browser.click(element_ref, timeout=timeout)
        step = NavigationStep(
            step_number=len(steps) + 1,
            action="click",
            url=snapshot.url,
            reasoning=f"Clicked element {element_ref}",
            links_found=len(snapshot.links),
        )
        steps.append(step)
        _log_step(step)
        return _format_snapshot_for_agent(snapshot)

    if tool_name == "extract_press_release_urls":
        snapshot = browser.get_snapshot()
        lines = [f"{link.text} → {link.url}" for link in snapshot.links]
        return "\n".join(lines) if lines else "No links found on this page."

    return f"Unknown tool: {tool_name}"


# ---------------------------------------------------------------------------
# Agent-based discovery (manual ReAct loop)
# ---------------------------------------------------------------------------


@traceable(name="discovery_agent", run_type="chain")
def _run_discovery_agent(
    bank: BankConfig,
    browser: BrowserAdapter,
    llm: BaseChatModel,
    steps: list[NavigationStep],
) -> list[DiscoveredPressRelease]:
    """Use an LLM agent to discover the press releases section.

    Implements a ReAct loop: call LLM → execute tool calls → repeat
    until the LLM responds without tool calls (indicating completion).
    """
    # Navigate to homepage first
    homepage = browser.navigate(str(bank.homepage_url), timeout=bank.page_load_timeout)
    step = NavigationStep(
        step_number=len(steps) + 1,
        action="navigate",
        url=str(bank.homepage_url),
        reasoning="Starting from homepage",
        links_found=len(homepage.links),
    )
    steps.append(step)
    _log_step(step)

    system_prompt = _NAVIGATOR_SYSTEM_PROMPT.format(
        bank_name=bank.name,
        country=bank.country,
    )

    initial_content = (
        "I've navigated to the homepage. Here's what I see:\n\n"
        + _format_snapshot_for_agent(homepage)
    )

    messages: list[BaseMessage] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=initial_content),
    ]

    # Bind tools to the LLM if it supports it, otherwise pass raw messages
    try:
        bound_llm = llm.bind_tools(_TOOL_SCHEMAS)
    except NotImplementedError:
        # FakeMessagesListChatModel and similar don't support bind_tools.
        # Fall back to using the raw LLM — it will return predetermined
        # AIMessages with tool_calls already set (for testing).
        bound_llm = llm

    for _ in range(_MAX_AGENT_STEPS):
        response = bound_llm.invoke(messages)
        messages.append(response)

        # If no tool calls, the agent is done
        if not isinstance(response, AIMessage) or not response.tool_calls:
            break

        # Execute each tool call and append results
        for tc in response.tool_calls:
            result_str = _execute_tool_call(
                tc["name"],
                tc.get("args", {}),
                browser,
                bank.page_load_timeout,
                steps,
            )
            messages.append(ToolMessage(content=result_str, tool_call_id=tc["id"]))

    # Extract press releases from the final page the agent landed on
    final_snapshot = browser.get_snapshot()
    return _extract_press_releases_from_snapshot(
        final_snapshot,
        llm,
        bank_name=bank.name,
        page_url=final_snapshot.url,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@traceable(name="find_press_releases", run_type="chain")
def find_press_releases(
    bank: BankConfig,
    browser: BrowserAdapter,
    llm: BaseChatModel,
    *,
    max_pages: int = 5,
) -> NavigationResult:
    """Navigate a central bank website and discover press release URLs.

    Two modes:

    1. **Direct URL**: If ``bank.press_releases_url`` is set, navigate there
       directly and scrape the listing page.
    2. **Discovery**: If only ``bank.homepage_url`` is available, use an LLM
       agent to autonomously find the press releases section.

    In both modes, the function paginates through the listing to discover
    additional press release URLs (up to *max_pages* pages).

    Args:
        bank: Configuration for the target central bank.
        browser: Browser adapter for page navigation.
        llm: LangChain chat model for agent reasoning.
        max_pages: Maximum number of listing pages to visit.

    Returns:
        NavigationResult with discovered press release URLs and step log.

    Raises:
        NavigationError: If the agent cannot find the press releases section.
    """
    steps: list[NavigationStep] = []

    # ── Mode 1: Direct URL ─────────────────────────────────────────────
    if bank.press_releases_url:
        snapshot = browser.navigate(
            str(bank.press_releases_url),
            timeout=bank.page_load_timeout,
        )
        step = NavigationStep(
            step_number=1,
            action="direct_url",
            url=str(bank.press_releases_url),
            reasoning="Bank config has press_releases_url configured",
            links_found=len(snapshot.links),
        )
        steps.append(step)
        _log_step(step)

        press_releases = _extract_press_releases_from_snapshot(
            snapshot,
            llm,
            bank_name=bank.name,
            page_url=str(bank.press_releases_url),
        )

        # Paginate for more releases
        if max_pages > 1:
            press_releases += _paginate(
                browser, llm, bank, snapshot, max_pages - 1, steps
            )

        return NavigationResult(
            bank_name=bank.name,
            press_releases=press_releases,
            navigation_steps=steps,
            listing_page_url=str(bank.press_releases_url),
            pages_visited=len(steps),
            used_direct_url=True,
        )

    # ── Mode 2: LLM agent discovery ───────────────────────────────────
    press_releases = _run_discovery_agent(bank, browser, llm, steps)

    # Paginate from wherever the agent ended up
    if max_pages > 1:
        current_snapshot = browser.get_snapshot()
        press_releases += _paginate(
            browser, llm, bank, current_snapshot, max_pages - 1, steps
        )

    return NavigationResult(
        bank_name=bank.name,
        press_releases=press_releases,
        navigation_steps=steps,
        listing_page_url=None,
        pages_visited=len(steps),
        used_direct_url=False,
    )
