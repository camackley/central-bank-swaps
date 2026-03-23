"""Agentic navigator — LLM-driven agent that finds press releases."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from bs4 import BeautifulSoup
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

if TYPE_CHECKING:
    from cbs.llm.claude_code_model import ClaudeCodeChatModel

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
link (for pagination). Look for links with text like:
- English: "Next", "Older", "Load More", ">", "»", page numbers (2, 3, ...)
- Spanish: "Siguiente", "Próxima", "Más", "›"
- French: "Suivant", "Page suivante"
- German: "Weiter", "Nächste"
- Portuguese: "Próxima", "Seguinte"
- Japanese/Chinese: page numbers or arrow symbols

Links:
{links_text}

Respond with ONLY a JSON object: {{"element_ref": "<url>"}} if you find \
a pagination link (use the exact URL shown), or "null" if there is no \
pagination link.
"""


# ---------------------------------------------------------------------------
# HTML-based URL extraction (main path for direct_url mode)
# ---------------------------------------------------------------------------

# After removing boilerplate (nav/header/footer/scripts/styles), listing pages
# are typically well under 100K chars. Cap at 150K as a safety margin.
_HTML_CHAR_LIMIT = 150_000

# Only strip by semantic HTML5 tag name — avoids class-based false positives
# on listing pages where article wrappers may have classes like "nav-section".
_BOILERPLATE_TAGS = {"nav", "header", "footer", "aside", "script", "style", "noscript"}


def _clean_html_for_llm(html: str) -> str:
    """Strip semantic boilerplate tags from HTML, return cleaned HTML string.

    Uses only tag-name matching (not class names) to avoid accidentally removing
    article content on listing pages. Reduces pages like BanRep (374K raw) to
    just the content area before sending to the LLM for URL extraction.
    """
    soup = BeautifulSoup(html, "html.parser")
    # Collect first, decompose after — avoids mutating the tree during iteration.
    to_remove = [tag for tag in soup.find_all(_BOILERPLATE_TAGS)]
    for tag in to_remove:
        tag.decompose()
    return str(soup)


_EXTRACT_URLS_PROMPT = """\
You are analyzing the HTML of {bank_name}'s press releases listing page.
URL: {page_url}

Extract ALL URLs that are individual press releases, news articles, or \
official announcements from this central bank. These are typically links \
with dates and specific event titles (rate decisions, policy statements, \
meeting minutes, swap agreements, etc.).

Exclude:
- Site navigation links (Home, About, Contact, Login)
- Footer links (Privacy, Terms, Sitemap)
- Social media links
- Category/filter controls
- Pagination links (Next, Previous, page numbers)
- Links to section index pages (not individual articles)

Err on the side of inclusion — a downstream classifier handles false positives.

Return ONLY a JSON array of absolute URL strings. Example:
["https://example.com/news/2024/article-1", "https://example.com/news/2024/article-2"]

HTML content:
{html_content}
"""


def _extract_urls_from_html(
    html: str,
    llm: BaseChatModel,
    *,
    bank_name: str,
    page_url: str,
) -> list[DiscoveredPressRelease]:
    """Ask Claude to extract press release URLs from the full rendered HTML.

    Falls back to an empty list on parse failure (caller handles gracefully).
    """
    stripped = _clean_html_for_llm(html)
    truncated = stripped[:_HTML_CHAR_LIMIT]
    prompt = _EXTRACT_URLS_PROMPT.format(
        bank_name=bank_name,
        page_url=page_url,
        html_content=truncated,
    )
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content if isinstance(response.content, str) else ""
        content = content.strip()
        if not content:
            raise ValueError("Empty LLM response")
        # Extract JSON array from anywhere in the response
        start = content.find("[")
        end = content.rfind("]")
        if start != -1 and end != -1 and end > start:
            content = content[start : end + 1]
        urls: list[str] = json.loads(content)
        if not isinstance(urls, list):
            raise ValueError("Expected list")
        results = [
            DiscoveredPressRelease(url=u, title=None)
            for u in urls
            if isinstance(u, str) and u.startswith("http")
        ]
        if not results:
            logger.warning(
                "HTML URL extraction for %s: LLM returned 0 URLs (raw: %r)",
                page_url,
                content[:200],
            )
            logger.debug("HTML head (first 500 chars): %r", html[:500])
        else:
            logger.info(
                "HTML URL extraction for %s: found %d URLs", page_url, len(results)
            )
        return results
    except Exception as exc:
        logger.warning(
            "HTML URL extraction failed for %s (%s) — falling back to snapshot links",
            page_url,
            exc,
        )
        logger.debug("HTML head (first 500 chars): %r", html[:500])
        return []


# ---------------------------------------------------------------------------
# Playwright MCP — browser tools let Claude navigate and extract directly
# ---------------------------------------------------------------------------

# Tools the claude subprocess is allowed to call via Playwright MCP.
_MCP_BROWSER_TOOLS = [
    "mcp__playwright__browser_navigate",
    "mcp__playwright__browser_snapshot",
    "mcp__playwright__browser_click",
    "mcp__playwright__browser_javascript",
]

# MCP calls involve multiple browser round-trips — allow generous timeout.
_MCP_TIMEOUT = 300

_MCP_DIRECT_PROMPT = """\
You have Playwright browser tools. Extract press release URLs from \
{bank_name} ({country}).

Steps:
1. browser_navigate to: {url}
2. browser_snapshot to see the page structure
3. From the snapshot, identify all links to individual press releases, \
news articles, or official announcements (rate decisions, policy \
statements, meeting minutes, swap agreements, etc.)
4. If the snapshot does not show full URLs for the links, use \
browser_javascript to run: \
Array.from(document.querySelectorAll('a[href]')).map(a => \
({{text: a.textContent.trim().slice(0,200), href: a.href}}))
{pagination_instruction}

Exclude: navigation links (Home, About, Contact), footer links, \
social media, category/filter controls, section index pages.

Err on the side of inclusion — a downstream classifier handles false positives.

Return ONLY a JSON array of absolute URL strings. No explanation, no markdown.
"""

_MCP_DISCOVERY_PROMPT = """\
You have Playwright browser tools. Find and extract press release URLs from \
{bank_name} ({country}).

Steps:
1. browser_navigate to: {url}
2. browser_snapshot to see the page structure
3. Look for sections like "Press Releases", "News", "Media", "Publications", \
"Announcements", "Press Room", "Newsroom", "Communications", "Noticias"
4. browser_click on the most promising link (use the element ref from snapshot)
5. browser_snapshot again — if you're on a listing page with individual \
article links, proceed to extraction. Otherwise click deeper.
6. Once on the listing page, identify all individual press release URLs.
7. If the snapshot does not show full URLs, use browser_javascript to run: \
Array.from(document.querySelectorAll('a[href]')).map(a => \
({{text: a.textContent.trim().slice(0,200), href: a.href}}))
{pagination_instruction}

Exclude: navigation links, footer links, social media, filters, index pages.
Err on the side of inclusion.

Return ONLY a JSON array of absolute URL strings. No explanation, no markdown.
"""


def _has_mcp_support(llm: BaseChatModel) -> bool:
    """Check if the LLM is a ClaudeCodeChatModel (supports MCP tools)."""
    from cbs.llm.claude_code_model import ClaudeCodeChatModel

    return isinstance(llm, ClaudeCodeChatModel)


@traceable(name="mcp_extract", run_type="chain")
def _extract_urls_via_mcp(
    llm: ClaudeCodeChatModel,
    url: str,
    bank: BankConfig,
    max_pages: int,
    *,
    is_direct_url: bool,
) -> list[DiscoveredPressRelease]:
    """Use Playwright MCP to navigate and extract press release URLs.

    Claude controls a headless Chromium directly via MCP tools within a
    single ``claude -p`` call.  The accessibility tree snapshot is ~2-5 KB
    per page (vs. ~150 KB of cleaned HTML), which eliminates the
    ``[Errno 7] Argument list too long`` issue and reduces token cost.

    Falls back to an empty list on failure — caller uses HTML approach.
    """
    pagination = ""
    if max_pages > 1:
        pagination = (
            f"\nAfter extracting URLs from the first page, look for pagination "
            f'("Next", "Siguiente", ">", page numbers). '
            f"If found, use browser_click on the pagination element ref, then "
            f"browser_snapshot and extract more URLs. "
            f"Repeat for up to {max_pages} pages total."
        )

    template = _MCP_DIRECT_PROMPT if is_direct_url else _MCP_DISCOVERY_PROMPT
    prompt = template.format(
        bank_name=bank.name,
        country=bank.country,
        url=url,
        pagination_instruction=pagination,
    )

    try:
        raw = llm._call_cli(
            system="",
            prompt=prompt,
            allowed_tools=_MCP_BROWSER_TOOLS,
            timeout_override=_MCP_TIMEOUT,
        )
        content = raw.strip()
        # Extract JSON array from anywhere in the response
        start = content.find("[")
        end = content.rfind("]")
        if start != -1 and end != -1 and end > start:
            content = content[start : end + 1]
        urls: list[str] = json.loads(content)
        if not isinstance(urls, list):
            raise ValueError("Expected list")
        results = [
            DiscoveredPressRelease(url=u, title=None)
            for u in urls
            if isinstance(u, str) and u.startswith("http")
        ]
        if results:
            logger.info(
                "MCP extraction for %s (%s): found %d URLs",
                bank.name,
                url,
                len(results),
            )
        else:
            logger.warning(
                "MCP extraction for %s: 0 URLs (raw: %r)",
                bank.name,
                raw[:300],
            )
        return results
    except Exception as exc:
        logger.warning("MCP extraction failed for %s: %s", bank.name, exc)
        return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_snapshot_for_agent(snapshot: PageSnapshot) -> str:
    """Format a page snapshot as a concise string for the LLM agent."""
    lines = [f"Page: {snapshot.title}", f"URL: {snapshot.url}", "", "Links:"]
    for link in snapshot.links:
        lines.append(f"  {link.text} → {link.url}")
    return "\n".join(lines)


def _format_links_text(snapshot: PageSnapshot) -> str:
    """Format links for the pagination prompt."""
    lines = []
    for link in snapshot.links:
        lines.append(f"{link.text} → {link.url}")
    return "\n".join(lines)


def _base_domain(netloc: str) -> str:
    """Strip common prefixes (www.) and return the base domain for comparison."""
    netloc = netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc


def _filter_off_domain(
    press_releases: list[DiscoveredPressRelease],
    bank_domain: str,
) -> list[DiscoveredPressRelease]:
    """Remove URLs that point outside the bank's domain."""
    base = _base_domain(bank_domain)
    filtered = []
    for pr in press_releases:
        pr_domain = urlparse(pr.url).netloc.lower()
        pr_base = _base_domain(pr_domain)
        if base in pr_base or pr_base in base:
            filtered.append(pr)
        else:
            logger.info("Filtering off-domain URL: %s", pr.url)
    return filtered


def _log_step(step: NavigationStep, snapshot: PageSnapshot | None = None) -> None:
    """Emit a log message for a navigation step."""
    logger.info(
        "Step %d [%s]: %s — %s (%d links)",
        step.step_number,
        step.action,
        step.url,
        step.reasoning,
        step.links_found,
    )
    if snapshot is not None and step.links_found == 0:
        logger.warning(
            "Step %d: page title=%r url=%r — zero links detected, possible bot block",
            step.step_number,
            snapshot.title,
            snapshot.url,
        )


# ---------------------------------------------------------------------------
# Pagination (uses focused LLM call, not full agent)
# ---------------------------------------------------------------------------


def _find_next_page_ref(
    llm: BaseChatModel,
    snapshot: PageSnapshot,
) -> str | None:
    """Ask the LLM to identify a pagination link on the current page.

    Returns the URL string if found, or None.
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
            logger.warning("Pagination click failed for %s — stopping", next_ref)
            break

        step = NavigationStep(
            step_number=len(steps) + 1,
            action="paginate",
            url=current_snapshot.url,
            reasoning=f"Navigated to pagination URL: {next_ref}",
            links_found=len(current_snapshot.links),
        )
        steps.append(step)
        _log_step(step)

        # Extract URLs from the new page's HTML
        html = browser.get_page_html()
        new_prs = _extract_urls_from_html(
            html,
            llm,
            bank_name=bank.name,
            page_url=current_snapshot.url,
        )
        if not new_prs:
            # Fallback: use snapshot links if HTML extraction found nothing
            new_prs = [
                DiscoveredPressRelease(url=link.url, title=link.text)
                for link in current_snapshot.links
            ]
        all_releases.extend(new_prs)

    return all_releases


# ---------------------------------------------------------------------------
# Tool execution (manual ReAct loop — discovery mode)
# ---------------------------------------------------------------------------

_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "click_link",
            "description": (
                "Navigate to a link on the current page by providing its URL. "
                "Use the exact URL shown next to the link in the page snapshot."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "element_ref": {
                        "type": "string",
                        "description": "The absolute URL of the link to navigate to.",
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
                "Signal that you have reached the press releases listing page "
                "and the system should extract all press release URLs."
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
        url = tool_args.get("element_ref", "")
        snapshot = browser.click(url, timeout=timeout)
        step = NavigationStep(
            step_number=len(steps) + 1,
            action="click",
            url=snapshot.url,
            reasoning=f"Navigated to {url}",
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

    try:
        bound_llm = llm.bind_tools(_TOOL_SCHEMAS)
    except NotImplementedError:
        bound_llm = llm

    for _ in range(_MAX_AGENT_STEPS):
        response = bound_llm.invoke(messages)
        messages.append(response)

        if not isinstance(response, AIMessage) or not response.tool_calls:
            break

        for tc in response.tool_calls:
            result_str = _execute_tool_call(
                tc["name"],
                tc.get("args", {}),
                browser,
                bank.page_load_timeout,
                steps,
            )
            messages.append(ToolMessage(content=result_str, tool_call_id=tc["id"]))

    # Extract from the final page using HTML → Claude
    html = browser.get_page_html()
    final_snapshot = browser.get_snapshot()
    prs = _extract_urls_from_html(
        html,
        llm,
        bank_name=bank.name,
        page_url=final_snapshot.url,
    )
    if not prs:
        # Fallback: use snapshot links
        prs = [
            DiscoveredPressRelease(url=link.url, title=link.text)
            for link in final_snapshot.links
        ]
    return prs


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

    Three strategies, tried in order:

    1. **Playwright MCP** (preferred when LLM is ``ClaudeCodeChatModel``):
       Claude controls a headless browser directly via MCP tools —
       navigates, reads the accessibility tree (~2-5 KB), paginates, and
       returns URLs.  No HTML parsing needed.  Falls back to (2) on failure.
    2. **Direct URL + HTML extraction**: Navigate to ``press_releases_url``,
       get the rendered HTML, clean it, and ask the LLM to extract URLs.
    3. **Discovery agent**: Navigate from the homepage using a ReAct loop
       to find the press releases section, then extract URLs.

    Args:
        bank: Configuration for the target central bank.
        browser: Playwright browser adapter (used for strategies 2/3 and for
            individual press release fetching by the bank processor).
        llm: LangChain chat model for agent reasoning and URL extraction.
        max_pages: Maximum number of listing pages to visit.

    Returns:
        NavigationResult with discovered press release URLs and step log.

    Raises:
        NavigationError: If the agent cannot find the press releases section.
    """
    steps: list[NavigationStep] = []

    # ── Strategy 1: Playwright MCP (preferred) ─────────────────────────────
    # Claude drives a separate headless browser via MCP tools.  The
    # accessibility tree is ~2-5 KB (vs. 150 KB of cleaned HTML), which
    # eliminates [Errno 7] and reduces token cost.
    if _has_mcp_support(llm):
        is_direct = bool(bank.press_releases_url)
        url = str(bank.press_releases_url or bank.homepage_url)

        press_releases = _extract_urls_via_mcp(
            llm,  # type: ignore[arg-type]
            url,
            bank,
            max_pages,
            is_direct_url=is_direct,
        )

        if press_releases:
            step = NavigationStep(
                step_number=1,
                action="mcp_extract",
                url=url,
                reasoning="Playwright MCP browser navigation + extraction",
                links_found=len(press_releases),
            )
            steps.append(step)
            _log_step(step)

            bank_domain = urlparse(url).netloc.lower()
            press_releases = _filter_off_domain(press_releases, bank_domain)

            return NavigationResult(
                bank_name=bank.name,
                press_releases=press_releases,
                navigation_steps=steps,
                listing_page_url=url if is_direct else None,
                pages_visited=1,
                used_direct_url=is_direct,
            )

        logger.warning(
            "MCP extraction returned 0 URLs for %s — falling back to HTML",
            bank.name,
        )

    # ── Strategy 2: Direct URL + HTML extraction ───────────────────────────
    if bank.press_releases_url:
        snapshot = browser.navigate(
            str(bank.press_releases_url),
            timeout=bank.page_load_timeout,
            wait_strategy=bank.wait_strategy,
            wait_for_selector=bank.wait_for_selector,
        )

        step = NavigationStep(
            step_number=len(steps) + 1,
            action="direct_url",
            url=str(bank.press_releases_url),
            reasoning="Bank config has press_releases_url configured",
            links_found=len(snapshot.links),
        )
        steps.append(step)
        _log_step(step, snapshot)

        # Extract press release URLs from the fully-rendered HTML
        html = browser.get_page_html()
        press_releases = _extract_urls_from_html(
            html,
            llm,
            bank_name=bank.name,
            page_url=str(bank.press_releases_url),
        )
        if not press_releases:
            # Fallback: use snapshot links (covers non-JS sites)
            press_releases = [
                DiscoveredPressRelease(url=link.url, title=link.text)
                for link in snapshot.links
            ]

        # Paginate for more releases
        if max_pages > 1:
            press_releases += _paginate(
                browser, llm, bank, snapshot, max_pages - 1, steps
            )

        bank_domain = urlparse(str(bank.press_releases_url)).netloc.lower()
        press_releases = _filter_off_domain(press_releases, bank_domain)

        return NavigationResult(
            bank_name=bank.name,
            press_releases=press_releases,
            navigation_steps=steps,
            listing_page_url=str(bank.press_releases_url),
            pages_visited=len(steps),
            used_direct_url=True,
        )

    # ── Strategy 3: LLM agent discovery ────────────────────────────────────
    press_releases = _run_discovery_agent(bank, browser, llm, steps)

    if max_pages > 1:
        current_snapshot = browser.get_snapshot()
        press_releases += _paginate(
            browser, llm, bank, current_snapshot, max_pages - 1, steps
        )

    bank_domain = urlparse(str(bank.homepage_url)).netloc.lower()
    press_releases = _filter_off_domain(press_releases, bank_domain)

    return NavigationResult(
        bank_name=bank.name,
        press_releases=press_releases,
        navigation_steps=steps,
        listing_page_url=None,
        pages_visited=len(steps),
        used_direct_url=False,
    )
