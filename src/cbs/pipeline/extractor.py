"""Swap data extractor — LLM-based structured extraction from press releases (FR-005).

Given an English press-release body that has been classified as swap-related,
use a LangChain chat model with structured output to produce one or more
``SwapRecord`` objects.  Each bilateral agreement yields two ``SwapDirection``
rows (one per currency side); if only one side is stated, a single direction is
acceptable.
"""

from __future__ import annotations

from decimal import Decimal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Pydantic models — these double as the LangChain structured-output schema
# ---------------------------------------------------------------------------


class SwapDirection(BaseModel):
    """One directional leg of a swap agreement.

    ``provider`` supplies the ``currency`` to the ``receiver``.
    """

    provider_central_bank: str = Field(
        description="Central bank providing the currency."
    )
    provider_country: str = Field(description="Country of the provider central bank.")
    receiver_central_bank: str = Field(
        description="Central bank receiving the currency."
    )
    receiver_country: str = Field(description="Country of the receiver central bank.")
    currency: str = Field(
        description="ISO 4217 currency code being provided (e.g. USD, EUR)."
    )
    swap_amount: Decimal | None = Field(
        default=None,
        description="Notional amount in this currency. NULL if not stated.",
    )


class SwapRecord(BaseModel):
    """A single swap agreement extracted from a press release.

    A bilateral swap has two ``directions`` (one per currency side).  If only
    one side is mentioned in the text, a single direction is acceptable.
    """

    directions: list[SwapDirection] = Field(
        description=(
            "Directional legs of the swap. Bilateral swaps have two "
            "(one per currency side). Include only sides explicitly "
            "stated in the text."
        ),
    )
    swap_type: str = Field(
        description=(
            'Type of swap arrangement, e.g. "bilateral", "standing", '
            '"liquidity", "temporary".'
        ),
    )
    announcement_type: str = Field(
        description=('One of: "new", "extension", "renewal", "modification".'),
    )
    type_of_change: str | None = Field(
        default=None,
        description=(
            "If announcement_type is 'modification', describes what "
            "changed. NULL otherwise."
        ),
    )
    conditions: str | None = Field(
        default=None,
        description="Interest rate, timing conditions, and other terms mentioned.",
    )
    reasons_for_swap: str | None = Field(
        default=None,
        description="Justification or context for why the swap was established.",
    )
    announcement_date: str | None = Field(
        default=None,
        description="Date the swap was announced (ISO 8601, e.g. 2008-12-12).",
    )
    effective_date: str | None = Field(
        default=None,
        description="Date the swap becomes effective (ISO 8601). NULL if not stated.",
    )
    maturity_date: str | None = Field(
        default=None,
        description=(
            "Parsed expiration date (ISO 8601). NULL if not stated or ambiguous."
        ),
    )
    maturity_text: str | None = Field(
        default=None,
        description=(
            "Original verbatim phrasing about duration/expiration "
            '(e.g. "until further notice", "for three years").'
        ),
    )
    duration_description: str | None = Field(
        default=None,
        description='Human-readable duration (e.g. "6 months", "indefinite").',
    )
    raw_extract: str = Field(
        description=(
            "The exact paragraph(s) from the press release from which "
            "data was extracted. Copied verbatim for audit."
        ),
    )


class ExtractionResult(BaseModel):
    """Top-level extraction output.  May contain zero or more swap agreements."""

    swaps: list[SwapRecord] = Field(
        default_factory=list,
        description="Swap agreements found in the press release.",
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a structured-data extraction agent for central bank swap line \
press releases. Given the English text of a press release that has been \
classified as swap-related, extract every swap agreement mentioned.

Rules:
1. Each bilateral swap produces TWO directional legs — one per currency \
   side (provider → receiver). If only one side is explicitly stated, \
   include only that side.
2. One press release may announce multiple swap agreements.
3. Use ISO 4217 currency codes (USD, EUR, JPY, GBP, etc.).
4. Use ISO 8601 dates (YYYY-MM-DD).
5. For announcement_type use exactly one of: "new", "extension", \
   "renewal", "modification".
6. Copy maturity_text verbatim from the source text.
7. Copy into raw_extract the exact paragraph(s) from which you extracted \
   each swap's data.
8. If a field cannot be determined from the text, set it to null. \
   Do NOT hallucinate values.
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_swaps(
    llm: BaseChatModel,
    press_release_text: str,
) -> ExtractionResult:
    """Extract structured swap data from a classified press release.

    Parameters
    ----------
    llm:
        A LangChain ``BaseChatModel`` (obtained via ``get_llm``).
    press_release_text:
        The English body of a press release already classified as swap-related.

    Returns
    -------
    ExtractionResult
        Zero or more ``SwapRecord`` objects, each with one or two
        ``SwapDirection`` legs.
    """
    structured_llm = llm.with_structured_output(ExtractionResult)
    raw = structured_llm.invoke(
        [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=press_release_text),
        ]
    )
    # with_structured_output returns BaseModel | dict; we requested
    # ExtractionResult so the runtime type is always ExtractionResult.
    assert isinstance(raw, ExtractionResult)
    return raw
