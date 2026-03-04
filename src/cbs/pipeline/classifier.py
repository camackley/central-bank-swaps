"""Swap classifier — determines if a press release is about central bank swap lines."""

from __future__ import annotations

from typing import cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

SYSTEM_PROMPT = (
    "You are an expert analyst specializing in central bank swap line agreements. "
    "Your task is to determine whether a press release is primarily about a central "
    "bank swap line or swap arrangement.\n\n"
    "A press release IS swap-related if its primary topic is:\n"
    "- Establishing a new currency swap line between central banks\n"
    "- Extending, renewing, or modifying an existing swap arrangement\n"
    "- Activating or drawing on an existing swap facility\n\n"
    "A press release is NOT swap-related if:\n"
    "- It merely mentions swap lines in passing while discussing other topics\n"
    "- Its primary topic is interest rate decisions, monetary policy, "
    "economic outlook, or other central bank activities\n"
    "- Swap lines are referenced only as background context\n\n"
    "Respond with a boolean classification and a brief reason."
)

HUMAN_PROMPT = (
    "Classify the following press release. Is its primary topic a central bank "
    "swap line agreement?\n\n"
    "---\n{text}\n---"
)

_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT),
    ]
)


class ClassificationResult(BaseModel):
    """Output schema for the swap classifier."""

    is_swap_related: bool = Field(
        description=(
            "True if the press release is primarily about "
            "a central bank swap agreement."
        ),
    )
    reason: str = Field(
        description=(
            "Brief explanation of why the press release was classified this way."
        ),
    )


def classify_press_release(
    llm: BaseChatModel,
    text: str,
) -> ClassificationResult:
    """Classify a press release as swap-related or not.

    Args:
        llm: A LangChain BaseChatModel instance (provider-agnostic).
        text: The English body text of the press release to classify.

    Returns:
        ClassificationResult with is_swap_related bool and reason string.
    """
    structured_llm = llm.with_structured_output(ClassificationResult)
    messages = _PROMPT.format_messages(text=text)
    result = structured_llm.invoke(messages)
    return cast(ClassificationResult, result)
