"""Translation pipeline — detect language and translate to English via LLM.

Slice 1.4 (FR-003).
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel


class TranslationResult(BaseModel):
    """Result of translating a press release body."""

    body_en: str
    original_language: str
    was_translated: bool


def detect_language(llm: BaseChatModel, text: str) -> str:
    """Detect the language of *text* and return its ISO 639-1 code.

    Uses the LLM to classify the language.  The returned value is a
    lower-case two-letter code (e.g. ``"en"``, ``"es"``, ``"ja"``).
    """
    messages = [
        SystemMessage(
            content=(
                "You are a language detection assistant. "
                "Given the following text, respond with ONLY the ISO 639-1 "
                "two-letter language code (e.g. 'en', 'es', 'ja', 'fr', 'zh'). "
                "Do not include any other text."
            )
        ),
        HumanMessage(content=text),
    ]
    response = llm.invoke(messages)
    return str(response.content).strip().lower()


def translate_text(
    llm: BaseChatModel,
    text: str,
    *,
    original_language: str,
) -> TranslationResult:
    """Translate *text* to English if it is not already English.

    If *original_language* is ``"en"``, the text is returned as-is without
    calling the LLM.  Otherwise the LLM translates the text and the result
    is wrapped in a :class:`TranslationResult`.
    """
    if original_language == "en":
        return TranslationResult(
            body_en=text,
            original_language="en",
            was_translated=False,
        )

    messages = [
        SystemMessage(
            content=(
                "You are a professional translator. "
                "Translate the following text to English. "
                "Preserve the original meaning and tone. "
                "Return ONLY the translated text, nothing else."
            )
        ),
        HumanMessage(content=text),
    ]
    response = llm.invoke(messages)
    translated = str(response.content).strip()

    return TranslationResult(
        body_en=translated,
        original_language=original_language,
        was_translated=True,
    )
