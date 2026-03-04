"""Tests for the translation pipeline — Slice 1.4 (FR-003)."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from cbs.pipeline.translator import (
    TranslationResult,
    detect_language,
    translate_text,
)

# ---------------------------------------------------------------------------
# Fake LLM for deterministic unit tests
# ---------------------------------------------------------------------------


class _FakeLLM(BaseChatModel):
    """Minimal fake that returns a canned response for every invoke."""

    response: str = ""

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:  # ChatResult
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=self.response))]
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEnglishTextReturnsUnchanged:
    """English text should pass through without hitting the LLM."""

    def test_english_body_returned_as_is(self) -> None:
        llm = _FakeLLM(response="SHOULD NOT BE CALLED")
        english_body = "The Federal Reserve announced a new swap line with the ECB."

        result = translate_text(llm, english_body, original_language="en")

        assert result.body_en == english_body
        assert result.original_language == "en"
        assert result.was_translated is False


class TestSpanishTextTriggersTranslation:
    """Non-English text must be translated via the LLM."""

    def test_spanish_body_is_translated(self) -> None:
        translated = "The Central Bank of Colombia announced a swap agreement."
        llm = _FakeLLM(response=translated)
        spanish_body = "El Banco de la República anunció un acuerdo de canje."

        result = translate_text(llm, spanish_body, original_language="es")

        assert result.body_en == translated
        assert result.original_language == "es"
        assert result.was_translated is True

    def test_japanese_body_is_translated(self) -> None:
        translated = "The Bank of Japan announced a swap line."
        llm = _FakeLLM(response=translated)
        japanese_body = "日本銀行はスワップラインを発表しました。"

        result = translate_text(llm, japanese_body, original_language="ja")

        assert result.body_en == translated
        assert result.original_language == "ja"
        assert result.was_translated is True


class TestLanguageDetectionReturnsIsoCode:
    """detect_language() must return an ISO 639-1 code."""

    def test_english_detected(self) -> None:
        llm = _FakeLLM(response="en")
        code = detect_language(
            llm,
            "The Federal Reserve announced a new bilateral swap line.",
        )
        assert code == "en"

    def test_spanish_detected(self) -> None:
        llm = _FakeLLM(response="es")
        code = detect_language(
            llm,
            "El Banco Central anunció una nueva línea de canje.",
        )
        assert code == "es"

    def test_whitespace_stripped(self) -> None:
        llm = _FakeLLM(response="  fr  \n")
        code = detect_language(llm, "La Banque de France a annoncé...")
        assert code == "fr"


class TestTranslationResultStoredInBodyEn:
    """TranslationResult must carry enough info to persist to press_releases.body_en."""

    def test_result_fields_for_english(self) -> None:
        llm = _FakeLLM(response="unused")
        body = "Some English text."
        result = translate_text(llm, body, original_language="en")

        assert isinstance(result, TranslationResult)
        assert result.body_en == body
        assert result.original_language == "en"
        assert result.was_translated is False

    def test_result_fields_for_translated(self) -> None:
        translated = "Translated text."
        llm = _FakeLLM(response=translated)
        result = translate_text(llm, "Texto original.", original_language="es")

        assert isinstance(result, TranslationResult)
        assert result.body_en == translated
        assert result.original_language == "es"
        assert result.was_translated is True
