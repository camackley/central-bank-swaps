"""Tests for the LLM provider abstraction — Slice 0.4."""

from __future__ import annotations

import pytest
from langchain_core.language_models.chat_models import BaseChatModel

from cbs.llm import LLMConfigError, get_llm
from cbs.llm.claude_code_model import ClaudeCodeChatModel


class TestGetLlmReturnsBaseChatModel:
    """get_llm() must return a LangChain BaseChatModel instance."""

    def test_anthropic_returns_base_chat_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
        llm = get_llm("anthropic", "claude-sonnet-4-20250514")
        assert isinstance(llm, BaseChatModel)

    def test_openai_returns_base_chat_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key-for-test")
        llm = get_llm("openai", "gpt-4o-mini")
        assert isinstance(llm, BaseChatModel)


class TestConfigChangeSwitchesProvider:
    """Changing the provider argument must yield a different model class."""

    def test_different_providers_return_different_types(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key-for-test")

        anthropic_llm = get_llm("anthropic", "claude-sonnet-4-20250514")
        openai_llm = get_llm("openai", "gpt-4o-mini")

        assert type(anthropic_llm) is not type(openai_llm)


class TestMissingApiKeyRaisesClearError:
    """Missing API key must raise LLMConfigError with a helpful message."""

    def test_missing_anthropic_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(LLMConfigError, match="ANTHROPIC_API_KEY"):
            get_llm("anthropic", "claude-sonnet-4-20250514")

    def test_missing_openai_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(LLMConfigError, match="OPENAI_API_KEY"):
            get_llm("openai", "gpt-4o-mini")

    def test_missing_google_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        with pytest.raises(LLMConfigError, match="GOOGLE_API_KEY"):
            get_llm("google-genai", "gemini-2.0-flash")


# ---------------------------------------------------------------------------
# TestClaudeCodeProvider
# ---------------------------------------------------------------------------


class TestClaudeCodeProvider:
    """get_llm('claude-code', ...) returns a ClaudeCodeChatModel."""

    def test_get_llm_claude_code_returns_claude_code_model(self) -> None:
        llm = get_llm("claude-code", "claude-sonnet-4-6")
        assert isinstance(llm, ClaudeCodeChatModel)

    def test_get_llm_claude_code_is_base_chat_model(self) -> None:
        llm = get_llm("claude-code", "claude-sonnet-4-6")
        assert isinstance(llm, BaseChatModel)

    def test_get_llm_claude_code_uses_given_model_name(self) -> None:
        llm = get_llm("claude-code", "claude-haiku-4-5")
        assert isinstance(llm, ClaudeCodeChatModel)
        assert llm.model_name == "claude-haiku-4-5"

    def test_get_llm_claude_code_requires_no_env_vars(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        llm = get_llm("claude-code", "claude-sonnet-4-6")
        assert isinstance(llm, ClaudeCodeChatModel)
