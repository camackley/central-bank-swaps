"""LLM provider abstraction — thin wrapper around LangChain's init_chat_model."""

from __future__ import annotations

import os

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel

PROVIDER_ENV_VARS: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google-genai": "GOOGLE_API_KEY",
}


class LLMConfigError(Exception):
    """Raised when LLM configuration is invalid or incomplete."""


def get_llm(provider: str, model: str) -> BaseChatModel:
    """Return a LangChain BaseChatModel for the given provider and model.

    Raises:
        LLMConfigError: If the required API key environment variable is not set.
    """
    env_var = PROVIDER_ENV_VARS.get(provider)
    if env_var and not os.environ.get(env_var):
        msg = (
            f"Missing API key: set {env_var} environment variable "
            f"for provider '{provider}'"
        )
        raise LLMConfigError(msg)

    return init_chat_model(model, model_provider=provider)
