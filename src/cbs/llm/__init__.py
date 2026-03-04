"""LLM provider abstraction — swappable LangChain chat models."""

from cbs.llm.provider import LLMConfigError, get_llm

__all__ = ["LLMConfigError", "get_llm"]
