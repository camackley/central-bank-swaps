"""LangSmith tracing configuration — opt-in via environment variables (FR-009)."""

from __future__ import annotations

import os

from pydantic import BaseModel

DEFAULT_PROJECT = "central-bank-swaps"


class TracingConfig(BaseModel):
    """Resolved LangSmith tracing configuration."""

    enabled: bool = False
    api_key: str | None = None
    project: str = DEFAULT_PROJECT

    @classmethod
    def from_env(cls) -> TracingConfig:
        """Build tracing config from environment variables.

        Tracing is enabled only when *both* ``LANGCHAIN_TRACING_V2`` is
        ``"true"`` (case-insensitive) **and** ``LANGCHAIN_API_KEY`` is set.
        This keeps tracing fully opt-in — tests and local dev never crash
        due to missing keys.
        """
        tracing_flag = os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true"
        api_key = os.environ.get("LANGCHAIN_API_KEY") or None
        project = os.environ.get("LANGCHAIN_PROJECT", DEFAULT_PROJECT)

        enabled = tracing_flag and api_key is not None

        return cls(enabled=enabled, api_key=api_key, project=project)


def configure_tracing() -> TracingConfig:
    """Read env vars, materialise the canonical env state, and return config.

    When tracing is enabled, ensures ``LANGCHAIN_TRACING_V2``,
    ``LANGCHAIN_API_KEY``, and ``LANGCHAIN_PROJECT`` are set in
    ``os.environ`` so that LangChain picks them up automatically.

    When tracing is disabled, explicitly sets ``LANGCHAIN_TRACING_V2=false``
    so that LangChain never attempts to phone home.
    """
    config = TracingConfig.from_env()

    if config.enabled:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = config.api_key  # type: ignore[assignment]
        os.environ["LANGCHAIN_PROJECT"] = config.project
    else:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"

    return config
