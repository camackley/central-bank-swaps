"""Tests for LangSmith tracing configuration (Slice 0.5 — FR-009)."""

from __future__ import annotations

import os
from unittest.mock import patch

from cbs.config.tracing import TracingConfig, configure_tracing


class TestTracingConfigReadsEnvVars:
    """Test that TracingConfig correctly reads LangSmith env vars."""

    def test_enabled_when_all_vars_present(self) -> None:
        env = {
            "LANGCHAIN_TRACING_V2": "true",
            "LANGCHAIN_API_KEY": "lsv2_pt_fake_key_123",
            "LANGCHAIN_PROJECT": "central-bank-swaps",
        }
        with patch.dict(os.environ, env, clear=False):
            config = TracingConfig.from_env()

        assert config.enabled is True
        assert config.api_key == "lsv2_pt_fake_key_123"
        assert config.project == "central-bank-swaps"

    def test_reads_tracing_v2_case_insensitive(self) -> None:
        env = {
            "LANGCHAIN_TRACING_V2": "True",
            "LANGCHAIN_API_KEY": "lsv2_pt_key",
            "LANGCHAIN_PROJECT": "my-project",
        }
        with patch.dict(os.environ, env, clear=False):
            config = TracingConfig.from_env()

        assert config.enabled is True

    def test_project_defaults_to_central_bank_swaps(self) -> None:
        env = {
            "LANGCHAIN_TRACING_V2": "true",
            "LANGCHAIN_API_KEY": "lsv2_pt_key",
        }
        with patch.dict(os.environ, env, clear=True):
            config = TracingConfig.from_env()

        assert config.project == "central-bank-swaps"

    def test_configure_tracing_sets_env_vars(self) -> None:
        env = {
            "LANGCHAIN_TRACING_V2": "true",
            "LANGCHAIN_API_KEY": "lsv2_pt_key",
            "LANGCHAIN_PROJECT": "test-project",
        }
        with patch.dict(os.environ, env, clear=True):
            config = configure_tracing()

            assert config.enabled is True
            assert os.environ.get("LANGCHAIN_TRACING_V2") == "true"
            assert os.environ.get("LANGCHAIN_API_KEY") == "lsv2_pt_key"
            assert os.environ.get("LANGCHAIN_PROJECT") == "test-project"


class TestTracingDisabledWhenEnvMissing:
    """Test that tracing is opt-in — no crash when keys are absent."""

    def test_disabled_when_no_env_vars(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            config = TracingConfig.from_env()

        assert config.enabled is False
        assert config.api_key is None
        assert config.project == "central-bank-swaps"

    def test_disabled_when_tracing_v2_false(self) -> None:
        env = {
            "LANGCHAIN_TRACING_V2": "false",
            "LANGCHAIN_API_KEY": "lsv2_pt_key",
        }
        with patch.dict(os.environ, env, clear=True):
            config = TracingConfig.from_env()

        assert config.enabled is False

    def test_disabled_when_api_key_missing(self) -> None:
        env = {"LANGCHAIN_TRACING_V2": "true"}
        with patch.dict(os.environ, env, clear=True):
            config = TracingConfig.from_env()

        assert config.enabled is False

    def test_configure_tracing_disables_when_env_missing(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            config = configure_tracing()

            assert config.enabled is False
            assert os.environ.get("LANGCHAIN_TRACING_V2") == "false"

    def test_configure_tracing_no_crash_when_keys_absent(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            config = configure_tracing()

        assert config is not None
        assert config.enabled is False
