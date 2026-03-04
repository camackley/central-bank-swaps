"""Tests for CLI entry point — Slice 1.13."""

from __future__ import annotations

import uuid

from cbs.pipeline.__main__ import build_parser


class TestDefaultArgsParse:
    def test_default_args(self) -> None:
        parser = build_parser()
        args = parser.parse_args([])

        assert args.config == "config/banks.yaml"
        assert args.db == "central_bank_swaps.duckdb"
        assert args.provider == "anthropic"
        assert args.model == "claude-sonnet-4-20250514"
        assert args.max_pages == 5
        assert args.resume is None


class TestResumeFlag:
    def test_resume_flag_accepted(self) -> None:
        run_id = str(uuid.uuid4())
        parser = build_parser()
        args = parser.parse_args(["--resume", run_id])

        assert args.resume == run_id


class TestModuleImportable:
    def test_module_importable(self) -> None:
        import cbs.pipeline.__main__  # noqa: F401
