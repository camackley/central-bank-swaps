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
        assert args.model == "claude-sonnet-4-6"
        assert args.max_pages == 5
        assert args.banks is None
        assert args.resume is None
        assert args.mode == "backfill"


class TestModeFlag:
    def test_incremental_mode(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--mode", "incremental"])
        assert args.mode == "incremental"

    def test_backfill_mode_explicit(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--mode", "backfill"])
        assert args.mode == "backfill"


class TestResumeFlag:
    def test_resume_flag_accepted(self) -> None:
        run_id = str(uuid.uuid4())
        parser = build_parser()
        args = parser.parse_args(["--resume", run_id])

        assert args.resume == run_id


class TestModelTieringFlags:
    def test_classify_model_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--classify-model", "claude-haiku-3-5"])
        assert args.classify_model == "claude-haiku-3-5"

    def test_extract_model_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--extract-model", "claude-sonnet-4-20250514"])
        assert args.extract_model == "claude-sonnet-4-20250514"

    def test_model_flags_default_to_none(self) -> None:
        parser = build_parser()
        args = parser.parse_args([])
        assert args.classify_model is None
        assert args.extract_model is None


class TestBanksFilter:
    def test_banks_flag_accepted(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--banks", "banrep", "australia"])
        assert args.banks == ["banrep", "australia"]

    def test_banks_multiple_values(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--banks", "japan", "china", "fed"])
        assert args.banks == ["japan", "china", "fed"]

    def test_banks_default_is_none(self) -> None:
        parser = build_parser()
        args = parser.parse_args([])
        assert args.banks is None


class TestModuleImportable:
    def test_module_importable(self) -> None:
        import cbs.pipeline.__main__  # noqa: F401
