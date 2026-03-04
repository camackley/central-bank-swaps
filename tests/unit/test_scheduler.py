"""Tests for PipelineScheduler — Slice 2.2."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from cbs.pipeline.models import RunSummary
from cbs.scheduler import PipelineScheduler


class TestSchedulerTriggersAtInterval:
    @patch("cbs.scheduler.BlockingScheduler")
    def test_adds_job_with_correct_interval(
        self, mock_scheduler_cls: MagicMock
    ) -> None:
        mock_scheduler = mock_scheduler_cls.return_value
        run_fn = MagicMock(return_value=RunSummary())

        scheduler = PipelineScheduler(run_fn=run_fn, interval_days=7)
        scheduler._scheduler = mock_scheduler
        scheduler.start()

        mock_scheduler.add_job.assert_called_once()
        call_args = mock_scheduler.add_job.call_args
        trigger = call_args[0][1]
        assert trigger.interval.days == 7

    @patch("cbs.scheduler.BlockingScheduler")
    def test_custom_interval(self, mock_scheduler_cls: MagicMock) -> None:
        mock_scheduler = mock_scheduler_cls.return_value
        run_fn = MagicMock(return_value=RunSummary())

        scheduler = PipelineScheduler(run_fn=run_fn, interval_days=14)
        scheduler._scheduler = mock_scheduler
        scheduler.start()

        trigger = mock_scheduler.add_job.call_args[0][1]
        assert trigger.interval.days == 14


class TestSchedulerLogsRunResult:
    def test_execute_calls_run_fn_and_logs(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        summary = RunSummary(
            banks_attempted=3,
            banks_succeeded=3,
            press_releases_found=15,
            swaps_extracted=8,
        )
        run_fn = MagicMock(return_value=summary)
        scheduler = PipelineScheduler(run_fn=run_fn)

        with caplog.at_level("INFO"):
            scheduler._execute()

        run_fn.assert_called_once()
        assert any("15 PRs" in msg for msg in caplog.messages)
        assert any("8 swaps" in msg for msg in caplog.messages)
