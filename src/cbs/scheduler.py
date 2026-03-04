"""Pipeline scheduler — automated periodic runs (Slice 2.2)."""

from __future__ import annotations

import logging
from collections.abc import Callable

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger

from cbs.pipeline.models import RunSummary

logger = logging.getLogger(__name__)


class PipelineScheduler:
    """Run the pipeline on a recurring interval.

    Wraps APScheduler's ``BlockingScheduler`` to execute a pipeline run
    function at a configurable interval (default: weekly).

    Usage::

        scheduler = PipelineScheduler(run_fn=my_incremental_run)
        scheduler.start()  # blocks forever, runs weekly
    """

    def __init__(
        self,
        run_fn: Callable[[], RunSummary],
        interval_days: int = 7,
    ) -> None:
        self._run_fn = run_fn
        self._interval_days = interval_days
        self._scheduler = BlockingScheduler()

    def start(self) -> None:
        """Start the scheduler. Blocks until interrupted."""
        self._scheduler.add_job(
            self._execute,
            IntervalTrigger(days=self._interval_days),
            id="incremental_run",
        )
        logger.info("Scheduler started — running every %d days", self._interval_days)
        self._scheduler.start()

    def _execute(self) -> None:
        """Execute a single scheduled run."""
        logger.info("Scheduled run starting")
        summary = self._run_fn()
        logger.info(
            "Scheduled run complete: %d/%d banks, %d PRs, %d swaps",
            summary.banks_succeeded,
            summary.banks_attempted,
            summary.press_releases_found,
            summary.swaps_extracted,
        )
