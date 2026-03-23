"""Per-worker resource creation for parallel bank processing."""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
from langchain_core.language_models.chat_models import BaseChatModel

from cbs.db.schema import init_db
from cbs.pipeline.bank_processor import DefaultBankProcessor
from cbs.pipeline.orchestrator import Orchestrator
from cbs.scraper.browser import BrowserAdapter


@dataclass
class WorkerResources:
    """Bundle of per-worker resources for parallel processing."""

    conn: duckdb.DuckDBPyConnection
    browser: BrowserAdapter
    orchestrator: Orchestrator
    processor: DefaultBankProcessor

    def close(self) -> None:
        """Release worker resources."""
        self.browser.close_session()
        self.conn.close()


def create_worker(
    db_path: str,
    instance_port: int,
    llm: BaseChatModel,
    *,
    classify_llm: BaseChatModel | None = None,
    extract_llm: BaseChatModel | None = None,
    translate_llm: BaseChatModel | None = None,
    max_pages: int = 5,
) -> WorkerResources:
    """Create a complete set of worker resources for one parallel worker.

    Each worker gets its own DuckDB connection (to the run's DB file) and
    BrowserAdapter (Playwright-backed, headless Chromium).
    LLMs are shared across workers (they are thread-safe).
    ``instance_port`` is kept for API compatibility but is no longer used
    (Playwright manages its own browser process).
    """
    conn = duckdb.connect(db_path)
    init_db(conn)

    browser = BrowserAdapter()

    orchestrator = Orchestrator(
        conn=conn,
        llm=llm,
        browser=browser,
        classify_llm=classify_llm,
        extract_llm=extract_llm,
        translate_llm=translate_llm,
    )

    processor = DefaultBankProcessor(
        orchestrator=orchestrator,
        browser=browser,
        llm=llm,
        max_pages=max_pages,
    )

    return WorkerResources(
        conn=conn,
        browser=browser,
        orchestrator=orchestrator,
        processor=processor,
    )
