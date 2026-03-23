"""Tests for worker_factory — per-worker resource creation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from cbs.pipeline.worker_factory import WorkerResources, create_worker


class TestCreateWorker:
    @patch("cbs.pipeline.worker_factory.duckdb")
    @patch("cbs.pipeline.worker_factory.init_db")
    @patch("cbs.pipeline.worker_factory.BrowserAdapter")
    @patch("cbs.pipeline.worker_factory.Orchestrator")
    @patch("cbs.pipeline.worker_factory.DefaultBankProcessor")
    def test_creates_all_resources(
        self,
        mock_processor_cls: MagicMock,
        mock_orchestrator_cls: MagicMock,
        mock_browser_cls: MagicMock,
        mock_init_db: MagicMock,
        mock_duckdb: MagicMock,
    ) -> None:
        mock_llm = MagicMock()
        mock_conn = MagicMock()
        mock_duckdb.connect.return_value = mock_conn

        worker = create_worker(
            db_path="test.duckdb",
            instance_port=9868,
            llm=mock_llm,
            max_pages=3,
        )

        mock_duckdb.connect.assert_called_once_with("test.duckdb")
        mock_init_db.assert_called_once_with(mock_conn)
        mock_browser_cls.assert_called_once_with()
        mock_orchestrator_cls.assert_called_once()
        mock_processor_cls.assert_called_once()

        assert isinstance(worker, WorkerResources)

    @patch("cbs.pipeline.worker_factory.duckdb")
    @patch("cbs.pipeline.worker_factory.init_db")
    @patch("cbs.pipeline.worker_factory.BrowserAdapter")
    @patch("cbs.pipeline.worker_factory.Orchestrator")
    @patch("cbs.pipeline.worker_factory.DefaultBankProcessor")
    def test_passes_llm_overrides(
        self,
        mock_processor_cls: MagicMock,
        mock_orchestrator_cls: MagicMock,
        mock_browser_cls: MagicMock,
        mock_init_db: MagicMock,
        mock_duckdb: MagicMock,
    ) -> None:
        mock_llm = MagicMock()
        mock_classify = MagicMock()
        mock_extract = MagicMock()
        mock_duckdb.connect.return_value = MagicMock()

        create_worker(
            db_path="test.duckdb",
            instance_port=9870,
            llm=mock_llm,
            classify_llm=mock_classify,
            extract_llm=mock_extract,
        )

        call_kwargs = mock_orchestrator_cls.call_args
        assert call_kwargs.kwargs["classify_llm"] is mock_classify
        assert call_kwargs.kwargs["extract_llm"] is mock_extract


class TestWorkerResourcesClose:
    def test_close_releases_resources(self) -> None:
        mock_conn = MagicMock()
        mock_browser = MagicMock()
        mock_orchestrator = MagicMock()
        mock_processor = MagicMock()

        worker = WorkerResources(
            conn=mock_conn,
            browser=mock_browser,
            orchestrator=mock_orchestrator,
            processor=mock_processor,
        )

        worker.close()

        mock_browser.close_session.assert_called_once()
        mock_conn.close.assert_called_once()
