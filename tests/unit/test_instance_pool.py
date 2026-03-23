"""Tests for PinchTabInstancePool — multi-instance lifecycle."""

from __future__ import annotations

import httpx
import pytest

from cbs.scraper.instance_pool import PinchTabInstancePool


class FakeTransport(httpx.BaseTransport):
    """Mock transport that tracks requests and returns preconfigured responses.

    Supports multiple responses for the same method+URL (queued in order).
    """

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self._responses: dict[str, list[httpx.Response]] = {}

    def add_response(self, method: str, url: str, response: httpx.Response) -> None:
        key = f"{method.upper()} {url}"
        self._responses.setdefault(key, []).append(response)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        key = f"{request.method} {request.url}"
        if key in self._responses and self._responses[key]:
            return self._responses[key].pop(0)
        return httpx.Response(404, text="Not found")


def _make_launch_response(name: str, port: int) -> httpx.Response:
    return httpx.Response(
        200,
        json={"id": f"{name}-{port}", "port": str(port), "status": "starting"},
    )


def _make_health_response() -> httpx.Response:
    return httpx.Response(200, json={"status": "ok"})


def _make_stop_response(instance_id: str) -> httpx.Response:
    return httpx.Response(200, json={"id": instance_id, "status": "stopped"})


class TestInstancePoolLaunch:
    def test_launches_correct_number_of_instances(self) -> None:
        transport = FakeTransport()
        for i in range(3):
            port = 9868 + i
            transport.add_response(
                "POST",
                "http://localhost:9867/instances/launch",
                _make_launch_response(f"cbs-worker-{i}", port),
            )
            transport.add_response(
                "GET",
                f"http://localhost:{port}/health",
                _make_health_response(),
            )
            transport.add_response(
                "POST",
                f"http://localhost:9867/instances/cbs-worker-{i}-{port}/stop",
                _make_stop_response(f"cbs-worker-{i}-{port}"),
            )

        client = httpx.Client(transport=transport)
        with PinchTabInstancePool(size=3, _http_client=client) as pool:
            assert len(pool.instances) == 3
            assert pool.instances[0].port == 9868
            assert pool.instances[1].port == 9869
            assert pool.instances[2].port == 9870

    def test_instance_base_urls(self) -> None:
        transport = FakeTransport()
        for i in range(2):
            port = 9868 + i
            transport.add_response(
                "POST",
                "http://localhost:9867/instances/launch",
                _make_launch_response(f"cbs-worker-{i}", port),
            )
            transport.add_response(
                "GET",
                f"http://localhost:{port}/health",
                _make_health_response(),
            )
            transport.add_response(
                "POST",
                f"http://localhost:9867/instances/cbs-worker-{i}-{port}/stop",
                _make_stop_response(f"cbs-worker-{i}-{port}"),
            )

        client = httpx.Client(transport=transport)
        with PinchTabInstancePool(size=2, _http_client=client) as pool:
            assert pool.instances[0].base_url == "http://localhost:9868"
            assert pool.instances[1].base_url == "http://localhost:9869"


class TestInstancePoolCleanup:
    def test_stop_called_on_exit(self) -> None:
        transport = FakeTransport()
        port = 9868
        transport.add_response(
            "POST",
            "http://localhost:9867/instances/launch",
            _make_launch_response("cbs-worker-0", port),
        )
        transport.add_response(
            "GET",
            f"http://localhost:{port}/health",
            _make_health_response(),
        )
        transport.add_response(
            "POST",
            f"http://localhost:9867/instances/cbs-worker-0-{port}/stop",
            _make_stop_response(f"cbs-worker-0-{port}"),
        )

        client = httpx.Client(transport=transport)
        with PinchTabInstancePool(size=1, _http_client=client) as pool:
            assert len(pool.instances) == 1

        # After exit, instances should be cleared
        assert len(pool.instances) == 0

        # Verify stop was called
        stop_requests = [r for r in transport.requests if "/stop" in str(r.url)]
        assert len(stop_requests) == 1

    def test_stop_errors_are_swallowed(self) -> None:
        transport = FakeTransport()
        port = 9868
        transport.add_response(
            "POST",
            "http://localhost:9867/instances/launch",
            _make_launch_response("cbs-worker-0", port),
        )
        transport.add_response(
            "GET",
            f"http://localhost:{port}/health",
            _make_health_response(),
        )
        # No stop response registered — will 404, but should not raise

        client = httpx.Client(transport=transport)
        with PinchTabInstancePool(size=1, _http_client=client):
            pass  # Should not raise on exit


class TestInstancePoolHealthCheck:
    def test_raises_on_health_timeout(self) -> None:
        transport = FakeTransport()
        port = 9868
        transport.add_response(
            "POST",
            "http://localhost:9867/instances/launch",
            _make_launch_response("cbs-worker-0", port),
        )
        # No health response — will always 404

        client = httpx.Client(transport=transport)
        pool = PinchTabInstancePool(size=1, _http_client=client)
        with pytest.raises(TimeoutError, match="did not become healthy"):
            pool.__enter__()
