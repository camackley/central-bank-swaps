"""PinchTab multi-instance pool — manages Chrome instance lifecycle."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from types import TracebackType

import httpx

logger = logging.getLogger(__name__)

_HEALTH_RETRIES = 10
_HEALTH_DELAY = 0.5


@dataclass(frozen=True)
class InstanceInfo:
    """Metadata for a launched PinchTab Chrome instance."""

    id: str
    port: int
    base_url: str


class PinchTabInstancePool:
    """Context manager that creates/destroys N PinchTab Chrome instances.

    Usage::

        with PinchTabInstancePool(size=3) as pool:
            for inst in pool.instances:
                browser = BrowserAdapter(base_url=inst.base_url)
                ...
    """

    def __init__(
        self,
        orchestrator_url: str = "http://localhost:9867",
        size: int = 1,
        *,
        _http_client: httpx.Client | None = None,
    ) -> None:
        self._orchestrator_url = orchestrator_url
        self._size = size
        self._http = _http_client or httpx.Client()
        self._instances: list[InstanceInfo] = []

    @property
    def instances(self) -> list[InstanceInfo]:
        return list(self._instances)

    def __enter__(self) -> PinchTabInstancePool:
        self._launch_instances()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._stop_instances()

    def _launch_instances(self) -> None:
        """Launch N Chrome instances via the PinchTab orchestrator."""
        base_port = 9868
        for i in range(self._size):
            port = base_port + i
            name = f"cbs-worker-{i}"
            resp = self._http.post(
                f"{self._orchestrator_url}/instances/launch",
                json={"name": name, "port": str(port), "mode": "headless"},
            )
            resp.raise_for_status()
            data = resp.json()
            instance_id = data["id"]
            instance_port = int(data["port"])
            info = InstanceInfo(
                id=instance_id,
                port=instance_port,
                base_url=f"http://localhost:{instance_port}",
            )
            self._instances.append(info)
            logger.info(
                "Launched PinchTab instance %s on port %d",
                instance_id,
                instance_port,
            )

        # Wait for all instances to become healthy
        for info in self._instances:
            self._wait_for_health(info)

    def _wait_for_health(self, info: InstanceInfo) -> None:
        """Poll /health on an instance port until it responds OK."""
        for _attempt in range(_HEALTH_RETRIES):
            try:
                resp = self._http.get(f"{info.base_url}/health", timeout=2)
                if resp.status_code == 200:
                    logger.debug("Instance %s healthy", info.id)
                    return
            except httpx.HTTPError:
                pass
            time.sleep(_HEALTH_DELAY)
        msg = f"Instance {info.id} on port {info.port} did not become healthy"
        raise TimeoutError(msg)

    def _stop_instances(self) -> None:
        """Stop all launched instances (best-effort)."""
        for info in self._instances:
            try:
                self._http.post(
                    f"{self._orchestrator_url}/instances/{info.id}/stop",
                )
                logger.info("Stopped PinchTab instance %s", info.id)
            except Exception:
                logger.warning("Failed to stop instance %s", info.id, exc_info=True)
        self._instances.clear()
