"""Application Content module."""

import socket
import time
from pathlib import Path

from server.config import AppSettings
from server.services_manager import ServicesManager

from .endpointregistry import EndpointRegistry
from .serviceprovider import ServiceProvider


class ApplicationContext:
    def __init__(
        self,
        endpoint_registry: EndpointRegistry,
        config: AppSettings,
        service_provider: ServiceProvider,
        services_manager: ServicesManager,
    ):
        self.endpoint_registry = endpoint_registry
        self.config = config
        self.service_provider = service_provider
        self.services_manager = services_manager
        self.allocated_ports = set[int]()

    async def load(self) -> None:
        """Load all service from bootstrap."""
        info = self.service_provider.load()
        for service_id, service_cfg in info["services"].items():
            start = time.time()
            await self.services_manager.load_service(service_id, service_cfg)
            print(f"{service_id} loaded in {round(time.time() - start, 1)}s")

    def get_free_port(self, start: int = 20_000, end: int = 30_000) -> int:
        """Get next free port."""
        for port in range(start, end + 1):
            if port in self.allocated_ports:
                continue
            if self.is_port_available(port):
                self.allocated_ports.add(port)
                return port
        raise RuntimeError("No free port in range", (start, end))

    def get_docker_compose_dir(self) -> Path:
        """Get docker compose dir."""
        dir = Path(__file__).resolve().parent.parent / "./storage/config"
        if not dir.is_dir():
            dir.mkdir(parents=True)
        return dir

    def get_service_dir(self, service: str) -> Path:
        """Get service dir."""
        dir = Path(__file__).resolve().parent.parent / f"./storage/services/{service}"
        if not dir.is_dir():
            dir.mkdir(parents=True)
        return dir

    def is_port_available(self, port: int) -> bool:
        """Check whether port is available."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("", port))
            sock.close()
            return True  # noqa: TRY300
        except OSError:
            return False
