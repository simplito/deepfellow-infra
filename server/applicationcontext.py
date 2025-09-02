"""Application Content module."""

import importlib.util
import socket
from pathlib import Path
from typing import Any

from server.config import AppSettings
from server.utils.exceptions import AppError

from .endpointregistry import EndpointRegistry
from .serviceprovider import ServiceProvider


class ApplicationContext:
    def __init__(self, endpoint_registry: EndpointRegistry, config: AppSettings, service_provider: ServiceProvider):
        self.endpoint_registry = endpoint_registry
        self.config = config
        self.service_provider = service_provider
        self.allocated_ports = set()

    async def run(self, args: list[str]) -> Any:  # noqa: ANN401
        """Run service with given args."""
        service_name = args[0]
        if not service_name or ".." in service_name or "/" in service_name or "\\" in service_name:
            raise AppError("Invalid module name")
        service_path = (Path(__file__).parent / f"./services/{service_name}/index.py").resolve()
        if not service_path.is_file():
            raise FileNotFoundError("Service not found", (service_name, service_path))

        spec = importlib.util.spec_from_file_location(service_name, service_path)
        if not spec:
            raise AppError("No spec for service", (service_name, service_path))
        if not spec.loader:
            raise AppError("No spec loader for service", (service_name, service_path))
        plugin = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(plugin)
        if not hasattr(plugin, "service"):
            raise AttributeError("Service does not define a 'service(context, args)' function", service_name)
        return await plugin.service(self, args)

    async def load(self) -> None:
        """Load all service from bootstrap."""
        info = self.service_provider.load()
        for x in info["bootstrapCommands"]:
            await self.run(x["args"])

    async def add_command_to_bootstrap(self, args: list[str]) -> None:
        """Add command to bootstrap."""
        await self.service_provider.save_command(args)

    async def remove_command_from_bootstrap(self, args: list[str]) -> None:
        """Remove command from bootstrap."""
        await self.service_provider.remove_command(args)

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
        dir = Path(__file__).resolve().parent / "../storage/config"
        if not dir.is_dir():
            dir.mkdir(parents=True)
        return dir

    def get_tts_dir(self) -> Path:
        """Get text to speach dir."""
        dir = Path(__file__).resolve().parent / "../storage/tts"
        if not dir.is_dir():
            dir.mkdir(parents=True)
        return dir

    def get_images_dir(self) -> Path:
        """Get images dir."""
        dir = Path(__file__).resolve().parent / "../storage/images"
        if not dir.is_dir():
            dir.mkdir(parents=True)
            for name in ["extensions", "outputs", "embeddings", "models"]:
                (dir / name).mkdir(parents=True)
        return dir

    def get_model_dir(self) -> Path:
        """Get model dir."""
        dir = Path(__file__).resolve().parent / "../storage/models"
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
