# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Application Content module."""

import asyncio
import json
import time

from server.config import AppSettings, get_services_json_path
from server.services_manager import ServicesManager
from server.utils.logger import logger

from .endpointregistry import EndpointRegistry
from .serviceprovider import ServiceProvider, ServiceRawConfig, ServicesMigrationError


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

    async def _load_service(self, service_id: str, service_cfg: ServiceRawConfig) -> None:
        """Load single service."""
        start = time.time()
        logger.info(f"{service_id} loading...")
        if service_id not in self.services_manager.services:
            logger.warning(f"{service_id} is present in the persisted config but is not registered in this build")
            await self._record_warning_safely(
                service_id, f"Service '{service_id}' is present in the persisted config but is not registered in this build."
            )
            return
        try:
            await self.services_manager.load_service(service_id, service_cfg)
            logger.info(f"{service_id} fully loaded in {round(time.time() - start, 1)}s")
        except Exception as exc:
            logger.exception(f"{service_id} error occurs during loading {round(time.time() - start, 1)}s")
            await self._record_warning_safely(service_id, f"Service '{service_id}' failed to load: {exc}")
            return
        await self._dismiss_warnings_safely(service_id)

    async def _record_warning_safely(self, service_id: str, message: str) -> None:
        """Record a warning, but never let a failure to record one abort startup or masquerade as a load failure."""
        try:
            await self.service_provider.add_warning(service_id, message)
        except Exception:
            logger.exception(f"{service_id} failed to record warning")

    async def _dismiss_warnings_safely(self, service_id: str) -> None:
        """Dismiss recovered warnings, but never let a failure here be reported as a load failure."""
        try:
            await self.service_provider.dismiss_warnings_matching(service_id)
        except Exception:
            logger.exception(f"{service_id} failed to dismiss warnings")

    async def load_services(self) -> None:
        """Load all service from bootstrap."""
        try:
            info = await self.service_provider.load()
        except ServicesMigrationError as exc:
            path = get_services_json_path(self.config)
            logger.exception(f"Could not migrate {path} to the current version, no installed service was loaded")
            self.service_provider.load_error = (
                f"Could not migrate the services configuration ({path}) to the current version: {_describe_load_error(exc.cause)}. "
                "Installed services were not loaded and the file was left unchanged. "
                "Make sure DF Infra can write the file, then restart DF Infra."
            )
            return
        except Exception as exc:
            path = get_services_json_path(self.config)
            logger.exception(f"Could not read {path}, no installed service was loaded")
            self.service_provider.load_error = (
                f"Could not read the services configuration ({path}): {_describe_load_error(exc)}. Installed services were not loaded "
                "and the file was left unchanged. Fix or restore the file, then restart DF Infra."
            )
            return
        tasks = [asyncio.create_task(self._load_service(service_id, service_cfg)) for service_id, service_cfg in info["services"].items()]
        await asyncio.gather(*tasks)


def _describe_load_error(exc: Exception) -> str:
    """Describe why services.json could not be read, without the character offset a JSON error adds."""
    if isinstance(exc, json.JSONDecodeError):
        # Some messages already end with "at", e.g. "Unterminated string starting at".
        msg = exc.msg if exc.msg.endswith(" at") else f"{exc.msg} at"
        return f"invalid JSON: {msg} line {exc.lineno}, column {exc.colno}"
    if isinstance(exc, OSError) and exc.strerror:
        return exc.strerror
    return str(exc) or type(exc).__name__


def get_base_url(host: str, port: int) -> str:
    """Get base url."""
    return f"http://{host}:{port}"
