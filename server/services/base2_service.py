"""Base2 service."""

import logging
import shutil
from abc import abstractmethod
from pathlib import Path
from typing import TypeVar

from fastapi import HTTPException
from pydantic import BaseModel

from server.applicationcontext import ApplicationContext
from server.endpointregistry import EndpointRegistry
from server.models.models import InstallModelIn, UninstallModelIn
from server.models.services import InstallServiceIn, UninstallServiceIn
from server.serviceprovider import ServiceProvider, ServiceRawConfig
from server.services.base_service import BaseService


class ModelConfig(BaseModel):
    model_id: str
    options: InstallModelIn


class ServiceConfig(BaseModel):
    options: InstallServiceIn
    models: list[ModelConfig]


T = TypeVar("T")

logger = logging.getLogger("uvicorn.error")


class Base2Service[T](BaseService):
    def __init__(self, application_context: ApplicationContext, endpoint_registry: EndpointRegistry, service_provider: ServiceProvider):
        super().__init__()
        self.application_context = application_context
        self.endpoint_registry = endpoint_registry
        self.service_provider = service_provider
        self.installed: T | None = None
        self.installing = False

    @abstractmethod
    def get_id(self) -> str:
        """Return the service id."""

    def is_installed(self) -> bool:
        """Check whether service is installed."""
        return self.installed is not None

    async def load(self, config: ServiceRawConfig) -> None:
        """Load service using the config."""
        cfg = ServiceConfig(**config)
        await self._install(cfg.options)
        logger.info(f"{self.get_id()} service checked")  # noqa: G004
        for model in cfg.models:
            logger.info(f"{self.get_id()} loading model {model.model_id}")  # noqa: G004
            await self._install_model(model.model_id, model.options)

    async def _save(self) -> None:
        info = self._check_installed()
        cfg = self._generate_config(info)
        await self.service_provider.save_service_config(self.get_id(), cfg.model_dump())

    @abstractmethod
    def _generate_config(self, info: T) -> ServiceConfig:
        """Generate config."""

    async def install(self, options: InstallServiceIn) -> None:
        """Install the service."""
        await self._install(options)
        await self._save()

    async def _install(self, options: InstallServiceIn) -> None:
        if self.installed is not None:
            raise HTTPException(status_code=400, detail=f"Service {self.get_id()} already installed")
        if self.installing:
            raise HTTPException(status_code=400, detail=f"Service {self.get_id()} already installing")

        self.installing = True
        try:
            self.installed = await self._install_core(options)
        finally:
            self.installing = False

    @abstractmethod
    async def _install_core(self, options: InstallServiceIn) -> T:
        """Install service."""

    async def uninstall(self, options: UninstallServiceIn) -> None:
        """Uninstall the service."""
        await self._uninstall(options)
        await self.service_provider.clear_service_config(self.get_id())

    @abstractmethod
    async def _uninstall(self, options: UninstallServiceIn) -> None:
        """Uninstall service."""

    async def install_model(self, model_id: str, options: InstallModelIn) -> None:
        """Install the model."""
        await self._install_model(model_id, options)
        await self._save()

    @abstractmethod
    async def _install_model(self, model_id: str, options: InstallModelIn) -> None:
        """Install the model."""

    async def uninstall_model(self, model_id: str, options: UninstallModelIn) -> None:
        """Uninstall the model."""
        await self._uninstall_model(model_id, options)
        await self._save()

    @abstractmethod
    async def _uninstall_model(self, model_id: str, options: UninstallModelIn) -> None:
        """Uninstall the model."""

    def _get_working_dir(self) -> Path:
        return self.application_context.get_service_dir(self.get_id())

    async def _clear_working_dir(self) -> None:
        working_dir = self._get_working_dir()
        if working_dir.exists():
            shutil.rmtree(working_dir)

    def _check_installed(self) -> T:
        if self.installed is None:
            raise HTTPException(status_code=400, detail=f"Service {self.get_id()} not installed")
        return self.installed

    def get_hugging_face_token(self) -> str:
        """Return Hugging Face Key."""
        return self.application_context.config.hugging_face_token

    def get_civitai_token(self) -> str:
        """Return Civitai Face Key."""
        return self.application_context.config.civitai_token
