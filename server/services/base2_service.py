# DeepFellow Software Framework.
# Copyright © 2025 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Base2 service."""

import asyncio
import logging
import shutil
import uuid
from abc import abstractmethod
from pathlib import Path
from typing import TypeVar

from fastapi import HTTPException
from pydantic import BaseModel

from server.applicationcontext import ApplicationContext
from server.docker import (
    DockerImage,
    docker_pull,
    get_docker_compose_logs,
    has_gpu_support_sync,
    is_docker_image_pulled,
    restart_docker_compose,
)
from server.endpointregistry import EndpointRegistry
from server.models.models import (
    AddCustomModelIn,
    CustomModelDefiniton,
    CustomModelId,
    InstallModelIn,
    InstallModelOut,
    UninstallModelIn,
)
from server.models.services import InstallServiceIn, InstallServiceOut, UninstallServiceIn
from server.serviceprovider import ServiceProvider, ServiceRawConfig
from server.services.base_service import BaseService
from server.utils.core import PromiseWithProgress, Stream, StreamChunk, StreamChunkProgress, Utils, convert_size_to_bytes
from server.utils.model_downloader import ModelDownloader


class ModelConfig(BaseModel):
    model_id: str
    options: InstallModelIn


class CustomModel(BaseModel):
    id: CustomModelId
    data: CustomModelDefiniton


class ServiceConfig(BaseModel):
    options: InstallServiceIn | None = None
    models: list[ModelConfig] | None = None
    custom: list[CustomModel] | None = None


T = TypeVar("T")

logger = logging.getLogger("uvicorn.error")


class Base2Service[T](BaseService):
    def __init__(
        self,
        application_context: ApplicationContext,
        endpoint_registry: EndpointRegistry,
        service_provider: ServiceProvider,
        model_downloader: ModelDownloader,
    ):
        super().__init__()
        self.application_context = application_context
        self.endpoint_registry = endpoint_registry
        self.service_provider = service_provider
        self.model_downloader = model_downloader
        self.installed: T | None = None
        self.installing = False
        self.custom = list[CustomModel]()
        self._after_init()

    def _after_init(self) -> None:
        """Do some custom initialization."""

    @abstractmethod
    def get_id(self) -> str:
        """Return the service id."""

    @abstractmethod
    def get_description(self) -> str:
        """Return the service description."""

    def is_installed(self) -> bool:
        """Check whether service is installed."""
        return self.installed is not None

    async def load_model(self, model: ModelConfig) -> None:
        """Load single model."""
        logger.info(f"{self.get_id()} loading model {model.model_id}")  # noqa: G004
        try:
            await (await self._install_model(model.model_id, model.options)).wait()
        except Exception:
            logger.exception(f"{self.get_id()} get error while loading model {model.model_id}")  # noqa: G004

    async def load(self, config: ServiceRawConfig) -> None:
        """Load service using the config."""
        cfg = ServiceConfig(**config)
        self.custom = cfg.custom or []
        for custom in self.custom:
            self._add_custom_model(custom)
        if not cfg.options:
            return
        promise = await self.install(cfg.options, save=False)
        await promise.wait()
        logger.info(f"{self.get_id()} service checked")  # noqa: G004
        tasks = [asyncio.create_task(self.load_model(model)) for model in cfg.models or []]
        await asyncio.gather(*tasks)

    async def _save(self) -> None:
        cfg = self._generate_config(self.installed)
        await self.service_provider.save_service_config(self.get_id(), cfg.model_dump())

    @abstractmethod
    def _generate_config(self, info: T | None) -> ServiceConfig:
        """Generate config."""

    async def install(self, options: InstallServiceIn, save: bool = True) -> PromiseWithProgress[InstallServiceOut, StreamChunk]:
        """Install the service."""
        if self.installed is not None:
            raise HTTPException(status_code=400, detail=f"Service {self.get_id()} already installed")
        if self.installing:
            raise HTTPException(status_code=400, detail=f"Service {self.get_id()} already installing")

        async def func(stream: Stream[StreamChunk]) -> InstallServiceOut:
            self.installing = True
            try:
                promise = await self._install_core(options)
                promise.progress.pipe(stream)
                self.installed = await promise.wait()
                if save:
                    await self._save()
                return InstallServiceOut(status="OK")
            finally:
                self.installing = False

        return PromiseWithProgress(func=func)

    @abstractmethod
    async def _install_core(self, options: InstallServiceIn) -> PromiseWithProgress[T, StreamChunk]:
        """Install service."""

    async def uninstall(self, options: UninstallServiceIn) -> None:
        """Uninstall the service."""
        await self._uninstall(options)
        await self._save()

    @abstractmethod
    async def _uninstall(self, options: UninstallServiceIn) -> None:
        """Uninstall service."""

    async def add_custom_model(self, options: AddCustomModelIn) -> CustomModelId:
        """Add custom model."""
        model = CustomModel(id=str(uuid.uuid4()), data=options.spec)
        self._add_custom_model(model)
        self.custom.append(model)
        await self._save()
        return model.id

    def _add_custom_model(self, model: CustomModel) -> None:  # noqa: ARG002
        """Add custom model."""
        raise HTTPException(400, "This service does not support custom models.")

    async def remove_custom_model(self, custom_model_id: CustomModelId) -> None:
        """Remove custom model."""
        model = next(x for x in self.custom if x.id == custom_model_id)
        if not model:
            return
        self._remove_custom_model(model)
        self.custom = [x for x in self.custom if x.id != custom_model_id]
        await self._save()

    def _remove_custom_model(self, model: CustomModel) -> None:  # noqa: ARG002
        """Remove custom model."""
        raise HTTPException(400, "This service does not support custom models.")

    async def install_model(self, model_id: str, options: InstallModelIn) -> PromiseWithProgress[InstallModelOut, StreamChunk]:
        """Install the model."""

        async def func(data: InstallModelOut) -> InstallModelOut:
            await self._save()
            return data

        return (await self._install_model(model_id, options)).next(func)

    @abstractmethod
    async def _install_model(self, model_id: str, options: InstallModelIn) -> PromiseWithProgress[InstallModelOut, StreamChunk]:
        """Install the model."""

    async def uninstall_model(self, model_id: str, options: UninstallModelIn) -> None:
        """Uninstall the model."""
        await self._uninstall_model(model_id, options)
        await self._save()

    @abstractmethod
    async def _uninstall_model(self, model_id: str, options: UninstallModelIn) -> None:
        """Uninstall the model."""

    async def get_docker_logs(self, model_id: str | None) -> str:
        """Get docker logs."""
        docker_compose_file_path = self.get_docker_compose_file_path(model_id)
        return await get_docker_compose_logs(docker_compose_file_path)

    async def get_docker_compose_file(self, model_id: str | None) -> str:
        """Get docker compose file."""
        docker_compose_file_path = self.get_docker_compose_file_path(model_id)
        return await Utils.read_file(docker_compose_file_path)

    async def restart_docker(self, model_id: str | None) -> None:
        """Get docker compose file."""
        docker_compose_file_path = self.get_docker_compose_file_path(model_id)
        await restart_docker_compose(docker_compose_file_path)

    def get_docker_compose_file_path(self, model_id: str | None) -> Path:  # noqa: ARG002
        """Get docker compose file path."""
        if not self.is_installed():
            raise HTTPException(400, "Service not installed")
        raise HTTPException(400, "Docker is not bound with this object")

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

    def _has_gpu_for_spec(self) -> str:
        return "true" if has_gpu_support_sync() else "false"

    async def _docker_pull(
        self,
        image: DockerImage,
        stream: Stream[StreamChunk],
        start_percentage: float = 0,
        end_percentage: float = 0.99,
    ) -> None:
        """Docker pull only if image does not exist."""
        multiplier = end_percentage - start_percentage
        stream.emit(StreamChunkProgress(type="progress", value=start_percentage))
        if not await is_docker_image_pulled(image.name):
            async for progress in docker_pull(image.name, convert_size_to_bytes(image.size) or 0):
                stream.emit(StreamChunkProgress(type="progress", value=progress * multiplier))
        stream.emit(StreamChunkProgress(type="progress", value=end_percentage))
