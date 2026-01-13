# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
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
from typing import Any, TypeVar

from fastapi import HTTPException
from pydantic import BaseModel

from server.config import AppSettings
from server.docker import (
    DockerImage,
    DockerOptions,
    DockerService,
)
from server.endpointregistry import EndpointRegistry
from server.models.models import (
    AddCustomModelIn,
    CustomModelDefiniton,
    CustomModelId,
    InstallModelIn,
    InstallModelOut,
    InstallModelProgress,
    UninstallModelIn,
)
from server.models.services import InstallServiceIn, InstallServiceOut, InstallServiceProgress, UninstallServiceIn
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
    downloaded: dict[str, Any] | None = None


InstalledInfoType = TypeVar("InstalledInfoType")
DownloadInfoType = TypeVar("DownloadInfoType")

logger = logging.getLogger("uvicorn.error")


class InstallingModel:
    last_chunk: StreamChunk | None = None

    def __init__(self, promise: PromiseWithProgress[InstallModelOut, StreamChunk]):
        self.promise = promise

        async def the_func() -> None:
            async for chunk in promise.progress.as_generator():
                self.last_chunk = chunk

        self.task = asyncio.create_task(the_func())


class InstallingService:
    last_chunk: StreamChunk | None = None

    def __init__(self, promise: PromiseWithProgress[InstallServiceOut, StreamChunk]):
        self.promise = promise

        async def the_func() -> None:
            async for chunk in promise.progress.as_generator():
                self.last_chunk = chunk

        self.task = asyncio.create_task(the_func())


class Base2Service[InstalledInfoType, DownloadInfoType](BaseService):
    config: AppSettings
    endpoint_registry: EndpointRegistry
    service_provider: ServiceProvider
    model_downloader: ModelDownloader
    docker_service: DockerService
    downloaded: dict[str, DownloadInfoType]
    custom: list[CustomModel]
    installing_model_progress: dict[str, InstallingModel]
    installing: InstallingService | None

    def __init__(
        self,
        config: AppSettings,
        endpoint_registry: EndpointRegistry,
        service_provider: ServiceProvider,
        model_downloader: ModelDownloader,
        docker_service: DockerService,
    ):
        super().__init__()
        self.config = config
        self.endpoint_registry = endpoint_registry
        self.service_provider = service_provider
        self.model_downloader = model_downloader
        self.docker_service = docker_service
        self.installed: InstalledInfoType | None = None
        self.installing = None
        self.downloaded = {}
        self.custom = list[CustomModel]()
        self.installing_model_progress = {}
        self._after_init()

    def _after_init(self) -> None:
        """Do some custom initialization."""

    @abstractmethod
    def get_id(self) -> str:
        """Return the service id."""

    @abstractmethod
    def get_description(self) -> str:
        """Return the service description."""

    def get_model_install_progress(self, model: str) -> PromiseWithProgress[InstallModelOut, StreamChunk]:
        """Return actually installing models."""
        if installing := self.installing_model_progress.get(model):
            return installing.promise

        raise HTTPException(404, "This model is not installing now.")

    def get_service_install_progress(self) -> PromiseWithProgress[InstallServiceOut, StreamChunk]:
        """Return actually installing service."""
        if self.installing:
            return self.installing.promise

        raise HTTPException(404, "This service is not installing now.")

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
        self.downloaded = cfg.downloaded or {}
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
    def _generate_config(self, info: InstalledInfoType | None) -> ServiceConfig:
        """Generate config."""

    async def install(self, options: InstallServiceIn, save: bool = True) -> PromiseWithProgress[InstallServiceOut, StreamChunk]:
        """Install the service."""
        if self.installed is not None:
            raise HTTPException(status_code=400, detail=f"Service {self.get_id()} already installed")
        if self.installing:
            raise HTTPException(status_code=400, detail=f"Service {self.get_id()} already installing")

        async def func(data: InstalledInfoType) -> InstallServiceOut:
            self.installed = data
            self.installing = None
            if save:
                await self._save()
            return InstallServiceOut(status="OK")

        def on_error(_e: Exception) -> None:
            self.installing = None

        promise = await self._install_core(options)
        next_promise = promise.next(func, on_error)
        self.installing = InstallingService(promise=next_promise)
        return next_promise

    @abstractmethod
    async def _install_core(self, options: InstallServiceIn) -> PromiseWithProgress[InstalledInfoType, StreamChunk]:
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
            del self.installing_model_progress[model_id]
            return data

        def on_error(_e: Exception) -> None:
            del self.installing_model_progress[model_id]

        promise = await self._install_model(model_id, options)

        self.installing_model_progress[model_id] = InstallingModel(promise)

        return promise.next(func, on_error)

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
        return await self.docker_service.get_docker_compose_logs(docker_compose_file_path)

    def _get_model_installed_info(self, model_id: str) -> bool | InstallModelProgress:
        if model_id in self.installing_model_progress:
            installing = self.installing_model_progress[model_id]
            if installing.last_chunk and installing.last_chunk["type"] == "progress":
                return InstallModelProgress(stage=installing.last_chunk["stage"], value=installing.last_chunk["value"])
            if installing.last_chunk and installing.last_chunk["type"] == "finish":
                return InstallModelProgress(stage="install", value=1)
            return InstallModelProgress(stage="download", value=0)
        return False

    def _get_service_installed_info(self) -> bool | InstallServiceProgress:
        if not self.installing:
            return False
        if self.installing.last_chunk and self.installing.last_chunk["type"] == "progress":
            return InstallServiceProgress(stage=self.installing.last_chunk["stage"], value=self.installing.last_chunk["value"])
        if self.installing.last_chunk and self.installing.last_chunk["type"] == "finish":
            return InstallServiceProgress(stage="install", value=1)
        return InstallServiceProgress(stage="download", value=0)

    async def get_docker_compose_file(self, model_id: str | None) -> str:
        """Get docker compose file."""
        docker_compose_file_path = self.get_docker_compose_file_path(model_id)
        return await Utils.read_file(docker_compose_file_path)

    async def restart_docker(self, model_id: str | None) -> None:
        """Get docker compose file."""
        docker_compose_file_path = self.get_docker_compose_file_path(model_id)
        await self.docker_service.restart_docker_compose(docker_compose_file_path)

    def get_docker_compose_file_path(self, model_id: str | None) -> Path:  # noqa: ARG002
        """Get docker compose file path."""
        if not self.is_installed():
            raise HTTPException(400, "Service not installed")
        raise HTTPException(400, "Docker is not bound with this object")

    def _get_working_dir(self) -> Path:
        return self._get_service_dir(self.get_id())

    def _get_service_dir(self, service: str) -> Path:
        """Get service dir."""
        dir = self.config.get_storage_services_dir() / f"./{service}"
        if not dir.is_dir():
            dir.mkdir(parents=True)
        return dir

    async def _clear_working_dir(self) -> None:
        working_dir = self._get_working_dir()
        if working_dir.exists():
            shutil.rmtree(working_dir)

    def _check_installed(self) -> InstalledInfoType:
        if self.installed is None:
            raise HTTPException(status_code=400, detail=f"Service {self.get_id()} not installed")
        return self.installed

    def get_hugging_face_token(self) -> str:
        """Return Hugging Face Key."""
        return self.config.hugging_face_token

    def get_civitai_token(self) -> str:
        """Return Civitai Face Key."""
        return self.config.civitai_token

    def _has_gpu_for_spec(self) -> str:
        return "true" if self.docker_service.has_gpu_support else "false"

    async def _docker_pull(
        self,
        image: DockerImage,
        stream: Stream[StreamChunk],
    ) -> None:
        """Docker pull only if image does not exist."""
        stream.emit(StreamChunkProgress(type="progress", stage="download", value=0))
        if not await self.docker_service.is_docker_image_pulled(image.name):
            size = await self.docker_service.get_docker_image_size(image.name)
            async for progress in self.docker_service.docker_pull(image.name, size or convert_size_to_bytes(image.size) or 0):
                stream.emit(StreamChunkProgress(type="progress", stage="download", value=progress))
        stream.emit(StreamChunkProgress(type="progress", stage="download", value=1))

    async def _stop_docker(self, docker_options: DockerOptions) -> None:
        """Stop docker and log error if it occurs."""
        try:
            await self.docker_service.stop_docker(docker_options)
        except Exception:
            logger.exception("Error during stopping docker compose %s", docker_options.name)

    async def _stop_dockers_parallel(self, docker_options_list: list[DockerOptions]) -> None:
        """Stop docker and log error if it occurs."""
        tasks = [asyncio.create_task(self._stop_docker(docker_options)) for docker_options in docker_options_list]
        await asyncio.gather(*tasks)

    async def _verify_docker_image(self, docker_image: str, ignore_warning: bool) -> None:
        warnings = await self.docker_service.get_image_warnings(docker_image)
        if len(warnings) > 0 and not ignore_warning:
            raise HTTPException(400, {"warnings": warnings})
