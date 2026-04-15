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
import contextlib
import logging
import shutil
import uuid
from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

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
    ModelField,
    UninstallModelIn,
)
from server.models.services import (
    InstallServiceIn,
    InstallServiceOut,
    InstallServiceProgress,
    OneOfOption,
    ServiceField,
    UninstallServiceIn,
)
from server.serviceprovider import ServiceProvider, ServiceRawConfig
from server.services.base_service import BaseService
from server.utils.core import PromiseWithProgress, Stream, StreamChunk, StreamChunkProgress, Utils, convert_size_to_bytes
from server.utils.hardware import GpuInfo, Hardware, HardwarePartInfo, NvidiaGpuInfo
from server.utils.model_downloader import ModelDownloader


class ModelConfig(BaseModel):
    model_id: str
    options: InstallModelIn


class CustomModel(BaseModel):
    id: CustomModelId
    data: CustomModelDefiniton


class InstanceConfig(BaseModel):
    options: InstallServiceIn | None = None
    models: list[ModelConfig] | None = None
    custom: list[CustomModel] | None = None


class ServiceConfig(BaseModel):
    instances: dict[str, InstanceConfig] | None = None
    downloaded: dict[str, Any] | None = None
    service_downloaded: bool | None = True


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


class InstallingInstance:
    last_chunk: StreamChunk | None = None

    def __init__(self, promise: PromiseWithProgress[InstallServiceOut, StreamChunk]):
        self.promise = promise

        async def the_func() -> None:
            async for chunk in promise.progress.as_generator():
                self.last_chunk = chunk

        self.task = asyncio.create_task(the_func())


@dataclass
class Instance[InstalledInfoType]:
    installed: InstalledInfoType | None
    installing: InstallingInstance | None
    installing_model_progress: dict[str, InstallingModel]
    config: InstanceConfig


class Base2Service(Generic[InstalledInfoType, DownloadInfoType], BaseService):  # noqa: UP046
    config: AppSettings
    endpoint_registry: EndpointRegistry
    service_provider: ServiceProvider
    model_downloader: ModelDownloader
    docker_service: DockerService
    models_downloaded: dict[str, DownloadInfoType]
    service_downloaded: bool
    hardware: Hardware
    instances_info: dict[str, Instance[InstalledInfoType]]
    images_download_progress: dict[str, Stream[StreamChunk]]
    models_download_progress: dict[str, Stream[StreamChunk]]

    def __init__(
        self,
        config: AppSettings,
        endpoint_registry: EndpointRegistry,
        service_provider: ServiceProvider,
        model_downloader: ModelDownloader,
        docker_service: DockerService,
        hardware: Hardware,
    ):
        super().__init__()
        self.config = config
        self.endpoint_registry = endpoint_registry
        self.service_provider = service_provider
        self.model_downloader = model_downloader
        self.docker_service = docker_service
        self.hardware = hardware
        self.models_downloaded = {}
        self.service_downloaded = False
        self.instances_info = {"default": Instance(None, None, {}, InstanceConfig())}
        self.models_download_progress = {}
        self.images_download_progress = {}
        self._after_init()

    def _after_init(self) -> None:
        """Do some custom initialization."""

    def load_default_models(self, instance: str) -> None:
        """Load default models to instance."""

    @abstractmethod
    def get_type(self) -> str:
        """Return the type."""

    @abstractmethod
    def get_description(self) -> str:
        """Return the service description."""

    def check_instance_exists(self, instance: str) -> None:
        """Check is instance exists."""
        if not self.instances_info.get(instance):
            raise HTTPException(404, "Instance doesn't exist.")

    def is_model_installed_in_other_instance(self, instance: str, model: str) -> bool:
        """Check is model installed in other instance."""
        return any(model in getattr(self.instances_info[i].installed, "models", []) for i in self.instances_info if i != instance)

    def get_instance_info(self, instance: str) -> Instance[InstalledInfoType]:
        """Return instance info."""
        instance_info = self.instances_info.get(instance)
        if not instance_info:
            raise HTTPException(404, "Instance doesn't exist.")
        return instance_info

    def get_model_install_progress(self, instance: str, model: str) -> PromiseWithProgress[InstallModelOut, StreamChunk]:
        """Return actually installing models."""
        installing = self.get_instance_info(instance).installing_model_progress.get(model)

        if not installing:
            raise HTTPException(404, "This model is not installing now.")

        return installing.promise

    def get_instance_install_progress(self, instance: str) -> PromiseWithProgress[InstallServiceOut, StreamChunk]:
        """Return actually installing service."""
        installing = self.get_instance_info(instance).installing

        if not installing:
            raise HTTPException(404, "This service is not installing now.")

        return installing.promise

    def is_installed(self, instance: str) -> bool:
        """Check whether service is installed."""
        return self.get_instance_info(instance).installed is not None

    def get_downloaded(self) -> bool:
        """Get service downloaded info."""
        return self.service_downloaded

    async def load_model(self, instance: str, model: ModelConfig) -> None:
        """Load single model."""
        logger.info(f"{self.get_id(instance)} loading model {model.model_id}")  # noqa: G004
        try:
            await (await self._install_model(instance, model.model_id, model.options)).wait()
        except Exception:
            logger.exception(f"{self.get_id(instance)} get error while loading model {model.model_id}")  # noqa: G004

    async def load_instance(self, instance: str, instance_data: InstanceConfig) -> None:
        """Load instance of service."""
        if not instance_data.options:
            return

        self.load_default_models(instance)

        if instance_data.custom:
            for custom in instance_data.custom:
                self._add_custom_model(instance, custom)

        promise = await self.install_instance(instance, instance_data.options, instance_data, save=False)
        await promise.wait()
        logger.info(f"{self.get_id(instance)} service checked")  # noqa: G004
        tasks = [asyncio.create_task(self.load_model(instance, model)) for model in instance_data.models or []]
        await asyncio.gather(*tasks)

    async def load_service(self, config: ServiceRawConfig) -> None:
        """Load service using the config."""
        cfg = ServiceConfig(**config)
        self.models_downloaded = ({key: self._load_download_info(value) for key, value in cfg.downloaded.items()}) if cfg.downloaded else {}
        self.service_downloaded = cfg.service_downloaded or False

        msg = f"{self.get_type()} service installed."
        logger.info(msg)
        if cfg.instances:
            tasks = [asyncio.create_task(self.load_instance(name, instance)) for name, instance in cfg.instances.items()]
            await asyncio.gather(*tasks)

    @abstractmethod
    def _load_download_info(self, data: dict[str, Any]) -> DownloadInfoType:
        pass

    async def _save(self) -> None:
        instances_config = {}
        for instance_name, instance in self.instances_info.items():
            instances_config[instance_name] = self._generate_instance_config(instance.installed, instance.config.custom)
        cfg = self.service_config(instances_config)
        await self.service_provider.save_service_config(self.get_type(), cfg.model_dump())

    @abstractmethod
    def _generate_instance_config(self, info: InstalledInfoType | None, custom: list[CustomModel] | None) -> InstanceConfig:
        """Generate instance config."""

    def service_config(self, instances_config: dict[str, InstanceConfig]) -> ServiceConfig:
        """Generate service config."""
        return ServiceConfig(instances=instances_config, downloaded=self.models_downloaded, service_downloaded=self.service_downloaded)

    async def install_instance(
        self, instance: str, options: InstallServiceIn, instance_config: InstanceConfig | None = None, save: bool = True
    ) -> PromiseWithProgress[InstallServiceOut, StreamChunk]:
        """Install the service."""
        if self.instances_info and self.instances_info.get(instance):
            if self.instances_info[instance].installed:
                raise HTTPException(status_code=400, detail=f"Service {self.get_id(instance)} on {instance} instance already installed")
            if self.instances_info[instance].installing:
                raise HTTPException(status_code=400, detail=f"Service {self.get_id(instance)} on {instance} instance already installing")

        self.instances_info[instance] = Instance(None, None, {}, instance_config or InstanceConfig())

        async def func(data: InstalledInfoType) -> InstallServiceOut:
            self.instances_info[instance].installed = data
            if save:
                await self._save()
            self.instances_info[instance].installing = None
            return InstallServiceOut(status="OK")

        def on_error(_e: Exception) -> None:
            self.instances_info[instance].installing = None

        promise = await self._install_instance(instance, options)
        next_promise = promise.next(func, on_error)
        self.instances_info[instance].installing = InstallingInstance(promise=next_promise)
        return next_promise

    @abstractmethod
    async def _install_instance(self, instance: str, options: InstallServiceIn) -> PromiseWithProgress[InstalledInfoType, StreamChunk]:
        """Install service."""

    async def uninstall_instance(self, instance: str, options: UninstallServiceIn) -> None:
        """Uninstall the service."""
        await self._uninstall_instance(instance, options)
        await self._save()

    @abstractmethod
    async def _uninstall_instance(self, instance: str, options: UninstallServiceIn) -> None:
        """Uninstall instance."""

    async def add_custom_model(self, instance: str, options: AddCustomModelIn) -> CustomModelId:
        """Add custom model."""
        model = CustomModel(id=str(uuid.uuid4()), data=options.spec)
        config = self.get_instance_info(instance).config
        self._add_custom_model(instance, model)
        if config.custom is None:
            config.custom = []

        config.custom.append(model)
        await self._save()
        return model.id

    def _add_custom_model(self, instance: str, model: CustomModel) -> None:  # noqa: ARG002
        """Add custom model."""
        raise HTTPException(400, "This service does not support custom models.")

    async def remove_custom_model(self, instance: str, custom_model_id: CustomModelId) -> None:
        """Remove custom model."""
        config = self.get_instance_info(instance).config
        model = next(x for x in config.custom or {} if x.id == custom_model_id)
        if not model:
            return
        self._remove_custom_model(instance, model)
        config.custom = [x for x in config.custom or {} if x.id != custom_model_id]
        await self._save()

    def _remove_custom_model(self, instance: str, model: CustomModel) -> None:  # noqa: ARG002
        """Remove custom model."""
        raise HTTPException(400, "This service does not support custom models.")

    async def install_model(
        self, instance: str, model_id: str, options: InstallModelIn
    ) -> PromiseWithProgress[InstallModelOut, StreamChunk]:
        """Install the model."""
        installing_model_progress = self.get_instance_info(instance).installing_model_progress

        async def func(data: InstallModelOut) -> InstallModelOut:
            await self._save()
            msg = f"{model_id} model installed."
            logger.debug(msg)
            with contextlib.suppress(KeyError):
                del installing_model_progress[model_id]
            return data

        def on_error(_e: Exception) -> None:
            del installing_model_progress[model_id]

        promise = await self._install_model(instance, model_id, options)

        installing_model_progress[model_id] = InstallingModel(promise)

        return promise.next(func, on_error)

    @abstractmethod
    async def _install_model(
        self, instance: str, model_id: str, options: InstallModelIn
    ) -> PromiseWithProgress[InstallModelOut, StreamChunk]:
        """Install the model."""

    async def uninstall_model(self, instance: str, model_id: str, options: UninstallModelIn) -> None:
        """Uninstall the model."""
        await self._uninstall_model(instance, model_id, options)
        await self._save()

    @abstractmethod
    async def _uninstall_model(self, instance: str, model_id: str, options: UninstallModelIn) -> None:
        """Uninstall the model."""

    async def get_docker_logs(self, instance: str, model_id: str | None) -> str:
        """Get docker logs."""
        docker_compose_file_path = self.get_docker_compose_file_path(instance, model_id)
        return await self.docker_service.get_docker_compose_logs(docker_compose_file_path)

    def _get_model_installed_info(self, instance: str, model_id: str) -> bool | InstallModelProgress:
        installing_model_progress = self.get_instance_info(instance).installing_model_progress

        if model_id in installing_model_progress:
            installing = installing_model_progress[model_id]
            if installing.last_chunk and installing.last_chunk["type"] == "progress":
                return InstallModelProgress(stage=installing.last_chunk["stage"], value=installing.last_chunk["value"])
            if installing.last_chunk and installing.last_chunk["type"] == "finish":
                return InstallModelProgress(stage="install", value=1)
            return InstallModelProgress(stage="download", value=0)
        return False

    def _get_service_installed_info(self, instance: str) -> bool | InstallServiceProgress:
        installing = self.get_instance_info(instance).installing

        if not installing:
            return False
        if installing.last_chunk and installing.last_chunk["type"] == "progress":
            return InstallServiceProgress(stage=installing.last_chunk["stage"], value=installing.last_chunk["value"])
        if installing.last_chunk and installing.last_chunk["type"] == "finish":
            return InstallServiceProgress(stage="install", value=1)
        return InstallServiceProgress(stage="download", value=0)

    async def get_docker_compose_file(self, instance: str, model_id: str | None) -> str:
        """Get docker compose file."""
        docker_compose_file_path = self.get_docker_compose_file_path(instance, model_id)
        return await Utils.read_file(docker_compose_file_path)

    async def restart_docker(self, instance: str, model_id: str | None) -> None:
        """Get docker compose file."""
        docker_compose_file_path = self.get_docker_compose_file_path(instance, model_id)
        await self.docker_service.restart_docker_compose(docker_compose_file_path)

    def get_docker_compose_file_path(self, instance: str, model_id: str | None) -> Path:  # noqa: ARG002
        """Get docker compose file path."""
        if not self.is_installed(instance):
            raise HTTPException(400, "Instance not installed")
        raise HTTPException(400, "Docker is not bound with this object")

    def _get_working_dir(self) -> Path:
        return self._get_service_dir(self.get_type())

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

    def get_instance_installed_info(self, instance: str) -> InstalledInfoType:
        """Get instance installed info or return exception."""
        installed = self.get_instance_info(instance).installed
        if installed is None:
            raise HTTPException(status_code=400, detail=f"Service {self.get_id(instance)} not installed")
        return installed

    def get_hugging_face_token(self) -> str:
        """Return Hugging Face Key."""
        return self.config.hugging_face_token

    def get_civitai_token(self) -> str:
        """Return Civitai Face Key."""
        return self.config.civitai_token

    def _has_gpu_for_spec(self) -> str:
        return "true" if self.docker_service.has_gpu_support else "false"

    async def _download_image_or_set_progress(self, stream: Stream[StreamChunk], image: DockerImage) -> None:
        if image.name not in self.images_download_progress:
            self.images_download_progress[image.name] = stream
            await self._docker_pull(image, stream)
            del self.images_download_progress[image.name]
        else:
            chunk: StreamChunk
            async for chunk in self.images_download_progress[image.name].as_generator():
                if chunk.get("type") == "progress" and chunk.get("stage") == "download":
                    stream.emit(chunk)
                else:
                    break

    async def _docker_pull(
        self,
        image: DockerImage,
        stream: Stream[StreamChunk],
    ) -> None:
        """Docker pull only if image does not exist."""
        stream.emit(StreamChunkProgress(type="progress", stage="download", value=0, data={}))
        if not await self.docker_service.is_docker_image_pulled(image.name):
            size = await self.docker_service.get_docker_image_size(image.name)
            async for progress in self.docker_service.docker_pull(image.name, size or convert_size_to_bytes(image.size) or 0):
                stream.emit(StreamChunkProgress(type="progress", stage="download", value=progress, data={}))
        stream.emit(StreamChunkProgress(type="progress", stage="download", value=1, data={}))

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

    @property
    def _supported_gpus(self) -> list[GpuInfo]:
        """Return GPUs supported by this service. Override to include non-NVIDIA GPUs."""
        return [gpu for gpu in self.hardware.gpus if isinstance(gpu, NvidiaGpuInfo)]

    def is_given_hardware_support_gpu(self, hardware_specification: str | bool | None) -> bool:
        """Return is gpu will be used."""
        if hardware_specification is None:
            return bool(self._supported_gpus)
        if isinstance(hardware_specification, str):
            has_gpu_support = hardware_specification.startswith("GPU")
            if has_gpu_support and not self._supported_gpus:
                raise HTTPException(400, "Given hardware specification is not supported")
            return has_gpu_support
        return hardware_specification

    def get_specified_hardware_parts(self, hardware_specification: str | bool | None) -> Sequence[HardwarePartInfo]:
        """Get specified hardware parts."""
        if hardware_specification is None:
            return self._supported_gpus if self._supported_gpus else [self.hardware.cpu]
        if (hardware_specification is False) or (isinstance(hardware_specification, str) and hardware_specification == "CPU"):
            return [self.hardware.cpu]

        if (hardware_specification is True) or (hardware_specification == "GPUs") or (hardware_specification == "GPU"):
            return self._supported_gpus

        gpus: list[GpuInfo] = []

        for gpu_name in hardware_specification.removeprefix("GPU | ").split(","):
            for gpu in self._supported_gpus:
                if gpu_name == gpu.long_name:
                    gpus.append(gpu)

        return gpus

    def add_hardware_field_to_spec(
        self,
        fields: list[ServiceField] | None = None,
        add_cpu_option_only_on_avx512_support: bool = False,
    ) -> list[ServiceField]:
        """Add hardware (CPU/GPU/GPUs) field to specification."""
        fields = fields or []
        options: list[str | OneOfOption] = []
        default: str | None = None
        if not add_cpu_option_only_on_avx512_support or self.hardware.cpu.avx512:
            options.append("CPU")
            default = "CPU"
        gpus = self._supported_gpus
        if len(gpus) == 1:
            options.append(OneOfOption(value="GPU", label="Default GPU"))
            default = "GPU"
        elif gpus:
            options.append(OneOfOption(value="GPUs", label="All GPUs"))
            default = "GPUs"
        options.extend([f"GPU | {gpu.long_name}" for gpu in gpus])

        gpus_select_field = ServiceField(type="oneof", name="hardware", description="Choose hardware:", values=options, default=default)
        fields.append(gpus_select_field)

        return fields

    def add_hardware_field_to_model_spec(
        self,
        fields: list[ModelField] | None = None,
        add_cpu_option_only_on_avx512_support: bool = False,
    ) -> list[ModelField]:
        """Add hardware (CPU/GPU/GPUs) field to specification."""
        fields = fields or []
        options: list[str | OneOfOption] = []
        default: str | None = None
        if not add_cpu_option_only_on_avx512_support or self.hardware.cpu.avx512:
            options.append("CPU")
            default = "CPU"
        gpus = self._supported_gpus
        if len(gpus) == 1:
            options.append(OneOfOption(value="GPU", label="Default GPU"))
            default = "GPU"
        elif gpus:
            options.append(OneOfOption(value="GPUs", label="All GPUs"))
            default = "GPUs"
        options.extend([f"GPU | {gpu.long_name}" for gpu in gpus])

        gpus_select_field = ModelField(type="oneof", name="hardware", description="Choose hardware:", values=options, default=default)
        fields.append(gpus_select_field)

        return fields
