# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Rerank service."""

import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from fastapi import HTTPException
from pydantic import BaseModel

from server.applicationcontext import get_base_url
from server.docker import DockerImage, DockerOptions
from server.endpointregistry import ProxyOptions, RegistrationId
from server.models.api import ModelProps
from server.models.models import (
    CustomModelField,
    CustomModelId,
    CustomModelSpecification,
    InstallModelIn,
    InstallModelOut,
    ListModelsFilters,
    ListModelsOut,
    ModelField,
    ModelInfo,
    ModelSpecification,
    RetrieveModelOut,
    UninstallModelIn,
)
from server.models.services import (
    InstallServiceIn,
    InstallServiceProgress,
    ServiceField,
    ServiceOptions,
    ServiceSize,
    ServiceSpecification,
    UninstallServiceIn,
)
from server.services.base2_service import Base2Service, CustomModel, Instance, InstanceConfig, ModelConfig
from server.utils.core import (
    DownloadedPacket,
    PreDownloadPacket,
    PromiseWithProgress,
    Stream,
    StreamChunk,
    StreamChunkProgress,
    convert_size_to_bytes,
    fetch_from,
    try_parse_pydantic,
)
from server.utils.hardware import GpuInfo, HardwarePartInfo, NvidiaGpuInfo
from server.utils.loading import Progress


class RerankModel(BaseModel):
    type: str
    size: str
    custom: CustomModelId | None = None


class RerankCustomModel(BaseModel):
    id: str
    hf_id: str
    size: str


type ImageTypes = Literal["cpu", "gpu"]


class RerankConst(BaseModel):
    images: dict[ImageTypes, DockerImage]
    models: dict[str, RerankModel]


_const = RerankConst(
    images={
        "gpu": DockerImage(name="hub.simplito.com/deepfellow/deepfellow-rerank:1.1.0-cuda12.8", size="8.29 GB"),
        "cpu": DockerImage(name="hub.simplito.com/deepfellow/deepfellow-rerank:1.1.0-cpu", size="1.14 GB"),
    },
    models={
        "cross-encoder/ms-marco-MiniLM-L6-v2": RerankModel(type="rerank", size="88MB"),
        "cross-encoder/ms-marco-MiniLM-L12-v2": RerankModel(type="rerank", size="129MB"),
        "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1": RerankModel(type="rerank", size="470MB"),
        "BAAI/bge-reranker-base": RerankModel(type="rerank", size="1.1GB"),
        "BAAI/bge-reranker-v2-m3": RerankModel(type="rerank", size="2.2GB"),
        "jinaai/jina-reranker-v2-base-multilingual": RerankModel(type="rerank", size="548MB"),
    },
)


@dataclass
class ModelInstalledInfo:
    id: str
    registered_name: str
    type: str
    options: InstallModelIn
    registration_id: RegistrationId

    def get_info(self) -> ModelInfo:
        """Get info."""
        return ModelInfo(spec=self.options.spec, registration_id=self.registration_id)


class RerankOptions(BaseModel):
    hardware: str | bool | None = None
    keep_alive: int | None = None


class RerankModelOptions(BaseModel):
    alias: str | None = None
    alive_time: int | None = None
    preload: bool | None = None


@dataclass
class InstalledInfo:
    docker: DockerOptions
    models: dict[str, ModelInstalledInfo]
    options: InstallServiceIn
    parsed_options: RerankOptions
    container_host: str
    container_port: int
    docker_exposed_port: int
    base_url: str


@dataclass
class DownloadedInfo:
    model_path: str | None


class RerankService(Base2Service[InstalledInfo, DownloadedInfo]):
    models: dict[str, dict[str, RerankModel]]

    def _after_init(self) -> None:
        self.models = {}
        self.load_default_models("default")

    def load_default_models(self, instance: str) -> None:
        """Load default models to instance."""
        self.models[instance] = _const.models.copy()

    def get_type(self) -> str:
        """Return the service id."""
        return "rerank"

    def get_description(self) -> str:
        """Return the service description."""
        return "Hosts reranking models."

    def get_size(self) -> ServiceSize:
        """Return the service size."""
        sizes = {"cpu": _const.images["cpu"].size}
        if self.hardware.nvidia_gpus:
            sizes["gpu"] = _const.images["gpu"].size
        return sizes

    @property
    def _supported_gpus(self) -> list[GpuInfo]:
        """Return GPUs supported by this service. Override to include non-NVIDIA GPUs."""
        return [gpu for gpu in self.hardware.gpus if isinstance(gpu, NvidiaGpuInfo)]

    def get_spec(self) -> ServiceSpecification:
        """Return the service specification."""
        fields = self.add_hardware_field_to_spec()

        fields.extend(
            [
                ServiceField(
                    type="text",
                    name="keep_alive",
                    description="The duration that models stay loaded in memory in seconds (default=300, instant_release=0, infinity=-1)",
                    required=False,
                ),
            ]
        )
        return ServiceSpecification(fields=fields)

    def get_model_spec(self) -> ModelSpecification:
        """Return the model specification."""
        return ModelSpecification(
            fields=[
                ModelField(type="text", name="alias", description="Model alias", required=False),
                ModelField(
                    type="text",
                    name="alive_time",
                    description="How long should this model last when it isn't used in second "
                    "(default=take value from service, instant_release=0, infinity=-1)",
                    required=False,
                ),
                ModelField(type="bool", name="preload", description="Load model to memory on service start", required=False),
            ]
        )

    def get_custom_model_spec(self) -> CustomModelSpecification | None:
        """Return the custom model specification or None if custom model is not supported."""
        return CustomModelSpecification(
            fields=[
                CustomModelField(type="text", name="id", description="Model ID", placeholder="my-custom-model"),
                CustomModelField(type="text", name="hf_id", description="Hugging face model ID", placeholder="google/gemma-3-270m-it"),
                CustomModelField(type="text", name="size", description="Model size", placeholder="1GB"),
            ]
        )

    def get_installed_info(self, instance: str) -> bool | InstallServiceProgress | ServiceOptions:
        """Get service installed info."""
        installed = self.get_instance_info(instance).installed
        return self._get_service_installed_info(instance) if installed is None else installed.options.spec

    def _generate_instance_config(self, info: InstalledInfo | None, custom: list[CustomModel] | None) -> InstanceConfig:
        return InstanceConfig(
            options=info.options if info else None,
            models=[ModelConfig(model_id=x.id, options=x.options) for x in info.models.values()] if info else [],
            custom=custom,
        )

    def _get_image(self, hardware: Sequence[HardwarePartInfo]) -> DockerImage:
        if any(isinstance(h, NvidiaGpuInfo) for h in hardware):
            return _const.images["gpu"]
        return _const.images["cpu"]

    def _load_download_info(self, data: dict[str, Any]) -> DownloadedInfo:
        return DownloadedInfo(**data)

    async def _install_instance(self, instance: str, options: InstallServiceIn) -> PromiseWithProgress[InstalledInfo, StreamChunk]:
        if not self.models.get(instance):
            self.load_default_models(instance)

        parsed_options = try_parse_pydantic(RerankOptions, options.spec)
        hardware_parts = self.get_specified_hardware_parts(parsed_options.hardware)
        image = self._get_image(hardware_parts)
        await self._verify_docker_image(image.name, options.ignore_warnings)

        async def func(stream: Stream[StreamChunk]) -> InstalledInfo:
            await self._download_image_or_set_progress(stream, image)
            self.service_downloaded = True
            stream.emit(StreamChunkProgress(type="progress", stage="install", value=0, data={}))
            volumes = [f"{self._get_working_dir()}/main:/root/.cache/huggingface"]
            subnet = self.docker_service.get_docker_subnet()
            name = f"{self.get_service_id(instance)}"
            envs = {}
            if parsed_options.keep_alive is not None:
                envs["DF_RERANK_KEEP_ALIVE"] = str(parsed_options.keep_alive)
            docker_options = DockerOptions(
                name=name,
                env_vars=envs,
                container_name=self.docker_service.get_docker_container_name(name),
                image=image.name,
                image_port=8089,
                hardware=hardware_parts,
                volumes=volumes,
                restart="unless-stopped",
                subnet=subnet,
                healthcheck={
                    "test": "wget -q --spider http://localhost:8089",
                    "interval": "30s",
                    "timeout": "10s",
                    "retries": "3",
                    "start_period": "30s",
                },
            )
            docker_exposed_port = await self.docker_service.install_and_run_docker(docker_options)
            container_host = self.docker_service.get_container_host(subnet, docker_options.name)
            container_port = self.docker_service.get_container_port(subnet, docker_exposed_port, docker_options.image_port)
            info = InstalledInfo(
                docker=docker_options,
                models={},
                options=options,
                parsed_options=parsed_options,
                container_host=container_host,
                container_port=container_port,
                docker_exposed_port=docker_exposed_port,
                base_url=get_base_url(container_host, container_port),
            )
            stream.emit(StreamChunkProgress(type="progress", stage="install", value=1, data={}))
            return info

        return PromiseWithProgress(func=func)

    async def _uninstall_instance(self, instance: str, options: UninstallServiceIn) -> None:
        installed = self.get_instance_info(instance).installed
        if installed:
            for model in installed.models.copy().values():
                if model.type == "rerank":
                    self.endpoint_registry.unregister_rerank(model.registered_name, model.registration_id)

                if not self.is_model_installed_in_other_instance(instance, model.id):
                    await self._uninstall_model(instance, model.id, UninstallModelIn(purge=options.purge))

            await self.docker_service.uninstall_docker(installed.docker)

        self.instances_info[instance].installed = None

        if options.purge:
            if len(self.instances_info) < 2:
                self.service_downloaded = False
                await self.docker_service.remove_image(_const.images["cpu"].name)
                await self.docker_service.remove_image(_const.images["gpu"].name)
                await self._clear_working_dir()
                self.models_downloaded = {}

            if instance == "default":
                self.instances_info["default"] = Instance(None, None, {}, InstanceConfig())
            else:
                del self.instances_info[instance]

    def get_docker_compose_file_path(self, instance: str, model_id: str | None) -> Path:
        """Get docker compose file path."""
        info = self.get_instance_installed_info(instance)
        if model_id:
            raise HTTPException(400, "Docker is not bound with this object")

        return self.docker_service.get_docker_compose_file_path(info.docker.name)

    def service_has_docker(self) -> bool:
        """Return true when docker is started when service is installed."""
        return True

    async def stop_instance(self, instance: str) -> None:
        """Stop the Rerank service Docker container."""
        installed = self.get_instance_info(instance).installed
        if not installed:
            return
        await self._stop_docker(installed.docker)

    async def list_models(self, input_instance: str | list[str] | None, filters: ListModelsFilters) -> ListModelsOut:
        """List models."""
        instances = [input_instance] if isinstance(input_instance, str) else input_instance if input_instance else self.instances_info

        for instance in instances:
            if instance not in self.instances_info:
                raise HTTPException(404, f"Instance {instance} doesn't exist.")

        out_list: list[RetrieveModelOut] = []
        for instance_name, instance_models in self.models.items():
            if instance_name not in instances:
                continue

            info = self.get_instance_installed_info(instance_name)
            for model_id, model in instance_models.items():
                if model_id in info.models:
                    installed = info.models[model_id].get_info()
                else:
                    installed = self._get_model_installed_info(instance_name, model_id)

                if filters.installed is None or filters.installed == bool(installed):
                    out_list.append(
                        RetrieveModelOut(
                            id=model_id,
                            service=self.get_id(instance_name),
                            type=model.type,
                            installed=installed,
                            downloaded=model_id in self.models_downloaded,
                            size=model.size,
                            spec=self.get_model_spec(),
                            has_docker=False,
                        )
                    )
        return ListModelsOut(list=out_list)

    async def get_model(self, instance: str, model_id: str) -> RetrieveModelOut:
        """Get the model."""
        info = self.get_instance_installed_info(instance)

        if not self.models.get(instance):
            self.models[instance] = {}

        if model_id not in _const.models:
            raise HTTPException(status_code=400, detail="Model not found")

        model = self.models[instance][model_id]
        installed = info.models[model_id].get_info() if model_id in info.models else self._get_model_installed_info(instance, model_id)
        return RetrieveModelOut(
            id=model_id,
            service=self.get_id(instance),
            type=model.type,
            installed=installed,
            downloaded=model_id in self.models_downloaded,
            size=model.size,
            spec=self.get_model_spec(),
            has_docker=False,
        )

    async def _download_model(self, stream: Stream[StreamChunk], model: RerankModel, model_id: str, model_dir: Path) -> None:
        progress = Progress(convert_size_to_bytes(model.size) or 0)
        stream.emit(StreamChunkProgress(type="progress", stage="download", value=0, data={}))
        async for packet in self.model_downloader.hugging_face_repo_with_blobs_downloader.download(model_id, model_dir, True):
            if isinstance(packet, DownloadedPacket) and packet.downloaded_bytes_size != 0:
                progress.add_to_actual_value(packet.downloaded_bytes_size)
                stream.emit(StreamChunkProgress(type="progress", stage="download", value=progress.get_percentage(), data={}))
            elif isinstance(packet, PreDownloadPacket):
                if max := packet.file_bytes_size:
                    progress.set_max_value(max)

        stream.emit(StreamChunkProgress(type="progress", stage="download", value=1, data={}))

    async def _download_model_or_set_progress(
        self, stream: Stream[StreamChunk], model: RerankModel, model_id: str, model_dir: Path
    ) -> None:
        if model_id not in self.models_download_progress:
            self.models_download_progress[model_id] = stream
            await self._download_model(stream, model, model_id, model_dir)
            del self.models_download_progress[model_id]
        else:
            chunk: StreamChunk
            async for chunk in self.models_download_progress[model_id].as_generator():
                if chunk.get("type") == "progress" and chunk.get("stage") == "download":
                    stream.emit(chunk)
                else:
                    break

    async def _install_model(
        self, instance: str, model_id: str, options: InstallModelIn
    ) -> PromiseWithProgress[InstallModelOut, StreamChunk]:
        parsed_model_options = try_parse_pydantic(RerankModelOptions, options.spec) if options.spec else RerankModelOptions()
        info = self.get_instance_installed_info(instance)

        if not self.models.get(instance):
            self.models[instance] = {}

        if model_id in info.models:
            return PromiseWithProgress(value=InstallModelOut(status="OK", details="Already installed"))

        if model_id not in _const.models:
            raise HTTPException(400, "Model not found")

        model = self.models[instance][model_id]

        async def func(stream: Stream[StreamChunk]) -> InstallModelOut:
            model_id_fixed = "models--" + model_id.replace("/", "--")
            models_dir = self._get_working_dir() / "main/hub"
            model_dir = models_dir / model_id_fixed
            model_dir.mkdir(parents=True, exist_ok=True)
            await self._download_model_or_set_progress(stream, model, model_id, model_dir)

            stream.emit(StreamChunkProgress(type="progress", stage="install", value=0, data={}))
            registered_name = parsed_model_options.alias if parsed_model_options.alias else model_id
            info.models[model_id] = model_info = ModelInstalledInfo(
                id=model_id,
                type=model.type,
                registered_name=registered_name,
                options=options,
                registration_id="",
            )
            if model.type == "rerank":
                model_info.registration_id = self.endpoint_registry.register_rerank_as_proxy(
                    model=registered_name,
                    props=ModelProps(
                        private=True,
                        type=model.type,
                        endpoints=["/v1/rerank"],
                    ),
                    options=ProxyOptions(url=f"{info.base_url}/v1/rerank", rewrite_model_to=model_id),
                    registration_options=None,
                )
            stream.emit(StreamChunkProgress(type="progress", stage="install", value=1, data={}))

            if parsed_model_options.alive_time is not None:
                await fetch_from(
                    f"{info.base_url}/v1/model/configure",
                    "POST",
                    {"model": model_id, "alive_time": parsed_model_options.alive_time},
                )
            if parsed_model_options.preload:
                await fetch_from(
                    f"{info.base_url}/v1/model/load",
                    "POST",
                    {"model": model_id},
                )
            self.models_downloaded[model_id] = DownloadedInfo(model_path=str(model_dir))
            return InstallModelOut(status="OK", details="Installed")

        return PromiseWithProgress(func=func)

    async def _uninstall_model(self, instance: str, model_id: str, options: UninstallModelIn) -> None:
        info = self.get_instance_installed_info(instance)
        if model_id in info.models:
            model = info.models[model_id]
            del info.models[model_id]
            if model.type == "rerank":
                self.endpoint_registry.unregister_rerank(model.registered_name, model.registration_id)
            await fetch_from(
                f"{info.base_url}/v1/model/unload",
                "POST",
                {"model": model_id},
            )

        if options.purge and model_id in self.models_downloaded:
            model = self.models_downloaded[model_id]
            if model_path := model.model_path:
                shutil.rmtree(Path(model_path))
            del self.models_downloaded[model_id]
