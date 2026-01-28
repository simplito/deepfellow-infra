# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Vllm service."""

import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from fastapi import HTTPException
from pydantic import BaseModel

from server.applicationcontext import get_base_url
from server.docker import (
    DockerImage,
    DockerOptions,
)
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
    ServiceOptions,
    ServiceSize,
    ServiceSpecification,
    UninstallServiceIn,
)
from server.services.base2_service import Base2Service, CustomModel, ModelConfig, ServiceConfig
from server.utils.core import (
    DownloadedPacket,
    PreDownloadPacket,
    PromiseWithProgress,
    Stream,
    StreamChunk,
    StreamChunkProgress,
    SuccessDownloadPacket,
    convert_size_to_bytes,
    normalize_name,
    try_parse_pydantic,
)
from server.utils.hardware import HardwarePartInfo
from server.utils.loading import Progress


class VllmModel(BaseModel):
    hf_id: str
    env_vars: dict[str, str] | None = None
    quantization: str | None = None
    dtype: str = "auto"
    shm_size: str = "16gb"
    ulimits: dict[str, str] | None = None
    max_model_len: int | None = None
    gpu_memory_utilization: float = 0.9
    size: str
    custom: CustomModelId | None = None


class VllmCustomModel(BaseModel):
    id: str
    hf_id: str
    size: str


type ImageTypes = Literal["cpu", "gpu"]


class VllmConst(BaseModel):
    images: dict[ImageTypes, DockerImage]
    model_type: str
    models: dict[str, VllmModel]


_const = VllmConst(
    images={
        "gpu": DockerImage(name="vllm/vllm-openai:v0.14.1-cu130", size="18.4 GB"),
        "cpu": DockerImage(name="public.ecr.aws/q9t5s3a7/vllm-cpu-release-repo:v0.14.1", size="3.4 GB"),
    },
    model_type="llm",
    models={
        "Qwen/Qwen3-0.6B": VllmModel(
            hf_id="Qwen/Qwen3-0.6B",
            max_model_len=8192,
            size="2GB",
        ),
        "speakleash/Bielik-4.5B-v3.0-Instruct": VllmModel(
            hf_id="speakleash/Bielik-4.5B-v3.0-Instruct",
            max_model_len=8192,
            size="10GB",
        ),
        "speakleash/Bielik-11B-v2.6-Instruct-FP8-Dynamic": VllmModel(
            hf_id="speakleash/Bielik-11B-v2.6-Instruct-FP8-Dynamic",
            max_model_len=4096,
            size="12GB",
        ),
        "google/gemma-3-1b-it": VllmModel(
            hf_id="google/gemma-3-1b-it",
            gpu_memory_utilization=0.85,
            max_model_len=None,
            size="2GB",
        ),
    },
)


@dataclass
class ModelInstalledInfo:
    id: str
    registered_name: str
    options: InstallModelIn
    docker: DockerOptions
    container_host: str
    container_port: int
    docker_exposed_port: int
    registration_id: RegistrationId
    model_path: Path
    base_url: str

    def get_info(self) -> ModelInfo:
        """Get info."""
        return ModelInfo(spec=self.options.spec, registration_id=self.registration_id)


class VllmOptions(BaseModel):
    hardware: str | bool | None = None


class VllmModelOptions(BaseModel):
    alias: str | None = None


@dataclass
class InstalledInfo:
    models: dict[str, ModelInstalledInfo]
    options: InstallServiceIn
    parsed_options: VllmOptions


@dataclass
class DownloadedInfo:
    model_path: str | None


class VllmService(Base2Service[InstalledInfo, DownloadedInfo]):
    hugging_face_cache_path = "/mnt/hf"
    models: dict[str, VllmModel]

    def _after_init(self) -> None:
        self.models = _const.models.copy()

    def get_id(self) -> str:
        """Return the service id."""
        return "vllm"

    def get_size(self) -> ServiceSize:
        """Return the service size."""
        sizes: dict[str, str] = {}
        if self.hardware.cpu.avx512:
            sizes["cpu"] = _const.images["cpu"].size
        if self.hardware.gpus:
            sizes["gpu"] = _const.images["gpu"].size
        return sizes

    def get_description(self) -> str:
        """Return the service description."""
        return "Self-hosted LLM high efficient model runner with complex configuration."

    def get_spec(self) -> ServiceSpecification:
        """Return the service specification."""
        fields = self.add_gpu_field_to_spec(add_cpu_option_only_on_avx512_support=True)
        return ServiceSpecification(fields=fields)

    def get_model_spec(self) -> ModelSpecification:
        """Return the model specification."""
        return ModelSpecification(
            fields=[
                ModelField(type="text", name="alias", description="Model alias", required=False),
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

    def get_installed_info(self) -> bool | InstallServiceProgress | ServiceOptions:
        """Get service installed info."""
        return self._get_service_installed_info() if self.installed is None else self.installed.options.spec

    def _generate_config(self, info: InstalledInfo | None) -> ServiceConfig:
        return ServiceConfig(
            options=info.options if info else None,
            models=[ModelConfig(model_id=x.id, options=x.options) for x in info.models.values()] if info else [],
            custom=self.custom,
            downloaded=self.models_downloaded,
            service_downloaded=self.service_downloaded,
        )

    def is_given_hardware_support_gpu(self, hardware_specification: str | bool | None) -> bool:
        """Return is gpu will be used."""
        if not self.hardware.cpu.avx512 and not self.hardware.gpus:
            raise HTTPException(
                400,
                (
                    "Your hardware doesn't support this service.\n"
                    "CPU does not support AVX 512 instructions.\n"
                    "There are not any graphic cards"
                ),
            )
        return super().is_given_hardware_support_gpu(hardware_specification)

    def get_specified_hardware_parts(self, hardware_specification: str | bool | None) -> Sequence[HardwarePartInfo]:
        """Get hardware based on user input."""
        if not self.hardware.cpu.avx512 and not self.hardware.gpus:
            raise HTTPException(
                400,
                (
                    "Your hardware doesn't support this service.\n"
                    "CPU does not support AVX 512 instructions.\n"
                    "There are not any graphic cards"
                ),
            )
        return super().get_specified_hardware_parts(hardware_specification)

    def _load_download_info(self, data: dict[str, Any]) -> DownloadedInfo:
        return DownloadedInfo(**data)

    async def _install_core(self, options: InstallServiceIn) -> PromiseWithProgress[InstalledInfo, StreamChunk]:
        if "hardware" not in options.spec:
            options.spec["hardware"] = options.spec.get("gpu", self.docker_service.has_gpu_support)
        parsed_options = try_parse_pydantic(VllmOptions, options.spec)
        image = self._get_image(self.is_given_hardware_support_gpu(parsed_options.hardware))
        await self._verify_docker_image(image.name, options.ignore_warnings)

        async def func(stream: Stream[StreamChunk]) -> InstalledInfo:
            await self._docker_pull(image, stream)
            self.service_downloaded = True
            return InstalledInfo(models={}, options=options, parsed_options=parsed_options)

        return PromiseWithProgress(func=func)

    async def _uninstall(self, options: UninstallServiceIn) -> None:
        if info := self.installed:
            for model in info.models.copy().values():
                await self._uninstall_model(model.id, UninstallModelIn(purge=options.purge))

        self.installed = None

        if options.purge:
            self.service_downloaded = False
            for image in _const.images.values():
                await self.docker_service.remove_image(image.name)
            await self._clear_working_dir()
            self.models_downloaded = {}

    def get_docker_compose_file_path(self, model_id: str | None) -> Path:
        """Get docker compose file path."""
        info = self.installed
        if not info:
            raise HTTPException(400, "Service not installed")
        if not model_id:
            raise HTTPException(400, "Docker is not bound with this object")
        installed = info.models.get(model_id, None)
        if not installed:
            raise HTTPException(status_code=400, detail="Model not installed")
        return self.docker_service.get_docker_compose_file_path(installed.docker.name)

    def _add_custom_model(self, model: CustomModel) -> None:
        parsed = try_parse_pydantic(VllmCustomModel, model.data)
        if parsed.id in self.models:
            raise HTTPException(400, "Model with given id already exists.")
        self.models[parsed.id] = VllmModel(hf_id=parsed.hf_id, size=parsed.size, custom=model.id)

    def _remove_custom_model(self, model: CustomModel) -> None:
        parsed = try_parse_pydantic(VllmCustomModel, model.data)
        if self.installed and parsed.id in self.installed.models:
            raise HTTPException(400, "Cannot remove custom model, it is in use, uninstall it first.")
        del self.models[parsed.id]

    async def list_models(self, filters: ListModelsFilters) -> ListModelsOut:
        """List models."""
        info = self._check_installed()
        out_list: list[RetrieveModelOut] = []
        for model_id, model in self.models.items():
            installed = info.models[model_id].get_info() if model_id in info.models else self._get_model_installed_info(model_id)
            if filters.installed is None or filters.installed == installed:
                out_list.append(
                    RetrieveModelOut(
                        id=model_id,
                        service=self.get_id(),
                        type=_const.model_type,
                        installed=installed,
                        downloaded=model_id in self.models_downloaded,
                        size=model.size,
                        custom=model.custom,
                        spec=self.get_model_spec(),
                        has_docker=True,
                    )
                )
        return ListModelsOut(list=out_list)

    async def get_model(self, model_id: str) -> RetrieveModelOut:
        """Get the model."""
        info = self._check_installed()
        if model_id not in self.models:
            raise HTTPException(status_code=400, detail="Model not found")
        model = self.models[model_id]
        installed = info.models[model_id].get_info() if model_id in info.models else self._get_model_installed_info(model_id)
        return RetrieveModelOut(
            id=model_id,
            service=self.get_id(),
            type=_const.model_type,
            installed=installed,
            downloaded=model_id in self.models_downloaded,
            size=model.size,
            custom=model.custom,
            spec=self.get_model_spec(),
            has_docker=True,
        )

    async def _install_model(self, model_id: str, options: InstallModelIn) -> PromiseWithProgress[InstallModelOut, StreamChunk]:  # noqa: C901
        parsed_model_options = try_parse_pydantic(VllmModelOptions, options.spec) if options.spec else VllmModelOptions()
        info = self._check_installed()
        if model_id in info.models:
            return PromiseWithProgress(value=InstallModelOut(status="OK", details="Already installed"))
        if model_id not in self.models:
            raise HTTPException(400, "Model not found")
        model = self.models[model_id]

        async def func(stream: Stream[StreamChunk]) -> InstallModelOut:  # noqa: C901
            model_id_fixed = model_id.replace("/", "-")
            models_dir = self._get_working_dir() / "models"
            model_dir = models_dir / model_id_fixed
            model_dir.mkdir(parents=True, exist_ok=True)
            progress = Progress(convert_size_to_bytes(model.size) or 0)
            local_model_path: Path | None = None
            stream.emit(StreamChunkProgress(type="progress", stage="download", value=0))
            async for packet in self.model_downloader.download(model_id, model_dir):
                if isinstance(packet, DownloadedPacket) and packet.downloaded_bytes_size != 0:
                    progress.add_to_actual_value(packet.downloaded_bytes_size)
                    stream.emit(StreamChunkProgress(type="progress", stage="download", value=progress.get_percentage()))
                elif isinstance(packet, PreDownloadPacket):
                    if max := packet.file_bytes_size:
                        progress.set_max_value(max)
                elif isinstance(packet, SuccessDownloadPacket):
                    local_model_path = packet.local_path

            stream.emit(StreamChunkProgress(type="progress", stage="download", value=1))
            stream.emit(StreamChunkProgress(type="progress", stage="install", value=0))

            docker_model_path = Path(self.hugging_face_cache_path) / "hub" / model_id_fixed
            volumes = [f"{local_model_path}:{docker_model_path}"]

            vllm_command = [
                "--model",
                str(docker_model_path),
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
                "--dtype",
                model.dtype,
                "--served-model-name",
                model_id,
            ]

            if model.quantization:
                vllm_command.extend(["--quantization", model.quantization])

            use_gpu = self.is_given_hardware_support_gpu(info.parsed_options.hardware)

            if use_gpu:
                vllm_command.extend(["--gpu-memory-utilization", str(model.gpu_memory_utilization)])

            if model.max_model_len and use_gpu:
                vllm_command.extend(["--max-model-len", str(model.max_model_len)])

            if not use_gpu:
                if model.max_model_len is None:
                    vllm_command.extend(["--disable-sliding-window"])
                else:
                    vllm_command.extend(["--max-model-len", str(model.max_model_len)])

            if not model.env_vars:
                model.env_vars = {}

            model.env_vars["HF_HUB_OFFLINE"] = "1"
            model.env_vars["HF_HOME"] = self.hugging_face_cache_path
            if not use_gpu:
                model.env_vars["VLLM_USE_V1"] = "1"

            image = self._get_image(use_gpu)

            subnet = self.docker_service.get_docker_subnet()
            service_name = f"{self.get_id()}-{normalize_name(model_id)}"
            docker_options = DockerOptions(
                name=service_name,
                container_name=self.docker_service.get_docker_container_name(service_name),
                image=image.name,
                command=" ".join(vllm_command),
                image_port=8000,
                env_vars=model.env_vars,
                restart="unless-stopped",
                volumes=volumes,
                hardware=self.get_specified_hardware_parts(info.parsed_options.hardware),
                shm_size=model.shm_size,
                ulimits=model.ulimits,
                subnet=subnet,
                healthcheck={
                    "test": "curl --fail http://localhost:8000/health || exit 1",
                    "interval": "60s",
                    "timeout": "10s",
                    "retries": "3",
                    "start_period": "240s",
                },
            )
            docker_exposed_port = await self.docker_service.install_and_run_docker(docker_options)
            registered_name = parsed_model_options.alias if parsed_model_options.alias else model_id
            container_host = self.docker_service.get_container_host(subnet, docker_options.name)
            container_port = self.docker_service.get_container_port(subnet, docker_exposed_port, docker_options.image_port)
            info.models[model_id] = model_info = ModelInstalledInfo(
                id=model_id,
                registered_name=registered_name,
                options=options,
                docker=docker_options,
                container_host=container_host,
                container_port=container_port,
                docker_exposed_port=docker_exposed_port,
                registration_id="",
                model_path=model_dir,
                base_url=get_base_url(container_host, container_port),
            )
            model_info.registration_id = self.endpoint_registry.register_chat_completion_as_proxy(
                model=registered_name,
                props=ModelProps(private=True),
                chat_completions=ProxyOptions(url=f"{model_info.base_url}/v1/chat/completions", rewrite_model_to=model_id),
                completions=ProxyOptions(url=f"{model_info.base_url}/v1/completions", rewrite_model_to=model_id),
                responses=ProxyOptions(url=f"{model_info.base_url}/v1/responses", rewrite_model_to=model_id),
                messages=ProxyOptions(url=f"{model_info.base_url}/v1/messages", rewrite_model_to=model_id),
                registration_options=None,
            )
            stream.emit(StreamChunkProgress(type="progress", stage="install", value=1))
            self.models_downloaded[model_id] = DownloadedInfo(str(local_model_path))
            return InstallModelOut(status="OK", details="Installed")

        return PromiseWithProgress(func=func)

    def _get_image(self, gpu: bool) -> DockerImage:
        return _const.images["gpu"] if gpu else _const.images["cpu"]

    async def _uninstall_model(self, model_id: str, options: UninstallModelIn) -> None:
        info = self._check_installed()
        if model_id in info.models:
            model = info.models[model_id]
            del info.models[model_id]
            self.endpoint_registry.unregister_chat_completion(model.registered_name, model.registration_id)
            await self.docker_service.uninstall_docker(model.docker)

        if options.purge and model_id in self.models_downloaded:
            model = self.models_downloaded[model_id]
            if model_path := model.model_path:
                shutil.rmtree(Path(model_path))
            del self.models_downloaded[model_id]

    async def stop(self) -> None:
        """Stop all the vLLM service Docker containers."""
        info = self.installed
        if not info:
            return
        await self._stop_dockers_parallel([model.docker for model in info.models.values()])
