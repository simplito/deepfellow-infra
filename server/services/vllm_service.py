# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Vllm service."""

import json
import logging
import re
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict

from fastapi import HTTPException
from pydantic import BaseModel

from server.applicationcontext import get_base_url
from server.config import get_main_dir
from server.docker import (
    DockerImage,
    DockerOptions,
)
from server.endpointregistry import ProxyOptions, RegistrationId
from server.models.api import LLM_ENDPOINTS, ModelProps
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
from server.services.base2_service import Base2Service, CustomModel, Instance, InstanceConfig, ModelConfig
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
from server.utils.files import get_model_dir_context_window
from server.utils.hardware import HardwarePartInfo
from server.utils.loading import Progress

logger = logging.getLogger("uvicorn.error")


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
    model_type: Literal["llm", "reranker"] = "llm"


class VllmCustomModel(BaseModel):
    id: str
    hf_id: str
    size: str
    model_type: Literal["llm", "reranker"] = "llm"


type ImageTypes = Literal["cpu", "gpu"]


class VllmRegistryEntry(TypedDict):
    name: str
    size: str


class VllmRegistry(TypedDict, total=False):
    llms: list[VllmRegistryEntry]
    rerankers: list[VllmRegistryEntry]


class VllmConst(BaseModel):
    images: dict[ImageTypes, DockerImage]
    models: dict[str, VllmModel]


def _read_models_from_json() -> dict[str, bool]:  # pyright: ignore[reportUnusedFunction]
    vllm_path = get_main_dir() / "./static/vllm.json"
    with vllm_path.open(encoding="utf-8") as f:
        data = json.loads(f.read())
        # tags = [x["tags"] for x in data["list"]]
        # flat: list[str] = [x["tag"] for sublist in tags for x in sublist]
        tags = [x["mainTags"] for x in data["list"]]
        flat: list[str] = [x.removesuffix(":latest") for sublist in tags for x in sublist]

        map: dict[str, bool] = {}
        for tag in flat:
            map[tag] = True
        return map


def _read_models() -> dict[str, VllmModel]:
    vllm_path = get_main_dir() / "./static/vllm-min.json"
    with vllm_path.open(encoding="utf-8") as f:
        registry: VllmRegistry = json.loads(f.read())
        map: dict[str, VllmModel] = {}
        for entry in registry.get("llms", []):
            map[entry["name"]] = VllmModel(
                hf_id=entry["name"],
                size=entry["size"],
                model_type="llm",
            )
        for entry in registry.get("rerankers", []):
            map[entry["name"]] = VllmModel(
                hf_id=entry["name"],
                size=entry["size"],
                model_type="reranker",
            )
        return map


_const = VllmConst(
    images={
        "gpu": DockerImage(name="vllm/vllm-openai:v0.14.1-cu130", size="18.4 GB"),
        "cpu": DockerImage(name="public.ecr.aws/q9t5s3a7/vllm-cpu-release-repo:v0.14.1", size="3.4 GB"),
    },
    models=_read_models(),
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
    gpu_memory_utilization: float | None
    model_type: Literal["llm", "reranker"] = "llm"

    def get_info(self) -> ModelInfo:
        """Get info."""
        return ModelInfo(spec=self.options.spec, registration_id=self.registration_id)


class VllmOptions(BaseModel):
    hardware: str | bool | None = None


class VllmModelOptions(BaseModel):
    alias: str | None = None
    gpu_memory_utilization: float | None = None
    max_model_length: int | None = None
    quantization: str | None = None
    extra_args: dict[str, str] = {}
    extra_envs: dict[str, str] = {}


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
    models: dict[str, dict[str, VllmModel]]
    gpu_memory_utilization = 0

    def _after_init(self) -> None:
        self.models = {}
        self.load_default_models("default")

    def load_default_models(self, instance: str) -> None:
        """Load default models to instance."""
        self.models[instance] = _const.models.copy()

    def get_type(self) -> str:
        """Return the service id."""
        return "vllm"

    def get_description(self) -> str:
        """Return the service description."""
        return "Self-hosted LLM high efficient model runner with complex configuration."

    def get_size(self) -> ServiceSize:
        """Return the service size."""
        sizes: dict[str, str] = {}
        if self.hardware.cpu.avx512:
            sizes["cpu"] = _const.images["cpu"].size
        if self._supported_gpus:
            sizes["gpu"] = _const.images["gpu"].size
        return sizes

    def get_spec(self) -> ServiceSpecification:
        """Return the service specification."""
        fields = self.add_hardware_field_to_spec(add_cpu_option_only_on_avx512_support=True)

        return ServiceSpecification(fields=fields)

    def get_model_spec(self, instance: str, model_type: str) -> ModelSpecification:
        """Return the model specification."""
        fields = [
            ModelField(type="text", name="alias", description="Model alias", required=False),
        ]
        if options := self.instances_info[instance].config.options:
            options.spec.get("")
            if "hardware" not in options.spec:
                options.spec["hardware"] = options.spec.get("gpu", self.docker_service.has_gpu_support)
            parsed_options = try_parse_pydantic(VllmOptions, options.spec)
            hardware = parsed_options.hardware
            if (isinstance(hardware, bool) and hardware is True) or (isinstance(hardware, str) and hardware.startswith("GPU")):
                fields.append(
                    ModelField(
                        type="number",
                        name="gpu_memory_utilization",
                        description="Gpu usage (in range 0.00 - 1.00)",
                        required=False,
                        default="0.9",
                    ),
                )
        fields.extend(
            [
                ModelField(type="number", name="max_model_length", description="Max model length ex. 4096", required=False),
                ModelField(type="text", name="quantization", description="Quantization", required=False),
                ModelField(
                    type="map",
                    name="extra_args",
                    description="Additional vLLM arguments. Key: flag name (e.g. --dtype), value: flag value (empty for boolean flags).",
                    required=False,
                    placeholder="extra_args",
                    default='{"--trust-remote-code":""}' if model_type == "reranker" else None,
                ),
                ModelField(
                    type="map",
                    name="extra_envs",
                    description="Additional environment variables passed to the vLLM container.",
                    required=False,
                    placeholder="extra_envs",
                ),
            ]
        )

        return ModelSpecification(fields=fields)

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

    def is_given_hardware_support_gpu(self, hardware_specification: str | bool | None) -> bool:
        """Return is gpu will be used."""
        if not self.hardware.cpu.avx512 and not self._supported_gpus:
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
        if not self.hardware.cpu.avx512 and not self._supported_gpus:
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

    async def _install_instance(self, instance: str, options: InstallServiceIn) -> PromiseWithProgress[InstalledInfo, StreamChunk]:
        if not self.models.get(instance):
            self.load_default_models(instance)

        if "hardware" not in options.spec:
            options.spec["hardware"] = options.spec.get("gpu", self.docker_service.has_gpu_support)
        parsed_options = try_parse_pydantic(VllmOptions, options.spec)
        image = self._get_image(self.is_given_hardware_support_gpu(parsed_options.hardware))
        await self._verify_docker_image(image.name, options.ignore_warnings)

        async def func(stream: Stream[StreamChunk]) -> InstalledInfo:
            await self._download_image_or_set_progress(stream, image)
            self.service_downloaded = True
            return InstalledInfo(models={}, options=options, parsed_options=parsed_options)

        return PromiseWithProgress(func=func)

    async def _uninstall_instance(self, instance: str, options: UninstallServiceIn) -> None:
        installed = self.get_instance_info(instance).installed
        if installed:
            for model in installed.models.copy().values():
                if not self.is_model_installed_in_other_instance(instance, model.id):
                    await self._uninstall_model(instance, model.id, UninstallModelIn(purge=options.purge))

        self.instances_info[instance].installed = None

        if options.purge:
            if len(self.instances_info) < 2:
                self.service_downloaded = False
                for image in _const.images.values():
                    await self.docker_service.remove_image(image.name)
                await self._clear_working_dir()
                self.models_downloaded = {}

            if instance == "default":
                self.instances_info["default"] = Instance(None, None, {}, InstanceConfig())
            else:
                del self.instances_info[instance]

    def get_docker_compose_file_path(self, instance: str, model_id: str | None) -> Path:
        """Get docker compose file path."""
        info = self.get_instance_installed_info(instance)
        if not model_id:
            raise HTTPException(400, "Docker is not bound with this object")

        model_installed = info.models.get(model_id, None)
        if not model_installed:
            raise HTTPException(status_code=400, detail="Model not installed")

        return self.docker_service.get_docker_compose_file_path(model_installed.docker.name)

    def _add_custom_model(self, instance: str, model: CustomModel) -> None:
        parsed = try_parse_pydantic(VllmCustomModel, model.data)

        if not self.models.get(instance):
            self.models[instance] = {}

        if parsed.id in self.models[instance]:
            raise HTTPException(400, "Model with given id already exists.")

        self.models[instance][parsed.id] = VllmModel(hf_id=parsed.hf_id, size=parsed.size, custom=model.id, model_type=parsed.model_type)

    def _remove_custom_model(self, instance: str, model: CustomModel) -> None:
        installed = self.get_instance_info(instance).installed
        parsed = try_parse_pydantic(VllmCustomModel, model.data)
        if installed and parsed.id in installed.models:
            raise HTTPException(400, "Cannot remove custom model, it is in use, uninstall it first.")
        del self.models[instance][parsed.id]

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
                            type=model.model_type,
                            installed=installed,
                            downloaded=model_id in self.models_downloaded,
                            size=model.size,
                            custom=model.custom,
                            spec=self.get_model_spec(instance_name, model.model_type),
                            has_docker=True,
                        )
                    )

        return ListModelsOut(list=out_list)

    async def get_model(self, instance: str, model_id: str) -> RetrieveModelOut:
        """Get the model."""
        info = self.get_instance_installed_info(instance)

        if not self.models.get(instance):
            self.models[instance] = {}

        if model_id not in self.models[instance]:
            raise HTTPException(status_code=400, detail="Model not found")

        model = self.models[instance][model_id]
        installed = info.models[model_id].get_info() if model_id in info.models else self._get_model_installed_info(instance, model_id)
        return RetrieveModelOut(
            id=model_id,
            service=self.get_id(instance),
            type=model.model_type,
            installed=installed,
            downloaded=model_id in self.models_downloaded,
            size=model.size,
            custom=model.custom,
            spec=self.get_model_spec(instance, model.model_type),
            has_docker=True,
        )

    async def _download_model(self, stream: Stream[StreamChunk], model_id: str, model: VllmModel, model_dir: Path) -> Path | None:
        local_model_path: Path | None = None
        progress = Progress(convert_size_to_bytes(model.size) or 0)
        stream.emit(StreamChunkProgress(type="progress", stage="download", value=0, data={}))
        async for packet in self.model_downloader.download(model_id, model_dir):
            if isinstance(packet, DownloadedPacket) and packet.downloaded_bytes_size != 0:
                progress.add_to_actual_value(packet.downloaded_bytes_size)
                stream.emit(StreamChunkProgress(type="progress", stage="download", value=progress.get_percentage(), data={}))
            elif isinstance(packet, PreDownloadPacket):
                if max := packet.file_bytes_size:
                    progress.set_max_value(max)
            elif isinstance(packet, SuccessDownloadPacket):
                local_model_path = packet.local_path

        stream.emit(StreamChunkProgress(type="progress", stage="download", value=1, data={"local_model_path": str(local_model_path)}))
        if local_model_path is None:
            raise HTTPException(500, f"Downloader doesn't return filepath in {model_dir} of {model_id} model")
        return local_model_path

    async def _download_model_or_set_progress(self, stream: Stream[StreamChunk], model_id: str, model: VllmModel, model_dir: Path) -> Path:
        local_model_path: Path | None = None
        if model_id not in self.models_download_progress:
            self.models_download_progress[model_id] = stream
            local_model_path = await self._download_model(stream, model_id, model, model_dir)
            del self.models_download_progress[model_id]
        else:
            chunk: StreamChunk
            async for chunk in self.models_download_progress[model_id].as_generator():
                if chunk.get("type") == "progress" and chunk.get("stage") == "download":
                    if data := chunk.get("data"):
                        local_model_path = Path(data.get("local_model_path", local_model_path) or "")
                    stream.emit(chunk)
                else:
                    break
        if local_model_path is None:
            raise HTTPException(500, f"Downloader doesn't return filepath in {model_dir} of {model_id} model")
        return local_model_path

    async def _get_gpu_memory_utilization(self, parsed_model_options: VllmModelOptions, model: VllmModel) -> float:
        gpu_memory_utilization = parsed_model_options.gpu_memory_utilization or model.gpu_memory_utilization or 0.95
        if gpu_memory_utilization > 1 and gpu_memory_utilization <= 0:
            raise HTTPException(422, "GPU utilization needs to be in range 0-1.")
        if self.gpu_memory_utilization + gpu_memory_utilization > 1:
            raise HTTPException(
                422,
                (
                    "GPU utilization in all vllm instances can be maximum 1. "
                    f"Actual GPU utilization is {self.gpu_memory_utilization}. "
                    f"You try add additional {gpu_memory_utilization}, "
                    f"which is {self.gpu_memory_utilization + gpu_memory_utilization - 1} more that 1."
                ),
            )
        self.gpu_memory_utilization += gpu_memory_utilization
        msg = f"VLLM gpu utilization = {self.gpu_memory_utilization}"
        logger.debug(msg)
        return gpu_memory_utilization

    async def _get_quantization(self, parsed_model_options: VllmModelOptions, model: VllmModel) -> RegistrationId | None:
        quantization = parsed_model_options.quantization or model.quantization or None
        if quantization:
            pattern = r"^[a-zA-Z0-9_-]+$"
            if not re.match(pattern, quantization):
                raise HTTPException(422, "Quantization options needs to be made from `a-zA-Z-_`.")
        return quantization

    async def _get_max_model_length(self, parsed_model_options: VllmModelOptions, model: VllmModel) -> None | int:
        max_model_length = parsed_model_options.max_model_length or model.max_model_len or None
        if max_model_length and not max_model_length.is_integer():
            raise HTTPException(422, "Max model length need to be integer.")
        return max_model_length

    def _build_vllm_command(
        self,
        docker_model_path: Path,
        model_id: str,
        opts: VllmModelOptions,
        quantization: str | None,
        gpu_memory_utilization: float | None,
        user_model_length: int | None,
        use_gpu: bool,
    ) -> list[str]:
        cmd = [
            str(docker_model_path),
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--served-model-name",
            model_id,
        ]

        kv_flags: list[tuple[str, str | None]] = [
            ("--quantization", quantization),
            ("--gpu-memory-utilization", str(gpu_memory_utilization) if gpu_memory_utilization is not None else None),
        ]
        for flag, value in kv_flags:
            if value:
                cmd.extend([flag, value])

        if use_gpu and user_model_length:
            cmd.extend(["--max-model-len", str(user_model_length)])
        elif not use_gpu:
            cmd.extend(["--max-model-len", str(user_model_length)] if user_model_length else ["--disable-sliding-window"])

        for flag, value in opts.extra_args.items():
            if value:
                cmd.extend([flag, value])
            else:
                cmd.append(flag)

        return cmd

    def _register_model_endpoint(
        self,
        model_info: "ModelInstalledInfo",
        model: VllmModel,
        registered_name: str,
        model_id: str,
        context_window: int | None,
        max_context_window: int | None,
    ) -> RegistrationId:
        """Register the model in the endpoint registry based on its type."""
        if model.model_type == "reranker":
            return self.endpoint_registry.register_rerank_as_proxy(
                model=registered_name,
                props=ModelProps(private=True, type="rerank", endpoints=["/v1/rerank"]),
                options=ProxyOptions(url=f"{model_info.base_url}/v1/rerank", rewrite_model_to=model_id),
                registration_options=None,
            )
        return self.endpoint_registry.register_chat_completion_as_proxy(
            model=registered_name,
            props=ModelProps(
                private=True,
                type="llm",
                endpoints=LLM_ENDPOINTS,
                context_window=context_window,
                max_context_window=max_context_window,
            ),
            chat_completions=ProxyOptions(url=f"{model_info.base_url}/v1/chat/completions", rewrite_model_to=model_id),
            completions=ProxyOptions(url=f"{model_info.base_url}/v1/completions", rewrite_model_to=model_id),
            responses=ProxyOptions(url=f"{model_info.base_url}/v1/responses", rewrite_model_to=model_id),
            messages=ProxyOptions(url=f"{model_info.base_url}/v1/messages", rewrite_model_to=model_id),
            registration_options=None,
        )

    async def _install_model(
        self, instance: str, model_id: str, options: InstallModelIn
    ) -> PromiseWithProgress[InstallModelOut, StreamChunk]:
        parsed_model_options = try_parse_pydantic(VllmModelOptions, options.spec) if options.spec else VllmModelOptions()

        info = self.get_instance_installed_info(instance)

        if model_id in info.models:
            return PromiseWithProgress(value=InstallModelOut(status="OK", details="Already installed"))

        if model_id not in self.models[instance]:
            raise HTTPException(400, "Model not found")

        model = self.models[instance][model_id]

        use_gpu = self.is_given_hardware_support_gpu(info.parsed_options.hardware)
        gpu_memory_utilization = await self._get_gpu_memory_utilization(parsed_model_options, model) if use_gpu else None
        user_model_length = await self._get_max_model_length(parsed_model_options, model)
        quantization = await self._get_quantization(parsed_model_options, model)

        async def func(stream: Stream[StreamChunk]) -> InstallModelOut:
            model_id_fixed = model_id.replace("/", "-")
            models_dir = self._get_working_dir() / "models"
            model_dir = models_dir / model_id_fixed
            model_dir.mkdir(parents=True, exist_ok=True)
            local_model_path: Path | None = await self._download_model_or_set_progress(stream, model_id, model, model_dir)

            stream.emit(StreamChunkProgress(type="progress", stage="install", value=0, data={}))
            docker_model_path = Path(self.hugging_face_cache_path) / "hub" / model_id_fixed
            volumes = [f"{local_model_path}:{docker_model_path}"]

            vllm_command = self._build_vllm_command(
                docker_model_path=docker_model_path,
                model_id=model_id,
                opts=parsed_model_options,
                quantization=quantization,
                gpu_memory_utilization=gpu_memory_utilization,
                user_model_length=user_model_length,
                use_gpu=use_gpu,
            )

            if not model.env_vars:
                model.env_vars = {}

            model.env_vars["HF_HUB_OFFLINE"] = "1"
            model.env_vars["HF_HOME"] = self.hugging_face_cache_path
            if not use_gpu:
                model.env_vars["VLLM_USE_V1"] = "1"
            model.env_vars.update(parsed_model_options.extra_envs)

            image = self._get_image(use_gpu)

            subnet = self.docker_service.get_docker_subnet()
            service_name = f"{self.get_service_id(instance)}-{normalize_name(model_id)}"
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
                    "interval": "80s",
                    "timeout": "60s",
                    "retries": "3",
                    "start_period": "240s",
                },
            )
            try:
                docker_exposed_port = await self.docker_service.install_and_run_docker(docker_options)
            except RuntimeError:
                if model.gpu_memory_utilization:
                    self.gpu_memory_utilization -= model.gpu_memory_utilization
                    if self.gpu_memory_utilization < 0:
                        self.gpu_memory_utilization = 0
                    msg = f"VLLM gpu utilization = {self.gpu_memory_utilization}"
                    logger.debug(msg)
                await self.docker_service.stop_docker(docker_options)
                raise
            registered_name = parsed_model_options.alias if parsed_model_options.alias else model_id
            container_host = self.docker_service.get_container_host(subnet, docker_options.name)
            container_port = self.docker_service.get_container_port(subnet, docker_exposed_port, docker_options.image_port)
            if model.model_type == "llm":
                max_context_window = await get_model_dir_context_window(local_model_path)
                context_window = user_model_length or max_context_window
            else:
                max_context_window = None
                context_window = None
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
                gpu_memory_utilization=gpu_memory_utilization,
                model_type=model.model_type,
            )
            model_info.registration_id = self._register_model_endpoint(
                model_info=model_info,
                model=model,
                registered_name=registered_name,
                model_id=model_id,
                context_window=context_window,
                max_context_window=max_context_window,
            )
            stream.emit(StreamChunkProgress(type="progress", stage="install", value=1, data={}))
            self.models_downloaded[model_id] = DownloadedInfo(str(local_model_path))
            return InstallModelOut(status="OK", details="Installed")

        return PromiseWithProgress(func=func)

    def _get_image(self, gpu: bool) -> DockerImage:
        return _const.images["gpu"] if gpu else _const.images["cpu"]

    async def _uninstall_model(self, instance: str, model_id: str, options: UninstallModelIn) -> None:
        info = self.get_instance_installed_info(instance)

        if model_id in info.models:
            model = info.models[model_id]
            if model.gpu_memory_utilization:
                self.gpu_memory_utilization -= model.gpu_memory_utilization
                if self.gpu_memory_utilization < 0:
                    self.gpu_memory_utilization = 0
                msg = f"VLLM gpu utilization = {self.gpu_memory_utilization}"
                logger.debug(msg)
            del info.models[model_id]
            if model.model_type == "reranker":
                self.endpoint_registry.unregister_rerank(model.registered_name, model.registration_id)
            else:
                self.endpoint_registry.unregister_chat_completion(model.registered_name, model.registration_id)
            await self.docker_service.uninstall_docker(model.docker)

        if options.purge and model_id in self.models_downloaded:
            model = self.models_downloaded[model_id]
            if model_path := model.model_path:
                shutil.rmtree(Path(model_path))
            del self.models_downloaded[model_id]

    async def stop_instance(self, instance: str) -> None:
        """Stop all the vLLM service Docker containers."""
        installed = self.get_instance_info(instance).installed
        if not installed:
            return
        await self._stop_dockers_parallel([model.docker for model in installed.models.values()])
