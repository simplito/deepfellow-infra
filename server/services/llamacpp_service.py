# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Llamacpp service."""

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
from server.utils.hardware import HardwarePartInfo, IntelGpuInfo, NvidiaGpuInfo
from server.utils.loading import Progress


class LlamacppModel(BaseModel):
    url: str
    size: str
    custom: CustomModelId | None = None
    jinja: bool = False


class LlamacppCustomModel(BaseModel):
    id: str
    url: str
    size: str


type ImageTypes = Literal["cpu", "gpu", "vulkan"]


class LlamacppConst(BaseModel):
    images: dict[ImageTypes, DockerImage]
    model_type: str
    models: dict[str, LlamacppModel]


_const = LlamacppConst(
    images={
        "gpu": DockerImage(name="ghcr.io/ggml-org/llama.cpp:server-cuda-b7836", size="2.8 GB"),
        "vulkan": DockerImage(name="ghcr.io/ggml-org/llama.cpp:server-vulkan-b7836", size="0.1 GB"),
        "cpu": DockerImage(name="ghcr.io/ggml-org/llama.cpp:server-b7836", size="0.1 GB"),
    },
    model_type="llm",
    models={
        "bartowski/mistral-community_pixtral-12b": LlamacppModel(
            url="https://huggingface.co/bartowski/mistral-community_pixtral-12b-GGUF/resolve/main/mistral-community_pixtral-12b-Q5_K_M.gguf",
            size="8.2GB",
        ),
        "bartowski/deepseek-ai_DeepSeek-R1-0528-Qwen3-8B": LlamacppModel(
            url="https://huggingface.co/bartowski/deepseek-ai_DeepSeek-R1-0528-Qwen3-8B-GGUF/resolve/main/deepseek-ai_DeepSeek-R1-0528-Qwen3-8B-Q5_K_M.gguf",
            size="5.5GB",
        ),
        "bartowski/Hermes-3-Llama-3.2-3B": LlamacppModel(
            url="https://huggingface.co/bartowski/Hermes-3-Llama-3.2-3B-GGUF/resolve/main/Hermes-3-Llama-3.2-3B-Q5_K_M.gguf",
            size="2.2GB",
        ),
        "bartowski/Meta-Llama-3.1-8B-Instruct": LlamacppModel(
            url="https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
            size="4.6GB",
        ),
        "mradermacher/PLLuM-12B-instruct": LlamacppModel(
            url="https://huggingface.co/mradermacher/PLLuM-12B-instruct-GGUF/resolve/main/PLLuM-12B-instruct.Q4_K_M.gguf",
            size="7.0GB",
        ),
        "google/gemma-3-1b-it-q8": LlamacppModel(
            url="https://huggingface.co/brittlewis12/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it.Q8_0.gguf",
            size="2.0GB",
        ),
        "google/gemma-3-1b-it-q4-k-m": LlamacppModel(
            url="https://huggingface.co/bartowski/google_gemma-3-1b-it-GGUF/resolve/main/google_gemma-3-1b-it-Q4_K_M.gguf",
            size="0.9GB",
        ),
        "lmstudio-community/gemma-3-270m-it-16f": LlamacppModel(
            url="https://huggingface.co/lmstudio-community/gemma-3-270m-it-GGUF/resolve/main/gemma-3-270m-it-F16.gguf",
            size="0.6GB",
        ),
        "google/gemma-2b": LlamacppModel(
            url="https://huggingface.co/google/gemma-2b/resolve/main/gemma-2b.gguf",
            size="10.0GB",
        ),
        "speakleash/Bielik-11B-v2.5-Instruct": LlamacppModel(
            url="https://huggingface.co/speakleash/Bielik-11B-v2.5-Instruct-GGUF/resolve/main/Bielik-11B-v2.5-Instruct.Q4_K_M.gguf",
            size="6.7GB",
            jinja=True,
        ),
        "speakleash/Bielik-11B-v2.6-Instruct": LlamacppModel(
            url="https://huggingface.co/speakleash/Bielik-11B-v2.6-Instruct-GGUF/resolve/main/Bielik-11B-v2.6-Instruct.Q4_K_M.gguf",
            size="6.7GB",
            jinja=True,
        ),
        "speakleash/Bielik-11B-v3.0-Instruct": LlamacppModel(
            url="https://huggingface.co/speakleash/Bielik-11B-v3.0-Instruct-GGUF/resolve/main/Bielik-11B-v3.0-Instruct.Q4_K_M.gguf",
            size="6.7GB",
            jinja=True,
        ),
    },
)


@dataclass
class ModelInstalledInfo:
    id: str
    registered_name: str
    options: InstallModelIn
    docker: DockerOptions
    model_path: Path
    container_host: str
    container_port: int
    docker_exposed_port: int
    registration_id: RegistrationId
    base_url: str

    def get_info(self) -> ModelInfo:
        """Get info."""
        return ModelInfo(spec=self.options.spec, registration_id=self.registration_id)


class LLamacppOptions(BaseModel):
    hardware: str | bool | None = None


class LLamacppModelOptions(BaseModel):
    alias: str | None = None


@dataclass
class InstalledInfo:
    models: dict[str, ModelInstalledInfo]
    options: InstallServiceIn
    parsed_options: LLamacppOptions


@dataclass
class DownloadedInfo:
    model_path: str


class LLamacppService(Base2Service[InstalledInfo, DownloadedInfo]):
    models: dict[str, dict[str, LlamacppModel]]

    def _after_init(self) -> None:
        self.models = {}
        self.load_default_models("default")

    def load_default_models(self, instance: str) -> None:
        """Load default models to instance."""
        self.models[instance] = _const.models.copy()

    def get_type(self) -> str:
        """Return the service id."""
        return "llamacpp"

    def get_description(self) -> str:
        """Return the service description."""
        return "Self-hosted LLM efficient model runner with advanced options."

    def get_size(self) -> ServiceSize:
        """Return the service size."""
        sizes = {"cpu": _const.images["cpu"].size}
        if self.hardware.nvidia_gpus:
            sizes["gpu"] = _const.images["gpu"].size
        if self.hardware.intel_gpus:
            sizes["vulkan"] = _const.images["vulkan"].size
        return sizes

    def get_spec(self) -> ServiceSpecification:
        """Return the service specification."""
        fields = self.add_hardware_field_to_spec()
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
                CustomModelField(
                    type="text", name="url", description="Model URL (gguf)", placeholder="https://model.registry.com/my-model.gguf"
                ),
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

    def _load_download_info(self, data: dict[str, Any]) -> DownloadedInfo:
        return DownloadedInfo(**data)

    async def _install_instance(self, instance: str, options: InstallServiceIn) -> PromiseWithProgress[InstalledInfo, StreamChunk]:
        if not self.models.get(instance):
            self.load_default_models(instance)

        if "hardware" not in options.spec:
            options.spec["hardware"] = options.spec.get("gpu", self.docker_service.has_gpu_support)
        parsed_options = try_parse_pydantic(LLamacppOptions, options.spec)
        hardware_parts = self.get_specified_hardware_parts(parsed_options.hardware)
        image = self._get_image(hardware_parts)
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
        parsed = try_parse_pydantic(LlamacppCustomModel, model.data)

        if not self.models.get(instance):
            self.models[instance] = {}

        if parsed.id in self.models[instance]:
            raise HTTPException(400, "Model with given id already exists.")
        self.models[instance][parsed.id] = LlamacppModel(url=parsed.url, size=parsed.size, custom=model.id)

    def _remove_custom_model(self, instance: str, model: CustomModel) -> None:
        installed = self.get_instance_info(instance).installed
        parsed = try_parse_pydantic(LlamacppCustomModel, model.data)
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

    async def get_model(self, instance: str, model_id: str) -> RetrieveModelOut:
        """Get the model."""
        info = self.get_instance_installed_info(instance)
        if model_id not in self.models[instance]:
            raise HTTPException(status_code=400, detail="Model not found")
        if not self.models.get(instance):
            self.models[instance] = {}

        model = self.models[instance][model_id]
        installed = info.models[model_id].get_info() if model_id in info.models else self._get_model_installed_info(instance, model_id)
        return RetrieveModelOut(
            id=model_id,
            service=self.get_id(instance),
            type=_const.model_type,
            installed=installed,
            downloaded=model_id in self.models_downloaded,
            size=model.size,
            custom=model.custom,
            spec=self.get_model_spec(),
            has_docker=True,
        )

    async def _download_model(self, stream: Stream[StreamChunk], model: LlamacppModel) -> tuple[Path | None, str]:
        model_dir = self._get_working_dir() / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        progress = Progress(convert_size_to_bytes(model.size) or 0)
        local_model_path: Path | None = None
        filename: str = ""
        stream.emit(StreamChunkProgress(type="progress", stage="download", value=0, data={}))
        async for packet in self.model_downloader.download(model.url, model_dir):
            if isinstance(packet, DownloadedPacket) and packet.downloaded_bytes_size != 0:
                progress.add_to_actual_value(packet.downloaded_bytes_size)
                stream.emit(StreamChunkProgress(type="progress", stage="download", value=progress.get_percentage(), data={}))
            elif isinstance(packet, PreDownloadPacket):
                if max := packet.file_bytes_size:
                    progress.set_max_value(max)
            elif isinstance(packet, SuccessDownloadPacket):
                local_model_path = packet.local_path
                filename = packet.filename

        stream.emit(
            StreamChunkProgress(
                type="progress", stage="download", value=0, data={"local_model_path": str(local_model_path), "filename": filename}
            )
        )
        return local_model_path, filename

    async def _download_model_or_set_progress(
        self, stream: Stream[StreamChunk], model: LlamacppModel, model_id: str
    ) -> tuple[Path | None, str]:
        local_model_path: Path | None = None
        filename: str = ""
        if model_id not in self.models_download_progress:
            self.models_download_progress[model_id] = stream
            local_model_path, filename = await self._download_model(stream, model)
            del self.models_download_progress[model_id]
        else:
            chunk: StreamChunk
            async for chunk in self.models_download_progress[model_id].as_generator():
                if chunk.get("type") == "progress" and chunk.get("stage") == "download":
                    if data := chunk.get("data"):
                        local_model_path = Path(data.get("local_model_path", local_model_path) or "")
                        filename = str(data.get("filename", filename))
                    stream.emit(chunk)
                else:
                    break

        return local_model_path, filename

    async def _install_model(
        self, instance: str, model_id: str, options: InstallModelIn
    ) -> PromiseWithProgress[InstallModelOut, StreamChunk]:
        parsed_model_options = try_parse_pydantic(LLamacppModelOptions, options.spec) if options.spec else LLamacppModelOptions()
        installed = self.get_instance_installed_info(instance)
        if model_id in installed.models:
            return PromiseWithProgress(value=InstallModelOut(status="OK", details="Already installed"))
        if model_id not in self.models[instance]:
            raise HTTPException(400, "Model not found")
        if not self.models.get(instance):
            self.models[instance] = {}
        model = self.models[instance][model_id]

        async def func(stream: Stream[StreamChunk]) -> InstallModelOut:
            local_model_path, model_filename = await self._download_model_or_set_progress(stream, model, model_id)

            stream.emit(StreamChunkProgress(type="progress", stage="download", value=1, data={}))

            stream.emit(StreamChunkProgress(type="progress", stage="install", value=0, data={}))
            if not local_model_path or not model_filename:
                raise HTTPException(400, "Local model path was not set up and not return by downloader.")
            if not model_filename:
                raise HTTPException(400, "Model filename was not set up or and return by downloader.")

            model_in_container = f"/models/{model_filename}"
            volumes = [f"{local_model_path.absolute()}:{model_in_container}:ro"]
            command_options = ["--host 0.0.0.0", "--port 8080", f"--model {model_in_container}"]
            if model.jinja:
                command_options.append("--jinja")
            command = " ".join(command_options)
            subnet = self.docker_service.get_docker_subnet()
            service_name = f"{self.get_id(instance)}-{normalize_name(model_id)}"
            hardware_parts = self.get_specified_hardware_parts(installed.parsed_options.hardware)
            image = self._get_image(hardware_parts)
            docker_options = DockerOptions(
                name=service_name,
                container_name=self.docker_service.get_docker_container_name(service_name),
                image=image.name,
                command=command,
                image_port=8080,
                restart="unless-stopped",
                volumes=volumes,
                hardware=hardware_parts,
                subnet=subnet,
            )
            docker_exposed_port = await self.docker_service.install_and_run_docker(docker_options)
            registered_name = parsed_model_options.alias if parsed_model_options.alias else model_id
            container_host = self.docker_service.get_container_host(subnet, docker_options.name)
            container_port = self.docker_service.get_container_port(subnet, docker_exposed_port, docker_options.image_port)
            installed.models[model_id] = model_info = ModelInstalledInfo(
                id=model_id,
                registered_name=registered_name,
                options=options,
                docker=docker_options,
                model_path=local_model_path.absolute(),
                container_host=container_host,
                container_port=container_port,
                docker_exposed_port=docker_exposed_port,
                registration_id="",
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
            stream.emit(StreamChunkProgress(type="progress", stage="install", value=1, data={}))
            self.models_downloaded[model_id] = DownloadedInfo(str(local_model_path))
            return InstallModelOut(status="OK", details="Installed")

        return PromiseWithProgress(func=func)

    def _get_image(self, hardware: Sequence[HardwarePartInfo]) -> DockerImage:
        if any(isinstance(h, NvidiaGpuInfo) for h in hardware):
            return _const.images["gpu"]
        if any(isinstance(h, IntelGpuInfo) for h in hardware):
            return _const.images["vulkan"]
        return _const.images["cpu"]

    async def _uninstall_model(self, instance: str, model_id: str, options: UninstallModelIn) -> None:
        info = self.get_instance_installed_info(instance)
        if model_id in info.models:
            model = info.models[model_id]
            del info.models[model_id]
            self.endpoint_registry.unregister_chat_completion(model.registered_name, model.registration_id)
            await self.docker_service.uninstall_docker(model.docker)

        if options.purge and model_id in self.models_downloaded:
            if self.models_downloaded[model_id].model_path:
                Path(self.models_downloaded[model_id].model_path).unlink()
            del self.models_downloaded[model_id]

    async def stop_instance(self, instance: str) -> None:
        """Stop all the Llamacpp service Docker containers."""
        installed = self.get_instance_info(instance).installed
        if not installed:
            return
        await self._stop_dockers_parallel([model.docker for model in installed.models.values()])
