# DeepFellow Software Framework.
# Copyright © 2025 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Llamacpp service."""

from collections.abc import AsyncGenerator
from pathlib import Path

from fastapi import HTTPException
from pydantic import BaseModel, Field

from server.applicationcontext import get_base_url, get_container_host, get_container_port
from server.docker import DockerImage, DockerOptions, docker_pull, has_gpu_support_sync, install_and_run_docker, uninstall_docker
from server.endpointregistry import ProxyOptions, RegistrationId
from server.models.api import ModelProps
from server.models.models import (
    CustomModelField,
    CustomModelId,
    CustomModelSpecification,
    InstallModelIn,
    ListModelsFilters,
    ListModelsOut,
    ModelField,
    ModelSpecification,
    RetrieveModelOut,
    UninstallModelIn,
)
from server.models.services import InstallServiceIn, ServiceField, ServiceOptions, ServiceSize, ServiceSpecification, UninstallServiceIn
from server.services.base2_service import Base2Service, CustomModel, ModelConfig, ServiceConfig
from server.utils.core import (
    StreamChunk,
    StreamChunkFinish,
    StreamChunkInstalledInfo,
    StreamChunkProgress,
    StreamingError,
    convert_size_to_bytes,
    normalize_name,
    try_parse_pydantic,
)
from server.utils.loading import Progress


class LlamacppModel(BaseModel):
    url: str
    size: str
    custom: CustomModelId | None = None


class LlamacppCustomModel(BaseModel):
    id: str
    url: str
    size: str


class LlamacppConst(BaseModel):
    image_gpu: DockerImage
    image_cpu: DockerImage
    model_type: str
    models: dict[str, LlamacppModel]


_const = LlamacppConst(
    image_gpu=DockerImage(name="ghcr.io/ggml-org/llama.cpp:server-cuda-b6620", size="2.6 GB"),
    image_cpu=DockerImage(name="ghcr.io/ggml-org/llama.cpp:server-b6617", size="0.1 GB"),
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
        "google/gemma-2b": LlamacppModel(
            url="https://huggingface.co/google/gemma-2b/resolve/main/gemma-2b.gguf",
            size="10.0GB",
        ),
    },
)


class ModelInstalledInfo:
    def __init__(
        self,
        id: str,
        registered_name: str,
        options: InstallModelIn,
        docker: DockerOptions,
        model_path: Path,
        container_host: str,
        container_port: int,
        docker_exposed_port: int,
        registration_id: RegistrationId,
    ):
        self.id = id
        self.registered_name = registered_name
        self.options = options
        self.docker = docker
        self.model_path = model_path
        self.container_host = container_host
        self.container_port = container_port
        self.docker_exposed_port = docker_exposed_port
        self.base_url = get_base_url(self.container_host, self.container_port)
        self.registration_id = registration_id


class LLamacppOptions(BaseModel):
    gpu: bool = Field(default_factory=lambda: has_gpu_support_sync())


class LLamacppModelOptions(BaseModel):
    alias: str | None = None


class InstalledInfo:
    def __init__(
        self,
        models: dict[str, ModelInstalledInfo],
        options: InstallServiceIn,
        parsed_options: LLamacppOptions,
    ):
        self.models = models
        self.options = options
        self.parsed_options = parsed_options


class LLamacppService(Base2Service[InstalledInfo]):
    models: dict[str, LlamacppModel]

    def _after_init(self) -> None:
        self.models = _const.models.copy()

    def get_id(self) -> str:
        """Return the service id."""
        return "llamacpp"

    def get_description(self) -> str:
        """Return the service description."""
        return "Self-hosted LLM efficient model runner with advanced options."

    def get_size(self) -> ServiceSize:
        """Return the service size."""
        return {"cpu": _const.image_cpu.size, "gpu": _const.image_gpu.size}

    def get_spec(self) -> ServiceSpecification:
        """Return the service specification."""
        return ServiceSpecification(
            fields=[
                ServiceField(type="bool", name="gpu", description="Run on GPU", required=False, default=self._has_gpu_for_spec()),
            ]
        )

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

    def get_installed_info(self) -> bool | ServiceOptions:
        """Get service installed info."""
        return False if self.installed is None else self.installed.options.spec

    def _generate_config(self, info: InstalledInfo | None) -> ServiceConfig:
        return ServiceConfig(
            options=info.options if info else None,
            models=[ModelConfig(model_id=x.id, options=x.options) for x in info.models.values()] if info else [],
            custom=self.custom,
        )

    async def _install_core(self, options: InstallServiceIn) -> AsyncGenerator[StreamChunk]:
        parsed_options = try_parse_pydantic(LLamacppOptions, options.spec)
        image = self._get_image(parsed_options.gpu)
        async for progress in docker_pull(image.name, convert_size_to_bytes(image.size) or 0):
            yield StreamChunkProgress(type="progress", value=progress * 0.99)

        info = InstalledInfo(models={}, options=options, parsed_options=parsed_options)

        yield StreamChunkProgress(type="progress", value=1)
        yield StreamChunkFinish(type="finish", status="ok", details="installed")
        yield StreamChunkInstalledInfo(type="installed_info", status="ok", details=info)

    async def _uninstall(self, options: UninstallServiceIn) -> None:
        info = self._check_installed()
        for model in info.models.copy().values():
            await self._uninstall_model(model.id, UninstallModelIn(purge=options.purge))
        self.installed = None
        if options.purge:
            await self._clear_working_dir()

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
        return self.application_context.get_docker_compose_file_path(installed.docker.name)

    def _add_custom_model(self, model: CustomModel) -> None:
        parsed = try_parse_pydantic(LlamacppCustomModel, model.data)
        if parsed.id in self.models:
            raise HTTPException(400, "Model with given id already exists.")
        self.models[parsed.id] = LlamacppModel(url=parsed.url, size=parsed.size, custom=model.id)

    def _remove_custom_model(self, model: CustomModel) -> None:
        parsed = try_parse_pydantic(LlamacppCustomModel, model.data)
        if self.installed and parsed.id in self.installed.models:
            raise HTTPException(400, "Cannot remove custom model, it is in use, uninstall it first.")
        del self.models[parsed.id]

    async def list_models(self, filters: ListModelsFilters) -> ListModelsOut:
        """List models."""
        info = self._check_installed()
        out_list: list[RetrieveModelOut] = []
        for model_id, model in self.models.items():
            installed = info.models[model_id].options if model_id in info.models else False
            if filters.installed is None or filters.installed == installed:
                out_list.append(
                    RetrieveModelOut(
                        id=model_id,
                        service=self.get_id(),
                        type=_const.model_type,
                        installed=installed,
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
        installed = info.models[model_id].options if model_id in info.models else False
        return RetrieveModelOut(
            id=model_id,
            service=self.get_id(),
            type=_const.model_type,
            installed=installed,
            size=model.size,
            custom=model.custom,
            spec=self.get_model_spec(),
            has_docker=True,
        )

    async def _install_model(self, model_id: str, options: InstallModelIn) -> AsyncGenerator[StreamChunk]:
        parsed_model_options = try_parse_pydantic(LLamacppModelOptions, options.spec) if options.spec else LLamacppModelOptions()
        info = self._check_installed()
        if model_id in info.models:
            yield StreamChunkFinish(type="finish", status="ok", details="Already installed")
            return
        if model_id not in self.models:
            raise StreamingError("Model not found")
        model = self.models[model_id]
        model_dir = self._get_working_dir() / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        progress = Progress(convert_size_to_bytes(model.size) or 0)
        local_model_path: Path | None = None
        model_filename: str = ""
        async for packet in self.model_downloader.download(model.url, model_dir):
            if packet.local_path and packet.filename:
                local_model_path = packet.local_path
                model_filename = packet.filename
            elif packet.downloaded_bytes_size != 0:
                progress.add_to_actual_value(packet.downloaded_bytes_size)

            yield StreamChunkProgress(type="progress", value=progress.get_percentage() * 0.99)

        if not local_model_path or not model_filename:
            raise StreamingError("Local model path was not set up and not return by downloader.")
        if not model_filename:
            raise StreamingError("Model filename was not set up or and return by downloader.")

        model_in_container = f"/models/{model_filename}"
        volumes = [f"{local_model_path.absolute()}:{model_in_container}:ro"]
        image = self._get_image(info.parsed_options.gpu)
        command = " ".join(["--host 0.0.0.0", "--port 8080", f"--model {model_in_container}"])
        subnet = self.application_context.get_docker_subnet()
        service_name = f"{self.get_id()}-{normalize_name(model_id)}"
        docker_options = DockerOptions(
            name=service_name,
            container_name=self.application_context.get_docker_container_name(service_name),
            image=image.name,
            command=command,
            image_port=8080,
            restart="unless-stopped",
            volumes=volumes,
            use_gpu=info.parsed_options.gpu,
            subnet=subnet,
        )
        docker_exposed_port = await install_and_run_docker(self.application_context, docker_options)
        registered_name = parsed_model_options.alias if parsed_model_options.alias else model_id
        info.models[model_id] = model_info = ModelInstalledInfo(
            id=model_id,
            registered_name=registered_name,
            options=options,
            docker=docker_options,
            model_path=local_model_path.absolute(),
            container_host=get_container_host(subnet, docker_options.name),
            container_port=get_container_port(subnet, docker_exposed_port, docker_options.image_port),
            docker_exposed_port=docker_exposed_port,
            registration_id="",
        )
        model_info.registration_id = self.endpoint_registry.register_chat_completion_as_proxy(
            model=registered_name,
            props=ModelProps(private=True),
            chat_completions=ProxyOptions(url=f"{model_info.base_url}/v1/chat/completions", rewrite_model_to=model_id),
            completions=ProxyOptions(url=f"{model_info.base_url}/v1/completions", rewrite_model_to=model_id),
            registration_options=None,
        )

        yield StreamChunkProgress(type="progress", value=1)
        yield StreamChunkFinish(type="finish", status="ok", details="Installed")

    def _get_image(self, gpu: bool) -> DockerImage:
        return _const.image_gpu if gpu else _const.image_cpu

    async def _uninstall_model(self, model_id: str, options: UninstallModelIn) -> None:
        info = self._check_installed()
        if model_id not in info.models:
            return
        model = info.models[model_id]
        del info.models[model_id]
        self.endpoint_registry.unregister_chat_completion(model.registered_name, model.registration_id)
        await uninstall_docker(self.application_context, model.docker)
        if options.purge:
            model.model_path.unlink()
