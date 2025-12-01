# DeepFellow Software Framework.
# Copyright © 2025 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ollama service."""

import json
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from fastapi import HTTPException
from pydantic import BaseModel, StringConstraints

from server.applicationcontext import get_base_url
from server.config import get_main_dir
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
    ModelSpecification,
    RetrieveModelOut,
    UninstallModelIn,
)
from server.models.services import InstallServiceIn, ServiceField, ServiceOptions, ServiceSpecification, UninstallServiceIn
from server.services.base2_service import Base2Service, CustomModel, ModelConfig, ServiceConfig
from server.utils.core import (
    PromiseWithProgress,
    Stream,
    StreamChunk,
    StreamChunkProgress,
    convert_size_to_bytes,
    fetch_from,
    stream_fetch_from,
    try_parse_pydantic,
)
from server.utils.loading import Progress


def _read_models_from_json() -> dict[str, bool]:  # pyright: ignore[reportUnusedFunction]
    ollama_path = get_main_dir() / "./static/ollama.json"
    with ollama_path.open(encoding="utf-8") as f:
        data = json.loads(f.read())
        # tags = [x["tags"] for x in data["list"]]
        # flat: list[str] = [x["tag"] for sublist in tags for x in sublist]
        tags = [x["mainTags"] for x in data["list"]]
        flat: list[str] = [x.removesuffix(":latest") for sublist in tags for x in sublist]

        map: dict[str, bool] = {}
        for tag in flat:
            map[tag] = True
        return map


class OllamaModel(BaseModel):
    id: str
    size: str
    type: str
    custom: CustomModelId | None = None


class OllamaCustomModel(BaseModel):
    id: str
    size: str
    type: Literal["llm", "embedding"]


class OllamaRegistryEntry(TypedDict):
    name: str
    size: str


class OllamaRegistry(TypedDict):
    llms: list[OllamaRegistryEntry]
    embeddings: list[OllamaRegistryEntry]


def _read_models() -> dict[str, OllamaModel]:
    ollama_path = get_main_dir() / "./static/ollama-min.json"
    with ollama_path.open(encoding="utf-8") as f:
        registry: OllamaRegistry = json.loads(f.read())
        map: dict[str, OllamaModel] = {}
        for tag in registry["llms"]:
            map[tag["name"]] = OllamaModel(id=tag["name"], size=tag["size"], type="llm")
        for tag in registry["embeddings"]:
            map[tag["name"]] = OllamaModel(id=tag["name"], size=tag["size"], type="embedding")
        return map


class OllamaAiConst(BaseModel):
    image: DockerImage
    models: dict[str, OllamaModel]


_const = OllamaAiConst(
    image=DockerImage(name="ollama/ollama:0.12.3", size="3.2 GB"),
    models=_read_models(),
)


class ModelInstalledInfo(BaseModel):
    id: str
    registered_name: str
    type: str
    options: InstallModelIn
    registration_id: RegistrationId


class OllamaOptions(BaseModel):
    gpu: bool
    num_parallel: int | None = None  # 3
    keep_alive: str = ""  # 60m
    is_flash_attention: bool | None = None  # False
    max_loaded_models: int | None = None  # 1
    kv_cache_type: str = ""  # f16


class OllamaModelOptions(BaseModel):
    alias: str | None = None
    alive_time: int | Annotated[str, StringConstraints(pattern=r"^(\d+[smh])?$", strict=True)] = ""


class InstalledInfo:
    def __init__(
        self,
        docker: DockerOptions,
        models: dict[str, ModelInstalledInfo],
        options: InstallServiceIn,
        parsed_options: OllamaOptions,
        container_host: str,
        container_port: int,
        docker_exposed_port: int,
    ):
        self.docker = docker
        self.models = models
        self.options = options
        self.parsed_options = parsed_options
        self.container_host = container_host
        self.container_port = container_port
        self.docker_exposed_port = docker_exposed_port
        self.base_url = get_base_url(self.container_host, self.container_port)


class OllamaService(Base2Service[InstalledInfo]):
    models: dict[str, OllamaModel]

    def _after_init(self) -> None:
        self.models = _const.models.copy()

    def get_id(self) -> str:
        """Return the service id."""
        return "ollama"

    def get_description(self) -> str:
        """Return the service description."""
        return "Self-hosted easy to use LLM model runner."

    def get_size(self) -> str:
        """Return the service size."""
        return _const.image.size

    def get_spec(self) -> ServiceSpecification:
        """Return the service specification."""
        return ServiceSpecification(
            fields=[
                ServiceField(type="bool", name="gpu", description="Run on GPU", required=False, default=self._has_gpu_for_spec()),
                ServiceField(
                    type="number",
                    name="num_parallel",
                    description="Maximum number of parallel requests (OLLAMA_NUM_PARALLEL)",
                    required=False,
                    default="3",
                ),
                ServiceField(
                    type="text",
                    name="keep_alive",
                    description="The duration that models stay loaded in memory (default 5m) (OLLAMA_KEEP_ALIVE)",
                    required=False,
                ),
                ServiceField(
                    type="bool",
                    name="is_flash_attention",
                    description="Enabled flash attention (OLLAMA_FLASH_ATTENTION)",
                    required=False,
                ),
                ServiceField(
                    type="number",
                    name="max_loaded_models",
                    description="Maximum number of loaded models per GPU (OLLAMA_MAX_LOADED_MODELS)",
                    required=False,
                ),
                ServiceField(
                    type="text",
                    name="kv_cache_type",
                    description="Quantization type for the K/V cache (default: f16, available: q4_0, q8_0) (OLLAMA_KV_CACHE_TYPE)",
                    required=False,
                ),
            ]
        )

    def get_model_spec(self) -> ModelSpecification:
        """Return the model specification."""
        return ModelSpecification(
            fields=[
                ModelField(type="text", name="alias", description="Model alias", required=False),
                ModelField(
                    type="text",
                    name="alive_time",
                    description="How long should this model last when it isn't used (e.g. 5m)",
                    required=False,
                ),
            ]
        )

    def get_custom_model_spec(self) -> CustomModelSpecification | None:
        """Return the custom model specification or None if custom model is not supported."""
        return CustomModelSpecification(
            fields=[
                CustomModelField(type="text", name="id", description="Model ID", placeholder="my-custom-model"),
                CustomModelField(type="text", name="size", description="Model size", placeholder="1GB"),
                CustomModelField(type="oneof", name="type", description="Model type", values=["llm", "embedding"]),
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

    def _get_image(self) -> DockerImage:
        return _const.image

    async def _install_core(self, options: InstallServiceIn) -> PromiseWithProgress[InstalledInfo, StreamChunk]:
        if "gpu" not in options.spec:
            options.spec["gpu"] = self.docker_service.has_gpu_support
        parsed_options = try_parse_pydantic(OllamaOptions, options.spec)
        image = self._get_image()

        async def func(stream: Stream[StreamChunk]) -> InstalledInfo:
            await self._docker_pull(image, stream)
            volumes = [f"{self._get_working_dir()}/main:/root/.ollama"]
            subnet = self.docker_service.get_docker_subnet()
            envs = dict[str, str]()
            if parsed_options.num_parallel is not None:
                envs["OLLAMA_NUM_PARALLEL"] = str(parsed_options.num_parallel)
            if parsed_options.keep_alive:
                envs["OLLAMA_KEEP_ALIVE"] = str(parsed_options.keep_alive)
            if parsed_options.is_flash_attention is not None:
                envs["OLLAMA_FLASH_ATTENTION"] = "1" if parsed_options.is_flash_attention else "0"
            if parsed_options.max_loaded_models is not None:
                envs["OLLAMA_MAX_LOADED_MODELS"] = str(parsed_options.max_loaded_models)
            if parsed_options.kv_cache_type:
                envs["OLLAMA_KV_CACHE_TYPE"] = str(parsed_options.kv_cache_type)
            docker_options = DockerOptions(
                name="ollama",
                container_name=self.docker_service.get_docker_container_name("ollama"),
                image=_const.image.name,
                image_port=11434,
                use_gpu=parsed_options.gpu,
                volumes=volumes,
                env_vars=envs,
                restart="unless-stopped",
                subnet=subnet,
                healthcheck={
                    "test": "ollama --version && ollama ps || exit 1",
                    "interval": "10s",
                    "timeout": "10s",
                    "retries": "10",
                    "start_period": "10s",
                },
            )
            docker_exposed_port = await self.docker_service.install_and_run_docker(docker_options)
            info = InstalledInfo(
                docker=docker_options,
                models={},
                options=options,
                parsed_options=parsed_options,
                container_host=self.docker_service.get_container_host(subnet, docker_options.name),
                container_port=self.docker_service.get_container_port(subnet, docker_exposed_port, docker_options.image_port),
                docker_exposed_port=docker_exposed_port,
            )
            stream.emit(StreamChunkProgress(type="progress", value=1))
            return info

        return PromiseWithProgress(func=func)

    async def _uninstall(self, options: UninstallServiceIn) -> None:
        info = self._check_installed()
        for model in info.models.copy().values():
            if model.type == "llm":
                self.endpoint_registry.unregister_chat_completion(model.registered_name, model.registration_id)
            if model.type == "embedding":
                self.endpoint_registry.unregister_embeddings(model.registered_name, model.registration_id)
        self.installed = None
        await self.docker_service.uninstall_docker(info.docker)
        if options.purge:
            await self._clear_working_dir()

    def get_docker_compose_file_path(self, model_id: str | None) -> Path:
        """Get docker compose file path."""
        info = self.installed
        if not info:
            raise HTTPException(400, "Service not installed")
        if model_id:
            raise HTTPException(400, "Docker is not bound with this object")
        return self.docker_service.get_docker_compose_file_path(info.docker.name)

    def service_has_docker(self) -> bool:
        """Return true when docker is started when service is installed."""
        return True

    def _add_custom_model(self, model: CustomModel) -> None:
        parsed = try_parse_pydantic(OllamaCustomModel, model.data)
        if parsed.id in self.models:
            raise HTTPException(400, "Model with given id already exists.")
        self.models[parsed.id] = OllamaModel(id=parsed.id, size=parsed.size, type=parsed.type, custom=model.id)

    def _remove_custom_model(self, model: CustomModel) -> None:
        parsed = try_parse_pydantic(OllamaCustomModel, model.data)
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
                        type=model.type,
                        installed=installed,
                        size=model.size,
                        custom=model.custom,
                        spec=self.get_model_spec(),
                        has_docker=False,
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
            type=model.type,
            installed=installed,
            size=model.size,
            custom=model.custom,
            spec=self.get_model_spec(),
            has_docker=False,
        )

    async def _install_model(self, model_id: str, options: InstallModelIn) -> PromiseWithProgress[InstallModelOut, StreamChunk]:  # noqa: C901
        parsed_model_options = try_parse_pydantic(OllamaModelOptions, options.spec) if options.spec else OllamaModelOptions()
        info = self._check_installed()
        if model_id in info.models:
            return PromiseWithProgress(value=InstallModelOut(status="OK", details="Already installed"))
        if model_id not in self.models:
            raise HTTPException(400, "Model not found")
        model = self.models[model_id]

        async def func(streamp: Stream[StreamChunk]) -> InstallModelOut:
            progress = Progress(convert_size_to_bytes(model.size) or 0)
            last_diggest: str = ""
            last_value: int = 0

            async for stream in stream_fetch_from(f"{info.base_url}/api/pull", "POST", {"model": model_id}):
                if (stream.status_code != 200 and stream.status_code != 201) or "error" in stream.data:
                    raise HTTPException(400, "Model not available")

                data_cleared: list[str] = stream.data.rstrip().split("\n")
                records = [json.loads(s) for s in data_cleared]
                if progress.max != 0:
                    for record in records:
                        if value := record.get("completed"):
                            digest = record.get("digest")
                            batch_download_bytes_size = value - last_value if digest == last_diggest else value
                            progress.add_to_actual_value(batch_download_bytes_size)
                            last_value = value
                            last_diggest = digest

                        elif record.get("status") == "success":
                            progress.set_actual_value(progress.max)

                        streamp.emit(StreamChunkProgress(type="progress", value=progress.get_percentage() * 0.99))

            streamp.emit(StreamChunkProgress(type="progress", value=0.99))
            if parsed_model_options.alive_time != "":
                await fetch_from(
                    f"{info.base_url}/api/generate",
                    "POST",
                    {"model": model_id, "keep_alive": parsed_model_options.alive_time},
                )

            registered_name = parsed_model_options.alias if parsed_model_options.alias else model_id
            info.models[model_id] = model_info = ModelInstalledInfo(
                id=model_id,
                type=model.type,
                registered_name=registered_name,
                options=options,
                registration_id="",
            )
            if model.type == "llm":
                model_info.registration_id = self.endpoint_registry.register_chat_completion_as_proxy(
                    model=registered_name,
                    props=ModelProps(private=True),
                    chat_completions=ProxyOptions(url=f"{info.base_url}/v1/chat/completions", rewrite_model_to=model_id),
                    completions=ProxyOptions(url=f"{info.base_url}/v1/completions", rewrite_model_to=model_id),
                    registration_options=None,
                )
            if model.type == "embedding":
                model_info.registration_id = self.endpoint_registry.register_embeddings_as_proxy(
                    model=registered_name,
                    props=ModelProps(private=True),
                    options=ProxyOptions(url=f"{info.base_url}/v1/embeddings", rewrite_model_to=model_id),
                    registration_options=None,
                )
            streamp.emit(StreamChunkProgress(type="progress", value=1))
            return InstallModelOut(status="OK", details="Installed")

        return PromiseWithProgress(func=func)

    async def _uninstall_model(self, model_id: str, options: UninstallModelIn) -> None:
        info = self._check_installed()
        if model_id not in info.models:
            return
        model = info.models[model_id]
        del info.models[model_id]
        if model.type == "llm":
            self.endpoint_registry.unregister_chat_completion(model.registered_name, model.registration_id)
        if model.type == "embedding":
            self.endpoint_registry.unregister_embeddings(model.registered_name, model.registration_id)

        if options.purge:
            await fetch_from(f"{info.base_url}/api/delete", "DELETE", {"name": model_id})
