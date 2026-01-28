# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ollama service."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

import aiofiles
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
    convert_size_to_bytes,
    fetch_from,
    stream_fetch_from,
    try_parse_pydantic,
)
from server.utils.loading import Progress

logger = logging.getLogger("uvicorn")


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


type Quantization = Literal[
    "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "q3_K_S", "q3_K_M", "q3_K_L", "q4_K_S", "q4_K_M", "q5_K_S", "q5_K_M", "q6_K"
]


class OllamaModel(BaseModel):
    id: str
    size: str
    type: str
    custom: CustomModelId | None = None
    modelfile: str | None = None
    quantization: Quantization | Literal[""] | None = None


class OllamaCustomModel(BaseModel):
    id: str
    size: str
    type: Literal["llm", "embedding"]
    modelfile: str | None = None
    quantization: Quantization | Literal[""] | None = None


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
            map[tag["name"]] = OllamaModel(
                id=tag["name"],
                size=tag["size"],
                type="llm",
                modelfile=tag.get("modelfile", ""),
                quantization=tag.get("quantization", ""),
            )
        for tag in registry["embeddings"]:
            map[tag["name"]] = OllamaModel(
                id=tag["name"],
                size=tag["size"],
                type="embedding",
                modelfile=tag.get("modelfile", ""),
                quantization=tag.get("quantization", ""),
            )
        return map


class OllamaAiConst(BaseModel):
    image: DockerImage
    models: dict[str, OllamaModel]


_const = OllamaAiConst(
    image=DockerImage(name="ollama/ollama:0.15.1", size="5.6 GB"),
    models=_read_models(),
)


@dataclass
class ModelInstalledInfo:
    id: str
    registered_name: str
    type: str
    options: InstallModelIn
    registration_id: RegistrationId
    internal_name: str | None

    def get_info(self) -> ModelInfo:
        """Get info."""
        return ModelInfo(spec=self.options.spec, registration_id=self.registration_id)


class OllamaOptions(BaseModel):
    hardware: str | bool | None = None
    num_parallel: int | None = None  # 3
    keep_alive: str = ""  # 60m
    is_flash_attention: bool | None = None  # False
    max_loaded_models: int | None = None  # 1
    kv_cache_type: str = ""  # f16
    context_length: int | None = None  # 4096


class OllamaModelOptions(BaseModel):
    alias: str | None = None
    alive_time: int | Annotated[str, StringConstraints(pattern=r"^(\d+[smh])?$", strict=True)] = ""
    context_length: int | None = None


@dataclass
class InstalledInfo:
    docker: DockerOptions
    models: dict[str, ModelInstalledInfo]
    options: InstallServiceIn
    parsed_options: OllamaOptions
    container_host: str
    container_port: int
    docker_exposed_port: int
    base_url: str


@dataclass
class DownloadedInfo:
    pass


class OllamaService(Base2Service[InstalledInfo, DownloadedInfo]):
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
        fields = self.add_gpu_field_to_spec()

        fields.extend(
            [
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
                ServiceField(
                    type="number",
                    name="context_length",
                    description="Default context length (OLLAMA_CONTEXT_LENGTH)",
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
                    description="How long should this model last when it isn't used (e.g. 5m)",
                    required=False,
                ),
                ModelField(
                    type="number",
                    name="context_length",
                    description="The size of context for this model (e.g. 8192)",
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
                CustomModelField(
                    type="textarea",
                    name="modelfile",
                    description="Modelfile",
                    placeholder="FROM gemma3:1b\nPARAMETER num_ctx 8192",
                    required=False,
                ),
                CustomModelField(type="text", name="quantization", description="Quantization", placeholder="q4_0", required=False),
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

    def _get_image(self) -> DockerImage:
        return _const.image

    def _load_download_info(self, data: dict[str, Any]) -> DownloadedInfo:
        return DownloadedInfo(**data)

    async def _install_core(self, options: InstallServiceIn) -> PromiseWithProgress[InstalledInfo, StreamChunk]:
        if "hardware" not in options.spec:
            options.spec["hardware"] = options.spec.get("gpu", self.docker_service.has_gpu_support)
        parsed_options = try_parse_pydantic(OllamaOptions, options.spec)
        image = self._get_image()
        await self._verify_docker_image(image.name, options.ignore_warnings)

        async def func(stream: Stream[StreamChunk]) -> InstalledInfo:
            await self._docker_pull(image, stream)
            self.service_downloaded = True
            stream.emit(StreamChunkProgress(type="progress", stage="install", value=0))
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
            if parsed_options.context_length:
                envs["OLLAMA_CONTEXT_LENGTH"] = str(parsed_options.context_length)
            docker_options = DockerOptions(
                name="ollama",
                container_name=self.docker_service.get_docker_container_name("ollama"),
                image=image.name,
                image_port=11434,
                hardware=self.get_specified_hardware_parts(parsed_options.hardware),
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
            stream.emit(StreamChunkProgress(type="progress", stage="install", value=1))

            return info

        return PromiseWithProgress(func=func)

    async def _uninstall(self, options: UninstallServiceIn) -> None:
        if self.installed:
            for model in self.installed.models.copy().values():
                if model.type == "llm":
                    self.endpoint_registry.unregister_chat_completion(model.registered_name, model.registration_id)
                if model.type == "embedding":
                    self.endpoint_registry.unregister_embeddings(model.registered_name, model.registration_id)
            await self.docker_service.uninstall_docker(self.installed.docker)
        self.installed = None
        if options.purge:
            self.service_downloaded = False
            await self.docker_service.remove_image(_const.image.name)
            await self._clear_working_dir()
            self.models_downloaded = {}

    async def stop(self) -> None:
        """Stop the Ollama service Docker container."""
        info = self.installed
        if not info:
            return
        await self._stop_docker(info.docker)

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
        self.models[parsed.id] = OllamaModel(
            id=parsed.id,
            size=parsed.size,
            type=parsed.type,
            custom=model.id,
            modelfile=parsed.modelfile,
            quantization=parsed.quantization,
        )

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
            installed = info.models[model_id].get_info() if model_id in info.models else self._get_model_installed_info(model_id)
            if filters.installed is None or filters.installed == installed:
                out_list.append(
                    RetrieveModelOut(
                        id=model_id,
                        service=self.get_id(),
                        type=model.type,
                        installed=installed,
                        downloaded=model_id in self.models_downloaded,
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
        installed = info.models[model_id].get_info() if model_id in info.models else self._get_model_installed_info(model_id)
        return RetrieveModelOut(
            id=model_id,
            service=self.get_id(),
            type=model.type,
            installed=installed,
            downloaded=model_id in self.models_downloaded,
            size=model.size,
            custom=model.custom,
            spec=self.get_model_spec(),
            has_docker=False,
        )

    async def _download_with_ollama(self, stream: Stream[StreamChunk], base_url: str, model_id: str, model_size: str) -> None:
        progress = Progress(convert_size_to_bytes(model_size) or 0)
        last_diggest: str = ""
        last_value: int = 0

        stream.emit(StreamChunkProgress(type="progress", stage="download", value=0))
        async for ollama_stream in stream_fetch_from(f"{base_url}/api/pull", "POST", {"model": model_id}, timeout=24 * 60 * 60):
            if (ollama_stream.status_code != 200 and ollama_stream.status_code != 201) or "error" in ollama_stream.data:
                raise HTTPException(400, "Model not available")

            data_cleared: list[str] = ollama_stream.data.rstrip().split("\n")
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

                    stream.emit(StreamChunkProgress(type="progress", stage="download", value=progress.get_percentage()))

        stream.emit(StreamChunkProgress(type="progress", stage="download", value=1))

    @staticmethod
    async def is_model_installed(base_url: str, model: str) -> bool:
        """Return is model installed."""
        result = await fetch_from(
            f"{base_url}/api/show",
            "POST",
            {"model": model},
        )
        return result.status_code == 200

    async def create_model_from_modelfile(
        self, compose_filepath: Path, service_name: str, model: str, model_path: str, quantization: str | None
    ) -> str:
        """Send message to ollama to create model from Modelfile.."""
        quantization_param = f"-q {quantization} " if quantization else ""
        cmd = f"ollama create {quantization_param}{model} -f {model_path}"
        return await self.docker_service.run_command_docker_compose(compose_filepath, service_name, cmd)

    async def _download(
        self,
        input_stream: Stream[StreamChunk],
        base_url: str,
        model_url: str,
        model_size: str,
        model_id: str,
        model_number: int,
        model_quantity: int,
    ) -> str:
        base_docker_dir: Path = Path("/root/.ollama/custom")
        local_models_dir = Path(self._get_working_dir()) / "main" / "custom"
        model_path = Path(model_url)

        if model_url.startswith(("http://", "https://")):
            dir = Path(Path(local_models_dir) / model_id)
            additional_params = ()
            if model_path.suffix == ".gguf":
                # If file then add filename to download.
                additional_params = (model_path.name,)
            else:
                # Download all files to subdirectory model
                dir = dir / "model"

            progress = Progress(convert_size_to_bytes(model_size) or 0)
            input_stream.emit(StreamChunkProgress(type="progress", stage="download", value=0))
            async for packet in self.model_downloader.download(model_url, dir, *additional_params):
                if isinstance(packet, DownloadedPacket) and packet.downloaded_bytes_size != 0:
                    progress.add_to_actual_value(packet.downloaded_bytes_size)
                    value = (model_quantity * model_number) + (progress.get_percentage() / model_quantity)
                    input_stream.emit(StreamChunkProgress(type="progress", stage="download", value=value))
                elif isinstance(packet, PreDownloadPacket):
                    if max := packet.file_bytes_size:
                        progress.set_max_value(max)

            # Return path to file or path to directory (with model)
            input_stream.emit(StreamChunkProgress(type="progress", stage="download", value=1))
            return str(base_docker_dir / model_id / model_path.name)

        await self._download_with_ollama(input_stream, base_url, model_url, model_size)
        return model_url

    async def create_docker_modelfile_content(self, model_name: str, modelfile: str) -> tuple[str, list[str]]:
        """Create docker modelfile content.

        Can edit local models path to docker model paths.
        """
        base_docker_dir: Path = Path("/root/.ollama/custom")
        local_models_dir = Path(self._get_working_dir()) / "main" / "custom"
        docker_modelfile_parts: list[str] = []
        models_to_download_with_duplicates: list[str] = []

        for line in modelfile.split("\n"):
            new_line = line
            if line.startswith(("FROM", "ADAPTER")):
                line_parts = line.split(" ")
                new_line_parts: list[str] = line_parts.copy()
                if len(line_parts) == 2:
                    model = line_parts[1]
                    path_model = Path(model)
                    if model.startswith(("http://", "https://")):
                        file_or_dir = path_model.name if path_model.suffix == ".gguf" else "model"
                        path = str(base_docker_dir / model_name / file_or_dir)
                        models_to_download_with_duplicates.append(model)
                    else:
                        docker_parent_path = base_docker_dir / path_model
                        local_parent_path = local_models_dir / path_model
                        if local_parent_path.exists():
                            # If exist then we don't want download. Replace to docker path
                            path = str(docker_parent_path)
                        else:
                            # If doesn't exist the we download with ollama
                            path = model
                            models_to_download_with_duplicates.append(model)

                    new_line_parts[1] = path

                new_line = " ".join(new_line_parts)

            docker_modelfile_parts.append(new_line)

        models = list(set(models_to_download_with_duplicates))

        return "\n".join(docker_modelfile_parts), models

    async def save_modelfile(self, model: str, modelfile: str) -> str:
        """Save modelfile."""
        base_docker_dir: Path = Path("root/.ollama/custom")
        base_local_dir: Path = Path(self._get_working_dir())
        docker_modelfile_path = base_docker_dir / model / "Modelfile"
        local_modelfile_path = base_local_dir / "main" / "custom" / model / "Modelfile"

        if local_modelfile_path.exists():
            async with aiofiles.open(local_modelfile_path) as f:
                content = await f.read()
                if modelfile == content:
                    return str(docker_modelfile_path)

        if not local_modelfile_path.parent.exists():
            local_modelfile_path.parent.mkdir(parents=True)

        async with aiofiles.open(local_modelfile_path, mode="w") as f:
            await f.write(modelfile)

        return str(docker_modelfile_path)

    async def _install_model(self, model_id: str, options: InstallModelIn) -> PromiseWithProgress[InstallModelOut, StreamChunk]:  # noqa: C901
        parsed_model_options = try_parse_pydantic(OllamaModelOptions, options.spec) if options.spec else OllamaModelOptions()
        info = self._check_installed()
        if model_id in info.models:
            return PromiseWithProgress(value=InstallModelOut(status="OK", details="Already installed"))
        if model_id not in self.models:
            raise HTTPException(400, "Model not found")
        model = self.models[model_id]

        async def func(input_stream: Stream[StreamChunk]) -> InstallModelOut:
            service_name = info.docker.name
            compose_filepath = self.docker_service.get_docker_compose_file_path(service_name)
            if not await self.is_model_installed(info.base_url, model_id):
                if not model.modelfile:
                    await self._download_with_ollama(input_stream, info.base_url, model_id, model.size)
                    input_stream.emit(StreamChunkProgress(type="progress", stage="install", value=0))
                else:
                    content, models_to_download = await self.create_docker_modelfile_content(model_id, model.modelfile)
                    models_quantity = len(set(models_to_download))
                    for i, model_to_download in enumerate(models_to_download):
                        await self._download(input_stream, info.base_url, model_to_download, model.size, model_id, i, models_quantity)
                    path = await self.save_modelfile(model_id, content)
                    quantization = model.quantization

                    msg = f"Create model {model_id} from {path}{f'with quantization {quantization}' if quantization else ''}"
                    logger.debug(msg)

                    input_stream.emit(StreamChunkProgress(type="progress", stage="install", value=0))
                    await self.create_model_from_modelfile(compose_filepath, service_name, model_id, path, quantization)
            else:
                input_stream.emit(StreamChunkProgress(type="progress", stage="install", value=0))

            if parsed_model_options.alive_time != "":
                await fetch_from(
                    f"{info.base_url}/api/generate",
                    "POST",
                    {"model": model_id, "keep_alive": parsed_model_options.alive_time},
                )
            rewrite_model_to = model_id
            internal_name: str | None = None
            if not model.modelfile and model.type == "llm" and parsed_model_options.context_length is not None:
                internal_name = model_id + "-customcontextlength"
                rewrite_model_to = internal_name
                content = f"FROM {model_id}\nPARAMETER num_ctx {parsed_model_options.context_length}"
                path = await self.save_modelfile(internal_name, content)
                await self.create_model_from_modelfile(compose_filepath, service_name, internal_name, path, "")

            registered_name = parsed_model_options.alias if parsed_model_options.alias else model_id
            info.models[model_id] = model_info = ModelInstalledInfo(
                id=model_id,
                type=model.type,
                registered_name=registered_name,
                options=options,
                registration_id="",
                internal_name=internal_name,
            )
            if model.type == "llm":
                model_info.registration_id = self.endpoint_registry.register_chat_completion_as_proxy(
                    model=registered_name,
                    props=ModelProps(private=True),
                    chat_completions=ProxyOptions(url=f"{info.base_url}/v1/chat/completions", rewrite_model_to=rewrite_model_to),
                    completions=ProxyOptions(url=f"{info.base_url}/v1/completions", rewrite_model_to=rewrite_model_to),
                    responses=ProxyOptions(url=f"{info.base_url}/v1/responses", rewrite_model_to=rewrite_model_to),
                    messages=ProxyOptions(url=f"{info.base_url}/v1/messages", rewrite_model_to=rewrite_model_to),
                    registration_options=None,
                )
            if model.type == "embedding":
                model_info.registration_id = self.endpoint_registry.register_embeddings_as_proxy(
                    model=registered_name,
                    props=ModelProps(private=True),
                    options=ProxyOptions(url=f"{info.base_url}/v1/embeddings", rewrite_model_to=rewrite_model_to),
                    registration_options=None,
                )
            input_stream.emit(StreamChunkProgress(type="progress", stage="install", value=1))
            self.models_downloaded[model_id] = DownloadedInfo()
            return InstallModelOut(status="OK", details="Installed")

        return PromiseWithProgress(func=func)

    async def _uninstall_model(self, model_id: str, options: UninstallModelIn) -> None:
        info = self._check_installed()
        if model_id in info.models:
            model = info.models[model_id]
            del info.models[model_id]
            if model.type == "llm":
                self.endpoint_registry.unregister_chat_completion(model.registered_name, model.registration_id)
            if model.type == "embedding":
                self.endpoint_registry.unregister_embeddings(model.registered_name, model.registration_id)

        if options.purge and model_id in self.models_downloaded:
            await fetch_from(f"{info.base_url}/api/delete", "DELETE", {"name": model_id})
            del self.models_downloaded[model_id]
