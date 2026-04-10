# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ollama service."""

import asyncio
import json
import logging
import shutil
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, NamedTuple, TypedDict

import aiofiles
from fastapi import HTTPException
from pydantic import BaseModel, StringConstraints

from server.applicationcontext import get_base_url
from server.config import get_main_dir
from server.docker import DockerImage, DockerOptions
from server.endpointregistry import ProxyOptions, RegistrationId
from server.models.api import EMBEDDINGS_ENDPOINTS, IMG_ENDPOINTS, LLM_ENDPOINTS, ModelProps
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
    stream_fetch_from,
    try_parse_pydantic,
)
from server.utils.files import get_gguf_context_window, get_model_dir_context_window
from server.utils.hardware import GpuInfo, HardwarePartInfo, IntelGpuInfo
from server.utils.loading import Progress

logger = logging.getLogger("uvicorn.error")


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
    hash: str
    context: int | None
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
    hash: str
    context: int | None


class OllamaRegistry(TypedDict):
    llms: list[OllamaRegistryEntry]
    embeddings: list[OllamaRegistryEntry]
    txt2img: list[OllamaRegistryEntry]


class ModelfileData(NamedTuple):
    content: str
    urls: list[str]
    docker_paths: list[str]
    local_paths: list[str]


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
                hash=tag["hash"],
                context=tag["context"],
            )
        for tag in registry["embeddings"]:
            map[tag["name"]] = OllamaModel(
                id=tag["name"],
                size=tag["size"],
                type="embedding",
                modelfile=tag.get("modelfile", ""),
                quantization=tag.get("quantization", ""),
                hash=tag["hash"],
                context=tag["context"],
            )
        for tag in registry["txt2img"]:
            map[tag["name"]] = OllamaModel(
                id=tag["name"],
                size=tag["size"],
                type="txt2img",
                modelfile=tag.get("modelfile", ""),
                quantization=tag.get("quantization", ""),
                hash=tag["hash"],
                context=tag["context"],
            )
        return map


class OllamaAiConst(BaseModel):
    image: DockerImage
    models: dict[str, OllamaModel]


_const = OllamaAiConst(
    image=DockerImage(name="ollama/ollama:0.20.4", size="6.2 GB"),
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
    model_paths: list[str] | None = None


class OllamaService(Base2Service[InstalledInfo, DownloadedInfo]):
    models: dict[str, dict[str, OllamaModel]]
    default_context_length: int

    @property
    def _supported_gpus(self) -> list[GpuInfo]:
        """Return GPUs supported by Ollama (including Intel via Vulkan)."""
        return self.hardware.gpus

    def _after_init(self) -> None:
        self.models = {}
        self.load_default_models("default")
        self.default_context_length = self.get_default_context_value()

    def load_default_models(self, instance: str) -> None:
        """Load default models to instance."""
        self.models[instance] = _const.models.copy()

    def get_type(self) -> str:
        """Return the type."""
        return "ollama"

    def get_description(self) -> str:
        """Return the service description."""
        return "Self-hosted easy to use LLM model runner."

    def get_size(self) -> str:
        """Return the service size."""
        return _const.image.size

    def get_spec(self) -> ServiceSpecification:
        """Return the service specification."""
        fields = self.add_hardware_field_to_spec()

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
                CustomModelField(type="oneof", name="type", description="Model type", values=["llm", "embedding", "txt2img"]),
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

    def _get_image(self) -> DockerImage:
        return _const.image

    def _load_download_info(self, data: dict[str, Any]) -> DownloadedInfo:
        return DownloadedInfo(**data)

    def _build_env_vars(self, parsed_options: OllamaOptions, hardware: Sequence[HardwarePartInfo]) -> dict[str, str]:
        """Build environment variables for the Ollama Docker container."""
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
        if any(isinstance(h, IntelGpuInfo) for h in hardware):
            envs["OLLAMA_VULKAN"] = "1"
        # Default envs
        envs["OLLAMA_NO_CLOUD"] = "1"
        return envs

    async def _install_instance(self, instance: str, options: InstallServiceIn) -> PromiseWithProgress[InstalledInfo, StreamChunk]:
        if not self.models.get(instance):
            self.load_default_models(instance)

        if "hardware" not in options.spec:
            options.spec["hardware"] = options.spec.get("gpu", self.docker_service.has_gpu_support)
        parsed_options = try_parse_pydantic(OllamaOptions, options.spec)
        image = self._get_image()
        await self._verify_docker_image(image.name, options.ignore_warnings)

        async def func(stream: Stream[StreamChunk]) -> InstalledInfo:
            await self._download_image_or_set_progress(stream, image)
            self.service_downloaded = True

            stream.emit(StreamChunkProgress(type="progress", stage="install", value=0, data={}))
            volumes = [f"{self._get_working_dir()}/main:/root/.ollama"]
            subnet = self.docker_service.get_docker_subnet()
            hardware = self.get_specified_hardware_parts(parsed_options.hardware)
            envs = self._build_env_vars(parsed_options, hardware)
            service_name = f"{self.get_service_id(instance)}"
            docker_options = DockerOptions(
                name=service_name,
                container_name=self.docker_service.get_docker_container_name(service_name),
                image=image.name,
                image_port=11434,
                hardware=hardware,
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
            stream.emit(StreamChunkProgress(type="progress", stage="install", value=1, data={}))

            return info

        return PromiseWithProgress(func=func)

    async def _uninstall_instance(self, instance: str, options: UninstallServiceIn) -> None:
        installed = self.get_instance_info(instance).installed
        if installed:
            for model in installed.models.copy().values():
                if model.type == "llm":
                    self.endpoint_registry.unregister_chat_completion(model.registered_name, model.registration_id)
                if model.type == "embedding":
                    self.endpoint_registry.unregister_embeddings(model.registered_name, model.registration_id)
                if model.type == "txt2img":
                    self.endpoint_registry.unregister_image_generations(model.registered_name, model.registration_id)
                if not self.is_model_installed_in_other_instance(instance, model.id):
                    await self._uninstall_model(instance, model.id, UninstallModelIn(purge=options.purge))

            await self.docker_service.uninstall_docker(installed.docker)

        self.instances_info[instance].installed = None
        if options.purge:
            if len(self.instances_info) < 2:
                self.service_downloaded = False
                await self.docker_service.remove_image(_const.image.name)
                await self._clear_working_dir()
                self.models_downloaded = {}

            if instance == "default":
                self.instances_info["default"] = Instance(None, None, {}, InstanceConfig())
            else:
                del self.instances_info[instance]

    async def stop_instance(self, instance: str) -> None:
        """Stop the Ollama service Docker container."""
        installed = self.get_instance_info(instance).installed
        if not installed:
            return
        await self._stop_docker(installed.docker)

    def get_docker_compose_file_path(self, instance: str, model_id: str | None) -> Path:
        """Get docker compose file path."""
        info = self.get_instance_installed_info(instance)
        if model_id:
            raise HTTPException(400, "Docker is not bound with this object")

        return self.docker_service.get_docker_compose_file_path(info.docker.name)

    def service_has_docker(self) -> bool:
        """Return true when docker is started when service is installed."""
        return True

    def _add_custom_model(self, instance: str, model: CustomModel) -> None:
        parsed = try_parse_pydantic(OllamaCustomModel, model.data)

        if not self.models.get(instance):
            self.models[instance] = {}

        if parsed.id in self.models[instance]:
            raise HTTPException(400, f"Model with {parsed.id} id already exists.")

        self.models[instance][parsed.id] = OllamaModel(
            id=parsed.id,
            size=parsed.size,
            type=parsed.type,
            custom=model.id,
            modelfile=parsed.modelfile,
            quantization=parsed.quantization,
            hash=parsed.id,
            context=None,
        )

    def _remove_custom_model(self, instance: str, model: CustomModel) -> None:
        installed = self.get_instance_info(instance).installed
        parsed = try_parse_pydantic(OllamaCustomModel, model.data)
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

        stream.emit(StreamChunkProgress(type="progress", stage="download", value=0, data={}))
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

                    stream.emit(StreamChunkProgress(type="progress", stage="download", value=progress.get_percentage(), data={}))

        stream.emit(StreamChunkProgress(type="progress", stage="download", value=1, data={}))

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

    async def _download_with_downloader(
        self,
        input_stream: Stream[StreamChunk],
        base_url: str,
        model_url: str,
        model_id: str,
        model_size: str,
        model_number: int,
        model_quantity: int,
    ) -> None:
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
            input_stream.emit(StreamChunkProgress(type="progress", stage="download", value=0, data={}))
            async for packet in self.model_downloader.download(model_url, dir, *additional_params):
                if isinstance(packet, DownloadedPacket) and packet.downloaded_bytes_size != 0:
                    progress.add_to_actual_value(packet.downloaded_bytes_size)
                    value = (model_quantity * model_number) + (progress.get_percentage() / model_quantity)
                    input_stream.emit(StreamChunkProgress(type="progress", stage="download", value=value, data={}))
                elif isinstance(packet, PreDownloadPacket):
                    if max := packet.file_bytes_size:
                        progress.set_max_value(max)

            # Return path to file or path to directory (with model)
            input_stream.emit(StreamChunkProgress(type="progress", stage="download", value=1, data={}))
        else:
            await self._download_with_ollama(input_stream, base_url, model_url, model_size)
        return

    def _process_from_line(
        self, line: str, model_name: str, instance: str, base_docker_dir: Path, local_models_dir: Path
    ) -> tuple[str, str | None, str | None, str | None]:
        """Return (new_line, url, docker_path, local_path) for a FROM line."""
        _, model = line.split(" ")
        path_model = Path(model)
        if model.startswith(("http://", "https://")):
            file_or_dir = path_model.name if path_model.suffix == ".gguf" else "model"
            docker_path = str(base_docker_dir / model_name / file_or_dir)
            local_path = str(local_models_dir / model_name / file_or_dir)
            return f"FROM {docker_path}", model, docker_path, local_path
        docker_path = str(base_docker_dir / path_model)
        local_path = str(local_models_dir / path_model)
        if model_name not in self.models[instance]:
            return f"FROM {docker_path}", None, docker_path, local_path
        return f"FROM {model}", model, None, None

    async def create_docker_modelfile_content(self, model_name: str, modelfile: str, instance: str) -> ModelfileData:
        """Create docker modelfile content.

        Can edit local models path to docker model paths.
        """
        base_docker_dir: Path = Path("/root/.ollama/custom")
        local_models_dir = Path(self._get_working_dir()) / "main" / "custom"
        new_lines: list[str] = []
        models: list[str] = []
        docker_paths: list[str] = []
        local_paths: list[str] = []

        for line in modelfile.split("\n"):
            if line.startswith("FROM"):
                new_line, url, docker_path, local_path = self._process_from_line(
                    line, model_name, instance, base_docker_dir, local_models_dir
                )
            elif line.startswith("ADAPTER"):
                _, url = line.split(" ")
                docker_path = str(base_docker_dir / model_name / "adapter.gguf")
                local_path = str(local_models_dir / model_name / "adapter.gguf")
                new_line = f"ADAPTER {docker_path}"
            else:
                new_line, url, docker_path, local_path = line, None, None, None
            new_lines.append(new_line)
            if url is not None and url not in models:
                models.append(url)
            if docker_path is not None and local_path is not None:
                docker_paths.append(docker_path)
                local_paths.append(local_path)

        return ModelfileData("\n".join(new_lines), models, docker_paths, local_paths)

    def _get_modelfile_path(self, model: str, instance: str) -> Path:
        """Return local (host) modelfile path."""
        return Path(self._get_working_dir()) / "main" / "custom" / model / instance / "Modelfile"

    def _get_docker_modelfile_path(self, model: str, instance: str) -> Path:
        """Return docker modelfile path."""
        return Path("root/.ollama/custom") / model / instance / "Modelfile"

    async def remove_modelfile(self, model: str, instance: str) -> None:
        """Remove modelfile."""
        local_modelfile_path = self._get_modelfile_path(model, instance)
        local_modelfile_path.unlink(missing_ok=True)
        parent_dir = local_modelfile_path.parent
        if parent_dir.is_dir() and not any(parent_dir.iterdir()):
            parent_dir.rmdir()

    async def save_modelfile(self, model: str, instance: str, modelfile: str) -> str:
        """Save modelfile."""
        docker_modelfile_path = self._get_docker_modelfile_path(model, instance)
        local_modelfile_path = self._get_modelfile_path(model, instance)

        # Remove old Modelfile
        old_path = Path(local_modelfile_path.parent.parent / "Modelfile")
        if old_path.exists():
            old_path.unlink()

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

    async def _download_model_or_set_progress(
        self,
        input_stream: Stream[StreamChunk],
        base_url: str,
        model_url: str,
        model_id: str,
        size: str,
        i: int = 1,
        models_quantity: int = 1,
    ) -> None:
        if model_id not in self.models_download_progress:
            self.models_download_progress[model_id] = input_stream
            await self._download_with_downloader(input_stream, base_url, model_url, model_id, size, i, models_quantity)
            del self.models_download_progress[model_id]
        else:
            chunk: StreamChunk
            async for chunk in self.models_download_progress[model_id].as_generator():
                if chunk.get("type") == "progress" and chunk.get("stage") == "download":
                    input_stream.emit(chunk)
                else:
                    break

    def get_default_context_window(self, model_context: int | None, service_context: int) -> int:
        """Return minimum default context length from model or service context length."""
        return min(model_context, service_context) if model_context else service_context

    async def _install_model(  # noqa: C901
        self, instance: str, model_id: str, options: InstallModelIn
    ) -> PromiseWithProgress[InstallModelOut, StreamChunk]:
        parsed_model_options = try_parse_pydantic(OllamaModelOptions, options.spec) if options.spec else OllamaModelOptions()
        info = self.get_instance_installed_info(instance)

        if not self.models.get(instance):
            self.models[instance] = {}

        if model_id in info.models:
            return PromiseWithProgress(value=InstallModelOut(status="OK", details="Already installed"))

        if model_id not in self.models[instance]:
            raise HTTPException(400, f"Model {model_id} not found")

        model = self.models[instance][model_id]

        async def func(input_stream: Stream[StreamChunk]) -> InstallModelOut:  # noqa: C901
            rewrite_model_to = model_id
            internal_name = model_id
            new_modelfile: str | None = None

            service_form_context_window = info.parsed_options.context_length
            model_form_context_window = parsed_model_options.context_length

            # Default model with custom context
            if model.type == "llm" and parsed_model_options.context_length is not None:
                if model.modelfile:
                    new_modelfile = f"{model.modelfile}\nPARAMETER num_ctx {parsed_model_options.context_length}"
                else:
                    internal_name = model_id + "-customcontextlength"
                    rewrite_model_to = internal_name
                    new_modelfile = f"FROM {model_id}\nPARAMETER num_ctx {parsed_model_options.context_length}"

            modelfile_recipe = new_modelfile or model.modelfile
            if modelfile_recipe:
                modelfile_data = await self.create_docker_modelfile_content(model_id, modelfile_recipe, instance)
                models_to_download = modelfile_data.urls
                paths = modelfile_data.local_paths
                path = await self.save_modelfile(model_id, instance, modelfile_data.content)
                quantization = model.quantization
                service_name = info.docker.name
                for i, model_to_download in enumerate(models_to_download):
                    if not await self.is_model_installed(info.base_url, model_id):
                        await self._download_model_or_set_progress(
                            input_stream, info.base_url, model_to_download, model_id, model.size, i, len(models_to_download)
                        )
                compose_filepath = self.docker_service.get_docker_compose_file_path(service_name)
                try:
                    await self.create_model_from_modelfile(compose_filepath, service_name, internal_name, path, quantization)
                except Exception:
                    for path in modelfile_data.local_paths:
                        if Path(path).is_dir():
                            with suppress(Exception):
                                shutil.rmtree(path)
                        else:
                            Path(path).unlink(missing_ok=True)
                if len(paths) != 0:
                    path = paths[0]
                    if path.endswith(".gguf"):
                        model.context = await get_gguf_context_window(path)
                    else:
                        temp_max_context_window = await get_model_dir_context_window(path)
                        if temp_max_context_window:
                            model.context = temp_max_context_window

            else:
                if not await self.is_model_installed(info.base_url, model_id):
                    await self._download_model_or_set_progress(input_stream, info.base_url, model.id, model.id, model.size)

            input_stream.emit(StreamChunkProgress(type="progress", stage="install", value=0, data={}))

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
                internal_name=internal_name,
            )
            model_max_context_window = model.context
            service_max_context_window = self.default_context_length
            default_context_window = self.get_default_context_window(model_max_context_window, service_max_context_window)
            max_context_window = model_max_context_window or service_max_context_window

            context_window = model_form_context_window or service_form_context_window or default_context_window
            if model.type == "llm":
                model_info.registration_id = self.endpoint_registry.register_chat_completion_as_proxy(
                    model=registered_name,
                    props=ModelProps(
                        private=True,
                        type=model.type,
                        endpoints=LLM_ENDPOINTS,
                        context_window=context_window,
                        max_context_window=max_context_window,
                    ),
                    chat_completions=ProxyOptions(url=f"{info.base_url}/v1/chat/completions", rewrite_model_to=rewrite_model_to),
                    completions=ProxyOptions(url=f"{info.base_url}/v1/completions", rewrite_model_to=rewrite_model_to),
                    responses=ProxyOptions(url=f"{info.base_url}/v1/responses", rewrite_model_to=rewrite_model_to),
                    messages=ProxyOptions(url=f"{info.base_url}/v1/messages", rewrite_model_to=rewrite_model_to),
                    registration_options=None,
                )
            if model.type == "embedding":
                model_info.registration_id = self.endpoint_registry.register_embeddings_as_proxy(
                    model=registered_name,
                    props=ModelProps(
                        private=True,
                        type=model.type,
                        endpoints=EMBEDDINGS_ENDPOINTS,
                        context_window=context_window,
                        max_context_window=max_context_window,
                    ),
                    options=ProxyOptions(url=f"{info.base_url}/v1/embeddings", rewrite_model_to=rewrite_model_to),
                    registration_options=None,
                )
            if model.type == "txt2img":
                model_info.registration_id = self.endpoint_registry.register_image_generations_as_proxy(
                    model=registered_name,
                    props=ModelProps(
                        private=True,
                        type=model.type,
                        endpoints=IMG_ENDPOINTS,
                        context_window=context_window,
                        max_context_window=max_context_window,
                    ),
                    options=ProxyOptions(url=f"{info.base_url}/v1/images/generations", rewrite_model_to=rewrite_model_to),
                    registration_options=None,
                )
            input_stream.emit(StreamChunkProgress(type="progress", stage="install", value=1, data={}))

            self.models_downloaded[model_id] = DownloadedInfo()

            if model.hash:
                for alias_model_id, alias_model in self.models[instance].items():
                    if alias_model.hash == model.hash:
                        self.models_downloaded[alias_model_id] = DownloadedInfo()

            return InstallModelOut(status="OK", details="Installed")

        return PromiseWithProgress(func=func)

    async def _uninstall_model(self, instance: str, model_id: str, options: UninstallModelIn) -> None:
        info = self.get_instance_installed_info(instance)
        if model_id in info.models:
            model = info.models[model_id]
            del info.models[model_id]
            if model.type == "llm":
                self.endpoint_registry.unregister_chat_completion(model.registered_name, model.registration_id)
            if model.type == "embedding":
                self.endpoint_registry.unregister_embeddings(model.registered_name, model.registration_id)
            if model.type == "txt2img":
                self.endpoint_registry.unregister_image_generations(model.registered_name, model.registration_id)

        if options.purge and model_id in self.models_downloaded:
            model = self.models[instance][model_id]
            tasks: list[asyncio.Task[Any]] = []

            tasks.append(asyncio.create_task(fetch_from(f"{info.base_url}/api/delete", "DELETE", {"name": model_id})))

            await self.remove_modelfile(model_id, instance)

            del self.models_downloaded[model_id]

            if not model.hash:
                return

            for alias_model_id, alias_model in self.models[instance].items():
                if alias_model.hash == model.hash and model_id != alias_model_id:
                    await self._uninstall_model(instance, alias_model_id, UninstallModelIn(purge=False))
                    tasks.append(asyncio.create_task(fetch_from(f"{info.base_url}/api/delete", "DELETE", {"name": alias_model_id})))
                    if alias_model_id in self.models_downloaded:
                        del self.models_downloaded[alias_model_id]

            await asyncio.gather(*tasks)

    def get_default_context_value(self) -> int:
        """Ollama default context value.

        Based on: https://docs.ollama.com/context-length
        """
        if self.hardware.total_vram_gb < 24:
            return 4096  # 4 * 1024
        if self.hardware.total_vram_gb > 256:
            return 262144  # 256 * 1024
        return 32768  # 32 * 1024
