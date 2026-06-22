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
import hashlib
import json
import logging
import os
import re
import shutil
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, NamedTuple, TypedDict

import aiofiles
import aiohttp
from fastapi import HTTPException
from pydantic import BaseModel, StringConstraints

from server.applicationcontext import get_base_url
from server.config import get_main_dir
from server.docker import DockerImage, DockerOptions, DockerPath
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
    MemoryLoadComponent,
    MemoryLoadOut,
    MemoryLoadSession,
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
from server.utils.files import detect_context_window_from_path
from server.utils.hardware import GpuInfo, HardwarePartInfo, IntelGpuInfo, get_vram_gb
from server.utils.loading import Progress
from server.utils.ollama import raise_ollama_pull_error
from server.utils.size_fetcher import fetch_ollama_ref_bytes, fmt_size
from server.utils.vram_calculator import GGUF_QUANTS, ArchParams, estimate_vram_gb, parse_cache_type_bits, parse_parameter_count

logger = logging.getLogger("uvicorn.error")


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
    instance_name: str
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


@dataclass
class OllamaModelFileLine:
    instruction: Literal["FROM", "ADAPTER"]
    value: str

    def render(self) -> str:
        """Render line."""
        return f"{self.instruction} {self.value}"


class OllamaModelFile:
    def __init__(self, lines: list[str | OllamaModelFileLine]):
        self.lines = lines

    def render(self) -> str:
        """Render modelfile."""
        return "\n".join([(line.render() if isinstance(line, OllamaModelFileLine) else line) for line in self.lines])

    @staticmethod
    def parse(modelfile: str) -> "OllamaModelFile":
        """Parse ollama modelfile."""
        lines: list[str | OllamaModelFileLine] = []
        for line in modelfile.split("\n"):
            if line.startswith("FROM "):
                value = line[5:].strip()
                lines.append(OllamaModelFileLine(instruction="FROM", value=value))
            elif line.startswith("ADAPTER "):
                value = line[8:].strip()
                lines.append(OllamaModelFileLine(instruction="ADAPTER", value=value))
            else:
                lines.append(line)
        return OllamaModelFile(lines)


class ModelSize(NamedTuple):
    size_bytes: int
    parameters: int | None
    bytes_weight: float | None


class OllamaService(Base2Service[InstalledInfo, DownloadedInfo]):
    _MEMORY_LINE_RE = re.compile(
        r'msg="(?P<name>model weights|kv cache|compute graph|total memory)"'
        r"(?:\s+device=(?P<device>\S+))?"
        r'.*?size="(?P<size>[^"]+)"'
    )
    _MODEL_BLOB_RE = re.compile(r"blobs/sha256-([a-f0-9]+)")

    models: dict[str, dict[str, OllamaModel]]
    default_context_length: int
    _arch_cache: dict[tuple[str, str], ArchParams]
    _blob_to_model: dict[str, str]
    _model_manifest_loaded: bool
    _vram_cache: dict[tuple[str, str], float]

    @property
    def _supported_gpus(self) -> list[GpuInfo]:
        """Return GPUs supported by Ollama (including Intel via Vulkan)."""
        return self.hardware.gpus

    def _after_init(self) -> None:
        self.models = {}
        self.load_default_models("default")
        self.default_context_length = self.get_default_context_value()
        self._arch_cache = {}
        self._blob_to_model = {}
        self._model_manifest_loaded = False
        self._vram_cache = {}

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
                CustomModelField(type="oneof", name="type", description="Model type", values=["llm", "embedding", "txt2img"]),
                CustomModelField(
                    type="textarea",
                    name="modelfile",
                    description="Modelfile",
                    placeholder="FROM gemma3:1b\nPARAMETER num_ctx 8192",
                    required=False,
                ),
                CustomModelField(type="text", name="quantization", description="Quantization", placeholder="q4_0", required=False),
                CustomModelField(type="text", name="size", description="Model size", placeholder="1 GB", required=False),
            ]
        )

    _OLLAMA_DOCKER_ROOT = "/root/.ollama"

    def _docker_path_to_host(self, path: str) -> Path | None:
        """Translate a Docker-side path to the corresponding host path via the volume mapping.

        Returns None for paths that are not under the known volume mount.
        The volume is: {working_dir}/main -> /root/.ollama
        """
        working = self._get_working_dir() / "main"
        if path.startswith("./"):
            return working / path[2:]
        if path == self._OLLAMA_DOCKER_ROOT or path.startswith(self._OLLAMA_DOCKER_ROOT + "/"):
            rel = path[len(self._OLLAMA_DOCKER_ROOT) :].lstrip("/")
            return working / rel
        return None

    async def _local_file_size_bytes(self, path: str, instance: str) -> int | None:
        """Return size in bytes for a local file referenced from an Ollama modelfile."""
        try:
            host_path = self._docker_path_to_host(path)
            if host_path is not None:
                return host_path.stat().st_size if host_path.exists() else None
            # Absolute path outside the known volume: fall back to docker exec stat
            instance_info = self.get_instance_info(instance)
            if not instance_info.installed:
                return None
            compose_filepath = self.docker_service.get_docker_compose_file_path(instance_info.installed.docker.name)
            service_name = instance_info.installed.docker.name
            result = await self.docker_service.run_command_docker_compose(compose_filepath, service_name, f"stat -c%s {path}")
            return int(result.strip())
        except Exception:
            return None

    async def _fetch_ref_bytes(self, ref: str, instance: str) -> int | None:
        if ref.startswith(("/", "./")):
            return await self._local_file_size_bytes(ref, instance)
        return await fetch_ollama_ref_bytes(ref)

    async def _resolve_custom_model_size(self, spec: dict[str, Any], instance: str = "") -> str | None:
        try:
            modelfile: str = spec.get("modelfile") or ""
            if modelfile.strip():
                refs: list[str] = []
                for line in modelfile.splitlines():
                    stripped = line.strip()
                    if stripped.startswith("FROM "):
                        refs.append(stripped[5:].strip())
                    elif stripped.startswith("ADAPTER "):
                        refs.append(stripped[8:].strip())
                if not refs:
                    return None
                total = 0
                any_resolved = False
                for ref in refs:
                    try:
                        n = await self._fetch_ref_bytes(ref, instance)
                        if n is not None:
                            total += n
                            any_resolved = True
                    except Exception:
                        pass
                return fmt_size(total) if any_resolved else None
            model_id: str = spec.get("id", "")
            n = await self._fetch_ref_bytes(model_id, instance)
            return fmt_size(n) if n is not None else None
        except Exception:
            return None

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
                instance_name=instance,
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
            models = list(installed.models.copy().values())

            for model in models:
                if model.type == "llm":
                    self.endpoint_registry.unregister_chat_completion(model.registered_name, model.registration_id)
                if model.type == "embedding":
                    self.endpoint_registry.unregister_embeddings(model.registered_name, model.registration_id)
                if model.type == "txt2img":
                    self.endpoint_registry.unregister_image_generations(model.registered_name, model.registration_id)

            await asyncio.gather(
                *[
                    self._uninstall_model(instance, model.id, UninstallModelIn(purge=options.purge))
                    for model in models
                    if not self.is_model_installed_in_other_instance(instance, model.id)
                ]
            )

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

    async def get_memory_load(self, instance: str) -> MemoryLoadOut:
        """Parse docker logs and return per-model-load memory breakdown sessions."""
        info = self.get_instance_installed_info(instance)
        container_name = info.docker.container_name or ""
        raw = await self._get_docker_logs(container_name)

        current_model = None
        sessions: list[MemoryLoadSession] = []
        current: list[MemoryLoadComponent] = []
        for line in raw.splitlines():
            if m_model := self._MODEL_BLOB_RE.search(line):
                current_model = await self.resolve_blob(f"sha256-{m_model.group(1)}")

            if data := self._MEMORY_LINE_RE.search(line):
                name = data.group("name")
                device = data.group("device")
                size = data.group("size")

                if name == "total memory":
                    sessions.append(MemoryLoadSession(model=current_model, components=current, total=size))
                    current = []
                else:
                    current.append(MemoryLoadComponent(name=name, device=device, size=size))

        return MemoryLoadOut(sessions=sessions)

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

    async def _get_arch_params(self, instance: str, base_url: str, model_name: str) -> ArchParams | None:
        """Fetch architecture parameters for a model from the Ollama /api/show endpoint.

        Results are cached by (instance, model_name) to avoid cross-instance contamination.
        Returns None if the request fails or the response is missing critical fields.
        """
        cache_key = (instance, model_name)
        if cache_key in self._arch_cache:
            return self._arch_cache[cache_key]

        with suppress(Exception):
            result = await fetch_from(f"{base_url}/api/show", method="POST", data={"name": model_name})
            if result.status_code == 200:
                data = json.loads(result.data)
                info = data.get("model_info", {})
                arch = info.get("general.architecture", "")
                params = ArchParams(
                    hidden_size=info.get(f"{arch}.embedding_length", 0),
                    num_attention_heads=info.get(f"{arch}.attention.head_count", 1),
                    num_key_value_heads=info.get(f"{arch}.attention.head_count_kv", 1),
                    num_hidden_layers=info.get(f"{arch}.block_count", 0),
                    sliding_window=info.get(f"{arch}.attention.sliding_window"),
                )

                if params.hidden_size and params.num_hidden_layers:
                    self._arch_cache[cache_key] = params
                    return params

        return None

    async def _get_model_sizes(self, base_url: str) -> dict[str, ModelSize]:
        """Return {model_name: ModelSize} for all models known to the Ollama instance."""
        with suppress(Exception):
            result = await fetch_from(f"{base_url}/api/tags")
            if result.status_code == 200:
                return {
                    model["name"]: ModelSize(
                        model.get("size", 0),
                        parse_parameter_count(details.get("parameter_size", "")),
                        GGUF_QUANTS.get(details.get("quantization_level", "")),
                    )
                    for model in json.loads(result.data).get("models", [])
                    for details in (model.get("details", {}),)
                }

        return {}

    async def _get_vram_estimate(
        self,
        instance: str,
        base_url: str,
        model_name: str,
        size_bytes: int,
        num_ctx: int | None = None,
        parameters: int | None = None,
        bytes_weight: float | None = None,
        num_parallel: int | None = None,
    ) -> float | None:
        arch = await self._get_arch_params(instance, base_url, model_name)

        if arch is None or num_ctx is None:
            return None

        cache_type = os.environ.get("OLLAMA_KV_CACHE_TYPE", "f16")
        cache_bit = parse_cache_type_bits(cache_type)
        num_parallel = num_parallel or int(os.environ.get("OLLAMA_NUM_PARALLEL", 1))
        return estimate_vram_gb(arch, size_bytes, num_ctx, cache_bit, num_parallel, parameters, bytes_weight)

    async def _get_vram_from_logs(self, instance: str, ollama_name: str, memory_load: MemoryLoadOut | None = None) -> float | None:
        """Return actual VRAM usage from Docker logs for the most recent load of a model."""
        try:
            if memory_load is None:
                memory_load = await self.get_memory_load(instance)
        except Exception:
            return None

        result: float | None = None
        for session in memory_load.sessions:
            if (session.model == ollama_name) and (parsed := get_vram_gb(session.total)) and (parsed > 0):
                result = round(parsed, 2)

        return result

    async def get_loaded_model_info(self, instance: str) -> dict[str, int] | None:
        """Return {model_name: context_length} for models currently loaded in VRAM."""
        info = self.get_instance_installed_info(instance)
        with suppress(Exception):
            result = await fetch_from(f"{info.base_url}/api/ps")
            if result.status_code == 200:
                data = json.loads(result.data)
                return {m["name"]: m.get("context_length", 0) for m in data.get("models", [])}

        return {}

    def _effective_context(self, model_context: int | None, parsed_options: OllamaOptions) -> int:
        """Return the context Ollama runs with for an idle model: native window capped by the service context."""
        service_context = parsed_options.context_length or self.default_context_length
        return self.get_default_context_window(model_context, service_context)

    async def _resolve_vram_info(
        self,
        instance: str,
        ollama_name: str,
        model_context: int | None,
        loaded_info: dict[str, int],
        sizes: dict[str, ModelSize],
        base_url: str,
        num_parallel: int | None,
        memory_load: MemoryLoadOut | None = None,
    ) -> tuple[bool, float | None]:
        """Return (is_loaded, vram_estimate_gb) for a model."""
        is_loaded = ollama_name in loaded_info
        cache_key = (instance, ollama_name)
        if not is_loaded:
            self._vram_cache.pop(cache_key, None)
            size_bytes, parameters, bpw = sizes.get(ollama_name, ModelSize(0, None, None))
            vram_estimate = await self._get_vram_estimate(
                instance, base_url, ollama_name, size_bytes, model_context, parameters, bpw, num_parallel
            )
            return False, vram_estimate

        if cache_key in self._vram_cache:
            return True, self._vram_cache[cache_key]

        vram_estimate = await self._get_vram_from_logs(instance, ollama_name, memory_load)

        if vram_estimate is None:
            num_ctx = loaded_info.get(ollama_name) or model_context
            size_bytes, parameters, bpw = sizes.get(ollama_name, ModelSize(0, None, None))
            vram_estimate = await self._get_vram_estimate(
                instance, base_url, ollama_name, size_bytes, num_ctx, parameters, bpw, num_parallel
            )

        if vram_estimate is not None:
            self._vram_cache[cache_key] = vram_estimate

        return True, vram_estimate

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
            loaded_info = await self.get_loaded_model_info(instance_name)
            sizes = await self._get_model_sizes(info.base_url)
            memory_load: MemoryLoadOut | None = None
            with suppress(Exception):
                memory_load = await self.get_memory_load(instance_name)

            for model_id, model in instance_models.items():
                if model_id in info.models:
                    installed = info.models[model_id].get_info()
                else:
                    installed = self._get_model_installed_info(instance_name, model_id)

                if filters.installed is None or filters.installed == bool(installed):
                    is_loaded: bool | None = None
                    vram_estimate: float | None = None
                    ollama_name: str = (info.models[model_id].internal_name if model_id in info.models else None) or model_id
                    if loaded_info is not None and model_id in info.models:
                        is_loaded, vram_estimate = await self._resolve_vram_info(
                            instance_name,
                            ollama_name,
                            self._effective_context(model.context, info.parsed_options),
                            loaded_info,
                            sizes,
                            info.base_url,
                            info.parsed_options.num_parallel,
                            memory_load,
                        )
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
                            is_loaded=is_loaded,
                            vram_estimate_gb=vram_estimate,
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
        loaded_info = await self.get_loaded_model_info(instance)
        sizes = await self._get_model_sizes(info.base_url)
        is_loaded: bool | None = None
        vram_estimate: float | None = None
        ollama_name: str = (info.models[model_id].internal_name if model_id in info.models else None) or model_id
        if loaded_info is not None:
            is_loaded, vram_estimate = await self._resolve_vram_info(
                instance,
                ollama_name,
                self._effective_context(model.context, info.parsed_options),
                loaded_info,
                sizes,
                info.base_url,
                info.parsed_options.num_parallel,
            )
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
            is_loaded=is_loaded,
            vram_estimate_gb=vram_estimate,
        )

    async def _download_with_ollama(self, stream: Stream[StreamChunk], base_url: str, model_id: str, model_size: str) -> None:
        progress = Progress(convert_size_to_bytes(model_size) or 0)
        last_values: dict[str, int] = {}

        stream.emit(StreamChunkProgress(type="progress", stage="download", value=0, data={}))
        async for ollama_stream in stream_fetch_from(f"{base_url}/api/pull", "POST", {"model": model_id}, timeout=24 * 60 * 60):
            if (ollama_stream.status_code != 200 and ollama_stream.status_code != 201) or "error" in ollama_stream.data:
                raise_ollama_pull_error(ollama_stream.data)

            data_cleared: list[str] = ollama_stream.data.rstrip().split("\n")
            records = [json.loads(s) for s in data_cleared if s]
            if progress.max != 0:
                for record in records:
                    if value := record.get("completed"):
                        digest: str = record.get("digest") or ""
                        increment = max(0, value - last_values.get(digest, 0))
                        if increment > 0:
                            progress.add_to_actual_value(increment)
                        last_values[digest] = value

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

    async def create_model_from_modelfile(self, info: InstalledInfo, model: str, model_path: str, quantization: str | None) -> str:
        """Send message to ollama to create model from Modelfile.."""
        compose_filepath = self.docker_service.get_docker_compose_file_path(info.docker.name)
        service_name = info.docker.name
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
        dest_path: Path | None = None,
    ) -> None:
        local_models_dir = Path(self._get_working_dir()) / "main" / "custom"
        model_path = Path(model_url)

        if model_url.startswith(("http://", "https://")):
            if dest_path is not None:
                dir = dest_path.parent
                additional_params: tuple[str, ...] = (dest_path.name,)
            else:
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

    def _get_local_modelfile_path(self, model: str, instance: str) -> Path:
        """Return local (host) modelfile path."""
        return Path(self._get_working_dir()) / "main" / "custom" / model / instance / "Modelfile"

    def _get_docker_modelfile_path(self, model: str, instance: str) -> Path:
        """Return docker modelfile path."""
        return Path("root/.ollama/custom") / model / instance / "Modelfile"

    async def remove_modelfile(self, model: str, instance: str) -> None:
        """Remove modelfile."""
        local_modelfile_path = self._get_local_modelfile_path(model, instance)
        local_modelfile_path.unlink(missing_ok=True)
        parent_dir = local_modelfile_path.parent
        if parent_dir.is_dir() and not any(parent_dir.iterdir()):
            parent_dir.rmdir()

    async def save_modelfile(self, model: str, instance: str, modelfile: str) -> str:
        """Save modelfile."""
        docker_modelfile_path = self._get_docker_modelfile_path(model, instance)
        local_modelfile_path = self._get_local_modelfile_path(model, instance)

        # Remove old Modelfile, code below can be deleted after migration in all instances
        old_path = Path(self._get_working_dir()) / "main" / "custom" / model / "Modelfile"
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
        dest_path: Path | None = None,
    ) -> None:
        if model_id not in self.models_download_progress:
            self.models_download_progress[model_id] = input_stream
            try:
                await self._download_with_downloader(input_stream, base_url, model_url, model_id, size, i, models_quantity, dest_path)
            finally:
                del self.models_download_progress[model_id]
        else:
            chunk: StreamChunk
            async for chunk in self.models_download_progress[model_id].as_generator():
                if chunk.get("type") == "progress" and chunk.get("stage") == "download":
                    input_stream.emit(chunk)
                else:
                    break

    def _get_modelfile_file_name_from_url(self, url: str) -> str:
        new_name = hashlib.sha1(url.encode()).hexdigest()
        path_model = Path(url)
        return new_name + ".gguf" if path_model.suffix == ".gguf" else new_name

    def _remove_paths(self, paths: Sequence[Path | str]) -> None:
        """Remove all given files and directories."""
        for local_path in paths:
            if Path(local_path).is_dir():
                with suppress(Exception):
                    shutil.rmtree(local_path)
            else:
                Path(local_path).unlink(missing_ok=True)

    async def _install_from_modelfile(  # noqa: C901
        self,
        input_stream: Stream[StreamChunk],
        model_id: str,
        modelfile_recipe: str,
        internal_model_name: str,
        instance_info: InstalledInfo,
        model: OllamaModel,
    ) -> int | None:
        """Download assets, create model in docker from modelfile, and detect and return context window."""
        paths_to_remove_on_error: list[Path] = []
        try:
            base_path = DockerPath(
                docker_path=Path("/root/.ollama"),
                local_path=Path(self._get_working_dir()) / "main",
            )
            model_context: int | None = None
            modelfile = OllamaModelFile.parse(modelfile_recipe)
            for line in modelfile.lines:
                if isinstance(line, OllamaModelFileLine):
                    if line.value.startswith(("http://", "https://")):
                        url = line.value
                        model_path = base_path.add(Path("custom") / model_id / self._get_modelfile_file_name_from_url(url))
                        if not model_path.local_path.exists():
                            paths_to_remove_on_error.append(model_path.local_path)
                            await self._download_model_or_set_progress(
                                input_stream,
                                instance_info.base_url,
                                url,
                                model_id,
                                model.size,
                                0,
                                1,
                                dest_path=model_path.local_path,
                            )
                        line.value = str(model_path.docker_path)
                        if line.instruction == "FROM":
                            model_context = await detect_context_window_from_path(model_path.local_path)
                    elif line.value.startswith("./"):
                        model_path = base_path.add(Path(line.value))
                        line.value = str(model_path.docker_path)
                        if not model_path.local_path.exists():
                            raise HTTPException(  # noqa: TRY301
                                status_code=400,
                                detail=f"Path {model_path.local_path} from {line.instruction} line does not exist",
                            )
                        if line.instruction == "FROM":
                            model_context = await detect_context_window_from_path(model_path.local_path)
                    else:
                        model_to_download = line.value
                        if not await self.is_model_installed(instance_info.base_url, model_to_download):
                            await self._download_model_or_set_progress(
                                input_stream, instance_info.base_url, model_to_download, model_id, model.size, 0, 1
                            )
            modelfile_path = await self.save_modelfile(model_id, instance_info.instance_name, modelfile.render())
            await self.create_model_from_modelfile(instance_info, internal_model_name, modelfile_path, model.quantization)
            return model_context  # noqa: TRY300
        except aiohttp.ClientConnectorError as e:
            self._remove_paths(paths_to_remove_on_error)
            raise HTTPException(status_code=400, detail=f"Cannot connect to model source: {e}") from e
        except HTTPException:
            self._remove_paths(paths_to_remove_on_error)
            raise
        except Exception as e:
            self._remove_paths(paths_to_remove_on_error)
            raise HTTPException(status_code=400, detail="Cannot install modelfile") from e

    def get_default_context_window(self, model_context: int | None, service_context: int) -> int:
        """Return minimum default context length from model or service context length."""
        return min(model_context, service_context) if model_context else service_context

    async def _install_model(  # noqa: C901
        self, instance: str, model_id: str, options: InstallModelIn
    ) -> PromiseWithProgress[InstallModelOut, StreamChunk]:
        self._arch_cache.pop((instance, model_id), None)
        self._model_manifest_loaded = False
        self._blob_to_model = {}
        self._log_cache.clear()
        parsed_model_options = try_parse_pydantic(OllamaModelOptions, options.spec) if options.spec else OllamaModelOptions()
        info = self.get_instance_installed_info(instance)

        if not self.models.get(instance):
            self.models[instance] = {}

        if model_id in info.models:
            return PromiseWithProgress(value=InstallModelOut(status="OK", details="Already installed"))

        if model_id not in self.models[instance]:
            raise HTTPException(400, f"Model {model_id!r} not found")

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
            model_context = model.context
            if modelfile_recipe:
                new_model_context = await self._install_from_modelfile(input_stream, model_id, modelfile_recipe, internal_name, info, model)
                if new_model_context is not None:
                    model_context = new_model_context

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
            model_max_context_window = model_context
            service_max_context_window = self.default_context_length
            default_context_window = self.get_default_context_window(model_max_context_window, service_max_context_window)
            max_context_window = model_form_context_window or model_max_context_window or service_max_context_window

            context_window = model_form_context_window or service_form_context_window or default_context_window
            context_window = min(context_window, max_context_window)
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
                    ollama_chat=ProxyOptions(url=f"{info.base_url}/api/chat", rewrite_model_to=rewrite_model_to),
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
        self._arch_cache.pop((instance, model_id), None)
        self._model_manifest_loaded = False
        self._blob_to_model = {}
        self._log_cache.clear()
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

    async def resolve_blob(self, blob_path: str) -> str:
        """Resolve a blob SHA256 digest (from Docker logs) to a human-readable model name (e.g. 'llama3').

        Ollama logs reference model layers by blob digest rather than model name.
        This method looks up the digest in a manifest index built from
        models/manifests/registry.ollama.ai to find the corresponding model tag,
        then returns only the final name component.
        Returns the original path if no digest is found, or 'unknown (<digest>)' if the digest is not in the index.
        """
        await self._load_model_manifest_index()
        sha_regex = r"sha256[-:]([a-f0-9]+)"
        if match := re.search(sha_regex, blob_path):
            digest = f"sha256:{match.group(1)}"
            resolved = self._blob_to_model.get(digest)
            return resolved.split("/")[-1] if resolved else f"unknown ({digest[:12]})"

        return blob_path

    async def _load_model_manifest_index(self) -> None:
        if self._model_manifest_loaded:
            return

        manifests_base = Path(self._get_working_dir()) / "main" / "models" / "manifests"
        if not manifests_base.exists():
            return

        manifest_paths = await asyncio.to_thread(
            lambda: [p for p in manifests_base.rglob("*") if p.is_file() and not p.name.startswith(".")]
        )

        new_index: dict[str, str] = {}
        for manifest_path in manifest_paths:
            try:
                async with aiofiles.open(manifest_path) as f:
                    content = await f.read()
                data = json.loads(content)
                rel = manifest_path.relative_to(manifests_base)
                model_tag = f"{rel.parent}:{rel.name}" if rel.parent.parts else rel.name
                digests = [data.get("config", {}).get("digest")]
                digests += [layer.get("digest") for layer in data.get("layers", [])]
                new_index.update({d: model_tag for d in digests if d})
            except Exception:
                logger.exception("Failed to parse manifest %s", manifest_path)

        self._blob_to_model = new_index
        self._model_manifest_loaded = True
