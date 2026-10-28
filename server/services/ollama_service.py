# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Ollama service."""

import asyncio
import errno
import hashlib
import json
import os
import shutil
import time
from collections.abc import Sequence
from contextlib import aclosing, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, NamedTuple, NotRequired, TypedDict
from urllib.parse import urlparse

import aiofiles
import aiohttp
from fastapi import HTTPException
from pydantic import BaseModel, Field, StringConstraints, field_validator

from server.applicationcontext import get_base_url
from server.config import get_main_dir
from server.docker import DockerImage, DockerOptions, DockerPath
from server.endpointregistry import ProxyOptions, RegistrationId, RegistrationOptions
from server.models.api import DECISION_ENDPOINTS, EMBEDDINGS_ENDPOINTS, IMG_ENDPOINTS, LLM_ENDPOINTS, ModelProps
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
    CatalogRefreshOut,
    InstallServiceIn,
    InstallServiceProgress,
    ServiceField,
    ServiceOptions,
    ServiceSpecification,
    UninstallServiceIn,
)
from server.services.base2_service import Base2Service, CustomModel, Instance, InstanceConfig, ModelConfig
from server.utils.core import (
    CommandFailedError,
    DownloadedPacket,
    PreDownloadPacket,
    PromiseWithProgress,
    Stream,
    StreamChunk,
    StreamChunkProgress,
    convert_size_to_bytes,
    fetch_from,
    positive_context_window_or_none,
    stream_fetch_from,
    try_parse_pydantic,
)
from server.utils.files import detect_context_window_from_path
from server.utils.hardware import GpuInfo, HardwarePartInfo, IntelGpuInfo
from server.utils.loading import Progress
from server.utils.logger import logger
from server.utils.ollama import (
    GGUF_QUANTIZATION_UNSUPPORTED_MESSAGE,
    decision_models_unsupported_message,
    get_ollama_create_error_message,
    raise_ollama_pull_error,
    supports_decision_models,
    supports_gguf_quantization,
)
from server.utils.ollama_catalog import OllamaCatalogClient
from server.utils.ollama_registry import (
    OLLAMA_REGISTRY_HOST,
    BlobLocks,
    OllamaModelRef,
    OllamaRegistryClient,
    OllamaRegistryError,
    parse_model_ref,
)
from server.utils.registry_client import image_without_registry_prefix, registry_for
from server.utils.size_fetcher import fetch_ollama_ref_bytes, fmt_size
from server.utils.vram_calculator import (
    GGUF_QUANTS,
    GIB,
    ArchParams,
    estimate_vram_gb,
    get_quant_overhead,
    parse_cache_type_bits,
    parse_parameter_count,
)

_PROGRESS_LOG_STEP = 0.1  # log every ~10% of progress so the console isn't flooded with updates
_WARMTH_POLL_INTERVAL_SECONDS = 15  # matches the staleness already tolerated for the UI's loaded-model display

_OLLAMA_VERSION_TIMEOUT_SECONDS = 5  # the version pre-check is best-effort; a slow Ollama just skips it
_DEFAULT_KV_CACHE_TYPE = "f16"  # matches Ollama's own real default when an instance doesn't set one
_DEFAULT_NUM_PARALLEL = 1  # conservative stand-in for Ollama's real "auto" (0) behavior, not a reproduction of it

_SHOW_TIMEOUT_SECONDS = 5  # /api/show should be near-instant (local metadata lookup); don't let a hung instance stall list calls
_SHOW_FAILURE_CACHE_SECONDS = 30  # how long a failed /api/show is remembered, so a down instance isn't re-hit on every list call


type Quantization = Literal[
    # GGUF sources (Ollama < 0.34.1)
    "q4_0",
    "q4_1",
    "q5_0",
    "q5_1",
    "q8_0",
    "q3_K_S",
    "q3_K_M",
    "q3_K_L",
    "q4_K_S",
    "q4_K_M",
    "q5_K_S",
    "q5_K_M",
    "q6_K",
    # safetensors sources (Ollama >= 0.34.1)
    "int4",
    "int8",
    "nvfp4",
    "mxfp4",
    "mxfp8",
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
    type: Literal["llm", "embedding", "decision"]
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
    decision: NotRequired[list[OllamaRegistryEntry]]


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
        for tag in registry.get("decision", []):
            map[tag["name"]] = OllamaModel(
                id=tag["name"],
                size=tag["size"],
                type="decision",
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
    image=DockerImage(name="ollama/ollama:0.40.12", size="3.4 GB"),
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
    num_parallel: int | None = Field(default=None, ge=0)  # 1
    keep_alive: str = ""  # 60m
    is_flash_attention: bool | None = None  # False
    max_loaded_models: int | None = None  # 1
    kv_cache_type: str = ""  # f16
    context_length: int | None = None  # 4096
    mode: Literal["online", "offline"] = "online"

    @field_validator("mode", mode="before")
    @classmethod
    def _blank_mode_is_online(cls, value: object) -> object:
        # An unset option arrives as None or a blank string from the form.
        return "online" if value is None or value == "" else value


class OllamaModelOptions(BaseModel):
    rename: str | None = None
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
    quantization_level: str | None = None


class LoadedModelInfo(NamedTuple):
    context_length: int
    vram_gb: float | None
    """Ollama's own live measurement of VRAM usage for this model (via /api/ps size_vram), if reported."""


@dataclass
class ModelCapabilities:
    """Modality support resolved from Ollama's `/api/show` `capabilities` list.

    `None` means unresolved (the field couldn't be determined, e.g. the instance's Ollama
    version doesn't report `capabilities`, or `/api/show` failed) - not "unsupported".
    """

    supports_vision: bool | None = None
    supports_audio: bool | None = None


class OllamaService(Base2Service[InstalledInfo, DownloadedInfo]):
    models: dict[str, dict[str, OllamaModel]]
    _dynamic_models: dict[str, OllamaModel]
    default_context_length: int
    _arch_cache: dict[tuple[str, str], ArchParams]
    _capabilities_cache: dict[tuple[str, str], ModelCapabilities]
    _show_failure_at: dict[tuple[str, str], float]
    """Monotonic timestamp of the last failed `/api/show` fetch, keyed like `_capabilities_cache`.

    Lets `_get_capabilities`/`_get_arch_params` skip re-hitting a down/hung instance for
    `_SHOW_FAILURE_CACHE_SECONDS` instead of retrying on every call.
    """
    _vram_cache: dict[tuple[str, str], float]
    _installing: set[tuple[str, str]]
    _warmth_poll_tasks: dict[str, asyncio.Task[None]]
    _ps_query_failing: set[str]
    _blob_locks: BlobLocks

    @property
    def _supported_gpus(self) -> list[GpuInfo]:
        """Return GPUs supported by Ollama (including Intel via Vulkan)."""
        return self.hardware.gpus

    def _after_init(self) -> None:
        self._dynamic_models = {}
        self.load_default_models("default")
        self.default_context_length = self.get_default_context_value()
        self._arch_cache = {}
        self._capabilities_cache = {}
        self._show_failure_at = {}
        self._vram_cache = {}
        self._installing = set()
        self._warmth_poll_tasks = {}
        self._ps_query_failing = set()
        self._blob_locks = BlobLocks()

    def load_default_models(self, instance: str) -> None:
        """Load default models to instance (merges static catalog with dynamic overlay, restoring any installed model dropped from both)."""
        self.models[instance] = {**_const.models, **{k: v for k, v in self._dynamic_models.items() if k not in _const.models}}
        persisted = self.instances_info.get(instance)
        for model_config in (persisted.config.models if persisted else None) or []:
            if model_config.model_id not in self.models[instance] and model_config.definition:
                self._restore_model_definition(instance, model_config.model_id, model_config.definition)

    async def refresh_catalog(self) -> PromiseWithProgress[CatalogRefreshOut, StreamChunk]:
        """Fetch trending models from the Ollama library and merge them into the dynamic catalog.

        Fast enough to complete within the request, so it's wrapped in an already-resolved
        promise rather than a background task — this only exists to match the shared
        `BaseService.refresh_catalog` contract used generically across service types.
        """
        client = OllamaCatalogClient()
        entries = await client.fetch_trending()

        added = 0
        for entry in entries:
            model_id = str(entry["id"])
            if model_id in _const.models or model_id in self._dynamic_models:
                continue
            self._dynamic_models[model_id] = OllamaModel(
                id=model_id,
                size=str(entry["size"]),
                type="llm",
                hash=str(entry["hash"]),
                context=0,
            )
            added += 1

        for instance in list(self.models):
            self.load_default_models(instance)

        total = len(_const.models) + len(self._dynamic_models)
        return PromiseWithProgress(value=CatalogRefreshOut(added=added, total=total))

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
                    default=str(_DEFAULT_NUM_PARALLEL),
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
                    description=f"Quantization type for the K/V cache (default: {_DEFAULT_KV_CACHE_TYPE}, available: q4_0, q8_0) "
                    "(OLLAMA_KV_CACHE_TYPE)",
                    required=False,
                ),
                ServiceField(
                    type="number",
                    name="context_length",
                    description="Default context length (OLLAMA_CONTEXT_LENGTH)",
                    required=False,
                ),
                ServiceField(
                    type="oneof",
                    name="mode",
                    description="online: the Ollama container pulls models itself. "
                    "offline: Infra downloads models from the Ollama registry, so the container does not need to reach it; "
                    "other registries (for example hf.co) are rejected, models with a direct download URL still work",
                    required=False,
                    default="online",
                    values=["online", "offline"],
                ),
                ServiceField(
                    type="docker-tags",
                    name="image_version",
                    description="Docker image version",
                    docker_image=_const.image.name.split(":")[0],
                    required=False,
                ),
            ]
        )

        return ServiceSpecification(fields=fields)

    def get_model_spec(self) -> ModelSpecification:
        """Return the model specification."""
        return ModelSpecification(
            fields=[
                ModelField(type="text", name="rename", description="Rename the model, hiding its original name", required=False),
                ModelField(
                    type="text", name="alias", description="Additional name for the model, the original name keeps working", required=False
                ),
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
                CustomModelField(type="oneof", name="type", description="Model type", values=["llm", "embedding", "txt2img", "decision"]),
                CustomModelField(
                    type="textarea",
                    name="modelfile",
                    description="Modelfile",
                    placeholder="FROM gemma3:1b\nPARAMETER num_ctx 8192",
                    required=False,
                ),
                CustomModelField(type="text", name="quantization", description="Quantization", placeholder="int4", required=False),
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
            service_name = instance_info.installed.docker.service_name
            result = await self.docker_service.run_command_docker_compose(compose_filepath, service_name, ["stat", "-c%s", path])
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

    def _generate_instance_config(self, instance: str, info: InstalledInfo | None, custom: list[CustomModel] | None) -> InstanceConfig:
        return InstanceConfig(
            options=info.options if info else None,
            models=[
                ModelConfig(
                    model_id=x.id,
                    options=x.options,
                    definition=self.models[instance][x.id].model_dump(mode="json")
                    if x.id in self.models[instance]
                    else self._get_persisted_model_definition(instance, x.id),
                )
                for x in info.models.values()
            ]
            if info
            else [],
            custom=custom,
            container_name=info.docker.container_name if info else None,
        )

    def _restore_model_definition(self, instance: str, model_id: str, definition: dict[str, Any]) -> None:
        self.models[instance][model_id] = OllamaModel.model_validate(definition)

    def _get_image(self, image_version: str | None = None) -> DockerImage:
        if image_version:
            base = _const.image.name.split(":")[0]
            return DockerImage(name=f"{base}:{image_version}", size=_const.image.size)
        return _const.image

    def get_default_docker_image_name(self) -> str | None:
        """Return the default docker image, for the upfront platform-compatibility check."""
        return self._get_image().name

    async def get_docker_tags(self, hardware: str | None) -> list[str]:
        """Fetch available Docker image tags from Docker Hub for the Ollama image."""
        base_image = _const.image.name.split(":")[0]
        client = registry_for(base_image, self.config.docker_hub_token or None)
        tags = await client.get_tags(image_without_registry_prefix(base_image))
        return self.filter_docker_tags(tags, hardware)

    def get_default_docker_tag(self, hardware: str | None) -> str | None:  # noqa: ARG002
        """Return the pinned default Ollama image tag."""
        return _const.image.name.split(":")[1]

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

    def _create_model_store_dirs(self) -> None:
        """Create the model store directories as the Infra user, so they are not created by the Docker daemon as root.

        Infra writes models there in offline mode, which fails on directories owned by root when Infra does not run as root.
        A store left owned by root by an earlier install does not fail the install: `_ensure_model_store_writable` takes it
        over before a model download.
        """
        models_dir = Path(self._get_working_dir()) / "main" / "models"
        library_dir = models_dir / "manifests" / OLLAMA_REGISTRY_HOST / "library"
        try:
            (models_dir / "blobs").mkdir(parents=True, exist_ok=True)
            library_dir.mkdir(parents=True, exist_ok=True)
            # Ollama removes empty directories under manifests/ on startup and after `ollama rm`, then recreates them as root.
            (library_dir / ".keep").touch(exist_ok=True)
        except PermissionError as e:
            logger.warning("Cannot prepare the Ollama model store %s, it is probably owned by root: %s", models_dir, e)

    def _model_store_paths_to_write(self, ref: OllamaModelRef) -> list[Path]:
        """Return the existing directories Infra has to write to when it stores the model `ref` in the Ollama model store."""
        models_dir = Path(self._get_working_dir()) / "main" / "models"
        paths: list[Path] = []
        for path in (models_dir, models_dir / "blobs", models_dir / ".infra-downloads", (models_dir / ref.manifest_path).parent):
            while not path.exists() and path != path.parent:
                path = path.parent
            paths.append(path)
        return paths

    async def _ensure_model_store_writable(self, ref: OllamaModelRef, info: InstalledInfo | None) -> None:
        """Take the model store back from root when Ollama (running as root) created directories there that Infra cannot write to.

        Checked before the download, so that a permission problem does not show up only after gigabytes were downloaded. The
        `chown` runs in the Ollama container, only when a directory is not writable: on rootless Docker the check passes and
        the container's uid does not map to the Infra user.
        """
        if info is None or all(os.access(path, os.W_OK | os.X_OK) for path in self._model_store_paths_to_write(ref)):
            return
        models_dir = Path(self._get_working_dir()) / "main" / "models"
        owner = f"{os.getuid()}:{os.getgid()}"
        try:
            compose_filepath = self.docker_service.get_docker_compose_file_path(info.docker.name)
            await self.docker_service.run_command_docker_compose(
                compose_filepath, info.docker.service_name, ["chown", "-R", owner, self._OLLAMA_DOCKER_ROOT + "/models"]
            )
        except Exception as e:
            logger.warning("Taking over the Ollama model store %s failed: %s", models_dir, e, exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"The Ollama model store {models_dir} is not writable by Infra and changing its owner from the "
                f"Ollama container failed: {e}. Run: sudo chown -R {owner} {models_dir}",
            ) from e

    async def _validate_update_options(self, options: InstallServiceIn) -> None:
        """Normalize the hardware spec and validate the requested image_version, if any."""
        if "hardware" not in options.spec:
            options.spec["hardware"] = options.spec.get("gpu", self.docker_service.has_gpu_support)
        options.spec["hardware"] = self.canonicalize_hardware_spec(options.spec["hardware"])
        await self.validate_docker_image_version(options.spec.get("image_version"), options.spec.get("hardware"))
        try_parse_pydantic(OllamaOptions, options.spec)

    async def _install_instance(
        self, instance: str, options: InstallServiceIn, container_name_override: str | None = None
    ) -> PromiseWithProgress[InstalledInfo, StreamChunk]:
        if not self.models.get(instance):
            self.load_default_models(instance)

        await self._validate_update_options(options)
        parsed_options = try_parse_pydantic(OllamaOptions, options.spec)
        image_version = options.spec.get("image_version")
        image = self._get_image(image_version)
        await self._verify_docker_image(image.name, options.ignore_warnings)

        async def func(stream: Stream[StreamChunk]) -> InstalledInfo:
            await self._download_image_or_set_progress(stream, image)
            self.service_downloaded = True

            stream.emit(StreamChunkProgress(type="progress", stage="install", value=0, data={}))
            self._create_model_store_dirs()
            volumes = [f"{self._get_working_dir()}/main:/root/.ollama"]
            subnet = self.docker_service.get_docker_subnet()
            hardware = self.get_specified_hardware_parts(parsed_options.hardware)
            envs = self._build_env_vars(parsed_options, hardware)
            service_name = f"{self.get_service_id(instance)}"
            docker_options = DockerOptions(
                name=service_name,
                container_name=container_name_override or self.docker_service.get_docker_container_name(service_name),
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
            docker_exposed_port, _, _ = await self.docker_service.install_and_run_docker(docker_options)
            container_host = self.docker_service.get_container_host(subnet, docker_options)
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
            self._start_warmth_poll_task(instance)

            return info

        return PromiseWithProgress(func=func)

    async def _uninstall_instance(self, instance: str, options: UninstallServiceIn) -> None:  # noqa: C901
        self._stop_warmth_poll_task(instance)
        installed = self.get_instance_info(instance).installed
        if installed:
            models = list(installed.models.copy().values())

            for model in models:
                if (instance, model.id) in self._installing:
                    raise HTTPException(409, f"Model {model.id!r} is currently being installed; cancel it before deleting the instance")

            for model in models:
                if model.type == "llm":
                    self.endpoint_registry.unregister_chat_completion(model.registered_name, model.registration_id)
                if model.type == "embedding":
                    self.endpoint_registry.unregister_embeddings(model.registered_name, model.registration_id)
                if model.type == "txt2img":
                    self.endpoint_registry.unregister_image_generations(model.registered_name, model.registration_id)
                if model.type == "decision":
                    self.endpoint_registry.unregister_systemone(model.registered_name, model.registration_id)

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
            if not any(i.installed for i in self.instances_info.values()):
                self.service_downloaded = False
                await self.docker_service.remove_image(_const.image.name)
                if installed:
                    await self.docker_service.remove_image(installed.docker.image)
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
        self._stop_warmth_poll_task(instance)
        await self._stop_docker(installed.docker)

    async def _unload_model_from_vram(self, model_id: str, ollama_name: str, base_url: str) -> None:
        """Force-unload the model from VRAM by setting its keep_alive to 0."""
        try:
            result = await fetch_from(f"{base_url}/api/generate", "POST", {"model": ollama_name, "keep_alive": 0}, timeout=10)
        except Exception:
            logger.warning("Failed to unload model %r from VRAM", model_id, exc_info=True)
            return
        if result.status_code != 200:
            logger.warning("Failed to unload model %r from VRAM: %s (%s)", model_id, result.data, result.status_code)

    async def _delete_ollama_model(self, model_id: str, base_url: str) -> None:
        """Delete a model from the Ollama instance, logging failures instead of raising."""
        try:
            result = await fetch_from(f"{base_url}/api/delete", "DELETE", {"name": model_id}, timeout=15)
        except Exception:
            logger.warning("Failed to delete model %r from Ollama", model_id, exc_info=True)
            return
        if result.status_code != 200:
            logger.warning("Failed to delete model %r from Ollama: %s (%s)", model_id, result.data, result.status_code)

    def get_docker_options(self, instance: str, model_id: str | None) -> DockerOptions:
        """Return the resolved DockerOptions for this instance/model."""
        info = self.get_instance_installed_info(instance)
        if model_id:
            raise HTTPException(400, "Docker is not bound with this object")

        return info.docker

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

    async def _fetch_show_info(self, instance: str, base_url: str, model_name: str) -> None:
        """Fetch `/api/show` once and populate both the arch-params and capabilities caches from it.

        Kept as a single POST shared by `_get_arch_params` and `_get_capabilities` so callers that
        need both (e.g. the VRAM estimate path plus capability detection) don't double-fetch. A
        bounded timeout keeps a down/hung instance from stalling callers on aiohttp's 300s default;
        a failure is remembered for `_SHOW_FAILURE_CACHE_SECONDS` so repeated lookups within that
        window don't keep re-hitting the network.
        """
        cache_key = (instance, model_name)
        try:
            result = await fetch_from(f"{base_url}/api/show", method="POST", data={"name": model_name}, timeout=_SHOW_TIMEOUT_SECONDS)
            if result.status_code != 200:
                self._show_failure_at[cache_key] = time.monotonic()
                return
            data = json.loads(result.data)
            info = data.get("model_info", {})
            arch = info.get("general.architecture", "")
            params = ArchParams(
                hidden_size=info.get(f"{arch}.embedding_length") or 0,
                num_attention_heads=info.get(f"{arch}.attention.head_count") or 1,
                num_key_value_heads=info.get(f"{arch}.attention.head_count_kv"),
                num_hidden_layers=info.get(f"{arch}.block_count") or 0,
                sliding_window=info.get(f"{arch}.attention.sliding_window"),
            )

            if params.hidden_size and params.num_hidden_layers:
                self._arch_cache[cache_key] = params

            # Modern Ollama reports a top-level "capabilities" list (e.g. ["completion", "vision",
            # "tools"]). When present it's authoritative for vision support; Ollama doesn't publicly
            # document an "audio" tag, so its absence must stay unresolved (None) rather than False.
            capabilities = data.get("capabilities")
            if isinstance(capabilities, list):
                self._capabilities_cache[cache_key] = ModelCapabilities(
                    supports_vision="vision" in capabilities,
                    supports_audio=True if "audio" in capabilities else None,
                )
            else:
                self._capabilities_cache[cache_key] = ModelCapabilities()
            self._show_failure_at.pop(cache_key, None)
        except Exception:
            self._show_failure_at[cache_key] = time.monotonic()

    def _show_recently_failed(self, cache_key: tuple[str, str]) -> bool:
        failed_at = self._show_failure_at.get(cache_key)
        return failed_at is not None and time.monotonic() - failed_at < _SHOW_FAILURE_CACHE_SECONDS

    async def _get_arch_params(self, instance: str, base_url: str, model_name: str) -> ArchParams | None:
        """Fetch architecture parameters for a model from the Ollama /api/show endpoint.

        Results are cached by (instance, model_name) to avoid cross-instance contamination.
        Returns None if the request fails or the response is missing critical fields.
        """
        cache_key = (instance, model_name)
        if cache_key not in self._arch_cache and not self._show_recently_failed(cache_key):
            await self._fetch_show_info(instance, base_url, model_name)

        return self._arch_cache.get(cache_key)

    async def _get_capabilities(self, instance: str, base_url: str, model_name: str) -> ModelCapabilities:
        """Return resolved modality support for a model, fetching/caching via `/api/show` if needed."""
        cache_key = (instance, model_name)
        if cache_key not in self._capabilities_cache and not self._show_recently_failed(cache_key):
            await self._fetch_show_info(instance, base_url, model_name)

        return self._capabilities_cache.get(cache_key, ModelCapabilities())

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
                        details.get("quantization_level"),
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
        quantization_level: str | None = None,
        *,
        kv_cache_type: str | None = None,
    ) -> float | None:
        arch = await self._get_arch_params(instance, base_url, model_name)

        if arch is None or num_ctx is None:
            return None

        try:
            cache_bit = parse_cache_type_bits(kv_cache_type or _DEFAULT_KV_CACHE_TYPE)
            num_parallel = num_parallel or _DEFAULT_NUM_PARALLEL
            quant_overhead = get_quant_overhead(quantization_level)
            env_margin = self.config.ollama_vram_overhead_factor
            estimate = estimate_vram_gb(
                arch, size_bytes, num_ctx, cache_bit, num_parallel, parameters, bytes_weight, overhead_factor=quant_overhead
            )
            return round(estimate * env_margin, 2) if estimate is not None else None
        except Exception:
            # Polled every few seconds per model, so debug — a broken estimate must not break the model list.
            logger.debug("VRAM estimate failed for Ollama model %r on instance %r (arch=%r)", model_name, instance, arch, exc_info=True)
            return None

    def _start_warmth_poll_task(self, instance: str) -> None:
        """Start (or restart) the background task that syncs registry warm state from Ollama's `/api/ps`."""
        if instance in self._warmth_poll_tasks:
            self._warmth_poll_tasks[instance].cancel()

        async def _poll_loop() -> None:
            while True:
                await asyncio.sleep(_WARMTH_POLL_INTERVAL_SECONDS)
                info = self.instances_info.get(instance)
                if not info or not info.installed:
                    break
                try:
                    await self._sync_warm_state(instance, info.installed)
                except Exception:
                    logger.exception(f"{self.get_id(instance)} warmth poll tick failed")

        def _forget_if_current(finished: asyncio.Task[None]) -> None:
            # Only drop the dict entry if it still points at this task: a restart may have
            # already replaced it with a newer one by the time this one finishes/cancels.
            if self._warmth_poll_tasks.get(instance) is finished:
                del self._warmth_poll_tasks[instance]
            if not finished.cancelled() and (exc := finished.exception()):
                logger.error(f"{self.get_id(instance)} warmth poll task died unexpectedly", exc_info=exc)

        task = asyncio.create_task(_poll_loop())
        task.add_done_callback(_forget_if_current)
        self._warmth_poll_tasks[instance] = task

    def _stop_warmth_poll_task(self, instance: str) -> None:
        if instance in self._warmth_poll_tasks:
            self._warmth_poll_tasks[instance].cancel()
            del self._warmth_poll_tasks[instance]

    async def _sync_warm_state(self, instance: str, info: InstalledInfo) -> None:
        """Reflect Ollama's live loaded/evicted model state into the endpoint registry's warm flag."""
        loaded = await self._get_loaded_models(instance)
        if loaded is None:
            return
        for model_id, model_info in info.models.items():
            if not model_info.registration_id:
                continue
            ollama_name = model_info.internal_name or model_id
            self.endpoint_registry.update_warm(model_info.registration_id, ollama_name in loaded)

    async def get_loaded_model_info(self, instance: str) -> dict[str, int] | None:
        """Return {model_name: context_length} for models currently loaded in VRAM. None if not applicable."""
        loaded = await self._get_loaded_models(instance)
        return None if loaded is None else {name: info.context_length for name, info in loaded.items()}

    async def _get_loaded_models(self, instance: str) -> dict[str, LoadedModelInfo] | None:
        """Return {model_name: LoadedModelInfo} for models currently loaded in VRAM, or None if the query failed.

        Returning None (rather than an empty dict) lets callers tell "confirmed nothing loaded"
        apart from "failed to ask Ollama", so a transient failure doesn't get treated as an unload.
        """
        info = self.get_instance_installed_info(instance)
        try:
            result = await fetch_from(f"{info.base_url}/api/ps")
            if result.status_code != 200:
                self._log_ps_query_failure("Ollama /api/ps on instance %r returned status %d", instance, result.status_code)
                return None
            data = json.loads(result.data)
            models = data.get("models", [])
            self._ps_query_failing.discard(instance)
            return {
                m["name"].removesuffix(":latest"): LoadedModelInfo(
                    context_length=m.get("context_length", 0),
                    vram_gb=round(size_vram / GIB, 2) if (size_vram := m.get("size_vram")) else None,
                )
                for m in models
            }
        except Exception:
            self._log_ps_query_failure("Failed to query Ollama /api/ps on instance %r", instance, exc_info=True)
            return None

    def _log_ps_query_failure(self, message: str, instance: str, *args: Any, exc_info: bool = False) -> None:
        """Warn once per instance on the first /api/ps failure, then drop to debug until it recovers.

        Polled every _WARMTH_POLL_INTERVAL_SECONDS; a stuck-but-installed instance would
        otherwise log a WARNING (with a full stack trace) forever.
        """
        if instance in self._ps_query_failing:
            logger.debug(message, instance, *args, exc_info=exc_info)
        else:
            self._ps_query_failing.add(instance)
            logger.warning(message, instance, *args, exc_info=exc_info)

    def _effective_context(self, model_context: int | None, parsed_options: OllamaOptions) -> int:
        """Return the context Ollama runs with for an idle model: native window capped by the service context."""
        service_context = parsed_options.context_length or self.default_context_length
        return self.get_default_context_window(model_context, service_context)

    async def _resolve_vram_info(
        self,
        instance: str,
        ollama_name: str,
        model_context: int | None,
        loaded_info: dict[str, LoadedModelInfo],
        sizes: dict[str, ModelSize],
        base_url: str,
        num_parallel: int | None,
        kv_cache_type: str | None,
    ) -> tuple[bool, float | None]:
        """Return (is_loaded, vram_estimate_gb) for a model.

        For loaded models, prefers Ollama's own live-measured VRAM usage (from loaded_info);
        falls back to a calculated estimate when Ollama doesn't report a measurement.
        """
        cache_key = (instance, ollama_name)
        loaded = loaded_info.get(ollama_name)
        if loaded is None:
            self._vram_cache.pop(cache_key, None)
            size_bytes, parameters, bpw, quantization_level = sizes.get(ollama_name, ModelSize(0, None, None, None))
            vram_estimate = await self._get_vram_estimate(
                instance,
                base_url,
                ollama_name,
                size_bytes,
                model_context,
                parameters,
                bpw,
                num_parallel,
                quantization_level,
                kv_cache_type=kv_cache_type,
            )
            return False, vram_estimate

        if loaded.vram_gb is not None:
            self._vram_cache[cache_key] = loaded.vram_gb
            return True, loaded.vram_gb

        if cache_key in self._vram_cache:
            return True, self._vram_cache[cache_key]

        num_ctx = loaded.context_length or model_context
        size_bytes, parameters, bpw, quantization_level = sizes.get(ollama_name, ModelSize(0, None, None, None))
        vram_estimate = await self._get_vram_estimate(
            instance,
            base_url,
            ollama_name,
            size_bytes,
            num_ctx,
            parameters,
            bpw,
            num_parallel,
            quantization_level,
            kv_cache_type=kv_cache_type,
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
            loaded_info = await self._get_loaded_models(instance_name)
            sizes = await self._get_model_sizes(info.base_url)

            for model_id, model in instance_models.items():
                if model_id in info.models:
                    installed = info.models[model_id].get_info()
                else:
                    installed = self._get_model_installed_info(instance_name, model_id)

                if filters.installed is None or filters.installed == bool(installed):
                    is_loaded: bool | None = None
                    vram_estimate: float | None = None
                    capabilities = ModelCapabilities()
                    ollama_name: str = (info.models[model_id].internal_name if model_id in info.models else None) or model_id
                    if model_id in info.models and loaded_info is not None:
                        capabilities = await self._get_capabilities(instance_name, info.base_url, ollama_name)
                    if loaded_info is not None and model_id in info.models:
                        is_loaded, vram_estimate = await self._resolve_vram_info(
                            instance_name,
                            ollama_name,
                            self._effective_context(model.context, info.parsed_options),
                            loaded_info,
                            sizes,
                            info.base_url,
                            info.parsed_options.num_parallel,
                            info.parsed_options.kv_cache_type,
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
                            supports_vision=capabilities.supports_vision,
                            supports_audio=capabilities.supports_audio,
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
        loaded_info = await self._get_loaded_models(instance)
        sizes = await self._get_model_sizes(info.base_url)
        is_loaded: bool | None = None
        vram_estimate: float | None = None
        capabilities = ModelCapabilities()
        ollama_name: str = (info.models[model_id].internal_name if model_id in info.models else None) or model_id
        if model_id in info.models and loaded_info is not None:
            capabilities = await self._get_capabilities(instance, info.base_url, ollama_name)
        if loaded_info is not None:
            is_loaded, vram_estimate = await self._resolve_vram_info(
                instance,
                ollama_name,
                self._effective_context(model.context, info.parsed_options),
                loaded_info,
                sizes,
                info.base_url,
                info.parsed_options.num_parallel,
                info.parsed_options.kv_cache_type,
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
            supports_vision=capabilities.supports_vision,
            supports_audio=capabilities.supports_audio,
        )

    async def _download_with_ollama(self, stream: Stream[StreamChunk], base_url: str, model_id: str, model_size: str) -> None:
        progress = Progress(convert_size_to_bytes(model_size) or 0)
        last_values: dict[str, int] = {}
        last_logged_percentage = 0.0

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

                    percentage = progress.get_percentage()
                    if last_logged_percentage < 1 and (percentage - last_logged_percentage >= _PROGRESS_LOG_STEP or percentage >= 1):
                        logger.info("Downloading model %s: %.0f%%", model_id, percentage * 100)
                        last_logged_percentage = percentage

                    stream.emit(StreamChunkProgress(type="progress", stage="download", value=percentage, data={}))

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
        service_name = info.docker.service_name
        quantization_args = ["-q", quantization] if quantization else []
        cmd = ["ollama", "create", *quantization_args, model, "-f", model_path]
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
        offline: bool = False,
        info: InstalledInfo | None = None,
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
        elif offline:
            await self._download_offline_registry_model(input_stream, base_url, model_url, model_size, info)
        else:
            await self._download_with_ollama(input_stream, base_url, model_url, model_size)
        return

    async def _download_offline_registry_model(
        self, stream: Stream[StreamChunk], base_url: str, model_url: str, model_size: str, info: InstalledInfo | None = None
    ) -> None:
        """Download a model of the Ollama registry in Infra and check that the Ollama container sees it afterwards."""
        if (ref := parse_model_ref(model_url)) is None:
            raise HTTPException(
                status_code=400,
                detail=f"Model {model_url!r} cannot be installed in offline mode: only models from the Ollama registry "
                "(registry.ollama.ai) are downloaded by Infra. Use an online instance or a model with a downloadable file.",
            )
        await self._ensure_model_store_writable(ref, info)
        await self._download_with_registry_client(stream, ref, model_size)
        if not await self.is_model_installed(base_url, model_url):
            raise HTTPException(
                status_code=500,
                detail=f"Model {model_url!r} was written to the Ollama store but Ollama does not see it; "
                "the store layout may differ in this Ollama version.",
            )

    async def _download_with_registry_client(self, stream: Stream[StreamChunk], ref: OllamaModelRef, model_size: str) -> None:
        """Download a registry model in Infra, straight into the Ollama model store mounted in the container.

        `model_size` only seeds the progress until the registry client reports the real total. Client errors are
        raised as `HTTPException`: registry errors keep the client's status (404 unknown model, 507 no disk space, 502
        otherwise), local write failures give 507 (disk full) or 500.
        """
        client = OllamaRegistryClient(Path(self._get_working_dir()) / "main" / "models", blob_locks=self._blob_locks)
        progress = Progress(convert_size_to_bytes(model_size) or 0)
        last_logged_percentage = 0.0

        stream.emit(StreamChunkProgress(type="progress", stage="download", value=0, data={}))
        try:
            async with aclosing(client.download(ref)) as packets:
                async for packet in packets:
                    if isinstance(packet, PreDownloadPacket):
                        if packet.file_bytes_size:
                            progress.set_max_value(packet.file_bytes_size)
                    elif isinstance(packet, DownloadedPacket) and progress.max:
                        progress.add_to_actual_value(packet.downloaded_bytes_size)
                        percentage = progress.get_percentage()
                        if last_logged_percentage < 1 and (percentage - last_logged_percentage >= _PROGRESS_LOG_STEP or percentage >= 1):
                            logger.info("Downloading model %s: %.0f%%", ref.manifest_path, percentage * 100)
                            last_logged_percentage = percentage
                        stream.emit(StreamChunkProgress(type="progress", stage="download", value=percentage, data={}))
        except OllamaRegistryError as e:
            logger.warning("Downloading model %s from the Ollama registry failed: %s", ref.manifest_path, e.message)
            raise HTTPException(status_code=e.status_code, detail=e.message) from e
        except OSError as e:
            logger.warning("Writing model %s to the Ollama store failed: %s", ref.manifest_path, e, exc_info=True)
            no_space = e.errno == errno.ENOSPC
            hint = (
                f" Run: sudo chown -R {os.getuid()}:{os.getgid()} {Path(self._get_working_dir()) / 'main' / 'models'}"
                if e.errno == errno.EACCES
                else ""
            )
            raise HTTPException(
                status_code=507 if no_space else 500,
                detail=f"Cannot write the model to the Ollama store: {e}.{hint}",
            ) from e

        stream.emit(StreamChunkProgress(type="progress", stage="download", value=1, data={}))

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
        offline: bool = False,
        info: InstalledInfo | None = None,
    ) -> None:
        if model_id not in self.models_download_progress:
            self.models_download_progress[model_id] = input_stream
            try:
                await self._download_with_downloader(
                    input_stream, base_url, model_url, model_id, size, i, models_quantity, dest_path, offline, info=info
                )
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
            if model.quantization:
                await self._reject_unsupported_quantization_source(instance_info.base_url, modelfile)
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
                                input_stream,
                                instance_info.base_url,
                                model_to_download,
                                model_id,
                                model.size,
                                0,
                                1,
                                offline=instance_info.parsed_options.mode == "offline",
                                info=instance_info,
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
            stderr = e.stderr if isinstance(e, CommandFailedError) else ""
            message = get_ollama_create_error_message(stderr) or "Cannot install modelfile"
            raise HTTPException(status_code=400, detail=message) from e

    async def _reject_unsupported_quantization_source(self, base_url: str, modelfile: OllamaModelFile) -> None:
        """Raise a friendly 400 before any download if this Ollama version can't quantize the modelfile's FROM source.

        Since Ollama 0.34.1 `ollama create -q` only works for safetensors imports, so a GGUF file or a model referenced
        by name always fails. Older versions (or an unknown version) are left to Ollama; its error is mapped afterwards.
        """
        from_values = [line.value for line in modelfile.lines if isinstance(line, OllamaModelFileLine) and line.instruction == "FROM"]
        if not any(self._is_gguf_or_named_source(value) for value in from_values):
            return
        ollama_version = await self._get_ollama_version(base_url)
        if ollama_version is not None and supports_gguf_quantization(ollama_version) is False:
            raise HTTPException(status_code=400, detail=GGUF_QUANTIZATION_UNSUPPORTED_MESSAGE)

    async def _reject_ollama_without_decision_models(self, base_url: str) -> None:
        """Raise a friendly 400 before any download if this Ollama is too old to serve decision models on `/v1/systemone`.

        An unknown version is left to Ollama, as in the quantization pre-check.
        """
        ollama_version = await self._get_ollama_version(base_url)
        if ollama_version is not None and supports_decision_models(ollama_version) is False:
            raise HTTPException(status_code=400, detail=decision_models_unsupported_message(ollama_version))

    @staticmethod
    def _is_gguf_or_named_source(from_value: str) -> bool:
        """Return whether a FROM value is a GGUF file or a model name, i.e. anything but a (safetensors) directory or file."""
        if from_value.startswith(("http://", "https://")):
            return Path(urlparse(from_value).path).suffix == ".gguf"
        if from_value.startswith(("./", "/")):
            return Path(from_value).suffix == ".gguf"
        return True

    @staticmethod
    async def _get_ollama_version(base_url: str) -> str | None:
        """Return the Ollama instance's version from `/api/version`, or None if it can't be read."""
        with suppress(Exception):
            result = await fetch_from(f"{base_url}/api/version", timeout=_OLLAMA_VERSION_TIMEOUT_SECONDS)
            if result.status_code == 200:
                ollama_version = json.loads(result.data).get("version")
                return ollama_version if isinstance(ollama_version, str) else None
        return None

    def get_default_context_window(self, model_context: int | None, service_context: int) -> int:
        """Return minimum default context length from model or service context length."""
        return min(model_context, service_context) if model_context else service_context

    async def _install_model(  # noqa: C901
        self,
        instance: str,
        model_id: str,
        options: InstallModelIn,
        container_name_override: str | None = None,  # noqa: ARG002
    ) -> PromiseWithProgress[InstallModelOut, StreamChunk]:
        parsed_model_options = try_parse_pydantic(OllamaModelOptions, options.spec) if options.spec else OllamaModelOptions()
        info = self.get_instance_installed_info(instance)

        # Cache entries are written under the resolved internal name (e.g. `<id>-customcontextlength`
        # for a custom-context install, see below), not `model_id` - evict both so a previous
        # install's real cache entry doesn't survive as stale data for the new one.
        previous_internal_name = info.models[model_id].internal_name if model_id in info.models else None
        self._arch_cache.pop((instance, model_id), None)
        self._capabilities_cache.pop((instance, model_id), None)
        if previous_internal_name and previous_internal_name != model_id:
            self._arch_cache.pop((instance, previous_internal_name), None)
            self._capabilities_cache.pop((instance, previous_internal_name), None)

        if not self.models.get(instance):
            self.models[instance] = {}

        key = (instance, model_id)
        if model_id in info.models or key in self._installing:
            return PromiseWithProgress(value=InstallModelOut(status="OK", details="Already installed or being installed right now."))

        if model_id not in self.models[instance]:
            raise HTTPException(400, f"Model {model_id!r} not found")

        model = self.models[instance][model_id]
        if model.type == "decision":
            await self._reject_ollama_without_decision_models(info.base_url)

        self._installing.add(key)

        async def func(input_stream: Stream[StreamChunk]) -> InstallModelOut:  # noqa: C901
            try:
                rewrite_model_to = model_id
                internal_name = model_id
                new_modelfile: str | None = None

                service_form_context_window = positive_context_window_or_none(info.parsed_options.context_length)
                model_form_context_window = positive_context_window_or_none(parsed_model_options.context_length)

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
                    new_model_context = await self._install_from_modelfile(
                        input_stream, model_id, modelfile_recipe, internal_name, info, model
                    )
                    if new_model_context is not None:
                        model_context = new_model_context

                else:
                    if not await self.is_model_installed(info.base_url, model_id):
                        await self._download_model_or_set_progress(
                            input_stream,
                            info.base_url,
                            model.id,
                            model.id,
                            model.size,
                            offline=info.parsed_options.mode == "offline",
                            info=info,
                        )

                input_stream.emit(StreamChunkProgress(type="progress", stage="install", value=0, data={}))

                if parsed_model_options.alive_time != "":
                    await fetch_from(
                        f"{info.base_url}/api/generate",
                        "POST",
                        {"model": model_id, "keep_alive": parsed_model_options.alive_time},
                    )

                registered_name = parsed_model_options.rename if parsed_model_options.rename else model_id
                registered_alias = parsed_model_options.alias
                info.models[model_id] = model_info = ModelInstalledInfo(
                    id=model_id,
                    type=model.type,
                    registered_name=registered_name,
                    options=options,
                    registration_id="",
                    internal_name=internal_name,
                )
                try:
                    model_max_context_window = positive_context_window_or_none(model_context)
                    service_max_context_window = self.default_context_length
                    default_context_window = self.get_default_context_window(model_max_context_window, service_max_context_window)
                    max_context_window = model_form_context_window or model_max_context_window or service_max_context_window

                    context_window = model_form_context_window or service_form_context_window or default_context_window
                    context_window = min(context_window, max_context_window)
                    # `or` (not `is not None`) is intentional here: Ollama's OLLAMA_NUM_PARALLEL=0 means
                    # "auto", so an explicit 0 must fall through to the fallback of 1 just like None does.
                    #
                    # WARNING for reviewers: the fallback of `1` is a *conservative approximation* of
                    # "auto", not Ollama's real behavior. Ollama's own auto-scaling picks up to 4
                    # concurrent requests depending on available VRAM at load time; this gateway has no
                    # visibility into that and cannot replicate it, so it substitutes a fixed default
                    # instead. Pros: never overcommits VRAM by assuming a higher concurrency than what's
                    # actually available - safe by construction. Cons: on hardware where Ollama would
                    # genuinely auto-scale to 2-4, this under-reports capacity and causes
                    # earlier-than-necessary spillover to other backends. Treat this as accepted
                    # routing-side conservatism, not a bug, unless the user explicitly wants capacity to
                    # track Ollama's real auto behavior.
                    capacity = info.parsed_options.num_parallel or _DEFAULT_NUM_PARALLEL
                    # The model is already pulled into Ollama's local store by this point (either via
                    # the modelfile install above or the plain pull), so `/api/show` can resolve its
                    # real modality support instead of registering with permanently-unknown values.
                    capabilities = await self._get_capabilities(instance, info.base_url, internal_name)
                    # Ollama lazily loads a model into memory on its first inference request, so a
                    # freshly-installed model is never warm yet - registering it as warm here would win
                    # ranking over an actually-warm instance and route the first requests into a cold load.
                    # The warmth-poll task (already running per-instance) corrects this to `True` within one
                    # tick once `/api/ps` confirms it's genuinely loaded (or leaves it `False` otherwise).
                    if model.type == "llm":
                        model_info.registration_id = self.endpoint_registry.register_chat_completion_as_proxy(
                            model=registered_name,
                            props=ModelProps(
                                private=True,
                                type=model.type,
                                endpoints=LLM_ENDPOINTS,
                                context_window=context_window,
                                max_context_window=max_context_window,
                                supports_vision=capabilities.supports_vision,
                                supports_audio=capabilities.supports_audio,
                            ),
                            chat_completions=ProxyOptions(
                                url=f"{info.base_url}/v1/chat/completions", rewrite_model_to=rewrite_model_to, forward_extra_fields=True
                            ),
                            completions=ProxyOptions(
                                url=f"{info.base_url}/v1/completions", rewrite_model_to=rewrite_model_to, forward_extra_fields=True
                            ),
                            responses=ProxyOptions(url=f"{info.base_url}/v1/responses", rewrite_model_to=rewrite_model_to),
                            messages=ProxyOptions(url=f"{info.base_url}/v1/messages", rewrite_model_to=rewrite_model_to),
                            ollama_chat=ProxyOptions(url=f"{info.base_url}/api/chat", rewrite_model_to=rewrite_model_to),
                            registration_options=RegistrationOptions(
                                origin="local", alias=registered_alias, owned_by=self.get_type(), capacity_state=capacity, warm=False
                            ),
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
                                supports_vision=capabilities.supports_vision,
                                supports_audio=capabilities.supports_audio,
                            ),
                            options=ProxyOptions(url=f"{info.base_url}/v1/embeddings", rewrite_model_to=rewrite_model_to),
                            registration_options=RegistrationOptions(
                                origin="local", alias=registered_alias, owned_by=self.get_type(), capacity_state=capacity, warm=False
                            ),
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
                                supports_vision=capabilities.supports_vision,
                                supports_audio=capabilities.supports_audio,
                            ),
                            options=ProxyOptions(url=f"{info.base_url}/v1/images/generations", rewrite_model_to=rewrite_model_to),
                            registration_options=RegistrationOptions(
                                origin="local", alias=registered_alias, owned_by=self.get_type(), capacity_state=capacity, warm=False
                            ),
                        )
                    if model.type == "decision":
                        model_info.registration_id = self.endpoint_registry.register_systemone_as_proxy(
                            model=registered_name,
                            props=ModelProps(
                                private=True,
                                type=model.type,
                                endpoints=DECISION_ENDPOINTS,
                                context_window=context_window,
                                max_context_window=max_context_window,
                                supports_vision=capabilities.supports_vision,
                                supports_audio=capabilities.supports_audio,
                            ),
                            options=ProxyOptions(url=f"{info.base_url}/v1/systemone", rewrite_model_to=rewrite_model_to),
                            registration_options=RegistrationOptions(
                                origin="local", alias=registered_alias, owned_by=self.get_type(), capacity_state=capacity, warm=False
                            ),
                        )
                    input_stream.emit(StreamChunkProgress(type="progress", stage="install", value=1, data={}))

                    self.models_downloaded[model_id] = DownloadedInfo()

                    if model.hash:
                        for alias_model_id, alias_model in self.models[instance].items():
                            if alias_model.hash == model.hash:
                                self.models_downloaded[alias_model_id] = DownloadedInfo()

                    return InstallModelOut(status="OK", details="Installed")
                except Exception:
                    if model_info.registration_id:
                        if model_info.type == "llm":
                            self.endpoint_registry.unregister_chat_completion(model_info.registered_name, model_info.registration_id)
                        if model_info.type == "embedding":
                            self.endpoint_registry.unregister_embeddings(model_info.registered_name, model_info.registration_id)
                        if model_info.type == "txt2img":
                            self.endpoint_registry.unregister_image_generations(model_info.registered_name, model_info.registration_id)
                        if model_info.type == "decision":
                            self.endpoint_registry.unregister_systemone(model_info.registered_name, model_info.registration_id)
                    if info.models.get(model_id) is model_info:
                        info.models.pop(model_id, None)
                    raise
            finally:
                self._installing.discard(key)

        return PromiseWithProgress(func=func)

    async def _uninstall_model(self, instance: str, model_id: str, options: UninstallModelIn) -> None:  # noqa: C901
        info = self.get_instance_installed_info(instance)
        if (instance, model_id) in self._installing:
            raise HTTPException(409, f"Model {model_id!r} is currently being installed; cancel the install before deleting")
        ollama_name = model_id
        if model_id in info.models:
            model = info.models[model_id]
            ollama_name = model.internal_name or model_id
            del info.models[model_id]
            if model.type == "llm":
                self.endpoint_registry.unregister_chat_completion(model.registered_name, model.registration_id)
            if model.type == "embedding":
                self.endpoint_registry.unregister_embeddings(model.registered_name, model.registration_id)
            if model.type == "txt2img":
                self.endpoint_registry.unregister_image_generations(model.registered_name, model.registration_id)
            if model.type == "decision":
                self.endpoint_registry.unregister_systemone(model.registered_name, model.registration_id)
            await self._unload_model_from_vram(model_id, ollama_name, info.base_url)

        # Cache entries are written under the resolved internal name (`ollama_name`), not
        # `model_id` - evict both so a custom-context install's real cache entry doesn't survive
        # as stale data (e.g. serving wrong capabilities after a tag change and reinstall).
        self._arch_cache.pop((instance, model_id), None)
        self._capabilities_cache.pop((instance, model_id), None)
        if ollama_name != model_id:
            self._arch_cache.pop((instance, ollama_name), None)
            self._capabilities_cache.pop((instance, ollama_name), None)

        if options.purge and model_id in self.models_downloaded:
            model = self.models[instance][model_id]
            tasks: list[asyncio.Task[Any]] = []

            tasks.append(asyncio.create_task(self._delete_ollama_model(ollama_name, info.base_url)))

            await self.remove_modelfile(model_id, instance)

            del self.models_downloaded[model_id]

            if not model.hash:
                await asyncio.gather(*tasks)
                return

            for alias_model_id, alias_model in self.models[instance].items():
                if alias_model.hash == model.hash and model_id != alias_model_id:
                    alias_ollama_name = (
                        (info.models[alias_model_id].internal_name or alias_model_id) if alias_model_id in info.models else alias_model_id
                    )
                    await self._uninstall_model(instance, alias_model_id, UninstallModelIn(purge=False))
                    tasks.append(asyncio.create_task(self._delete_ollama_model(alias_ollama_name, info.base_url)))
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
