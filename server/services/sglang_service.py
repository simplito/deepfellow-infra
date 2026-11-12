# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Sglang service."""

import asyncio
import functools
import json
import re
import shutil
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict

import aiohttp
from fastapi import HTTPException
from pydantic import BaseModel, Field

from scripts.get_huggingface_models import fetch_registry_entries
from server.applicationcontext import get_base_url
from server.config import get_main_dir
from server.docker import (
    DockerImage,
    DockerOptions,
)
from server.endpointregistry import CapacityState, ProxyOptions, RegistrationId, RegistrationOptions
from server.models.api import EMBEDDINGS_ENDPOINTS, ModelProps
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
    OneOfOption,
    ServiceField,
    ServiceOptions,
    ServiceSize,
    ServiceSpecification,
    UninstallServiceIn,
)
from server.services import model_catalog_refresh
from server.services.base2_service import Base2Service, CustomModel, Instance, InstanceConfig, ModelConfig
from server.services.model_catalog_refresh import huggingface_refresh_guard
from server.utils.core import (
    PromiseWithProgress,
    Stream,
    StreamChunk,
    StreamChunkProgress,
    SuccessDownloadPacket,
    convert_size_to_bytes,
    normalize_name,
    try_parse_pydantic,
    with_legacy_key_carried_over,
)
from server.utils.files import get_model_dir_context_window
from server.utils.hardware import HardwarePartInfo
from server.utils.loading import Progress
from server.utils.logger import logger
from server.utils.model_downloader import track_download_progress
from server.utils.registry_client import image_without_registry_prefix, registry_for
from server.utils.served_model_len import fetch_served_max_model_len
from server.utils.size_fetcher import fetch_huggingface_model_size
from server.utils.tokenizer_archive import tokenizer_from_dir

# Qwen3-Reranker-family models refuse to start with `--is-embedding`; every other reranker in the
# registry is a cross-encoder, which accepts (and defaults to) it.
_QWEN3_RERANKER_HINT = "qwen3"

# hf_id substring -> `--reasoning-parser` value. Order matters: more specific patterns first.
# Only covers families we can identify with confidence; unmatched models get no reasoning parser,
# which preserves today's behavior (raw <think> tags in the response) rather than guessing wrong.
_REASONING_PARSER_HINTS: tuple[tuple[str, str], ...] = (
    ("deepseek-r1", "deepseek-r1"),
    ("deepseek-v3", "deepseek-v3"),
    ("qwen3", "qwen3"),
    ("glm-4.5", "glm45"),
    ("glm4.5", "glm45"),
    ("gpt-oss", "gpt-oss"),
    ("kimi", "kimi"),
    ("step3", "step3"),
)

# hf_id substring -> `--tool-call-parser` value. Order matters: more specific patterns first
# (e.g. "qwen3-coder" before the generic "qwen" family match).
_TOOL_CALL_PARSER_HINTS: tuple[tuple[str, str], ...] = (
    ("qwen3-coder", "qwen3_coder"),
    ("qwen2.5", "qwen25"),
    ("qwen25", "qwen25"),
    ("qwen", "qwen"),
    ("deepseek-v3", "deepseekv3"),
    ("llama-3", "llama3"),
    ("llama3", "llama3"),
    ("mistral", "mistral"),
    ("glm-4.5", "glm45"),
    ("glm4.5", "glm45"),
    ("kimi", "kimi_k2"),
)


# Mirrors the `just get-sglang-models` release recipe (see justfile): one fetch per model type,
# each merged into its own registry key so a refresh reproduces exactly what the offline, reviewed
# release process would generate. Unlike vLLM, SGLang's reranker fetch allows generative
# candidates, tagging each with `is_generative` instead of dropping it.
_SGLANG_FETCH_PLAN: list[tuple[int, int, int, str, bool]] = [
    (200, 200, 50, "llm", False),
    (50, 30, 30, "reranker", True),
    (10, 0, 10, "embedding", False),
]


class SglangModel(BaseModel):
    hf_id: str
    revision: str | None = None
    env_vars: dict[str, str] | None = None
    quantization: str | None = None
    dtype: str = "auto"
    shm_size: str = "32gb"
    ulimits: dict[str, str] | None = None
    max_model_len: int | None = None
    gpu_memory_utilization: float | None = None
    size: str
    custom: CustomModelId | None = None
    model_type: Literal["llm", "reranker", "embedding"] = "llm"
    # Only meaningful when model_type == "reranker". True/False comes from the registry's
    # architecture-derived `is_generative` flag (see `just get-sglang-models`); None means the
    # model has no such metadata (e.g. a hand-added custom model), so callers fall back to the
    # by-name Qwen3-Reranker-family heuristic.
    is_generative_reranker: bool | None = None
    # Best-effort, derived from the HF repo's `config.architectures` at catalog-build time (see
    # `classify_modality_support`). None means unresolved (e.g. a hand-added custom model, or a
    # non-"llm" registry entry, for which modality support isn't computed).
    supports_vision: bool | None = None
    supports_audio: bool | None = None


class SglangCustomModel(BaseModel):
    id: str
    hf_id: str
    revision: str | None = None
    size: str
    model_type: Literal["llm", "reranker", "embedding"] = "llm"


class SglangRegistryEntry(TypedDict):
    name: str
    size: str


class SglangRerankerRegistryEntry(SglangRegistryEntry, total=False):
    is_generative: bool


class SglangLlmRegistryEntry(SglangRegistryEntry, total=False):
    supports_vision: bool
    supports_audio: bool


class SglangRegistry(TypedDict, total=False):
    llms: list[SglangLlmRegistryEntry]
    rerankers: list[SglangRerankerRegistryEntry]
    embeddings: list[SglangRegistryEntry]


class SglangConst(BaseModel):
    image: DockerImage
    models: dict[str, SglangModel]


def _read_models() -> dict[str, SglangModel]:
    # SGLang has its own catalog, generated by `just get-sglang-models` (see justfile), so its
    # reranker pool can diverge from vLLM's without being constrained by vLLM's filtering.
    sglang_path = get_main_dir() / "./static/sglang-min.json"
    with sglang_path.open(encoding="utf-8") as f:
        registry: SglangRegistry = json.loads(f.read())
        map: dict[str, SglangModel] = {}
        for entry in registry.get("llms", []):
            map[entry["name"]] = SglangModel(
                hf_id=entry["name"],
                size=entry["size"],
                model_type="llm",
                supports_vision=entry.get("supports_vision"),
                supports_audio=entry.get("supports_audio"),
            )
        for entry in registry.get("rerankers", []):
            map[entry["name"]] = SglangModel(
                hf_id=entry["name"],
                size=entry["size"],
                model_type="reranker",
                is_generative_reranker=entry.get("is_generative"),
            )
        for entry in registry.get("embeddings", []):
            map[entry["name"]] = SglangModel(
                hf_id=entry["name"],
                size=entry["size"],
                model_type="embedding",
            )
        return map


_const = SglangConst(
    image=DockerImage(name="lmsysorg/sglang:v0.5.23-cu130", size="15.2 GB"),
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
    model_type: Literal["llm", "reranker", "embedding"] = "llm"
    capacity: int | None = None
    capacity_known: bool = False

    def get_info(self) -> ModelInfo:
        """Get info."""
        return ModelInfo(spec=self.options.spec, registration_id=self.registration_id)


class SglangOptions(BaseModel):
    hardware: str | bool | None = None


class SglangModelOptions(BaseModel):
    rename: str | None = None
    alias: str | None = None
    gpu_memory_utilization: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    gpu_priority: float | None = Field(default=None, ge=0.01, le=1)
    max_model_length: int | None = None
    quantization: str | None = None
    extra_args: dict[str, str] = {}
    extra_envs: dict[str, str] = {}


@dataclass
class InstalledInfo:
    models: dict[str, ModelInstalledInfo]
    options: InstallServiceIn
    parsed_options: SglangOptions


@dataclass
class DownloadedInfo:
    model_path: str | None


class SglangService(Base2Service[InstalledInfo, DownloadedInfo]):
    hugging_face_cache_path = "/mnt/hf"
    models: dict[str, dict[str, SglangModel]]
    gpu_memory_utilization = 0
    _default_gpu_memory_utilization = 0.9
    _gpu_memory_free_safety_margin = 0.9
    _min_gpu_memory_utilization = 0.01
    _vram_cache: dict[tuple[str, str], float]
    _installing: set[tuple[str, str]]

    def _after_init(self) -> None:
        self.load_default_models("default")
        self._vram_cache = {}
        self._installing = set()
        self._init_reconciliation()

    def load_default_models(self, instance: str) -> None:
        """Load default models to instance."""
        self.models[instance] = {model_id: model.model_copy(deep=True) for model_id, model in _const.models.items()}

    def get_catalog_refresh_unavailable_reason(self) -> str | None:
        """Return why a catalog refresh can't run right now, or None if it can."""
        if not model_catalog_refresh.catalog_refresh_supported:
            return "The static model-catalog directory isn't writable, so catalog refresh is disabled."
        return None

    async def refresh_catalog(self) -> PromiseWithProgress[CatalogRefreshOut, StreamChunk]:
        """Fetch current top SGLang-compatible LLM/reranker/embedding models from HuggingFace.

        Runs as a background job (dozens-to-hundreds of HTTP requests) and, on success, writes the
        merged registry to `static/sglang-min.json` and reloads `_const.models` in memory. Failure
        (including the shared in-flight guard rejecting a concurrent refresh) leaves the existing
        file and in-memory catalog untouched.
        """
        if not model_catalog_refresh.catalog_refresh_supported:
            raise HTTPException(405, self.get_catalog_refresh_unavailable_reason())

        service_id = self.get_type()
        huggingface_refresh_guard.acquire(service_id)

        async def func(stream: Stream[StreamChunk]) -> CatalogRefreshOut:
            try:
                before_ids = set(_const.models.keys())
                sglang_path = get_main_dir() / "./static/sglang-min.json"
                existing_registry: dict[str, Any] = json.loads(sglang_path.read_text(encoding="utf-8")) if sglang_path.exists() else {}

                merged: dict[str, list[dict[str, Any]]] = {}
                async with aiohttp.ClientSession() as session:
                    for i, (top_downloads, top_likes, top_trending, model_type, allow_generative) in enumerate(_SGLANG_FETCH_PLAN):
                        stream.emit(
                            StreamChunkProgress(
                                type="progress", stage="install", value=i / len(_SGLANG_FETCH_PLAN), data={"stage": model_type}
                            )
                        )
                        registry_key, entries = await fetch_registry_entries(
                            session,
                            top_downloads,
                            top_likes,
                            top_trending,
                            model_type,
                            log=lambda _msg: None,
                            allow_generative_rerankers=allow_generative,
                        )
                        try:
                            model_catalog_refresh.ensure_catalog_not_degraded(
                                registry_key, len(entries), len(existing_registry.get(registry_key, []))
                            )
                        except model_catalog_refresh.CatalogTooSmallError as exc:
                            raise HTTPException(502, str(exc)) from exc
                        merged[registry_key] = entries

                registry = {**existing_registry, **merged}
                model_catalog_refresh.atomic_write_json(sglang_path, registry)

                _const.models = _read_models()
                for instance in list(self.models):
                    self._reload_instance_models(instance)

                after_ids = set(_const.models.keys())
                stream.emit(StreamChunkProgress(type="progress", stage="install", value=1, data={}))
                return CatalogRefreshOut(added=len(after_ids - before_ids), total=len(after_ids))
            finally:
                huggingface_refresh_guard.release()

        return PromiseWithProgress(func=func)

    def _reload_instance_models(self, instance: str) -> None:
        """Reload `instance`'s models from the refreshed catalog, preserving custom models and installed ones that fell out of it."""
        installed = self.get_instance_info(instance).installed
        installed_ids = set(installed.models) if installed else set()
        custom_models = {mid: m for mid, m in self.models[instance].items() if m.custom is not None}
        self.load_default_models(instance)
        self.models[instance].update(custom_models)
        for model_id in installed_ids:
            if model_id not in self.models[instance] and (definition := self._get_persisted_model_definition(instance, model_id)):
                self._restore_model_definition(instance, model_id, definition)

        # An installed model's registration keeps the supports_vision/supports_audio it was
        # registered with at install time; a catalog refresh (e.g. a `classify_modality_support`
        # fix, or a regenerated static catalog) can correct those values in the catalog without
        # anything else re-registering the model, so re-sync them here.
        if installed:
            for model_id in installed_ids:
                model_info = installed.models.get(model_id)
                model = self.models[instance].get(model_id)
                if model_info and model and model_info.registration_id:
                    self.endpoint_registry.update_modality_support(
                        model_info.registration_id, supports_vision=model.supports_vision, supports_audio=model.supports_audio
                    )

    async def sync_models(self, instance: str) -> None:  # noqa: ARG002
        """Trigger a background catalog refresh (fire-and-forget; matches the base no-blocking sync contract)."""
        with suppress(HTTPException):
            promise = await self.refresh_catalog()

            async def _consume_result() -> None:
                try:
                    await promise.wait()
                except Exception:
                    logger.warning("Background catalog refresh triggered by sync_models failed", exc_info=True)

            asyncio.create_task(_consume_result())  # noqa: RUF006 - fire-and-forget, retrieves the promise's exception

    def _restore_model_definition(self, instance: str, model_id: str, definition: dict[str, Any]) -> None:
        self.models[instance][model_id] = SglangModel.model_validate(definition)

    def get_type(self) -> str:
        """Return the service id."""
        return "sglang"

    def get_description(self) -> str:
        """Return the service description."""
        return "Self-hosted SGLang model runner with RadixAttention for high-throughput serving."

    def get_size(self) -> ServiceSize:
        """Return the service size."""
        sizes: dict[str, str] = {}
        if self._supported_gpus:
            sizes["gpu"] = _const.image.size
        return sizes

    def get_spec(self) -> ServiceSpecification:
        """Return the service specification. GPU-only: no CPU hardware option is offered."""
        options: list[str | OneOfOption] = []
        default: str | None = None
        gpus = self._supported_gpus
        if len(gpus) == 1:
            options.append(OneOfOption(value="GPU", label="Default GPU"))
            default = "GPU"
        elif gpus:
            options.append(OneOfOption(value="GPUs", label="All GPUs"))
            default = "GPUs"
        options.extend([f"GPU | {gpu.long_name}" for gpu in gpus])

        fields = [ServiceField(type="oneof", name="hardware", description="Choose hardware:", values=options, default=default)]
        fields.append(
            ServiceField(
                type="docker-tags",
                name="image_version",
                description="Docker image version",
                docker_image=_const.image.name.split(":")[0],
                depends_on="hardware",
                required=False,
            )
        )
        return ServiceSpecification(fields=fields)

    def _get_model_install_fields(self, instance: str, model_id: str) -> Sequence[ModelField]:
        """Return the install spec fields of a model, or none if the model or instance is unknown."""
        model = self.models.get(instance, {}).get(model_id)
        if model is None or instance not in self.instances_info:
            return []
        return self.get_model_type_spec(instance, model.model_type).fields

    def get_model_type_spec(self, instance: str, model_type: str) -> ModelSpecification:
        """Return the install specification of models of the given type."""
        fields = [
            ModelField(type="text", name="rename", description="Rename the model, hiding its original name", required=False),
            ModelField(
                type="text", name="alias", description="Additional name for the model, the original name keeps working", required=False
            ),
        ]
        if options := self.instances_info[instance].config.options:
            if "hardware" not in options.spec:
                options.spec["hardware"] = options.spec.get("gpu", self.docker_service.has_gpu_support)
            parsed_options = try_parse_pydantic(SglangOptions, options.spec)
            hardware = parsed_options.hardware
            if (isinstance(hardware, bool) and hardware is True) or (isinstance(hardware, str) and hardware.startswith("GPU")):
                fields.append(
                    ModelField(
                        type="gpu-priority",
                        name="gpu_priority",
                        description="Share of currently-free GPU memory (0.01 - 1.00) to reserve for this model.",
                        required=False,
                    ),
                )
        fields.extend(
            [
                ModelField(type="number", name="max_model_length", description="Max model length ex. 4096", required=False),
                ModelField(type="text", name="quantization", description="Quantization", required=False),
                ModelField(
                    type="map",
                    name="extra_args",
                    description="Additional SGLang arguments. Key: flag name (e.g. --dtype), value: flag value (empty for boolean flags).",
                    required=False,
                    placeholder="extra_args",
                    default='{"--trust-remote-code":""}' if model_type in ("reranker", "embedding") else None,
                ),
                ModelField(
                    type="map",
                    name="extra_envs",
                    description="Additional environment variables passed to the SGLang container.",
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
                CustomModelField(
                    type="text",
                    name="revision",
                    description="Hugging face branch, tag, or commit SHA (defaults to the repo's main branch)",
                    placeholder="main",
                    required=False,
                ),
                CustomModelField(type="text", name="size", description="Model size", placeholder="1 GB", required=False),
                CustomModelField(
                    type="oneof",
                    name="model_type",
                    description="Model type",
                    values=["llm", "reranker", "embedding"],
                    default="llm",
                    required=False,
                ),
            ]
        )

    async def _resolve_custom_model_size(self, spec: dict[str, Any], instance: str = "") -> str | None:  # noqa: ARG002
        try:
            return await fetch_huggingface_model_size(spec["hf_id"], spec.get("revision"), token=self.get_hugging_face_token())
        except Exception:
            logger.debug("Couldn't resolve size for a custom model id = %s, revision = %s", spec["hf_id"], spec.get("revision"))
            return None

    def get_installed_info(self, instance: str) -> bool | InstallServiceProgress | ServiceOptions:
        """Get service installed info."""
        installed = self.get_instance_info(instance).installed
        return self._get_service_installed_info(instance) if installed is None else installed.options.spec

    def _generate_instance_config(
        self,
        instance: str,  # noqa: ARG002
        info: InstalledInfo | None,
        custom: list[CustomModel] | None,
    ) -> InstanceConfig:
        return InstanceConfig(
            options=info.options if info else None,
            models=[
                ModelConfig(
                    model_id=x.id,
                    options=x.options,
                    capacity=x.capacity,
                    capacity_known=x.capacity_known,
                    gpu_memory_utilization=x.gpu_memory_utilization,
                    container_name=x.docker.container_name,
                )
                for x in info.models.values()
            ]
            if info
            else [],
            custom=custom,
        )

    def get_hardware_unsupported_reason(self) -> str | None:
        """Return why SGLang can't run here. SGLang is GPU-only: no supported GPU means it can't run at all."""
        if not self._supported_gpus:
            return "Your hardware doesn't support this service.\nThere are not any graphic cards. Restart Infra if this has changed."
        return None

    def get_default_docker_image_name(self) -> str | None:
        """Return the default docker image, for the upfront platform-compatibility check."""
        return _const.image.name

    def is_given_hardware_support_gpu(self, hardware_specification: str | bool | None) -> bool:
        """Return is gpu will be used. SGLang is GPU-only: no supported GPU means the service can't run at all."""
        if reason := self.get_hardware_unsupported_reason():
            raise HTTPException(400, reason)
        return super().is_given_hardware_support_gpu(hardware_specification)

    def get_specified_hardware_parts(self, hardware_specification: str | bool | None) -> Sequence[HardwarePartInfo]:
        """Get hardware based on user input."""
        if reason := self.get_hardware_unsupported_reason():
            raise HTTPException(400, reason)
        return super().get_specified_hardware_parts(hardware_specification)

    def _load_download_info(self, data: dict[str, Any]) -> DownloadedInfo:
        return DownloadedInfo(**data)

    async def _validate_update_options(self, options: InstallServiceIn) -> None:
        """Normalize the hardware spec and validate the requested image_version, if any."""
        if "hardware" not in options.spec:
            options.spec["hardware"] = options.spec.get("gpu", self.docker_service.has_gpu_support)
        options.spec["hardware"] = self.canonicalize_hardware_spec(options.spec["hardware"])
        await self.validate_docker_image_version(options.spec.get("image_version"), options.spec.get("hardware"))

    async def _install_instance(
        self,
        instance: str,
        options: InstallServiceIn,
        container_name_override: str | None = None,  # noqa: ARG002
    ) -> PromiseWithProgress[InstalledInfo, StreamChunk]:
        if not self.models.get(instance):
            self.load_default_models(instance)

        await self._validate_update_options(options)
        parsed_options = try_parse_pydantic(SglangOptions, options.spec)
        self.is_given_hardware_support_gpu(parsed_options.hardware)
        image_version = options.spec.get("image_version")
        image = self._get_image(image_version)
        await self._verify_docker_image(image.name, options.ignore_warnings)

        async def func(stream: Stream[StreamChunk]) -> InstalledInfo:
            await self._download_image_or_set_progress(stream, image)
            self.service_downloaded = True
            self._start_reconciliation_task(instance)
            return InstalledInfo(models={}, options=options, parsed_options=parsed_options)

        return PromiseWithProgress(func=func)

    async def _uninstall_instance(self, instance: str, options: UninstallServiceIn) -> None:
        self._stop_reconciliation_task(instance)
        installed = self.get_instance_info(instance).installed
        installed_images = {model.docker.image for model in installed.models.values()} if installed else set()
        if installed:
            await asyncio.gather(
                *[
                    self._uninstall_model(instance, model.id, UninstallModelIn(purge=options.purge))
                    for model in installed.models.copy().values()
                    if not self.is_model_installed_in_other_instance(instance, model.id)
                ]
            )

        self.instances_info[instance].installed = None

        if options.purge:
            if not any(i.installed for i in self.instances_info.values()):
                self.service_downloaded = False
                await self.docker_service.remove_image(_const.image.name)
                for image_name in installed_images:
                    await self.docker_service.remove_image(image_name)
                await self._clear_working_dir()
                self.models_downloaded = {}

            if instance == "default":
                self.instances_info["default"] = Instance(None, None, {}, InstanceConfig())
            else:
                del self.instances_info[instance]

    def _reconcile_container_name(self, instance: str, info: InstalledInfo, model_info: ModelInstalledInfo) -> str | None:  # noqa: ARG002
        return model_info.docker.container_name or model_info.docker.name

    async def _release_dead_model(
        self,
        instance: str,  # noqa: ARG002
        info: InstalledInfo,
        model_id: str,
        model_info: ModelInstalledInfo,
    ) -> None:
        await self.docker_service.uninstall_docker(model_info.docker)
        self._release_gpu_utilization(model_info.gpu_memory_utilization)
        info.models.pop(model_id, None)
        if model_info.model_type == "reranker":
            self.endpoint_registry.unregister_rerank(model_info.registered_name, model_info.registration_id)
        elif model_info.model_type == "embedding":
            self.endpoint_registry.unregister_embeddings(model_info.registered_name, model_info.registration_id)
        else:
            self.endpoint_registry.unregister_chat_completion(model_info.registered_name, model_info.registration_id)
        await self._save()

    def get_docker_options(self, instance: str, model_id: str | None) -> DockerOptions:
        """Return the resolved DockerOptions for this instance/model."""
        info = self.get_instance_installed_info(instance)
        if not model_id:
            raise HTTPException(400, "Docker is not bound with this object")

        model_installed = info.models.get(model_id, None)
        if not model_installed:
            raise HTTPException(status_code=400, detail="Model not installed")

        return model_installed.docker

    def _add_custom_model(self, instance: str, model: CustomModel) -> None:
        parsed = try_parse_pydantic(SglangCustomModel, model.data)

        if not self.models.get(instance):
            self.models[instance] = {}

        if parsed.id in self.models[instance]:
            raise HTTPException(400, "Model with given id already exists.")

        self.models[instance][parsed.id] = SglangModel(
            hf_id=parsed.hf_id, revision=parsed.revision, size=parsed.size, custom=model.id, model_type=parsed.model_type
        )

    def _remove_custom_model(self, instance: str, model: CustomModel) -> None:
        installed = self.get_instance_info(instance).installed
        parsed = try_parse_pydantic(SglangCustomModel, model.data)
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
                    is_loaded = model_id in info.models
                    vram_estimate_gb = await self._get_cached_vram_estimate(instance_name, model_id, is_loaded)
                    out_list.append(
                        RetrieveModelOut(
                            id=model_id,
                            service=self.get_id(instance_name),
                            type=model.model_type,
                            installed=installed,
                            downloaded=model_id in self.models_downloaded,
                            size=model.size,
                            custom=model.custom,
                            spec=self.get_model_type_spec(instance_name, model.model_type),
                            has_docker=True,
                            is_loaded=is_loaded,
                            vram_estimate_gb=vram_estimate_gb,
                            supports_vision=model.supports_vision,
                            supports_audio=model.supports_audio,
                        )
                    )

        return ListModelsOut(list=out_list)

    async def get_model(self, instance: str, model_id: str) -> RetrieveModelOut:
        """Get the model."""
        info = self.get_instance_installed_info(instance)

        if model_id not in self.models[instance]:
            raise HTTPException(status_code=400, detail="Model not found")

        model = self.models[instance][model_id]
        installed = info.models[model_id].get_info() if model_id in info.models else self._get_model_installed_info(instance, model_id)
        is_loaded = model_id in info.models
        vram_estimate_gb = await self._get_cached_vram_estimate(instance, model_id, is_loaded)
        return RetrieveModelOut(
            id=model_id,
            service=self.get_id(instance),
            type=model.model_type,
            installed=installed,
            downloaded=model_id in self.models_downloaded,
            size=model.size,
            custom=model.custom,
            spec=self.get_model_type_spec(instance, model.model_type),
            has_docker=True,
            is_loaded=is_loaded,
            vram_estimate_gb=vram_estimate_gb,
            supports_vision=model.supports_vision,
            supports_audio=model.supports_audio,
        )

    async def _download_model(self, stream: Stream[StreamChunk], model: SglangModel, model_dir: Path) -> Path | None:
        local_model_path: Path | None = None
        progress = Progress(convert_size_to_bytes(model.size) or 0)
        packets = self.model_downloader.download(model.hf_id, model_dir, revision=model.revision)
        async for packet in track_download_progress(stream, packets, progress):
            if isinstance(packet, SuccessDownloadPacket):
                local_model_path = packet.local_path

        stream.emit(
            StreamChunkProgress(
                type="progress", stage="download", value=1, data={"local_model_path": str(local_model_path)} if local_model_path else {}
            )
        )
        if local_model_path is None:
            raise HTTPException(500, f"Downloader doesn't return filepath in {model_dir} of {model.hf_id} model")
        return local_model_path

    def _purge_stale_download(self, model_id: str, local_model_path: Path) -> None:
        """Delete the previous download directory for `model_id` if this install now points at a different one.

        An edit that changes hf_id/revision reinstalls the model into a new directory but leaves the old
        one on disk with nothing referencing it anymore (`models_downloaded[model_id]` is about to be
        overwritten) - clean it up here rather than leaking it forever.
        """
        old_downloaded = self.models_downloaded.get(model_id)
        if old_downloaded and old_downloaded.model_path and old_downloaded.model_path != str(local_model_path):
            shutil.rmtree(Path(old_downloaded.model_path), ignore_errors=True)

    async def _download_model_or_set_progress(
        self, stream: Stream[StreamChunk], download_key: str, model: SglangModel, model_dir: Path
    ) -> Path:
        """Download `model`, deduplicating concurrent installs by `download_key`.

        `download_key` must uniquely identify the (hf_id, revision) pair being downloaded - using the
        bare hf_id would make two concurrent installs of different revisions of the same repository
        collide, handing the second caller the first one's (wrong) `model_dir`.
        """
        local_model_path: Path | None = None
        if download_key not in self.models_download_progress:
            self.models_download_progress[download_key] = stream
            try:
                local_model_path = await self._download_model(stream, model, model_dir)
            finally:
                del self.models_download_progress[download_key]
        else:
            chunk: StreamChunk
            async for chunk in self.models_download_progress[download_key].as_generator():
                if chunk.get("type") == "progress" and chunk.get("stage") == "download":
                    if (data := chunk.get("data")) and (path := data.get("local_model_path")):
                        local_model_path = Path(path)
                    stream.emit(chunk)
                else:
                    break
        if local_model_path is None:
            raise HTTPException(500, f"Downloader doesn't return filepath in {model_dir} of {model.hf_id} model")
        return local_model_path

    def _carry_over_edited_install_options(self, old_options: InstallModelIn, new_options: InstallModelIn) -> InstallModelIn:
        """Keep a legacy absolute `gpu_memory_utilization` across an edit.

        The edit form only renders `gpu_priority`, so without this a model installed with the old absolute
        key would silently lose it on any edit (even one touching only prefix/envs) and be re-sized.
        """
        spec = with_legacy_key_carried_over(old_options.spec, new_options.spec, "gpu_memory_utilization", "gpu_priority")
        return new_options if spec is new_options.spec else new_options.model_copy(update={"spec": spec})

    async def _get_default_gpu_memory_utilization(self) -> float:
        """Size the default utilization to the fraction of VRAM currently free.

        See `VllmService._get_default_gpu_memory_utilization` for the rationale: a flat default can
        request more memory than is actually free, which makes SGLang fail to start.
        """
        stats = await self.hardware.get_realtime_stats(force_refresh=True)
        if not stats or stats.total_vram_gb <= 0:
            logger.warning(
                f"SGLang: GPU stats unavailable, assuming {self._default_gpu_memory_utilization} of total VRAM is free; "
                "a gpu_priority share is applied to that and may exceed the memory actually free"
            )
            return self._default_gpu_memory_utilization
        free_fraction = max(0.0, stats.total_vram_gb - stats.used_vram_gb) / stats.total_vram_gb
        return round(min(self._default_gpu_memory_utilization, free_fraction * self._gpu_memory_free_safety_margin), 2)

    async def _get_gpu_memory_utilization(
        self, instance: str, model_id: str, parsed_model_options: SglangModelOptions, model: SglangModel, *, reserve: bool = True
    ) -> float:
        """Size the utilization for a model; `reserve=False` only validates, without taking the share."""
        gpu_memory_utilization = parsed_model_options.gpu_memory_utilization
        if gpu_memory_utilization is not None and gpu_memory_utilization <= 0:
            # A persisted 0 (the old free-text field allowed it) used to mean "unset" - keep it that way
            # rather than launching the engine with no memory, which fails to start.
            gpu_memory_utilization = None
        if gpu_memory_utilization is None and parsed_model_options.gpu_priority is None:
            # An explicit priority must win over the model definition's own default utilization.
            gpu_memory_utilization = model.gpu_memory_utilization
        if gpu_memory_utilization is None:
            gpu_memory_utilization = self._get_persisted_model_gpu_memory_utilization(instance, model_id, parsed_model_options.gpu_priority)
        if gpu_memory_utilization is None and parsed_model_options.gpu_priority is not None:
            available = await self._get_default_gpu_memory_utilization()
            gpu_memory_utilization = round(parsed_model_options.gpu_priority * available, 2)
            if gpu_memory_utilization < self._min_gpu_memory_utilization:
                raise HTTPException(
                    422,
                    (
                        f"GPU priority {parsed_model_options.gpu_priority} of the currently free VRAM ({available}) "
                        f"gives {gpu_memory_utilization} utilization, below the minimum {self._min_gpu_memory_utilization}. "
                        "Raise the priority or free up GPU memory."
                    ),
                )
        if gpu_memory_utilization is None:
            gpu_memory_utilization = await self._get_default_gpu_memory_utilization()
            if gpu_memory_utilization < self._min_gpu_memory_utilization:
                raise HTTPException(
                    422,
                    f"Not enough free GPU memory: only {gpu_memory_utilization} of VRAM is free. "
                    "Wait for running installs to finish or free up GPU memory.",
                )
        if self.gpu_memory_utilization + gpu_memory_utilization > 1:
            raise HTTPException(
                422,
                (
                    "GPU utilization in all SGLang instances can be maximum 1. "
                    f"Actual GPU utilization is {self.gpu_memory_utilization}. "
                    f"You try add additional {gpu_memory_utilization}, "
                    f"which is {self.gpu_memory_utilization + gpu_memory_utilization - 1} more that 1."
                ),
            )
        if not reserve:
            return gpu_memory_utilization
        self.gpu_memory_utilization += gpu_memory_utilization
        msg = f"SGLang gpu utilization = {self.gpu_memory_utilization}"
        logger.debug(msg)
        return gpu_memory_utilization

    def _release_gpu_utilization(self, utilization: float | None) -> None:
        if utilization:
            self.gpu_memory_utilization -= utilization
            if self.gpu_memory_utilization < 0:
                self.gpu_memory_utilization = 0
            msg = f"SGLang gpu utilization = {self.gpu_memory_utilization}"
            logger.debug(msg)

    async def _get_quantization(self, parsed_model_options: SglangModelOptions, model: SglangModel) -> str | None:
        quantization = parsed_model_options.quantization or model.quantization or None
        if quantization:
            pattern = r"^[a-zA-Z0-9_-]+$"
            if not re.match(pattern, quantization):
                raise HTTPException(422, "Quantization options needs to be made from `a-zA-Z-_`.")
        return quantization

    @staticmethod
    def _is_qwen3_reranker(hf_id: str) -> bool:
        """Return whether `hf_id` names a Qwen3-Reranker-family model, by naming convention.

        Fallback only, used when a reranker carries no architecture-derived
        `is_generative_reranker` verdict (i.e. a hand-added custom model, not one from the
        registry). These models must launch WITHOUT `--is-embedding` (the engine rejects it at
        startup); every other reranker matched by this heuristic is assumed to be a cross-encoder,
        which accepts it.
        """
        lowered = hf_id.lower()
        return _QWEN3_RERANKER_HINT in lowered and "rerank" in lowered

    @classmethod
    def _is_generative_reranker(cls, model: SglangModel) -> bool:
        """Return whether `model` is a generative reranker that must launch WITHOUT `--is-embedding`.

        Registry rerankers carry an architecture-derived verdict (`is_generative_reranker`,
        computed by `just get-sglang-models` from the model's HF `config.json`); custom rerankers
        have none, so they fall back to the by-name heuristic.
        """
        if model.is_generative_reranker is not None:
            return model.is_generative_reranker
        return cls._is_qwen3_reranker(model.hf_id)

    @staticmethod
    def _is_rerank_or_embedding_hf_id(hf_id: str) -> bool:
        """Return whether `hf_id` names a reranker or embedding model, by naming convention.

        Reasoning/tool-call parsers only make sense for chat models; rerankers and embedding
        models virtually always carry "rerank" or "embed" in their hf_id, so this is used as a
        belt-and-suspenders guard alongside the caller's `model_type == "llm"` check.
        """
        lowered = hf_id.lower()
        return "rerank" in lowered or "embed" in lowered

    @staticmethod
    def _detect_reasoning_parser(hf_id: str) -> str | None:
        """Guess SGLang's `--reasoning-parser` value from the model family, or None if unsure."""
        lowered = hf_id.lower()
        if SglangService._is_rerank_or_embedding_hf_id(hf_id):
            return None
        if "qwen3" in lowered and "coder" in lowered:
            # Qwen3-Coder variants are non-thinking tool-calling models, not reasoning models.
            return None
        for hint, parser in _REASONING_PARSER_HINTS:
            if hint in lowered:
                return parser
        return None

    @staticmethod
    def _detect_tool_call_parser(hf_id: str) -> str | None:
        """Guess SGLang's `--tool-call-parser` value from the model family, or None if unsure."""
        if SglangService._is_rerank_or_embedding_hf_id(hf_id):
            return None
        lowered = hf_id.lower()
        for hint, parser in _TOOL_CALL_PARSER_HINTS:
            if hint in lowered:
                return parser
        return None

    def _build_sglang_command(
        self,
        docker_model_path: Path,
        model_id: str,
        model: SglangModel,
        opts: SglangModelOptions,
        quantization: str | None,
        gpu_memory_utilization: float | None,
        max_model_length: int | None,
    ) -> list[str]:
        cmd = [
            "python3",
            "-m",
            "sglang.launch_server",
            "--model-path",
            str(docker_model_path),
            "--host",
            "0.0.0.0",
            "--port",
            "30000",
            "--served-model-name",
            model_id,
        ]

        kv_flags: list[tuple[str, str | None]] = [
            ("--quantization", quantization),
            ("--mem-fraction-static", str(gpu_memory_utilization) if gpu_memory_utilization is not None else None),
            ("--context-length", str(max_model_length) if max_model_length else None),
        ]
        for flag, value in kv_flags:
            if value:
                cmd.extend([flag, value])

        if model.model_type == "embedding" or (model.model_type == "reranker" and not self._is_generative_reranker(model)):
            cmd.append("--is-embedding")

        if model.model_type == "reranker":
            # RoBERTa/XLM-RoBERTa cross-encoders (e.g. BAAI/bge-reranker-*, jina-reranker-v2) crash with
            # `assert torch.equal(positions, expected_pos)` in sglang's roberta.py once the radix cache
            # returns cache-hit position offsets instead of a clean 0..N-1 range. Disabling it sidesteps
            # the crash for every RoBERTa-family model without needing a per-architecture allowlist; the
            # prefix-cache reuse it gives up barely matters for one-off rerank requests anyway.
            # See https://github.com/sgl-project/sglang/issues/8741.
            cmd.append("--disable-radix-cache")

        if model.model_type == "llm":
            if "--reasoning-parser" not in opts.extra_args and (reasoning_parser := self._detect_reasoning_parser(model.hf_id)):
                cmd.extend(["--reasoning-parser", reasoning_parser])
            if "--tool-call-parser" not in opts.extra_args and (tool_call_parser := self._detect_tool_call_parser(model.hf_id)):
                cmd.extend(["--tool-call-parser", tool_call_parser])

        for flag, value in opts.extra_args.items():
            if value:
                cmd.extend([flag, value])
            else:
                cmd.append(flag)

        return cmd

    def _register_model_endpoint(
        self,
        model_info: "ModelInstalledInfo",
        model: SglangModel,
        registered_name: str,
        model_id: str,
        context_window: int | None,
        max_context_window: int | None,
        capacity: int | None,
        registered_alias: str | None = None,
        tokenizer_dir: Path | None = None,
    ) -> RegistrationId:
        """Register the model in the endpoint registry based on its type; embeddings serve their tokenizer from `tokenizer_dir`."""
        # SGLang always sizes a real, finite `max_running_requests` from KV-cache memory at startup;
        # `capacity is None` here means we failed to determine it (docker-log fetch or parse failure),
        # never that it's unbounded.
        capacity_state: CapacityState = capacity if capacity is not None else "unknown"
        if model.model_type == "reranker":
            return self.endpoint_registry.register_rerank_as_proxy(
                model=registered_name,
                props=ModelProps(private=True, type="rerank", endpoints=["/v1/rerank"]),
                options=ProxyOptions(url=f"{model_info.base_url}/v1/rerank", rewrite_model_to=model_id),
                registration_options=RegistrationOptions(
                    origin="local", alias=registered_alias, owned_by=self.get_type(), capacity_state=capacity_state
                ),
                normalize_sglang_response=True,
            )
        if model.model_type == "embedding":
            return self.endpoint_registry.register_embeddings_as_proxy(
                model=registered_name,
                props=ModelProps(
                    private=True,
                    type="embedding",
                    endpoints=EMBEDDINGS_ENDPOINTS,
                    context_window=context_window,
                    max_context_window=max_context_window,
                ),
                options=ProxyOptions(url=f"{model_info.base_url}/v1/embeddings", rewrite_model_to=model_id),
                tokenizer=functools.partial(tokenizer_from_dir, tokenizer_dir) if tokenizer_dir else None,
                registration_options=RegistrationOptions(
                    origin="local", alias=registered_alias, owned_by=self.get_type(), capacity_state=capacity_state
                ),
            )
        return self.endpoint_registry.register_chat_completion_as_proxy(
            model=registered_name,
            props=ModelProps(
                private=True,
                type="llm",
                context_window=context_window,
                max_context_window=max_context_window,
                supports_vision=model.supports_vision,
                supports_audio=model.supports_audio,
            ),
            chat_completions=ProxyOptions(
                url=f"{model_info.base_url}/v1/chat/completions", rewrite_model_to=model_id, forward_extra_fields=True
            ),
            completions=ProxyOptions(url=f"{model_info.base_url}/v1/completions", rewrite_model_to=model_id, forward_extra_fields=True),
            responses=ProxyOptions(url=f"{model_info.base_url}/v1/responses", rewrite_model_to=model_id),
            messages=ProxyOptions(url=f"{model_info.base_url}/v1/messages", rewrite_model_to=model_id),
            ollama_chat=None,
            registration_options=RegistrationOptions(
                origin="local", alias=registered_alias, owned_by=self.get_type(), capacity_state=capacity_state
            ),
        )

    @staticmethod
    def _is_embedding_mismatch_error(raw: str) -> bool:
        """Return whether container logs report the engine rejected the resolved `--is-embedding` choice."""
        return "relaunch without --is-embedding" in raw.lower()

    @staticmethod
    def _parse_max_concurrency(raw: str) -> int | None:
        """Parse SGLang's dynamically-computed max running requests from its KV-cache startup log line.

        SGLang logs e.g. "max_total_num_tokens=665690, chunked_prefill_size=8192, max_prefill_tokens=16384,
        max_running_requests=4096, context_len=65536, available_gpu_mem=13.50 GB" once it has sized the KV
        cache from real free VRAM; `max_running_requests` is the number of concurrent requests the instance
        can serve. On (re)start it may log this line more than once (e.g. across an embedding-flag retry),
        so the *last* occurrence reflects the config the instance actually launched with. A missing log line
        (e.g. the log format changed across an SGLang version) falls back to `None` rather than failing
        install; `None` here means "failed to determine," never "unbounded" - callers must treat it as
        unknown capacity.
        """
        matches = list(re.finditer(r"max_running_requests=(\d+)", raw))
        if not matches:
            return None
        raw_concurrency = int(matches[-1].group(1))
        if raw_concurrency < 1:
            logger.warning(
                "SGLang reported max_running_requests=%d, which is below 1; treating capacity as 1 instead of unknown.",
                raw_concurrency,
            )
        return max(1, raw_concurrency)

    async def _resolve_context_windows(
        self, model: SglangModel, model_id: str, local_model_path: Path, base_url: str, max_model_length: int | None
    ) -> tuple[int | None, int | None]:
        """Return `(context_window, max_context_window)` for a new model; for embeddings, one below SGLang's `context_len`."""
        if model.model_type == "llm":
            max_context_window = await get_model_dir_context_window(local_model_path)
            return max_model_length or max_context_window, max_context_window
        if model.model_type == "embedding":
            context_length = await fetch_served_max_model_len(base_url, model_id) or max_model_length
            limit = context_length - 1 if context_length else None
            return limit, limit
        return None, None

    async def _install_model(
        self, instance: str, model_id: str, options: InstallModelIn, container_name_override: str | None = None
    ) -> PromiseWithProgress[InstallModelOut, StreamChunk]:
        parsed_model_options = try_parse_pydantic(SglangModelOptions, options.spec) if options.spec else SglangModelOptions()

        info = self.get_instance_installed_info(instance)

        key = (instance, model_id)
        if model_id in info.models or key in self._installing:
            return PromiseWithProgress(value=InstallModelOut(status="OK", details="Already installed or being installed right now."))

        if model_id not in self.models[instance]:
            raise HTTPException(400, f"Model {model_id!r} not found")

        self._installing.add(key)
        model = self.models[instance][model_id]

        try:
            # Fail fast (before a long download) on options that can never fit; the share is really taken in `func`.
            await self._get_gpu_memory_utilization(instance, model_id, parsed_model_options, model, reserve=False)
            max_model_length = parsed_model_options.max_model_length or model.max_model_len or None
            quantization = await self._get_quantization(parsed_model_options, model)
        except BaseException:
            self._installing.discard(key)
            raise

        async def func(stream: Stream[StreamChunk]) -> InstallModelOut:
            model_info: ModelInstalledInfo | None = None
            docker_options: DockerOptions | None = None
            gpu_memory_utilization: float | None = None
            adopted = False
            try:
                model_id_fixed = model.hf_id.replace("/", "-")
                if model.revision and model.revision != "main":
                    model_id_fixed = f"{model_id_fixed}--{normalize_name(model.revision)}"
                models_dir = self._get_working_dir() / "models"
                model_dir = models_dir / model_id_fixed
                model_dir.mkdir(parents=True, exist_ok=True)
                local_model_path: Path | None = await self._download_model_or_set_progress(stream, model_id_fixed, model, model_dir)

                stream.emit(StreamChunkProgress(type="progress", stage="install", value=0, data={}))
                docker_model_path = Path(self.hugging_face_cache_path) / "hub" / model_id_fixed
                model_cache_dir = model_dir / ".cache"
                model_cache_dir.mkdir(parents=True, exist_ok=True)
                volumes = [
                    f"{local_model_path}:{docker_model_path}",
                    f"{model_cache_dir}:{docker_model_path}/.cache",
                ]

                # Sizing and the container start/health wait are serialized: once the previous container is healthy its
                # weights are in `used_vram_gb`, so the next model is sized from a measurement that already includes it.
                # Downloading stays outside the lock, so installs still download in parallel.
                async with self.hardware.gpu_load_lock:
                    gpu_memory_utilization = await self._get_gpu_memory_utilization(instance, model_id, parsed_model_options, model)
                    sglang_command = self._build_sglang_command(
                        docker_model_path=docker_model_path,
                        model_id=model_id,
                        model=model,
                        opts=parsed_model_options,
                        quantization=quantization,
                        gpu_memory_utilization=gpu_memory_utilization,
                        max_model_length=max_model_length,
                    )

                    env_vars = dict(model.env_vars or {})
                    env_vars["HF_HUB_OFFLINE"] = "1"
                    env_vars["HF_HOME"] = self.hugging_face_cache_path
                    env_vars.update(parsed_model_options.extra_envs)

                    image = self._get_image(info.options.spec.get("image_version") if info.options else None)

                    subnet = self.docker_service.get_docker_subnet()
                    service_name = f"{self.get_service_id(instance)}-{normalize_name(model_id)}"
                    docker_options = DockerOptions(
                        name=service_name,
                        container_name=container_name_override or self.docker_service.get_docker_container_name(service_name),
                        image=image.name,
                        command=sglang_command,
                        image_port=30000,
                        env_vars=env_vars,
                        restart="unless-stopped",
                        volumes=volumes,
                        hardware=self.get_specified_hardware_parts(info.parsed_options.hardware),
                        shm_size=model.shm_size,
                        ulimits=model.ulimits,
                        subnet=subnet,
                        healthcheck={
                            "test": "curl --fail http://localhost:30000/health_generate || exit 1",
                            "interval": "80s",
                            "timeout": "60s",
                            "retries": "3",
                            "start_period": "240s",
                        },
                    )
                    docker_exposed_port, restarted, adopted = await self._start_container_with_embedding_fallback(
                        docker_options, sglang_command
                    )

                capacity, capacity_known = await self._resolve_model_capacity(
                    instance, model_id, "SGLang", docker_options.container_name or "", restarted, self._parse_max_concurrency
                )

                registered_name = parsed_model_options.rename if parsed_model_options.rename else model_id
                registered_alias = parsed_model_options.alias
                container_host = self.docker_service.get_container_host(subnet, docker_options)
                container_port = self.docker_service.get_container_port(subnet, docker_exposed_port, docker_options.image_port)
                context_window, max_context_window = await self._resolve_context_windows(
                    model, model_id, local_model_path, get_base_url(container_host, container_port), max_model_length
                )
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
                    capacity=capacity,
                    capacity_known=capacity_known,
                )
                model_info.registration_id = self._register_model_endpoint(
                    model_info=model_info,
                    model=model,
                    registered_name=registered_name,
                    registered_alias=registered_alias,
                    model_id=model_id,
                    context_window=context_window,
                    max_context_window=max_context_window,
                    capacity=capacity,
                    tokenizer_dir=local_model_path,
                )
                stream.emit(StreamChunkProgress(type="progress", stage="install", value=1, data={}))
                self._purge_stale_download(model_id, local_model_path)
                self.models_downloaded[model_id] = DownloadedInfo(str(local_model_path))
                return InstallModelOut(status="OK", details="Installed")
            except BaseException:
                self._release_gpu_utilization(gpu_memory_utilization)
                if model_info is not None and info.models.get(model_id) is model_info:
                    info.models.pop(model_id, None)
                if docker_options is not None:
                    await self._rollback_failed_install_docker(docker_options, adopted=adopted)
                raise
            finally:
                self._installing.discard(key)

        return PromiseWithProgress(func=func)

    async def _start_container_with_embedding_fallback(
        self, docker_options: DockerOptions, sglang_command: list[str]
    ) -> tuple[int, bool, bool]:
        """Start the container, retrying once without `--is-embedding` if the engine rejects it.

        Our `--is-embedding` heuristic (model_type/reranker-family based) isn't perfect; when SGLang
        itself rejects the flag at startup, retrying without it lets the install succeed without
        requiring the user to intervene.
        Returns (docker_exposed_port, whether the container was (re)started, whether it was adopted).
        """
        try:
            return await self.docker_service.install_and_run_docker(docker_options)
        except Exception as exc:
            await self.docker_service.stop_docker(docker_options)
            if not (self._is_embedding_mismatch_error(str(exc)) and "--is-embedding" in sglang_command):
                raise self._describe_install_failure(exc) from exc

        docker_options.command = [arg for arg in sglang_command if arg != "--is-embedding"]
        try:
            return await self.docker_service.install_and_run_docker(docker_options)
        except Exception as retry_exc:
            await self.docker_service.stop_docker(docker_options)
            raise self._describe_install_failure(retry_exc) from retry_exc

    def _describe_install_failure(self, exc: Exception) -> Exception:
        """Turn an `--is-embedding` startup mismatch into an actionable error, otherwise pass through."""
        if self._is_embedding_mismatch_error(str(exc)):
            return HTTPException(
                400,
                "SGLang rejected the '--is-embedding' flag for this model. "
                "This usually means the model belongs to a reranker family (e.g. Qwen3-Reranker) that "
                "must be launched without it.",
            )
        return exc

    def _get_image(self, image_version: str | None = None) -> DockerImage:
        if image_version:
            base_name = _const.image.name.split(":")[0]
            return DockerImage(name=f"{base_name}:{image_version}", size=_const.image.size)
        return _const.image

    def get_docker_image_repo(self, hardware: str | None) -> str:  # noqa: ARG002
        """Return the Docker repo tags are fetched from (SGLang is GPU-only, so hardware doesn't select a repo)."""
        return _const.image.name.split(":")[0]

    async def get_docker_tags(self, hardware: str | None) -> list[str]:
        """Fetch available Docker image tags for the SGLang image."""
        base_image = self.get_docker_image_repo(hardware)
        client = registry_for(base_image, self.config.docker_hub_token or None)
        tags = await client.get_tags(image_without_registry_prefix(base_image))
        return self.filter_docker_tags(tags, hardware)

    def filter_docker_tags(self, tags: list[str], hardware: str | None) -> list[str]:  # noqa: ARG002
        """Filter tags to CUDA (cuNN) GPU variants, excluding ROCm/CANN/Xeon/CPU builds."""
        return [t for t in tags if re.search(r"cu\d", t, re.IGNORECASE) and "xeon" not in t.lower()]

    def get_default_docker_tag(self, hardware: str | None) -> str | None:  # noqa: ARG002
        """Return the pinned default SGLang image tag."""
        return _const.image.name.split(":")[1]

    async def _uninstall_model(self, instance: str, model_id: str, options: UninstallModelIn) -> None:
        info = self.get_instance_installed_info(instance)

        if model_id in info.models:
            model = info.models[model_id]
            self._release_gpu_utilization(model.gpu_memory_utilization)
            del info.models[model_id]
            if model.model_type == "reranker":
                self.endpoint_registry.unregister_rerank(model.registered_name, model.registration_id)
            elif model.model_type == "embedding":
                self.endpoint_registry.unregister_embeddings(model.registered_name, model.registration_id)
            else:
                self.endpoint_registry.unregister_chat_completion(model.registered_name, model.registration_id)
            await self.docker_service.uninstall_docker(model.docker)

        if options.purge and model_id in self.models_downloaded:
            model = self.models_downloaded[model_id]
            if model_path := model.model_path:
                shutil.rmtree(Path(model_path))
            del self.models_downloaded[model_id]

    def _get_vram_estimate(self, model_info: ModelInstalledInfo) -> float | None:
        """Return estimated VRAM in GiB based on gpu_memory_utilization fraction of total VRAM."""
        if (utilization := model_info.gpu_memory_utilization) is None or (total_vram := self.hardware.total_vram_gb) <= 0:
            return None
        return round(utilization * total_vram, 2)

    async def _get_cached_vram_estimate(self, instance_name: str, model_id: str, is_loaded: bool) -> float | None:
        """Return cached VRAM estimate for a loaded model, evicting the cache when the model is unloaded."""
        cache_key = (instance_name, model_id)
        if not is_loaded:
            self._vram_cache.pop(cache_key, None)
            return None
        if cache_key in self._vram_cache:
            return self._vram_cache[cache_key]
        vram_estimate_gb: float | None = None
        info = self.get_instance_installed_info(instance_name)
        if model_info := info.models.get(model_id):
            vram_estimate_gb = self._get_vram_estimate(model_info)
        if vram_estimate_gb is not None:
            self._vram_cache[cache_key] = vram_estimate_gb
        return vram_estimate_gb

    async def stop_instance(self, instance: str) -> None:
        """Stop all the SGLang service Docker containers."""
        self._stop_reconciliation_task(instance)
        installed = self.get_instance_info(instance).installed
        if not installed:
            return
        await self._stop_dockers_parallel([model.docker for model in installed.models.values()])
