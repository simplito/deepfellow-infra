# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Vllm service."""

import asyncio
import contextlib
import functools
import json
import re
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict

import aiohttp
from fastapi import HTTPException
from pydantic import BaseModel, Field

from scripts.get_huggingface_models import (
    ModelIncompatibleError,
    fetch_model_metadata,
    fetch_registry_entries,
    validate_model_compatibility,
)
from server.applicationcontext import get_base_url
from server.config import get_main_dir
from server.docker import (
    DockerImage,
    DockerOptions,
)
from server.endpointregistry import CapacityState, ProxyOptions, RegistrationId, RegistrationOptions
from server.models.api import EMBEDDINGS_ENDPOINTS, STT_ENDPOINTS, ModelProps
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
    ServiceSize,
    ServiceSpecification,
    UninstallServiceIn,
)
from server.services import model_catalog_refresh
from server.services.base2_service import Base2Service, CustomModel, Instance, InstanceConfig, ModelConfig
from server.services.finetune_service import FinetuneService
from server.services.model_catalog_refresh import huggingface_refresh_guard
from server.utils.core import (
    DownloadedPacket,
    PreDownloadPacket,
    PromiseWithProgress,
    Stream,
    StreamChunk,
    StreamChunkProgress,
    SuccessDownloadPacket,
    Utils,
    convert_size_to_bytes,
    get_cpu_architecture,
    normalize_name,
    try_parse_pydantic,
    with_legacy_key_carried_over,
)
from server.utils.exceptions import AppError
from server.utils.files import get_model_dir_context_window
from server.utils.hardware import HardwarePartInfo
from server.utils.loading import Progress
from server.utils.logger import logger
from server.utils.model_downloader import AdapterRegistryDownloader
from server.utils.registry_client import image_without_registry_prefix, registry_for
from server.utils.served_model_len import fetch_served_max_model_len
from server.utils.size_fetcher import fetch_huggingface_model_size, fmt_size, sum_siblings_size_bytes
from server.utils.tokenizer_archive import tokenizer_from_dir

# Mirrors the `just get-vllm-models` release recipe (see justfile): one fetch per model type,
# each merged into its own registry key so a refresh reproduces exactly what the offline,
# reviewed release process would generate.
_VLLM_FETCH_PLAN: list[tuple[int, int, int, str, bool]] = [
    (200, 200, 50, "llm", False),
    (10, 0, 10, "reranker", False),
    (10, 0, 10, "embedding", False),
    (50, 0, 10, "stt", False),
]

# vLLM's `--quantization` engine argument choices, as reported by this pinned image's own
# `ModelConfig` validation error (differs from vLLM's public docs, which lag this build).
# Left unset (empty string, not omitted - see the `default=""` on the fields using this list,
# needed so the webui's oneof-field auto-select-first-value-when-no-default behavior doesn't
# silently pick "awq" here instead of leaving quantization unset), vLLM auto-detects it from
# the model config.
# Kept as a plain tuple (not a shared list constant) since `ModelField.values` and
# `CustomModelField.values` declare incompatible (invariant) list element types.
_VLLM_QUANTIZATION_METHODS = (
    "awq",
    "auto_awq",
    "fp8",
    "fbgemm_fp8",
    "fp_quant",
    "modelopt",
    "modelopt_fp4",
    "modelopt_mxfp8",
    "modelopt_mixed",
    "auto_gptq",
    "gptq",
    "gptq_marlin",
    "awq_marlin",
    "humming",
    "compressed-tensors",
    "bitsandbytes",
    "experts_int8",
    "quark",
    "moe_wna16",
    "torchao",
    "inc",
    "mxfp4",
    "gpt_oss_mxfp4",
    "deepseek_v4_fp8",
    "online",
    "fp8_per_tensor",
    "fp8_per_block",
    "fp8_per_channel",
    "int8_per_channel_weight_only",
    "nvfp4_per_token",
    "mxfp8",
)

#: The two files vLLM's static `--lora-modules` loader reads from a LoRA adapter directory.
_ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")

#: File in a downloaded adapter directory holding the registry's `created` timestamp of the downloaded weights.
_ADAPTER_CREATED_MARKER = ".registry_created"

#: The only `--max-lora-rank` values vLLM accepts; any other value makes the container fail to start.
_VLLM_ALLOWED_LORA_RANKS = (1, 8, 16, 32, 64, 128, 256, 320, 512)


def _round_up_lora_rank(rank: int) -> int:
    """Round a LoRA adapter's rank up to the smallest `--max-lora-rank` value vLLM accepts.

    `--max-lora-rank` is an upper bound, so rounding up still serves the adapter at its real rank.
    Ranks above vLLM's maximum are rejected when the model is added; one that got persisted anyway is
    passed through unchanged so vLLM reports it rather than silently serving a truncated bound.
    """
    return next((allowed for allowed in _VLLM_ALLOWED_LORA_RANKS if allowed >= rank), rank)


def _is_truthy_flag(value: Any) -> bool:  # noqa: ANN401
    """Interpret a raw (pre-pydantic) custom-model spec value as a bool checkbox field.

    Spec values arrive as whatever JSON the caller sent - typically a real `bool`, but a form-style
    submission could send a string. Mirrors Pydantic's own string-to-bool coercion (used when the
    spec is later parsed as `VllmCustomModel`) so a direct API caller isn't seen differently by the
    two: "false"/"no"/"off"/"0" (case-insensitive) or empty count as unset/off; anything else is on.
    """
    if isinstance(value, str):
        return value.strip().lower() not in ("", "false", "no", "off", "0")
    return bool(value)


type VllmModelType = Literal["llm", "reranker", "embedding", "stt"]


def _hf_source(spec: dict[str, Any]) -> tuple[Any, Any, Any]:
    """Return the custom-model spec fields the HuggingFace validation depends on: repo, revision, model type.

    Falsy values are normalized (to `None`, or to `"llm"` for `model_type`, its default) so a field
    missing from an older stored spec doesn't compare as different from one submitted as `""`.
    """
    return (spec.get("hf_id") or None, spec.get("revision") or None, spec.get("model_type") or "llm")


class VllmModel(BaseModel):
    hf_id: str
    revision: str | None = None
    env_vars: dict[str, str] | None = None
    quantization: str | None = None
    dtype: str = "auto"
    shm_size: str = "16gb"
    ulimits: dict[str, str] | None = None
    max_model_len: int | None = None
    gpu_memory_utilization: float | None = None
    size: str
    custom: CustomModelId | None = None
    model_type: VllmModelType = "llm"
    adapter_id: str | None = None
    """df-finetune adapter id this model serves as a LoRA on top of `hf_id`, if any."""
    lora_rank: int | None = None
    """LoRA rank resolved from the df-finetune adapter registry; required when `adapter_id` is set."""
    # Best-effort, derived from the HF repo's `config.architectures` at catalog-build time (see
    # `classify_modality_support`). None means unresolved (e.g. a hand-added custom model, or a
    # non-"llm" registry entry, for which modality support isn't computed).
    supports_vision: bool | None = None
    supports_audio: bool | None = None


class VllmCustomModel(BaseModel):
    id: str
    hf_id: str
    revision: str | None = None
    size: str
    model_type: VllmModelType = "llm"
    skip_validation: bool = False
    adapter_id: str | None = None
    lora_rank: int | None = None
    quantization: str | None = None


type ImageTypes = Literal["cpu", "gpu"]


class VllmRegistryEntry(TypedDict):
    name: str
    size: str


class VllmLlmRegistryEntry(VllmRegistryEntry, total=False):
    supports_vision: bool
    supports_audio: bool


class VllmRegistry(TypedDict, total=False):
    llms: list[VllmLlmRegistryEntry]
    rerankers: list[VllmRegistryEntry]
    embeddings: list[VllmRegistryEntry]
    stts: list[VllmRegistryEntry]


class VllmConst(BaseModel):
    images: dict[ImageTypes, DockerImage]
    models: dict[str, VllmModel]


# "v0.31.0", "v0.31.0-ubuntu2404", "v0.31.0-x86_64": the default (CUDA-suffix-less) build of a release
_PLAIN_RELEASE_TAG = re.compile(r"^v\d+\.\d+\.\d+(-(aarch64|x86_64))?(-ubuntu\d+)?$")


def _is_gpu_tag(tag: str) -> bool:
    """Whether the tag of the GPU vLLM repo is a CUDA build: explicit `-cu*` or the plain release tag."""
    if "cpu" in tag.lower():
        return False
    return bool(re.search(r"cu\d", tag, re.IGNORECASE) or _PLAIN_RELEASE_TAG.match(tag))


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
                supports_vision=entry.get("supports_vision"),
                supports_audio=entry.get("supports_audio"),
            )
        for entry in registry.get("rerankers", []):
            map[entry["name"]] = VllmModel(
                hf_id=entry["name"],
                size=entry["size"],
                model_type="reranker",
            )
        for entry in registry.get("embeddings", []):
            map[entry["name"]] = VllmModel(
                hf_id=entry["name"],
                size=entry["size"],
                model_type="embedding",
            )
        for entry in registry.get("stts", []):
            map[entry["name"]] = VllmModel(
                hf_id=entry["name"],
                size=entry["size"],
                model_type="stt",
            )
        return map


_const = VllmConst(
    images={
        # Plain CUDA 13 build: the -cu129 builds of v0.30-v0.32 don't start.
        "gpu": DockerImage(name="vllm/vllm-openai:v0.33.0", size="9.2 GB"),
        "cpu": DockerImage(name="vllm/vllm-openai-cpu:v0.33.0", size="1.6 GB"),
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
    model_type: VllmModelType = "llm"
    capacity: int | None = None
    capacity_known: bool = False

    def get_info(self) -> ModelInfo:
        """Get info."""
        return ModelInfo(spec=self.options.spec, registration_id=self.registration_id)


class VllmOptions(BaseModel):
    hardware: str | bool | None = None


class VllmModelOptions(BaseModel):
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
    parsed_options: VllmOptions


@dataclass
class DownloadedInfo:
    model_path: str | None
    adapter_path: str | None = None


@dataclass(frozen=True)
class AdapterMetadata:
    """A df-finetune adapter's registry metadata; `created` is None when the registry doesn't know it."""

    base_model: str
    rank: int
    created: int | None


class VllmService(Base2Service[InstalledInfo, DownloadedInfo]):
    hugging_face_cache_path = "/mnt/hf"
    models: dict[str, dict[str, VllmModel]]
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
        """Fetch current top vLLM-compatible LLM/reranker/embedding/STT models from HuggingFace.

        Runs as a background job (dozens-to-hundreds of HTTP requests) and, on success, writes the
        merged registry to `static/vllm-min.json` and reloads `_const.models` in memory. Failure
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
                vllm_path = get_main_dir() / "./static/vllm-min.json"
                existing_registry: dict[str, Any] = json.loads(vllm_path.read_text(encoding="utf-8")) if vllm_path.exists() else {}

                merged: dict[str, list[dict[str, Any]]] = {}
                async with aiohttp.ClientSession() as session:
                    for i, (top_downloads, top_likes, top_trending, model_type, allow_generative) in enumerate(_VLLM_FETCH_PLAN):
                        stream.emit(
                            StreamChunkProgress(
                                type="progress", stage="install", value=i / len(_VLLM_FETCH_PLAN), data={"stage": model_type}
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
                model_catalog_refresh.atomic_write_json(vllm_path, registry)

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
        with contextlib.suppress(HTTPException):
            promise = await self.refresh_catalog()

            async def _consume_result() -> None:
                try:
                    await promise.wait()
                except Exception:
                    logger.warning("Background catalog refresh triggered by sync_models failed", exc_info=True)

            asyncio.create_task(_consume_result())  # noqa: RUF006 - fire-and-forget, retrieves the promise's exception

    def get_type(self) -> str:
        """Return the service id."""
        return "vllm"

    def get_description(self) -> str:
        """Return the service description."""
        return "Self-hosted LLM high efficient model runner with complex configuration."

    def get_size(self) -> ServiceSize:
        """Return the service size."""
        sizes: dict[str, str] = {}
        if self._cpu_supported:
            sizes["cpu"] = _const.images["cpu"].size
        if self._supported_gpus:
            sizes["gpu"] = _const.images["gpu"].size
        return sizes

    def get_spec(self) -> ServiceSpecification:
        """Return the service specification."""
        fields = self.add_hardware_field_to_spec(cpu_supported=self._cpu_supported)
        fields.append(
            ServiceField(
                type="docker-tags",
                name="image_version",
                description="Docker image version",
                docker_image=_const.images["gpu"].name.split(":")[0],
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
            options.spec.get("")
            if "hardware" not in options.spec:
                options.spec["hardware"] = options.spec.get("gpu", self.docker_service.has_gpu_support)
            parsed_options = try_parse_pydantic(VllmOptions, options.spec)
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
        if model_type != "stt":
            fields.append(ModelField(type="number", name="max_model_length", description="Max model length ex. 4096", required=False))
        fields.extend(
            [
                ModelField(
                    type="oneof",
                    name="quantization",
                    description="Quantization",
                    values=[*_VLLM_QUANTIZATION_METHODS],
                    default="",
                    required=False,
                ),
                ModelField(
                    type="map",
                    name="extra_args",
                    description="Additional vLLM arguments. Key: flag name (e.g. --dtype), value: flag value (empty for boolean flags).",
                    required=False,
                    placeholder="extra_args",
                    default='{"--trust-remote-code":""}' if model_type in ("reranker", "embedding") else None,
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
                CustomModelField(
                    type="text",
                    name="adapter_id",
                    description="df-finetune adapter ID (leave blank to install a plain base model via hf_id instead)",
                    placeholder="pirate-llama",
                    required=False,
                ),
                CustomModelField(
                    type="text",
                    name="hf_id",
                    description="Hugging face model ID (ignored if adapter_id is set - resolved automatically from the adapter)",
                    placeholder="google/gemma-3-270m-it",
                    required=False,
                ),
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
                    values=["llm", "reranker", "embedding", "stt"],
                    default="llm",
                    required=False,
                ),
                CustomModelField(
                    type="oneof",
                    name="quantization",
                    description="Quantization",
                    values=[*_VLLM_QUANTIZATION_METHODS],
                    default="",
                    required=False,
                ),
                CustomModelField(
                    type="bool",
                    name="skip_validation",
                    description=(
                        "Skip the HuggingFace compatibility heuristics: GGUF, chat template, architecture "
                        "(use only if they wrongly reject a repo that does work). Repo existence and access are still checked"
                    ),
                    default="false",
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

    async def _validate_custom_model(
        self,
        spec: dict[str, Any],
        instance: str = "",  # noqa: ARG002
        *,
        previous: dict[str, Any] | None = None,
    ) -> None:
        """Validate hf_id compatibility, opportunistically filling in `size` from the same fetch.

        `_resolve_custom_model_size` would otherwise issue a second, near-identical HuggingFace
        request for the same repo right after this one - filling `size` here whenever the metadata
        was already fetched successfully lets `add_custom_model`'s `if not spec.get("size")` guard
        skip that redundant call. This ordering (validate, then resolve size only if still unset) is
        load-bearing for that optimization to take effect.

        The heuristics `validate_model_compatibility` relies on (GGUF name matching, architecture
        detection) are tuned for filtering a large curated catalog, where an occasional false
        positive silently drops one candidate among hundreds - here they gate a single explicit user
        action instead, so `spec["skip_validation"]` lets a user bypass those heuristics for a repo
        they've confirmed works despite being misdetected. It doesn't bypass `fetch_model_metadata`'s
        check that the repo/revision exists and is accessible: the download uses the same host and
        token, so a definite "not found"/"no access" there can't be a false positive.

        On an edit (`previous` set) that keeps the repo, revision and model type, HuggingFace isn't
        queried at all: the model already passed this validation when it was added, and its weights
        may already be downloaded, so an edit of an unrelated field (e.g. `quantization`) mustn't be
        blocked by a since-revoked or removed token, or a repo that has since become gated.

        Sends the configured HuggingFace token, if any, so a gated or private repo that
        `add_custom_model` can already download validates the same way it downloads, instead of
        being wrongly rejected as missing/inaccessible when accessed anonymously.

        If `adapter_id` is set, this is a df-finetune LoRA adapter install rather than a plain HF
        model: `hf_id`/`lora_rank` are resolved from the adapter registry (overwriting whatever was
        submitted) before the usual HF-compatibility check runs against the resolved `hf_id`.
        """
        adapter_id = spec.get("adapter_id")
        if adapter_id:
            await self._resolve_adapter_metadata(spec, adapter_id)
        elif not spec.get("hf_id"):
            raise HTTPException(400, "Either 'hf_id' or 'adapter_id' must be provided.")
        hf_id = spec.get("hf_id")
        if not hf_id:
            return
        if previous is not None and _hf_source(previous) == _hf_source(spec):
            return
        token = self.get_hugging_face_token()
        revision = spec.get("revision")
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20), headers=Utils.create_bearer_header(token)) as session:
            try:
                if _is_truthy_flag(spec.get("skip_validation")):
                    data = await fetch_model_metadata(session, hf_id, revision, authenticated=bool(token))
                else:
                    data = await validate_model_compatibility(
                        session, hf_id, revision, spec.get("model_type") or "llm", authenticated=bool(token)
                    )
            except ModelIncompatibleError as exc:
                raise HTTPException(400, str(exc)) from exc
        if data is not None and not spec.get("size"):
            total = sum_siblings_size_bytes(data.get("siblings", []))
            if total:
                spec["size"] = fmt_size(total)

    def _get_adapter_registry_base_url(self) -> str:
        """Live base URL of the running `deepfellow-finetune` container.

        Resolved from `FinetuneService`'s installed-model bookkeeping instead of a static config
        value - the container's host port is assigned dynamically by `PortService` at install
        time and isn't guaranteed to stay the same across reinstalls.
        """
        finetune_service = self.services_manager.services.get("finetune")
        base_url = finetune_service.get_installed_base_url() if isinstance(finetune_service, FinetuneService) else None
        if not base_url:
            raise HTTPException(400, "deepfellow-finetune is not installed. Install the Finetune service before referencing an adapter_id.")
        return base_url

    async def _resolve_adapter_metadata(self, spec: dict[str, Any], adapter_id: str) -> None:
        """Resolve base `hf_id` and LoRA `rank` from the df-finetune adapter registry, mutating `spec` in place."""
        metadata = await self._fetch_adapter_metadata(adapter_id)
        spec["hf_id"] = metadata.base_model
        spec["lora_rank"] = metadata.rank

    async def _fetch_adapter_metadata(self, adapter_id: str) -> AdapterMetadata:
        """Fetch and validate an adapter's metadata from the df-finetune adapter registry."""
        base_url = self._get_adapter_registry_base_url()
        url = f"{base_url.rstrip('/')}/v1/adapters/{Utils.str_encode(adapter_id)}"
        headers = Utils.create_bearer_header(self.config.adapter_registry_secret.get_secret_value())
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20), headers=headers) as session:
            try:
                async with session.get(url) as resp:
                    if resp.status == 404:
                        raise HTTPException(400, f"Adapter {adapter_id!r} not found in adapter registry.")
                    resp.raise_for_status()
                    data = await resp.json()
            except (aiohttp.ClientError, TimeoutError) as exc:
                raise HTTPException(400, f"Could not reach adapter registry at {base_url}: {exc!r}") from exc
        if "safetensors" not in (data.get("formats") or []):
            raise HTTPException(400, f"Adapter {adapter_id!r} has no safetensors LoRA files available.")
        rank = (data.get("lora") or {}).get("rank")
        base_model = data.get("base_model")
        if not base_model or not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
            raise HTTPException(400, f"Adapter {adapter_id!r} is missing base_model or has invalid lora.rank metadata: {rank!r}.")
        max_rank = _VLLM_ALLOWED_LORA_RANKS[-1]
        if rank > max_rank:
            raise HTTPException(400, f"Adapter {adapter_id!r} has LoRA rank {rank}, above vLLM's maximum of {max_rank}.")
        created = data.get("created")
        # df-finetune reports 0 when it has no creation timestamp for the adapter
        known_created = created if isinstance(created, int) and not isinstance(created, bool) and created > 0 else None
        return AdapterMetadata(base_model=base_model, rank=rank, created=known_created)

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

    def _restore_model_definition(self, instance: str, model_id: str, definition: dict[str, Any]) -> None:
        self.models[instance][model_id] = VllmModel.model_validate(definition)

    def get_hardware_unsupported_reason(self) -> str | None:
        """Return why vLLM can't run here, or None if it can."""
        if not self._cpu_supported and not self._supported_gpus:
            return (
                "Your hardware doesn't support this service.\nCPU does not support AVX 512 instructions."
                "\nThere are not any graphic cards. Restart Infra if this has changed."
            )
        return None

    @property
    def _cpu_supported(self) -> bool:
        """Return whether the vLLM CPU image can run here: AVX-512 on x86, no such requirement on ARM (AVX-512 is x86-only)."""
        return self._is_arm_host() or self.hardware.cpu.avx512

    @staticmethod
    def _is_arm_host() -> bool:
        return get_cpu_architecture() == "arm64"

    def get_default_docker_image_name(self) -> str | None:
        """Return the default docker image, for the upfront platform-compatibility check.

        Mirrors the default hardware selection in `is_given_hardware_support_gpu(None)`, without
        going through that method - it raises when hardware is unsupported, which must not happen
        here (this runs unconditionally at startup, before hardware support is even relevant).
        """
        return self._get_image(bool(self._supported_gpus)).name

    def _ensure_hardware_supported(self) -> None:
        if reason := self.get_hardware_unsupported_reason():
            raise HTTPException(400, reason)

    def is_given_hardware_support_gpu(self, hardware_specification: str | bool | None) -> bool:
        """Return is gpu will be used."""
        self._ensure_hardware_supported()
        return super().is_given_hardware_support_gpu(hardware_specification)

    def get_specified_hardware_parts(self, hardware_specification: str | bool | None) -> Sequence[HardwarePartInfo]:
        """Get hardware based on user input."""
        self._ensure_hardware_supported()
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
        parsed_options = try_parse_pydantic(VllmOptions, options.spec)
        image_version = options.spec.get("image_version")
        image = self._get_image(self.is_given_hardware_support_gpu(parsed_options.hardware), image_version)
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
                for image in _const.images.values():
                    await self.docker_service.remove_image(image.name)
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

    async def _release_dead_model(self, instance: str, info: InstalledInfo, model_id: str, model_info: ModelInstalledInfo) -> None:  # noqa: ARG002
        await self.docker_service.uninstall_docker(model_info.docker)
        self._release_gpu_utilization(model_info.gpu_memory_utilization)
        info.models.pop(model_id, None)
        self._unregister_model_endpoint(model_info)
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
        parsed = try_parse_pydantic(VllmCustomModel, model.data)

        if not self.models.get(instance):
            self.models[instance] = {}

        if parsed.id in self.models[instance]:
            raise HTTPException(400, "Model with given id already exists.")

        self.models[instance][parsed.id] = VllmModel(
            hf_id=parsed.hf_id,
            revision=parsed.revision,
            size=parsed.size,
            custom=model.id,
            model_type=parsed.model_type,
            adapter_id=parsed.adapter_id,
            lora_rank=parsed.lora_rank,
            quantization=parsed.quantization,
        )

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

    async def _download_model(self, stream: Stream[StreamChunk], model: VllmModel, model_dir: Path) -> Path | None:
        local_model_path: Path | None = None
        progress = Progress(convert_size_to_bytes(model.size) or 0)
        stream.emit(StreamChunkProgress(type="progress", stage="download", value=0, data={}))
        async for packet in self.model_downloader.download(model.hf_id, model_dir, revision=model.revision):
            if isinstance(packet, DownloadedPacket) and packet.downloaded_bytes_size != 0:
                progress.add_to_actual_value(packet.downloaded_bytes_size)
                stream.emit(
                    StreamChunkProgress(type="progress", stage="download", value=progress.get_percentage(), data=progress.to_data())
                )
            elif isinstance(packet, PreDownloadPacket):
                if max := packet.file_bytes_size:
                    progress.set_max_value(max)
            elif isinstance(packet, SuccessDownloadPacket):
                local_model_path = packet.local_path

        stream.emit(
            StreamChunkProgress(
                type="progress", stage="download", value=1, data={"local_model_path": str(local_model_path)} if local_model_path else {}
            )
        )
        if local_model_path is None:
            raise HTTPException(500, f"Downloader doesn't return filepath in {model_dir} of {model.hf_id} model")
        return local_model_path

    def _purge_stale_download(self, model_id: str, local_model_path: Path, local_adapter_path: Path | None = None) -> None:
        """Delete the previous download/adapter directories for `model_id` if this install now points at different ones.

        An edit that changes hf_id/revision (or adapter_id) reinstalls the model into new directories but
        leaves the old ones on disk with nothing referencing them anymore (`models_downloaded[model_id]` is
        about to be overwritten) - clean them up here rather than leaking them forever.
        """
        old_downloaded = self.models_downloaded.get(model_id)
        if not old_downloaded:
            return
        if old_downloaded.model_path and old_downloaded.model_path != str(local_model_path):
            self._remove_download_dir(old_downloaded.model_path, model_id, ignore_errors=True)
        new_adapter_path = str(local_adapter_path) if local_adapter_path else None
        if old_downloaded.adapter_path and old_downloaded.adapter_path != new_adapter_path:
            self._remove_download_dir(old_downloaded.adapter_path, model_id, ignore_errors=True)

    def _is_path_used_by_other_model(self, path: str, model_id: str) -> bool:
        """Whether any model other than `model_id` has `path` as its downloaded base-model or adapter directory.

        Download directories are keyed by hf_id/revision and adapter_id, not by model id, so several
        models can share one - e.g. two adapters trained on the same base model share its weights.
        """
        return any(
            path in (downloaded.model_path, downloaded.adapter_path)
            for other_id, downloaded in self.models_downloaded.items()
            if other_id != model_id
        )

    def _remove_download_dir(self, path: str, model_id: str, ignore_errors: bool = False) -> None:
        """Delete a download directory of `model_id`, unless another model still uses it."""
        if not self._is_path_used_by_other_model(path, model_id):
            shutil.rmtree(Path(path), ignore_errors=ignore_errors)

    async def _download_adapter(self, model: VllmModel, adapter_id: str, models_dir: Path) -> tuple[Path, Path]:
        """Download a LoRA adapter's safetensors files from the df-finetune registry.

        Returns (local_dir, docker_dir). Fetches the two files vLLM's static `--lora-modules`
        loader reads (`adapter_config.json`, `adapter_model.safetensors`) directly via a freshly
        built `AdapterRegistryDownloader`, from the finetune container's current base URL.

        An adapter id normally keeps pointing at the same weights, but df-finetune can rewrite them
        (e.g. a job retrained under the same id, or `run.py` writing into an existing adapter
        directory). So the registry's `created` timestamp is stored next to the downloaded files, and
        when it no longer matches, all files are downloaded again. `model.lora_rank` is refreshed
        from the registry too, so `--max-lora-rank` always matches the served weights.

        If both files are already on disk and the registry can't be reached, the on-disk copy is used
        as-is - services load concurrently at startup, so a reload of an adapter model can run before
        the Finetune service has registered its container, and requiring it here would fail a reload
        that needs nothing from it.
        """
        adapter_dir = models_dir / f"adapter-{normalize_name(adapter_id)}"
        docker_adapter_path = Path(self.hugging_face_cache_path) / "loras" / normalize_name(adapter_id)
        marker = adapter_dir / _ADAPTER_CREATED_MARKER
        missing = [filename for filename in _ADAPTER_FILES if not (adapter_dir / filename).exists()]
        try:
            metadata = await self._fetch_adapter_metadata(adapter_id)
        except HTTPException as exc:
            if missing:
                raise
            logger.warning("Adapter registry unavailable (%s), using on-disk copy of adapter %r", exc.detail, adapter_id)
            return adapter_dir, docker_adapter_path
        if metadata.base_model != model.hf_id:
            msg = f"Adapter {adapter_id!r} is now trained on {metadata.base_model!r}, not {model.hf_id!r}. Add it as a new model instead."
            raise HTTPException(400, msg)
        cached_created = marker.read_text().strip() if marker.exists() else None
        is_current = metadata.created is None or cached_created == str(metadata.created)
        to_fetch = missing if is_current else list(_ADAPTER_FILES)
        if to_fetch:
            if not is_current and not missing:
                logger.info("Adapter %r changed in the registry, downloading it again", adapter_id)
            marker.unlink(missing_ok=True)
            adapter_dir.mkdir(parents=True, exist_ok=True)
            base_url = self._get_adapter_registry_base_url()
            adapter_base_url = f"{base_url.rstrip('/')}/v1/adapters/{Utils.str_encode(adapter_id)}/safetensors"
            downloader = AdapterRegistryDownloader(self.config.adapter_registry_secret.get_secret_value())
            temp_dir = self.model_downloader.temp_dir
            for filename in to_fetch:
                # The downloader skips files already on disk, so a stale copy must go first
                (adapter_dir / filename).unlink(missing_ok=True)
                async for _packet in downloader.download(f"{adapter_base_url}/{filename}", adapter_dir, temp_dir, filename):
                    pass
        if metadata.created is not None:
            marker.write_text(str(metadata.created))
        model.lora_rank = metadata.rank
        return adapter_dir, docker_adapter_path

    async def _download_model_or_set_progress(
        self, stream: Stream[StreamChunk], download_key: str, model: VllmModel, model_dir: Path
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
        """Size the default utilization to the fraction of VRAM currently free instead of a flat value.

        A flat default can request more memory than is actually free (e.g. when other processes already
        hold some VRAM), which makes vLLM fail to start. Sizing the default off live free memory avoids that.
        """
        stats = await self.hardware.get_realtime_stats(force_refresh=True)
        if not stats or stats.total_vram_gb <= 0:
            logger.warning(
                f"vLLM: GPU stats unavailable, assuming {self._default_gpu_memory_utilization} of total VRAM is free; "
                "a gpu_priority share is applied to that and may exceed the memory actually free"
            )
            return self._default_gpu_memory_utilization
        free_fraction = max(0.0, stats.total_vram_gb - stats.used_vram_gb) / stats.total_vram_gb
        return round(min(self._default_gpu_memory_utilization, free_fraction * self._gpu_memory_free_safety_margin), 2)

    async def _get_gpu_memory_utilization(
        self, instance: str, model_id: str, parsed_model_options: VllmModelOptions, model: VllmModel, *, reserve: bool = True
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
                    "GPU utilization in all vllm instances can be maximum 1. "
                    f"Actual GPU utilization is {self.gpu_memory_utilization}. "
                    f"You try add additional {gpu_memory_utilization}, "
                    f"which is {self.gpu_memory_utilization + gpu_memory_utilization - 1} more that 1."
                ),
            )
        if not reserve:
            return gpu_memory_utilization
        self.gpu_memory_utilization += gpu_memory_utilization
        msg = f"VLLM gpu utilization = {self.gpu_memory_utilization}"
        logger.debug(msg)
        return gpu_memory_utilization

    def _release_gpu_utilization(self, utilization: float | None) -> None:
        if utilization:
            self.gpu_memory_utilization -= utilization
            if self.gpu_memory_utilization < 0:
                self.gpu_memory_utilization = 0
            msg = f"VLLM gpu utilization = {self.gpu_memory_utilization}"
            logger.debug(msg)

    async def _get_quantization(self, parsed_model_options: VllmModelOptions, model: VllmModel) -> RegistrationId | None:
        quantization = parsed_model_options.quantization or model.quantization or None
        if quantization:
            pattern = r"^[a-zA-Z0-9_-]+$"
            if not re.match(pattern, quantization):
                raise HTTPException(422, "Quantization options needs to be made from `a-zA-Z-_`.")
        return quantization

    def _build_vllm_command(
        self,
        docker_model_path: Path,
        model_id: str,
        opts: VllmModelOptions,
        quantization: str | None,
        gpu_memory_utilization: float | None,
        user_model_length: int | None,
        use_gpu: bool,
        docker_adapter_path: Path | None = None,
        lora_rank: int | None = None,
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

        if docker_adapter_path is not None:
            cmd.extend(["--enable-lora", "--lora-modules", f"{model_id}={docker_adapter_path}"])
            if lora_rank is not None:
                cmd.extend(["--max-lora-rank", str(_round_up_lora_rank(lora_rank))])

        for flag, value in opts.extra_args.items():
            if value:
                cmd.extend([flag, value])
            else:
                cmd.append(flag)

        return cmd

    def _unregister_model_endpoint(self, model_info: ModelInstalledInfo) -> None:
        """Remove the model's registration from the endpoint registry based on its type."""
        name, registration_id = model_info.registered_name, model_info.registration_id
        if model_info.model_type == "reranker":
            self.endpoint_registry.unregister_rerank(name, registration_id)
        elif model_info.model_type == "embedding":
            self.endpoint_registry.unregister_embeddings(name, registration_id)
        elif model_info.model_type == "stt":
            self.endpoint_registry.unregister_audio_transcriptions(name, registration_id)
        else:
            self.endpoint_registry.unregister_chat_completion(name, registration_id)

    def _register_model_endpoint(
        self,
        model_info: "ModelInstalledInfo",
        model: VllmModel,
        registered_name: str,
        model_id: str,
        context_window: int | None,
        max_context_window: int | None,
        capacity: int | None,
        registered_alias: str | None = None,
        tokenizer_dir: Path | None = None,
    ) -> RegistrationId:
        """Register the model in the endpoint registry based on its type; embeddings serve their tokenizer from `tokenizer_dir`."""
        # vLLM always has a real, finite concurrency limit; `capacity is None` here means we
        # failed to determine it (docker-log fetch or parse failure), never that it's unbounded.
        capacity_state: CapacityState = capacity if capacity is not None else "unknown"
        if model.model_type == "reranker":
            return self.endpoint_registry.register_rerank_as_proxy(
                model=registered_name,
                props=ModelProps(private=True, type="rerank", endpoints=["/v1/rerank"]),
                options=ProxyOptions(url=f"{model_info.base_url}/v1/rerank", rewrite_model_to=model_id),
                registration_options=RegistrationOptions(
                    origin="local", alias=registered_alias, owned_by=self.get_type(), capacity_state=capacity_state
                ),
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
        if model.model_type == "stt":
            return self.endpoint_registry.register_audio_transcriptions_as_proxy(
                model=registered_name,
                props=ModelProps(private=True, type="stt", endpoints=STT_ENDPOINTS),
                options=ProxyOptions(url=f"{model_info.base_url}/v1/audio/transcriptions", rewrite_model_to=model_id),
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

    async def _start_vllm_container_with_retry(
        self,
        docker_options: DockerOptions,
        model_id: str,
        docker_model_path: Path,
        parsed_model_options: VllmModelOptions,
        quantization: str | None,
        gpu_memory_utilization: float | None,
        user_model_length: int | None,
        use_gpu: bool,
        docker_adapter_path: Path | None = None,
        lora_rank: int | None = None,
    ) -> tuple[int, int | None, bool, bool]:
        """Start the vLLM container, tearing it down on any failure.

        If startup fails because the model's default context length needs more KV cache than is
        free, retries once with `--max-model-len` lowered to vLLM's own suggestion. Any other
        failure (including a retry that still doesn't fit) is translated into a clear "not enough
        GPU VRAM" or "missing model files" AppError when recognized, instead of the raw docker/vLLM
        traceback.
        Returns (docker_exposed_port, effective user_model_length, whether the container was (re)started, whether it was adopted).
        """
        try:
            docker_exposed_port, restarted, adopted = await self.docker_service.install_and_run_docker(docker_options)
        except Exception as exc:
            await self.docker_service.stop_docker(docker_options)
            if missing_files_msg := self._diagnose_missing_model_files(model_id, str(exc)):
                raise AppError(missing_files_msg) from exc
            suggested_max_length = self._parse_kv_cache_max_len_suggestion(str(exc)) if use_gpu and user_model_length is None else None
            if suggested_max_length is None:
                if use_gpu and (vram_msg := self._diagnose_insufficient_vram(model_id, str(exc))):
                    raise AppError(vram_msg) from exc
                raise
            logger.warning(
                "vLLM rejected the model's default context length for %r (insufficient KV cache memory); retrying with --max-model-len %d.",
                model_id,
                suggested_max_length,
            )
            user_model_length = suggested_max_length
            vllm_command = self._build_vllm_command(
                docker_model_path=docker_model_path,
                model_id=model_id,
                opts=parsed_model_options,
                quantization=quantization,
                gpu_memory_utilization=gpu_memory_utilization,
                user_model_length=user_model_length,
                use_gpu=use_gpu,
                docker_adapter_path=docker_adapter_path,
                lora_rank=lora_rank,
            )
            docker_options.command = vllm_command
            try:
                docker_exposed_port, restarted, adopted = await self.docker_service.install_and_run_docker(docker_options)
            except Exception as retry_exc:
                await self.docker_service.stop_docker(docker_options)
                if missing_files_msg := self._diagnose_missing_model_files(model_id, str(retry_exc)):
                    raise AppError(missing_files_msg) from retry_exc
                if vram_msg := self._diagnose_insufficient_vram(model_id, str(retry_exc)):
                    raise AppError(vram_msg) from retry_exc
                raise
        return docker_exposed_port, user_model_length, restarted, adopted

    async def _resolve_context_windows(
        self, model: VllmModel, model_id: str, local_model_path: Path, base_url: str, user_model_length: int | None
    ) -> tuple[int | None, int | None]:
        """Return `(context_window, max_context_window)` to register a freshly started model with."""
        if model.model_type == "llm":
            max_context_window = await get_model_dir_context_window(local_model_path)
            return user_model_length or max_context_window, max_context_window
        if model.model_type == "embedding":
            limit = await fetch_served_max_model_len(base_url, model_id) or user_model_length
            return limit, limit
        return None, None

    async def _install_model(  # noqa: C901
        self, instance: str, model_id: str, options: InstallModelIn, container_name_override: str | None = None
    ) -> PromiseWithProgress[InstallModelOut, StreamChunk]:
        parsed_model_options = try_parse_pydantic(VllmModelOptions, options.spec) if options.spec else VllmModelOptions()

        info = self.get_instance_installed_info(instance)

        key = (instance, model_id)
        if model_id in info.models or key in self._installing:
            return PromiseWithProgress(value=InstallModelOut(status="OK", details="Already installed or being installed right now."))

        if model_id not in self.models[instance]:
            raise HTTPException(400, f"Model {model_id!r} not found")

        self._installing.add(key)
        model = self.models[instance][model_id]

        try:
            use_gpu = self.is_given_hardware_support_gpu(info.parsed_options.hardware)
            if use_gpu:
                # Fail fast (before a long download) on options that can never fit; the share is really taken in `func`.
                await self._get_gpu_memory_utilization(instance, model_id, parsed_model_options, model, reserve=False)
            user_model_length = parsed_model_options.max_model_length or model.max_model_len or None
            quantization = await self._get_quantization(parsed_model_options, model)
        except BaseException:
            self._installing.discard(key)
            raise

        async def func(stream: Stream[StreamChunk]) -> InstallModelOut:
            nonlocal user_model_length
            model_info: ModelInstalledInfo | None = None
            docker_options: DockerOptions | None = None
            docker_stopped = False
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

                local_adapter_path: Path | None = None
                docker_adapter_path: Path | None = None
                if model.adapter_id:
                    local_adapter_path, docker_adapter_path = await self._download_adapter(model, model.adapter_id, models_dir)

                stream.emit(StreamChunkProgress(type="progress", stage="install", value=0, data={}))
                docker_model_path = Path(self.hugging_face_cache_path) / "hub" / model_id_fixed
                compile_cache_dir = model_dir / "compile_cache"
                compile_cache_dir.mkdir(parents=True, exist_ok=True)
                model_cache_dir = model_dir / ".cache"
                model_cache_dir.mkdir(parents=True, exist_ok=True)
                volumes = [
                    f"{local_model_path}:{docker_model_path}",
                    f"{compile_cache_dir}:/root/.cache/vllm",
                    f"{model_cache_dir}:{docker_model_path}/.cache",
                ]
                if local_adapter_path is not None:
                    volumes.append(f"{local_adapter_path}:{docker_adapter_path}")

                # Sizing and the container start/health wait are serialized: once the previous container is healthy its
                # weights are in `used_vram_gb`, so the next model is sized from a measurement that already includes it.
                # Downloading stays outside the lock, so installs still download in parallel.
                async with self.hardware.gpu_load_lock if use_gpu else contextlib.nullcontext():
                    if use_gpu:
                        gpu_memory_utilization = await self._get_gpu_memory_utilization(instance, model_id, parsed_model_options, model)
                    vllm_command = self._build_vllm_command(
                        docker_model_path=docker_model_path,
                        model_id=model_id,
                        opts=parsed_model_options,
                        quantization=quantization,
                        gpu_memory_utilization=gpu_memory_utilization,
                        user_model_length=user_model_length,
                        use_gpu=use_gpu,
                        docker_adapter_path=docker_adapter_path,
                        lora_rank=model.lora_rank,
                    )

                    env_vars = dict(model.env_vars or {})
                    env_vars["HF_HUB_OFFLINE"] = "1"
                    env_vars["HF_HOME"] = self.hugging_face_cache_path
                    if not use_gpu:
                        env_vars["VLLM_USE_V1"] = "1"
                    env_vars.update(parsed_model_options.extra_envs)

                    image = self._get_image(use_gpu, info.options.spec.get("image_version") if info.options else None)

                    subnet = self.docker_service.get_docker_subnet()
                    service_name = f"{self.get_service_id(instance)}-{normalize_name(model_id)}"
                    docker_options = DockerOptions(
                        name=service_name,
                        container_name=container_name_override or self.docker_service.get_docker_container_name(service_name),
                        image=image.name,
                        command=vllm_command,
                        image_port=8000,
                        env_vars=env_vars,
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
                        docker_exposed_port, user_model_length, restarted, adopted = await self._start_vllm_container_with_retry(
                            docker_options=docker_options,
                            model_id=model_id,
                            docker_model_path=docker_model_path,
                            parsed_model_options=parsed_model_options,
                            quantization=quantization,
                            gpu_memory_utilization=gpu_memory_utilization,
                            user_model_length=user_model_length,
                            use_gpu=use_gpu,
                            docker_adapter_path=docker_adapter_path,
                            lora_rank=model.lora_rank,
                        )
                    except Exception:
                        docker_stopped = True
                        raise
                capacity, capacity_known = await self._resolve_model_capacity(
                    instance, model_id, "vLLM", docker_options.container_name or "", restarted, self._parse_max_concurrency
                )
                registered_name = parsed_model_options.rename if parsed_model_options.rename else model_id
                registered_alias = parsed_model_options.alias
                container_host = self.docker_service.get_container_host(subnet, docker_options)
                container_port = self.docker_service.get_container_port(subnet, docker_exposed_port, docker_options.image_port)
                context_window, max_context_window = await self._resolve_context_windows(
                    model, model_id, local_model_path, get_base_url(container_host, container_port), user_model_length
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
                self._purge_stale_download(model_id, local_model_path, local_adapter_path)
                self.models_downloaded[model_id] = DownloadedInfo(
                    str(local_model_path), str(local_adapter_path) if local_adapter_path else None
                )
                return InstallModelOut(status="OK", details="Installed")
            except BaseException:
                self._release_gpu_utilization(gpu_memory_utilization)
                if model_info is not None and info.models.get(model_id) is model_info:
                    info.models.pop(model_id, None)
                if docker_options is not None and not docker_stopped:
                    await self._rollback_failed_install_docker(docker_options, adopted=adopted)
                raise
            finally:
                self._installing.discard(key)

        return PromiseWithProgress(func=func)

    def _get_image(self, gpu: bool, image_version: str | None = None) -> DockerImage:
        base = _const.images["gpu"] if gpu else _const.images["cpu"]
        if image_version:
            base_name = base.name.split(":")[0]
            return DockerImage(name=f"{base_name}:{image_version}", size=base.size)
        return base

    def _use_gpu_image(self, hardware: str | None) -> bool:
        """Return whether the given hardware string selects the GPU image variant."""
        return hardware is None or not str(hardware).lower().startswith("cpu")

    def get_docker_image_repo(self, hardware: str | None) -> str:
        """Return the Docker repo tags are fetched from for the given hardware (GPU and CPU use different repos)."""
        base = _const.images["gpu"] if self._use_gpu_image(hardware) else _const.images["cpu"]
        return base.name.split(":")[0]

    async def get_docker_tags(self, hardware: str | None) -> list[str]:
        """Fetch available Docker image tags for the vLLM image matching the selected hardware."""
        base_image = self.get_docker_image_repo(hardware)
        client = registry_for(base_image, self.config.docker_hub_token or None)
        tags = await client.get_tags(image_without_registry_prefix(base_image))
        return self.filter_docker_tags(tags, hardware)

    def filter_docker_tags(self, tags: list[str], hardware: str | None) -> list[str]:
        """Filter tags to GPU variants (-cu* and plain release tags), or for CPU to the tags usable on this host's architecture."""
        if hardware is None:
            return tags
        if self._use_gpu_image(hardware):
            return [t for t in tags if _is_gpu_tag(t)]
        other_arch_suffix = "-x86_64" if self._is_arm_host() else "-arm64"
        return [t for t in tags if not t.endswith(other_arch_suffix)]

    def get_default_docker_tag(self, hardware: str | None) -> str | None:
        """Return the pinned default vLLM image tag for the given hardware variant."""
        base = _const.images["gpu"] if self._use_gpu_image(hardware) else _const.images["cpu"]
        return base.name.split(":")[1]

    async def _uninstall_model(self, instance: str, model_id: str, options: UninstallModelIn) -> None:
        info = self.get_instance_installed_info(instance)

        if model_id in info.models:
            model = info.models[model_id]
            self._release_gpu_utilization(model.gpu_memory_utilization)
            del info.models[model_id]
            self._unregister_model_endpoint(model)
            await self.docker_service.uninstall_docker(model.docker)

        if options.purge and model_id in self.models_downloaded:
            model = self.models_downloaded[model_id]
            if model_path := model.model_path:
                self._remove_download_dir(model_path, model_id)
            if adapter_path := model.adapter_path:
                self._remove_download_dir(adapter_path, model_id, ignore_errors=True)
            del self.models_downloaded[model_id]

    @staticmethod
    def _parse_max_concurrency(raw: str) -> int | None:
        """Parse vLLM's dynamically-computed max concurrency from its KV-cache startup log line.

        vLLM logs e.g. "Maximum concurrency for 4096 tokens per request: 12.34x" once it has sized
        the KV cache from real free VRAM; the trailing multiplier is the number of concurrent
        max-length requests the instance can serve. On (re)start it may log this line more than
        once (e.g. across a KV-cache-too-small retry), so the *last* occurrence reflects the
        config the instance actually launched with. A failed parse (e.g. the log format changed
        across a vLLM version) falls back to `None` rather than failing install; `None` here means
        "failed to determine," never "unbounded" - callers must treat it as unknown capacity.
        """
        matches = list(re.finditer(r"[Mm]aximum concurrency for [\d,]+ tokens per request:\s*([\d.]+)x", raw))
        if not matches:
            return None
        captured = matches[-1].group(1)
        try:
            raw_concurrency = float(captured)
        except (ValueError, OverflowError):
            logger.warning("Failed to parse vLLM's reported max concurrency %r as a number; capacity will be unknown.", captured)
            return None
        if raw_concurrency < 1:
            logger.warning(
                "vLLM reported maximum concurrency of %.2fx, which is below 1; treating capacity as 1 instead of unknown.",
                raw_concurrency,
            )
        try:
            return max(1, round(raw_concurrency))
        except OverflowError:
            logger.warning("vLLM's reported max concurrency %.2fx overflowed rounding; capacity will be unknown.", raw_concurrency)
            return None

    @staticmethod
    def _parse_kv_cache_max_len_suggestion(raw: str) -> int | None:
        """Parse vLLM's suggested max model length from a KV-cache-too-small startup failure.

        When a model's default context length needs more KV cache than is actually free, vLLM aborts
        startup but reports the largest length that would fit (e.g. "the estimated maximum model length
        is 110256"). Reusing that value lets install retry once instead of failing outright.
        """
        if m := re.search(r"estimated maximum model length is (\d+)", raw):
            return int(m.group(1))
        return None

    _VRAM_INSUFFICIENT_PATTERNS = (
        "larger than the available kv cache memory",
        "no available memory for the cache blocks",
        "cuda out of memory",
        "cuda error: out of memory",
    )

    @classmethod
    def _diagnose_insufficient_vram(cls, model_id: str, raw: str) -> str | None:
        """Return a user-facing message if *raw* indicates the GPU doesn't have enough free VRAM for the model."""
        lower = raw.lower()
        if any(p in lower for p in cls._VRAM_INSUFFICIENT_PATTERNS):
            return (
                f"Not enough GPU VRAM to run {model_id!r}. Try a smaller or quantized model, reduce the "
                "max context length, or free VRAM by stopping other running models."
            )
        return None

    _MISSING_FILES_PATTERNS = (
        "localentrynotfounderror",
        "outgoing traffic has been disabled",
        "couldn't connect to 'https://huggingface.co'",
    )

    @classmethod
    def _diagnose_missing_model_files(cls, model_id: str, raw: str) -> str | None:
        """Return a user-facing message if *raw* indicates vLLM couldn't find required files in the local download.

        The container always runs with `HF_HUB_OFFLINE=1` (only the pre-downloaded model directory is mounted in),
        so this happens when the download is incomplete or missing files vLLM expects, not a real network outage.
        """
        lower = raw.lower()
        if any(p in lower for p in cls._MISSING_FILES_PATTERNS):
            return (
                f"Model {model_id!r} is missing files it needs to start (the download may be incomplete or "
                "the repository layout isn't fully supported). Try prune and install model again"
            )
        return None

    @staticmethod
    def _parse_vllm_vram_gb(raw: str) -> float | None:
        """Parse vLLM docker logs and return total estimated VRAM in GiB.

        Tries the profiling summary first, then falls back to summing weights + KV cache.
        """
        total_re = re.compile(r"Total \(incl\. non-KV cache overhead\):\s*([\d.]+)\s*GiB", re.IGNORECASE)
        if m := total_re.search(raw):
            return float(m.group(1))

        weights_re = re.compile(r"[Mm]odel weights(?: take)?\s*[:\s]*([\d.]+)\s*GiB")
        kv_re = re.compile(r"KV [Cc]ache(?:\s+memory)?:\s*([\d.]+)\s*GiB")
        total = 0.0
        found = False
        if m := weights_re.search(raw):
            total += float(m.group(1))
            found = True
        if m := kv_re.search(raw):
            total += float(m.group(1))
            found = True
        return round(total, 2) if found else None

    async def _get_vram_from_logs(self, instance: str, model_id: str) -> float | None:
        """Return actual GPU VRAM usage in GiB for an installed vLLM model by parsing its container logs."""
        info = self.get_instance_installed_info(instance)
        if (model_info := info.models.get(model_id)) and (container_name := model_info.docker.container_name or ""):
            try:
                raw = await self._get_docker_logs(container_name)
                return self._parse_vllm_vram_gb(raw)
            except Exception:
                logger.debug("Could not determine VRAM usage from logs for %s", container_name)
        return None

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
        vram_estimate_gb = await self._get_vram_from_logs(instance_name, model_id)
        if vram_estimate_gb is None:
            info = self.get_instance_installed_info(instance_name)
            if model_info := info.models.get(model_id):
                vram_estimate_gb = self._get_vram_estimate(model_info)
        if vram_estimate_gb is not None:
            self._vram_cache[cache_key] = vram_estimate_gb
        return vram_estimate_gb

    async def stop_instance(self, instance: str) -> None:
        """Stop all the vLLM service Docker containers."""
        self._stop_reconciliation_task(instance)
        installed = self.get_instance_info(instance).installed
        if not installed:
            return
        await self._stop_dockers_parallel([model.docker for model in installed.models.values()])
