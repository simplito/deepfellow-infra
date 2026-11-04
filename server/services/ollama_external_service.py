# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Ollama external service."""

import asyncio
import functools
import json
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from fastapi import HTTPException
from packaging import version
from pydantic import BaseModel, Field, StringConstraints

from server.config import get_main_dir
from server.endpointregistry import ProxyOptions, RegistrationId, RegistrationOptions
from server.models.api import DECISION_ENDPOINTS, EMBEDDINGS_ENDPOINTS, ModelProps
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
from server.utils.logger import logger
from server.utils.ollama import decision_models_unsupported_message, raise_ollama_pull_error, supports_decision_models
from server.utils.ollama_embedding_limit import OllamaEmbeddingTokenLimits
from server.utils.ollama_tokenizer import OllamaTokenizers
from server.utils.size_fetcher import fetch_ollama_ref_bytes, fmt_size


class OllamaModel(BaseModel):
    id: str
    size: str
    type: str
    custom: CustomModelId | None = None


class OllamaCustomModel(BaseModel):
    id: str
    size: str
    type: Literal["llm", "embedding", "decision"]


class OllamaRegistryEntry(BaseModel):
    name: str
    size: str


class OllamaRegistry(BaseModel):
    llms: list[OllamaRegistryEntry]
    embeddings: list[OllamaRegistryEntry]
    decision: list[OllamaRegistryEntry] = []


def _read_models() -> dict[str, OllamaModel]:
    ollama_path = get_main_dir() / "./static/ollama-min.json"
    with ollama_path.open(encoding="utf-8") as f:
        registry_data = json.loads(f.read())
        registry = OllamaRegistry(
            llms=registry_data["llms"], embeddings=registry_data["embeddings"], decision=registry_data.get("decision", [])
        )
        map: dict[str, OllamaModel] = {}
        for tag in registry.llms:
            map[tag.name] = OllamaModel(id=tag.name, size=tag.size, type="llm")
        for tag in registry.embeddings:
            map[tag.name] = OllamaModel(id=tag.name, size=tag.size, type="embedding")
        for tag in registry.decision:
            map[tag.name] = OllamaModel(id=tag.name, size=tag.size, type="decision")
        return map


class OllamaAiConst(BaseModel):
    models: dict[str, OllamaModel]


_const = OllamaAiConst(
    models=_read_models(),
)

_WARMTH_POLL_INTERVAL_SECONDS = 15  # matches the staleness already tolerated for the UI's loaded-model display
_EMBEDDING_LIMIT_MAX_AGE_SECONDS = 24 * 3600


@dataclass
class ModelInstalledInfo:
    id: str
    registered_name: str
    type: str
    options: InstallModelIn
    registration_id: RegistrationId
    supports_vision: bool | None = None
    supports_audio: bool | None = None

    def get_info(self) -> ModelInfo:
        """Get info."""
        return ModelInfo(spec=self.options.spec, registration_id=self.registration_id)


class OllamaExternalOptions(BaseModel):
    url: str = "http://localhost:11434"
    sync_interval: int = Field(default=60, ge=10)


class OllamaModelOptions(BaseModel):
    rename: str | None = None
    alias: str | None = None
    alive_time: int | Annotated[str, StringConstraints(pattern=r"^(\d+[smh])?$", strict=True)] = ""


@dataclass
class InstalledInfo:
    models: dict[str, ModelInstalledInfo]
    options: InstallServiceIn
    parsed_options: OllamaExternalOptions
    base_url: str


@dataclass
class DownloadedInfo:
    pass


class OllamaExternalService(Base2Service[InstalledInfo, DownloadedInfo]):
    models: dict[str, dict[str, OllamaModel]]
    support_responses: bool
    support_messages: bool
    support_systemone: bool
    ollama_version: str
    _installing: set[tuple[str, str]]
    _warmth_poll_tasks: dict[str, asyncio.Task[None]]
    _embedding_token_limits: OllamaEmbeddingTokenLimits
    _tokenizers: OllamaTokenizers
    _persists_model_definitions = False

    def _after_init(self) -> None:
        self._sync_tasks: dict[str, asyncio.Task[None]] = {}
        self.load_default_models("default")
        self._installing = set()
        self._warmth_poll_tasks = {}
        self._embedding_token_limits = OllamaEmbeddingTokenLimits(
            self.config.get_storage_services_dir() / self.get_type() / "embedding_token_limits.json",
            max_age_seconds=_EMBEDDING_LIMIT_MAX_AGE_SECONDS,
        )
        self._tokenizers = OllamaTokenizers()

    def load_default_models(self, instance: str) -> None:
        """Load default models to instance."""
        self.models[instance] = _const.models.copy()

    def get_type(self) -> str:
        """Return the service id."""
        return "ollama-external"

    def get_description(self) -> str:
        """Return the service description."""
        return "Connect to an external Ollama instance."

    def get_size(self) -> str:
        """Return the service size."""
        return ""

    def get_spec(self) -> ServiceSpecification:
        """Return the service specification."""
        return ServiceSpecification(
            fields=[
                ServiceField(type="text", name="url", description="URL to external Ollama instance", default="http://localhost:11434"),
                ServiceField(
                    type="number", name="sync_interval", description="Background sync interval in seconds", default="60", required=False
                ),
            ]
        )

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
            ]
        )

    def get_custom_model_spec(self) -> CustomModelSpecification | None:
        """Return the custom model specification or None if custom model is not supported."""
        return CustomModelSpecification(
            fields=[
                CustomModelField(type="text", name="id", description="Model ID", placeholder="my-custom-model"),
                CustomModelField(type="oneof", name="type", description="Model type", values=["llm", "embedding", "decision"]),
                CustomModelField(type="text", name="size", description="Model size", placeholder="1 GB", required=False),
            ]
        )

    async def _resolve_custom_model_size(self, spec: dict[str, Any], instance: str = "") -> str | None:  # noqa: ARG002
        try:
            model_id: str = spec.get("id", "")
            n = await fetch_ollama_ref_bytes(model_id)
            return fmt_size(n) if n is not None else None
        except Exception:
            return None

    def get_installed_info(self, instance: str) -> bool | InstallServiceProgress | ServiceOptions:
        """Get service installed info."""
        installed = self.get_instance_info(instance).installed
        return self._get_service_installed_info(instance) if installed is None else installed.options.spec

    def _generate_instance_config(self, instance: str, info: InstalledInfo | None, custom: list[CustomModel] | None) -> InstanceConfig:  # noqa: ARG002
        return InstanceConfig(
            options=info.options if info else None,
            models=[ModelConfig(model_id=x.id, options=x.options) for x in info.models.values()] if info else [],
            custom=custom,
        )

    async def load_instance(self, instance: str, instance_data: InstanceConfig) -> None:
        """Load instance, skipping models no longer present in external Ollama."""
        if not instance_data.options:
            return

        self.load_default_models(instance)

        if instance_data.custom:
            for custom in instance_data.custom:
                self._add_custom_model(instance, custom)

        promise = await self.install_instance(instance, instance_data.options, instance_data, save=False)
        await promise.wait()
        await self._save()

        instance_info = self.instances_info[instance]
        installed = instance_info.installed
        if installed:
            instance_info.config.models = [ModelConfig(model_id=mid, options=installed.models[mid].options) for mid in installed.models]

    def service_has_docker(self) -> bool:
        """Return true when docker is started when service is installed."""
        return False

    async def stop_instance(self, instance: str) -> None:
        """Stop the service gracefully.

        External Ollama service has no containers to stop.
        """
        self._stop_warmth_poll_task(instance)

    def _add_custom_model(self, instance: str, model: CustomModel) -> None:
        parsed = try_parse_pydantic(OllamaCustomModel, model.data)

        if not self.models.get(instance):
            self.models[instance] = {}

        if parsed.id in self.models[instance]:
            raise HTTPException(400, "Model with given id already exists.")

        self.models[instance][parsed.id] = OllamaModel(id=parsed.id, size=parsed.size, type=parsed.type, custom=model.id)

    def _remove_custom_model(self, instance: str, model: CustomModel) -> None:
        installed = self.get_instance_info(instance).installed
        parsed = try_parse_pydantic(OllamaCustomModel, model.data)
        if installed and parsed.id in installed.models:
            raise HTTPException(400, "Cannot remove custom model, it is in use, uninstall it first.")
        del self.models[instance][parsed.id]

    def _load_download_info(self, data: dict[str, Any]) -> DownloadedInfo:
        return DownloadedInfo(**data)

    def _determine_model_type(self, model_name: str, instance: str, supports_decision: bool | None = None) -> str:
        known = self.models.get(instance, {}).get(model_name)
        if known is not None and known.custom is not None:
            return known.type
        if supports_decision:
            return "decision"
        if known is not None:
            return known.type
        if model_name in _const.models:
            return _const.models[model_name].type
        return "llm"

    async def _install_instance(
        self,
        instance: str,
        options: InstallServiceIn,
        container_name_override: str | None = None,  # noqa: ARG002
    ) -> PromiseWithProgress[InstalledInfo, StreamChunk]:
        if not self.models.get(instance):
            self.load_default_models(instance)

        parsed_options = try_parse_pydantic(OllamaExternalOptions, options.spec)

        def _raise_connection_error(url: str) -> None:
            msg = f"Cannot connect to Ollama at {url}"
            raise HTTPException(status_code=400, detail=msg)

        async def func(stream: Stream[StreamChunk]) -> InstalledInfo:
            # Verify connection to external Ollama
            stream.emit(StreamChunkProgress(type="progress", stage="install", value=0, data={}))
            try:
                res = await fetch_from(f"{parsed_options.url.rstrip('/')}/api/version", "GET", None)
                if res.status_code != 200:
                    _raise_connection_error(parsed_options.url)
                try:
                    data_json = json.loads(res.data)
                    actual_version: str = data_json["version"]
                    self.support_responses = bool(version.parse(actual_version) > version.parse("0.14.1"))
                    self.support_messages = bool(version.parse(actual_version) >= version.parse("0.14.0"))
                    self.support_systemone = supports_decision_models(actual_version) is True
                    self.ollama_version = actual_version
                except json.JSONDecodeError as err:
                    raise HTTPException(500, "Ollama external server doesn't return version.") from err
            except HTTPException:
                raise
            except Exception as e:
                msg = f"Cannot connect to Ollama at {parsed_options.url}: {e!s}"
                raise HTTPException(status_code=400, detail=msg) from e

            stream.emit(StreamChunkProgress(type="progress", stage="install", value=1, data={}))
            installed_info = InstalledInfo(
                models={}, options=options, parsed_options=parsed_options, base_url=parsed_options.url.rstrip("/")
            )
            await self._sync_models_from_external_ollama(instance, installed_info, is_initial_sync=True)
            self._start_sync_task(instance, parsed_options.sync_interval)
            self._start_warmth_poll_task(instance)
            return installed_info

        return PromiseWithProgress(func=func)

    def _start_sync_task(self, instance: str, sync_interval: int) -> None:
        if instance in self._sync_tasks:
            self._sync_tasks[instance].cancel()

        async def _sync_loop() -> None:
            while True:
                await asyncio.sleep(sync_interval)
                info = self.instances_info.get(instance)
                if not info or not info.installed:
                    break
                try:
                    await self._sync_models_from_external_ollama(instance, info.installed)
                    await self._refresh_embedding_token_limits(instance, info.installed)
                    await self._save()
                except Exception:
                    logger.exception(f"{self.get_id(instance)} model sync tick failed")

        self._sync_tasks[instance] = asyncio.create_task(_sync_loop())

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
                    await self._sync_warm_state(info.installed)
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

    async def _sync_warm_state(self, info: InstalledInfo) -> None:
        """Reflect the external Ollama's live loaded/evicted model state into the endpoint registry's warm flag."""
        loaded = await self._get_loaded_model_names(info.base_url)
        if loaded is None:
            return
        for model_info in info.models.values():
            if not model_info.registration_id:
                continue
            self.endpoint_registry.update_warm(model_info.registration_id, model_info.id in loaded)

    async def _get_loaded_model_names(self, base_url: str) -> set[str] | None:
        """Return the set of model names currently loaded in VRAM, or None if the query failed.

        Returning None (rather than an empty set) lets callers tell "confirmed nothing loaded"
        apart from "failed to ask Ollama", so a transient failure doesn't get treated as an unload.
        """
        try:
            result = await fetch_from(f"{base_url}/api/ps", "GET", None)
            if result.status_code != 200:
                logger.warning("Ollama /api/ps at %r returned status %d", base_url, result.status_code)
                return None
            data = json.loads(result.data)
            return {m["name"].removesuffix(":latest") for m in data.get("models", [])}
        except Exception:
            logger.warning("Failed to query Ollama /api/ps at %r", base_url, exc_info=True)
            return None

    def _register_synced_model(
        self,
        instance: str,
        installed_info: InstalledInfo,
        model_id: str,
        size_bytes: int,
        context_length: int | None = None,
        supports_vision: bool | None = None,
        supports_audio: bool | None = None,
        supports_decision: bool | None = None,
    ) -> None:
        model_type = self._determine_model_type(model_id, instance, supports_decision)
        size_str = fmt_size(size_bytes) if size_bytes else ""

        ollama_model = OllamaModel(id=model_id, size=size_str, type=model_type)
        self.models[instance][model_id] = ollama_model
        self.models_downloaded[model_id] = DownloadedInfo()

        if model_type == "decision" and not self.support_systemone:
            logger.warning(
                "Skipping decision model %r at %r: %s",
                model_id,
                installed_info.base_url,
                decision_models_unsupported_message(self.ollama_version),
            )
            return

        model_info = ModelInstalledInfo(
            id=model_id,
            type=model_type,
            registered_name=model_id,
            options=InstallModelIn(),
            registration_id="",
            supports_vision=supports_vision,
            supports_audio=supports_audio,
        )
        if model_type == "llm":
            model_info.registration_id = self.endpoint_registry.register_chat_completion_as_proxy(
                model=model_id,
                props=ModelProps(
                    private=True,
                    type="llm",
                    context_window=context_length,
                    max_context_window=context_length,
                    supports_vision=supports_vision,
                    supports_audio=supports_audio,
                ),
                chat_completions=ProxyOptions(
                    url=f"{installed_info.base_url}/v1/chat/completions", rewrite_model_to=model_id, forward_extra_fields=True
                ),
                completions=ProxyOptions(
                    url=f"{installed_info.base_url}/v1/completions", rewrite_model_to=model_id, forward_extra_fields=True
                ),
                responses=(
                    ProxyOptions(url=f"{installed_info.base_url}/v1/responses", rewrite_model_to=model_id)
                    if self.support_responses
                    else None
                ),
                messages=(
                    ProxyOptions(url=f"{installed_info.base_url}/v1/messages", rewrite_model_to=model_id) if self.support_messages else None
                ),
                ollama_chat=ProxyOptions(url=f"{installed_info.base_url}/api/chat", rewrite_model_to=model_id),
                # Loaded state is unknown until the warmth-poll task (already running per-instance)
                # confirms it via `/api/ps` - assume cold rather than winning ranking on a guess.
                registration_options=RegistrationOptions(origin="local", owned_by=self.get_type(), warm=False),
            )
        if model_type == "embedding":
            model_info.registration_id = self.endpoint_registry.register_embeddings_as_proxy(
                model=model_id,
                props=ModelProps(
                    private=True,
                    type="embedding",
                    endpoints=EMBEDDINGS_ENDPOINTS,
                    context_window=context_length,
                    max_context_window=context_length,
                ),
                options=ProxyOptions(url=f"{installed_info.base_url}/v1/embeddings", rewrite_model_to=model_id),
                registration_options=RegistrationOptions(origin="local", owned_by=self.get_type(), warm=False),
                tokenizer=functools.partial(self._tokenizers.archive, installed_info.base_url, model_id),
            )
        if model_type == "decision":
            model_info.registration_id = self.endpoint_registry.register_systemone_as_proxy(
                model=model_id,
                props=ModelProps(
                    private=True,
                    type="decision",
                    endpoints=DECISION_ENDPOINTS,
                    context_window=context_length,
                    max_context_window=context_length,
                    supports_vision=supports_vision,
                ),
                options=ProxyOptions(url=f"{installed_info.base_url}/v1/systemone", rewrite_model_to=model_id),
                registration_options=RegistrationOptions(origin="local", owned_by=self.get_type(), warm=False),
            )
        installed_info.models[model_id] = model_info

    async def _fetch_show_info(self, base_url: str, model_id: str) -> tuple[int | None, bool | None, bool | None, bool | None]:
        """Fetch `/api/show` once, returning `(context_length, supports_vision, supports_audio, supports_decision)`.

        `supports_audio` stays unresolved (`None`) when the response doesn't report it - Ollama does
        not publicly document an "audio" capability tag, so its absence must not be read as "no
        audio support".
        """
        try:
            result = await fetch_from(f"{base_url}/api/show", "POST", {"name": model_id})
            if result.status_code == 200:
                data = json.loads(result.data)
                info = data.get("model_info", {})
                arch = info.get("general.architecture", "")
                context_length = info.get(f"{arch}.context_length")
                context_length = context_length if isinstance(context_length, int) and context_length > 0 else None

                supports_vision: bool | None = None
                supports_audio: bool | None = None
                supports_decision: bool | None = None
                capabilities = data.get("capabilities")
                if isinstance(capabilities, list):
                    supports_vision = "vision" in capabilities
                    supports_audio = True if "audio" in capabilities else None
                    supports_decision = "decision" in capabilities

                return context_length, supports_vision, supports_audio, supports_decision
        except Exception:
            pass
        return None, None, None, None

    async def _fetch_synced_model_info(
        self, instance: str, installed_info: InstalledInfo, model_id: str
    ) -> tuple[int | None, bool | None, bool | None, bool | None]:
        """Return a synced model's context length (an embedding model's real token limit) and capabilities."""
        context_length, supports_vision, supports_audio, supports_decision = await self._fetch_show_info(installed_info.base_url, model_id)
        if self._determine_model_type(model_id, instance, supports_decision) == "embedding":
            context_length = await self._embedding_context_length(instance, installed_info.base_url, model_id, context_length)
        return context_length, supports_vision, supports_audio, supports_decision

    async def _embedding_context_length(self, instance: str, base_url: str, model_id: str, context_length: int | None) -> int | None:
        """Narrow an embedding model's window to the tokens Ollama really embeds; it silently truncates the rest."""
        token_limit = await self._embedding_token_limits.resolve(instance, base_url, model_id)
        if token_limit is None:
            return context_length
        return min(context_length, token_limit) if context_length else token_limit

    async def _refresh_embedding_token_limits(self, instance: str, installed_info: InstalledInfo) -> None:
        """Measure registered embedding models again whose last measurement failed or expired."""
        stale = [
            model.id
            for model in installed_info.models.values()
            if model.type == "embedding" and not self._embedding_token_limits.has_fresh_measurement(instance, model.id)
        ]
        await asyncio.gather(*(self._refresh_embedding_token_limit(instance, installed_info, model_id) for model_id in stale))

    async def _refresh_embedding_token_limit(self, instance: str, installed_info: InstalledInfo, model_id: str) -> None:
        """Update a registered embedding model's window once a fresh measurement is in; a failed one keeps the previous value."""
        context_length, _, _, _ = await self._fetch_show_info(installed_info.base_url, model_id)
        context_length = await self._embedding_context_length(instance, installed_info.base_url, model_id, context_length)
        model_info = installed_info.models.get(model_id)
        if model_info is None or not model_info.registration_id:
            return
        if not self._embedding_token_limits.has_fresh_measurement(instance, model_id):
            return
        self.endpoint_registry.update_context_window(
            model_info.registration_id, context_window=context_length, max_context_window=context_length
        )

    async def _retry_unresolved_capabilities(self, installed_info: InstalledInfo, model_id: str) -> None:
        """Retry `/api/show` for an already-registered model whose capabilities are still unknown.

        A model already in `installed_info.models` is otherwise skipped on every sync - fine for
        settled fields, but if `/api/show` failed (or returned no `capabilities` list) on first
        sync, `supports_vision`/`supports_audio` would stay `None` forever without this retry, and
        a tag re-pulled under the same name would never pick up newly-resolved values either.
        Models with at least one already-resolved value are left alone: only "never determined"
        (both still `None`) is retried, so this doesn't re-hit the network for every synced model.
        """
        model_info = installed_info.models[model_id]
        if model_info.supports_vision is not None or model_info.supports_audio is not None:
            return
        _, supports_vision, supports_audio, _ = await self._fetch_show_info(installed_info.base_url, model_id)
        if supports_vision is None and supports_audio is None:
            return
        model_info.supports_vision = supports_vision
        model_info.supports_audio = supports_audio
        if model_info.registration_id:
            self.endpoint_registry.update_modality_support(
                model_info.registration_id, supports_vision=supports_vision, supports_audio=supports_audio
            )

    def _remove_stale_models(self, installed_info: InstalledInfo, stale_ids: set[str]) -> None:
        for removed_id in stale_ids:
            model = installed_info.models.pop(removed_id, None)
            if model is not None:
                if model.type == "llm":
                    self.endpoint_registry.unregister_chat_completion(model.registered_name, model.registration_id)
                if model.type == "embedding":
                    self.endpoint_registry.unregister_embeddings(model.registered_name, model.registration_id)
                if model.type == "decision":
                    self.endpoint_registry.unregister_systemone(model.registered_name, model.registration_id)
            self.models_downloaded.pop(removed_id, None)

    def _is_registered_or_installing(self, instance: str, installed_info: InstalledInfo, model_id: str) -> bool:
        """Tell whether `model_id` is already registered or an install of it, which registers it itself, is underway."""
        return model_id in installed_info.models or (instance, model_id) in self._installing

    async def _sync_models_from_external_ollama(
        self, instance: str, installed_info: InstalledInfo, *, is_initial_sync: bool = False
    ) -> None:
        res_tags = await fetch_from(f"{installed_info.base_url}/api/tags", "GET", None)
        if res_tags.status_code != 200:
            return

        tag_data = json.loads(res_tags.data)
        saved_model_ids = {m.model_id for m in (self.get_instance_info(instance).config.models or [])}
        current_model_ids: set[str] = set()

        new_entries: list[tuple[str, int]] = []
        for entry in tag_data.get("models", []):
            model_id: str = entry["name"].removesuffix(":latest")
            current_model_ids.add(model_id)

            if model_id in installed_info.models:
                await self._retry_unresolved_capabilities(installed_info, model_id)
                continue
            if not is_initial_sync and model_id in self.models_downloaded:
                continue
            if self._is_registered_or_installing(instance, installed_info, model_id):
                continue
            new_entries.append((model_id, entry.get("size", 0)))

        infos = await asyncio.gather(*(self._fetch_synced_model_info(instance, installed_info, model_id) for model_id, _ in new_entries))
        for (model_id, size), (context_length, supports_vision, supports_audio, supports_decision) in zip(new_entries, infos, strict=True):
            if self._is_registered_or_installing(instance, installed_info, model_id):
                continue
            self._register_synced_model(
                instance, installed_info, model_id, size, context_length, supports_vision, supports_audio, supports_decision
            )

        self._remove_stale_models(installed_info, (set(installed_info.models) | saved_model_ids) - current_model_ids)

    async def _uninstall_instance(self, instance: str, options: UninstallServiceIn) -> None:
        if instance in self._sync_tasks:
            self._sync_tasks[instance].cancel()
            del self._sync_tasks[instance]
        self._stop_warmth_poll_task(instance)

        installed = self.get_instance_info(instance).installed
        if installed:
            models = list(installed.models.values())

            for model in models:
                if model.type == "llm":
                    self.endpoint_registry.unregister_chat_completion(model.registered_name, model.registration_id)
                if model.type == "embedding":
                    self.endpoint_registry.unregister_embeddings(model.registered_name, model.registration_id)
                if model.type == "decision":
                    self.endpoint_registry.unregister_systemone(model.registered_name, model.registration_id)

            await asyncio.gather(
                *[
                    self._uninstall_model(instance, model.id, UninstallModelIn(purge=options.purge))
                    for model in models
                    if not self.is_model_installed_in_other_instance(instance, model.id)
                ]
            )

        self.instances_info[instance].installed = None

        if options.purge:
            if not any(i.installed for i in self.instances_info.values()):
                self.service_downloaded = False
                await self._clear_working_dir()
                self.models_downloaded = {}

            if instance == "default":
                self.instances_info["default"] = Instance(None, None, {}, InstanceConfig())
            else:
                del self.instances_info[instance]

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
                supports_vision: bool | None = None
                supports_audio: bool | None = None
                if model_id in info.models:
                    installed = info.models[model_id].get_info()
                    supports_vision = info.models[model_id].supports_vision
                    supports_audio = info.models[model_id].supports_audio
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
                            supports_vision=supports_vision,
                            supports_audio=supports_audio,
                        )
                    )
        return ListModelsOut(list=out_list)

    async def get_model(self, instance: str, model_id: str) -> RetrieveModelOut:
        """Get the model."""
        info = self.get_instance_installed_info(instance)

        if model_id not in self.models[instance]:
            raise HTTPException(status_code=400, detail="Model not found")

        model = self.models[instance][model_id]
        supports_vision: bool | None = None
        supports_audio: bool | None = None
        if model_id in info.models:
            installed = info.models[model_id].get_info()
            supports_vision = info.models[model_id].supports_vision
            supports_audio = info.models[model_id].supports_audio
        else:
            installed = self._get_model_installed_info(instance, model_id)
        return RetrieveModelOut(
            id=model_id,
            service=self.get_id(instance),
            type=model.type,
            installed=installed,
            downloaded=model_id in self.models_downloaded,
            size=model.size,
            custom=model.custom,
            supports_vision=supports_vision,
            supports_audio=supports_audio,
            spec=self.get_model_spec(),
            has_docker=False,
        )

    async def _download_model(self, stream: Stream[StreamChunk], model: OllamaModel, model_id: str, base_url: str) -> None:
        progress = Progress(convert_size_to_bytes(model.size) or 0)
        last_values: dict[str, int] = {}

        stream.emit(StreamChunkProgress(type="progress", stage="download", value=0, data={}))
        async for ollama_stream in stream_fetch_from(f"{base_url}/api/pull", "POST", {"model": model_id}, timeout=24 * 60 * 60):
            if (ollama_stream.status_code != 200 and ollama_stream.status_code != 201) or "error" in ollama_stream.data:
                raise_ollama_pull_error(ollama_stream.data)

            data_cleared: list[str] = ollama_stream.data.rstrip().split("\n")
            records = [json.loads(s) for s in data_cleared]
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

    async def _download_model_or_set_progress(
        self,
        stream: Stream[StreamChunk],
        model: OllamaModel,
        model_id: str,
        base_url: str,
    ) -> None:
        if model_id not in self.models_download_progress:
            self.models_download_progress[model_id] = stream
            try:
                await self._download_model(stream, model, model_id, base_url)
            finally:
                del self.models_download_progress[model_id]
        else:
            chunk: StreamChunk
            async for chunk in self.models_download_progress[model_id].as_generator():
                if chunk.get("type") == "progress" and chunk.get("stage") == "download":
                    stream.emit(chunk)
                else:
                    break

    async def _install_model(  # noqa: C901
        self,
        instance: str,
        model_id: str,
        options: InstallModelIn,
        container_name_override: str | None = None,  # noqa: ARG002
    ) -> PromiseWithProgress[InstallModelOut, StreamChunk]:
        parsed_model_options = try_parse_pydantic(OllamaModelOptions, options.spec) if options.spec else OllamaModelOptions()
        info = self.get_instance_installed_info(instance)

        key = (instance, model_id)
        if model_id in info.models or key in self._installing:
            return PromiseWithProgress(value=InstallModelOut(status="OK", details="Already installed or being installed right now."))

        if model_id not in self.models[instance]:
            raise HTTPException(400, "Model not found")

        model = self.models[instance][model_id]
        if model.type == "decision" and not self.support_systemone:
            raise HTTPException(400, decision_models_unsupported_message(self.ollama_version))

        self._installing.add(key)

        async def func(stream: Stream[StreamChunk]) -> InstallModelOut:  # noqa: C901
            try:
                await self._download_model_or_set_progress(stream, model, model_id, info.base_url)

                stream.emit(StreamChunkProgress(type="progress", stage="install", value=0, data={}))
                if parsed_model_options.alive_time != "":
                    await fetch_from(
                        f"{info.base_url}/api/generate",
                        "POST",
                        {"model": model_id, "keep_alive": parsed_model_options.alive_time},
                    )

                context_length, supports_vision, supports_audio, _ = await self._fetch_show_info(info.base_url, model_id)
                if model.type == "embedding":
                    context_length = await self._embedding_context_length(instance, info.base_url, model_id, context_length)
                registered_name = parsed_model_options.rename if parsed_model_options.rename else model_id
                registered_alias = parsed_model_options.alias
                info.models[model_id] = model_info = ModelInstalledInfo(
                    id=model_id,
                    type=model.type,
                    registered_name=registered_name,
                    options=options,
                    registration_id="",
                    supports_vision=supports_vision,
                    supports_audio=supports_audio,
                )
                try:
                    if model.type == "llm":
                        model_info.registration_id = self.endpoint_registry.register_chat_completion_as_proxy(
                            model=registered_name,
                            props=ModelProps(
                                private=True,
                                type="llm",
                                context_window=context_length,
                                max_context_window=context_length,
                                supports_vision=supports_vision,
                                supports_audio=supports_audio,
                            ),
                            chat_completions=ProxyOptions(
                                url=f"{info.base_url}/v1/chat/completions", rewrite_model_to=model_id, forward_extra_fields=True
                            ),
                            completions=ProxyOptions(
                                url=f"{info.base_url}/v1/completions", rewrite_model_to=model_id, forward_extra_fields=True
                            ),
                            responses=(
                                ProxyOptions(url=f"{info.base_url}/v1/responses", rewrite_model_to=model_id)
                                if self.support_responses
                                else None
                            ),
                            messages=(
                                ProxyOptions(url=f"{info.base_url}/v1/messages", rewrite_model_to=model_id)
                                if self.support_messages
                                else None
                            ),
                            ollama_chat=ProxyOptions(url=f"{info.base_url}/api/chat", rewrite_model_to=model_id),
                            # Ollama lazily loads a model into memory on its first inference request, so a
                            # freshly-installed model is never warm yet - registering it as warm here would win
                            # ranking over an actually-warm instance and route the first requests into a cold load.
                            # The warmth-poll task (already running per-instance) corrects this to `True` within one
                            # tick once `/api/ps` confirms it's genuinely loaded (or leaves it `False` otherwise).
                            registration_options=RegistrationOptions(
                                origin="local", alias=registered_alias, owned_by=self.get_type(), warm=False
                            ),
                        )
                    if model.type == "embedding":
                        model_info.registration_id = self.endpoint_registry.register_embeddings_as_proxy(
                            model=registered_name,
                            props=ModelProps(
                                private=True,
                                type="embedding",
                                endpoints=EMBEDDINGS_ENDPOINTS,
                                context_window=context_length,
                                max_context_window=context_length,
                            ),
                            options=ProxyOptions(url=f"{info.base_url}/v1/embeddings", rewrite_model_to=model_id),
                            registration_options=RegistrationOptions(
                                origin="local", alias=registered_alias, owned_by=self.get_type(), warm=False
                            ),
                            tokenizer=functools.partial(self._tokenizers.archive, info.base_url, model_id),
                        )
                    if model.type == "decision":
                        model_info.registration_id = self.endpoint_registry.register_systemone_as_proxy(
                            model=registered_name,
                            props=ModelProps(
                                private=True,
                                type="decision",
                                endpoints=DECISION_ENDPOINTS,
                                context_window=context_length,
                                max_context_window=context_length,
                                supports_vision=supports_vision,
                            ),
                            options=ProxyOptions(url=f"{info.base_url}/v1/systemone", rewrite_model_to=model_id),
                            registration_options=RegistrationOptions(
                                origin="local", alias=registered_alias, owned_by=self.get_type(), warm=False
                            ),
                        )
                    stream.emit(StreamChunkProgress(type="progress", stage="install", value=1, data={}))
                    self.models_downloaded[model_id] = DownloadedInfo()
                    await self._sync_models_from_external_ollama(instance, info, is_initial_sync=False)
                    await self._save()
                    return InstallModelOut(status="OK", details="Installed")
                except Exception:
                    if model_info.registration_id:
                        if model.type == "llm":
                            self.endpoint_registry.unregister_chat_completion(model_info.registered_name, model_info.registration_id)
                        if model.type == "embedding":
                            self.endpoint_registry.unregister_embeddings(model_info.registered_name, model_info.registration_id)
                        if model.type == "decision":
                            self.endpoint_registry.unregister_systemone(model_info.registered_name, model_info.registration_id)
                    if info.models.get(model_id) is model_info:
                        info.models.pop(model_id, None)
                    raise
            finally:
                self._installing.discard(key)

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
            if model.type == "decision":
                self.endpoint_registry.unregister_systemone(model.registered_name, model.registration_id)

        if options.purge and model_id in self.models_downloaded:
            await fetch_from(f"{info.base_url}/api/delete", "DELETE", {"name": model_id})
            del self.models_downloaded[model_id]

        await self._sync_models_from_external_ollama(instance, info, is_initial_sync=False)
        await self._save()

    async def sync_models(self, instance: str) -> None:
        """Trigger an immediate sync from external Ollama."""
        info = self.get_instance_installed_info(instance)
        await self._sync_models_from_external_ollama(instance, info, is_initial_sync=False)
        await self._save()
