# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Remote service."""

import asyncio
from abc import abstractmethod
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar, get_args

from fastapi import HTTPException
from pydantic import BaseModel

from server.endpointregistry import ProxyOptions, RegistrationId, RegistrationOptions
from server.models.api import DECISION_ENDPOINTS, EMBEDDINGS_ENDPOINTS, IMG_ENDPOINTS, STT_ENDPOINTS, TTS_ENDPOINTS, ModelProps
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
from server.services.base2_service import Base2Service, CustomModel, Instance, InstanceConfig, ModelConfig
from server.utils.core import (
    PromiseWithProgress,
    Stream,
    StreamChunk,
    StreamChunkProgress,
    Utils,
    positive_context_window_or_none,
    try_parse_pydantic,
)
from server.utils.logger import logger

RemoteModelType = Literal["llm", "embedding", "stt", "tts", "txt2img", "decision"]
REMOTE_MODEL_TYPES: list[str] = list(get_args(RemoteModelType))


class RemoteModel(BaseModel):
    # None means the model's type/capabilities couldn't be resolved from either the live listing
    # or the known RemoteConst overlay -- see `capabilities_resolved` below. Statically-declared
    # RemoteConst entries always set a real type.
    type: RemoteModelType | None
    real_model_name: str | None = None
    messages: bool = True
    responses: bool = True
    completions: bool = True
    legacy_completions: bool = True
    custom: CustomModelId | None = None
    private: bool = False
    context_length: int | None = None
    max_context_length: int | None = None
    # True when a catalog entry was previously seen (and possibly installed) but is no longer
    # present in the most recent live provider listing. Installed models stay registered while
    # stale — see cloud-model-autolist-policy design.md Decision 3.
    stale: bool = False

    @property
    def capabilities_resolved(self) -> bool:
        """Whether `type`/capabilities are known -- derived from `type`, not independently settable.

        False for entries populated from a live provider listing whose type/capabilities could not
        be resolved (no live field, no match in the service's hardcoded RemoteConst overlay).
        Statically-declared RemoteConst entries always set a real type and are always resolved.
        """
        return self.type is not None


class LiveModelEntry(BaseModel):
    """A single model as reported by a provider's live model-listing API.

    Fields left as None mean "not reported by the provider" — resolution falls back to the
    service's hardcoded RemoteConst overlay for that model ID, per cloud-model-autolist-policy.
    """

    id: str
    type: RemoteModelType | None = None
    messages: bool | None = None
    responses: bool | None = None
    completions: bool | None = None
    legacy_completions: bool | None = None
    context_length: int | None = None
    max_context_length: int | None = None


class RemoteCustomModel(BaseModel):
    id: str
    type: RemoteModelType
    completions: bool = True
    legacy_completions: bool = True
    messages: bool = True
    responses: bool = True
    context_length: int | None = None
    max_context_length: int | None = None


# Endpoints advertised by non-LLM models. The completions/responses/messages flags only apply to LLMs.
_NON_LLM_ENDPOINTS: dict[RemoteModelType, list[str]] = {
    "embedding": EMBEDDINGS_ENDPOINTS,
    "stt": STT_ENDPOINTS,
    "tts": TTS_ENDPOINTS,
    "txt2img": IMG_ENDPOINTS,
    "decision": DECISION_ENDPOINTS,
}


def get_model_endpoints(model: RemoteModel | RemoteCustomModel) -> list[str]:
    """Get the endpoints the model accepts, based on its type."""
    if model.type is None:
        return []
    if model.type != "llm":
        return list(_NON_LLM_ENDPOINTS[model.type])
    endpoints: list[str] = []
    if model.legacy_completions:
        endpoints.append("/v1/completions")
    if model.completions:
        endpoints.append("/v1/chat/completions")
    if model.responses:
        endpoints.append("/v1/responses")
    if model.messages:
        endpoints.append("/v1/messages")
    return endpoints


def get_model_props(model: RemoteModel | RemoteCustomModel) -> ModelProps:
    """Get model props."""
    return ModelProps(
        private=False,
        type=model.type or "unknown",
        endpoints=get_model_endpoints(model),
        context_window=positive_context_window_or_none(model.context_length),
        max_context_window=positive_context_window_or_none(model.max_context_length),
    )


class RemoteConst(BaseModel):
    models: dict[str, RemoteModel]


@dataclass
class ModelInstalledInfo:
    id: str
    registered_name: str
    type: str
    options: InstallModelIn
    completions: bool
    legacy_completions: bool
    registration_id: RegistrationId

    def get_info(self) -> ModelInfo:
        """Get info."""
        return ModelInfo(spec=self.options.spec, registration_id=self.registration_id)


class BaseServiceOptions(BaseModel):
    """Base class for provider-specific HTTP headers."""

    api_url: str

    @property
    def headers(self) -> dict[str, str]:
        """Resolve model fields into final HTTP headers."""
        raise NotImplementedError


class DefaultRemoteServiceOptions(BaseServiceOptions):
    """Default contains an OpenAI compatible authorization header."""

    api_key: str = ""

    @property
    def headers(self) -> dict[str, str]:
        """Translate data into header."""
        return {"Authorization": f"Bearer {self.api_key}"}


# Configurable options type for remote services.
# Defaults to DefaultRemoteServiceOptions (Bearer token auth).
# Override with provider-specific options (e.g. ClaudeServiceOptions).
T_Options = TypeVar("T_Options", bound=BaseServiceOptions, default=DefaultRemoteServiceOptions)


class RemoteModelOptions(BaseModel):
    rename: str | None = None
    alias: str | None = None


@dataclass
class InstalledInfo(Generic[T_Options]):
    """State of an installed remote service instance."""

    models: dict[str, ModelInstalledInfo]
    options: InstallServiceIn
    parsed_options: T_Options


@dataclass
class DownloadedInfo:
    pass


class RemoteService(Base2Service[InstalledInfo[T_Options], DownloadedInfo]):
    """Base class for remote API services (OpenAI, Anthropic, etc.)."""

    api_version: str = "v1/"
    models: dict[str, dict[str, RemoteModel]]
    options_class: type[T_Options]  # set by each subclass to parse frontend input
    _installing: set[tuple[str, str]]
    _persists_model_definitions = False

    def _after_init(self) -> None:
        self._installing = set()
        self.load_default_models("default")

    def load_default_models(self, instance: str) -> None:
        """Load default models to instance."""
        self.models[instance] = self.get_models_registry().models.copy()

    def get_supported_model_types(self) -> Collection[str] | None:
        """Model `type`s this service accepts from a live provider listing.

        Defaults to the distinct types already declared in the hardcoded RemoteConst overlay
        (empty overlay means no opinion, i.e. no filtering). Override for a service whose live
        listing may plausibly include types its RemoteConst overlay doesn't happen to cover yet.
        """
        types = {model.type for model in self.get_models_registry().models.values() if model.type is not None}
        return types or None

    async def _fetch_live_models(self, instance: str) -> Sequence[LiveModelEntry] | None:  # noqa: ARG002
        """Fetch the provider's live model listing for `instance`.

        Returns None if this service doesn't support live listing (the default); a
        provider-specific subclass overrides this to call the provider's own model-list API.
        """
        return None

    def _merge_live_models(self, instance: str, live_entries: Sequence[LiveModelEntry]) -> dict[str, RemoteModel]:
        """Merge a live provider listing into `instance`'s model catalog.

        Applies the type-based eligibility filter, resolves capability metadata (live fields ->
        RemoteConst overlay -> unresolved), and marks previously catalogued, currently installed
        models that are absent from this listing as stale instead of dropping them. See
        openspec/changes/cloud-model-autolist-policy/design.md.
        """
        overlay = self.get_models_registry().models
        allowed_types = self.get_supported_model_types()
        existing = self.models.get(instance, {})
        instance_info = self.instances_info.get(instance)
        installed_ids = set(instance_info.installed.models) if instance_info and instance_info.installed else set()

        # Seed with custom models so a refresh never drops a user-defined model that isn't part
        # of the provider's live listing (custom models aren't provider-catalogued to begin with).
        merged: dict[str, RemoteModel] = {model_id: model for model_id, model in existing.items() if model.custom is not None}
        for entry in live_entries:
            if entry.id in merged:
                logger.debug(
                    "%s live model %r on instance %r shadowed by an existing custom model, skipping", self.get_type(), entry.id, instance
                )
                continue  # a custom model already claims this id -- leave it alone
            known = overlay.get(entry.id)
            resolved_type = entry.type or (known.type if known else None)
            if resolved_type is not None and allowed_types is not None and resolved_type not in allowed_types:
                logger.debug(
                    "%s live model %r on instance %r has type %r, not in supported types %s, dropping",
                    self.get_type(),
                    entry.id,
                    instance,
                    resolved_type,
                    allowed_types,
                )
                continue
            merged[entry.id] = RemoteModel(
                type=resolved_type,
                real_model_name=known.real_model_name if known else None,
                messages=entry.messages if entry.messages is not None else (known.messages if known else True),
                responses=entry.responses if entry.responses is not None else (known.responses if known else True),
                completions=entry.completions if entry.completions is not None else (known.completions if known else True),
                legacy_completions=entry.legacy_completions
                if entry.legacy_completions is not None
                else (known.legacy_completions if known else True),
                context_length=entry.context_length if entry.context_length is not None else (known.context_length if known else None),
                max_context_length=entry.max_context_length
                if entry.max_context_length is not None
                else (known.max_context_length if known else None),
                stale=False,
            )

        for model_id in installed_ids - merged.keys():
            previous = existing.get(model_id)
            if previous is not None:
                merged[model_id] = previous.model_copy(update={"stale": True})

        return merged

    async def refresh_catalog(self) -> PromiseWithProgress[CatalogRefreshOut, StreamChunk]:
        """Refresh model catalogs from live provider listings, for services that support one.

        Applies `_merge_live_models` per instance. Raises the inherited 405 if this provider
        doesn't override `_fetch_live_models` at all (no live listing support). If it does but
        every instance's live fetch failed (bad key, network error, etc.), raises 502 instead of
        the misleading 405 -- the provider is supported, it's just currently unreachable.
        """
        supports_live = type(self)._fetch_live_models is not RemoteService._fetch_live_models
        if not supports_live:
            return await super().refresh_catalog()

        added = 0
        total = 0
        any_live = False
        failed_instances: list[str] = []
        for instance in list(self.models):
            live_entries = await self._fetch_live_models(instance)
            if live_entries is None:
                failed_instances.append(instance)
                continue
            any_live = True
            before_ids = set(self.models[instance])
            merged = self._merge_live_models(instance, live_entries)
            added += len(set(merged) - before_ids)
            self.models[instance] = merged
            total += len(merged)

        if not any_live:
            raise HTTPException(502, "Failed to fetch the live model catalog from the provider. Check the API key and connectivity.")
        if failed_instances:
            logger.warning(
                "%s live catalog refresh failed for instance(s) %s; their catalogs were left unchanged", self.get_type(), failed_instances
            )

        return PromiseWithProgress(value=CatalogRefreshOut(added=added, total=total))

    async def _after_install(self, instance: str) -> None:
        """Best-effort live catalog refresh right after (re)installing `instance`.

        Providers without live listing (`_fetch_live_models` returning None) are left on the
        static RemoteConst overlay, same as before a manual refresh; a live provider that's
        merely unreachable at install time is picked up on the next manual or periodic refresh.
        Errors from `_merge_live_models` itself (as opposed to the network fetch, which already
        catches its own failures) are swallowed here too -- a bug in the merge must not surface as
        an install/update/startup failure, since the caller has already committed `installed`.
        """
        try:
            live_entries = await self._fetch_live_models(instance)
            if live_entries is not None:
                self.models[instance] = self._merge_live_models(instance, live_entries)
        except Exception:
            logger.exception("%s failed to refresh the live catalog for instance %r after install", self.get_type(), instance)

    def get_size(self) -> ServiceSize:
        """Return the service size."""
        return ""

    async def stop_instance(self, instance: str) -> None:
        """Stop the service gracefully.

        Remote service has no containers to stop.
        """

    @abstractmethod
    def get_default_url(self) -> str:
        """Return the default url."""

    @abstractmethod
    def get_models_registry(self) -> RemoteConst:
        """Return the models registry."""

    def get_spec(self) -> ServiceSpecification:
        """Provide the specification for the install service modal fields compatible with OpenAI."""
        return ServiceSpecification(
            fields=[
                ServiceField(
                    type="text",
                    name="api_url",
                    description="API URL",
                    placeholder="https://api.example.com (with or without a trailing /v1)",
                    required=False,
                    default=self.get_default_url(),
                ),
                ServiceField(type="password", name="api_key", description="API Key", required=False),
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
            ]
        )

    def get_custom_model_spec(self) -> CustomModelSpecification | None:
        """Return the custom model specification or None if custom model is not supported."""
        return CustomModelSpecification(
            fields=[
                CustomModelField(type="text", name="id", description="Model ID", placeholder="my-custom-model"),
                CustomModelField(type="oneof", name="type", description="Model Type", values=REMOTE_MODEL_TYPES),
                CustomModelField(
                    type="bool",
                    name="completions",
                    description="Support /v1/chat/completions",
                    default="true",
                    display="type=llm",
                ),
                CustomModelField(
                    type="bool",
                    name="legacy_completions",
                    description="Support /v1/completions",
                    default="true",
                    display="type=llm",
                ),
                CustomModelField(
                    type="bool",
                    name="responses",
                    description="Support /v1/responses",
                    default="true",
                    display="type=llm",
                ),
                CustomModelField(
                    type="bool",
                    name="messages",
                    description="Support /v1/messages",
                    default="true",
                    display="type=llm",
                ),
                CustomModelField(
                    type="number", name="context_length", description="Context window size", display="type=llm", required=False
                ),
                CustomModelField(
                    type="number", name="max_context_length", description="Maximum context window size", display="type=llm", required=False
                ),
            ]
        )

    def get_installed_info(self, instance: str) -> bool | InstallServiceProgress | ServiceOptions:
        """Get service installed info."""
        installed = self.get_instance_info(instance).installed
        return self._get_service_installed_info(instance) if installed is None else installed.options.spec

    def _generate_instance_config(
        self,
        instance: str,  # noqa: ARG002
        info: InstalledInfo[T_Options] | None,
        custom: list[CustomModel] | None,
    ) -> InstanceConfig:
        return InstanceConfig(
            options=info.options if info else None,
            models=[ModelConfig(model_id=x.id, options=x.options) for x in info.models.values()] if info else [],
            custom=custom,
        )

    def _load_download_info(self, data: dict[str, Any]) -> DownloadedInfo:
        return DownloadedInfo(**data)

    async def _install_instance(
        self,
        instance: str,
        options: InstallServiceIn,
        container_name_override: str | None = None,  # noqa: ARG002
    ) -> PromiseWithProgress[InstalledInfo[T_Options], StreamChunk]:
        """Install a remote service instance with validated frontend options."""
        if not self.models.get(instance):
            self.load_default_models(instance)

        if "api_url" not in options.spec:
            options.spec["api_url"] = self.get_default_url()

        parsed_options = try_parse_pydantic(self.options_class, options.spec)

        async def func(stream: Stream[StreamChunk]) -> InstalledInfo[T_Options]:  # noqa: ARG001
            self.service_downloaded = True
            return InstalledInfo(models={}, options=options, parsed_options=parsed_options)

        return PromiseWithProgress(func=func)

    async def _uninstall_instance(self, instance: str, options: UninstallServiceIn) -> None:  # noqa: C901
        instance_info = self.get_instance_info(instance)
        if instance_info.installed:
            models = list(instance_info.installed.models.copy().values())

            for model in models:
                if model.type == "llm":
                    self.endpoint_registry.unregister_chat_completion(model.registered_name, model.registration_id)
                if model.type == "tts":
                    self.endpoint_registry.unregister_audio_speech(model.registered_name, model.registration_id)
                if model.type == "stt":
                    self.endpoint_registry.unregister_audio_transcriptions(model.registered_name, model.registration_id)
                if model.type == "txt2img":
                    self.endpoint_registry.unregister_image_generations(model.registered_name, model.registration_id)
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

    def _add_custom_model(self, instance: str, model: CustomModel) -> None:
        parsed = try_parse_pydantic(RemoteCustomModel, model.data)

        if not self.models.get(instance):
            self.models[instance] = {}

        if parsed.id in self.models[instance]:
            raise HTTPException(400, "Model with given id already exists.")

        self.models[instance][parsed.id] = RemoteModel(
            type=parsed.type,
            real_model_name=None,
            messages=parsed.messages,
            responses=parsed.responses,
            completions=parsed.completions,
            legacy_completions=parsed.legacy_completions,
            custom=model.id,
        )

    def _remove_custom_model(self, instance: str, model: CustomModel) -> None:
        installed = self.get_instance_info(instance).installed
        parsed = try_parse_pydantic(RemoteCustomModel, model.data)
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
                            type=model.type or "unknown",
                            installed=installed,
                            downloaded=model_id in self.models_downloaded,
                            size="",
                            custom=model.custom,
                            spec=self.get_model_spec(),
                            has_docker=False,
                            capabilities_resolved=model.capabilities_resolved,
                            stale=model.stale,
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
            type=model.type or "unknown",
            installed=installed,
            downloaded=model_id in self.models_downloaded,
            size="",
            custom=model.custom,
            spec=self.get_model_spec(),
            has_docker=False,
            capabilities_resolved=model.capabilities_resolved,
            stale=model.stale,
        )

    async def _install_model(  # noqa: C901
        self,
        instance: str,
        model_id: str,
        options: InstallModelIn,
        container_name_override: str | None = None,  # noqa: ARG002
    ) -> PromiseWithProgress[InstallModelOut, StreamChunk]:
        parsed_model_options = try_parse_pydantic(RemoteModelOptions, options.spec) if options.spec else RemoteModelOptions()
        info = self.get_instance_installed_info(instance)

        if not self.models.get(instance):
            self.models[instance] = {}

        key = (instance, model_id)
        if model_id in info.models or key in self._installing:
            return PromiseWithProgress(value=InstallModelOut(status="OK", details="Already installed"))

        if model_id not in self.models[instance]:
            raise HTTPException(400, "Model not found")

        model = self.models[instance][model_id]

        if not model.capabilities_resolved:
            raise HTTPException(
                400,
                f"Cannot install '{model_id}': its capabilities (type, context window, supported endpoints) "
                "could not be resolved from the provider's live listing or known model metadata.",
            )

        headers = info.parsed_options.headers
        self._installing.add(key)

        async def func(stream: Stream[StreamChunk]) -> InstallModelOut:
            try:
                stream.emit(StreamChunkProgress(type="progress", stage="install", value=0, data={}))
                registered_name = parsed_model_options.rename if parsed_model_options.rename else model_id
                registered_alias = parsed_model_options.alias
                info.models[model_id] = model_info = ModelInstalledInfo(
                    id=model_id,
                    type=model.type or "unknown",
                    registered_name=registered_name,
                    options=options,
                    completions=model.completions,
                    legacy_completions=model.legacy_completions,
                    registration_id="",
                )

                url_base = Utils.join_versioned_url(info.parsed_options.api_url, self.api_version)
                props = get_model_props(model)
                try:
                    if model.type == "llm":
                        model_info.registration_id = self.endpoint_registry.register_chat_completion_as_proxy(
                            model=registered_name,
                            props=props,
                            messages=ProxyOptions(
                                url=Utils.join_url(url_base, "messages"),
                                rewrite_model_to=model.real_model_name or model_id,
                                headers=headers,
                            )
                            if model.messages
                            else None,
                            responses=ProxyOptions(
                                url=Utils.join_url(url_base, "responses"),
                                rewrite_model_to=model.real_model_name or model_id,
                                headers=headers,
                            )
                            if model.responses
                            else None,
                            chat_completions=ProxyOptions(
                                url=Utils.join_url(url_base, "chat/completions"),
                                rewrite_model_to=model.real_model_name or model_id,
                                headers=headers,
                            )
                            if model.completions
                            else None,
                            completions=ProxyOptions(
                                url=Utils.join_url(url_base, "completions"),
                                rewrite_model_to=model.real_model_name or model_id,
                                headers=headers,
                            )
                            if model.legacy_completions
                            else None,
                            ollama_chat=None,
                            registration_options=RegistrationOptions(origin="local", alias=registered_alias, owned_by=self.get_type()),
                        )
                    if model.type == "tts":
                        url = Utils.join_url(url_base, "audio/speech")
                        model_info.registration_id = self.endpoint_registry.register_audio_speech_as_proxy(
                            model=registered_name,
                            props=props,
                            options=ProxyOptions(
                                url=url,
                                rewrite_model_to=model.real_model_name or model_id,
                                headers=headers,
                            ),
                            registration_options=RegistrationOptions(origin="local", alias=registered_alias, owned_by=self.get_type()),
                        )
                    if model.type == "stt":
                        url = Utils.join_url(url_base, "audio/transcriptions")
                        model_info.registration_id = self.endpoint_registry.register_audio_transcriptions_as_proxy(
                            model=registered_name,
                            props=props,
                            options=ProxyOptions(
                                url=url,
                                rewrite_model_to=model.real_model_name or model_id,
                                headers=headers,
                            ),
                            registration_options=RegistrationOptions(origin="local", alias=registered_alias, owned_by=self.get_type()),
                        )
                    if model.type == "txt2img":
                        url = Utils.join_url(url_base, "images/generations")
                        model_info.registration_id = self.endpoint_registry.register_image_generations_as_proxy(
                            model=registered_name,
                            props=props,
                            options=ProxyOptions(
                                url=url,
                                rewrite_model_to=model.real_model_name or model_id,
                                headers=headers,
                            ),
                            registration_options=RegistrationOptions(origin="local", alias=registered_alias, owned_by=self.get_type()),
                        )
                    if model.type == "embedding":
                        url = Utils.join_url(url_base, "embeddings")
                        model_info.registration_id = self.endpoint_registry.register_embeddings_as_proxy(
                            model=registered_name,
                            props=props,
                            options=ProxyOptions(
                                url=url,
                                rewrite_model_to=model.real_model_name or model_id,
                                headers=headers,
                            ),
                            registration_options=RegistrationOptions(origin="local", alias=registered_alias, owned_by=self.get_type()),
                        )
                    if model.type == "decision":
                        url = Utils.join_url(url_base, "systemone")
                        model_info.registration_id = self.endpoint_registry.register_systemone_as_proxy(
                            model=registered_name,
                            props=props,
                            options=ProxyOptions(
                                url=url,
                                rewrite_model_to=model.real_model_name or model_id,
                                headers=headers,
                            ),
                            registration_options=RegistrationOptions(origin="local", alias=registered_alias, owned_by=self.get_type()),
                        )
                    stream.emit(StreamChunkProgress(type="progress", stage="install", value=1, data={}))
                    self.models_downloaded[model_id] = DownloadedInfo()
                    return InstallModelOut(status="OK", details="Installed")
                except Exception:
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
            if model.type == "tts":
                self.endpoint_registry.unregister_audio_speech(model.registered_name, model.registration_id)
            if model.type == "stt":
                self.endpoint_registry.unregister_audio_transcriptions(model.registered_name, model.registration_id)
            if model.type == "txt2img":
                self.endpoint_registry.unregister_image_generations(model.registered_name, model.registration_id)
            if model.type == "embedding":
                self.endpoint_registry.unregister_embeddings(model.registered_name, model.registration_id)
            if model.type == "decision":
                self.endpoint_registry.unregister_systemone(model.registered_name, model.registration_id)

        if options.purge and model_id in self.models_downloaded:
            del self.models_downloaded[model_id]
