# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Remote service."""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar
from urllib.parse import urljoin

from fastapi import HTTPException
from pydantic import BaseModel

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
    try_parse_pydantic,
)


class RemoteModel(BaseModel):
    type: str
    real_model_name: str | None = None
    messages: bool = True
    responses: bool = True
    completions: bool = True
    legacy_completions: bool = True
    custom: CustomModelId | None = None
    private: bool = False
    context_length: int | None = None
    max_context_length: int | None = None


class RemoteCustomModel(BaseModel):
    id: str
    type: Literal["llm", "embedding", "stt", "tts", "txt2img"]
    completions: bool = True
    legacy_completions: bool = True
    messages: bool = True
    responses: bool = True
    context_length: int | None = None
    max_context_length: int | None = None


def get_model_props(model: RemoteModel | RemoteCustomModel) -> ModelProps:
    """Get model props."""
    endpoints = []
    if model.legacy_completions:
        endpoints.append("/v1/completions")
    if model.completions:
        endpoints.append("/v1/chat/completions")
    if model.responses:
        endpoints.append("/v1/responses")
    if model.messages:
        endpoints.append("/v1/messages")
    return ModelProps(
        private=False,
        type=model.type,
        endpoints=endpoints,
        context_window=model.context_length,
        max_context_window=model.max_context_length,
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

    @staticmethod
    def get_spec(default_api_url: str | None = None) -> ServiceSpecification:
        """Provide the specification for the install service modal fields."""
        raise NotImplementedError


class DefaultRemoteServiceOptions(BaseServiceOptions):
    """Default contains an OpenAI compatible authorization header."""

    api_key: str = ""

    @property
    def headers(self) -> dict[str, str]:
        """Translate data into header."""
        return {"Authorization": f"Bearer {self.api_key}"}

    @staticmethod
    def get_spec(default_api_url: str | None = None) -> ServiceSpecification:
        """Provide the specification for the install service modal fields compatible with OpenAI."""
        return ServiceSpecification(
            fields=[
                ServiceField(type="text", name="api_url", description="API URL", required=False, default=default_api_url),
                ServiceField(type="password", name="api_key", description="API Key (required for OpenAI)", required=False),
            ]
        )


# Configurable options type for remote services.
# Defaults to DefaultRemoteServiceOptions (Bearer token auth).
# Override with provider-specific options (e.g. ClaudeServiceOptions).
T_Options = TypeVar("T_Options", bound=BaseServiceOptions, default=DefaultRemoteServiceOptions)


class RemoteModelOptions(BaseModel):
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

    def _after_init(self) -> None:
        self.models = {}
        self.load_default_models("default")

    def load_default_models(self, instance: str) -> None:
        """Load default models to instance."""
        self.models[instance] = self.get_models_registry().models.copy()

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
        """Return the service specification."""
        return self.options_class.get_spec(default_api_url=self.get_default_url())

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
                CustomModelField(type="oneof", name="type", description="Model Type", values=["llm", "embedding", "stt", "tts", "txt2img"]),
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

    def _generate_instance_config(self, info: InstalledInfo[T_Options] | None, custom: list[CustomModel] | None) -> InstanceConfig:
        return InstanceConfig(
            options=info.options if info else None,
            models=[ModelConfig(model_id=x.id, options=x.options) for x in info.models.values()] if info else [],
            custom=custom,
        )

    def _load_download_info(self, data: dict[str, Any]) -> DownloadedInfo:
        return DownloadedInfo(**data)

    async def _install_instance(
        self, instance: str, options: InstallServiceIn
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
            for model in instance_info.installed.models.copy().values():
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

                if not self.is_model_installed_in_other_instance(instance, model.id):
                    await self._uninstall_model(instance, model.id, UninstallModelIn(purge=options.purge))

        self.instances_info[instance].installed = None

        if options.purge:
            if len(self.instances_info) < 2:
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
                            type=model.type,
                            installed=installed,
                            downloaded=model_id in self.models_downloaded,
                            size="",
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
            size="",
            custom=model.custom,
            spec=self.get_model_spec(),
            has_docker=False,
        )

    async def _install_model(
        self, instance: str, model_id: str, options: InstallModelIn
    ) -> PromiseWithProgress[InstallModelOut, StreamChunk]:
        parsed_model_options = try_parse_pydantic(RemoteModelOptions, options.spec) if options.spec else RemoteModelOptions()
        info = self.get_instance_installed_info(instance)

        if not self.models.get(instance):
            self.models[instance] = {}

        if model_id in info.models:
            return PromiseWithProgress(value=InstallModelOut(status="OK", details="Already installed"))

        if model_id not in self.models[instance]:
            raise HTTPException(400, "Model not found")

        model = self.models[instance][model_id]
        headers = info.parsed_options.headers

        async def func(stream: Stream[StreamChunk]) -> InstallModelOut:
            stream.emit(StreamChunkProgress(type="progress", stage="install", value=0, data={}))
            registered_name = parsed_model_options.alias if parsed_model_options.alias else model_id
            info.models[model_id] = model_info = ModelInstalledInfo(
                id=model_id,
                type=model.type,
                registered_name=registered_name,
                options=options,
                completions=model.completions,
                legacy_completions=model.legacy_completions,
                registration_id="",
            )

            url_base = urljoin(info.parsed_options.api_url, self.api_version)
            props = get_model_props(model)
            if model.type == "llm":
                model_info.registration_id = self.endpoint_registry.register_chat_completion_as_proxy(
                    model=registered_name,
                    props=props,
                    messages=ProxyOptions(
                        url=urljoin(url_base, "messages"),
                        rewrite_model_to=model.real_model_name,
                        headers=headers,
                    )
                    if model.messages
                    else None,
                    responses=ProxyOptions(
                        url=urljoin(url_base, "responses"),
                        rewrite_model_to=model.real_model_name,
                        headers=headers,
                    )
                    if model.responses
                    else None,
                    chat_completions=ProxyOptions(
                        url=urljoin(url_base, "chat/completions"),
                        rewrite_model_to=model.real_model_name or model_id,
                        headers=headers,
                    )
                    if model.completions
                    else None,
                    completions=ProxyOptions(
                        url=urljoin(url_base, "completions"),
                        rewrite_model_to=model.real_model_name,
                        headers=headers,
                    )
                    if model.legacy_completions
                    else None,
                    registration_options=None,
                )
            if model.type == "tts":
                url = urljoin(url_base, "audio/speech")
                model_info.registration_id = self.endpoint_registry.register_audio_speech_as_proxy(
                    model=registered_name,
                    props=props,
                    options=ProxyOptions(
                        url=url,
                        rewrite_model_to=model.real_model_name,
                        headers=headers,
                    ),
                    registration_options=None,
                )
            if model.type == "stt":
                url = urljoin(url_base, "audio/transcriptions")
                model_info.registration_id = self.endpoint_registry.register_audio_transcriptions_as_proxy(
                    model=registered_name,
                    props=props,
                    options=ProxyOptions(
                        url=url,
                        rewrite_model_to=model.real_model_name,
                        headers=headers,
                    ),
                    registration_options=None,
                )
            if model.type == "txt2img":
                url = urljoin(url_base, "images/generations")
                model_info.registration_id = self.endpoint_registry.register_image_generations_as_proxy(
                    model=registered_name,
                    props=props,
                    options=ProxyOptions(
                        url=url,
                        rewrite_model_to=model.real_model_name,
                        headers=headers,
                    ),
                    registration_options=None,
                )
            if model.type == "embedding":
                url = urljoin(url_base, "embeddings")
                model_info.registration_id = self.endpoint_registry.register_embeddings_as_proxy(
                    model=registered_name,
                    props=props,
                    options=ProxyOptions(
                        url=url,
                        rewrite_model_to=model.real_model_name,
                        headers=headers,
                    ),
                    registration_options=None,
                )
            stream.emit(StreamChunkProgress(type="progress", stage="install", value=1, data={}))
            self.models_downloaded[model_id] = DownloadedInfo()
            return InstallModelOut(status="OK", details="Installed")

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

        if options.purge and model_id in self.models_downloaded:
            del self.models_downloaded[model_id]
