# DeepFellow Software Framework.
# Copyright © 2025 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Remote service."""

from abc import abstractmethod
from typing import Literal
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
from server.services.base2_service import Base2Service, CustomModel, ModelConfig, ServiceConfig
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
    completions: bool = True
    legacy_completions: bool = True
    props: ModelProps = ModelProps(private=False)
    custom: CustomModelId | None = None


class RemoteCustomModel(BaseModel):
    id: str
    type: Literal["llm", "embedding", "stt", "tts", "txt2img"]
    completions: bool = True
    legacy_completions: bool = True


class RemoteConst(BaseModel):
    models: dict[str, RemoteModel]


class ModelInstalledInfo(BaseModel):
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


class RemoteOptions(BaseModel):
    api_url: str
    api_key: str


class RemoteModelOptions(BaseModel):
    alias: str | None = None


class InstalledInfo:
    def __init__(
        self,
        models: dict[str, ModelInstalledInfo],
        options: InstallServiceIn,
        parsed_options: RemoteOptions,
    ):
        self.models = models
        self.options = options
        self.parsed_options = parsed_options


class RemoteService(Base2Service[InstalledInfo]):
    url_prefix: str = "v1/"
    models: dict[str, RemoteModel]

    def _after_init(self) -> None:
        self.models = self.get_models_registry().models.copy()

    def get_size(self) -> ServiceSize:
        """Return the service size."""
        return ""

    async def stop(self) -> None:
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
        return ServiceSpecification(
            fields=[
                ServiceField(type="text", name="api_url", description="API URL", required=False, default=self.get_default_url()),
                ServiceField(type="password", name="api_key", description="API Key", required=True),
            ]
        )

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
        )

    async def _install_core(self, options: InstallServiceIn) -> PromiseWithProgress[InstalledInfo, StreamChunk]:
        if "api_url" not in options.spec:
            options.spec["api_url"] = self.get_default_url()
        parsed_options = try_parse_pydantic(RemoteOptions, options.spec)

        async def func(stream: Stream[StreamChunk]) -> InstalledInfo:  # noqa: ARG001
            return InstalledInfo(models={}, options=options, parsed_options=parsed_options)

        return PromiseWithProgress(func=func)

    async def _uninstall(self, options: UninstallServiceIn) -> None:
        info = self._check_installed()
        for model in info.models.copy().values():
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
        self.installed = None
        if options.purge:
            await self._clear_working_dir()

    def _add_custom_model(self, model: CustomModel) -> None:
        parsed = try_parse_pydantic(RemoteCustomModel, model.data)
        if parsed.id in self.models:
            raise HTTPException(400, "Model with given id already exists.")
        self.models[parsed.id] = RemoteModel(
            type=parsed.type,
            completions=parsed.completions,
            legacy_completions=parsed.legacy_completions,
            real_model_name=None,
            props=ModelProps(private=False),
            custom=model.id,
        )

    def _remove_custom_model(self, model: CustomModel) -> None:
        parsed = try_parse_pydantic(RemoteCustomModel, model.data)
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
                        size="",
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
            size="",
            custom=model.custom,
            spec=self.get_model_spec(),
            has_docker=False,
        )

    async def _install_model(self, model_id: str, options: InstallModelIn) -> PromiseWithProgress[InstallModelOut, StreamChunk]:
        parsed_model_options = try_parse_pydantic(RemoteModelOptions, options.spec) if options.spec else RemoteModelOptions()
        info = self._check_installed()
        if model_id in info.models:
            return PromiseWithProgress(value=InstallModelOut(status="OK", details="Already installed"))
        if model_id not in self.models:
            raise HTTPException(400, "Model not found")
        model = self.models[model_id]

        async def func(stream: Stream[StreamChunk]) -> InstallModelOut:
            stream.emit(StreamChunkProgress(type="progress", stage="install", value=0))
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
            url_base = urljoin(info.parsed_options.api_url, self.url_prefix)
            if model.type == "llm":
                model_info.registration_id = self.endpoint_registry.register_chat_completion_as_proxy(
                    model=registered_name,
                    props=model.props,
                    chat_completions=ProxyOptions(
                        url=urljoin(url_base, "chat/completions"),
                        rewrite_model_to=model.real_model_name or model_id,
                        headers={"Authorization": f"Bearer {info.parsed_options.api_key}"},
                    )
                    if model.completions
                    else None,
                    completions=ProxyOptions(
                        url=urljoin(url_base, "completions"),
                        rewrite_model_to=model.real_model_name,
                        headers={"Authorization": f"Bearer {info.parsed_options.api_key}"},
                    )
                    if model.legacy_completions
                    else None,
                    registration_options=None,
                )
            if model.type == "tts":
                url = urljoin(url_base, "v1/audio/speech")
                model_info.registration_id = self.endpoint_registry.register_audio_speech_as_proxy(
                    model=registered_name,
                    props=model.props,
                    options=ProxyOptions(
                        url=url,
                        rewrite_model_to=model.real_model_name,
                        headers={"Authorization": f"Bearer {info.parsed_options.api_key}"},
                    ),
                    registration_options=None,
                )
            if model.type == "stt":
                url = urljoin(url_base, "v1/audio/transcriptions")
                model_info.registration_id = self.endpoint_registry.register_audio_transcriptions_as_proxy(
                    model=registered_name,
                    props=model.props,
                    options=ProxyOptions(
                        url=url,
                        rewrite_model_to=model.real_model_name,
                        headers={"Authorization": f"Bearer {info.parsed_options.api_key}"},
                    ),
                    registration_options=None,
                )
            if model.type == "txt2img":
                url = urljoin(url_base, "v1/images/generations")
                model_info.registration_id = self.endpoint_registry.register_image_generations_as_proxy(
                    model=registered_name,
                    props=model.props,
                    options=ProxyOptions(
                        url=url,
                        rewrite_model_to=model.real_model_name,
                        headers={"Authorization": f"Bearer {info.parsed_options.api_key}"},
                    ),
                    registration_options=None,
                )
            if model.type == "embedding":
                url = urljoin(url_base, "v1/embeddings")
                model_info.registration_id = self.endpoint_registry.register_embeddings_as_proxy(
                    model=registered_name,
                    props=model.props,
                    options=ProxyOptions(
                        url=url,
                        rewrite_model_to=model.real_model_name,
                        headers={"Authorization": f"Bearer {info.parsed_options.api_key}"},
                    ),
                    registration_options=None,
                )
            stream.emit(StreamChunkProgress(type="progress", stage="install", value=1))
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
        if model.type == "tts":
            self.endpoint_registry.unregister_audio_speech(model.registered_name, model.registration_id)
        if model.type == "stt":
            self.endpoint_registry.unregister_audio_transcriptions(model.registered_name, model.registration_id)
        if model.type == "txt2img":
            self.endpoint_registry.unregister_image_generations(model.registered_name, model.registration_id)
        if model.type == "embedding":
            self.endpoint_registry.unregister_embeddings(model.registered_name, model.registration_id)

        if options.purge:
            # unsupported
            pass
