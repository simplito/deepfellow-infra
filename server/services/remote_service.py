"""Remote service."""

from abc import abstractmethod
from urllib.parse import urljoin

from fastapi import HTTPException
from pydantic import BaseModel

from server.endpointregistry import ProxyOptions, RegistrationId
from server.models.models import InstallModelIn, ListModelsFilters, ListModelsOut, RetrieveModelOut, UninstallModelIn
from server.models.services import InstallServiceIn, ServiceField, ServiceOptions, ServiceSpecification, UninstallServiceIn
from server.services.base2_service import Base2Service, ModelConfig, ServiceConfig


class RemoteModel(BaseModel):
    type: str
    real_model_name: str | None = None
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
    alternative_registration_id: RegistrationId


class RemoteOptions(BaseModel):
    api_url: str
    api_key: str


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
                ServiceField(type="text", name="api_url", description="API URL", default=self.get_default_url()),
                ServiceField(type="password", name="api_key", description="API Key"),
            ]
        )

    def get_installed_info(self) -> bool | ServiceOptions:
        """Get service installed info."""
        return False if self.installed is None else self.installed.options.spec

    def _generate_config(self, info: InstalledInfo) -> ServiceConfig:
        return ServiceConfig(options=info.options, models=[ModelConfig(model_id=x.id, options=x.options) for x in info.models.values()])

    async def _install_core(self, options: InstallServiceIn) -> InstalledInfo:
        parsed_options = RemoteOptions(**options.spec)
        return InstalledInfo(models={}, options=options, parsed_options=parsed_options)

    async def _uninstall(self, options: UninstallServiceIn) -> None:
        info = self._check_installed()
        for model in info.models.copy().values():
            if model.type == "llm":
                if model.completions:
                    self.endpoint_registry.unregister_chat_completion(model.registered_name, model.registration_id)
                if model.legacy_completions:
                    self.endpoint_registry.unregister_completion(model.registered_name, model.alternative_registration_id)
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

    async def list_models(self, filters: ListModelsFilters) -> ListModelsOut:
        """List models."""
        info = self._check_installed()
        out_list: list[RetrieveModelOut] = []
        _const = self.get_models_registry()
        for model_id, model in _const.models.items():
            installed = model_id in info.models
            if filters.installed is None or filters.installed == installed:
                out_list.append(RetrieveModelOut(id=model_id, service=self.get_id(), type=model.type, installed=installed))
        return ListModelsOut(list=out_list)

    async def get_model(self, model_id: str) -> RetrieveModelOut:
        """Get the model."""
        info = self._check_installed()
        _const = self.get_models_registry()
        if model_id not in _const.models:
            raise HTTPException(status_code=400, detail="Model not found")
        model = _const.models[model_id]
        installed = model_id in info.models
        return RetrieveModelOut(id=model_id, service=self.get_id(), type=model.type, installed=installed)

    async def _install_model(self, model_id: str, options: InstallModelIn) -> None:
        info = self._check_installed()
        if model_id in info.models:
            return
        _const = self.get_models_registry()
        if model_id not in _const.models:
            raise HTTPException(status_code=400, detail="Model not found")
        model = _const.models[model_id]
        registered_name = options.alias if options.alias is not None else model_id
        info.models[model_id] = model_info = ModelInstalledInfo(
            id=model_id,
            type=model.type,
            registered_name=registered_name,
            options=options,
            completions=model.completions,
            legacy_completions=model.legacy_completions,
            registration_id="",
            alternative_registration_id="",
        )
        url_base = urljoin(info.parsed_options.api_url, self.url_prefix)
        if model.type == "llm":
            if model.completions:
                url = urljoin(url_base, "chat/completions")
                model_info.registration_id = self.endpoint_registry.register_chat_completion_as_proxy(
                    registered_name,
                    ProxyOptions(
                        url=url,
                        rewrite_model_to=model.real_model_name or model_id,
                        headers={"Authorization": f"Bearer {info.parsed_options.api_key}"},
                    ),
                )
            if model.legacy_completions:
                url = urljoin(url_base, "completions")
                model_info.alternative_registration_id = self.endpoint_registry.register_completion_as_proxy(
                    registered_name,
                    ProxyOptions(
                        url=url,
                        rewrite_model_to=model.real_model_name,
                        headers={"Authorization": f"Bearer {info.parsed_options.api_key}"},
                    ),
                )
        if model.type == "tts":
            url = urljoin(url_base, "v1/audio/speech")
            model_info.registration_id = self.endpoint_registry.register_audio_speech_as_proxy(
                registered_name,
                ProxyOptions(
                    url=url,
                    rewrite_model_to=model.real_model_name,
                    headers={"Authorization": f"Bearer {info.parsed_options.api_key}"},
                ),
            )
        if model.type == "stt":
            url = urljoin(url_base, "v1/audio/transcriptions")
            model_info.registration_id = self.endpoint_registry.register_audio_transcriptions_as_proxy(
                registered_name,
                ProxyOptions(
                    url=url,
                    rewrite_model_to=model.real_model_name,
                    headers={"Authorization": f"Bearer {info.parsed_options.api_key}"},
                ),
            )
        if model.type == "txt2img":
            url = urljoin(url_base, "v1/images/generations")
            model_info.registration_id = self.endpoint_registry.register_image_generations_as_proxy(
                registered_name,
                ProxyOptions(
                    url=url,
                    rewrite_model_to=model.real_model_name,
                    headers={"Authorization": f"Bearer {info.parsed_options.api_key}"},
                ),
            )
        if model.type == "embedding":
            url = urljoin(url_base, "v1/embeddings")
            model_info.registration_id = self.endpoint_registry.register_embeddings_as_proxy(
                registered_name,
                ProxyOptions(
                    url=url,
                    rewrite_model_to=model.real_model_name,
                    headers={"Authorization": f"Bearer {info.parsed_options.api_key}"},
                ),
            )

    async def _uninstall_model(self, model_id: str, options: UninstallModelIn) -> None:
        info = self._check_installed()
        if model_id not in info.models:
            return
        model = info.models[model_id]
        del info.models[model_id]
        if model.type == "llm":
            if model.completions:
                self.endpoint_registry.unregister_chat_completion(model.registered_name, model.registration_id)
            if model.legacy_completions:
                self.endpoint_registry.unregister_completion(model.registered_name, model.alternative_registration_id)
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
