# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ollama external service."""

import json
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from fastapi import HTTPException
from packaging import version
from pydantic import BaseModel, StringConstraints

from server.config import get_main_dir
from server.endpointregistry import ProxyOptions, RegistrationId
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


class OllamaModel(BaseModel):
    id: str
    size: str
    type: str
    custom: CustomModelId | None = None


class OllamaCustomModel(BaseModel):
    id: str
    size: str
    type: Literal["llm", "embedding"]


class OllamaRegistryEntry(BaseModel):
    name: str
    size: str


class OllamaRegistry(BaseModel):
    llms: list[OllamaRegistryEntry]
    embeddings: list[OllamaRegistryEntry]


def _read_models() -> dict[str, OllamaModel]:
    ollama_path = get_main_dir() / "./static/ollama-min.json"
    with ollama_path.open(encoding="utf-8") as f:
        registry_data = json.loads(f.read())
        registry = OllamaRegistry(llms=registry_data["llms"], embeddings=registry_data["embeddings"])
        map: dict[str, OllamaModel] = {}
        for tag in registry.llms:
            map[tag.name] = OllamaModel(id=tag.name, size=tag.size, type="llm")
        for tag in registry.embeddings:
            map[tag.name] = OllamaModel(id=tag.name, size=tag.size, type="embedding")
        return map


class OllamaAiConst(BaseModel):
    models: dict[str, OllamaModel]


_const = OllamaAiConst(
    models=_read_models(),
)


@dataclass
class ModelInstalledInfo:
    id: str
    registered_name: str
    type: str
    options: InstallModelIn
    registration_id: RegistrationId

    def get_info(self) -> ModelInfo:
        """Get info."""
        return ModelInfo(spec=self.options.spec, registration_id=self.registration_id)


class OllamaExternalOptions(BaseModel):
    url: str = "http://localhost:11434"


class OllamaModelOptions(BaseModel):
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

    def _after_init(self) -> None:
        self.models = {}
        self.load_default_models("default")

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
            ]
        )

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
            ]
        )

    def get_custom_model_spec(self) -> CustomModelSpecification | None:
        """Return the custom model specification or None if custom model is not supported."""
        return CustomModelSpecification(
            fields=[
                CustomModelField(type="text", name="id", description="Model ID", placeholder="my-custom-model"),
                CustomModelField(type="text", name="size", description="Model size", placeholder="1GB"),
                CustomModelField(type="oneof", name="type", description="Model type", values=["llm", "embedding"]),
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

    def service_has_docker(self) -> bool:
        """Return true when docker is started when service is installed."""
        return False

    async def stop_instance(self, instance: str) -> None:
        """Stop the service gracefully.

        External Ollama service has no containers to stop.
        """

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

    async def _install_instance(self, instance: str, options: InstallServiceIn) -> PromiseWithProgress[InstalledInfo, StreamChunk]:
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
                except json.JSONDecodeError as err:
                    raise HTTPException(500, "Ollama external server doesn't return version.") from err
            except HTTPException:
                raise
            except Exception as e:
                msg = f"Cannot connect to Ollama at {parsed_options.url}: {e!s}"
                raise HTTPException(status_code=400, detail=msg) from e

            stream.emit(StreamChunkProgress(type="progress", stage="install", value=1, data={}))
            return InstalledInfo(models={}, options=options, parsed_options=parsed_options, base_url=parsed_options.url.rstrip("/"))

        return PromiseWithProgress(func=func)

    async def _uninstall_instance(self, instance: str, options: UninstallServiceIn) -> None:
        installed = self.get_instance_info(instance).installed
        if installed:
            for model in installed.models.copy().values():
                if model.type == "llm":
                    self.endpoint_registry.unregister_chat_completion(model.registered_name, model.registration_id)
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

    async def _download_model(self, stream: Stream[StreamChunk], model: OllamaModel, model_id: str, base_url: str) -> None:
        progress = Progress(convert_size_to_bytes(model.size) or 0)
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

    async def _download_model_or_set_progress(
        self,
        stream: Stream[StreamChunk],
        model: OllamaModel,
        model_id: str,
        base_url: str,
    ) -> None:
        if model_id not in self.models_download_progress:
            self.models_download_progress[model_id] = stream
            await self._download_model(stream, model, model_id, base_url)
            del self.models_download_progress[model_id]
        else:
            chunk: StreamChunk
            async for chunk in self.models_download_progress[model_id].as_generator():
                if chunk.get("type") == "progress" and chunk.get("stage") == "download":
                    stream.emit(chunk)
                else:
                    break

    async def _install_model(
        self, instance: str, model_id: str, options: InstallModelIn
    ) -> PromiseWithProgress[InstallModelOut, StreamChunk]:
        parsed_model_options = try_parse_pydantic(OllamaModelOptions, options.spec) if options.spec else OllamaModelOptions()
        info = self.get_instance_installed_info(instance)

        if not self.models.get(instance):
            self.models[instance] = {}

        if model_id in info.models:
            return PromiseWithProgress(value=InstallModelOut(status="OK", details="Already installed"))

        if model_id not in self.models[instance]:
            raise HTTPException(400, "Model not found")

        model = self.models[instance][model_id]

        async def func(stream: Stream[StreamChunk]) -> InstallModelOut:
            await self._download_model_or_set_progress(stream, model, model_id, info.base_url)

            stream.emit(StreamChunkProgress(type="progress", stage="install", value=0, data={}))
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
            )
            if model.type == "llm":
                model_info.registration_id = self.endpoint_registry.register_chat_completion_as_proxy(
                    model=registered_name,
                    props=ModelProps(private=True, type="llm", endpoints=self.get_supported_endpoints()),
                    chat_completions=ProxyOptions(url=f"{info.base_url}/v1/chat/completions", rewrite_model_to=model_id),
                    completions=ProxyOptions(url=f"{info.base_url}/v1/completions", rewrite_model_to=model_id),
                    responses=(
                        ProxyOptions(url=f"{info.base_url}/v1/responses", rewrite_model_to=model_id) if self.support_responses else None
                    ),
                    messages=(
                        ProxyOptions(url=f"{info.base_url}/v1/messages", rewrite_model_to=model_id) if self.support_messages else None
                    ),
                    registration_options=None,
                )
            if model.type == "embedding":
                model_info.registration_id = self.endpoint_registry.register_embeddings_as_proxy(
                    model=registered_name,
                    props=ModelProps(private=True, type="embedding", endpoints=EMBEDDINGS_ENDPOINTS),
                    options=ProxyOptions(url=f"{info.base_url}/v1/embeddings", rewrite_model_to=model_id),
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
            if model.type == "embedding":
                self.endpoint_registry.unregister_embeddings(model.registered_name, model.registration_id)

        if options.purge and model_id in self.models_downloaded:
            await fetch_from(f"{info.base_url}/api/delete", "DELETE", {"name": model_id})
            del self.models_downloaded[model_id]

    def get_supported_endpoints(self) -> list[str]:
        """Get supported endpoints."""
        endpoints = ["/v1/completions", "/v1/chat/completions"]
        if self.support_responses:
            endpoints.append("/v1/responses")
        if self.support_messages:
            endpoints.append("/v1/messages")
        return endpoints
