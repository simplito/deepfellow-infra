# DeepFellow Software Framework.
# Copyright © 2025 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ollama external service."""

import json
from typing import Annotated, Literal

from fastapi import HTTPException
from pydantic import BaseModel, StringConstraints

from server.config import get_main_dir
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
    ModelSpecification,
    RetrieveModelOut,
    UninstallModelIn,
)
from server.models.services import InstallServiceIn, ServiceField, ServiceOptions, ServiceSpecification, UninstallServiceIn
from server.services.base2_service import Base2Service, CustomModel, ModelConfig, ServiceConfig
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


class ModelInstalledInfo(BaseModel):
    id: str
    registered_name: str
    type: str
    options: InstallModelIn
    registration_id: RegistrationId


class OllamaExternalOptions(BaseModel):
    url: str = "http://localhost:11434"


class OllamaModelOptions(BaseModel):
    alias: str | None = None
    alive_time: int | Annotated[str, StringConstraints(pattern=r"^(\d+[smh])?$", strict=True)] = ""


class InstalledExternalInfo:
    def __init__(
        self,
        models: dict[str, ModelInstalledInfo],
        options: InstallServiceIn,
        parsed_options: OllamaExternalOptions,
    ):
        self.models = models
        self.options = options
        self.parsed_options = parsed_options
        self.base_url = parsed_options.url.rstrip("/")


class OllamaExternalService(Base2Service[InstalledExternalInfo]):
    models: dict[str, OllamaModel]

    def _after_init(self) -> None:
        self.models = _const.models.copy()

    def get_id(self) -> str:
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

    def get_installed_info(self) -> bool | ServiceOptions:
        """Get service installed info."""
        return False if self.installed is None else self.installed.options.spec

    def _generate_config(self, info: InstalledExternalInfo | None) -> ServiceConfig:
        return ServiceConfig(
            options=info.options if info else None,
            models=[ModelConfig(model_id=x.id, options=x.options) for x in info.models.values()] if info else [],
            custom=self.custom,
        )

    def service_has_docker(self) -> bool:
        """Return true when docker is started when service is installed."""
        return False

    async def stop(self) -> None:
        """Stop the service gracefully.

        External Ollama service has no containers to stop.
        """

    def _add_custom_model(self, model: CustomModel) -> None:
        parsed = try_parse_pydantic(OllamaCustomModel, model.data)
        if parsed.id in self.models:
            raise HTTPException(400, "Model with given id already exists.")
        self.models[parsed.id] = OllamaModel(id=parsed.id, size=parsed.size, type=parsed.type, custom=model.id)

    def _remove_custom_model(self, model: CustomModel) -> None:
        parsed = try_parse_pydantic(OllamaCustomModel, model.data)
        if self.installed and parsed.id in self.installed.models:
            raise HTTPException(400, "Cannot remove custom model, it is in use, uninstall it first.")
        del self.models[parsed.id]

    async def _install_core(self, options: InstallServiceIn) -> PromiseWithProgress[InstalledExternalInfo, StreamChunk]:
        parsed_options = try_parse_pydantic(OllamaExternalOptions, options.spec)

        def _raise_connection_error(url: str) -> None:
            msg = f"Cannot connect to Ollama at {url}"
            raise HTTPException(status_code=400, detail=msg)

        async def func(stream: Stream[StreamChunk]) -> InstalledExternalInfo:
            # Verify connection to external Ollama
            try:
                res = await fetch_from(f"{parsed_options.url.rstrip('/')}/api/tags", "GET", None)
                if res.status_code != 200:
                    _raise_connection_error(parsed_options.url)
            except HTTPException:
                raise
            except Exception as e:
                msg = f"Cannot connect to Ollama at {parsed_options.url}: {e!s}"
                raise HTTPException(status_code=400, detail=msg) from e

            stream.emit(StreamChunkProgress(type="progress", value=1))
            return InstalledExternalInfo(
                models={},
                options=options,
                parsed_options=parsed_options,
            )

        return PromiseWithProgress(func=func)

    async def _uninstall(self, options: UninstallServiceIn) -> None:
        info = self._check_installed()
        for model in info.models.copy().values():
            if model.type == "llm":
                self.endpoint_registry.unregister_chat_completion(model.registered_name, model.registration_id)
            if model.type == "embedding":
                self.endpoint_registry.unregister_embeddings(model.registered_name, model.registration_id)
        self.installed = None
        if options.purge:
            await self._clear_working_dir()

    async def list_models(self, filters: ListModelsFilters) -> ListModelsOut:
        """List models."""
        info = self._check_installed()
        out_list: list[RetrieveModelOut] = []
        for model_id, model in self.models.items():
            installed = info.models[model_id].options if model_id in info.models else False
            if filters.installed is None or filters.installed == installed:
                out_list.append(
                    RetrieveModelOut(
                        id=model_id,
                        service=self.get_id(),
                        type=model.type,
                        installed=installed,
                        size=model.size,
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
        installed = info.models[model_id].options if model_id in info.models else False
        return RetrieveModelOut(
            id=model_id,
            service=self.get_id(),
            type=model.type,
            installed=installed,
            size=model.size,
            custom=model.custom,
            spec=self.get_model_spec(),
            has_docker=False,
        )

    async def _install_model(self, model_id: str, options: InstallModelIn) -> PromiseWithProgress[InstallModelOut, StreamChunk]:  # noqa: C901
        parsed_model_options = try_parse_pydantic(OllamaModelOptions, options.spec) if options.spec else OllamaModelOptions()
        info = self._check_installed()
        if model_id in info.models:
            return PromiseWithProgress(value=InstallModelOut(status="OK", details="Already installed"))
        if model_id not in self.models:
            raise HTTPException(400, "Model not found")
        model = self.models[model_id]

        async def func(streamp: Stream[StreamChunk]) -> InstallModelOut:
            progress = Progress(convert_size_to_bytes(model.size) or 0)
            last_diggest: str = ""
            last_value: int = 0

            async for stream in stream_fetch_from(f"{info.base_url}/api/pull", "POST", {"model": model_id}):
                if (stream.status_code != 200 and stream.status_code != 201) or "error" in stream.data:
                    raise HTTPException(400, "Model not available")

                data_cleared: list[str] = stream.data.rstrip().split("\n")
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

                        streamp.emit(StreamChunkProgress(type="progress", value=progress.get_percentage() * 0.99))

            streamp.emit(StreamChunkProgress(type="progress", value=0.99))
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
                    props=ModelProps(private=True),
                    chat_completions=ProxyOptions(url=f"{info.base_url}/v1/chat/completions", rewrite_model_to=model_id),
                    completions=ProxyOptions(url=f"{info.base_url}/v1/completions", rewrite_model_to=model_id),
                    registration_options=None,
                )
            if model.type == "embedding":
                model_info.registration_id = self.endpoint_registry.register_embeddings_as_proxy(
                    model=registered_name,
                    props=ModelProps(private=True),
                    options=ProxyOptions(url=f"{info.base_url}/v1/embeddings", rewrite_model_to=model_id),
                    registration_options=None,
                )
            streamp.emit(StreamChunkProgress(type="progress", value=1))
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
        if model.type == "embedding":
            self.endpoint_registry.unregister_embeddings(model.registered_name, model.registration_id)

        if options.purge:
            await fetch_from(f"{info.base_url}/api/delete", "DELETE", {"name": model_id})
