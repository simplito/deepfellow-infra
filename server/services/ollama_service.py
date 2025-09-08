"""Ollama service."""

import json
from pathlib import Path
from typing import Any

import aiohttp
from fastapi import HTTPException
from pydantic import BaseModel

from server.docker import DockerOptions, install_and_run_docker, uninstall_docker
from server.endpointregistry import ProxyOptions
from server.models.models import InstallModelIn, ListModelsFilters, ListModelsOut, RetrieveModelOut, UninstallModelIn
from server.models.services import InstallServiceIn, UninstallServiceIn
from server.services.base2_service import Base2Service, ModelConfig, ServiceConfig


def _read_models_from_json() -> dict[str, bool]:  # pyright: ignore[reportUnusedFunction]
    ollama_path = Path(__file__).parent.parent.parent / "./static/ollama.json"
    with ollama_path.open(encoding="utf-8") as f:
        data = json.loads(f.read())
        # tags = [x["tags"] for x in data["list"]]
        # flat: list[str] = [x["tag"] for sublist in tags for x in sublist]
        tags = [x["mainTags"] for x in data["list"]]
        flat: list[str] = [x.removesuffix(":latest") for sublist in tags for x in sublist]

        map: dict[str, bool] = {}
        for tag in flat:
            map[tag] = True
        return map


def _read_models() -> dict[str, str]:
    ollama_path = Path(__file__).parent.parent.parent / "./static/ollama-min.json"
    with ollama_path.open(encoding="utf-8") as f:
        registry = json.loads(f.read())
        map: dict[str, str] = {}
        for tag in registry["llms"]:
            map[tag] = "llm"
        for tag in registry["embeddings"]:
            map[tag] = "embedding"
        return map


class OllamaAiConst(BaseModel):
    image: str
    models: dict[str, str]


_const = OllamaAiConst(
    image="ollama/ollama",
    models=_read_models(),
)


class ModelInstalledInfo(BaseModel):
    id: str
    registered_name: str
    type: str
    options: InstallModelIn


class InstalledInfo:
    def __init__(
        self,
        docker: DockerOptions,
        port: int,
        models: dict[str, ModelInstalledInfo],
        options: InstallServiceIn,
    ):
        self.docker = docker
        self.port = port
        self.models = models
        self.options = options


class FetchResult(BaseModel):
    status_code: int
    data: Any


class OllamaService(Base2Service[InstalledInfo]):
    def get_id(self) -> str:
        """Return the service id."""
        return "ollama"

    def _generate_config(self, info: InstalledInfo) -> ServiceConfig:
        return ServiceConfig(options=info.options, models=[ModelConfig(model_id=x.id, options=x.options) for x in info.models.values()])

    async def _install_core(self, options: InstallServiceIn) -> InstalledInfo:
        volumes = [f"{self._get_working_dir()}/main:/root/.ollama"]

        docker_options = DockerOptions(
            name="ollama",
            image=_const.image,
            image_port=11434,
            use_gpu=options.gpu,
            volumes=volumes,
            restart="unless-stopped",
        )
        port = await install_and_run_docker(self.application_context, docker_options)
        return InstalledInfo(docker=docker_options, port=port, models={}, options=options)

    async def _uninstall(self, options: UninstallServiceIn) -> None:
        info = self._check_installed()
        for model in info.models.values():
            if model.type == "llm":
                self.endpoint_registry.unregister_chat_completion(model.registered_name)
            if model.type == "embedding":
                self.endpoint_registry.unregister_embeddings(model.registered_name)
        self.installed = None
        await uninstall_docker(self.application_context, info.docker)
        if options.purge:
            await self._clear_working_dir()

    async def list_models(self, filters: ListModelsFilters) -> ListModelsOut:
        """List models."""
        info = self._check_installed()
        out_list: list[RetrieveModelOut] = []
        for model_id, model_type in _const.models.items():
            installed = model_id in info.models
            if filters.installed is None or filters.installed == installed:
                out_list.append(RetrieveModelOut(id=model_id, service=self.get_id(), type=model_type, installed=installed))
        return ListModelsOut(list=out_list)

    async def get_model(self, model_id: str) -> RetrieveModelOut:
        """Get the model."""
        info = self._check_installed()
        if model_id not in _const.models:
            raise HTTPException(status_code=400, detail="Model not found")
        model_type = _const.models[model_id]
        installed = model_id in info.models
        return RetrieveModelOut(id=model_id, service=self.get_id(), type=model_type, installed=installed)

    async def _install_model(self, model_id: str, options: InstallModelIn) -> None:
        info = self._check_installed()
        if model_id in info.models:
            return
        if model_id not in _const.models:
            raise HTTPException(status_code=400, detail="Model not found")
        model_type = _const.models[model_id]
        res = await self._fetch(info.port, "/api/pull", "POST", {"model": model_id})
        if res.status_code != 200 and res.status_code != 201:
            print("Error when install model in ollama", model_id, res.status_code, res.data)
            raise HTTPException(status_code=400, detail="Model not avaialble")
        registered_name = options.alias if options.alias is not None else model_id
        info.models[model_id] = ModelInstalledInfo(id=model_id, type=model_type, registered_name=registered_name, options=options)
        if model_type == "llm":
            self.endpoint_registry.register_chat_completion_as_proxy(
                registered_name, ProxyOptions(url=f"http://localhost:{info.port}/v1/chat/completions")
            )
        if model_type == "embedding":
            self.endpoint_registry.register_embeddings_as_proxy(
                registered_name, ProxyOptions(url=f"http://localhost:{info.port}/v1/embeddings")
            )

    async def _uninstall_model(self, model_id: str, options: UninstallModelIn) -> None:
        info = self._check_installed()
        if model_id not in info.models:
            return
        model = info.models[model_id]
        del info.models[model_id]
        if model.type == "llm":
            self.endpoint_registry.unregister_chat_completion(model.registered_name)
        if model.type == "embedding":
            self.endpoint_registry.unregister_embeddings(model.registered_name)

        if options.purge:
            await self._fetch(info.port, "/api/delete", "DELETE", {"name": model_id})

    async def _fetch(self, port: int, url: str, method: str = "GET", data: dict | None = None) -> FetchResult:
        full_url = f"http://localhost:{port}{url}"
        async with aiohttp.ClientSession() as session, session.request(method, full_url, json=data) as response:
            return FetchResult(status_code=response.status, data=await response.text())
