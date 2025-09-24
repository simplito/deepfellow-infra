"""Ollama service."""

import json

from fastapi import HTTPException
from pydantic import BaseModel

from server.applicationcontext import get_base_url, get_container_host, get_container_port
from server.config import get_main_dir
from server.docker import DockerOptions, install_and_run_docker, uninstall_docker
from server.endpointregistry import ProxyOptions, RegistrationId
from server.models.models import InstallModelIn, ListModelsFilters, ListModelsOut, RetrieveModelOut, UninstallModelIn
from server.models.services import InstallServiceIn, ServiceField, ServiceOptions, ServiceSpecification, UninstallServiceIn
from server.services.base2_service import Base2Service, ModelConfig, ServiceConfig
from server.utils.core import fetch_from


def _read_models_from_json() -> dict[str, bool]:  # pyright: ignore[reportUnusedFunction]
    ollama_path = get_main_dir() / "./static/ollama.json"
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
    ollama_path = get_main_dir() / "./static/ollama-min.json"
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
    registration_id: RegistrationId


class OllamaOptions(BaseModel):
    gpu: bool
    num_parallel: int = 3


class InstalledInfo:
    def __init__(
        self,
        docker: DockerOptions,
        models: dict[str, ModelInstalledInfo],
        options: InstallServiceIn,
        parsed_options: OllamaOptions,
        container_host: str,
        container_port: int,
        docker_exposed_port: int,
    ):
        self.docker = docker
        self.models = models
        self.options = options
        self.parsed_options = parsed_options
        self.container_host = container_host
        self.container_port = container_port
        self.docker_exposed_port = docker_exposed_port
        self.base_url = get_base_url(self.container_host, self.container_port)


class OllamaService(Base2Service[InstalledInfo]):
    def get_id(self) -> str:
        """Return the service id."""
        return "ollama"

    def get_spec(self) -> ServiceSpecification:
        """Return the service specification."""
        return ServiceSpecification(
            fields=[
                ServiceField(type="bool", name="gpu", description="Run on GPU"),
                ServiceField(
                    type="number",
                    name="num_parallel",
                    description="How many copies of one model can be loaded (OLLAMA_NUM_PARALLEL)",
                    default="3",
                ),
            ]
        )

    def get_installed_info(self) -> bool | ServiceOptions:
        """Get service installed info."""
        return False if self.installed is None else self.installed.options.spec

    def _generate_config(self, info: InstalledInfo) -> ServiceConfig:
        return ServiceConfig(options=info.options, models=[ModelConfig(model_id=x.id, options=x.options) for x in info.models.values()])

    async def _install_core(self, options: InstallServiceIn) -> InstalledInfo:
        parsed_options = OllamaOptions(**options.spec)
        volumes = [f"{self._get_working_dir()}/main:/root/.ollama"]

        subnet = self.application_context.get_docker_subnet()
        docker_options = DockerOptions(
            name="ollama",
            image=_const.image,
            image_port=11434,
            use_gpu=parsed_options.gpu,
            volumes=volumes,
            env_vars={
                "OLLAMA_NUM_PARALLEL": str(parsed_options.num_parallel),
            },
            restart="unless-stopped",
            subnet=subnet,
        )
        docker_exposed_port = await install_and_run_docker(self.application_context, docker_options)
        return InstalledInfo(
            docker=docker_options,
            models={},
            options=options,
            parsed_options=parsed_options,
            container_host=get_container_host(subnet, docker_options.name),
            container_port=get_container_port(subnet, docker_exposed_port, docker_options.image_port),
            docker_exposed_port=docker_exposed_port,
        )

    async def _uninstall(self, options: UninstallServiceIn) -> None:
        info = self._check_installed()
        for model in info.models.copy().values():
            if model.type == "llm":
                self.endpoint_registry.unregister_chat_completion(model.registered_name, model.registration_id)
            if model.type == "embedding":
                self.endpoint_registry.unregister_embeddings(model.registered_name, model.registration_id)
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
        res = await fetch_from(f"{info.base_url}/api/pull", "POST", {"model": model_id})
        if res.status_code != 200 and res.status_code != 201:
            print("Error when install model in ollama", model_id, res.status_code, res.data)
            raise HTTPException(status_code=400, detail="Model not avaialble")
        registered_name = options.alias if options.alias is not None else model_id
        info.models[model_id] = model_info = ModelInstalledInfo(
            id=model_id,
            type=model_type,
            registered_name=registered_name,
            options=options,
            registration_id="",
        )
        if model_type == "llm":
            model_info.registration_id = self.endpoint_registry.register_chat_completion_as_proxy(
                model=registered_name,
                chat_completions=ProxyOptions(url=f"{info.base_url}/v1/chat/completions", rewrite_model_to=model_id),
                completions=ProxyOptions(url=f"{info.base_url}/v1/completions", rewrite_model_to=model_id),
            )
        if model_type == "embedding":
            model_info.registration_id = self.endpoint_registry.register_embeddings_as_proxy(
                registered_name, ProxyOptions(url=f"{info.base_url}/v1/embeddings")
            )

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
