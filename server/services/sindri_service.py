"""Sindri service."""

from fastapi import HTTPException
from pydantic import BaseModel

from server.docker import DockerOptions, install_and_run_docker, uninstall_docker
from server.endpointregistry import ProxyOptions
from server.models.models import InstallModelIn, ListModelsFilters, ListModelsOut, RetrieveModelOut, UninstallModelIn
from server.models.services import InstallServiceIn, ServiceField, ServiceOptions, ServiceSpecification, UninstallServiceIn
from server.services.base2_service import Base2Service, ModelConfig, ServiceConfig


class SindriAiModel(BaseModel):
    type: str
    real_model_name: str


class SindriAiConst(BaseModel):
    image: str
    models: dict[str, SindriAiModel]


_const = SindriAiConst(
    image="sindrilabs/evllm-proxy:v0.0.8",
    models={
        "gemma3:27b": SindriAiModel(type="llm", real_model_name="gemma3"),
    },
)


class ModelInstalledInfo(BaseModel):
    id: str
    registered_name: str
    type: str
    options: InstallModelIn


class SindriOptions(BaseModel):
    api_url: str
    api_key: str


class InstalledInfo:
    def __init__(
        self,
        docker: DockerOptions,
        container_host: str,
        port: int,
        models: dict[str, ModelInstalledInfo],
        options: InstallServiceIn,
        parsed_options: SindriOptions,
    ):
        self.docker = docker
        self.container_host = container_host
        self.port = port
        self.models = models
        self.options = options
        self.parsed_options = parsed_options


class SindriService(Base2Service[InstalledInfo]):
    def get_id(self) -> str:
        """Return the service id."""
        return "sindri"

    def get_spec(self) -> ServiceSpecification:
        """Return the service specification."""
        return ServiceSpecification(
            fields=[
                ServiceField(type="text", name="api_url", description="API URL", default="https://sindri.app/api/ai/v1/openai"),
                ServiceField(type="password", name="api_key", description="API Key"),
            ]
        )

    def get_installed_info(self) -> bool | ServiceOptions:
        """Get service installed info."""
        return False if self.installed is None else self.installed.options.spec

    def _generate_config(self, info: InstalledInfo) -> ServiceConfig:
        return ServiceConfig(options=info.options, models=[ModelConfig(model_id=x.id, options=x.options) for x in info.models.values()])

    async def _install_core(self, options: InstallServiceIn) -> InstalledInfo:
        parsed_options = SindriOptions(**options.spec)
        config_path = self._get_working_dir() / "config.yaml"
        data = f"""listenAddress: 0.0.0.0
listenPort: 8080
appMode: release

sindriClient:
  baseURL: {parsed_options.api_url}
  apiKey: {parsed_options.api_key}
  requestTimeoutSeconds: 10
  
  encryption:
    enabled: true
    keySource: ephemeral  # or 'value' or 'file'
    # Configure keys if not using ephemeral
"""
        with config_path.open("w", encoding="utf-8") as f:
            f.write(data)
        volumes = [f"{config_path}:/config.yaml"]
        docker_options = DockerOptions(
            name="sindri",
            image=_const.image,
            command="serve /config.yaml",
            image_port=8080,
            volumes=volumes,
            restart="unless-stopped",
        )
        port = await install_and_run_docker(self.application_context, docker_options)
        return InstalledInfo(
            docker=docker_options,
            container_host=self.application_context.get_container_host(docker_options.name),
            port=port,
            models={},
            options=options,
            parsed_options=parsed_options,
        )

    async def _uninstall(self, options: UninstallServiceIn) -> None:
        info = self._check_installed()
        for model in info.models.values():
            if model.type == "llm":
                self.endpoint_registry.unregister_chat_completion(model.registered_name)
        self.installed = None
        await uninstall_docker(self.application_context, info.docker)
        if options.purge:
            await self._clear_working_dir()

    async def list_models(self, filters: ListModelsFilters) -> ListModelsOut:
        """List models."""
        info = self._check_installed()
        out_list: list[RetrieveModelOut] = []
        for model_id, model in _const.models.items():
            installed = model_id in info.models
            if filters.installed is None or filters.installed == installed:
                out_list.append(RetrieveModelOut(id=model_id, service=self.get_id(), type=model.type, installed=installed))
        return ListModelsOut(list=out_list)

    async def get_model(self, model_id: str) -> RetrieveModelOut:
        """Get the model."""
        info = self._check_installed()
        if model_id not in _const.models:
            raise HTTPException(status_code=400, detail="Model not found")
        model = _const.models[model_id]
        installed = model_id in info.models
        return RetrieveModelOut(id=model_id, service=self.get_id(), type=model.type, installed=installed)

    async def _install_model(self, model_id: str, options: InstallModelIn) -> None:
        info = self._check_installed()
        if model_id in info.models:
            return
        if model_id not in _const.models:
            raise HTTPException(status_code=400, detail="Model not found")
        model = _const.models[model_id]
        registered_name = options.alias if options.alias is not None else model_id
        info.models[model_id] = ModelInstalledInfo(id=model_id, type=model.type, registered_name=registered_name, options=options)
        if model.type == "llm":
            self.endpoint_registry.register_chat_completion_as_proxy(
                registered_name,
                ProxyOptions(url=f"http://{info.container_host}:{info.port}/v1/chat/completions", rewrite_model_to=model.real_model_name),
            )

    async def _uninstall_model(self, model_id: str, options: UninstallModelIn) -> None:
        info = self._check_installed()
        if model_id not in info.models:
            return
        model = info.models[model_id]
        del info.models[model_id]
        if model.type == "llm":
            self.endpoint_registry.unregister_chat_completion(model.registered_name)

        if options.purge:
            # unsupported
            pass
