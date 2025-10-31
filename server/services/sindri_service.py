# DeepFellow Software Framework.
# Copyright © 2025 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sindri service."""

from pathlib import Path

from fastapi import HTTPException
from pydantic import BaseModel

from server.applicationcontext import get_base_url, get_container_host, get_container_port
from server.docker import DockerImage, DockerOptions, install_and_run_docker, uninstall_docker
from server.endpointregistry import ProxyOptions, RegistrationId
from server.models.api import ModelProps
from server.models.models import (
    CustomModelSpecification,
    InstallModelIn,
    ListModelsFilters,
    ListModelsOut,
    ModelField,
    ModelSpecification,
    RetrieveModelOut,
    UninstallModelIn,
)
from server.models.services import InstallServiceIn, ServiceField, ServiceOptions, ServiceSize, ServiceSpecification, UninstallServiceIn
from server.services.base2_service import Base2Service, ModelConfig, ServiceConfig
from server.utils.core import try_parse_pydantic


class SindriAiModel(BaseModel):
    type: str
    real_model_name: str


class SindriAiConst(BaseModel):
    image: DockerImage
    models: dict[str, SindriAiModel]


_const = SindriAiConst(
    image=DockerImage(name="sindrilabs/evllm-proxy:v0.0.8", size="0.1 GB"),
    models={
        "gemma3:27b": SindriAiModel(type="llm", real_model_name="gemma3"),
    },
)


class ModelInstalledInfo(BaseModel):
    id: str
    registered_name: str
    type: str
    options: InstallModelIn
    registration_id: RegistrationId


class SindriOptions(BaseModel):
    api_url: str
    api_key: str


class SindriModelOptions(BaseModel):
    alias: str | None = None


class InstalledInfo:
    def __init__(
        self,
        docker: DockerOptions,
        models: dict[str, ModelInstalledInfo],
        options: InstallServiceIn,
        parsed_options: SindriOptions,
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


class SindriService(Base2Service[InstalledInfo]):
    def get_id(self) -> str:
        """Return the service id."""
        return "sindri"

    def get_size(self) -> ServiceSize:
        """Return the service size."""
        return _const.image.size

    def get_description(self) -> str:
        """Return the service description."""
        return "Remote encrypted access to Sindri models."

    def get_spec(self) -> ServiceSpecification:
        """Return the service specification."""
        return ServiceSpecification(
            fields=[
                ServiceField(type="text", name="api_url", description="API URL", default="https://sindri.app/api/ai/v1/openai"),
                ServiceField(type="password", name="api_key", description="API Key"),
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
        return None

    def get_installed_info(self) -> bool | ServiceOptions:
        """Get service installed info."""
        return False if self.installed is None else self.installed.options.spec

    def _generate_config(self, info: InstalledInfo | None) -> ServiceConfig:
        return ServiceConfig(
            options=info.options if info else None,
            models=[ModelConfig(model_id=x.id, options=x.options) for x in info.models.values()] if info else [],
            custom=self.custom,
        )

    async def _install_core(self, options: InstallServiceIn) -> InstalledInfo:
        parsed_options = try_parse_pydantic(SindriOptions, options.spec)
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
        subnet = self.application_context.get_docker_subnet()
        docker_options = DockerOptions(
            name="sindri",
            container_name=self.application_context.get_docker_container_name("sindri"),
            image=_const.image.name,
            command="serve /config.yaml",
            image_port=8080,
            volumes=volumes,
            restart="unless-stopped",
            subnet=subnet,
            healthcheck={
                "test": "wget -q --spider http://localhost:8080",
                "interval": "30s",
                "timeout": "10s",
                "retries": "3",
                "start_period": "5s",
            },
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
        self.installed = None
        await uninstall_docker(self.application_context, info.docker)
        if options.purge:
            await self._clear_working_dir()

    def get_docker_compose_file_path(self, model_id: str | None) -> Path:
        """Get docker compose file path."""
        info = self.installed
        if not info:
            raise HTTPException(400, "Service not installed")
        if model_id:
            raise HTTPException(400, "Docker is not bound with this object")
        return self.application_context.get_docker_compose_file_path(info.docker.name)

    def service_has_docker(self) -> bool:
        """Return true when docker is started when service is installed."""
        return True

    async def list_models(self, filters: ListModelsFilters) -> ListModelsOut:
        """List models."""
        info = self._check_installed()
        out_list: list[RetrieveModelOut] = []
        for model_id, model in _const.models.items():
            installed = info.models[model_id].options if model_id in info.models else False
            if filters.installed is None or filters.installed == installed:
                out_list.append(
                    RetrieveModelOut(
                        id=model_id,
                        service=self.get_id(),
                        type=model.type,
                        installed=installed,
                        size="",
                        spec=self.get_model_spec(),
                        has_docker=False,
                    )
                )
        return ListModelsOut(list=out_list)

    async def get_model(self, model_id: str) -> RetrieveModelOut:
        """Get the model."""
        info = self._check_installed()
        if model_id not in _const.models:
            raise HTTPException(status_code=400, detail="Model not found")
        model = _const.models[model_id]
        installed = info.models[model_id].options if model_id in info.models else False
        return RetrieveModelOut(
            id=model_id,
            service=self.get_id(),
            type=model.type,
            installed=installed,
            size="",
            spec=self.get_model_spec(),
            has_docker=False,
        )

    async def _install_model(self, model_id: str, options: InstallModelIn) -> None:
        parsed_model_options = try_parse_pydantic(SindriModelOptions, options.spec) if options.spec else SindriModelOptions()
        info = self._check_installed()
        if model_id in info.models:
            return
        if model_id not in _const.models:
            raise HTTPException(status_code=400, detail="Model not found")
        model = _const.models[model_id]
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
                chat_completions=ProxyOptions(url=f"{info.base_url}/v1/chat/completions", rewrite_model_to=model.real_model_name),
                completions=None,
                registration_options=None,
            )

    async def _uninstall_model(self, model_id: str, options: UninstallModelIn) -> None:
        info = self._check_installed()
        if model_id not in info.models:
            return
        model = info.models[model_id]
        del info.models[model_id]
        if model.type == "llm":
            self.endpoint_registry.unregister_chat_completion(model.registered_name, model.registration_id)

        if options.purge:
            # unsupported
            pass
