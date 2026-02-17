# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sindri service."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel

from server.applicationcontext import get_base_url
from server.docker import DockerImage, DockerOptions
from server.endpointregistry import ProxyOptions, RegistrationId
from server.models.api import ModelProps
from server.models.models import (
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


class SindriOptions(BaseModel):
    api_url: str = "https://sindri.app/api/ai/v1/openai"
    api_key: str


class SindriModelOptions(BaseModel):
    alias: str | None = None


@dataclass
class InstalledInfo:
    docker: DockerOptions
    models: dict[str, ModelInstalledInfo]
    options: InstallServiceIn
    parsed_options: SindriOptions
    container_host: str
    container_port: int
    docker_exposed_port: int
    base_url: str


@dataclass
class DownloadedInfo:
    pass


class SindriService(Base2Service[InstalledInfo, DownloadedInfo]):
    models: dict[str, dict[str, SindriAiModel]]

    def _after_init(self) -> None:
        self.models = {}
        self.load_default_models("default")

    def load_default_models(self, instance: str) -> None:
        """Load default models to instance."""
        self.models[instance] = _const.models.copy()

    def get_type(self) -> str:
        """Return the service id."""
        return "sindri"

    def get_description(self) -> str:
        """Return the service description."""
        return "Remote encrypted access to Sindri models."

    def get_size(self) -> ServiceSize:
        """Return the service size."""
        return _const.image.size

    def get_spec(self) -> ServiceSpecification:
        """Return the service specification."""
        return ServiceSpecification(
            fields=[
                ServiceField(
                    type="text", name="api_url", description="API URL", required=False, default="https://sindri.app/api/ai/v1/openai"
                ),
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
        return None

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

    def _get_image(self) -> DockerImage:
        return _const.image

    def _load_download_info(self, data: dict[str, Any]) -> DownloadedInfo:
        return DownloadedInfo(**data)

    async def _install_instance(self, instance: str, options: InstallServiceIn) -> PromiseWithProgress[InstalledInfo, StreamChunk]:
        if not self.models.get(instance):
            self.load_default_models(instance)

        parsed_options = try_parse_pydantic(SindriOptions, options.spec)
        image = self._get_image()
        await self._verify_docker_image(image.name, options.ignore_warnings)

        async def func(stream: Stream[StreamChunk]) -> InstalledInfo:
            await self._download_image_or_set_progress(stream, image)
            self.service_downloaded = True
            stream.emit(StreamChunkProgress(type="progress", stage="install", value=0, data={}))
            config_path = self._get_working_dir() / "config.yaml"
            data = (
                "listenAddress: 0.0.0.0\n"
                "listenPort: 8080\n"
                "appMode: release\n"
                "sindriClient:\n"
                f"  baseURL: {parsed_options.api_url}\n"
                f"  apiKey: {parsed_options.api_key}\n"
                "  requestTimeoutSeconds: 10\n"
                "\n"
                "  encryption:\n"
                "    enabled: true\n"
                "    keySource: ephemeral  # or 'value' or 'file'"
                "    # Configure keys if not using ephemeral"
            )
            with config_path.open("w", encoding="utf-8") as f:
                f.write(data)
            volumes = [f"{config_path}:/config.yaml"]
            subnet = self.docker_service.get_docker_subnet()
            name = f"{self.get_service_id(instance)}"
            docker_options = DockerOptions(
                name=name,
                container_name=self.docker_service.get_docker_container_name(name),
                image=image.name,
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
            docker_exposed_port = await self.docker_service.install_and_run_docker(docker_options)
            container_host = self.docker_service.get_container_host(subnet, docker_options.name)
            container_port = self.docker_service.get_container_port(subnet, docker_exposed_port, docker_options.image_port)
            info = InstalledInfo(
                docker=docker_options,
                models={},
                options=options,
                parsed_options=parsed_options,
                container_host=container_host,
                container_port=container_port,
                docker_exposed_port=docker_exposed_port,
                base_url=get_base_url(container_host, container_port),
            )
            stream.emit(StreamChunkProgress(type="progress", stage="install", value=1, data={}))
            return info

        return PromiseWithProgress(func=func)

    async def _uninstall_instance(self, instance: str, options: UninstallServiceIn) -> None:
        installed = self.get_instance_info(instance).installed
        if installed:
            for model in installed.models.copy().values():
                if model.type == "llm":
                    self.endpoint_registry.unregister_chat_completion(model.registered_name, model.registration_id)

                if not self.is_model_installed_in_other_instance(instance, model.id):
                    await self._uninstall_model(instance, model.id, UninstallModelIn(purge=options.purge))

            await self.docker_service.uninstall_docker(installed.docker)

        self.instances_info[instance].installed = None

        if options.purge:
            if len(self.instances_info) < 2:
                self.service_downloaded = False
                image = self._get_image()
                await self.docker_service.remove_image(image.name)
                await self._clear_working_dir()
                self.models_downloaded = {}

            if instance == "default":
                self.instances_info["default"] = Instance(None, None, {}, InstanceConfig())
            else:
                del self.instances_info[instance]

    def get_docker_compose_file_path(self, instance: str, model_id: str | None) -> Path:
        """Get docker compose file path."""
        info = self.get_instance_installed_info(instance)
        if model_id:
            raise HTTPException(400, "Docker is not bound with this object")

        return self.docker_service.get_docker_compose_file_path(info.docker.name)

    def service_has_docker(self) -> bool:
        """Return true when docker is started when service is installed."""
        return True

    async def stop_instance(self, instance: str) -> None:
        """Stop the Sindri service Docker container."""
        installed = self.get_instance_info(instance).installed
        if not installed:
            return
        await self._stop_docker(installed.docker)

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

        if model_id not in _const.models:
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
            spec=self.get_model_spec(),
            has_docker=False,
        )

    async def _install_model(
        self, instance: str, model_id: str, options: InstallModelIn
    ) -> PromiseWithProgress[InstallModelOut, StreamChunk]:
        parsed_model_options = try_parse_pydantic(SindriModelOptions, options.spec) if options.spec else SindriModelOptions()
        info = self.get_instance_installed_info(instance)

        if not self.models.get(instance):
            self.models[instance] = {}

        if model_id in info.models:
            return PromiseWithProgress(value=InstallModelOut(status="OK", details="Already installed"))

        if model_id not in _const.models:
            raise HTTPException(400, "Model not found")

        model = self.models[instance][model_id]

        async def func(stream: Stream[StreamChunk]) -> InstallModelOut:
            stream.emit(StreamChunkProgress(type="progress", stage="install", value=0, data={}))
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
                    messages=None,
                    responses=None,
                    chat_completions=ProxyOptions(url=f"{info.base_url}/v1/chat/completions", rewrite_model_to=model.real_model_name),
                    completions=None,
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

        if options.purge and model_id in self.models_downloaded:
            del self.models_downloaded[model_id]
