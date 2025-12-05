# DeepFellow Software Framework.
# Copyright © 2025 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Custom service."""

from collections.abc import Callable
from pathlib import Path
from typing import Annotated

from fastapi import HTTPException
from pydantic import BaseModel, Field

from server.applicationcontext import get_base_url
from server.docker import DockerImage, DockerOptions
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
from server.models.services import InstallServiceIn, ServiceOptions, ServiceSize, ServiceSpecification, UninstallServiceIn
from server.services.base2_service import Base2Service, CustomModel, ModelConfig, ServiceConfig
from server.utils.core import (
    PromiseWithProgress,
    Stream,
    StreamChunk,
    StreamChunkProgress,
    normalize_name,
    try_parse_pydantic,
)

type SrvCustomModelX = Callable[["CustomService", str | None], SrvCustomModel]


class SrvCustomModel:
    def __init__(
        self,
        model_props: ModelProps,
        model_spec: ModelSpecification,
        model_type: str,
        default_prefix: str,
        size: str,
        options: DockerOptions,
        custom: CustomModelId | None = None,
    ):
        self.model_props = model_props
        self.model_spec = model_spec
        self.model_type = model_type
        self.default_prefix = default_prefix
        self.size = size
        self.options = options
        self.custom = custom


class SrvCustomCustomModel(BaseModel):
    id: str
    private: bool = True
    default_prefix: Annotated[str, Field(pattern=r"^[a-zA-Z0-9_-]+$")]
    size: str
    image: str
    image_port: int
    command: str | None = None
    use_gpu: bool = False
    volumes: list[str] | None = None
    envs: dict[str, str] | None = None
    healthcheck_cmd: str | None = None
    healthcheck_start_period: str | None = None  # TODO change to str time


class CustomConst:
    def __init__(
        self,
        models: dict[str, SrvCustomModelX],
    ):
        self.models = models


class CustomModelOptions(BaseModel):
    prefix: Annotated[str, Field(pattern=r"^[a-zA-Z0-9_-]+$")]


class ModelInstalledInfo:
    def __init__(
        self,
        id: str,
        options: InstallModelIn,
        docker_options: DockerOptions,
        container_host: str,
        container_port: int,
        docker_exposed_port: int,
        registration_id: RegistrationId,
        prefix: str,
    ):
        self.id = id
        self.options = options
        self.docker_options = docker_options
        self.container_host = container_host
        self.container_port = container_port
        self.docker_exposed_port = docker_exposed_port
        self.base_url = get_base_url(self.container_host, self.container_port)
        self.registration_id = registration_id
        self.prefix = prefix

    def get_info(self) -> ModelInfo:
        """Get info."""
        return ModelInfo(spec=self.options.spec, registration_id=self.registration_id)


class InstalledInfo:
    def __init__(
        self,
        models: dict[str, ModelInstalledInfo],
        options: InstallServiceIn,
    ):
        self.models = models
        self.options = options


class CustomService(Base2Service[InstalledInfo]):
    models: dict[str, "SrvCustomModel"]

    def _after_init(self) -> None:
        self.models = dict[str, "SrvCustomModel"]()
        subnet = self.docker_service.get_docker_subnet()
        for model in _const.models.copy():
            self.models[model] = _const.models[model](self, subnet)

    def get_id(self) -> str:
        """Return the service id."""
        return "custom"

    def get_description(self) -> str:
        """Return the service description."""
        return "Your option to add custom service."

    def get_size(self) -> ServiceSize:
        """Return the service size."""
        return ""

    def get_spec(self) -> ServiceSpecification:
        """Return the service specification."""
        return ServiceSpecification(fields=[])

    def get_defaut_model_spec(self, default_prefix: str) -> ModelSpecification:
        """Return the model specification."""
        return ModelSpecification(
            fields=[
                ModelField(
                    type="text",
                    name="prefix",
                    description="Endpoint prefix",
                    required=True,
                    placeholder="my-prefix",
                    default=default_prefix,
                ),
            ]
        )

    async def stop(self) -> None:
        """Stop all custom service Docker containers."""
        info = self.installed
        if not info:
            return
        await self._stop_dockers_parallel([model.docker_options for model in info.models.values()])

    def get_custom_model_spec(self) -> CustomModelSpecification | None:
        """Return the custom model specification or None if custom model is not supported."""
        return CustomModelSpecification(
            fields=[
                CustomModelField(type="text", name="id", description="Model ID", placeholder="my-custom-model"),
                CustomModelField(type="bool", name="private", description="Model is private", default="true"),
                CustomModelField(
                    type="text",
                    name="default_prefix",
                    description="Default model endpoint prefix [a-zA-Z0-9_-]",
                    placeholder="custom-model",
                ),
                CustomModelField(type="text", name="size", description="Model size", placeholder="1GB"),
                CustomModelField(type="text", name="image", description="Docker image", placeholder="company/image"),
                CustomModelField(type="text", name="image_port", description="Docker image port", placeholder="8000"),
                CustomModelField(type="text", name="command", description="Docker command", placeholder="/bin/myapp", required=False),
                CustomModelField(
                    type="text",
                    name="healthcheck_cmd",
                    description="Healthcheck command",
                    placeholder="curl --fail 127.0.0.1:8000 | exit 1",
                    required=False,
                ),
                CustomModelField(
                    type="text",
                    name="healthcheck_start_period",
                    description="Healthcheck start period",
                    placeholder="10s",
                    required=False,
                ),
                CustomModelField(type="list", name="volumes", description="Docker volumes", placeholder="/work/storage"),
                CustomModelField(type="map", name="envs", description="Docker environment variables"),
            ]
        )

    def get_installed_info(self) -> bool | ServiceOptions:
        """Get service installed info."""
        return False if self.installed is None else self.installed.options.spec

    def _generate_config(self, info: InstalledInfo | None) -> ServiceConfig:
        return ServiceConfig(
            options=info.options if info else None,
            models=[ModelConfig(model_id=x.id, options=x.options) for x in info.models.values()] if info else [],
            custom=self.custom,
        )

    async def _install_core(self, options: InstallServiceIn) -> PromiseWithProgress[InstalledInfo, StreamChunk]:
        async def func(stream: Stream[StreamChunk]) -> InstalledInfo:  # noqa: ARG001
            return InstalledInfo(models={}, options=options)

        return PromiseWithProgress(func=func)

    async def _uninstall(self, options: UninstallServiceIn) -> None:
        info = self._check_installed()
        for model in info.models.copy().values():
            await self._uninstall_model(model.id, UninstallModelIn(purge=options.purge))
        self.installed = None
        if options.purge:
            await self._clear_working_dir()

    def get_docker_compose_file_path(self, model_id: str | None) -> Path:
        """Get docker compose file path."""
        info = self.installed
        if not info:
            raise HTTPException(400, "Service not installed")
        if not model_id:
            raise HTTPException(400, "Docker is not bound with this object")
        installed = info.models.get(model_id, None)
        if not installed:
            raise HTTPException(status_code=400, detail="Model not installed")
        return self.docker_service.get_docker_compose_file_path(installed.docker_options.name)

    def _add_custom_model(self, model: CustomModel) -> None:
        parsed = try_parse_pydantic(SrvCustomCustomModel, model.data)
        if parsed.id in self.models:
            raise HTTPException(400, "Model with given id already exists.")
        name = normalize_name(parsed.id)
        subnet = self.docker_service.get_docker_subnet()
        self.models[parsed.id] = SrvCustomModel(
            model_props=ModelProps(private=parsed.private),
            model_spec=self.get_defaut_model_spec(parsed.default_prefix),
            model_type="custom",
            default_prefix=parsed.default_prefix,
            size=parsed.size,
            options=DockerOptions(
                image_port=parsed.image_port,
                name=name,
                container_name=self.docker_service.get_docker_container_name(name),
                image=parsed.image,
                command=parsed.command,
                use_gpu=parsed.use_gpu,
                env_vars=parsed.envs,
                restart="unless-stopped",
                volumes=[f"{self.get_working_dir()}/{name}/volume_{i}:{volume}" for i, volume in enumerate(parsed.volumes or [])],
                subnet=subnet,
                healthcheck={
                    "test": parsed.healthcheck_cmd,
                    "interval": "30s",
                    "timeout": "10s",
                    "retries": "3",
                    "start_period": parsed.healthcheck_start_period or "10s",
                }
                if parsed.healthcheck_cmd
                else None,
            ),
            custom=model.id,
        )

    def _remove_custom_model(self, model: CustomModel) -> None:
        parsed = try_parse_pydantic(SrvCustomCustomModel, model.data)
        if self.installed and parsed.id in self.installed.models:
            raise HTTPException(400, "Cannot remove custom model, it is in use, uninstall it first.")
        del self.models[parsed.id]

    async def list_models(self, filters: ListModelsFilters) -> ListModelsOut:
        """List models."""
        info = self._check_installed()
        out_list: list[RetrieveModelOut] = []
        for model_id in self.models:
            model = self.models[model_id]
            installed = info.models[model_id].get_info() if model_id in info.models else False
            if filters.installed is None or filters.installed == installed:
                out_list.append(
                    RetrieveModelOut(
                        id=model_id,
                        service=self.get_id(),
                        type=model.model_type,
                        installed=installed,
                        size=model.size,
                        custom=model.custom,
                        spec=model.model_spec,
                        has_docker=True,
                    )
                )
        return ListModelsOut(list=out_list)

    async def get_model(self, model_id: str) -> RetrieveModelOut:
        """Get the model."""
        info = self._check_installed()
        if model_id not in self.models:
            raise HTTPException(status_code=400, detail="Model not found")
        model = self.models[model_id]
        installed = info.models[model_id].get_info() if model_id in info.models else False
        return RetrieveModelOut(
            id=model_id,
            service=self.get_id(),
            type=model.model_type,
            installed=installed,
            size=model.size,
            custom=model.custom,
            spec=model.model_spec,
            has_docker=True,
        )

    async def _install_model(self, model_id: str, options: InstallModelIn) -> PromiseWithProgress[InstallModelOut, StreamChunk]:
        info = self._check_installed()
        if model_id in info.models:
            return PromiseWithProgress(value=InstallModelOut(status="OK", details="Already installed"))
        if model_id not in self.models:
            raise HTTPException(400, "Model not found")
        model = self.models[model_id]
        if not options.spec:
            options.spec = {}
        if "prefix" not in options.spec:
            options.spec["prefix"] = model.default_prefix
        parsed_model_options = try_parse_pydantic(CustomModelOptions, options.spec)

        async def func(stream: Stream[StreamChunk]) -> InstallModelOut:
            model_dir = self._get_working_dir() / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            subnet = self.docker_service.get_docker_subnet()
            docker_options = model.options
            await self._docker_pull(DockerImage(name=docker_options.image, size=model.size), stream)
            stream.emit(StreamChunkProgress(type="progress", stage="install", value=0))
            docker_exposed_port = await self.docker_service.install_and_run_docker(docker_options)
            info.models[model_id] = model_info = ModelInstalledInfo(
                id=model_id,
                options=options,
                docker_options=docker_options,
                container_host=self.docker_service.get_container_host(subnet, docker_options.name),
                container_port=self.docker_service.get_container_port(subnet, docker_exposed_port, docker_options.image_port),
                docker_exposed_port=docker_exposed_port,
                registration_id="",
                prefix=parsed_model_options.prefix,
            )
            model_info.registration_id = self.endpoint_registry.register_custom_endpoint_as_proxy(
                url=model_info.prefix,
                props=model.model_props,
                options=ProxyOptions(url=model_info.base_url),
                registration_options=None,
            )
            stream.emit(StreamChunkProgress(type="progress", stage="install", value=1))
            return InstallModelOut(status="OK", details="Installed")

        return PromiseWithProgress(func=func)

    async def _uninstall_model(self, model_id: str, options: UninstallModelIn) -> None:
        info = self._check_installed()
        if (model_id not in info.models) or (model_id not in self.models):
            return
        model = info.models[model_id]
        del info.models[model_id]
        self.endpoint_registry.unregister_custom_endpoint(model.prefix, model.registration_id)
        await self.docker_service.uninstall_docker(model.docker_options)
        if options.purge:
            # unsupported
            pass

    def get_working_dir(self) -> Path:
        """Get working dir."""
        return self._get_working_dir()


_const = CustomConst(
    models={
        "bentoml/example-summarization": lambda custom_service, subnet: SrvCustomModel(
            model_props=ModelProps(private=True),
            model_spec=custom_service.get_defaut_model_spec("bentoml-summarize"),
            model_type="custom",
            default_prefix="bentoml-summarize",
            size="12013.49MB",
            options=DockerOptions(
                image_port=3000,
                name="bentoml",
                container_name=custom_service.docker_service.get_docker_container_name("bentoml"),
                image="gitlab2.simplito.com:5050/df/deepfellow-infra/bentomlexample:1.0.1",
                command="serve",
                env_vars={},
                subnet=subnet,
            ),
        ),
        "easyOCR": lambda custom_service, subnet: SrvCustomModel(
            model_props=ModelProps(private=True),
            model_spec=custom_service.get_defaut_model_spec("ocr"),
            model_type="custom",
            default_prefix="ocr",
            size="10075.38MB",
            options=DockerOptions(
                image_port=8000,
                name="easyocr",
                container_name=custom_service.docker_service.get_docker_container_name("easyocr"),
                image="gitlab2.simplito.com:5050/df/df-ocr:1.0.1",
                use_gpu=False,
                env_vars={},
                restart="unless-stopped",
                volumes=[f"{custom_service.get_working_dir()}/easyocr/model:/root/.EasyOCR/model"],
                subnet=subnet,
            ),
        ),
    },
)
