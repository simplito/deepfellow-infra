# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Custom service."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

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
    InstallModelOptions,
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
    normalize_name,
    try_parse_pydantic,
)
from server.utils.hardware import NvidiaGpuInfo

type SrvCustomModelX = Callable[["CustomService", str | None], SrvCustomModel]
type DockerOptionsOrCallable = DockerOptions | Callable[[InstallModelOptions], DockerOptions]


@dataclass
class SrvCustomModel:
    model_props: ModelProps
    model_spec: ModelSpecification
    model_type: str
    default_prefix: str
    size: str
    options: DockerOptionsOrCallable
    custom: CustomModelId | None = None


class SrvCustomCustomModel(BaseModel):
    id: str
    private: bool = True
    default_prefix: Annotated[str, Field(pattern=r"^[a-zA-Z0-9_-]+$")]
    size: str
    image: str
    image_port: int
    command: str | None = None
    hardware: str | bool | None = None
    volumes: list[str] | None = None
    envs: dict[str, str] | None = None
    healthcheck_cmd: str | None = None
    healthcheck_start_period: str | None = None  # TODO change to str time


@dataclass
class CustomConst:
    models: dict[str, SrvCustomModelX]


class CustomModelOptions(BaseModel):
    prefix: Annotated[str, Field(pattern=r"^[a-zA-Z0-9_-]+$")]


@dataclass
class ModelInstalledInfo:
    id: str
    options: InstallModelIn
    docker_options: DockerOptions
    container_host: str
    container_port: int
    docker_exposed_port: int
    registration_id: RegistrationId
    prefix: str
    base_url: str

    def get_info(self) -> ModelInfo:
        """Get info."""
        return ModelInfo(spec=self.options.spec, registration_id=self.registration_id)


@dataclass
class InstalledInfo:
    models: dict[str, ModelInstalledInfo]
    options: InstallServiceIn


@dataclass
class DownloadedInfo:
    image: str


class CustomService(Base2Service[InstalledInfo, DownloadedInfo]):
    models: dict[str, dict[str, "SrvCustomModel"]]

    def _after_init(self) -> None:
        self.models = {}
        self.load_default_models("default")

    def load_default_models(self, instance: str) -> None:
        """Load default models to instance."""
        self.models[instance] = {}
        subnet = self.docker_service.get_docker_subnet()
        for model in _const.models.copy():
            self.models[instance][model] = _const.models[model](self, subnet)

    def get_type(self) -> str:
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

    def get_default_model_spec(self, default_prefix: str) -> ModelSpecification:
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

    async def stop_instance(self, instance: str) -> None:
        """Stop all custom service Docker containers."""
        installed = self.get_instance_info(instance).installed
        if not installed:
            return
        await self._stop_dockers_parallel([model.docker_options for model in installed.models.values()])

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
                CustomModelField(type="list", name="volumes", description="Docker volumes", placeholder="/work/storage", required=False),
                CustomModelField(type="map", name="envs", description="Docker environment variables", required=False),
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

    def _load_download_info(self, data: dict[str, Any]) -> DownloadedInfo:
        return DownloadedInfo(**data)

    async def _install_instance(self, instance: str, options: InstallServiceIn) -> PromiseWithProgress[InstalledInfo, StreamChunk]:
        if not self.models.get(instance):
            self.load_default_models(instance)

        async def func(stream: Stream[StreamChunk]) -> InstalledInfo:  # noqa: ARG001
            self.service_downloaded = True
            return InstalledInfo(models={}, options=options)

        return PromiseWithProgress(func=func)

    async def _uninstall_instance(self, instance: str, options: UninstallServiceIn) -> None:
        installed = self.get_instance_info(instance).installed
        if installed:
            for model in installed.models.copy().values():
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

    def get_docker_compose_file_path(self, instance: str, model_id: str | None) -> Path:
        """Get docker compose file path."""
        info = self.get_instance_installed_info(instance)
        if not model_id:
            raise HTTPException(400, "Docker is not bound with this object")

        model_installed = info.models.get(model_id, None)
        if not model_installed:
            raise HTTPException(status_code=400, detail="Model not installed")

        return self.docker_service.get_docker_compose_file_path(model_installed.docker_options.name)

    def _add_custom_model(self, instance: str, model: CustomModel) -> None:
        parsed = try_parse_pydantic(SrvCustomCustomModel, model.data)

        if not self.models.get(instance):
            self.models[instance] = {}

        if parsed.id in self.models[instance]:
            raise HTTPException(400, "Model with given id already exists.")
        name = normalize_name(f"{parsed.id}-{instance}")
        subnet = self.docker_service.get_docker_subnet()
        self.models[instance][parsed.id] = SrvCustomModel(
            model_props=ModelProps(private=parsed.private, type="custom", endpoints=[f"/custom/{parsed.default_prefix}"]),
            model_spec=self.get_default_model_spec(parsed.default_prefix),
            model_type="custom",
            default_prefix=parsed.default_prefix,
            size=parsed.size,
            options=DockerOptions(
                image_port=parsed.image_port,
                name=name,
                container_name=self.docker_service.get_docker_container_name(name),
                image=parsed.image,
                command=parsed.command,
                hardware=self.get_specified_hardware_parts(parsed.hardware),
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

    def _remove_custom_model(self, instance: str, model: CustomModel) -> None:
        installed = self.get_instance_info(instance).installed
        parsed = try_parse_pydantic(SrvCustomCustomModel, model.data)
        if installed and parsed.id in installed.models:
            raise HTTPException(400, "Cannot remove custom model, it is in use, uninstall it first.")
        del self.models[instance][parsed.id]

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
                            type=model.model_type,
                            installed=installed,
                            downloaded=model_id in self.models_downloaded,
                            size=model.size,
                            custom=model.custom,
                            spec=model.model_spec,
                            has_docker=True,
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
            type=model.model_type,
            installed=installed,
            downloaded=model_id in self.models_downloaded,
            size=model.size,
            custom=model.custom,
            spec=model.model_spec,
            has_docker=True,
        )

    async def _install_model(
        self, instance: str, model_id: str, options: InstallModelIn
    ) -> PromiseWithProgress[InstallModelOut, StreamChunk]:
        info = self.get_instance_installed_info(instance)
        if not self.models.get(instance):
            self.models[instance] = {}
        if model_id in info.models:
            return PromiseWithProgress(value=InstallModelOut(status="OK", details="Already installed"))
        if model_id not in self.models[instance]:
            raise HTTPException(400, "Model not found")
        model = self.models[instance][model_id]
        if not options.spec:
            options.spec = {}
        if "prefix" not in options.spec:
            options.spec["prefix"] = model.default_prefix
        model.model_props.prefix = options.spec["prefix"]
        parsed_model_options = try_parse_pydantic(CustomModelOptions, options.spec)
        docker_options = model.options(options.spec) if isinstance(model.options, Callable) else model.options
        await self._verify_docker_image(docker_options.image, options.ignore_warnings)

        async def func(stream: Stream[StreamChunk]) -> InstallModelOut:
            model_dir = self._get_working_dir() / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            subnet = self.docker_service.get_docker_subnet()
            image = DockerImage(name=docker_options.image, size=model.size)
            await self._download_image_or_set_progress(stream, image)
            stream.emit(StreamChunkProgress(type="progress", stage="install", value=0, data={}))
            docker_exposed_port = await self.docker_service.install_and_run_docker(docker_options)
            container_host = self.docker_service.get_container_host(subnet, docker_options.name)
            container_port = self.docker_service.get_container_port(subnet, docker_exposed_port, docker_options.image_port)
            info.models[model_id] = model_info = ModelInstalledInfo(
                id=model_id,
                options=options,
                docker_options=docker_options,
                container_host=container_host,
                container_port=container_port,
                docker_exposed_port=docker_exposed_port,
                registration_id="",
                prefix=parsed_model_options.prefix,
                base_url=get_base_url(container_host, container_port),
            )
            model_info.registration_id = self.endpoint_registry.register_custom_endpoint_as_proxy(
                url=model_info.prefix,
                props=model.model_props,
                options=ProxyOptions(url=model_info.base_url),
                registration_options=None,
            )
            self.models_downloaded[model_id] = DownloadedInfo(image.name)
            stream.emit(StreamChunkProgress(type="progress", stage="install", value=1, data={}))
            return InstallModelOut(status="OK", details="Installed")

        return PromiseWithProgress(func=func)

    async def _uninstall_model(self, instance: str, model_id: str, options: UninstallModelIn) -> None:
        info = self.get_instance_installed_info(instance)
        if model_id in info.models:
            model = info.models[model_id]
            self.endpoint_registry.unregister_custom_endpoint(model.prefix, model.registration_id)
            await self.docker_service.uninstall_docker(model.docker_options)
            del info.models[model_id]

        if options.purge and model_id in self.models_downloaded:
            await self.docker_service.remove_image(self.models_downloaded[model_id].image)
            del self.models_downloaded[model_id]

    def get_working_dir(self) -> Path:
        """Get working dir."""
        return self._get_working_dir()


def create_lemmatizer_model(custom_service: CustomService, subnet: str | None) -> SrvCustomModel:
    """Create lemmatizer model."""
    fields: list[ModelField] = custom_service.add_hardware_field_to_model_spec()
    fields.extend(
        [
            ModelField(
                type="text",
                name="prefix",
                description="Endpoint prefix",
                required=True,
                placeholder="lemmatizer",
                default="lemmatizer",
            ),
            ModelField(
                type="number",
                name="num_workers",
                description="Number of worker threads",
                required=False,
                placeholder="4",
                default="4",
            ),
            ModelField(
                type="number",
                name="queue_max",
                description="Maximum size of the job queue",
                required=False,
                placeholder="100000",
                default="100000",
            ),
            ModelField(
                type="number",
                name="shutdown_timeout",
                description="Grace period for finishing lemmatizer tasks on shutdown.",
                required=False,
                placeholder="30",
                default="30",
            ),
        ]
    )

    def generate_docker_options(model_fields: InstallModelOptions) -> DockerOptions:
        hardware_parts = custom_service.get_specified_hardware_parts(model_fields.get("hardware"))
        image = (
            "hub.simplito.com/deepfellow/deepfellow-lemmatizer:1.0.0-cuda-12.8"
            if any(isinstance(h, NvidiaGpuInfo) for h in hardware_parts)
            else "hub.simplito.com/deepfellow/deepfellow-lemmatizer:1.0.0-cpu"
        )
        return DockerOptions(
            image_port=8090,
            name="df-lemmatizer",
            container_name="df-lemmatizer",
            image=image,
            restart="unless-stopped",
            volumes=[f"{custom_service.get_working_dir()}/lemmatizer/cache/stanza:/root/.cache/stanza"],
            hardware=hardware_parts,
            subnet=subnet,
            env_vars={
                "DF_LEMMATIZER_NUM_WORKERS": model_fields.get("num_workers", 4),
                "DF_LEMMATIZER_QUEUE_MAX": model_fields.get("queue_max", 100000),
                "DF_LEMMATIZER_SHUTDOWN_TIMEOUT": model_fields.get("shutdown_timeout", 30),
            },
            healthcheck={
                "test": "wget -q --spider http://localhost:8090/health",
                "interval": "30s",
                "timeout": "10s",
                "retries": "3",
                "start_period": "30s",
            },
        )

    return SrvCustomModel(
        model_props=ModelProps(private=True, type="custom", endpoints=["/custom/v1/lemmatize"]),
        model_spec=ModelSpecification(fields=fields),
        model_type="custom",
        default_prefix="lemmatizer",
        size="1.89GB",
        options=generate_docker_options,
    )


_const = CustomConst(
    models={
        "bentoml/example-summarization": lambda custom_service, subnet: SrvCustomModel(
            model_props=ModelProps(private=True, type="custom", endpoints=["/custom/bentoml-summarize"]),
            model_spec=custom_service.get_default_model_spec("bentoml-summarize"),
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
            model_props=ModelProps(private=True, type="custom", endpoints=["/custom/ocr"]),
            model_spec=custom_service.get_default_model_spec("ocr"),
            model_type="custom",
            default_prefix="ocr",
            size="10075.38MB",
            options=DockerOptions(
                image_port=8000,
                name="easyocr",
                container_name=custom_service.docker_service.get_docker_container_name("easyocr"),
                image="gitlab2.simplito.com:5050/df/df-ocr:1.0.1",
                env_vars={},
                restart="unless-stopped",
                volumes=[f"{custom_service.get_working_dir()}/easyocr/model:/root/.EasyOCR/model"],
                subnet=subnet,
            ),
        ),
        "lemmatizer": create_lemmatizer_model,
    }
)
