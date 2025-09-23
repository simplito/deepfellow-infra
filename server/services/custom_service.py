"""Custom service."""

from collections.abc import Callable
from pathlib import Path

from fastapi import HTTPException

from server.applicationcontext import get_base_url, get_container_host, get_container_port
from server.docker import DockerOptions, install_and_run_docker, uninstall_docker
from server.endpointregistry import ProxyOptions, RegistrationId
from server.models.models import InstallModelIn, ListModelsFilters, ListModelsOut, RetrieveModelOut, UninstallModelIn
from server.models.services import InstallServiceIn, ServiceOptions, ServiceSpecification, UninstallServiceIn
from server.services.base2_service import Base2Service, ModelConfig, ServiceConfig


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
    ):
        self.id = id
        self.options = options
        self.docker_options = docker_options
        self.container_host = container_host
        self.container_port = container_port
        self.docker_exposed_port = docker_exposed_port
        self.base_url = get_base_url(self.container_host, self.container_port)
        self.registration_id = registration_id


class InstalledInfo:
    def __init__(
        self,
        models: dict[str, ModelInstalledInfo],
        options: InstallServiceIn,
    ):
        self.models = models
        self.options = options


class CustomService(Base2Service[InstalledInfo]):
    def get_id(self) -> str:
        """Return the service id."""
        return "custom"

    def get_spec(self) -> ServiceSpecification:
        """Return the service specification."""
        return ServiceSpecification(fields=[])

    def get_installed_info(self) -> bool | ServiceOptions:
        """Get service installed info."""
        return False if self.installed is None else self.installed.options.spec

    def _generate_config(self, info: InstalledInfo) -> ServiceConfig:
        return ServiceConfig(options=info.options, models=[ModelConfig(model_id=x.id, options=x.options) for x in info.models.values()])

    async def _install_core(self, options: InstallServiceIn) -> InstalledInfo:
        return InstalledInfo(models={}, options=options)

    async def _uninstall(self, options: UninstallServiceIn) -> None:
        info = self._check_installed()
        for model in info.models.copy().values():
            await self._uninstall_model(model.id, UninstallModelIn(purge=options.purge))
        self.installed = None
        if options.purge:
            await self._clear_working_dir()

    async def list_models(self, filters: ListModelsFilters) -> ListModelsOut:
        """List models."""
        info = self._check_installed()
        out_list: list[RetrieveModelOut] = []
        for model_id in _const.models:
            model = _const.models[model_id]
            installed = model_id in info.models
            if filters.installed is None or filters.installed == installed:
                out_list.append(RetrieveModelOut(id=model_id, service=self.get_id(), type=model.model_type, installed=installed))
        return ListModelsOut(list=out_list)

    async def get_model(self, model_id: str) -> RetrieveModelOut:
        """Get the model."""
        info = self._check_installed()
        if model_id not in _const.models:
            raise HTTPException(status_code=400, detail="Model not found")
        model = _const.models[model_id]
        installed = model_id in info.models
        return RetrieveModelOut(id=model_id, service=self.get_id(), type=model.model_type, installed=installed)

    async def _install_model(self, model_id: str, options: InstallModelIn) -> None:
        info = self._check_installed()
        if model_id in info.models:
            return
        if model_id not in _const.models:
            raise HTTPException(status_code=400, detail="Model not found")
        model = _const.models[model_id]
        model_dir = self._get_working_dir() / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        subnet = self.application_context.get_docker_subnet()
        docker_options = model.options(self, subnet) if callable(model.options) else model.options
        docker_exposed_port = await install_and_run_docker(self.application_context, docker_options)
        info.models[model_id] = model_info = ModelInstalledInfo(
            id=model_id,
            options=options,
            docker_options=docker_options,
            container_host=get_container_host(subnet, docker_options.name),
            container_port=get_container_port(subnet, docker_exposed_port, docker_options.image_port),
            docker_exposed_port=docker_exposed_port,
            registration_id="",
        )
        model_info.registration_id = self.endpoint_registry.register_custom_endpoint_as_proxy(
            model.custom_endpoint, ProxyOptions(url=f"{model_info.base_url}{model.custom_endpoint}")
        )

    async def _uninstall_model(self, model_id: str, options: UninstallModelIn) -> None:
        info = self._check_installed()
        if (model_id not in info.models) or (model_id not in _const.models):
            return
        model = info.models[model_id]
        model_const = _const.models[model_id]
        del info.models[model_id]
        self.endpoint_registry.unregister_custom_endpoint(model_const.custom_endpoint, model.registration_id)
        await uninstall_docker(self.application_context, model.docker_options)
        if options.purge:
            # unsupported
            pass

    def get_docker_options(self) -> DockerOptions:
        """Return docker options."""
        return DockerOptions(
            image_port=8000,
            name="easyocr",
            image="gitlab2.simplito.com:5050/df/df-ocr:1.0.0",
            use_gpu=False,
            env_vars={},
            restart="unless-stopped",
            volumes=[f"{self._get_working_dir()}/easyocr/model:/root/.EasyOCR/model"],
            subnet=self.application_context.get_docker_subnet(),
        )

    def get_working_dir(self) -> Path:
        """Get working dir."""
        return self._get_working_dir()


class CustomModel:
    def __init__(
        self,
        model_type: str,
        custom_endpoint: str,
        options: DockerOptions | Callable[[CustomService, str | None], DockerOptions],
    ):
        self.model_type = model_type
        self.custom_endpoint = custom_endpoint
        self.options = options


class CustomConst:
    def __init__(
        self,
        models: dict[str, CustomModel],
    ):
        self.models = models


_const = CustomConst(
    models={
        "bentoml/example-summarization": CustomModel(
            model_type="custom",
            custom_endpoint="/summarize",
            options=lambda _, subnet: DockerOptions(
                image_port=3000,
                name="bentoml",
                image="gitlab2.simplito.com:5050/df/deepfellow-infra/bentomlexample:1.0.0",
                command="serve",
                env_vars={},
                subnet=subnet,
            ),
        ),
        "easyOCR": CustomModel(
            model_type="custom",
            custom_endpoint="/v1/ocr",
            options=lambda custom_service, subnet: DockerOptions(
                image_port=8000,
                name="easyocr",
                image="gitlab2.simplito.com:5050/df/df-ocr:1.0.0",
                use_gpu=False,
                env_vars={},
                restart="unless-stopped",
                volumes=[f"{custom_service.get_working_dir()}/easyocr/model:/root/.EasyOCR/model"],
                subnet=subnet,
            ),
        ),
    },
)
