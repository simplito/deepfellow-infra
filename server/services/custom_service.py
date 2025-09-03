"""Custom service."""

from collections.abc import Callable

from fastapi import HTTPException

from server.docker import DockerOptions, install_and_run_docker, uninstall_docker
from server.endpointregistry import ProxyOptions
from server.models.models import InstallModelIn, ListModelsFilters, ListModelsOut, RetrieveModelOut, UninstallModelIn
from server.models.services import InstallServiceIn, UninstallServiceIn
from server.services.base2_service import Base2Service, ModelConfig, ServiceConfig


class ModelInstalledInfo:
    def __init__(
        self,
        id: str,
        options: InstallModelIn,
        port: int,
        docker_options: DockerOptions,
    ):
        self.id = id
        self.options = options
        self.port = port
        self.docker_options = docker_options


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

    def _generate_config(self, info: InstalledInfo) -> ServiceConfig:
        return ServiceConfig(options=info.options, models=[ModelConfig(model_id=x.id, options=x.options) for x in info.models.values()])

    async def _install_core(self, options: InstallServiceIn) -> InstalledInfo:
        return InstalledInfo(models={}, options=options)

    async def _uninstall(self, options: UninstallServiceIn) -> None:
        info = self._check_installed()
        for model in info.models.values():
            await self._uninstall_model(model.id, UninstallModelIn())
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

        docker_options = model.options(self) if callable(model.options) else model.options
        port = await install_and_run_docker(self.application_context, docker_options)
        info.models[model_id] = ModelInstalledInfo(
            id=model_id,
            options=options,
            port=port,
            docker_options=docker_options,
        )
        self.endpoint_registry.register_custom_endpoint_as_proxy(
            model.custom_endpoint, ProxyOptions(url=f"http://localhost:{port}{model.custom_endpoint}")
        )

    async def _uninstall_model(self, model_id: str, options: UninstallModelIn) -> None:
        info = self._check_installed()
        if (model_id not in info.models) or (model_id not in _const.models):
            return
        model = info.models[model_id]
        model_const = _const.models[model_id]
        del info.models[model_id]
        self.endpoint_registry.unregister_custom_endpoint(model_const.custom_endpoint)
        await uninstall_docker(self.application_context, model.docker_options)
        if options.purge:
            # unsupported
            pass


def _load_easy_ocr(service: CustomService) -> DockerOptions:
    return DockerOptions(
        image_port=8000,
        name="easyocr",
        image="gitlab2.simplito.com:5050/df/df-ocr:1.0.0",
        use_gpu=False,
        env_vars={},
        restart="unless-stopped",
        volumes=[f"{service._get_working_dir()}/easyocr/model:/root/.EasyOCR/model"],
    )


class CustomModel:
    def __init__(
        self,
        model_type: str,
        custom_endpoint: str,
        options: DockerOptions | Callable[[CustomService], DockerOptions],
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
            options=DockerOptions(
                image_port=3000,
                name="bentoml",
                image="gitlab2.simplito.com:5050/df/deepfellow-infra/bentomlexample:1.0.0",
                command="serve",
                env_vars={},
            ),
        ),
        "easyOCR": CustomModel(
            model_type="custom",
            custom_endpoint="/v1/ocr",
            options=_load_easy_ocr,
        ),
    },
)
