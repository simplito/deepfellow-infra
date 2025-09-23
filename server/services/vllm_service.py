"""Vllm service."""

from fastapi import HTTPException
from pydantic import BaseModel

from server.applicationcontext import get_base_url, get_container_host, get_container_port
from server.docker import DockerOptions, docker_pull, install_and_run_docker, uninstall_docker
from server.endpointregistry import ProxyOptions, RegistrationId
from server.models.models import InstallModelIn, ListModelsFilters, ListModelsOut, RetrieveModelOut, UninstallModelIn
from server.models.services import InstallServiceIn, ServiceField, ServiceOptions, ServiceSpecification, UninstallServiceIn
from server.services.base2_service import Base2Service, ModelConfig, ServiceConfig


class VllmModel(BaseModel):
    docker_name: str
    hf_id: str
    env_vars: dict[str, str] | None = None
    quantization: str | None = None
    dtype: str = "auto"
    shm_size: str = "16gb"
    ulimits: dict[str, str] | None = None
    max_model_len: int | None = None
    gpu_memory_utilization: float = 0.9


class VllmConst(BaseModel):
    image_gpu: str
    image_cpu: str
    model_type: str
    hf_token: str | None
    models: dict[str, VllmModel]


_const = VllmConst(
    image_gpu="vllm/vllm-openai:v0.8.4",
    image_cpu="vllm/vllm-openai:v0.8.4",  # "opea/erag-vllm-cpu:latest",
    model_type="llm",
    hf_token=None,
    models={
        "speakleash/Bielik-11B-v2.6-Instruct-AWQ": VllmModel(
            docker_name="speakleash__bielik-11b-v2.6-instruct-awq",
            hf_id="speakleash/Bielik-11B-v2.6-Instruct-AWQ",
            env_vars={"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512"},
            quantization="awq_marlin",
            gpu_memory_utilization=0.95,
            max_model_len=4096,
        ),
    },
)


class ModelInstalledInfo:
    def __init__(
        self,
        id: str,
        registered_name: str,
        options: InstallModelIn,
        docker: DockerOptions,
        container_host: str,
        container_port: int,
        docker_exposed_port: int,
        completions_registration_id: RegistrationId,
        chat_completions_registration_id: RegistrationId,
    ):
        self.id = id
        self.registered_name = registered_name
        self.options = options
        self.docker = docker
        self.container_host = container_host
        self.container_port = container_port
        self.docker_exposed_port = docker_exposed_port
        self.base_url = get_base_url(self.container_host, self.container_port)
        self.completions_registration_id = completions_registration_id
        self.chat_completions_registration_id = chat_completions_registration_id


class VllmOptions(BaseModel):
    gpu: bool


class InstalledInfo:
    def __init__(
        self,
        models: dict[str, ModelInstalledInfo],
        options: InstallServiceIn,
        parsed_options: VllmOptions,
    ):
        self.models = models
        self.options = options
        self.parsed_options = parsed_options


class VllmService(Base2Service[InstalledInfo]):
    def get_id(self) -> str:
        """Return the service id."""
        return "vllm"

    def get_spec(self) -> ServiceSpecification:
        """Return the service specification."""
        return ServiceSpecification(
            fields=[
                ServiceField(type="bool", name="gpu", description="Run on GPU"),
            ]
        )

    def get_installed_info(self) -> bool | ServiceOptions:
        """Get service installed info."""
        return False if self.installed is None else self.installed.options.spec

    def _generate_config(self, info: InstalledInfo) -> ServiceConfig:
        return ServiceConfig(options=info.options, models=[ModelConfig(model_id=x.id, options=x.options) for x in info.models.values()])

    async def _install_core(self, options: InstallServiceIn) -> InstalledInfo:
        parsed_options = VllmOptions(**options.spec)
        image = self._get_image(parsed_options.gpu)
        await docker_pull(image)
        return InstalledInfo(models={}, options=options, parsed_options=parsed_options)

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
            installed = model_id in info.models
            if filters.installed is None or filters.installed == installed:
                out_list.append(RetrieveModelOut(id=model_id, service=self.get_id(), type=_const.model_type, installed=installed))
        return ListModelsOut(list=out_list)

    async def get_model(self, model_id: str) -> RetrieveModelOut:
        """Get the model."""
        info = self._check_installed()
        if model_id not in _const.models:
            raise HTTPException(status_code=400, detail="Model not found")
        installed = model_id in info.models
        return RetrieveModelOut(id=model_id, service=self.get_id(), type=_const.model_type, installed=installed)

    async def _install_model(self, model_id: str, options: InstallModelIn) -> None:
        info = self._check_installed()
        if model_id in info.models:
            return
        if model_id not in _const.models:
            raise HTTPException(status_code=400, detail="Model not found")
        model = _const.models[model_id]
        model_dir = self._get_working_dir() / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        vllm_command = [
            "--model",
            model.hf_id,
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--dtype",
            model.dtype,
            "--gpu-memory-utilization",
            str(model.gpu_memory_utilization),
        ]

        if model.quantization:
            vllm_command.extend(["--quantization", model.quantization])

        if model.max_model_len:
            vllm_command.extend(["--max-model-len", str(model.max_model_len)])

        image = self._get_image(info.parsed_options.gpu)
        volumes = [f"{self._get_working_dir() / 'models'}:/models"]

        subnet = self.application_context.get_docker_subnet()
        docker_options = DockerOptions(
            name=f"{self.get_id()}-{model.docker_name}",
            image=image,
            command=" ".join(vllm_command),
            image_port=8000,
            env_vars=model.env_vars,
            restart="unless-stopped",
            volumes=volumes,
            use_gpu=info.parsed_options.gpu,
            shm_size=model.shm_size,
            ulimits=model.ulimits,
            subnet=subnet,
        )
        docker_exposed_port = await install_and_run_docker(self.application_context, docker_options)
        registered_name = options.alias if options.alias is not None else model_id
        info.models[model_id] = model_info = ModelInstalledInfo(
            id=model_id,
            registered_name=registered_name,
            options=options,
            docker=docker_options,
            container_host=get_container_host(subnet, docker_options.name),
            container_port=get_container_port(subnet, docker_exposed_port, docker_options.image_port),
            docker_exposed_port=docker_exposed_port,
            chat_completions_registration_id="",
            completions_registration_id="",
        )
        model_info.chat_completions_registration_id = self.endpoint_registry.register_chat_completion_as_proxy(
            registered_name, ProxyOptions(url=f"{model_info.base_url}/v1/chat/completions", rewrite_model_to=model_id)
        )
        model_info.completions_registration_id = self.endpoint_registry.register_completion_as_proxy(
            registered_name, ProxyOptions(url=f"{model_info.base_url}/v1/completions", rewrite_model_to=model_id)
        )

    def _get_image(self, gpu: bool) -> str:
        return _const.image_gpu if gpu else _const.image_cpu

    async def _uninstall_model(self, model_id: str, options: UninstallModelIn) -> None:
        info = self._check_installed()
        if model_id not in info.models:
            return
        model = info.models[model_id]
        del info.models[model_id]
        self.endpoint_registry.unregister_chat_completion(model.registered_name, model.chat_completions_registration_id)
        self.endpoint_registry.unregister_completion(model.registered_name, model.completions_registration_id)
        await uninstall_docker(self.application_context, model.docker)
        if options.purge:
            # unsupported
            pass
