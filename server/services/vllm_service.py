"""Vllm service."""

from collections.abc import Mapping

from fastapi import HTTPException
from pydantic import BaseModel

from server.docker import DockerOptions, docker_pull, install_and_run_docker, uninstall_docker
from server.endpointregistry import ProxyOptions
from server.models.models import InstallModelIn, ListModelsFilters, ListModelsOut, RetrieveModelOut, UninstallModelIn
from server.models.services import InstallServiceIn, UninstallServiceIn
from server.services.base2_service import Base2Service, ModelConfig, ServiceConfig


class VllmModel(BaseModel):
    docker_name: str
    hf_id: str
    env_vars: Mapping[str, str] | None = None
    quantization: str | None = None
    dtype: str = "auto"
    shm_size: str = "16gb"
    ulimits: Mapping[str, str] | None = None
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
        port: int,
    ):
        self.id = id
        self.registered_name = registered_name
        self.options = options
        self.docker = docker
        self.port = port


class InstalledInfo:
    def __init__(
        self,
        models: dict[str, ModelInstalledInfo],
        options: InstallServiceIn,
    ):
        self.models = models
        self.options = options


class VllmService(Base2Service[InstalledInfo]):
    def get_id(self) -> str:
        """Return the service id."""
        return "vllm"

    def _generate_config(self, info: InstalledInfo) -> ServiceConfig:
        return ServiceConfig(options=info.options, models=[ModelConfig(model_id=x.id, options=x.options) for x in info.models.values()])

    async def _install_core(self, options: InstallServiceIn) -> InstalledInfo:
        image = self._get_image(options.gpu)
        await docker_pull(image)
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

        image = self._get_image(info.options.gpu)
        volumes = [f"{self._get_working_dir() / 'models'}:/models"]

        docker_options = DockerOptions(
            name=f"{self.get_id()}-{model.docker_name}",
            image=image,
            command=" ".join(vllm_command),
            image_port=8000,
            env_vars=model.env_vars,
            restart="unless-stopped",
            volumes=volumes,
            use_gpu=info.options.gpu,
            shm_size=model.shm_size,
            ulimits=model.ulimits,
        )
        port = await install_and_run_docker(self.application_context, docker_options)
        registered_name = options.alias if options.alias is not None else model_id
        info.models[model_id] = ModelInstalledInfo(
            id=model_id,
            registered_name=registered_name,
            options=options,
            docker=docker_options,
            port=port,
        )
        self.endpoint_registry.register_chat_completion_as_proxy(
            registered_name, ProxyOptions(url=f"http://localhost:{port}/v1/chat/completions")
        )

    def _get_image(self, gpu: bool) -> str:
        return _const.image_gpu if gpu else _const.image_cpu

    async def _uninstall_model(self, model_id: str, options: UninstallModelIn) -> None:
        info = self._check_installed()
        if model_id not in info.models:
            return
        model = info.models[model_id]
        del info.models[model_id]
        self.endpoint_registry.unregister_chat_completion(model.registered_name)
        await uninstall_docker(self.application_context, model.docker)
        if options.purge:
            # unsupported
            pass
