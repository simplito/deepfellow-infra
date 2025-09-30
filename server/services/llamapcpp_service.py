"""Llamacpp service."""

from pathlib import Path

from fastapi import HTTPException
from pydantic import BaseModel

from server.applicationcontext import get_base_url, get_container_host, get_container_port
from server.docker import DockerImage, DockerOptions, docker_pull, install_and_run_docker, uninstall_docker
from server.endpointregistry import ProxyOptions, RegistrationId
from server.models.api import ModelProps
from server.models.models import InstallModelIn, ListModelsFilters, ListModelsOut, RetrieveModelOut, UninstallModelIn
from server.models.services import InstallServiceIn, ServiceField, ServiceOptions, ServiceSize, ServiceSpecification, UninstallServiceIn
from server.services.base2_service import Base2Service, ModelConfig, ServiceConfig
from server.utils.core import Utils


class LlamacppModel(BaseModel):
    docker_name: str
    model_path: str
    size: str


class LlamacppConst(BaseModel):
    image_gpu: DockerImage
    image_cpu: DockerImage
    model_type: str
    models: dict[str, LlamacppModel]


_const = LlamacppConst(
    image_gpu=DockerImage(name="ghcr.io/ggml-org/llama.cpp:server-cuda", size="2680.57 MB"),
    image_cpu=DockerImage(name="ghcr.io/ggml-org/llama.cpp:server", size="99.76 MB"),
    model_type="llm",
    models={
        "bartowski/mistral-community_pixtral-12b": LlamacppModel(
            docker_name="bartowski__mistral-community_pixtral-12b",
            model_path="https://huggingface.co/bartowski/mistral-community_pixtral-12b-GGUF/resolve/main/mistral-community_pixtral-12b-Q5_K_M.gguf",
            size="8.2GB",
        ),
        "bartowski/deepseek-ai_DeepSeek-R1-0528-Qwen3-8B": LlamacppModel(
            docker_name="bartowski__deepseek-ai_deepseek-r1-0528-qwen3-8b",
            model_path="https://huggingface.co/bartowski/deepseek-ai_DeepSeek-R1-0528-Qwen3-8B-GGUF/resolve/main/deepseek-ai_DeepSeek-R1-0528-Qwen3-8B-Q5_K_M.gguf",
            size="5.5GB",
        ),
        "bartowski/Hermes-3-Llama-3.2-3B": LlamacppModel(
            docker_name="bartowski__hermes-3-llama-3_2-3Bb",
            model_path="https://huggingface.co/bartowski/Hermes-3-Llama-3.2-3B-GGUF/resolve/main/Hermes-3-Llama-3.2-3B-Q5_K_M.gguf",
            size="2.2GB",
        ),
        "bartowski/Meta-Llama-3.1-8B-Instruct": LlamacppModel(
            docker_name="bartowski__meta-llama-3_1-8b-instruct",
            model_path="https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
            size="4.6GB",
        ),
        "mradermacher/PLLuM-12B-instruct": LlamacppModel(
            docker_name="mradermacher__pllum-12b-instruct",
            model_path="https://huggingface.co/mradermacher/PLLuM-12B-instruct-GGUF/resolve/main/PLLuM-12B-instruct.Q4_K_M.gguf",
            size="7.0GB",
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
        model_path: Path,
        container_host: str,
        container_port: int,
        docker_exposed_port: int,
        registration_id: RegistrationId,
    ):
        self.id = id
        self.registered_name = registered_name
        self.options = options
        self.docker = docker
        self.model_path = model_path
        self.container_host = container_host
        self.container_port = container_port
        self.docker_exposed_port = docker_exposed_port
        self.base_url = get_base_url(self.container_host, self.container_port)
        self.registration_id = registration_id


class LLamacppOptions(BaseModel):
    gpu: bool


class InstalledInfo:
    def __init__(
        self,
        models: dict[str, ModelInstalledInfo],
        options: InstallServiceIn,
        parsed_options: LLamacppOptions,
    ):
        self.models = models
        self.options = options
        self.parsed_options = parsed_options


class LLamacppService(Base2Service[InstalledInfo]):
    def get_id(self) -> str:
        """Return the service id."""
        return "llamacpp"

    def get_size(self) -> ServiceSize:
        """Return the service size."""
        return {"cpu": _const.image_cpu.size, "gpu": _const.image_gpu.size}

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
        parsed_options = LLamacppOptions(**options.spec)
        image = self._get_image(parsed_options.gpu)
        await docker_pull(image.name)
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
        for model_id, model in _const.models.items():
            installed = model_id in info.models
            if filters.installed is None or filters.installed == installed:
                out_list.append(
                    RetrieveModelOut(
                        id=model_id,
                        service=self.get_id(),
                        type=_const.model_type,
                        installed=installed,
                        size=model.size,
                    )
                )
        return ListModelsOut(list=out_list)

    async def get_model(self, model_id: str) -> RetrieveModelOut:
        """Get the model."""
        info = self._check_installed()
        if model_id not in _const.models:
            raise HTTPException(status_code=400, detail="Model not found")
        model = _const.models[model_id]
        installed = model_id in info.models
        return RetrieveModelOut(id=model_id, service=self.get_id(), type=_const.model_type, installed=installed, size=model.size)

    async def _install_model(self, model_id: str, options: InstallModelIn) -> None:
        info = self._check_installed()
        if model_id in info.models:
            return
        if model_id not in _const.models:
            raise HTTPException(status_code=400, detail="Model not found")
        model = _const.models[model_id]
        model_dir = self._get_working_dir() / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        local_model_path, model_filename = await Utils.ensure_model_downloaded(model.model_path, model_dir)
        model_in_container = f"/models/{model_filename}"
        volumes = [f"{local_model_path.absolute()}:{model_in_container}:ro"]
        image = self._get_image(info.parsed_options.gpu)

        subnet = self.application_context.get_docker_subnet()
        docker_options = DockerOptions(
            name=f"{self.get_id()}-{model.docker_name}",
            image=image.name,
            command=f"--model {model_in_container} --host 0.0.0.0 --port 8080",
            image_port=8080,
            restart="unless-stopped",
            volumes=volumes,
            use_gpu=info.parsed_options.gpu,
            subnet=subnet,
        )
        docker_exposed_port = await install_and_run_docker(self.application_context, docker_options)
        registered_name = options.alias if options.alias is not None else model_id
        info.models[model_id] = model_info = ModelInstalledInfo(
            id=model_id,
            registered_name=registered_name,
            options=options,
            docker=docker_options,
            model_path=local_model_path.absolute(),
            container_host=get_container_host(subnet, docker_options.name),
            container_port=get_container_port(subnet, docker_exposed_port, docker_options.image_port),
            docker_exposed_port=docker_exposed_port,
            registration_id="",
        )
        model_info.registration_id = self.endpoint_registry.register_chat_completion_as_proxy(
            model=registered_name,
            props=ModelProps(private=True),
            chat_completions=ProxyOptions(url=f"{model_info.base_url}/v1/chat/completions", rewrite_model_to=model_id),
            completions=ProxyOptions(url=f"{model_info.base_url}/v1/completions", rewrite_model_to=model_id),
        )

    def _get_image(self, gpu: bool) -> DockerImage:
        return _const.image_gpu if gpu else _const.image_cpu

    async def _uninstall_model(self, model_id: str, options: UninstallModelIn) -> None:
        info = self._check_installed()
        if model_id not in info.models:
            return
        model = info.models[model_id]
        del info.models[model_id]
        self.endpoint_registry.unregister_chat_completion(model.registered_name, model.registration_id)
        await uninstall_docker(self.application_context, model.docker)
        if options.purge:
            model.model_path.unlink()
