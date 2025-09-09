"""Llamacpp service."""

from pathlib import Path
from platform import system

from fastapi import HTTPException
from pydantic import BaseModel

from server.docker import DockerOptions, docker_pull, install_and_run_docker, uninstall_docker
from server.models.models import InstallModelIn, ListModelsFilters, ListModelsOut, RetrieveModelOut, UninstallModelIn
from server.models.services import InstallServiceIn, UninstallServiceIn
from server.services.base2_service import Base2Service, ModelConfig, ServiceConfig
from server.utils.core import Utils


class LlamacppModel(BaseModel):
    docker_name: str
    model_path: str


class LlamacppConst(BaseModel):
    image_gpu: str
    image_cpu: str
    image_cpu_arm64: str
    model_type: str
    models: dict[str, LlamacppModel]


_const = LlamacppConst(
    image_gpu="ghcr.io/ggml-org/llama.cpp:server",
    image_cpu="ghcr.io/ggml-org/llama.cpp:server-cuda",
    image_cpu_arm64="deepfellow-llamacpp-arm64:latest",
    model_type="llm",
    models={
        "bartowski/mistral-community_pixtral-12b": LlamacppModel(
            docker_name="bartowski__mistral-community_pixtral-12b",
            model_path="https://huggingface.co/bartowski/mistral-community_pixtral-12b-GGUF/resolve/main/mistral-community_pixtral-12b-Q5_K_M.gguf",
        ),
        "bartowski/deepseek-ai_DeepSeek-R1-0528-Qwen3-8B": LlamacppModel(
            docker_name="bartowski__deepseek-ai_deepseek-r1-0528-qwen3-8b",
            model_path="https://huggingface.co/bartowski/deepseek-ai_DeepSeek-R1-0528-Qwen3-8B-GGUF/resolve/main/deepseek-ai_DeepSeek-R1-0528-Qwen3-8B-Q5_K_M.gguf",
        ),
        "bartowski/Hermes-3-Llama-3.2-3B": LlamacppModel(
            docker_name="bartowski__hermes-3-llama-3_2-3Bb",
            model_path="https://huggingface.co/bartowski/Hermes-3-Llama-3.2-3B-GGUF/resolve/main/Hermes-3-Llama-3.2-3B-Q5_K_M.gguf",
        ),
        "bartowski/Meta-Llama-3.1-8B-Instruct": LlamacppModel(
            docker_name="bartowski__meta-llama-3_1-8b-instruct",
            model_path="https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        ),
        "mradermacher/PLLuM-12B-instruct": LlamacppModel(
            docker_name="mradermacher__pllum-12b-instruct",
            model_path="https://huggingface.co/mradermacher/PLLuM-12B-instruct-GGUF/resolve/main/PLLuM-12B-instruct.Q4_K_M.gguf",
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
        model_path: Path,
    ):
        self.id = id
        self.registered_name = registered_name
        self.options = options
        self.docker = docker
        self.port = port
        self.model_path = model_path


class InstalledInfo:
    def __init__(
        self,
        models: dict[str, ModelInstalledInfo],
        options: InstallServiceIn,
    ):
        self.models = models
        self.options = options


class LLamacppService(Base2Service[InstalledInfo]):
    def get_id(self) -> str:
        """Return the service id."""
        return "llamacpp"

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
        local_model_path, model_filename = await Utils.ensure_model_downloaded(model.model_path, model_dir)
        model_in_container = f"/models/{model_filename}"
        volumes = [f"{local_model_path.absolute()}:{model_in_container}:ro"]
        image = self._get_image(info.options.gpu)

        docker_options = DockerOptions(
            name=f"{self.get_id()}-{model.docker_name}",
            image=image,
            command=f"--model {model_in_container} --host 0.0.0.0 --port 8080",
            image_port=8080,
            restart="unless-stopped",
            volumes=volumes,
            use_gpu=info.options.gpu,
        )
        port = await install_and_run_docker(self.application_context, docker_options)
        registered_name = options.alias if options.alias is not None else model_id
        info.models[model_id] = ModelInstalledInfo(
            id=model_id,
            registered_name=registered_name,
            options=options,
            docker=docker_options,
            port=port,
            model_path=local_model_path.absolute(),
        )
        self.endpoint_registry.register_all_completions_as_proxy(registered_name, f"http://localhost:{port}")

    def _get_image(self, gpu: bool) -> str:
        return _const.image_cpu_arm64 if system() == "Darwin" else _const.image_gpu if gpu else _const.image_cpu

    async def _uninstall_model(self, model_id: str, options: UninstallModelIn) -> None:
        info = self._check_installed()
        if model_id not in info.models:
            return
        model = info.models[model_id]
        del info.models[model_id]
        self.endpoint_registry.unregister_all_completions(model.registered_name)
        await uninstall_docker(self.application_context, model.docker)
        if options.purge:
            model.model_path.unlink()
