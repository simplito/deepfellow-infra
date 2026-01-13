# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Coqui service."""

from collections.abc import AsyncGenerator
from pathlib import Path

from aiohttp import ClientSession
from attr import dataclass
from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from server.applicationcontext import get_base_url
from server.docker import (
    DockerImage,
    DockerOptions,
)
from server.endpointregistry import EndpointCallback, RegistrationId, SimpleEndpoint
from server.ffmpeg import ffmpeg_audio_convert_async_gen
from server.models.api import CreateSpeechRequest, ModelProps
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
from server.services.base2_service import Base2Service, ModelConfig, ServiceConfig
from server.utils.core import (
    PromiseWithProgress,
    Stream,
    StreamChunk,
    StreamChunkProgress,
    Utils,
    try_parse_pydantic,
)


class CoquiModel(BaseModel):
    docker_name: str
    default_speaker: str
    response_format: str = "mp3"
    model_type: str
    language: str | None = None
    size: str


class CoquiConst(BaseModel):
    image_gpu: DockerImage
    image_cpu: DockerImage
    models: dict[str, CoquiModel]


_const = CoquiConst(
    image_gpu=DockerImage(name="ghcr.io/coqui-ai/tts:dbf1a08a0d4e47fdad6172e433eeb34bc6b13b4e", size="11.0 GB"),
    image_cpu=DockerImage(name="ghcr.io/coqui-ai/tts-cpu:dbf1a08a0d4e47fdad6172e433eeb34bc6b13b4e", size="11.0 GB"),
    models={
        "tts_models/en/vctk/vits": CoquiModel(
            docker_name="en-vctk-vits",
            default_speaker="p225",
            model_type="tts",
            size="152MB",
        ),
    },
)


class ModelInstalledInfo:
    def __init__(
        self,
        id: str,
        type: str,
        registered_name: str,
        options: InstallModelIn,
        docker: DockerOptions,
        container_host: str,
        container_port: int,
        docker_exposed_port: int,
        registration_id: RegistrationId,
    ):
        self.id = id
        self.type = type
        self.registered_name = registered_name
        self.options = options
        self.docker = docker
        self.container_host = container_host
        self.container_port = container_port
        self.docker_exposed_port = docker_exposed_port
        self.base_url = get_base_url(self.container_host, self.container_port)
        self.registration_id = registration_id

    def get_info(self) -> ModelInfo:
        """Get info."""
        return ModelInfo(spec=self.options.spec, registration_id=self.registration_id)


class CoquiOptions(BaseModel):
    gpu: bool


class CoquiModelOptions(BaseModel):
    alias: str | None = None


class InstalledInfo:
    def __init__(
        self,
        models: dict[str, ModelInstalledInfo],
        options: InstallServiceIn,
        parsed_options: CoquiOptions,
    ):
        self.models = models
        self.options = options
        self.parsed_options = parsed_options


@dataclass
class DownloadedInfo:
    pass


class CoquiCmdOptions:
    def __init__(
        self,
        model_name: str,
        model_path: str | None,
        cuda: bool,
        language: str | None,
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.cuda = cuda
        self.language = language


class CoquiService(Base2Service[InstalledInfo, DownloadedInfo]):
    def get_id(self) -> str:
        """Return the service id."""
        return "coqui"

    def get_description(self) -> str:
        """Return the service description."""
        return "Self-hosted Text-to-Speech model runner"

    def get_size(self) -> ServiceSize:
        """Return the service size."""
        return {"cpu": _const.image_cpu.size, "gpu": _const.image_gpu.size}

    def get_spec(self) -> ServiceSpecification:
        """Return the service specification."""
        return ServiceSpecification(
            fields=[
                ServiceField(type="bool", name="gpu", description="Run on GPU", required=False, default=self._has_gpu_for_spec()),
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

    def get_installed_info(self) -> bool | InstallServiceProgress | ServiceOptions:
        """Get service installed info."""
        return self._get_service_installed_info() if self.installed is None else self.installed.options.spec

    def _generate_config(self, info: InstalledInfo | None) -> ServiceConfig:
        return ServiceConfig(
            options=info.options if info else None,
            models=[ModelConfig(model_id=x.id, options=x.options) for x in info.models.values()] if info else [],
            custom=self.custom,
            downloaded=self.downloaded,
        )

    async def _install_core(self, options: InstallServiceIn) -> PromiseWithProgress[InstalledInfo, StreamChunk]:
        if "gpu" not in options.spec:
            options.spec["gpu"] = self.docker_service.has_gpu_support
        parsed_options = try_parse_pydantic(CoquiOptions, options.spec)
        image = self._get_image(parsed_options.gpu)
        await self._verify_docker_image(image.name, options.ignore_warnings)

        async def func(stream: Stream[StreamChunk]) -> InstalledInfo:
            await self._docker_pull(image, stream)
            return InstalledInfo(models={}, options=options, parsed_options=parsed_options)

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
        return self.docker_service.get_docker_compose_file_path(installed.docker.name)

    async def list_models(self, filters: ListModelsFilters) -> ListModelsOut:
        """List models."""
        info = self._check_installed()
        out_list: list[RetrieveModelOut] = []
        for model_id, model in _const.models.items():
            installed = info.models[model_id].get_info() if model_id in info.models else self._get_model_installed_info(model_id)
            if filters.installed is None or filters.installed == installed:
                out_list.append(
                    RetrieveModelOut(
                        id=model_id,
                        service=self.get_id(),
                        type=model.model_type,
                        installed=installed,
                        downloaded=model_id in self.downloaded,
                        size=model.size,
                        spec=self.get_model_spec(),
                        has_docker=True,
                    )
                )
        return ListModelsOut(list=out_list)

    async def get_model(self, model_id: str) -> RetrieveModelOut:
        """Get the model."""
        info = self._check_installed()
        if model_id not in _const.models:
            raise HTTPException(status_code=400, detail="Model not found")
        model = _const.models[model_id]
        installed = info.models[model_id].get_info() if model_id in info.models else self._get_model_installed_info(model_id)
        return RetrieveModelOut(
            id=model_id,
            service=self.get_id(),
            type=model.model_type,
            installed=installed,
            downloaded=model_id in self.downloaded,
            size=model.size,
            spec=self.get_model_spec(),
            has_docker=True,
        )

    async def _install_model(self, model_id: str, options: InstallModelIn) -> PromiseWithProgress[InstallModelOut, StreamChunk]:
        parsed_model_options = try_parse_pydantic(CoquiModelOptions, options.spec) if options.spec else CoquiModelOptions()
        info = self._check_installed()
        if model_id in info.models:
            return PromiseWithProgress(value=InstallModelOut(status="OK", details="Already installed"))
        if model_id not in _const.models:
            raise HTTPException(400, "Model not found")
        model = _const.models[model_id]

        async def func(stream: Stream[StreamChunk]) -> InstallModelOut:
            volumes = [f"{self._get_working_output_dir()}:/root/tts-output", f"{self._get_working_dir()}/models:/root/.local/share/tts"]
            image = self._get_image(info.parsed_options.gpu)

            subnet = self.docker_service.get_docker_subnet()
            stream.emit(StreamChunkProgress(type="progress", stage="install", value=0))
            docker_options = DockerOptions(
                name=f"{self.get_id()}-{model.docker_name}",
                container_name=self.docker_service.get_docker_container_name(f"{self.get_id()}-{model.docker_name}"),
                image=image.name,
                command=self._build_coqui_command(
                    CoquiCmdOptions(
                        model_name=model_id,
                        model_path=None,
                        cuda=info.parsed_options.gpu,
                        language=model.language,
                    )
                ),
                entrypoint="/bin/bash",
                image_port=5002,
                restart="unless-stopped",
                volumes=volumes,
                use_gpu=info.parsed_options.gpu,
                subnet=subnet,
                healthcheck={
                    "test": (
                        """python3 -c 'import requests, sys; r = requests.get("http://localhost:5002");"""
                        """ r.raise_for_status(); print("Success");'"""
                    ),
                    "interval": "30s",
                    "timeout": "10s",
                    "retries": "3",
                    "start_period": "5s",
                },
            )
            docker_exposed_port = await self.docker_service.install_and_run_docker(docker_options)
            registered_name = parsed_model_options.alias if parsed_model_options.alias else model_id
            info.models[model_id] = model_info = ModelInstalledInfo(
                id=model_id,
                type=model.model_type,
                registered_name=registered_name,
                options=options,
                docker=docker_options,
                container_host=self.docker_service.get_container_host(subnet, docker_options.name),
                container_port=self.docker_service.get_container_port(subnet, docker_exposed_port, docker_options.image_port),
                docker_exposed_port=docker_exposed_port,
                registration_id="",
            )
            model_info.registration_id = self.endpoint_registry.register_audio_speech(
                model=registered_name,
                props=ModelProps(private=True),
                endpoint=SimpleEndpoint(on_request=_create_handler(model_info.base_url, model.default_speaker, model.response_format)),
                registration_options=None,
            )
            stream.emit(StreamChunkProgress(type="progress", stage="install", value=1))
            self.downloaded[model_id] = DownloadedInfo()
            return InstallModelOut(status="OK", details="Installed")

        return PromiseWithProgress(func=func)

    async def stop(self) -> None:
        """Stop all the Coqui service Docker containers."""
        info = self.installed
        if not info:
            return
        await self._stop_dockers_parallel([model.docker for model in info.models.values()])

    def _get_image(self, gpu: bool) -> DockerImage:
        return _const.image_gpu if gpu else _const.image_cpu

    def _get_working_output_dir(self) -> Path:
        path = self._get_working_dir() / "output"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _build_coqui_command(self, options: CoquiCmdOptions) -> str:
        cmd_args = ["python3", "TTS/server/server.py"]

        if options.model_path:
            cmd_args.extend(["--model_path", options.model_path])
        elif options.model_name:
            cmd_args.extend(["--model_name", options.model_name])
        else:
            raise ValueError("Either model_path or model_name must be provided")

        cmd_args.extend(["--port", "5002"])

        if options.cuda:
            cmd_args.extend(["--use_cuda", "true"])

        if options.language:
            cmd_args.extend(["--language", options.language])

        command_string = " ".join(cmd_args)
        return f"-c {Utils.shell_escape(command_string)}"

    async def _uninstall_model(self, model_id: str, options: UninstallModelIn) -> None:
        info = self._check_installed()
        if model_id in info.models:
            model = info.models[model_id]
            del info.models[model_id]
            if model.type == "tts":
                self.endpoint_registry.unregister_audio_speech(model.registered_name, model.registration_id)
            await self.docker_service.uninstall_docker(model.docker)

        if options.purge and model_id in self.downloaded:
            del self.downloaded[model_id]
            # unsupported


def _create_handler(base_url: str, default_speaker: str | None, response_format: str | None) -> EndpointCallback[CreateSpeechRequest]:
    async def _proxy_post_request(url: str) -> AsyncGenerator[bytes]:
        async with ClientSession() as session, session.get(url) as resp:
            async for chunk in resp.content.iter_any():
                if chunk:
                    yield chunk

    async def coqui_handler(body: CreateSpeechRequest, _req: Request | None) -> StreamingResponse:
        text = body.input
        voice = body.voice or default_speaker
        response_format2 = body.format or response_format
        if response_format2 is None:
            response_format2 = "wav"

        encoded_text = Utils.str_encode(text)
        coqui_url = f"{base_url}/api/tts?text={encoded_text}"

        if voice is not None:
            voice_encoded = Utils.str_encode(voice)
            coqui_url += f"&speaker_id={voice_encoded}"
        return StreamingResponse(
            ffmpeg_audio_convert_async_gen(_proxy_post_request(coqui_url), "wav", response_format2),
            media_type=f"audio/{response_format2}",
        )

    return coqui_handler
