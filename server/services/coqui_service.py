"""Coqui service."""

from collections.abc import AsyncGenerator
from pathlib import Path

from aiohttp import ClientSession
from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from server.applicationcontext import get_base_url, get_container_host, get_container_port
from server.docker import DockerImage, DockerOptions, docker_pull, install_and_run_docker, uninstall_docker
from server.endpointregistry import EndpointCallback, RegistrationId, SimpleEndpoint
from server.ffmpeg import ffmpeg_audio_convert_async_gen
from server.models.api import CreateSpeechRequest, ModelProps
from server.models.models import InstallModelIn, ListModelsFilters, ListModelsOut, RetrieveModelOut, UninstallModelIn
from server.models.services import InstallServiceIn, ServiceField, ServiceOptions, ServiceSize, ServiceSpecification, UninstallServiceIn
from server.services.base2_service import Base2Service, ModelConfig, ServiceConfig
from server.utils.core import Utils


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
    image_gpu=DockerImage(name="ghcr.io/coqui-ai/tts", size="10549.64 MB"),
    image_cpu=DockerImage(name="ghcr.io/coqui-ai/tts-cpu", size="10387.64 MB"),
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


class CoquiOptions(BaseModel):
    gpu: bool


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


class CoquiService(Base2Service[InstalledInfo]):
    def get_id(self) -> str:
        """Return the service id."""
        return "coqui"

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
        parsed_options = CoquiOptions(**options.spec)
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
                        type=model.model_type,
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
        return RetrieveModelOut(id=model_id, service=self.get_id(), type=model.model_type, installed=installed, size=model.size)

    async def _install_model(self, model_id: str, options: InstallModelIn) -> None:
        info = self._check_installed()
        if model_id in info.models:
            return
        if model_id not in _const.models:
            raise HTTPException(status_code=400, detail="Model not found")
        model = _const.models[model_id]
        volumes = [f"{self._get_working_output_dir()}:/root/tts-output", f"{self._get_working_dir()}/models:/root/.local/share/tts"]
        image = self._get_image(info.parsed_options.gpu)

        subnet = self.application_context.get_docker_subnet()
        docker_options = DockerOptions(
            name=f"{self.get_id()}-{model.docker_name}",
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
        )
        docker_exposed_port = await install_and_run_docker(self.application_context, docker_options)
        registered_name = options.alias if options.alias is not None else model_id
        info.models[model_id] = model_info = ModelInstalledInfo(
            id=model_id,
            type=model.model_type,
            registered_name=registered_name,
            options=options,
            docker=docker_options,
            container_host=get_container_host(subnet, docker_options.name),
            container_port=get_container_port(subnet, docker_exposed_port, docker_options.image_port),
            docker_exposed_port=docker_exposed_port,
            registration_id="",
        )
        model_info.registration_id = self.endpoint_registry.register_audio_speech(
            model=registered_name,
            props=ModelProps(private=True),
            endpoint=SimpleEndpoint(on_request=_create_handler(model_info.base_url, model.default_speaker, model.response_format)),
        )

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
        if model_id not in info.models:
            return
        model = info.models[model_id]
        del info.models[model_id]
        if model.type == "tts":
            self.endpoint_registry.unregister_audio_speech(model.registered_name, model.registration_id)
        await uninstall_docker(self.application_context, model.docker)
        if options.purge:
            # unsupported
            pass


def _create_handler(base_url: str, default_speaker: str | None, response_format: str | None) -> EndpointCallback[CreateSpeechRequest]:
    async def _proxy_post_request(url: str) -> AsyncGenerator[bytes]:
        async with ClientSession() as session, session.get(url) as resp:
            async for chunk in resp.content.iter_any():
                if chunk:
                    yield chunk

    async def coqui_handler(body: CreateSpeechRequest, _req: Request) -> StreamingResponse:
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
