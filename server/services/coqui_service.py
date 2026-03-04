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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from aiohttp import ClientSession
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
from server.models.api import TTS_ENDPOINTS, CreateSpeechRequest, ModelProps
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


type ImageTypes = Literal["cpu", "gpu"]


class CoquiConst(BaseModel):
    images: dict[ImageTypes, DockerImage]
    models: dict[str, CoquiModel]


_const = CoquiConst(
    images={
        "cpu": DockerImage(name="ghcr.io/coqui-ai/tts:dbf1a08a0d4e47fdad6172e433eeb34bc6b13b4e", size="11.0 GB"),
        "gpu": DockerImage(name="ghcr.io/coqui-ai/tts-cpu:dbf1a08a0d4e47fdad6172e433eeb34bc6b13b4e", size="11.0 GB"),
    },
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
    hardware: str | bool | None = None


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
    models: dict[str, dict[str, CoquiModel]]

    def _after_init(self) -> None:
        self.models = {}
        self.load_default_models("default")

    def load_default_models(self, instance: str) -> None:
        """Load default models to instance."""
        self.models[instance] = _const.models.copy()

    def get_type(self) -> str:
        """Return the type."""
        return "coqui"

    def get_description(self) -> str:
        """Return the service description."""
        return "Self-hosted Text-to-Speech model runner"

    def get_size(self) -> ServiceSize:
        """Return the service size."""
        sizes = {"cpu": _const.images["cpu"].size}
        if self._supported_gpus:
            sizes["gpu"] = _const.images["gpu"].size
        return sizes

    def get_spec(self) -> ServiceSpecification:
        """Return the service specification."""
        fields = self.add_hardware_field_to_spec()
        return ServiceSpecification(fields=fields)

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

        if "hardware" not in options.spec:
            options.spec["hardware"] = options.spec.get("gpu", self.docker_service.has_gpu_support)
        parsed_options = try_parse_pydantic(CoquiOptions, options.spec)
        image = self._get_image(self.is_given_hardware_support_gpu(parsed_options.hardware))
        await self._verify_docker_image(image.name, options.ignore_warnings)

        async def func(stream: Stream[StreamChunk]) -> InstalledInfo:
            # Coqui download docker which is used in every model.
            await self._download_image_or_set_progress(stream, image)
            self.service_downloaded = True
            return InstalledInfo(models={}, options=options, parsed_options=parsed_options)

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
                for image in _const.images.values():
                    await self.docker_service.remove_image(image.name)
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

        return self.docker_service.get_docker_compose_file_path(model_installed.docker.name)

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
                            spec=self.get_model_spec(),
                            has_docker=True,
                        )
                    )

        return ListModelsOut(list=out_list)

    async def get_model(self, instance: str, model_id: str) -> RetrieveModelOut:
        """Get the model."""
        """Get the model."""
        info = self.get_instance_installed_info(instance)
        if not self.models.get(instance):
            self.models[instance] = {}
        if model_id not in self.models[instance]:
            raise HTTPException(status_code=400, detail="Model not found")

        model = _const.models[model_id]
        installed = info.models[model_id].get_info() if model_id in info.models else self._get_model_installed_info(instance, model_id)
        return RetrieveModelOut(
            id=model_id,
            service=self.get_id(instance),
            type=model.model_type,
            installed=installed,
            downloaded=model_id in self.models_downloaded,
            size=model.size,
            spec=self.get_model_spec(),
            has_docker=True,
        )

    async def _install_model(
        self, instance: str, model_id: str, options: InstallModelIn
    ) -> PromiseWithProgress[InstallModelOut, StreamChunk]:
        parsed_model_options = try_parse_pydantic(CoquiModelOptions, options.spec) if options.spec else CoquiModelOptions()
        info = self.get_instance_installed_info(instance)

        if not self.models.get(instance):
            self.models[instance] = {}

        if model_id in info.models:
            return PromiseWithProgress(value=InstallModelOut(status="OK", details="Already installed"))

        if model_id not in _const.models:
            raise HTTPException(400, "Model not found")

        model = _const.models[model_id]

        async def func(stream: Stream[StreamChunk]) -> InstallModelOut:
            volumes = [f"{self._get_working_output_dir()}:/root/tts-output", f"{self._get_working_dir()}/models:/root/.local/share/tts"]
            use_gpu = self.is_given_hardware_support_gpu(info.parsed_options.hardware)
            image = self._get_image(use_gpu)
            subnet = self.docker_service.get_docker_subnet()
            stream.emit(StreamChunkProgress(type="progress", stage="install", value=0, data={}))
            service_name = f"{self.get_service_id(instance)}-{model.docker_name}"
            docker_options = DockerOptions(
                name=service_name,
                container_name=self.docker_service.get_docker_container_name(service_name),
                image=image.name,
                command=self._build_coqui_command(
                    CoquiCmdOptions(
                        model_name=model_id,
                        model_path=None,
                        cuda=use_gpu,
                        language=model.language,
                    )
                ),
                entrypoint="/bin/bash",
                image_port=5002,
                restart="unless-stopped",
                volumes=volumes,
                hardware=self.get_specified_hardware_parts(info.parsed_options.hardware),
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
                props=ModelProps(private=True, type="tts", endpoints=TTS_ENDPOINTS),
                endpoint=SimpleEndpoint(on_request=_create_handler(model_info.base_url, model.default_speaker, model.response_format)),
                registration_options=None,
            )
            stream.emit(StreamChunkProgress(type="progress", stage="install", value=1, data={}))
            self.models_downloaded[model_id] = DownloadedInfo()
            return InstallModelOut(status="OK", details="Installed")

        return PromiseWithProgress(func=func)

    async def stop_instance(self, instance: str) -> None:
        """Stop all the Coqui service Docker containers."""
        installed = self.get_instance_info(instance).installed
        if not installed:
            return
        await self._stop_dockers_parallel([model.docker for model in installed.models.values()])

    def _get_image(self, gpu: bool) -> DockerImage:
        return _const.images["gpu"] if gpu else _const.images["cpu"]

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

    async def _uninstall_model(self, instance: str, model_id: str, options: UninstallModelIn) -> None:
        info = self.get_instance_installed_info(instance)
        if model_id in info.models:
            model = info.models[model_id]
            del info.models[model_id]
            if model.type == "tts":
                self.endpoint_registry.unregister_audio_speech(model.registered_name, model.registration_id)
            await self.docker_service.uninstall_docker(model.docker)

        if options.purge and model_id in self.models_downloaded:
            del self.models_downloaded[model_id]
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
