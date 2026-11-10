# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Speaches AI service."""

import asyncio
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse
from langdetect import detect  # type: ignore
from pydantic import BaseModel, BeforeValidator, Field

from server.applicationcontext import get_base_url
from server.config import get_main_dir
from server.docker import DockerImage, DockerOptions
from server.endpointregistry import ProxyOptions, RegistrationId, RegistrationOptions, SimpleEndpoint, post_json
from server.models.api import STT_ENDPOINTS, TTS_ENDPOINTS, CreateSpeechRequest, ModelId, ModelProps
from server.models.models import (
    CustomModelField,
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
from server.services.base2_service import Base2Service, CustomModel, Instance, InstanceConfig, ModelConfig, default_gpu_cpu_image_name
from server.utils.core import (
    DownloadedPacket,
    PreDownloadPacket,
    PromiseWithProgress,
    Stream,
    StreamChunk,
    StreamChunkProgress,
    convert_size_to_bytes,
    load_json_registry,
    try_parse_pydantic,
)
from server.utils.loading import Progress

type ModelType = Literal["tts", "stt"]
type CustomModelId = str


type ImageTypes = Literal["cpu", "gpu"]


class SpeachesAiConst(BaseModel):
    images: dict[ImageTypes, DockerImage]
    audio_speech_models: list[tuple[str, str]]
    audio_transcriptions_models: list[tuple[str, str]]


class SpeachesRegistryEntry(TypedDict):
    name: str
    size: str


class SpeachesRegistry(TypedDict):
    tts: list[SpeachesRegistryEntry]
    stt: list[SpeachesRegistryEntry]


def _build_speech_models(registry: SpeachesRegistry) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    tts = [(entry["name"], entry["size"]) for entry in registry["tts"]]
    stt = [(entry["name"], entry["size"]) for entry in registry["stt"]]
    return tts, stt


def _read_speech_models() -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    speaches_path = get_main_dir() / "./static/speaches-min.json"
    return load_json_registry(speaches_path, "speaches", _build_speech_models, ([], []))


_speech_models, _transcriptions_models = _read_speech_models()

_const = SpeachesAiConst(
    images={
        "gpu": DockerImage(name="ghcr.io/speaches-ai/speaches:0.9.0-rc.3-cuda", size="5.9 GB"),
        "cpu": DockerImage(name="ghcr.io/speaches-ai/speaches:0.9.0-rc.3-cpu", size="1.6 GB"),
    },
    audio_speech_models=_speech_models,
    audio_transcriptions_models=_transcriptions_models,
)


type SupportedLanguages = (
    str
    | Literal[
        "af",
        "ar",
        "bg",
        "bn",
        "ca",
        "cs",
        "cy",
        "da",
        "de",
        "el",
        "en",
        "es",
        "et",
        "fa",
        "fi",
        "fr",
        "gu",
        "he",
        "hi",
        "hr",
        "hu",
        "id",
        "it",
        "ja",
        "kn",
        "ko",
        "lt",
        "lv",
        "mk",
        "ml",
        "mr",
        "ne",
        "nl",
        "no",
        "pa",
        "pl",
        "pt",
        "ro",
        "ru",
        "sk",
        "sl",
        "so",
        "sq",
        "sv",
        "sw",
        "ta",
        "te",
        "th",
        "tl",
        "tr",
        "uk",
        "ur",
        "vi",
        "zh-cn",
        "zh-tw",
    ]
)


@dataclass
class ModelInstalledInfo:
    id: str
    type: str
    registered_name: str
    options: InstallModelIn
    registration_id: RegistrationId
    model_path: Path
    default_model: str | None
    langs_models: dict[SupportedLanguages, str] | None

    def get_info(self) -> ModelInfo:
        """Get info."""
        return ModelInfo(spec=self.options.spec, registration_id=self.registration_id)


def _empty_to_none(value: object) -> object:
    return None if value == "" else value


# An emptied optional form field arrives as "", which means unset (speaches' own default of 300 s).
type OptionalModelTtl = Annotated[Annotated[int, Field(ge=-1)] | None, BeforeValidator(_empty_to_none)]


class SpeachesAIOptions(BaseModel):
    hardware: str | bool | None = None
    keep_alive: OptionalModelTtl = None


class SpeachesAIModelOptions(BaseModel):
    rename: str | None = None
    alias: str | None = None


@dataclass
class InstalledInfo:
    docker: DockerOptions
    models: dict[str, ModelInstalledInfo]
    options: InstallServiceIn
    parsed_options: SpeachesAIOptions
    container_host: str
    container_port: int
    docker_exposed_port: int
    base_url: str


@dataclass
class DownloadedInfo:
    model_path: str


class SrvSpeachesCustomModel(BaseModel):
    id: str
    default_model: str
    langs_models: dict[SupportedLanguages, str] | None = None


@dataclass
class SpeachesModel:
    id: str
    type: ModelType
    size: str = ""
    custom: CustomModelId | None = None
    default_model: str | None = None
    langs_models: dict[SupportedLanguages, str] = field(default_factory=dict[SupportedLanguages, str])


class SpeachesAIService(Base2Service[InstalledInfo, DownloadedInfo]):
    models: dict[str, dict[str, SpeachesModel]]
    _installing: set[tuple[str, str]]

    def _after_init(self) -> None:
        self._installing = set()
        self.load_default_models("default")

    def load_default_models(self, instance: str) -> None:
        """Load default models to instance."""
        self.models[instance] = {}
        for model in _const.audio_speech_models.copy():
            self.models[instance][model[0]] = SpeachesModel(id=model[0], type="tts", size=model[1])
        for model in _const.audio_transcriptions_models.copy():
            self.models[instance][model[0]] = SpeachesModel(id=model[0], type="stt", size=model[1])

    def get_type(self) -> str:
        """Return the service id."""
        return "speaches-ai"

    def get_description(self) -> str:
        """Return the service description."""
        return "Self-hosted Speech-to-Text and Text-to-Speech model runner."

    def get_size(self) -> ServiceSize:
        """Return the service size."""
        sizes = {"cpu": _const.images["cpu"].size}
        if self._supported_gpus:
            sizes["gpu"] = _const.images["gpu"].size
        return sizes

    def get_spec(self) -> ServiceSpecification:
        """Return the service specification."""
        fields = self.add_hardware_field_to_spec()
        fields.append(
            ServiceField(
                type="text",
                name="keep_alive",
                description="The duration that models stay loaded in memory in seconds (default=300, instant_release=0, infinity=-1)",
                required=False,
            )
        )
        return ServiceSpecification(fields=fields)

    def get_model_spec(self) -> ModelSpecification:
        """Return the model specification."""
        return ModelSpecification(
            fields=[
                ModelField(type="text", name="rename", description="Rename the model, hiding its original name", required=False),
                ModelField(
                    type="text", name="alias", description="Additional name for the model, the original name keeps working", required=False
                ),
            ]
        )

    def get_custom_model_spec(self) -> CustomModelSpecification | None:
        """Return the custom model specification or None if custom model is not supported."""
        return CustomModelSpecification(
            fields=[
                CustomModelField(type="text", name="id", description="Model ID", placeholder="my-custom-model"),
                CustomModelField(
                    type="text", name="default_model", description="Default model", placeholder="speaches-ai/piper-en_US-john-medium"
                ),
                CustomModelField(
                    type="map",
                    name="langs_models",
                    description="ISO 639-1 language codes and model id pairs.",
                ),
            ]
        )

    def get_installed_info(self, instance: str) -> bool | InstallServiceProgress | ServiceOptions:
        """Get service installed info."""
        installed = self.get_instance_info(instance).installed
        return self._get_service_installed_info(instance) if installed is None else installed.options.spec

    def _generate_instance_config(self, instance: str, info: InstalledInfo | None, custom: list[CustomModel] | None) -> InstanceConfig:
        return InstanceConfig(
            options=info.options if info else None,
            models=[
                ModelConfig(
                    model_id=x.id,
                    options=x.options,
                    definition=asdict(self.models[instance][x.id])
                    if x.id in self.models[instance]
                    else self._get_persisted_model_definition(instance, x.id),
                )
                for x in info.models.values()
            ]
            if info
            else [],
            custom=custom,
            container_name=info.docker.container_name if info else None,
        )

    def _restore_model_definition(self, instance: str, model_id: str, definition: dict[str, Any]) -> None:
        self.models[instance][model_id] = SpeachesModel(**definition)

    def _get_image(self, gpu: bool) -> DockerImage:
        return _const.images["gpu"] if gpu else _const.images["cpu"]

    def get_default_docker_image_name(self) -> str | None:
        """Return the default docker image, for the upfront platform-compatibility check."""
        return default_gpu_cpu_image_name(self._supported_gpus, self._get_image)

    def _load_download_info(self, data: dict[str, Any]) -> DownloadedInfo:
        return DownloadedInfo(**data)

    def _get_hub_dir(self) -> Path:
        path = self._get_working_dir() / "cache"
        path.mkdir(parents=True, exist_ok=True)
        return path

    async def _validate_update_options(self, options: InstallServiceIn) -> None:
        """Normalize the hardware spec and validate the options spec."""
        if "hardware" not in options.spec:
            options.spec["hardware"] = options.spec.get("gpu", self.docker_service.has_gpu_support)
        options.spec["hardware"] = self.canonicalize_hardware_spec(options.spec["hardware"])
        try_parse_pydantic(SpeachesAIOptions, options.spec)

    async def _install_instance(
        self, instance: str, options: InstallServiceIn, container_name_override: str | None = None
    ) -> PromiseWithProgress[InstalledInfo, StreamChunk]:
        if not self.models.get(instance):
            self.load_default_models(instance)

        await self._validate_update_options(options)
        parsed_options = try_parse_pydantic(SpeachesAIOptions, options.spec)
        volumes = [f"{self._get_hub_dir()}:/home/ubuntu/.cache/huggingface/hub"]

        image = self._get_image(self.is_given_hardware_support_gpu(parsed_options.hardware))

        await self._verify_docker_image(image.name, options.ignore_warnings)

        async def func(stream: Stream[StreamChunk]) -> InstalledInfo:
            await self._download_image_or_set_progress(stream, image)
            self.service_downloaded = True
            stream.emit(StreamChunkProgress(type="progress", stage="install", value=0, data={}))
            subnet = self.docker_service.get_docker_subnet()
            name = f"{self.get_service_id(instance)}"
            env_vars = {"ENABLE_UI": "False"}
            if parsed_options.keep_alive is not None:
                env_vars["STT_MODEL_TTL"] = str(parsed_options.keep_alive)
                env_vars["TTS_MODEL_TTL"] = str(parsed_options.keep_alive)
            docker_options = DockerOptions(
                name=name,
                container_name=container_name_override or self.docker_service.get_docker_container_name(name),
                image=image.name,
                image_port=8000,
                hardware=self.get_specified_hardware_parts(parsed_options.hardware),
                volumes=volumes,
                env_vars=env_vars,
                restart="unless-stopped",
                # user pinned to 0:0 - get_user_for_docker() returns the host uid:gid on rootful Docker, but the image's /home/ubuntu
                # is 0750 ubuntu:ubuntu, so any uid other than 1000 cannot reach the venv on PATH - the container then dies with
                # "exec: uvicorn: not found" (exit 127) and restart-loops, never going healthy.
                user="0:0",
                subnet=subnet,
                healthcheck={
                    "test": "curl --fail http://localhost:8000/health || exit 1",
                    "interval": "30s",
                    "timeout": "10s",
                    "retries": "3",
                    "start_period": "5s",
                },
            )
            try:
                docker_exposed_port, _, _ = await self.docker_service.install_and_run_docker(docker_options)
            except RuntimeError as e:
                stderr = e.args[1][3] if len(e.args) > 1 and len(e.args[1]) > 3 else ""
                if "cuda>=" in stderr or "please update your driver" in stderr:
                    raise HTTPException(
                        503,
                        "Insufficient CUDA version. The speeches-ai GPU service requires cuda>=12.9. "
                        "Please update your NVIDIA GPU drivers to a newer version.",
                    ) from e
                raise
            container_host = self.docker_service.get_container_host(subnet, docker_options)
            container_port = self.docker_service.get_container_port(subnet, docker_exposed_port, docker_options.image_port)
            info = InstalledInfo(
                docker=docker_options,
                models={},
                options=options,
                parsed_options=parsed_options,
                container_host=container_host,
                container_port=container_port,
                docker_exposed_port=docker_exposed_port,
                base_url=get_base_url(container_host, container_port),
            )
            stream.emit(StreamChunkProgress(type="progress", stage="install", value=1, data={}))
            return info

        return PromiseWithProgress(func=func)

    async def _uninstall_instance(self, instance: str, options: UninstallServiceIn) -> None:
        installed = self.get_instance_info(instance).installed
        if installed:
            models = list(installed.models.copy().values())

            for model in models:
                if model.type == "tts":
                    self.endpoint_registry.unregister_audio_speech(model.registered_name, model.registration_id)
                if model.type == "stt":
                    self.endpoint_registry.unregister_audio_transcriptions(model.registered_name, model.registration_id)

            await asyncio.gather(
                *[
                    self._uninstall_model(instance, model.id, UninstallModelIn(purge=options.purge))
                    for model in models
                    if not self.is_model_installed_in_other_instance(instance, model.id)
                ]
            )

            await self.docker_service.uninstall_docker(installed.docker)

        self.instances_info[instance].installed = None

        if options.purge:
            if not any(i.installed for i in self.instances_info.values()):
                self.service_downloaded = False
                for image in _const.images.values():
                    await self.docker_service.remove_image(image.name)
                await self._clear_working_dir()
                self.models_downloaded = {}

            if instance == "default":
                self.instances_info["default"] = Instance(None, None, {}, InstanceConfig())
            else:
                del self.instances_info[instance]

    async def stop_instance(self, instance: str) -> None:
        """Stop the Speaches AI service Docker container."""
        installed = self.get_instance_info(instance).installed
        if not installed:
            return
        await self._stop_docker(installed.docker)

    def get_docker_options(self, instance: str, model_id: str | None) -> DockerOptions:
        """Return the resolved DockerOptions for this instance/model."""
        info = self.get_instance_installed_info(instance)
        if model_id:
            raise HTTPException(400, "Docker is not bound with this object")

        return info.docker

    def service_has_docker(self) -> bool:
        """Return true when docker is started when service is installed."""
        return True

    def _add_custom_model(self, instance: str, model: CustomModel) -> None:
        parsed = try_parse_pydantic(SrvSpeachesCustomModel, model.data)

        if not self.models.get(instance):
            self.models[instance] = {}

        if parsed.id in self.models[instance]:
            raise HTTPException(400, "Model with given id already exists.")

        if parsed.langs_models:
            for lang_model_id in parsed.langs_models.values():
                if lang_model_id not in self.models[instance]:
                    raise HTTPException(400, f"Model {lang_model_id} not found")
        if parsed.default_model and parsed.default_model not in self.models[instance]:
            raise HTTPException(400, f"Model {parsed.default_model} not found")

        self.models[instance][parsed.id] = SpeachesModel(
            id=parsed.id,
            type="tts",
            custom=model.id,
            default_model=parsed.default_model,
            langs_models=parsed.langs_models if parsed.langs_models else {},
        )

    def _remove_custom_model(self, instance: str, model: CustomModel) -> None:
        installed = self.get_instance_info(instance).installed
        parsed = try_parse_pydantic(SrvSpeachesCustomModel, model.data)
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
                            type=model.type,
                            installed=installed,
                            downloaded=model_id in self.models_downloaded,
                            size=model.size,
                            custom=model.custom,
                            spec=self.get_model_spec(),
                            has_docker=False,
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
            type=model.type,
            installed=installed,
            downloaded=model_id in self.models_downloaded,
            size=model.size,
            spec=self.get_model_spec(),
            has_docker=False,
        )

    @staticmethod
    def choose_model_for_language(text: str, default_model: str, langs_models: dict[str, str]) -> ModelId:
        """Choose model for language."""
        text_parts = text.split(" ")
        text_sample = " ".join(text_parts[0:9]) if len(text_parts) > 10 else text
        detected_language = detect(text_sample)  # type: ignore
        if isinstance(detected_language, str) and (model := langs_models.get(detected_language)):
            return model

        return default_model

    async def _download_model(self, stream: Stream[StreamChunk], model: SpeachesModel, model_id: str, model_dir: Path) -> None:
        progress = Progress(convert_size_to_bytes(model.size) or 0)
        stream.emit(StreamChunkProgress(type="progress", stage="download", value=0, data={}))
        async for packet in self.model_downloader.hugging_face_repo_with_blobs_downloader.download(model_id, model_dir):
            if isinstance(packet, DownloadedPacket) and packet.downloaded_bytes_size != 0:
                progress.add_to_actual_value(packet.downloaded_bytes_size)
                stream.emit(
                    StreamChunkProgress(type="progress", stage="download", value=progress.get_percentage(), data=progress.to_data())
                )
            elif isinstance(packet, PreDownloadPacket):
                if max := packet.file_bytes_size:
                    progress.set_max_value(max)

        stream.emit(StreamChunkProgress(type="progress", stage="download", value=1, data={}))

    async def _download_model_or_set_progress(
        self, stream: Stream[StreamChunk], model: SpeachesModel, model_id: str, model_dir: Path
    ) -> None:
        if model_id not in self.models_download_progress:
            self.models_download_progress[model_id] = stream
            try:
                await self._download_model(stream, model, model_id, model_dir)
            finally:
                del self.models_download_progress[model_id]
        else:
            chunk: StreamChunk
            async for chunk in self.models_download_progress[model_id].as_generator():
                if chunk.get("type") == "progress" and chunk.get("stage") == "download":
                    stream.emit(chunk)
                else:
                    break

    async def _install_model(  # noqa: C901
        self,
        instance: str,
        model_id: str,
        options: InstallModelIn,
        container_name_override: str | None = None,  # noqa: ARG002
    ) -> PromiseWithProgress[InstallModelOut, StreamChunk]:
        parsed_model_options = try_parse_pydantic(SpeachesAIModelOptions, options.spec) if options.spec else SpeachesAIModelOptions()
        info = self.get_instance_installed_info(instance)

        if not self.models.get(instance):
            self.models[instance] = {}

        key = (instance, model_id)
        if model_id in info.models or key in self._installing:
            return PromiseWithProgress(value=InstallModelOut(status="OK", details="Already installed or being installed right now."))

        if model_id not in self.models[instance]:
            raise HTTPException(400, "Model not found")

        model = self.models[instance][model_id]
        self._installing.add(key)

        async def func(stream: Stream[StreamChunk]) -> InstallModelOut:
            try:
                model_dir = Path()
                model_id_fixed = f"models--{model_id.replace('/', '--')}"
                models_dir = self._get_hub_dir()
                model_dir = models_dir / model_id_fixed
                model_dir.mkdir(parents=True, exist_ok=True)

                if not model.custom and not model.langs_models:
                    await self._download_model_or_set_progress(stream, model, model_id, model_dir)

                stream.emit(StreamChunkProgress(type="progress", stage="install", value=0, data={}))

                registered_name = parsed_model_options.rename if parsed_model_options.rename else model_id
                registered_alias = parsed_model_options.alias
                info.models[model_id] = model_info = ModelInstalledInfo(
                    id=model_id,
                    type=model.type,
                    registered_name=registered_name,
                    options=options,
                    registration_id="",
                    model_path=model_dir,
                    default_model=model.default_model or None,
                    langs_models=model.langs_models or None,
                )
                try:
                    if model.type == "tts":

                        async def on_request(body: CreateSpeechRequest, request: Request | None) -> StreamingResponse:
                            body.voice = "" if not body.voice else body.voice  # Fix for 422 status code
                            proxy_options = ProxyOptions(url=f"{info.base_url}/v1/audio/speech", rewrite_model_to=model_id)
                            if model.id and model.default_model and model.langs_models:
                                proxy_options.rewrite_model_to = body.model = self.choose_model_for_language(
                                    body.input, model.default_model, model.langs_models
                                )
                            return await post_json(body, proxy_options, request)

                        model_info.registration_id = self.endpoint_registry.register_audio_speech(
                            model=registered_name,
                            props=ModelProps(private=True, type=model.type, endpoints=TTS_ENDPOINTS),
                            endpoint=SimpleEndpoint(on_request=on_request),
                            registration_options=RegistrationOptions(origin="local", alias=registered_alias, owned_by=self.get_type()),
                        )
                    if model.type == "stt":
                        model_info.registration_id = self.endpoint_registry.register_audio_transcriptions_as_proxy(
                            model=registered_name,
                            props=ModelProps(private=True, type=model.type, endpoints=STT_ENDPOINTS),
                            options=ProxyOptions(url=f"{info.base_url}/v1/audio/transcriptions", rewrite_model_to=model_id),
                            registration_options=RegistrationOptions(origin="local", alias=registered_alias, owned_by=self.get_type()),
                        )
                    stream.emit(StreamChunkProgress(type="progress", stage="install", value=1, data={}))
                    self.models_downloaded[model_id] = DownloadedInfo(model_path=str(model_dir))
                    return InstallModelOut(status="OK", details="Installed")
                except Exception:
                    if info.models.get(model_id) is model_info:
                        info.models.pop(model_id, None)
                    raise
            finally:
                self._installing.discard(key)

        return PromiseWithProgress(func=func)

    async def _uninstall_model(self, instance: str, model_id: str, options: UninstallModelIn) -> None:
        info = self.get_instance_installed_info(instance)
        if model_id in info.models:
            model = info.models[model_id]

            for check_model_id, check_model in info.models.items():
                if (isinstance(check_model.langs_models, dict) and model_id in check_model.langs_models.values()) or (
                    check_model.default_model and model_id in check_model.default_model
                ):
                    raise HTTPException(409, f"Model is used in custom model {check_model_id}. Remove it first.")

            del info.models[model_id]
            if model.type == "tts":
                self.endpoint_registry.unregister_audio_speech(model.registered_name, model.registration_id)
            if model.type == "stt":
                self.endpoint_registry.unregister_audio_transcriptions(model.registered_name, model.registration_id)

        if options.purge and model_id in self.models_downloaded:
            if self.models_downloaded[model_id].model_path and self.models_downloaded[model_id].model_path != ".":
                shutil.rmtree(Path(self.models_downloaded[model_id].model_path))
            del self.models_downloaded[model_id]
