# DeepFellow Software Framework.
# Copyright © 2025 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Stable diffusion service."""

import asyncio
import base64
import io
import json
import platform
import re
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal, NamedTuple, NotRequired, TypedDict

import aiofiles
from aiohttp import ClientSession
from fastapi import HTTPException, Request
from PIL import Image
from pydantic import BaseModel, Field, ValidationError

from server.applicationcontext import get_base_url
from server.docker import (
    DockerImage,
    DockerOptions,
)
from server.endpointregistry import EndpointCallback, ProxyOptions, RegistrationId, SimpleEndpoint
from server.models.api import ImagesRequest, ModelProps
from server.models.models import (
    CustomModelField,
    CustomModelId,
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
from server.models.services import InstallServiceIn, ServiceField, ServiceOptions, ServiceSize, ServiceSpecification, UninstallServiceIn
from server.services.base2_service import Base2Service, CustomModel, ModelConfig, ServiceConfig
from server.utils.core import (
    PromiseWithProgress,
    Stream,
    StreamChunk,
    StreamChunkProgress,
    convert_size_to_bytes,
    try_parse_pydantic,
)
from server.utils.loading import Progress

ModelType = Literal["txt2img", "lora"]

# Type names by sdnext models folders names.
# Model type must be appropriate folder.
FileType = Literal[
    "chaiNNer",
    "Codeformer",
    "control",
    "Diffusers",
    "embeddings",
    "ESRGAN",
    "GFPGAN",
    "huggingface",
    "LDSR",
    "Lora",
    "ONNX",
    "REALESGRAN",
    "SCUNet",
    "Stable-diffusion",
    "styles",
    "SwinIR",
    "Text-encoder",
    "tunable",
    "UNET",
    "VAE",
    "wildcards",
    "YOLO",
]


class StableDiffusionModel(BaseModel):
    filetype: FileType
    type: ModelType
    url: str
    filename: str
    model_url: str | None = None
    size: str
    custom: CustomModelId | None = None


class StableDiffusionCustomModel(BaseModel):
    id: str
    filetype: FileType
    url: str
    filename: str
    size: str


class StableDiffusionConst(BaseModel):
    image_gpu: DockerImage
    image_cpu: DockerImage
    models: dict[str, StableDiffusionModel]


_const = StableDiffusionConst(
    image_gpu=DockerImage(
        name="vladmandic/sdnext-cuda:latest@sha256:10f9ab600c245b9ce83be5a55abb64b46e115ef8508f5ffc69eed8fa0fc28ce8", size="9.4 GB"
    ),
    image_cpu=DockerImage(
        name="vladmandic/sdnext-cuda:latest@sha256:10f9ab600c245b9ce83be5a55abb64b46e115ef8508f5ffc69eed8fa0fc28ce8", size="9.4 GB"
    ),
    models={
        "Plant Milk Walnut": StableDiffusionModel(
            filetype="Stable-diffusion",
            type="txt2img",
            url="https://civitai.com/api/download/models/1714002?type=Model&format=SafeTensor&size=pruned&fp=fp16",
            filename="plantMilkModelSuite_walnut.safetensors",
            model_url="https://civitai.com/models/1162518?modelVersionId=1714002",
            size="6.46GB",
        ),
        "Fantastic Landscapes": StableDiffusionModel(
            filetype="Lora",
            type="lora",
            url="https://civitai.com/api/download/models/180314?type=Model&format=SafeTensor",
            filename="FantasticLandscapes.safetensors",
            model_url="https://civitai.com/models/160272/lora-fantastic-landscapes",
            size="36.11MB",
        ),
        "Pixiv AenuaV1": StableDiffusionModel(
            filetype="Lora",
            type="lora",
            url="https://civitai.com/api/download/models/694191?type=Model&format=SafeTensor",
            filename="AenuaV1.safetensors",
            model_url="https://civitai.com/models/620966?modelVersionId=694191",
            size="435.34MB",
        ),
        "Minecraft square style": StableDiffusionModel(
            filetype="Lora",
            type="lora",
            url="https://civitai.com/api/download/models/1145880?type=Model&format=SafeTensor",
            filename="minecraft filter [IL]_1.safetensors",
            model_url="https://civitai.com/models/113741/minecraft-square-style",
            size="54.77MB",
        ),
        "SDXL Lighting 8 step": StableDiffusionModel(
            filetype="Stable-diffusion",
            type="txt2img",
            url="https://huggingface.co/ByteDance/SDXL-Lightning/resolve/main/sdxl_lightning_8step.safetensors?download=true",
            filename="sdxl_lightning_8step.safetensors",
            model_url="https://huggingface.co/ByteDance/SDXL-Lightning",
            size="6.94GB",
        ),
        "Semi-realistic": StableDiffusionModel(
            filetype="Stable-diffusion",
            type="txt2img",
            url="https://civitai.com/api/download/models/2202259?type=Model&format=SafeTensor&size=pruned&fp=fp16",
            filename="semi-realistic.safetensors",
            model_url="https://civitai.com/models/1945811/semi-real-illustrious-or-mm",
            size="6.94GB",
        ),
        "CyberRealistic-XL-FP16": StableDiffusionModel(
            filetype="Stable-diffusion",
            type="txt2img",
            url="https://civitai.com/api/download/models/2152184?type=Model&format=SafeTensor&size=pruned&fp=fp16",
            filename="cyberrealistic-xl-fp16.safetensors",
            model_url="https://civitai.com/models/312530/cyberrealistic-xl",
            size="6.46GB",
        ),
        "CyberRealistic-XL-FP32": StableDiffusionModel(
            filetype="Stable-diffusion",
            type="txt2img",
            url="https://civitai.com/api/download/models/2152184?type=Model&format=SafeTensor&size=pruned&fp=fp32",
            filename="cyberrealistic-xl-fp32.safetensors",
            model_url="https://civitai.com/models/312530/cyberrealistic-xl",
            size="12.92GB",
        ),
        "yomama-2.5D": StableDiffusionModel(
            filetype="Stable-diffusion",
            type="txt2img",
            url="https://civitai.com/api/download/models/1085088?type=Model&format=SafeTensor&size=pruned&fp=fp16",
            filename="yomama-25d.safetensors",
            model_url="https://civitai.com/models/959233/yomama-25d-illustrious-pony?modelVersionId=1085088",
            size="6.46GB",
        ),
    },
)


class RemoveBackgroundRequest(BaseModel):
    input_image: str
    model: str = "u2net"
    return_mask: bool = False
    alpha_matting: bool = False
    alpha_matting_foreground_threshold: int = 240
    alpha_matting_background_threshold: int = 10
    alpha_matting_erode_size: int = 10
    refine: bool = False


class ModelInstalledInfo(BaseModel):
    id: str
    type: str
    registered_name: str
    options: InstallModelIn
    model_path: Path
    registration_id: RegistrationId

    def get_info(self) -> ModelInfo:
        """Get info."""
        return ModelInfo(spec=self.options.spec, registration_id=self.registration_id)


class SDOptions(BaseModel):
    gpu: bool
    expose_api_at_prefix: Annotated[str, Field(pattern=r"^[a-zA-Z0-9_-]+$")] = ""


class SDModelOptions(BaseModel):
    alias: str | None = None


class InstalledInfo:
    def __init__(
        self,
        docker: DockerOptions,
        models: dict[str, ModelInstalledInfo],
        options: InstallServiceIn,
        parsed_options: SDOptions,
        container_host: str,
        container_port: int,
        docker_exposed_port: int,
        proxy_registration_id: RegistrationId | None,
    ):
        self.docker = docker
        self.models = models
        self.options = options
        self.parsed_options = parsed_options
        self.container_host = container_host
        self.container_port = container_port
        self.docker_exposed_port = docker_exposed_port
        self.base_url = get_base_url(self.container_host, self.container_port)
        self.proxy_registration_id = proxy_registration_id


class DefaultSdNextConfig(NamedTuple):
    samples_format: str = "png"
    samples_save: bool = False
    keep_incomplete: bool = False
    save_selected_only: bool = False


class StableDiffusionService(Base2Service[InstalledInfo]):
    models: dict[str, StableDiffusionModel]

    def _after_init(self) -> None:
        self.models = _const.models.copy()

    def get_id(self) -> str:
        """Return the service id."""
        return "stable-diffusion"

    def get_description(self) -> str:
        """Return the service description."""
        return "Self-hosted graphic models runner."

    def get_size(self) -> ServiceSize:
        """Return the service size."""
        if _const.image_gpu.name != _const.image_gpu.name:
            return {"cpu": _const.image_cpu.size, "gpu": _const.image_gpu.size}

        return _const.image_cpu.size

    def get_spec(self) -> ServiceSpecification:
        """Return the service specification."""
        return ServiceSpecification(
            fields=[
                ServiceField(type="bool", name="gpu", description="Run on GPU", required=False, default=self._has_gpu_for_spec()),
                ServiceField(type="text", name="expose_api_at_prefix", description="Expose SD API at prefix", required=False, default="sd"),
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
        return CustomModelSpecification(
            fields=[
                CustomModelField(type="text", name="id", description="Model ID", placeholder="my-custom-model"),
                CustomModelField(type="oneof", name="filetype", description="File type", values=["Stable-diffusion", "Lora"]),
                CustomModelField(
                    type="text",
                    name="url",
                    description="Model File URL",
                    placeholder="https://civitai.com/api/download/models/123456789?type=Model&format=SafeTensor&size=pruned&fp=fp16",
                ),
                CustomModelField(type="text", name="filename", description="Model filename", placeholder="mymodel.safetensors"),
                CustomModelField(type="text", name="size", description="Model size", placeholder="1GB"),
            ]
        )

    def get_installed_info(self) -> bool | ServiceOptions:
        """Get service installed info."""
        return False if self.installed is None else self.installed.options.spec

    def _generate_config(self, info: InstalledInfo | None) -> ServiceConfig:
        return ServiceConfig(
            options=info.options if info else None,
            models=[ModelConfig(model_id=x.id, options=x.options) for x in info.models.values()] if info else [],
            custom=self.custom,
        )

    async def update_config(self) -> None:
        """Edit SD Next config file."""
        file_path = self._get_working_data_dir() / "config.json"
        data = DefaultSdNextConfig()._asdict()

        async def read_file() -> dict[Any, Any]:
            try:
                async with aiofiles.open(file_path, encoding="utf-8") as f:
                    content = await f.read()
            except FileNotFoundError:
                content = "{}"
            try:
                obj = json.loads(content)
                return obj if isinstance(obj, dict) else {}  # pyright: ignore[reportUnknownVariableType]
            except Exception:
                return {}

        old_content = await read_file()

        async with aiofiles.open(file_path, mode="w+") as f:
            data.update(old_content)
            await f.write(json.dumps(data, indent=4))

    async def _install_core(self, options: InstallServiceIn) -> PromiseWithProgress[InstalledInfo, StreamChunk]:
        if "gpu" not in options.spec:
            options.spec["gpu"] = self.docker_service.has_gpu_support
        if platform.system() == "Darwin":
            raise HTTPException(400, "Stable Diffusion is not supported on macOS")
        parsed_options = try_parse_pydantic(SDOptions, options.spec)
        volumes = [
            f"{self._get_working_models_dir()}:/mnt/models",
            f"{self._get_working_data_dir()}:/mnt/data",
            f"{self._get_working_logs()}:/app/sdnext.log",
        ]
        if parsed_options.gpu:
            image = _const.image_gpu
            if not self.docker_service.has_gpu_support:
                raise HTTPException(400, "Docker doesn't support GPU on this machine.")
        else:
            image = _const.image_cpu

        async def func(stream: Stream[StreamChunk]) -> InstalledInfo:
            stream.emit(StreamChunkProgress(type="progress", value=0))
            await self._docker_pull(image, stream)
            stream.emit(StreamChunkProgress(type="progress", value=0.99))
            await self.update_config()
            subnet = self.docker_service.get_docker_subnet()
            docker_options = DockerOptions(
                name="stable-diffusion",
                container_name=self.docker_service.get_docker_container_name("stable-diffusion"),
                image=image.name,
                env_vars={
                    "SD_DOCS": "true",
                },
                image_port=7860,
                use_gpu=parsed_options.gpu,
                volumes=volumes,
                restart="unless-stopped",
                subnet=subnet,
                user=await self.docker_service.get_user_for_docker(),
                healthcheck={
                    "test": "curl --fail http://localhost:7860/sdapi/v1/status || exit 1",
                    "interval": "40s",
                    "timeout": "10s",
                    "retries": "3",
                    "start_period": "60s",
                },
            )
            docker_exposed_port = await self.docker_service.install_and_run_docker(docker_options)

            host = self.docker_service.get_container_host(subnet, docker_options.name)
            port = self.docker_service.get_container_port(subnet, docker_exposed_port, docker_options.image_port)

            proxy_registration_id = (
                self.endpoint_registry.register_custom_endpoint_as_proxy(
                    parsed_options.expose_api_at_prefix,
                    ModelProps(private=False),
                    ProxyOptions(get_base_url(host, port)),
                    registration_options=None,
                )
                if parsed_options.expose_api_at_prefix
                else None
            )

            info = InstalledInfo(
                docker=docker_options,
                models={},
                options=options,
                parsed_options=parsed_options,
                container_host=host,
                container_port=port,
                docker_exposed_port=docker_exposed_port,
                proxy_registration_id=proxy_registration_id,
            )
            stream.emit(StreamChunkProgress(type="progress", value=1))
            return info

        return PromiseWithProgress(func=func)

    async def _uninstall(self, options: UninstallServiceIn) -> None:
        info = self._check_installed()
        if info.parsed_options.expose_api_at_prefix and info.proxy_registration_id:
            self.endpoint_registry.unregister_custom_endpoint(
                info.parsed_options.expose_api_at_prefix,
                info.proxy_registration_id,
            )
        for model in info.models.copy().values():
            if model.type == "txt2img":
                self.endpoint_registry.unregister_image_generations(model.registered_name, model.registration_id)
        self.installed = None
        await self.docker_service.uninstall_docker(info.docker)
        if options.purge:
            await self._clear_working_dir()

    async def stop(self) -> None:
        """Stop the Stable Diffusion service Docker container."""
        info = self.installed
        if not info:
            return
        await self._stop_docker(info.docker)

    def get_docker_compose_file_path(self, model_id: str | None) -> Path:
        """Get docker compose file path."""
        info = self.installed
        if not info:
            raise HTTPException(400, "Service not installed")
        if model_id:
            raise HTTPException(400, "Docker is not bound with this object")
        return self.docker_service.get_docker_compose_file_path(info.docker.name)

    def service_has_docker(self) -> bool:
        """Return true when docker is started when service is installed."""
        return True

    def _add_custom_model(self, model: CustomModel) -> None:
        parsed = try_parse_pydantic(StableDiffusionCustomModel, model.data)
        if parsed.id in self.models:
            raise HTTPException(400, "Model with given id already exists.")
        self.models[parsed.id] = StableDiffusionModel(
            filetype=parsed.filetype,
            type="txt2img" if parsed.filetype == "Stable-diffusion" else "lora",
            url=parsed.url,
            filename=parsed.filename,
            model_url=None,
            size=parsed.size,
            custom=model.id,
        )

    def _remove_custom_model(self, model: CustomModel) -> None:
        parsed = try_parse_pydantic(StableDiffusionCustomModel, model.data)
        if self.installed and parsed.id in self.installed.models:
            raise HTTPException(400, "Cannot remove custom model, it is in use, uninstall it first.")
        del self.models[parsed.id]

    async def list_models(self, filters: ListModelsFilters) -> ListModelsOut:
        """List models."""
        info = self._check_installed()
        out_list: list[RetrieveModelOut] = []
        for model_id, model in self.models.items():
            installed = info.models[model_id].get_info() if model_id in info.models else False
            if filters.installed is None or filters.installed == installed:
                out_list.append(
                    RetrieveModelOut(
                        id=model_id,
                        service=self.get_id(),
                        type=model.type,
                        installed=installed,
                        size=model.size,
                        custom=model.custom,
                        spec=self.get_model_spec(),
                        has_docker=False,
                    )
                )
        return ListModelsOut(list=out_list)

    async def get_model(self, model_id: str) -> RetrieveModelOut:
        """Get the model."""
        info = self._check_installed()
        if model_id not in self.models:
            raise HTTPException(status_code=400, detail="Model not found")
        model = self.models[model_id]
        installed = info.models[model_id].get_info() if model_id in info.models else False
        return RetrieveModelOut(
            id=model_id,
            service=self.get_id(),
            type=model.type,
            installed=installed,
            size=model.size,
            custom=model.custom,
            spec=self.get_model_spec(),
            has_docker=False,
        )

    async def refresh_checkpoints(self, client: ClientSession, base_url: str) -> None:
        """Refresh SD Next checkpoints models."""
        await client.post(base_url + "/sdapi/v1/refresh-checkpoints")  # type: ignore

    async def refresh_vae(self, client: ClientSession, base_url: str) -> None:
        """Refresh SD Next vae models."""
        await client.post(base_url + "/sdapi/v1/refresh-vae")  # type: ignore

    async def refresh_loras(self, client: ClientSession, base_url: str) -> None:
        """Refresh SD Next loras models."""
        await client.post(base_url + "/sdapi/v1/refresh-loras")  # type: ignore

    async def refresh_models(self) -> None:
        """Refresh models in SD Next."""
        info = self._check_installed()
        async with ClientSession(info.base_url) as client:
            refresh_functions = [self.refresh_checkpoints, self.refresh_vae, self.refresh_loras]
            tasks = [asyncio.create_task(f(client, info.base_url)) for f in refresh_functions]
            await asyncio.gather(*tasks)

    async def _install_model(self, model_id: str, options: InstallModelIn) -> PromiseWithProgress[InstallModelOut, StreamChunk]:
        parsed_model_options = try_parse_pydantic(SDModelOptions, options.spec) if options.spec else SDModelOptions()
        info = self._check_installed()
        if model_id in info.models:
            return PromiseWithProgress(value=InstallModelOut(status="OK", details="Already installed"))
        if model_id not in self.models:
            raise HTTPException(400, "Model not found")
        model = self.models[model_id]

        async def func(stream: Stream[StreamChunk]) -> InstallModelOut:
            model_dir = self._get_working_models_dir() / model.filetype
            progress = Progress(convert_size_to_bytes(model.size) or 0)
            local_model_path: Path | None = None
            model_filename = ""
            async for packet in self.model_downloader.download(model.url, model_dir, model.filename):
                if packet.local_path and packet.filename and not local_model_path and not model_filename:
                    local_model_path = packet.local_path
                    model_filename = packet.filename
                elif packet.downloaded_bytes_size != 0:
                    progress.add_to_actual_value(packet.downloaded_bytes_size)

                stream.emit(StreamChunkProgress(type="progress", value=progress.get_percentage() * 0.99))

            if not local_model_path:
                raise HTTPException(400, "Something went wrong with downloading")

            registered_name = parsed_model_options.alias if parsed_model_options.alias else model_id
            info.models[model_id] = model_info = ModelInstalledInfo(
                id=model_id,
                type=model.type,
                registered_name=registered_name,
                options=options,
                model_path=local_model_path.absolute(),
                registration_id="",
            )
            model_filename = model.filename.split(".")[0]
            if model.type == "txt2img":
                model_info.registration_id = self.endpoint_registry.register_image_generations(
                    model=registered_name,
                    props=ModelProps(private=True),
                    endpoint=SimpleEndpoint(on_request=_stable_diffusion_handler(info.base_url, model_filename)),
                    registration_options=None,
                )

            await self.refresh_models()
            stream.emit(StreamChunkProgress(type="progress", value=1))
            return InstallModelOut(status="OK", details="Installed")

        return PromiseWithProgress(func=func)

    async def _uninstall_model(self, model_id: str, options: UninstallModelIn) -> None:
        info = self._check_installed()
        if model_id not in info.models:
            return
        model = info.models[model_id]
        del info.models[model_id]
        if model.type == "txt2img":
            self.endpoint_registry.unregister_image_generations(model.registered_name, model.registration_id)

        if options.purge:
            model.model_path.unlink()

        await self.refresh_models()

    def _get_working_models_dir(self) -> Path:
        path = self._get_working_dir() / "models"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _get_working_data_dir(self) -> Path:
        path = self._get_working_dir() / "data"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _get_working_logs(self) -> Path:
        path = self._get_working_dir() / "logs"
        path.mkdir(parents=True, exist_ok=True)
        file = path / "sdnext.log"
        with Path.open(file, "a"):
            pass

        return file


class InputTokensDetails(BaseModel):
    image_tokens: int
    text_tokens: int


class Usage(BaseModel):
    input_tokens: int
    input_tokens_details: InputTokensDetails
    output_tokens: int
    total_tokens: int


class B64Data(BaseModel):
    b64_json: str
    revised_prompt: str = ""


class ImagesResponse(BaseModel):
    data: list[B64Data]
    created: int = int(datetime.now(UTC).timestamp())


class OverrideSettings(BaseModel):
    sd_model_checkpoint: str = "your_model_name.safetensors"


class StableDiffusionInputSettings(BaseModel):
    sd_model_checkpoint: str
    prompt: str = ""  # Consider generate from prompt
    negative_prompt: str = ""  # Consider generate from prompt
    sampler_name: str = "DPM++ 2M"
    hr_sampler_name: str = "Kerras"
    clip_skip: int = 2
    steps: Annotated[int, Field(ge=1, le=50)] = 20
    n_iter: Annotated[int, Field(ge=1, le=4)] = 1
    cfg_scale: Annotated[float, Field(ge=1, le=30)] = 7
    width: Annotated[int, Field(ge=64, le=2048)] = 512
    height: Annotated[int, Field(ge=64, le=2048)] = 512


class QualityLevel(Enum):
    high = 35
    medium = 20
    auto = 20  # noqa: PIE796
    low = 15


class StableDiffusionOptions(TypedDict):
    sd_model_checkpoint: NotRequired[str]
    prompt: NotRequired[str]
    negative_prompt: NotRequired[str]
    sampler_name: NotRequired[str]
    hr_sampler_name: NotRequired[str]
    clip_skip: NotRequired[int]
    steps: NotRequired[int]
    n_iter: NotRequired[int]
    cfg_scale: NotRequired[int]
    width: NotRequired[int]
    height: NotRequired[int]


def split_text_to_json_and_prompt(text: str) -> tuple[StableDiffusionOptions, str]:
    """Split text into prompt and JSON data for settings."""
    # With this regex we get two matches
    # (one match with whole <sd>example text</sd>)
    # and one with what inside sd html element like example text
    json_candidates = re.findall(r"(<sd>(.*?)</sd>)", text, re.DOTALL)
    if not json_candidates:
        # No json found in text
        return {}, text

    extracted_text, json_candidate = json_candidates[0]

    try:
        # Convert text to dict
        json_output: StableDiffusionOptions = json.loads(json_candidate)
    except json.JSONDecodeError:
        # Settings json not correct, return error.
        raise HTTPException(
            422, "Something went wrong with stable diffusion settings.Please place correct json in prompt in html <sd></sd> tags."
        ) from None

    # Remove whole sd html element from prompt text
    remaining_text = text.replace(extracted_text, "")

    return json_output, remaining_text


def get_image_size(size: str) -> tuple[int, int]:
    """Convert size str to width and height.

    Example:
        input: "517x768"
        output: (517, 768)

    """
    sizes = size.split("x")
    if len(sizes) != 2:
        raise ValueError("Incorect size dimension. Should be 2. Format should be like '517x768'.")

    try:
        width, height = int(sizes[0]), int(sizes[1])
    except ValueError:
        raise ValueError("Sizes are not ints. Format should be like '517x768'") from None

    return width, height


def convert_b64png_to_b64jpg(b64_png: str, quality: int = 90) -> str:
    """Convert a Base64 encoded PNG image string to a Base64 encoded JPG image string.

    Args:
        b64_png: The Base64 encoded PNG string. Should NOT include the "data:image/png;base64," prefix.
        quality: The quality of the JPG image, between 0 and 100, with 100 being the highest quality.

    Returns:
        str: The Base64 encoded JPG string, or None if an error occurs.
    """
    png_bytes = base64.b64decode(b64_png)

    image_stream = io.BytesIO(png_bytes)
    img = Image.open(image_stream)

    if img.mode == "RGBA":
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img = background
    else:  # img.mode == 'RGB'
        img = img.convert("RGB")

    output_buffer = io.BytesIO()
    img.save(output_buffer, format="JPEG", quality=quality)

    base64_jpg_bytes = base64.b64encode(output_buffer.getvalue())

    return base64_jpg_bytes.decode("utf-8")


def convert_b64png_to_b64webp(b64_png: str, quality: int = 90) -> str:
    """Convert a Base64 encoded PNG image string to a Base64 encoded WebP image string.

    Args:
        b64_png: The Base64 encoded PNG string. Should NOT include the "data:image/png;base64," prefix.
        quality: The quality of the JPG image, between 0 and 100, with 100 being the highest quality.

    Returns:
        str: The Base64 encoded WebP string, or None if an error occurs.
    """
    png_bytes = base64.b64decode(b64_png)

    image_stream = io.BytesIO(png_bytes)
    img = Image.open(image_stream)

    output_buffer = io.BytesIO()
    img.save(output_buffer, format="WEBP", quality=quality)

    base64_webp_bytes = base64.b64encode(output_buffer.getvalue())

    return base64_webp_bytes.decode("utf-8")


def _get_in_format(img: str, format: str = "png", quality: int = 95) -> str:
    match format:
        case "jpg":
            return convert_b64png_to_b64jpg(img, quality)
        case "jpeg":
            return convert_b64png_to_b64jpg(img, quality)
        case "webp":
            return convert_b64png_to_b64webp(img, quality)
        case "png":
            return img
        case _:
            raise ValueError("Not Supported format")


def _add_body_config(settings_original: StableDiffusionOptions, body: ImagesRequest, remaining_text: str) -> StableDiffusionOptions:
    settings = settings_original.copy()

    if body.response_format is not None and body.response_format != "b64_json":
        raise ValueError("Response format not supported", body.response_format)

    if body.partial_images is not None and body.partial_images != 0:
        raise ValueError("Partial images not supported")

    if body.stream:
        raise ValueError("Stream not supported")

    if not settings.get("prompt") and remaining_text:
        settings["prompt"] = remaining_text

    if not settings.get("n_iter") and body.n:
        settings["n_iter"] = body.n

    if not settings.get("width") and not settings.get("height") and body.size and body.size != "auto":
        settings["width"], settings["height"] = get_image_size(body.size)

    if not settings.get("steps") and body.quality:
        settings["steps"] = QualityLevel[body.quality].value

    return settings


async def remove_background_from_img(base_url: str, img: str) -> str:
    """Remove background from img."""
    rm_bg_url = f"{base_url}/rembg/"
    request_model = RemoveBackgroundRequest(input_image=img)
    async with ClientSession(rm_bg_url) as client:
        response = await client.post(rm_bg_url, json=request_model.model_dump(), timeout=600)  # type: ignore
        response_json = await response.json()
        try:
            return response_json.get("image", "")
        except Exception:
            return img


def _stable_diffusion_handler(base_url: str, model_filename: str) -> EndpointCallback[ImagesRequest]:
    async def handler(body: ImagesRequest, _req: Request | None) -> ImagesResponse:
        settings_raw, remaining_text = split_text_to_json_and_prompt(body.prompt)
        settings_raw["sd_model_checkpoint"] = model_filename

        try:
            settings_edited = _add_body_config(settings_raw, body, remaining_text)
        except ValueError as err:
            raise HTTPException(422, str(err)) from None

        try:
            settings = StableDiffusionInputSettings(**settings_edited)
        except ValidationError:
            raise HTTPException(422, "Incorrect settings.") from None

        gen_img_url = f"{base_url}/sdapi/v1/txt2img/"
        async with ClientSession(gen_img_url) as client:
            response = await client.post(gen_img_url, json=settings.model_dump(), timeout=600)  # type: ignore

            if response.status != 200:
                raise HTTPException(
                    500, (f"Something went wrong inside Stable Diffusion Web UI.Error {response.status}: {response.content}")
                )

            # try:
            data_raw = await response.json()
            # except Exception:
            #     raise HTTPException(500, "There is no images in Stable Diffusion response.")

        imgs = data_raw["images"]

        if body.background == "transparent" and body.output_format != "jpeg":
            tasks = [asyncio.create_task(remove_background_from_img(base_url, img)) for img in imgs]
            new_imgs: list[str] = await asyncio.gather(*tasks)
            imgs = new_imgs

        try:
            imgs = [_get_in_format(img, body.output_format or "png", body.output_compression or 95) for img in imgs]
        except ValueError:
            raise HTTPException(422, f"Not supported image format: {body.output_format}") from None
        except Exception as exc:
            raise HTTPException(500, f"Something went wrong: {exc}") from None

        return ImagesResponse(data=[B64Data(b64_json=img, revised_prompt=settings.prompt) for img in imgs])

    return handler
