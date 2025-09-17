"""Stable diffusion service."""

import base64
import io
import json
import platform
import re
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Literal, NotRequired, TypedDict

from aiohttp import ClientSession
from fastapi import HTTPException, Request
from PIL import Image
from pydantic import BaseModel, Field, ValidationError

from server.docker import DockerOptions, install_and_run_docker, uninstall_docker
from server.endpointregistry import EndpointCallback, SimpleEndpoint
from server.models.api import ImagesRequest
from server.models.models import InstallModelIn, ListModelsFilters, ListModelsOut, RetrieveModelOut, UninstallModelIn
from server.models.services import InstallServiceIn, ServiceField, ServiceOptions, ServiceSpecification, UninstallServiceIn
from server.services.base2_service import Base2Service, ModelConfig, ServiceConfig
from server.utils.core import Utils

ModelType = Literal["txt2img"]


class StableDiffusionModel(BaseModel):
    type: ModelType
    url: str
    filename: str
    model_url: str | None = None


class StableDiffusionConst(BaseModel):
    image_gpu: str
    image_cpu: str
    models: dict[str, StableDiffusionModel]


_const = StableDiffusionConst(
    image_gpu="ghcr.io/ai-dock/stable-diffusion-webui:latest",
    image_cpu="ghcr.io/ai-dock/stable-diffusion-webui:v2-cpu-22.04-v1.10.1",
    models={
        "HiDream-I1-Full": StableDiffusionModel(
            type="txt2img",
            url="https://civitai.com/api/download/models/509959?type=Model&format=SafeTensor&size=pruned&fp=fp16",
            filename="unstableIllusionPRO_pro.safetensors",
            model_url="https://civitai.com/models/147687/unstable-illusion-pro?modelVersionId=509959",
        )
    },
)


class ModelInstalledInfo(BaseModel):
    id: str
    type: str
    registered_name: str
    options: InstallModelIn
    model_path: Path


class SDOptions(BaseModel):
    gpu: bool


class InstalledInfo:
    def __init__(
        self,
        docker: DockerOptions,
        container_host: str,
        port: int,
        models: dict[str, ModelInstalledInfo],
        options: InstallServiceIn,
        parsed_options: SDOptions,
    ):
        self.docker = docker
        self.container_host = container_host
        self.port = port
        self.models = models
        self.options = options
        self.parsed_options = parsed_options


class StableDiffusionService(Base2Service[InstalledInfo]):
    def get_id(self) -> str:
        """Return the service id."""
        return "stable-diffusion"

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
        if platform.system() == "Darwin":
            raise NotImplementedError("Stable Diffusion is not supported on macOS")
        parsed_options = SDOptions(**options.spec)
        volumes = [
            f"{self._get_working_stable_diffusion_dir()}:/opt/stable-diffusion-webui/models/Stable-diffusion",
            f"{self._get_working_lora_dir()}:/opt/stable-diffusion-webui/models/Lora",
        ]
        image = _const.image_gpu if parsed_options.gpu else _const.image_cpu

        docker_options = DockerOptions(
            name="stable-diffusion",
            image=image,
            env_vars={
                "COMMANDLINE_ARGS": "--listen --api --no-half-vae",  #  --xformers
            },
            image_port=17860,
            use_gpu=parsed_options.gpu,
            volumes=volumes,
            restart="unless-stopped",
        )
        port = await install_and_run_docker(self.application_context, docker_options)
        return InstalledInfo(
            docker=docker_options,
            container_host=self.application_context.get_container_host(docker_options.name),
            port=port,
            models={},
            options=options,
            parsed_options=parsed_options,
        )

    async def _uninstall(self, options: UninstallServiceIn) -> None:
        info = self._check_installed()
        for model in info.models.values():
            if model.type == "txt2img":
                self.endpoint_registry.unregister_image_generations(model.registered_name)
        self.installed = None
        await uninstall_docker(self.application_context, info.docker)
        if options.purge:
            await self._clear_working_dir()

    async def list_models(self, filters: ListModelsFilters) -> ListModelsOut:
        """List models."""
        info = self._check_installed()
        out_list: list[RetrieveModelOut] = []
        for model_id, model in _const.models.items():
            installed = model_id in info.models
            if filters.installed is None or filters.installed == installed:
                out_list.append(RetrieveModelOut(id=model_id, service=self.get_id(), type=model.type, installed=installed))
        return ListModelsOut(list=out_list)

    async def get_model(self, model_id: str) -> RetrieveModelOut:
        """Get the model."""
        info = self._check_installed()
        if model_id not in _const.models:
            raise HTTPException(status_code=400, detail="Model not found")
        model = _const.models[model_id]
        installed = model_id in info.models
        return RetrieveModelOut(id=model_id, service=self.get_id(), type=model.type, installed=installed)

    async def _install_model(self, model_id: str, options: InstallModelIn) -> None:
        info = self._check_installed()
        if model_id in info.models:
            return
        if model_id not in _const.models:
            raise HTTPException(status_code=400, detail="Model not found")
        model = _const.models[model_id]

        local_model_path, _ = await Utils.ensure_model_downloaded(model.url, self._get_working_stable_diffusion_dir(), model.filename)
        registered_name = options.alias if options.alias is not None else model_id
        info.models[model_id] = ModelInstalledInfo(
            id=model_id, type=model.type, registered_name=registered_name, options=options, model_path=local_model_path.absolute()
        )
        if model.type == "txt2img":
            self.endpoint_registry.register_image_generations(
                registered_name, SimpleEndpoint(on_request=_stable_diffusion_handler(info.container_host, info.port))
            )

    async def _uninstall_model(self, model_id: str, options: UninstallModelIn) -> None:
        info = self._check_installed()
        if model_id not in info.models:
            return
        model = info.models[model_id]
        del info.models[model_id]
        if model.type == "txt2img":
            self.endpoint_registry.unregister_image_generations(model.registered_name)

        if options.purge:
            model.model_path.unlink()

    def _get_working_stable_diffusion_dir(self) -> Path:
        path = self._get_working_dir() / "models/Stable-diffusion"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _get_working_lora_dir(self) -> Path:
        path = self._get_working_dir() / "models/Lora"
        path.mkdir(parents=True, exist_ok=True)
        return path


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
    created: int = Field(default_factory=lambda: int(datetime.now(UTC).timestamp()))


class OverrideSettings(BaseModel):
    sd_model_checkpoint: str = "your_model_name.safetensors"


class StableDiffusionInputSettings(BaseModel):
    prompt: str = ""  # Consider generate from prompt
    negative_prompt: str = ""  # Consider generate from prompt
    sampler_name: str = "DPM++ 2M"
    scheduler: str = "Kerras"
    steps: int = Field(default=20, ge=1, le=50)
    n_iter: int = Field(default=1, ge=1, le=4)
    cfg_scale: float = Field(default=7, ge=1, le=30)
    width: int = Field(default=512, ge=64, le=2048)
    height: int = Field(default=512, ge=64, le=2048)
    override_settings: OverrideSettings = OverrideSettings()
    override_settings_restore_afterwards: bool = True


class QualityLevel(Enum):
    high = 35
    medium = 20
    auto = 20  # noqa: PIE796
    low = 15


class StableDiffusionOptions(TypedDict):
    prompt: NotRequired[str]
    n_iter: NotRequired[int]
    width: NotRequired[int]
    height: NotRequired[int]
    steps: NotRequired[int]


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


def _stable_diffusion_handler(container_host: str, port: int) -> EndpointCallback[ImagesRequest]:
    async def handler(body: ImagesRequest, _req: Request) -> ImagesResponse:
        url = f"http://{container_host}:{port}/sdapi/v1/txt2img/"

        settings_raw, remaining_text = split_text_to_json_and_prompt(body.prompt)
        try:
            settings_edited = _add_body_config(settings_raw, body, remaining_text)
        except ValueError as err:
            raise HTTPException(422, str(err)) from None

        try:
            settings = StableDiffusionInputSettings(**settings_edited)
        except ValidationError:
            raise HTTPException(422, "Incorrect settings.") from None

        async with ClientSession(url) as client:
            response = await client.post(url, json=settings.model_dump(), timeout=120)  # type: ignore

            if response.status != 200:
                raise HTTPException(
                    500, (f"Something went wrong inside Stable Diffusion Web UI.Error {response.status}: {response.content}")
                )

            # try:
            data_raw = await response.json()
            # except Exception:
            #     raise HTTPException(500, "There is no images in Stable Diffusion response.")

        try:
            imgs = [_get_in_format(img, body.output_format or "png", body.output_compression or 95) for img in data_raw["images"]]
        except ValueError:
            raise HTTPException(422, f"Not supported image format: {body.output_format}") from None
        except Exception as exc:
            raise HTTPException(500, f"Somethign went wrong: {exc}") from None

        return ImagesResponse(data=[B64Data(b64_json=img, revised_prompt=settings.prompt) for img in imgs])

    return handler
