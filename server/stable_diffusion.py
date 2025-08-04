import base64
from datetime import UTC, datetime
from enum import Enum
import io
import json
import re
from PIL import Image
from typing import Callable, Literal, List, Dict, Mapping, Any
from platform import system
from aiohttp import ClientSession
from attr import dataclass
from fastapi import HTTPException
from pydantic import BaseModel, Field, ValidationError

from .applicationcontext import ApplicationContext
from .docker import docker, DockerOptions, FunctionHandler, ImageGenerationsOptions


class ImagesRequest(BaseModel):
    # Supported
    prompt: str = Field(..., examples=['A painting of a cat <sd>{"negative_prompt": "low quality"}</sd>'])
    model: str = ""
    size: str = Field("auto", examples=["512x512", "auto"])
    quality: Literal["low", "medium", "high", "auto"] = "auto"
    output_format: Literal["png", "webp", "jpeg"] = "png"
    output_compression: int = Field(95, ge=0, le=100)
    n: int = Field(1, ge=1, le=10)
    # Not supported yet. Possible to do.
    background: Literal["auto", "transparent", "opaque"] = "auto" # LayerDiffusion in forge version.
    style: str = "vivid"
    moderation: str = "auto"
    # Not supported
    response_format: Literal["url", "b64_json"] = "b64_json"
    partial_images: int = 0
    stream: bool = False
    user: str = ""


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
    prompt: str = "" # Consider generate from prompt
    negative_prompt: str = "" # Consider generate from prompt
    sampler_name: str = "DPM++ 2M"
    scheduler: str = "Kerras"
    steps: int = Field(default=20, ge=1, le=50)
    n_iter: int = Field(default=1, ge=1, le=4)
    cfg_scale: float = Field(default=7, ge=1, le=30)
    width: int = Field(default=512, ge=64, le=2048)
    height: int = Field(default=512, ge=64, le=2048)
    override_settings: OverrideSettings = OverrideSettings()
    override_settings_restore_afterwards: bool = True


@dataclass
class StableDiffusionOptions:
    name: str
    model_name: str
    env_vars: Mapping = {}
    image: str = "ghcr.io/ai-dock/stable-diffusion-webui:latest"
    image_cpu: str = "ghcr.io/ai-dock/stable-diffusion-webui:v2-cpu-22.04-v1.10.1"
    hf_token: str | None = None
    use_gpu: bool = True
    image_endpoint: str = "/v1/images/generation"
    additional_bootstrap_args: list[str] = []
    image_generation: ImageGenerationsOptions | None = None


class QualityLevel(Enum):
    high = 35
    medium = 20
    auto = 20
    low = 15


def split_text_to_json_and_prompt(text: str) -> tuple[dict, str]:
    """Split text into prompt and JSON data for settings."""
    # With this regex we get two matches
    # (one match with whole <sd>example text</sd>)
    # and one with what inside sd html element like example text
    json_candidates = re.findall(r'(<sd>(.*?)</sd>)', text, re.DOTALL)
    if not json_candidates:
        # No json found in text
        return {}, text

    extracted_text, json_candidate = json_candidates[0]

    try:
        # Convert text to dict
        json_output: dict = json.loads(json_candidate)
    except json.JSONDecodeError:
        # Settings json not correct, return error.
        raise HTTPException(
            422,
            "Something went wrong with stable diffusion settings."
            "Please place correct json in prompt in html <sd></sd> tags."
        )

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
        raise ValueError("Sizes are not ints. Format should be like '517x768'")

    return width, height


def convert_b64png_to_b64jpg(b64_png: str, quality: int=90) -> str:
    """
    Converts a Base64 encoded PNG image string to a Base64 encoded JPG image string.

    Args:
        b64_png: The Base64 encoded PNG string. Should NOT include the "data:image/png;base64," prefix.
        quality: The quality of the JPG image, between 0 and 100, with 100 being the highest quality.

    Returns:
        str: The Base64 encoded JPG string, or None if an error occurs.
    """
    png_bytes = base64.b64decode(b64_png)

    image_stream = io.BytesIO(png_bytes)
    img = Image.open(image_stream)

    if img.mode == 'RGBA':
        background = Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img = background
    else: # img.mode == 'RGB'
        img = img.convert('RGB')

    output_buffer = io.BytesIO()
    img.save(output_buffer, format="JPEG", quality=quality)

    base64_jpg_bytes = base64.b64encode(output_buffer.getvalue())
    base64_jpg_string = base64_jpg_bytes.decode('utf-8')

    return base64_jpg_string


def convert_b64png_to_b64webp(b64_png: str, quality: int=90) -> str:
    """
    Converts a Base64 encoded PNG image string to a Base64 encoded WebP image string.

    Args:
        b64_png_string: The Base64 encoded PNG string. Should NOT include the "data:image/png;base64," prefix.
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
    base64_webp_string = base64_webp_bytes.decode('utf-8')

    return base64_webp_string



def get_in_format(img: str, format: str = "png", quality: int = 95):
    match format:
        case "jpg":
            return convert_b64png_to_b64jpg(img, quality)
        case "webp":
            return convert_b64png_to_b64webp(img, quality)
        case "png":
            return img
        case _:
            raise ValueError("Not Supported format")


def add_body_config(settings_original: dict, body: ImagesRequest, remaining_text: str) -> dict:
    settings = settings_original.copy()

    if body.response_format != "b64_json":
        raise ValueError(f"Response format {body.response_format} not supported")

    if body.partial_images != 0:
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


def stable_diffusion(options: StableDiffusionOptions) -> Callable[[ApplicationContext, List[str]], Any]:
    async def stable_diffusion_handler(port: int, body: ImagesRequest, req=None) -> ImagesResponse:
        url  = f"http://localhost:{port}/sdapi/v1/txt2img/"

        settings_raw, remaining_text = split_text_to_json_and_prompt(body.prompt)
        try:
            settings_edited = add_body_config(settings_raw, body, remaining_text)
        except ValueError as err:
            raise HTTPException(422, str(err))

        try:
            settings = StableDiffusionInputSettings(**settings_edited)
        except ValidationError:
            raise HTTPException(422, "Incorrect settings.")

        async with ClientSession(url) as client:
            response = await client.post(url, json=settings.model_dump(), timeout=120) # type: ignore

            if response.status != 200:
                raise HTTPException(
                    500,
                    (
                        "Something went wrong inside Stable Diffusion Web UI."
                        f"Error {response.status}: {response.content}"
                    )
                )

            # try:
            data_raw = await response.json()
            # except Exception:
            #     raise HTTPException(500, "There is no images in Stable Diffusion response.")

        try:
            imgs = [get_in_format(img, body.output_format, body.output_compression) for img in data_raw["images"]]
        except ValueError:
            raise HTTPException(422, f"Not supported image format: {body.output_format}")
        except Exception as exc:
            raise HTTPException(500, f"Somethign went wrong: {exc}")

        return ImagesResponse(data=[B64Data(b64_json=img, revised_prompt=settings.prompt) for img in imgs])


    async def handler(ctx: ApplicationContext, args: List[str]) -> Dict[str, Any]:
        if system() == "Darwin":
            raise NotImplementedError("Stable Diffusion is not supported on macOS")

        images_dir = ctx.get_images_dir()

        volumes = [
            f"{images_dir}/models/Stable-diffusion:/opt/stable-diffusion-webui/models/Stable-diffusion",
            f"{images_dir}/models/Lora:/opt/stable-diffusion-webui/models/Lora",
            # f"{images_dir}/outputs:/opt/stable-diffusion-webui/outputs",
            # f"{images_dir}/embeddings:/opt/stable-diffusion-webui/embeddings",
            # f"{images_dir}/extensions:/opt/stable-diffusion-webui/extensions"
        ]

        image = options.image if options.use_gpu else options.image_cpu
        env_vars = {
            "COMMANDLINE_ARGS": "--listen --api --xformers --no-half-vae",
        }

        return await docker(DockerOptions(
            name=f"stable-diffusion-{options.name}",
            image=image,
            env_vars=env_vars,
            image_port=17860,
            use_gpu=options.use_gpu,
            volumes=volumes,
            restart="unless-stopped",
            api_endpoint=options.image_endpoint,
            additional_bootstrap_args=options.additional_bootstrap_args,
            image_generations=ImageGenerationsOptions(
                model = options.model_name,
                handler = FunctionHandler(stable_diffusion_handler)
            )
        ))(ctx, args)

    return handler
