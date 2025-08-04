from aiohttp import ClientSession
from collections.abc import AsyncGenerator
from typing import Callable, Optional, List, Dict, Any
from platform import system
from starlette.responses import StreamingResponse

from .utils import Utils
from .applicationcontext import ApplicationContext
from .docker import docker, DockerOptions, AudioSpeechOptions, FunctionHandler
from .ffmpeg import ffmpeg_audio_convert_async_gen

class CoquiOptions:
    def __init__(
        self,
        name: str,
        model_name: str,
        model_path: str = None,
        default_speaker: Optional[str] = None,
        cuda: Optional[bool] = True,
        language: Optional[str] = None,
        response_format: Optional[str] = None,
        additional_bootstrap_args: Optional[List[str]] = None,
    ):
        self.name = name
        self.model_name = model_name
        self.audio_endpoint = "/v1/audio/speech"
        self.image_gpu = "ghcr.io/coqui-ai/tts"
        self.image_cpu = "ghcr.io/coqui-ai/tts-cpu"
        self.model_path = model_path
        self.cuda = cuda
        self.language = language
        self.response_format = response_format or "mp3"
        self.additional_bootstrap_args = additional_bootstrap_args or []
        self.default_speaker = default_speaker

def coqui(options: CoquiOptions) -> Callable[[ApplicationContext, List[str]], Any]:
    async def _proxy_post_request(url: str) -> AsyncGenerator[bytes]:
        async with ClientSession() as session, session.get(url) as resp:
            async for chunk in resp.content.iter_any():
                if chunk:
                    yield chunk

    async def coqui_handler(port: int, body, req=None):
        text = body.get("input", "")
        voice = body.get("voice", None) or options.default_speaker
        response_format = body.get("response_format", options.response_format)
        if response_format is None:
            response_format = "wav"

        encoded_text = Utils.str_encode(text)
        coqui_url = f"http://localhost:{port}/api/tts?text={encoded_text}"

        if voice is not None: 
            voice_encoded = Utils.str_encode(voice)
            coqui_url += f"&speaker_id={voice_encoded}"
        return StreamingResponse(
            ffmpeg_audio_convert_async_gen(_proxy_post_request(coqui_url), "wav", response_format),
            media_type=f"audio/{response_format}",
        )

    def _build_coqui_command(options: CoquiOptions) -> str:
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

    async def handler(ctx: ApplicationContext, args: List[str]) -> Dict[str, Any]:
        if system() ==  "Darwin":
            raise NotImplementedError()
        
        tts_dir = ctx.get_tts_dir()
        
        volumes = [
            f"{tts_dir}:/root/tts-output",
            f"{ctx.get_model_dir()}:/root/.local/share/tts"
        ]

        command = _build_coqui_command(options)
        
        image = options.image_gpu if options.cuda else options.image_cpu

        return await docker(DockerOptions(
            name=f"coqui-tts-{options.name}",
            image=image,
            command=command,
            image_port=5002,
            use_gpu=options.cuda,
            volumes=volumes,
            entrypoint="/bin/bash",
            restart="unless-stopped",
            api_endpoint=options.audio_endpoint,
            audio_generation = AudioSpeechOptions(
                model=options.model_name,
                handler=FunctionHandler(
                    func = coqui_handler
                )
            ),
            additional_bootstrap_args=options.additional_bootstrap_args
        ))(ctx, args)

    return handler


