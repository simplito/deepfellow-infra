from typing import Callable, Optional, List, Dict, Mapping, Any
from platform import system

from .utils import Utils
from .applicationcontext import ApplicationContext
from .docker import docker, DockerOptions, ChatCompletionsOptions

class LlamacppOptions:
    def __init__(
        self,
        name: str,
        model_path: str, 
        env_vars: Mapping,
        command: str,
        hf_token: str,
        additional_bootstrap_args: Optional[List[str]] = None,
    ):
        self.image_cuda = "ghcr.io/ggml-org/llama.cpp:server-cuda"
        self.image_cpu = "ghcr.io/ggml-org/llama.cpp:server"
        self.image_cpu_arm64 = "deepfellow-llamacpp-arm64:latest"
        self.name = name
        self.model_path = model_path
        self.env_vars = env_vars
        self.command = command
        self.additional_bootstrap_args = additional_bootstrap_args or []
        self.use_gpu = True
        self.service_name = Utils.sanitize_service_name(name)
        self.hf_token = hf_token


def llamacpp(options: LlamacppOptions) -> Callable[[ApplicationContext, List[str]], Any]:
    async def handler(ctx: ApplicationContext, args: List[str]) -> Dict[str, Any]:
        local_model_path, model_filename = await Utils.ensure_model_downloaded(ctx, options.model_path)
        model_in_container = f"/models/{model_filename}"
        os = system()
        if os == "Darwin":
            options.use_gpu = False
            image = options.image_cpu_arm64
        else:
            image = options.image_cuda if options.use_gpu else options.image_cpu
            
        return await docker(DockerOptions(
            name=options.name,
            image=image,
            command=f'--model {model_in_container} --host 0.0.0.0 --port 8080',
            image_port=8080,
            additional_bootstrap_args = options.additional_bootstrap_args,
            env_vars=options.env_vars,
            restart='unless-stopped',
            volumes=[f"{local_model_path.absolute()}:{model_in_container}:ro"],
            use_gpu=options.use_gpu,
            chat_completion=ChatCompletionsOptions(
                model = options.name,
                remove_model = True
            )
        ))(ctx, args)
    return handler