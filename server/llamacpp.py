"""llamacpp backend."""

from collections.abc import Callable, Mapping
from platform import system
from typing import Any

from .applicationcontext import ApplicationContext
from .docker import ChatCompletionsOptions, DockerOptions, docker
from .utils.core import Utils


class LlamacppOptions:
    def __init__(
        self,
        name: str,
        model_path: str,
        env_vars: Mapping,
        additional_bootstrap_args: list[str] | None = None,
    ):
        self.image_cuda = "ghcr.io/ggml-org/llama.cpp:server-cuda"
        self.image_cpu = "ghcr.io/ggml-org/llama.cpp:server"
        self.image_cpu_arm64 = "deepfellow-llamacpp-arm64:latest"
        self.name = name
        self.model_path = model_path
        self.env_vars = env_vars
        self.additional_bootstrap_args = additional_bootstrap_args or []
        self.use_gpu = True
        self.service_name = Utils.sanitize_service_name(name)


def llamacpp(options: LlamacppOptions) -> Callable[[ApplicationContext, list[str]], Any]:
    """Prepare llamacpp setup."""

    async def handler(ctx: ApplicationContext, args: list[str]) -> dict[str, Any]:
        local_model_path, model_filename = await Utils.ensure_model_downloaded(ctx, options.model_path)
        model_in_container = f"/models/{model_filename}"
        os = system()
        if os == "Darwin":
            options.use_gpu = False
            image = options.image_cpu_arm64
        else:
            image = options.image_cuda if options.use_gpu else options.image_cpu

        return await docker(
            DockerOptions(
                name=options.name,
                image=image,
                command=f"--model {model_in_container} --host 0.0.0.0 --port 8080",
                image_port=8080,
                additional_bootstrap_args=options.additional_bootstrap_args,
                env_vars=options.env_vars,
                restart="unless-stopped",
                volumes=[f"{local_model_path.absolute()}:{model_in_container}:ro"],
                use_gpu=options.use_gpu,
                chat_completion=ChatCompletionsOptions(model=options.name, remove_model=True),
            )
        )(ctx, args)

    return handler
