"""VLLM backend."""

from collections.abc import Callable, Mapping
from typing import Any

from .applicationcontext import ApplicationContext
from .docker import ChatCompletionsOptions, DockerOptions, docker
from .utils.core import Utils


class VllmOptions:
    def __init__(
        self,
        model_name: str,
        model_hf_id: str,
        hf_token: str | None = None,
        env_vars: Mapping | None = None,
        quantization: str | None = None,  # 'awq', 'gptq', 'squeezellm', 'bitsandbytes'
        dtype: str = "auto",
        shm_size: str = "16gb",
        ulimits: Mapping | None = None,
        max_model_len: int | None = None,
        gpu_memory_utilization: float = 0.9,
        additional_bootstrap_args: list[str] | None = None,
    ):
        self.model_name = model_name
        self.image = "vllm/vllm-openai:v0.8.4"
        self.model_hf_id = model_hf_id
        self.hf_token = hf_token
        self.env_vars = dict(env_vars or {})
        self.env_vars["HF_HOME"] = "/models/cache"
        if hf_token:
            self.env_vars["HUGGING_FACE_HUB_TOKEN"] = hf_token
        self.quantization = quantization
        self.dtype = dtype
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.additional_bootstrap_args = additional_bootstrap_args or []
        self.use_gpu = True
        self.service_name = Utils.sanitize_service_name(model_hf_id)
        self.shm_size = shm_size
        self.ulimits = ulimits or {"memlock": -1, "stack": 67108864}


def vllm(options: VllmOptions) -> Callable[[ApplicationContext, list[str]], Any]:
    """Prepare coqui setup."""

    async def handler(ctx: ApplicationContext, args: list[str]) -> dict[str, Any]:
        vllm_command = [
            "--model",
            options.model_hf_id,
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--dtype",
            options.dtype,
            "--gpu-memory-utilization",
            str(options.gpu_memory_utilization),
        ]

        if options.quantization:
            vllm_command.extend(["--quantization", options.quantization])

        if options.max_model_len:
            vllm_command.extend(["--max-model-len", str(options.max_model_len)])

        return await docker(
            DockerOptions(
                name=options.model_hf_id,
                image=options.image,
                command=" ".join(vllm_command),
                image_port=8000,
                additional_bootstrap_args=options.additional_bootstrap_args,
                env_vars=options.env_vars,
                restart="unless-stopped",
                volumes=[f"{ctx.get_model_dir().absolute()}:/models"],
                use_gpu=options.use_gpu,
                shm_size=options.shm_size,
                ulimits=options.ulimits,
                chat_completion=ChatCompletionsOptions(model=options.model_name, rewrite_model_to=options.model_hf_id),
            )
        )(ctx, args)

    return handler
