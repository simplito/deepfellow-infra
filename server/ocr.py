from typing import Callable, Optional, List, Dict, Mapping, Any
from platform import system

from .utils import Utils
from .applicationcontext import ApplicationContext
from .docker import docker, DockerOptions, ChatCompletionsOptions

class OcrOptions:
    def __init__(
        self,
        name: str,
        language: str, 
        additional_bootstrap_args: Optional[List[str]] = None,
    ):
        self.name = name
        self.image = "gitlab2.simplito.com:5050/df/df-ocr:1.0.0"
        self.language = language
        self.additional_bootstrap_args = additional_bootstrap_args or []
        self.use_gpu = True
        self.service_name = Utils.sanitize_service_name(name)

def ocr(options: OcrOptions) -> Callable[[ApplicationContext, List[str]], Any]:
    async def handler(ctx: ApplicationContext, args: List[str]) -> Dict[str, Any]:
        os = system()
        if os == "Darwin":
            raise NotImplementedError()
            
        return await docker(DockerOptions(
            name=options.name,
            image=options.image,
            image_port=8000,
            additional_bootstrap_args = options.additional_bootstrap_args,
            restart='unless-stopped',
            volumes=[f"{ctx.get_model_dir()}/ocr:/root/.EasyOCR/model"],
            use_gpu=options.use_gpu,
            api_endpoint="/v1/ocr"
        ))(ctx, args)

    return handler