from typing import Callable, Optional, List, Dict, Any
from server.utils import Utils
from server.applicationcontext import ApplicationContext
from .endpointregistry import ProxyOptions

class OllamaOptions:
    def __init__(
        self,
        model: str,
        chat_completions: Optional[bool] = True,
        additional_bootstrap_args: Optional[List[str]] = None,
    ):
        self.model = model
        self.chat_completions = chat_completions
        self.additional_bootstrap_args = additional_bootstrap_args or []

def ollama(options: OllamaOptions) -> Callable[[ApplicationContext, List[str]], Any]:
    async def handler(ctx: ApplicationContext, args: List[str]) -> Dict[str, Any]:
        service, command = args
        bootstrap_args = [service, "install"] + (options.additional_bootstrap_args or [])

        if command == "install":
            await Utils.run_command_for_success(f"{ctx.config.values.ollama.cmd} pull {options.model}")
            if options.chat_completions is not False:
                url = Utils.join_url(ctx.config.values.ollama.url, "/v1/chat/completions")
                ctx.endpoint_registry.register_chat_completion_as_proxy(options.model, ProxyOptions(
                    url=url,  
                ))
            await ctx.add_command_to_bootstrap(bootstrap_args)
            return {"success": True}

        elif command == "uninstall":
            ctx.endpoint_registry.unregister_chat_completion(options.model)
            await ctx.remove_command_from_bootstrap(bootstrap_args)
            return {"success": True}

        elif command == "status":
            res = await Utils.run_command(f"{ctx.config.values.ollama.cmd} show {options.model}")
            if res.exit_code == 1 and res.stderr.strip() == f"Error: model '{options.model}' not found":
                return {"success": True, "info": "not found"}
            return {"success": True, "info": res.stdout}

        return {"success": False, "error": f"Unknown command {command}"}

    return handler