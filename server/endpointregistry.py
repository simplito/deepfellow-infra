from typing import Callable, NamedTuple, Any, Mapping, Optional
from starlette.responses import StreamingResponse
from aiohttp.client import ClientSession
from collections.abc import AsyncGenerator

class SimpleEndpoint(NamedTuple):
    on_request: Callable

class ProxyOptions:
    def __init__(
            self,
            url: str,
            rewrite_model_to: Optional[str] = None,
            remove_model: bool = False,
        ):
            self.url = url
            self.rewrite_model_to = rewrite_model_to
            self.remove_model = remove_model


class EndpointRegistry:
    def __init__(self):
        self.chat_completion_endpoints: Mapping = {}
        self.audio_speech_endpoints: Mapping = {}
        self.custom_endpoints: Mapping = {}
        self.images_generations_endpoints: Mapping = {}

    def register_chat_completion(self, model: str, endpoint):
        if model in self.chat_completion_endpoints:
            raise Exception(f"There is already registered endpoint for given model {model}")
        self.chat_completion_endpoints[model] = endpoint

    def register_chat_completion_as_proxy(self, model: str, options: ProxyOptions):
        def on_request(body, req):
            return self.proxy(body, options, req)
        self.register_chat_completion(model, SimpleEndpoint(on_request=on_request))

    def unregister_chat_completion(self, model: str):
        del self.chat_completion_endpoints[model]

    def register_audio_speech(self, model: str, endpoint):
        if model in self.audio_speech_endpoints:
            raise Exception(f"There is already registered endpoint for given model {model}")
        self.audio_speech_endpoints[model] = endpoint

    def register_audio_speech_as_proxy(self, model: str, options: ProxyOptions):
        def on_request(body, req):
            return self.proxy(body, options, req)

        self.register_audio_speech(model, SimpleEndpoint(on_request=on_request))

    def unregister_audio_speech(self, model: str):
        del self.audio_speech_endpoints[model]

    def register_image_generations(self, model: str, endpoint):
        if model in self.images_generations_endpoints:
            raise Exception(f"There is already registered endpoint for given model {model}")
        self.images_generations_endpoints[model] = endpoint

    def register_image_generations_as_proxy(self, model: str, options: ProxyOptions):
        def on_request(body, req):
            return self.proxy(body, options, req)

        self.register_image_generations(model, SimpleEndpoint(on_request=on_request))

    def unregister_image_generations(self, model: str):
        del self.images_generations_endpoints[model]

    def register_custom_endpoint(self, url: str, endpoint):
        if url in self.custom_endpoints:
            raise Exception(f"There is already registered endpoint for given url {url}")
        self.custom_endpoints[url] = endpoint

    def register_custom_endpoint_as_proxy(self, url: str, options: ProxyOptions):
        def on_request(body, req):
            return self.proxy(body, options, req)
        self.register_custom_endpoint(url, SimpleEndpoint(on_request=on_request))

    def unregister_custom_endpoint(self, model: str):
        del self.custom_endpoints[model]

    def has_chat_completion_model(self, model: str):
        return model in self.chat_completion_endpoints

    def has_audio_speech_model(self, model: str):
        return model in self.audio_speech_endpoints

    def has_image_generations_model(self, model: str):
        return model in self.images_generations_endpoints

    def has_custom_endpoint(self, url: str):
        return url in self.custom_endpoints

    def execute_chat_completion(self, data, req):
        endpoint = self.chat_completion_endpoints[data["model"]]
        if not endpoint:
            raise Exception(f"Given model ${data["model"]} is not supported")
        return endpoint.on_request(data, req)

    def execute_images_generations(self, data, req):
        endpoint = self.images_generations_endpoints[data.model]
        if not endpoint:
            raise Exception(f"Given model ${data.model} is not supported")
        return endpoint.on_request(data, req)

    async def execute_audio_speech(self, data, req):
        endpoint = self.audio_speech_endpoints.get(data["model"])
        if not endpoint:
            raise Exception(f"Given model {data["model"]} is not supported")
        return await endpoint.on_request(data, req)

    def execute_custom_endpoints(self, url: str, body, req):
        endpoint = self.custom_endpoints[url]
        if not endpoint:
            raise Exception(f"Given url {url} is not supported")
        return endpoint.on_request(body, req)


    def proxy(self, body, options: ProxyOptions, req):
        if options.remove_model:
            del body["model"]
        if options.rewrite_model_to:
            body["model"] = options.rewrite_model_to

        return StreamingResponse(
            proxy_post_request(options.url, body),
            media_type="application/json",
        )

async def proxy_post_request(
    url: str, body: dict[str, Any], headers: dict[str, str] | None = None
) -> AsyncGenerator[bytes]:

    async with ClientSession() as session, session.post(url, json=body, headers=headers or {}) as resp:
        async for line in resp.content.iter_any():
            if line:
                yield line
