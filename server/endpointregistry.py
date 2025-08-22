"""Endpoint registry holds callbacks for given endpoints and models."""

from collections.abc import AsyncGenerator, Callable
from typing import Any, NamedTuple

from aiohttp.client import ClientSession
from fastapi import Request
from starlette.responses import StreamingResponse

from server.utils.exceptions import AppError


class SimpleEndpoint(NamedTuple):
    on_request: Callable[[dict, Request], Any]


class ProxyOptions:
    def __init__(
        self,
        url: str,
        rewrite_model_to: str | None = None,
        remove_model: bool = False,
    ):
        self.url = url
        self.rewrite_model_to = rewrite_model_to
        self.remove_model = remove_model


class EndpointRegistry:
    def __init__(self):
        self.chat_completion_endpoints: dict[str, SimpleEndpoint] = {}
        self.audio_speech_endpoints: dict[str, SimpleEndpoint] = {}
        self.custom_endpoints: dict[str, SimpleEndpoint] = {}
        self.images_generations_endpoints: dict[str, SimpleEndpoint] = {}

    def register_chat_completion(self, model: str, endpoint: SimpleEndpoint) -> None:
        """Register chat completion endpoint for given model."""
        if model in self.chat_completion_endpoints:
            raise AppError("There is already registered endpoint for given model", model)
        self.chat_completion_endpoints[model] = endpoint

    def register_chat_completion_as_proxy(self, model: str, options: ProxyOptions) -> None:
        """Register chat completion for given model as a proxy."""

        def on_request(body: dict, req: Request) -> StreamingResponse:
            return self._proxy(body, options, req)

        self.register_chat_completion(model, SimpleEndpoint(on_request=on_request))

    def unregister_chat_completion(self, model: str) -> None:
        """Unregister chat completion for given model."""
        if model in self.chat_completion_endpoints:
            del self.chat_completion_endpoints[model]

    def register_audio_speech(self, model: str, endpoint: SimpleEndpoint) -> None:
        """Register audio speech endpoint for given model."""
        if model in self.audio_speech_endpoints:
            raise AppError("There is already registered endpoint for given model", model)
        self.audio_speech_endpoints[model] = endpoint

    def register_audio_speech_as_proxy(self, model: str, options: ProxyOptions) -> None:
        """Register audio speech for given model as a proxy."""

        def on_request(body: dict, req: Request) -> StreamingResponse:
            return self._proxy(body, options, req)

        self.register_audio_speech(model, SimpleEndpoint(on_request=on_request))

    def unregister_audio_speech(self, model: str) -> None:
        """Unregister audio speech for given model."""
        if model in self.audio_speech_endpoints:
            del self.audio_speech_endpoints[model]

    def register_image_generations(self, model: str, endpoint: SimpleEndpoint) -> None:
        """Register image generations endpoint for given model."""
        if model in self.images_generations_endpoints:
            raise AppError("There is already registered endpoint for given model", model)
        self.images_generations_endpoints[model] = endpoint

    def register_image_generations_as_proxy(self, model: str, options: ProxyOptions) -> None:
        """Register image generations for given model as a proxy."""

        def on_request(body: dict, req: Request) -> StreamingResponse:
            return self._proxy(body, options, req)

        self.register_image_generations(model, SimpleEndpoint(on_request=on_request))

    def unregister_image_generations(self, model: str) -> None:
        """Unregister image generations for given model."""
        if model in self.images_generations_endpoints:
            del self.images_generations_endpoints[model]

    def register_custom_endpoint(self, url: str, endpoint: SimpleEndpoint) -> None:
        """Register custom endpoint."""
        if url in self.custom_endpoints:
            raise AppError("There is already registered endpoint for given url", url)
        self.custom_endpoints[url] = endpoint

    def register_custom_endpoint_as_proxy(self, url: str, options: ProxyOptions) -> None:
        """Register custom endpoint as a proxy."""

        def on_request(body: dict, req: Request) -> StreamingResponse:
            return self._proxy(body, options, req)

        self.register_custom_endpoint(url, SimpleEndpoint(on_request=on_request))

    def unregister_custom_endpoint(self, model: str) -> None:
        """Unregister custom endpoint."""
        if model in self.custom_endpoints:
            del self.custom_endpoints[model]

    def has_chat_completion_model(self, model: str) -> bool:
        """Check whether the chat completion model is registered."""
        return model in self.chat_completion_endpoints

    def has_audio_speech_model(self, model: str) -> bool:
        """Check whether the audio speech model is registered."""
        return model in self.audio_speech_endpoints

    def has_image_generations_model(self, model: str) -> bool:
        """Check whether the image generations model is registered."""
        return model in self.images_generations_endpoints

    def has_custom_endpoint(self, url: str) -> bool:
        """Check whether the custom endpoint is registered."""
        return url in self.custom_endpoints

    def execute_chat_completion(self, data: dict, req: Request) -> Any:  # noqa: ANN401
        """Process chat completion request."""
        endpoint = self.chat_completion_endpoints[data["model"]]
        if not endpoint:
            raise AppError("Given model is not supported", data["model"])
        return endpoint.on_request(data, req)

    def execute_images_generations(self, data: dict, req: Request) -> Any:  # noqa: ANN401
        """Process images generations request."""
        endpoint = self.images_generations_endpoints[data["model"]]
        if not endpoint:
            raise AppError("Given model is not supported", data["model"])
        return endpoint.on_request(data, req)

    async def execute_audio_speech(self, data: dict, req: Request) -> Any:  # noqa: ANN401
        """Process audio speech request."""
        endpoint = self.audio_speech_endpoints.get(data["model"])
        if not endpoint:
            raise AppError("Given model is not supported", data["model"])
        return await endpoint.on_request(data, req)

    def execute_custom_endpoints(self, url: str, body: dict, req: Request) -> Any:  # noqa: ANN401
        """Process custom endpoint request."""
        endpoint = self.custom_endpoints[url]
        if not endpoint:
            raise AppError("Given url is not supported", url)
        return endpoint.on_request(body, req)

    def _proxy(self, body: dict, options: ProxyOptions, _req: Request) -> StreamingResponse:
        if options.remove_model:
            del body["model"]
        if options.rewrite_model_to:
            body["model"] = options.rewrite_model_to

        return StreamingResponse(
            proxy_post_request(options.url, body),
            media_type="application/json",
        )


async def proxy_post_request(url: str, body: dict[str, Any], headers: dict[str, str] | None = None) -> AsyncGenerator[bytes]:
    """Make request to given url and stream it."""
    async with ClientSession() as session, session.post(url, json=body, headers=headers or {}) as resp:
        async for line in resp.content.iter_any():
            if line:
                yield line
