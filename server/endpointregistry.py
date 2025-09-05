"""Endpoint registry holds callbacks for given endpoints and models."""

from collections.abc import AsyncGenerator, Callable
from typing import Any, NamedTuple

from aiohttp import FormData
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
        form: bool = False,
    ):
        self.url = url
        self.rewrite_model_to = rewrite_model_to
        self.remove_model = remove_model
        self.form = form


class EndpointRegistry:
    def __init__(self):
        self.chat_completion_endpoints: dict[str, SimpleEndpoint] = {}
        self.embeddings_endpoints: dict[str, SimpleEndpoint] = {}
        self.audio_speech_endpoints: dict[str, SimpleEndpoint] = {}
        self.audio_transcriptions_endpoints: dict[str, SimpleEndpoint] = {}
        self.custom_endpoints: dict[str, SimpleEndpoint] = {}
        self.images_generations_endpoints: dict[str, SimpleEndpoint] = {}

    def get_chat_completions_models(self) -> list:
        """Get chat completions models."""
        models = []
        for model_id in self.chat_completion_endpoints:
            models.append({"id": model_id, "object": "model", "created": 0, "owned_by": "unknown"})
        return models

    def register_chat_completion(self, model: str, endpoint: SimpleEndpoint) -> None:
        """Register chat completion endpoint for given model."""
        if model in self.chat_completion_endpoints:
            raise AppError("There is already registered endpoint for given model", model)
        self.chat_completion_endpoints[model] = endpoint

    def register_chat_completion_as_proxy(self, model: str, options: ProxyOptions) -> None:
        """Register chat completion for given model as a proxy."""

        async def on_request(body: dict, req: Request) -> StreamingResponse:
            return await self._proxy(body, options, req)

        self.register_chat_completion(model, SimpleEndpoint(on_request=on_request))

    def unregister_chat_completion(self, model: str) -> None:
        """Unregister chat completion for given model."""
        if model in self.chat_completion_endpoints:
            del self.chat_completion_endpoints[model]

    def register_embeddings(self, model: str, endpoint: SimpleEndpoint) -> None:
        """Register embeddings endpoint for given model."""
        if model in self.embeddings_endpoints:
            raise AppError("There is already registered endpoint for given model", model)
        self.embeddings_endpoints[model] = endpoint

    def register_embeddings_as_proxy(self, model: str, options: ProxyOptions) -> None:
        """Register embeddings for given model as a proxy."""

        async def on_request(body: dict, req: Request) -> StreamingResponse:
            return await self._proxy(body, options, req)

        self.register_embeddings(model, SimpleEndpoint(on_request=on_request))

    def unregister_embeddings(self, model: str) -> None:
        """Unregister embeddings for given model."""
        if model in self.embeddings_endpoints:
            del self.embeddings_endpoints[model]

    def register_audio_speech(self, model: str | list[str], endpoint: SimpleEndpoint) -> None:
        """Register audio speech endpoint for given model."""
        models = model if isinstance(model, list) else [model]
        for model in models:
            if model in self.audio_speech_endpoints:
                raise AppError("There is already registered endpoint for given model", model)

        for model in models:
            self.audio_speech_endpoints[model] = endpoint

    def register_audio_speech_as_proxy(self, model: str | list[str], options: ProxyOptions) -> None:
        """Register audio speech for given model as a proxy."""

        async def on_request(body: dict, req: Request) -> StreamingResponse:
            return await self._proxy(body, options, req)

        self.register_audio_speech(model, SimpleEndpoint(on_request=on_request))

    def unregister_audio_speech(self, model: str | list[str]) -> None:
        """Unregister audio speech for given model."""
        models = model if isinstance(model, list) else [model]
        for model in models:
            if model in self.audio_speech_endpoints:
                del self.audio_speech_endpoints[model]

    def register_audio_transcriptions(self, model: str | list[str], endpoint: SimpleEndpoint) -> None:
        """Register audio transcriptions endpoint for given model."""
        models = model if isinstance(model, list) else [model]
        for model in models:
            if model in self.audio_transcriptions_endpoints:
                raise AppError("There is already registered endpoint for given model", model)

        for model in models:
            self.audio_transcriptions_endpoints[model] = endpoint

    def register_audio_transcriptions_as_proxy(self, model: str | list[str], options: ProxyOptions) -> None:
        """Register audio transcriptions for given model as a proxy."""

        async def on_request(body: dict, req: Request) -> StreamingResponse:
            form_data = FormData()
            for key in body:
                if key == "file":
                    form_data.add_field("file", await body["file"].read(), filename="audio", content_type="application/octet-stream")
                else:
                    form_data.add_field(key, body[key])
            return await self._proxy(form_data, options, req)

        self.register_audio_transcriptions(model, SimpleEndpoint(on_request=on_request))

    def unregister_audio_transcriptions(self, model: str | list[str]) -> None:
        """Unregister audio transcriptions for given model."""
        models = model if isinstance(model, list) else [model]
        for model in models:
            if model in self.audio_transcriptions_endpoints:
                del self.audio_transcriptions_endpoints[model]

    def register_image_generations(self, model: str, endpoint: SimpleEndpoint) -> None:
        """Register image generations endpoint for given model."""
        if model in self.images_generations_endpoints:
            raise AppError("There is already registered endpoint for given model", model)
        self.images_generations_endpoints[model] = endpoint

    def register_image_generations_as_proxy(self, model: str, options: ProxyOptions) -> None:
        """Register image generations for given model as a proxy."""

        async def on_request(body: dict, req: Request) -> StreamingResponse:
            return await self._proxy(body, options, req)

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

        async def on_request(body: dict, req: Request) -> StreamingResponse:
            return await self._proxy(body, options, req)

        self.register_custom_endpoint(url, SimpleEndpoint(on_request=on_request))

    def unregister_custom_endpoint(self, model: str) -> None:
        """Unregister custom endpoint."""
        if model in self.custom_endpoints:
            del self.custom_endpoints[model]

    def has_chat_completion_model(self, model: str) -> bool:
        """Check whether the chat completion model is registered."""
        return model in self.chat_completion_endpoints

    def has_embeddings_model(self, model: str) -> bool:
        """Check whether the embeddings model is registered."""
        return model in self.embeddings_endpoints

    def has_audio_speech_model(self, model: str) -> bool:
        """Check whether the audio speech model is registered."""
        return model in self.audio_speech_endpoints

    def has_audio_transcriptions_model(self, model: str) -> bool:
        """Check whether the audio transcriptions model is registered."""
        return model in self.audio_transcriptions_endpoints

    def has_image_generations_model(self, model: str) -> bool:
        """Check whether the image generations model is registered."""
        return model in self.images_generations_endpoints

    def has_custom_endpoint(self, url: str) -> bool:
        """Check whether the custom endpoint is registered."""
        return url in self.custom_endpoints

    async def execute_chat_completion(self, data: dict, req: Request) -> Any:  # noqa: ANN401
        """Process chat completion request."""
        endpoint = self.chat_completion_endpoints[data["model"]]
        if not endpoint:
            raise AppError("Given model is not supported", data["model"])
        return await endpoint.on_request(data, req)

    async def execute_embeddings(self, data: dict, req: Request) -> Any:  # noqa: ANN401
        """Process embeddings request."""
        endpoint = self.embeddings_endpoints[data["model"]]
        if not endpoint:
            raise AppError("Given model is not supported", data["model"])
        return await endpoint.on_request(data, req)

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

    async def execute_audio_transcriptions(self, data: dict, req: Request) -> Any:  # noqa: ANN401
        """Process audio transcriptions request."""
        endpoint = self.audio_transcriptions_endpoints.get(data["model"])
        if not endpoint:
            raise AppError("Given model is not supported", data["model"])
        return await endpoint.on_request(data, req)

    def execute_custom_endpoints(self, url: str, body: dict, req: Request) -> Any:  # noqa: ANN401
        """Process custom endpoint request."""
        endpoint = self.custom_endpoints[url]
        if not endpoint:
            raise AppError("Given url is not supported", url)
        return endpoint.on_request(body, req)

    async def _proxy(self, body: Any, options: ProxyOptions, _req: Request) -> StreamingResponse:  # noqa: ANN401
        if options.remove_model:
            del body["model"]
        if options.rewrite_model_to:
            body["model"] = options.rewrite_model_to

        (status_code, media_type, generator) = await proxy_post_request(options.url, body, form=options.form)
        return StreamingResponse(generator, media_type=media_type, status_code=status_code)


async def proxy_post_request(
    url: str,
    body: Any,  # noqa: ANN401
    headers: dict[str, str] | None = None,
    form: bool = False,
) -> tuple[int, str, AsyncGenerator[bytes]]:
    """Make request to given url and stream it."""
    session = ClientSession()
    try:
        if form:
            resp = await session.post(url, data=body, headers=headers or {})
        else:
            resp = await session.post(url, json=body, headers=headers or {})
        status_code = resp.status
        content_type = resp.headers.get("content-type", "application/octet-stream")

        async def generator() -> AsyncGenerator[bytes]:
            try:
                async for chunk in resp.content.iter_any():
                    if chunk:
                        yield chunk
            finally:
                await resp.release()
                await session.close()

        return status_code, content_type, generator()

    except Exception:
        await session.close()
        raise
