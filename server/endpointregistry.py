"""Endpoint registry holds callbacks for given endpoints and models."""

from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import NamedTuple

from aiohttp import FormData
from aiohttp.client import ClientSession
from fastapi import HTTPException, Request
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from server.models.api import (
    ChatCompletionModel,
    ChatCompletionModels,
    ChatCompletionRequest,
    CompletionLegacyRequest,
    CreateSpeechRequest,
    CreateTranscriptionRequest,
    EmbeddingRequest,
    ImagesRequest,
)
from server.models.common import FormFields, JsonSerializable, StarletteResponse
from server.utils.exceptions import AppError

type EndpointCallback[T] = Callable[[T, Request], Awaitable[StarletteResponse]]


class SimpleEndpoint[T](NamedTuple):
    on_request: EndpointCallback[T]


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
        self.chat_completion_endpoints: dict[str, SimpleEndpoint[ChatCompletionRequest]] = {}
        self.completion_endpoints: dict[str, SimpleEndpoint[CompletionLegacyRequest]] = {}
        self.embeddings_endpoints: dict[str, SimpleEndpoint[EmbeddingRequest]] = {}
        self.audio_speech_endpoints: dict[str, SimpleEndpoint[CreateSpeechRequest]] = {}
        self.audio_transcriptions_endpoints: dict[str, SimpleEndpoint[CreateTranscriptionRequest]] = {}
        self.custom_endpoints: dict[str, SimpleEndpoint[None]] = {}
        self.images_generations_endpoints: dict[str, SimpleEndpoint[ImagesRequest]] = {}

    def get_chat_completions_models(self) -> ChatCompletionModels:
        """Get chat completions models."""
        models: list[ChatCompletionModel] = []
        for model_id in self.chat_completion_endpoints:
            models.append(ChatCompletionModel(id=model_id, object="model", created=0, owned_by="unknown"))
        return ChatCompletionModels(data=models)

    def get_chat_completions_model(self, model_id: str) -> ChatCompletionModel:
        """Get chat completions model."""
        if model_id not in self.chat_completion_endpoints:
            raise HTTPException(404, f"Model not found {model_id}")
        return ChatCompletionModel(id=model_id, object="model", created=0, owned_by="unknown")

    def register_chat_completion(self, model: str, endpoint: SimpleEndpoint[ChatCompletionRequest]) -> None:
        """Register chat completion endpoint for given model."""
        if model in self.chat_completion_endpoints:
            raise AppError("There is already registered endpoint for given model", model)
        self.chat_completion_endpoints[model] = endpoint

    def register_chat_completion_as_proxy(self, model: str, options: ProxyOptions) -> None:
        """Register chat completion for given model as a proxy."""

        async def on_request(body: ChatCompletionRequest, request: Request) -> StreamingResponse:
            return await self._proxy(body, options, request)

        self.register_chat_completion(model, SimpleEndpoint(on_request=on_request))

    def unregister_chat_completion(self, model: str) -> None:
        """Unregister chat completion for given model."""
        if model in self.chat_completion_endpoints:
            del self.chat_completion_endpoints[model]

    def register_completion(self, model: str, endpoint: SimpleEndpoint[CompletionLegacyRequest]) -> None:
        """Register completion endpoint for given model."""
        if model in self.completion_endpoints:
            raise AppError("There is already registered endpoint for given model", model)
        self.completion_endpoints[model] = endpoint

    def register_completion_as_proxy(self, model: str, options: ProxyOptions) -> None:
        """Register completion for given model as a proxy."""

        async def on_request(body: CompletionLegacyRequest, request: Request) -> StreamingResponse:
            return await self._proxy(body, options, request)

        self.register_completion(model, SimpleEndpoint(on_request=on_request))

    def unregister_completion(self, model: str) -> None:
        """Unregister completion for given model."""
        if model in self.completion_endpoints:
            del self.completion_endpoints[model]

    def register_all_completions_as_proxy(self, model: str, url: str) -> None:
        """Register given model to chat completions and legacy completions."""
        self.register_chat_completion_as_proxy(model, ProxyOptions(url=f"{url}/v1/chat/completions"))
        self.register_completion_as_proxy(model, ProxyOptions(url=f"{url}/v1/completions"))

    def unregister_all_completions(self, model: str) -> None:
        """Unregister given model from chat completions and legacy completions."""
        self.unregister_chat_completion(model)
        self.unregister_completion(model)

    def register_embeddings(self, model: str, endpoint: SimpleEndpoint[EmbeddingRequest]) -> None:
        """Register embeddings endpoint for given model."""
        if model in self.embeddings_endpoints:
            raise AppError("There is already registered endpoint for given model", model)
        self.embeddings_endpoints[model] = endpoint

    def register_embeddings_as_proxy(self, model: str, options: ProxyOptions) -> None:
        """Register embeddings for given model as a proxy."""

        async def on_request(body: EmbeddingRequest, request: Request) -> StreamingResponse:
            return await self._proxy(body, options, request)

        self.register_embeddings(model, SimpleEndpoint(on_request=on_request))

    def unregister_embeddings(self, model: str) -> None:
        """Unregister embeddings for given model."""
        if model in self.embeddings_endpoints:
            del self.embeddings_endpoints[model]

    def register_audio_speech(self, model: str | list[str], endpoint: SimpleEndpoint[CreateSpeechRequest]) -> None:
        """Register audio speech endpoint for given model."""
        models = model if isinstance(model, list) else [model]
        for model in models:
            if model in self.audio_speech_endpoints:
                raise AppError("There is already registered endpoint for given model", model)

        for model in models:
            self.audio_speech_endpoints[model] = endpoint

    def register_audio_speech_as_proxy(self, model: str | list[str], options: ProxyOptions) -> None:
        """Register audio speech for given model as a proxy."""

        async def on_request(body: CreateSpeechRequest, request: Request) -> StreamingResponse:
            return await self._proxy(body, options, request)

        self.register_audio_speech(model, SimpleEndpoint(on_request=on_request))

    def unregister_audio_speech(self, model: str | list[str]) -> None:
        """Unregister audio speech for given model."""
        models = model if isinstance(model, list) else [model]
        for model in models:
            if model in self.audio_speech_endpoints:
                del self.audio_speech_endpoints[model]

    def register_audio_transcriptions(self, model: str | list[str], endpoint: SimpleEndpoint[CreateTranscriptionRequest]) -> None:
        """Register audio transcriptions endpoint for given model."""
        models = model if isinstance(model, list) else [model]
        for model in models:
            if model in self.audio_transcriptions_endpoints:
                raise AppError("There is already registered endpoint for given model", model)

        for model in models:
            self.audio_transcriptions_endpoints[model] = endpoint

    def register_audio_transcriptions_as_proxy(self, model: str | list[str], options: ProxyOptions) -> None:
        """Register audio transcriptions for given model as a proxy."""

        async def on_request(body: CreateTranscriptionRequest, request: Request) -> StreamingResponse:
            form = FormData()

            for field_name in CreateTranscriptionRequest.model_fields:
                if field_name == "file":
                    form.add_field("file", await body.file.read(), filename=body.file.filename, content_type=body.file.content_type)
                else:
                    field_value = getattr(body, field_name)
                    if field_value:
                        if isinstance(field_value, list):
                            for element in field_value:  # pyright: ignore[reportUnknownVariableType]
                                form.add_field(field_name + "[]", element)
                        elif isinstance(field_value, bool):
                            form.add_field(field_name, "true" if field_value else "false")
                        else:
                            form.add_field(field_name, field_value)

            return await self._proxy(form, options, request)

        self.register_audio_transcriptions(model, SimpleEndpoint(on_request=on_request))

    def unregister_audio_transcriptions(self, model: str | list[str]) -> None:
        """Unregister audio transcriptions for given model."""
        models = model if isinstance(model, list) else [model]
        for model in models:
            if model in self.audio_transcriptions_endpoints:
                del self.audio_transcriptions_endpoints[model]

    def register_image_generations(self, model: str, endpoint: SimpleEndpoint[ImagesRequest]) -> None:
        """Register image generations endpoint for given model."""
        if model in self.images_generations_endpoints:
            raise AppError("There is already registered endpoint for given model", model)
        self.images_generations_endpoints[model] = endpoint

    def register_image_generations_as_proxy(self, model: str, options: ProxyOptions) -> None:
        """Register image generations for given model as a proxy."""

        async def on_request(body: ImagesRequest, request: Request) -> StreamingResponse:
            return await self._proxy(body, options, request)

        self.register_image_generations(model, SimpleEndpoint(on_request=on_request))

    def unregister_image_generations(self, model: str) -> None:
        """Unregister image generations for given model."""
        if model in self.images_generations_endpoints:
            del self.images_generations_endpoints[model]

    def register_custom_endpoint(self, url: str, endpoint: SimpleEndpoint[None]) -> None:
        """Register custom endpoint."""
        if url in self.custom_endpoints:
            raise AppError("There is already registered endpoint for given url", url)
        self.custom_endpoints[url] = endpoint

    def register_custom_endpoint_as_proxy(self, url: str, options: ProxyOptions) -> None:
        """Register custom endpoint as a proxy."""

        async def on_request(_body: None, request: Request) -> StreamingResponse:
            data = await request.body()
            return await self._proxy(data, options, request)

        self.register_custom_endpoint(url, SimpleEndpoint(on_request=on_request))

    def unregister_custom_endpoint(self, model: str) -> None:
        """Unregister custom endpoint."""
        if model in self.custom_endpoints:
            del self.custom_endpoints[model]

    def has_chat_completion_model(self, model: str) -> bool:
        """Check whether the chat completion model is registered."""
        return model in self.chat_completion_endpoints

    def has_completion_model(self, model: str) -> bool:
        """Check whether the completion model is registered."""
        return model in self.completion_endpoints

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

    async def execute_chat_completion(self, data: ChatCompletionRequest, request: Request) -> StarletteResponse:
        """Process chat completion request."""
        endpoint = self.chat_completion_endpoints[data.model]
        if not endpoint:
            raise AppError("Given model is not supported", data.model)
        return await endpoint.on_request(data, request)

    async def execute_completion(self, data: CompletionLegacyRequest, request: Request) -> StarletteResponse:
        """Process completion request."""
        endpoint = self.completion_endpoints[data.model]
        if not endpoint:
            raise AppError("Given model is not supported", data.model)
        return await endpoint.on_request(data, request)

    async def execute_embeddings(self, data: EmbeddingRequest, request: Request) -> StarletteResponse:
        """Process embeddings request."""
        endpoint = self.embeddings_endpoints[data.model]
        if not endpoint:
            raise AppError("Given model is not supported", data.model)
        return await endpoint.on_request(data, request)

    async def execute_images_generations(self, data: ImagesRequest, request: Request) -> StarletteResponse:
        """Process images generations request."""
        endpoint = self.images_generations_endpoints[data.model]
        if not endpoint:
            raise AppError("Given model is not supported", data.model)
        return await endpoint.on_request(data, request)

    async def execute_audio_speech(self, data: CreateSpeechRequest, request: Request) -> StarletteResponse:
        """Process audio speech request."""
        endpoint = self.audio_speech_endpoints.get(data.model)
        if not endpoint:
            raise AppError("Given model is not supported", data.model)
        return await endpoint.on_request(data, request)

    async def execute_audio_transcriptions(self, data: CreateTranscriptionRequest, request: Request) -> StarletteResponse:
        """Process audio transcriptions request."""
        endpoint = self.audio_transcriptions_endpoints.get(data.model)
        if not endpoint:
            raise AppError("Given model is not supported", data.model)
        return await endpoint.on_request(data, request)

    async def execute_custom_endpoints(self, url: str, request: Request) -> StarletteResponse:
        """Process custom endpoint request."""
        endpoint = self.custom_endpoints[url]
        if not endpoint:
            raise AppError("Given url is not supported", url)
        return await endpoint.on_request(None, request)

    async def _proxy(
        self,
        body: bytes | BaseModel | JsonSerializable | FormFields | FormData,
        options: ProxyOptions,
        _request: Request,
        headers: dict[str, str] | None = None,
    ) -> StreamingResponse:
        if not isinstance(body, (FormData | bytes | bytearray | memoryview)):
            body = body.model_dump() if isinstance(body, BaseModel) else body
            if options.remove_model:
                del body["model"]
            if options.rewrite_model_to:
                body["model"] = options.rewrite_model_to

        (status_code, media_type, generator) = await proxy_post_request(
            options.url,
            body,
            headers=headers,
            form=options.form,
        )
        return StreamingResponse(generator, media_type=media_type, status_code=status_code)


async def proxy_post_request(
    url: str,
    body: bytes | JsonSerializable | FormFields | FormData,
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
