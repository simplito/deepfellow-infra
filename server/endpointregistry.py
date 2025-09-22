"""Endpoint registry holds callbacks for given endpoints and models."""

import logging
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import NamedTuple
from urllib.parse import urljoin

from aiohttp import ClientResponse, FormData, JsonPayload, Payload
from aiohttp.client import ClientSession
from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from server.models.api import (
    ApiModel,
    ApiModels,
    ChatCompletionRequest,
    CompletionLegacyRequest,
    CreateSpeechRequest,
    CreateTranscriptionRequest,
    EmbeddingRequest,
    FormSerializable,
    ImagesRequest,
)
from server.models.common import StarletteResponse
from server.utils.exceptions import AppError
from server.websockets.utils import handle_usage

logger = logging.getLogger("uvicorn.error")


type EndpointCallback[T] = Callable[[T, Request], Awaitable[StarletteResponse]]


class SimpleEndpoint[T](NamedTuple):
    on_request: EndpointCallback[T]


class ProxyOptions(NamedTuple):
    url: str
    rewrite_model_to: str | None = None
    remove_model: bool = False
    headers: dict[str, str] | None = None
    allowed_response_headers: list[str] | None = None
    allowed_request_headers: list[str] | None = None

    def get_request_headers(self, request: Request | None) -> dict[str, str]:
        """Get request headers."""
        headers = self.headers or {}
        if not request:
            return headers
        allowed_request_headers = self.allowed_request_headers or []
        request_headers = {k: v for k, v in dict(request.headers).items() if k in allowed_request_headers}
        return request_headers | headers


class EndpointRegistry:
    def __init__(self):
        self.chat_completion_endpoints: dict[str, SimpleEndpoint[ChatCompletionRequest]] = {}
        self.completion_endpoints: dict[str, SimpleEndpoint[CompletionLegacyRequest]] = {}
        self.embeddings_endpoints: dict[str, SimpleEndpoint[EmbeddingRequest]] = {}
        self.audio_speech_endpoints: dict[str, SimpleEndpoint[CreateSpeechRequest]] = {}
        self.audio_transcriptions_endpoints: dict[str, SimpleEndpoint[CreateTranscriptionRequest]] = {}
        self.custom_endpoints: dict[str, SimpleEndpoint[None]] = {}
        self.images_generations_endpoints: dict[str, SimpleEndpoint[ImagesRequest]] = {}

    def get_models(self) -> ApiModels:
        """Get chat completions models."""
        models: list[ApiModel] = []
        for model_id in self.chat_completion_endpoints:
            models.append(ApiModel(id=model_id, object="model", created=0, owned_by="unknown"))
        for model_id in self.completion_endpoints:
            if model_id not in self.chat_completion_endpoints:
                models.append(ApiModel(id=model_id, object="model", created=0, owned_by="unknown"))
        for model_id in self.embeddings_endpoints:
            models.append(ApiModel(id=model_id, object="model", created=0, owned_by="unknown"))
        for model_id in self.audio_speech_endpoints:
            models.append(ApiModel(id=model_id, object="model", created=0, owned_by="unknown"))
        for model_id in self.audio_transcriptions_endpoints:
            models.append(ApiModel(id=model_id, object="model", created=0, owned_by="unknown"))
        for model_id in self.images_generations_endpoints:
            models.append(ApiModel(id=model_id, object="model", created=0, owned_by="unknown"))
        return ApiModels(data=models)

    def get_model(self, model_id: str) -> ApiModel:
        """Get chat completions model."""
        models = self.get_models()
        model = next((x for x in models.data if x.id == model_id), None)
        if model is None:
            raise HTTPException(404, f"Model not found {model_id}")
        return model

    def register_chat_completion(self, model: str, endpoint: SimpleEndpoint[ChatCompletionRequest]) -> None:
        """Register chat completion endpoint for given model."""
        if model in self.chat_completion_endpoints:
            raise AppError("There is already registered endpoint for given model", model)
        self.chat_completion_endpoints[model] = endpoint

    def register_chat_completion_as_proxy(self, model: str, options: ProxyOptions) -> None:
        """Register chat completion for given model as a proxy."""

        async def on_request(body: ChatCompletionRequest, request: Request) -> StreamingResponse:
            return await post_json(body, options, request)

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
            return await post_json(body, options, request)

        self.register_completion(model, SimpleEndpoint(on_request=on_request))

    def unregister_completion(self, model: str) -> None:
        """Unregister completion for given model."""
        if model in self.completion_endpoints:
            del self.completion_endpoints[model]

    def register_all_completions_as_proxy(self, model: str, url: str, rewrite_model_to: str | None = None) -> None:
        """Register given model to chat completions and legacy completions."""
        self.register_chat_completion_as_proxy(model, ProxyOptions(url=f"{url}/v1/chat/completions", rewrite_model_to=rewrite_model_to))
        self.register_completion_as_proxy(model, ProxyOptions(url=f"{url}/v1/completions", rewrite_model_to=rewrite_model_to))

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
            return await post_json(body, options, request)

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
            return await post_json(body, options, request)

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
            return await post_form(body, options, request)

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
            return await post_json(body, options, request)

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
            return (
                await make_http_request(
                    url=urljoin(options.url, request.url.path[7:]),
                    method=request.method,
                    data=request.stream(),
                    headers=options.get_request_headers(request),
                )
            ).as_streaming_response(options.allowed_response_headers)

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
        print(url, self.custom_endpoints.keys())
        return url in self.custom_endpoints

    @handle_usage
    async def execute_chat_completion(self, request: Request, model: str, body: ChatCompletionRequest) -> StarletteResponse:
        """Process chat completion request."""
        endpoint = self.chat_completion_endpoints[model]
        if not endpoint:
            raise AppError("Given model is not supported", model)
        return await endpoint.on_request(body, request)

    @handle_usage
    async def execute_completion(self, request: Request, model: str, body: CompletionLegacyRequest) -> StarletteResponse:
        """Process completion request."""
        endpoint = self.completion_endpoints[model]
        if not endpoint:
            raise AppError("Given model is not supported", model)
        return await endpoint.on_request(body, request)

    @handle_usage
    async def execute_embeddings(self, request: Request, model: str, body: EmbeddingRequest) -> StarletteResponse:
        """Process embeddings request."""
        endpoint = self.embeddings_endpoints[model]
        if not endpoint:
            raise AppError("Given model is not supported", model)
        return await endpoint.on_request(body, request)

    @handle_usage
    async def execute_images_generations(self, request: Request, model: str, body: ImagesRequest) -> StarletteResponse:
        """Process images generations request."""
        endpoint = self.images_generations_endpoints[model]
        if not endpoint:
            raise AppError("Given model is not supported", model)
        return await endpoint.on_request(body, request)

    @handle_usage
    async def execute_audio_speech(self, request: Request, model: str, body: CreateSpeechRequest) -> StarletteResponse:
        """Process audio speech request."""
        endpoint = self.audio_speech_endpoints.get(model)
        if not endpoint:
            raise AppError("Given model is not supported", model)
        return await endpoint.on_request(body, request)

    @handle_usage
    async def execute_audio_transcriptions(
        self,
        request: Request,
        model: str,
        body: CreateTranscriptionRequest,
    ) -> StarletteResponse:
        """Process audio transcriptions request."""
        endpoint = self.audio_transcriptions_endpoints.get(model)
        if not endpoint:
            raise AppError("Given model is not supported", model)
        return await endpoint.on_request(body, request)

    async def execute_custom_endpoints(self, request: Request, url: str) -> StarletteResponse:
        """Process custom endpoint request."""
        endpoint = self.custom_endpoints[url]
        if not endpoint:
            raise AppError("Given url is not supported", url)
        return await endpoint.on_request(None, request)


async def post_json(data: BaseModel, options: ProxyOptions, request: Request | None = None) -> StreamingResponse:
    """Make HTTP POST request sending data as JSON."""
    raw = data.model_dump(exclude_none=True)
    if options.remove_model:
        del raw["model"]  # pyright: ignore[reportIndexIssue]
    if options.rewrite_model_to:
        raw["model"] = options.rewrite_model_to  # pyright: ignore[reportIndexIssue]
    return (
        await make_http_request(
            url=options.url,
            method="POST",
            data=JsonPayload(raw),
            headers=options.get_request_headers(request),
        )
    ).as_streaming_response(options.allowed_response_headers)


async def post_form(data: FormSerializable, options: ProxyOptions, request: Request | None = None) -> StreamingResponse:
    """Make HTTP POST request sending data as form."""
    return (
        await make_http_request(
            url=options.url,
            method="POST",
            data=await data.to_form(options.remove_model, options.rewrite_model_to),
            headers=options.get_request_headers(request),
        )
    ).as_streaming_response(options.allowed_response_headers)


class HttpResponse:
    def __init__(self, response: ClientResponse, content: AsyncGenerator[bytes]):
        self.response = response
        self.content = content

    def as_streaming_response(self, allowed_response_headers: list[str] | None = None) -> StreamingResponse:
        """Return as StreamingResponse."""
        allowed_response_headers = allowed_response_headers or []
        response_headers = {k: v for k, v in dict(self.response.headers).items() if k in allowed_response_headers}
        return StreamingResponse(
            self.content, media_type=self.response.content_type, status_code=self.response.status, headers=response_headers
        )


async def make_http_request(
    url: str,
    method: str = "GET",
    data: AsyncGenerator[bytes] | bytes | FormData | Payload | None = None,
    headers: dict[str, str] | None = None,
) -> HttpResponse:
    """Make HTTP request to given url."""
    headers = headers or {}
    session = ClientSession()
    try:
        response = await session.request(method=method, url=url, data=data, headers=headers)

        async def generator() -> AsyncGenerator[bytes]:
            try:
                async for chunk in response.content.iter_any():
                    if chunk:
                        yield chunk
            finally:
                await response.release()
                await session.close()

        return HttpResponse(response=response, content=generator())

    except Exception:
        await session.close()
        raise
