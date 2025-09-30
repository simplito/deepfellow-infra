"""Endpoint registry holds callbacks for given endpoints and models."""

import logging
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import TYPE_CHECKING, NamedTuple
from urllib.parse import urljoin

from aiohttp import ClientResponse, FormData, JsonPayload, Payload
from aiohttp.client import ClientSession
from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from server.config import AppSettings
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
    ModelProps,
)
from server.models.common import StarletteResponse

if TYPE_CHECKING:
    from server.websockets.usage_manager import UsageManager

logger = logging.getLogger("uvicorn.error")


type EndpointCallback[T] = Callable[[T, Request], Awaitable[StarletteResponse]]


class SimpleEndpoint[T](NamedTuple):
    on_request: EndpointCallback[T]


class ChatCompletionEndpoint:
    on_chat_completion: EndpointCallback[ChatCompletionRequest] | None = None
    on_completion: EndpointCallback[CompletionLegacyRequest] | None = None


type ModelId = str
type RegistrationId = str


class RegisteredModel[T](NamedTuple):
    id: RegistrationId
    props: ModelProps
    endpoint: T


class Endpoint[T]:
    def __init__(self):
        self.models = dict[ModelId, dict[RegistrationId, RegisteredModel[T]]]()

    def add_model(self, model: ModelId, props: ModelProps, endpoint: T) -> RegistrationId:
        """Add model to registry."""
        registered_model = RegisteredModel(id=str(uuid.uuid4()), props=props, endpoint=endpoint)
        if model not in self.models:
            self.models[model] = {}
        self.models[model][registered_model.id] = registered_model
        return registered_model.id

    def remove_model(self, model: ModelId, registration_id: RegistrationId) -> None:
        """Remove model from registry."""
        if model not in self.models:
            return
        if registration_id in self.models[model]:
            del self.models[model][registration_id]
        if len(self.models[model]) == 0:
            del self.models[model]

    def has_model(self, model: str) -> bool:
        """Have model in registry."""
        return model in self.models and len(self.models[model]) > 0

    def get_model(self, model: str, filter: Callable[[T], bool] | None = None) -> T | None:
        """Get model from registry."""
        if model not in self.models:
            return None
        # TODO make load balancing
        for x in self.models[model].values():
            if not filter or filter(x.endpoint):
                return x.endpoint
        return None

    def list_models(self) -> list[ApiModel]:
        """List models from registry."""
        res = list[ApiModel]()
        for model_id, map in self.models.items():
            private = False
            for item in map.values():
                if item.props.private:
                    private = True
            res.append(ApiModel(id=model_id, object="model", created=0, owned_by="unknown", props=ModelProps(private=private)))
        return res


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


class ModelInfo(NamedTuple):
    id: str
    type: str


class EndpointRegistry:
    usage_manager: "UsageManager"

    def __init__(
        self,
        config: AppSettings,
    ):
        self.config = config
        self.chat_completion_endpoints = Endpoint[ChatCompletionEndpoint]()
        self.embeddings_endpoints = Endpoint[SimpleEndpoint[EmbeddingRequest]]()
        self.audio_speech_endpoints = Endpoint[SimpleEndpoint[CreateSpeechRequest]]()
        self.audio_transcriptions_endpoints = Endpoint[SimpleEndpoint[CreateTranscriptionRequest]]()
        self.custom_endpoints = Endpoint[SimpleEndpoint[None]]()
        self.images_generations_endpoints = Endpoint[SimpleEndpoint[ImagesRequest]]()

    def get_models(self) -> ApiModels:
        """Get models for api."""
        models: list[ApiModel] = []
        for model in self.chat_completion_endpoints.list_models():
            models.append(model)
        for model in self.embeddings_endpoints.list_models():
            models.append(model)
        for model in self.audio_speech_endpoints.list_models():
            models.append(model)
        for model in self.audio_transcriptions_endpoints.list_models():
            models.append(model)
        for model in self.images_generations_endpoints.list_models():
            models.append(model)
        return ApiModels(data=models)

    def get_model(self, model_id: str) -> ApiModel:
        """Get model for api."""
        models = self.get_models()
        model = next((x for x in models.data if x.id == model_id), None)
        if model is None:
            raise HTTPException(404, f"Model not found {model_id}")
        return model

    def list_models(self) -> list[ModelInfo]:
        """List models for load balancing."""
        models = list[ModelInfo]()
        for model in self.chat_completion_endpoints.list_models():
            models.append(ModelInfo(id=model.id, type="llm"))
        for model in self.embeddings_endpoints.list_models():
            models.append(ModelInfo(id=model.id, type="llm"))
        for model in self.audio_speech_endpoints.list_models():
            models.append(ModelInfo(id=model.id, type="llm"))
        for model in self.audio_transcriptions_endpoints.list_models():
            models.append(ModelInfo(id=model.id, type="llm"))
        for model in self.images_generations_endpoints.list_models():
            models.append(ModelInfo(id=model.id, type="llm"))
        return models

    def register_chat_completion(self, model: str, props: ModelProps, endpoint: ChatCompletionEndpoint) -> RegistrationId:
        """Register chat completion endpoint for given model."""
        return self.chat_completion_endpoints.add_model(model, props, endpoint)

    def register_chat_completion_as_proxy(
        self,
        model: str,
        props: ModelProps,
        chat_completions: ProxyOptions | None,
        completions: ProxyOptions | None,
    ) -> RegistrationId:
        """Register chat completion for given model as a proxy."""
        endpoint = ChatCompletionEndpoint()
        if chat_completions:

            async def on_chat_completions_request(body: ChatCompletionRequest, request: Request) -> StreamingResponse:
                return await post_json(body, chat_completions, request)

            endpoint.on_chat_completion = on_chat_completions_request
        if completions:

            async def on_completion_request(body: CompletionLegacyRequest, request: Request) -> StreamingResponse:
                return await post_json(body, completions, request)

            endpoint.on_completion = on_completion_request

        return self.register_chat_completion(model, props, endpoint)

    def unregister_chat_completion(self, model: str, registration_id: RegistrationId) -> None:
        """Unregister chat completion for given model."""
        self.chat_completion_endpoints.remove_model(model, registration_id)

    def register_embeddings(self, model: str, props: ModelProps, endpoint: SimpleEndpoint[EmbeddingRequest]) -> RegistrationId:
        """Register embeddings endpoint for given model."""
        return self.embeddings_endpoints.add_model(model, props, endpoint)

    def register_embeddings_as_proxy(self, model: str, props: ModelProps, options: ProxyOptions) -> RegistrationId:
        """Register embeddings for given model as a proxy."""

        async def on_request(body: EmbeddingRequest, request: Request) -> StreamingResponse:
            return await post_json(body, options, request)

        return self.register_embeddings(model, props, SimpleEndpoint(on_request=on_request))

    def unregister_embeddings(self, model: str, registration_id: RegistrationId) -> None:
        """Unregister embeddings for given model."""
        self.embeddings_endpoints.remove_model(model, registration_id)

    def register_audio_speech(self, model: str, props: ModelProps, endpoint: SimpleEndpoint[CreateSpeechRequest]) -> RegistrationId:
        """Register audio speech endpoint for given model."""
        return self.audio_speech_endpoints.add_model(model, props, endpoint)

    def register_audio_speech_as_proxy(self, model: str, props: ModelProps, options: ProxyOptions) -> RegistrationId:
        """Register audio speech for given model as a proxy."""

        async def on_request(body: CreateSpeechRequest, request: Request) -> StreamingResponse:
            return await post_json(body, options, request)

        return self.register_audio_speech(model, props, SimpleEndpoint(on_request=on_request))

    def unregister_audio_speech(self, model: str, registration_id: RegistrationId) -> None:
        """Unregister audio speech for given model."""
        self.audio_speech_endpoints.remove_model(model, registration_id)

    def register_audio_transcriptions(
        self,
        model: str,
        props: ModelProps,
        endpoint: SimpleEndpoint[CreateTranscriptionRequest],
    ) -> RegistrationId:
        """Register audio transcriptions endpoint for given model."""
        return self.audio_transcriptions_endpoints.add_model(model, props, endpoint)

    def register_audio_transcriptions_as_proxy(self, model: str, props: ModelProps, options: ProxyOptions) -> RegistrationId:
        """Register audio transcriptions for given model as a proxy."""

        async def on_request(body: CreateTranscriptionRequest, request: Request) -> StreamingResponse:
            return await post_form(body, options, request)

        return self.register_audio_transcriptions(model, props, SimpleEndpoint(on_request=on_request))

    def unregister_audio_transcriptions(self, model: str, registration_id: RegistrationId) -> None:
        """Unregister audio transcriptions for given model."""
        self.audio_transcriptions_endpoints.remove_model(model, registration_id)

    def register_image_generations(self, model: str, props: ModelProps, endpoint: SimpleEndpoint[ImagesRequest]) -> RegistrationId:
        """Register image generations endpoint for given model."""
        return self.images_generations_endpoints.add_model(model, props, endpoint)

    def register_image_generations_as_proxy(self, model: str, props: ModelProps, options: ProxyOptions) -> RegistrationId:
        """Register image generations for given model as a proxy."""

        async def on_request(body: ImagesRequest, request: Request) -> StreamingResponse:
            return await post_json(body, options, request)

        return self.register_image_generations(model, props, SimpleEndpoint(on_request=on_request))

    def unregister_image_generations(self, model: str, registration_id: RegistrationId) -> None:
        """Unregister image generations for given model."""
        self.images_generations_endpoints.remove_model(model, registration_id)

    def register_custom_endpoint(self, url: str, props: ModelProps, endpoint: SimpleEndpoint[None]) -> RegistrationId:
        """Register custom endpoint."""
        return self.custom_endpoints.add_model(url, props, endpoint)

    def register_custom_endpoint_as_proxy(self, url: str, props: ModelProps, options: ProxyOptions) -> RegistrationId:
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

        return self.register_custom_endpoint(url, props, SimpleEndpoint(on_request=on_request))

    def unregister_custom_endpoint(self, model: str, registration_id: RegistrationId) -> None:
        """Unregister custom endpoint."""
        self.custom_endpoints.remove_model(model, registration_id)

    def has_chat_completion_model(self, model: str) -> bool:
        """Check whether the chat completion model is registered."""
        endpoint = self.chat_completion_endpoints.get_model(model)
        return endpoint is not None and endpoint.on_chat_completion is not None

    def has_completion_model(self, model: str) -> bool:
        """Check whether the completion model is registered."""
        endpoint = self.chat_completion_endpoints.get_model(model)
        return endpoint is not None and endpoint.on_completion is not None

    def has_embeddings_model(self, model: str) -> bool:
        """Check whether the embeddings model is registered."""
        return self.embeddings_endpoints.has_model(model)

    def has_audio_speech_model(self, model: str) -> bool:
        """Check whether the audio speech model is registered."""
        return self.audio_speech_endpoints.has_model(model)

    def has_audio_transcriptions_model(self, model: str) -> bool:
        """Check whether the audio transcriptions model is registered."""
        return self.audio_transcriptions_endpoints.has_model(model)

    def has_image_generations_model(self, model: str) -> bool:
        """Check whether the image generations model is registered."""
        return self.images_generations_endpoints.has_model(model)

    def has_custom_endpoint(self, url: str) -> bool:
        """Check whether the custom endpoint is registered."""
        return self.custom_endpoints.has_model(url)

    async def execute_chat_completion(self, request: Request, model: str, body: ChatCompletionRequest) -> StarletteResponse:
        """Process chat completion request."""
        if self.config.is_log_payloads_enabled():
            logger.info(f"DUMP REQUEST PAYLOAD /v1/chat/completions {body.model_dump_json(exclude_none=True)}")  # noqa: G004

        async def func() -> StarletteResponse:
            endpoint = self.chat_completion_endpoints.get_model(model, lambda x: x.on_chat_completion is not None)
            if not endpoint or not endpoint.on_chat_completion:
                msg = (
                    "Given model is only supported in legacy /v1/completions"
                    if self.chat_completion_endpoints.has_model(model)
                    else "Given model is not supported"
                )
                raise HTTPException(400, msg)
            return await endpoint.on_chat_completion(body, request)

        return await self.usage_manager.with_usage(model, func, self.config.is_log_payloads_enabled())

    async def execute_completion(self, request: Request, model: str, body: CompletionLegacyRequest) -> StarletteResponse:
        """Process completion request."""

        async def func() -> StarletteResponse:
            endpoint = self.chat_completion_endpoints.get_model(model, lambda x: x.on_completion is not None)
            if not endpoint or not endpoint.on_completion:
                msg = (
                    "Given model is only supported in /v1/chat/completions"
                    if self.chat_completion_endpoints.has_model(model)
                    else "Given model is not supported"
                )
                raise HTTPException(400, msg)
            return await endpoint.on_completion(body, request)

        return await self.usage_manager.with_usage(model, func)

    async def execute_embeddings(self, request: Request, model: str, body: EmbeddingRequest) -> StarletteResponse:
        """Process embeddings request."""

        async def func() -> StarletteResponse:
            endpoint = self.embeddings_endpoints.get_model(model)
            if not endpoint:
                raise HTTPException(400, "Given model is not supported")
            return await endpoint.on_request(body, request)

        return await self.usage_manager.with_usage(model, func)

    async def execute_images_generations(self, request: Request, model: str, body: ImagesRequest) -> StarletteResponse:
        """Process images generations request."""

        async def func() -> StarletteResponse:
            endpoint = self.images_generations_endpoints.get_model(model)
            if not endpoint:
                raise HTTPException(400, "Given model is not supported")
            return await endpoint.on_request(body, request)

        return await self.usage_manager.with_usage(model, func)

    async def execute_audio_speech(self, request: Request, model: str, body: CreateSpeechRequest) -> StarletteResponse:
        """Process audio speech request."""

        async def func() -> StarletteResponse:
            endpoint = self.audio_speech_endpoints.get_model(model)
            if not endpoint:
                raise HTTPException(400, "Given model is not supported")
            return await endpoint.on_request(body, request)

        return await self.usage_manager.with_usage(model, func)

    async def execute_audio_transcriptions(
        self,
        request: Request,
        model: str,
        body: CreateTranscriptionRequest,
    ) -> StarletteResponse:
        """Process audio transcriptions request."""

        async def func() -> StarletteResponse:
            endpoint = self.audio_transcriptions_endpoints.get_model(model)
            if not endpoint:
                raise HTTPException(400, "Given model is not supported")
            return await endpoint.on_request(body, request)

        return await self.usage_manager.with_usage(model, func)

    async def execute_custom_endpoints(self, request: Request, url: str) -> StarletteResponse:
        """Process custom endpoint request."""
        endpoint = self.custom_endpoints.get_model(url)
        if not endpoint:
            raise HTTPException(400, "Given url is not supported")
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
