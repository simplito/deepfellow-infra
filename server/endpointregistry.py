# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Endpoint registry holds callbacks for given endpoints and models."""

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple, TypeVar, cast
from urllib.parse import parse_qs, urljoin, urlparse

import aiohttp
from aiohttp import JsonPayload
from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from server.config import AppSettings
from server.metrics_registry import MetricsRegistry
from server.models.api import (
    ALL_LLM_SUFFIXES,
    LLM_SUFFIX_MAP,
    ApiModel,
    ApiModels,
    ChatCompletionRequest,
    CompletionLegacyRequest,
    CreateSpeechRequest,
    CreateTranscriptionRequest,
    EmbeddingRequest,
    FormSerializable,
    ImagesRequest,
    MessagesRequest,
    Model,
    ModelId,
    ModelProps,
    ModelType,
    OllamaChatRequest,
    RerankRequest,
    ResponsesRequest,
)
from server.models.common import JsonSerializable, StarletteResponse
from server.utils.core import HttpClientError, Utils, make_http_request
from server.websockets.models import RegistrationId, UsageChangeRequest
from server.websockets.parent_infra_group import ParentInfraGroup

if TYPE_CHECKING:
    from server.model_tester import ModelTester

logger = logging.getLogger("uvicorn.error")
T = TypeVar("T")

type EndpointCallback[T] = Callable[[T, Request | None], Awaitable[StarletteResponse]]
type CustomEndpointCallback = Callable[[Request], Awaitable[StarletteResponse]]
type McpEndpointCallback = Callable[[Request], Awaitable[StarletteResponse]]


class SimpleEndpoint[T](NamedTuple):
    on_request: EndpointCallback[T]


class CustomEndpoint(NamedTuple):
    on_request: CustomEndpointCallback


class McpEndpoint(NamedTuple):
    on_request: McpEndpointCallback


class ChatCompletionEndpoint:
    on_chat_completion: EndpointCallback[ChatCompletionRequest] | None = None
    on_completion: EndpointCallback[CompletionLegacyRequest] | None = None
    on_responses: EndpointCallback[ResponsesRequest] | None = None
    on_messages: EndpointCallback[MessagesRequest] | None = None
    on_ollama_chat: EndpointCallback[OllamaChatRequest] | None = None


class RegisteredModel[T](BaseModel):
    id: RegistrationId
    name: ModelId
    origin: str
    props: ModelProps
    type: str
    endpoint: T
    usage: int


class RegistrationOptions(NamedTuple):
    origin: str
    id: RegistrationId | None = None
    usage: int | None = None
    send_notification: bool = True


class RegistryEntry(NamedTuple):
    model_id: ModelId
    endpoint: "Endpoint[Any]"
    registered_model: RegisteredModel[Any]


class Endpoint[T]:
    def __init__(
        self,
        registry: dict[RegistrationId, RegistryEntry],
        parent_infra: ParentInfraGroup,
    ):
        self.registry = registry
        self.parent_infra = parent_infra
        self.models = dict[ModelId, dict[RegistrationId, RegisteredModel[T]]]()

    def add_model(
        self, model_id: ModelId, props: ModelProps, endpoint: T, type: str, options: RegistrationOptions | None
    ) -> RegistrationId:
        """Add model to registry."""
        registered_model = RegisteredModel(
            id=options.id if options and options.id is not None else str(uuid.uuid4()),
            name=model_id,
            origin=options.origin if options else "local",
            props=props,
            endpoint=endpoint,
            type=type,
            usage=options.usage if options and options.usage is not None else 0,
        )
        if model_id not in self.models:
            self.models[model_id] = {}
        self.models[model_id][registered_model.id] = registered_model
        self.registry[registered_model.id] = RegistryEntry(model_id=model_id, endpoint=self, registered_model=registered_model)
        if not options or options.send_notification:
            self.parent_infra.send_models_list()
        return registered_model.id

    def remove_model(self, model_id: ModelId, registration_id: RegistrationId, send_notification: bool = True) -> None:
        """Remove model from registry."""
        if model_id not in self.models:
            return
        if registration_id in self.models[model_id]:
            del self.models[model_id][registration_id]
        if len(self.models[model_id]) == 0:
            del self.models[model_id]
        if registration_id in self.registry:
            del self.registry[registration_id]
        if send_notification:
            self.parent_infra.send_models_list()

    def has_model(self, model_id: ModelId) -> bool:
        """Have model in registry."""
        return model_id in self.models and len(self.models[model_id]) > 0

    def get_model(
        self,
        model_id: ModelId,
        filter: Callable[[T], bool] | None = None,
        registration_id: RegistrationId | None = None,
    ) -> RegisteredModel[T] | None:
        """Get model from registry."""
        if model_id not in self.models:
            return None
        lowest: RegisteredModel[T] | None = None
        for x in self.models[model_id].values():
            if (not filter or filter(x.endpoint)) and (lowest is None or x.usage < lowest.usage):
                if registration_id:
                    if x.id == registration_id:
                        return x
                else:
                    if x.usage == 0:
                        logger.debug(f"Choosen model origin={x.origin} id={x.id} name={x.name} usage={x.usage}")  # noqa: G004
                        return x
                    lowest = x
        if lowest:
            logger.debug(f"Choosen model origin={lowest.origin} id={lowest.id} name={lowest.name} usage={lowest.usage}")  # noqa: G004
        return lowest

    def is_model_private(self, model: dict[RegistrationId, RegisteredModel[T]]) -> bool:
        """Return is model private."""
        return any(item.props.private for item in model.values())

    def get_model_type(self, model: dict[RegistrationId, RegisteredModel[T]]) -> str:
        """Return unique model types."""
        return next(iter({item.type for item in model.values()}))

    def get_model_available_endpoints(self, model: dict[RegistrationId, RegisteredModel[T]]) -> list[str]:
        """Get model available endpoints."""
        return list({endpoint for item in model.values() for endpoint in item.props.endpoints})

    def get_model_context_window(self, model: dict[RegistrationId, RegisteredModel[T]]) -> int:
        """Get model context window."""
        return max([0, *list({item.props.context_window for item in model.values() if item.props.context_window})])

    def get_max_context_window(self, model: dict[RegistrationId, RegisteredModel[T]]) -> int:
        """Get model max context window."""
        return max([0, *list({item.props.max_context_window for item in model.values() if item.props.max_context_window})])

    def get_models(self) -> list[ApiModel]:
        """List models from registry."""
        res = list[ApiModel]()
        for model_id, model in self.models.items():
            res.append(
                ApiModel(
                    id=model_id,
                    object="model",
                    created=0,
                    owned_by="unknown",
                    props=ModelProps(
                        private=self.is_model_private(model),
                        type=self.get_model_type(model),
                        endpoints=self.get_model_available_endpoints(model),
                        context_window=self.get_model_context_window(model),
                        max_context_window=self.get_max_context_window(model),
                    ),
                )
            )
        return res

    def list_models(self) -> list[Model]:
        """List models from registry."""
        res = list[Model]()
        for model_id, map in self.models.items():
            for item in map.values():
                res.append(
                    Model(
                        id=item.id,
                        name=model_id,
                        type=cast("ModelType", item.type.split("-")[0]),
                        props=item.props,
                        usage=item.usage,
                    )
                )
        return res


class McpSseSessionStore:
    """Maps session IDs to upstream message endpoint URLs for SSE transport.

    All mutations are plain dict operations with no await points inside them,
    so they are atomic from asyncio's single-threaded event loop perspective.
    No asyncio.Lock is needed here — that would only be required with threads or
    if an await were introduced inside a mutation.
    """

    def __init__(self, ttl_seconds: int = 300, max_sessions: int = 128) -> None:
        self._sessions: dict[str, tuple[str, float]] = {}
        self._ttl = ttl_seconds
        self._max = max_sessions

    def _evict_expired(self) -> None:
        """Remove all sessions whose TTL has elapsed."""
        now = time.monotonic()
        expired = [sid for sid, (_, ts) in self._sessions.items() if now - ts >= self._ttl]
        for sid in expired:
            del self._sessions[sid]

    def add(self, session_id: str, upstream_url: str) -> None:
        """Store session → upstream messages URL mapping, evicting expired/oldest if at capacity."""
        self._evict_expired()
        if len(self._sessions) >= self._max:
            # evict oldest (first inserted) to make room
            oldest = next(iter(self._sessions))
            del self._sessions[oldest]
        self._sessions[session_id] = (upstream_url, time.monotonic())

    def get(self, session_id: str) -> str | None:
        """Return upstream messages URL for a session, or None if unknown or expired."""
        entry = self._sessions.get(session_id)
        if entry is None:
            return None
        url, ts = entry
        if time.monotonic() - ts >= self._ttl:
            del self._sessions[session_id]
            return None
        return url

    def remove(self, session_id: str) -> None:
        """Remove a session mapping."""
        self._sessions.pop(session_id, None)


async def _rewrite_sse_endpoint_events(
    content: AsyncGenerator[bytes],
    upstream_sse_url: str,
    proxy_endpoint_url: str,
    session_store: McpSseSessionStore,
) -> AsyncGenerator[bytes]:
    """Stream SSE events, rewriting 'endpoint' event data URLs to point through the proxy."""
    buffer = ""
    in_endpoint_event = False

    async for chunk in content:
        buffer += chunk.decode("utf-8", errors="replace")

        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.rstrip("\r")

            if not line:
                in_endpoint_event = False
                yield b"\n"
                continue

            if line.startswith("event:"):
                in_endpoint_event = line[6:].strip() == "endpoint"
                yield (line + "\n").encode()
            elif line.startswith("data:") and in_endpoint_event:
                raw = line[5:].strip()
                upstream_url = raw if raw.startswith("http") else urljoin(upstream_sse_url, raw)
                parsed = urlparse(upstream_url)
                session_id = (parse_qs(parsed.query).get("sessionId") or [None])[0]
                if session_id:
                    session_store.add(session_id, upstream_url)
                    yield f"data: {proxy_endpoint_url}?sessionId={session_id}\n".encode()
                else:
                    yield (line + "\n").encode()
            else:
                yield (line + "\n").encode()

    if buffer:
        yield buffer.encode()


@dataclass
class ProxyOptions:
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
    def __init__(
        self,
        config: AppSettings,
        parent_infra: ParentInfraGroup,
        model_tester: "ModelTester",
        metrics_registry: MetricsRegistry,
    ):
        self.config = config
        self.parent_infra = parent_infra
        self.parent_infra.endpoint_registry = self
        self.model_tester = model_tester
        self.metrics_registry = metrics_registry
        self.registry = dict[RegistrationId, RegistryEntry]()
        self.chat_completion_endpoints = Endpoint[ChatCompletionEndpoint](self.registry, self.parent_infra)
        self.embeddings_endpoints = Endpoint[SimpleEndpoint[EmbeddingRequest]](self.registry, self.parent_infra)
        self.audio_speech_endpoints = Endpoint[SimpleEndpoint[CreateSpeechRequest]](self.registry, self.parent_infra)
        self.audio_transcriptions_endpoints = Endpoint[SimpleEndpoint[CreateTranscriptionRequest]](self.registry, self.parent_infra)
        self.custom_endpoints = Endpoint[CustomEndpoint](self.registry, self.parent_infra)
        self.images_generations_endpoints = Endpoint[SimpleEndpoint[ImagesRequest]](self.registry, self.parent_infra)
        self.rerank_endpoints = Endpoint[SimpleEndpoint[RerankRequest]](self.registry, self.parent_infra)
        self.mcp_endpoints = Endpoint[McpEndpoint](self.registry, self.parent_infra)

    def get_models(self) -> ApiModels:
        """Get models for api."""
        models: list[ApiModel] = []
        for model in self.chat_completion_endpoints.get_models():
            models.append(model)
        for model in self.embeddings_endpoints.get_models():
            models.append(model)
        for model in self.audio_speech_endpoints.get_models():
            models.append(model)
        for model in self.audio_transcriptions_endpoints.get_models():
            models.append(model)
        for model in self.images_generations_endpoints.get_models():
            models.append(model)
        for model in self.rerank_endpoints.get_models():
            models.append(model)
        return ApiModels(data=models)

    def get_model(self, model_id: str) -> ApiModel:
        """Get model for api."""
        models = self.get_models()
        model = next((x for x in models.data if x.id == model_id), None)
        if model is None:
            raise HTTPException(404, f"Model not found {model_id}")
        return model

    def list_models(self) -> list[Model]:
        """List models for load balancing."""
        models = list[Model]()
        models.extend(self.chat_completion_endpoints.list_models())
        models.extend(self.embeddings_endpoints.list_models())
        models.extend(self.audio_speech_endpoints.list_models())
        models.extend(self.audio_transcriptions_endpoints.list_models())
        models.extend(self.images_generations_endpoints.list_models())
        models.extend(self.rerank_endpoints.list_models())
        models.extend(self.custom_endpoints.list_models())
        models.extend(self.mcp_endpoints.list_models())
        return models

    def model_exists(self, model_id: ModelId) -> bool:
        """Check if model exists in any endpoint."""
        for reg in (
            self.chat_completion_endpoints,
            self.embeddings_endpoints,
            self.audio_speech_endpoints,
            self.audio_transcriptions_endpoints,
            self.images_generations_endpoints,
            self.rerank_endpoints,
        ):
            if model_id in reg.models:
                return True
        return False

    def register_chat_completion(
        self,
        model: str,
        props: ModelProps,
        endpoint: ChatCompletionEndpoint,
        registration_options: RegistrationOptions | None,
    ) -> RegistrationId:
        """Register chat completion endpoint for given model."""
        suffixes = [suffix for field, suffix in LLM_SUFFIX_MAP if getattr(endpoint, field)]

        model_type = "llm" if len(suffixes) == len(ALL_LLM_SUFFIXES) else "llm-" + "-".join(suffixes)
        return self.chat_completion_endpoints.add_model(model, props, endpoint, model_type, registration_options)

    def register_chat_completion_as_proxy(  # noqa: C901
        self,
        model: str,
        props: ModelProps,
        chat_completions: ProxyOptions | None,
        completions: ProxyOptions | None,
        responses: ProxyOptions | None,
        messages: ProxyOptions | None,
        ollama_chat: ProxyOptions | None,
        registration_options: RegistrationOptions | None,
    ) -> RegistrationId:
        """Register chat completion for given model as a proxy."""
        endpoint = ChatCompletionEndpoint()
        if not chat_completions and not completions:
            raise RuntimeError("Chat completions nor completions registered " + model)
        if chat_completions:

            async def on_chat_completions_request(body: ChatCompletionRequest, request: Request | None) -> StreamingResponse:
                return await post_json(body, chat_completions, request)

            endpoint.on_chat_completion = on_chat_completions_request
        if completions:

            async def on_completion_request(body: CompletionLegacyRequest, request: Request | None) -> StreamingResponse:
                return await post_json(body, completions, request)

            endpoint.on_completion = on_completion_request
        if responses:

            async def on_responses_request(body: ResponsesRequest, request: Request | None) -> StreamingResponse:
                return await post_json(body, responses, request)

            endpoint.on_responses = on_responses_request

        if messages:

            async def on_messages_request(body: MessagesRequest, request: Request | None) -> StreamingResponse:
                return await post_json(body, messages, request)

            endpoint.on_messages = on_messages_request

        if ollama_chat:

            async def on_ollama_chat_request(body: OllamaChatRequest, request: Request | None) -> StreamingResponse:
                return await post_json(body, ollama_chat, request)

            endpoint.on_ollama_chat = on_ollama_chat_request

        return self.register_chat_completion(model, props, endpoint, registration_options)

    def unregister_chat_completion(self, model: str, registration_id: RegistrationId) -> None:
        """Unregister chat completion for given model."""
        self.chat_completion_endpoints.remove_model(model, registration_id)

    def register_embeddings(
        self,
        model: str,
        props: ModelProps,
        endpoint: SimpleEndpoint[EmbeddingRequest],
        registration_options: RegistrationOptions | None,
    ) -> RegistrationId:
        """Register embeddings endpoint for given model."""
        return self.embeddings_endpoints.add_model(model, props, endpoint, "embedding", registration_options)

    def register_embeddings_as_proxy(
        self,
        model: str,
        props: ModelProps,
        options: ProxyOptions,
        registration_options: RegistrationOptions | None,
    ) -> RegistrationId:
        """Register embeddings for given model as a proxy."""

        async def on_request(body: EmbeddingRequest, request: Request | None) -> StreamingResponse:
            return await post_json(body, options, request)

        return self.register_embeddings(model, props, SimpleEndpoint(on_request=on_request), registration_options)

    def unregister_embeddings(self, model: str, registration_id: RegistrationId) -> None:
        """Unregister embeddings for given model."""
        self.embeddings_endpoints.remove_model(model, registration_id)

    def register_audio_speech(
        self,
        model: str,
        props: ModelProps,
        endpoint: SimpleEndpoint[CreateSpeechRequest],
        registration_options: RegistrationOptions | None,
    ) -> RegistrationId:
        """Register audio speech endpoint for given model."""
        return self.audio_speech_endpoints.add_model(model, props, endpoint, "tts", registration_options)

    def register_audio_speech_as_proxy(
        self,
        model: str,
        props: ModelProps,
        options: ProxyOptions,
        registration_options: RegistrationOptions | None,
    ) -> RegistrationId:
        """Register audio speech for given model as a proxy."""

        async def on_request(body: CreateSpeechRequest, request: Request | None) -> StreamingResponse:
            return await post_json(body, options, request)

        return self.register_audio_speech(model, props, SimpleEndpoint(on_request=on_request), registration_options)

    def unregister_audio_speech(self, model: str, registration_id: RegistrationId) -> None:
        """Unregister audio speech for given model."""
        self.audio_speech_endpoints.remove_model(model, registration_id)

    def register_audio_transcriptions(
        self,
        model: str,
        props: ModelProps,
        endpoint: SimpleEndpoint[CreateTranscriptionRequest],
        registration_options: RegistrationOptions | None,
    ) -> RegistrationId:
        """Register audio transcriptions endpoint for given model."""
        return self.audio_transcriptions_endpoints.add_model(model, props, endpoint, "stt", registration_options)

    def register_audio_transcriptions_as_proxy(
        self,
        model: str,
        props: ModelProps,
        options: ProxyOptions,
        registration_options: RegistrationOptions | None,
    ) -> RegistrationId:
        """Register audio transcriptions for given model as a proxy."""

        async def on_request(body: CreateTranscriptionRequest, request: Request | None) -> StreamingResponse:
            return await post_form(body, options, request)

        return self.register_audio_transcriptions(model, props, SimpleEndpoint(on_request=on_request), registration_options)

    def unregister_audio_transcriptions(self, model: str, registration_id: RegistrationId) -> None:
        """Unregister audio transcriptions for given model."""
        self.audio_transcriptions_endpoints.remove_model(model, registration_id)

    def register_image_generations(
        self,
        model: str,
        props: ModelProps,
        endpoint: SimpleEndpoint[ImagesRequest],
        registration_options: RegistrationOptions | None,
    ) -> RegistrationId:
        """Register image generations endpoint for given model."""
        return self.images_generations_endpoints.add_model(model, props, endpoint, "txt2img", registration_options)

    def register_image_generations_as_proxy(
        self,
        model: str,
        props: ModelProps,
        options: ProxyOptions,
        registration_options: RegistrationOptions | None,
    ) -> RegistrationId:
        """Register image generations for given model as a proxy."""

        async def on_request(body: ImagesRequest, request: Request | None) -> StreamingResponse:
            return await post_json(body, options, request)

        return self.register_image_generations(model, props, SimpleEndpoint(on_request=on_request), registration_options)

    def unregister_image_generations(self, model: str, registration_id: RegistrationId) -> None:
        """Unregister image generations for given model."""
        self.images_generations_endpoints.remove_model(model, registration_id)

    def register_rerank(
        self,
        model: str,
        props: ModelProps,
        endpoint: SimpleEndpoint[RerankRequest],
        registration_options: RegistrationOptions | None,
    ) -> RegistrationId:
        """Register rerank endpoint for given model."""
        return self.rerank_endpoints.add_model(model, props, endpoint, "rerank", registration_options)

    def register_rerank_as_proxy(
        self,
        model: str,
        props: ModelProps,
        options: ProxyOptions,
        registration_options: RegistrationOptions | None,
    ) -> RegistrationId:
        """Register rerank for given model as a proxy."""

        async def on_request(body: RerankRequest, request: Request | None) -> StreamingResponse:
            return await post_json(body, options, request)

        return self.register_rerank(model, props, SimpleEndpoint(on_request=on_request), registration_options)

    def unregister_rerank(self, model: str, registration_id: RegistrationId) -> None:
        """Unregister rerank for given model."""
        self.rerank_endpoints.remove_model(model, registration_id)

    def register_custom_endpoint(
        self,
        url: str,
        props: ModelProps,
        endpoint: CustomEndpoint,
        registration_options: RegistrationOptions | None,
    ) -> RegistrationId:
        """Register custom endpoint."""
        return self.custom_endpoints.add_model(url, props, endpoint, "custom", registration_options)

    def register_custom_endpoint_as_proxy(
        self,
        url: str,
        props: ModelProps,
        options: ProxyOptions,
        registration_options: RegistrationOptions | None,
    ) -> RegistrationId:
        """Register custom endpoint as a proxy."""

        async def on_request(request: Request) -> StreamingResponse:
            headers = options.get_request_headers(request)
            headers["content-type"] = request.headers.get("content-type") or "application/octet-stream"
            _, _, sub_path = request.path_params["full_path"].partition("/")
            full_url = Utils.join_url(options.url, sub_path) if sub_path else options.url
            if request.url.query:
                full_url = f"{full_url}?{request.url.query}"
            return (
                await make_http_request(
                    url=full_url,
                    method=request.method,
                    data=request.stream(),
                    headers=headers,
                )
            ).as_streaming_response(options.allowed_response_headers)

        return self.register_custom_endpoint(url, props, CustomEndpoint(on_request=on_request), registration_options)

    def unregister_custom_endpoint(self, model: str, registration_id: RegistrationId) -> None:
        """Unregister custom endpoint."""
        self.custom_endpoints.remove_model(model, registration_id)

    def register_mcp_endpoint(
        self,
        url: str,
        props: ModelProps,
        endpoint: McpEndpoint,
        registration_options: RegistrationOptions | None,
    ) -> RegistrationId:
        """Register mcp endpoint."""
        return self.mcp_endpoints.add_model(url, props, endpoint, "mcp", registration_options)

    def register_mcp_endpoint_as_proxy(
        self,
        url: str,
        props: ModelProps,
        options: ProxyOptions,
        registration_options: RegistrationOptions | None,
    ) -> RegistrationId:
        """Register mcp endpoint as a proxy."""

        async def on_request(request: Request) -> StreamingResponse:
            headers = options.get_request_headers(request)
            headers["content-type"] = request.headers.get("content-type") or "application/octet-stream"
            full_url = options.url
            if request.url.query:
                full_url = f"{full_url}?{request.url.query}"
            logger.debug("MCP proxy: %s %s -> %s", request.method, request.path_params["full_path"], full_url)
            response = await make_http_request(url=full_url, method=request.method, data=request.stream(), headers=headers)
            logger.debug("MCP proxy response: %s", response.response.status)
            return response.as_streaming_response(options.allowed_response_headers)

        return self.register_mcp_endpoint(url, props, McpEndpoint(on_request=on_request), registration_options)

    def register_mcp_sse_endpoint_as_proxy(
        self,
        url: str,
        props: ModelProps,
        options: ProxyOptions,
        registration_options: RegistrationOptions | None,
    ) -> RegistrationId:
        """Register an SSE-transport MCP endpoint as a proxy.

        GET  → streams upstream SSE, rewriting the 'endpoint' event URL to route
               through this proxy so the client never speaks directly to upstream.
        POST → forwards the client message to the upstream messages URL stored in
               the session store that was populated during the GET phase.
        """
        session_store = McpSseSessionStore(
            ttl_seconds=self.config.mcp_sse_session_ttl_seconds,
            max_sessions=self.config.mcp_sse_max_sessions,
        )

        async def on_request(request: Request) -> StreamingResponse:
            headers = options.get_request_headers(request)

            if request.method == "GET":
                headers["accept"] = "text/event-stream"
                headers.pop("content-type", None)
                proxy_endpoint_url = str(request.url).split("?")[0]
                upstream_response = await make_http_request(url=options.url, method="GET", headers=headers)

                async def rewritten() -> AsyncGenerator[bytes]:
                    session_id: str | None = None
                    try:
                        async for chunk in _rewrite_sse_endpoint_events(
                            upstream_response.content,
                            options.url,
                            proxy_endpoint_url,
                            session_store,
                        ):
                            if session_id is None and b"sessionId=" in chunk:
                                session_id = chunk.decode(errors="replace").split("sessionId=")[-1].strip()
                            yield chunk
                    finally:
                        if session_id:
                            session_store.remove(session_id)
                        await upstream_response.response.release()

                return StreamingResponse(
                    rewritten(),
                    media_type="text/event-stream",
                    status_code=200,
                    headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
                )

            session_id = request.query_params.get("sessionId")
            if not session_id:
                raise HTTPException(400, "Missing sessionId query parameter")
            upstream_url = session_store.get(session_id)
            if upstream_url is None:
                raise HTTPException(404, f"Unknown session: {session_id}")
            headers["content-type"] = request.headers.get("content-type") or "application/json"
            response = await make_http_request(url=upstream_url, method="POST", data=request.stream(), headers=headers)
            return response.as_streaming_response(options.allowed_response_headers)

        return self.register_mcp_endpoint(url, props, McpEndpoint(on_request=on_request), registration_options)

    def unregister_mcp_endpoint(self, model: str, registration_id: RegistrationId) -> None:
        """Unregister mcp endpoint."""
        self.mcp_endpoints.remove_model(model, registration_id)

    def update_usage(self, usage: UsageChangeRequest) -> None:
        """Update model usage."""
        entry = self.registry.get(usage.id, None)
        if entry and entry.registered_model.usage != usage.usage:
            entry.registered_model.usage = usage.usage
            self._refresh_usage(entry.registered_model)

    def update_models(self, prev_list: list[Model], new_list: list[Model], api_url: str, api_key: str) -> None:
        """Update models, it remove old models and add new ones."""
        registered = set[RegistrationId]()
        changed = False
        for model in new_list:
            registered.add(model.id)
            if model.id not in self.registry:
                changed = True
                logger.info(f"Register new model origin={api_url} id={model.id} name={model.name} type={model.type}")  # noqa: G004
                self._register_proxy(
                    model_id=model.name,
                    type=model.type,
                    props=model.props,
                    url=api_url,
                    api_key=api_key,
                    registration_options=RegistrationOptions(
                        id=model.id,
                        origin=api_url,
                        usage=model.usage,
                        send_notification=False,
                    ),
                )
        for model in prev_list:
            if model.id not in registered:
                prev = self.registry.get(model.id, None)
                if prev:
                    changed = True
                    logger.info(f"Remove old model origin={api_url} id={model.id} name={model.name} type={model.type}")  # noqa: G004
                    prev.endpoint.remove_model(model_id=prev.model_id, registration_id=model.id, send_notification=False)
        if changed:
            self.parent_infra.send_models_list()

    def _register_proxy(
        self,
        model_id: ModelId,
        type: str,
        props: ModelProps,
        url: str,
        api_key: str,
        registration_options: RegistrationOptions | None,
    ) -> None:
        """Register proxy."""
        if type == "llm" or type.startswith("llm-"):
            headers = {"Authorization": f"Bearer {api_key}"}
            parts = set(ALL_LLM_SUFFIXES) if type == "llm" else set(type.split("-")[1:])

            def _proxy(path: str) -> ProxyOptions:
                return ProxyOptions(url=urljoin(url, path), headers=headers)

            self.register_chat_completion_as_proxy(
                model=model_id,
                props=props,
                completions=_proxy("v1/completions") if "v1" in parts else None,
                chat_completions=_proxy("v1/chat/completions") if "v2" in parts else None,
                responses=_proxy("v1/responses") if "v3" in parts else None,
                messages=_proxy("v1/messages") if "ant" in parts else None,
                ollama_chat=_proxy("api/chat") if "ollama" in parts else None,
                registration_options=registration_options,
            )
        elif type == "tts":
            self.register_audio_speech_as_proxy(
                model=model_id,
                props=props,
                options=ProxyOptions(
                    url=urljoin(url, "v1/audio/speech"),
                    headers={"Authorization": f"Bearer {api_key}"},
                ),
                registration_options=registration_options,
            )
        elif type == "stt":
            self.register_audio_transcriptions_as_proxy(
                model=model_id,
                props=props,
                options=ProxyOptions(
                    url=urljoin(url, "v1/audio/transcriptions"),
                    headers={"Authorization": f"Bearer {api_key}"},
                ),
                registration_options=registration_options,
            )
        elif type == "txt2img":
            self.register_image_generations_as_proxy(
                model=model_id,
                props=props,
                options=ProxyOptions(
                    url=urljoin(url, "v1/images/generations"),
                    headers={"Authorization": f"Bearer {api_key}"},
                ),
                registration_options=registration_options,
            )
        elif type == "embedding":
            self.register_embeddings_as_proxy(
                model=model_id,
                props=props,
                options=ProxyOptions(
                    url=urljoin(url, "v1/embeddings"),
                    headers={"Authorization": f"Bearer {api_key}"},
                ),
                registration_options=registration_options,
            )
        elif type == "rerank":
            self.register_rerank_as_proxy(
                model=model_id,
                props=props,
                options=ProxyOptions(
                    url=urljoin(url, "v1/rerank"),
                    headers={"Authorization": f"Bearer {api_key}"},
                ),
                registration_options=registration_options,
            )
        elif type == "custom":
            self.register_custom_endpoint_as_proxy(
                url=model_id,
                props=props,
                options=ProxyOptions(
                    url=urljoin(url, "custom"),
                    headers={"Authorization": f"Bearer {api_key}"},
                ),
                registration_options=registration_options,
            )
        elif type == "mcp":
            self.register_mcp_endpoint_as_proxy(
                url=model_id,
                props=props,
                options=ProxyOptions(
                    url=urljoin(url, "mcp"),
                    headers={"Authorization": f"Bearer {api_key}"},
                ),
                registration_options=registration_options,
            )
        else:
            logger.warning(f"Cannot register proxy with model_type={type}")  # noqa: G004

    async def execute_messages(
        self,
        body: MessagesRequest,
        request: Request | None = None,
        registration_id: RegistrationId | None = None,
    ) -> StarletteResponse:
        """Process messages request."""
        if self.config.is_log_payloads_enabled():
            logger.info(f"DUMP REQUEST PAYLOAD /v1/messages {body.model_dump_json(exclude_none=True)}")  # noqa: G004

        endpoint = self.chat_completion_endpoints.get_model(
            body.model,
            filter=lambda x: x.on_messages is not None,
            registration_id=registration_id,
        )
        on_messages = endpoint.endpoint.on_messages if endpoint else None
        if not endpoint or not on_messages:
            if not self.model_exists(body.model):
                raise HTTPException(404, "Model not found")
            msg = "Given model not support this endpoint.\n"
            if endpoint:
                supported_endpoints = []
                if endpoint.endpoint.on_responses:
                    supported_endpoints.append("/v1/responses")
                if endpoint.endpoint.on_chat_completion:
                    supported_endpoints.append("/v1/chat/completions")
                if endpoint.endpoint.on_completion:
                    supported_endpoints.append("/v1/completions")

                msg = msg + "Supported endpoints:\n" + ", ".join(supported_endpoints)

            raise HTTPException(400, msg)

        async def func() -> StarletteResponse:
            return await on_messages(body, request)

        return await self.with_usage(endpoint, func, self.config.is_log_payloads_enabled())

    async def execute_responses(
        self,
        body: ResponsesRequest,
        request: Request | None = None,
        registration_id: RegistrationId | None = None,
    ) -> StarletteResponse:
        """Process responses request."""
        if self.config.is_log_payloads_enabled():
            logger.info(f"DUMP REQUEST PAYLOAD /v1/responses {body.model_dump_json(exclude_none=True)}")  # noqa: G004

        endpoint = self.chat_completion_endpoints.get_model(
            body.model,
            filter=lambda x: x.on_responses is not None,
            registration_id=registration_id,
        )
        on_responses = endpoint.endpoint.on_responses if endpoint else None
        if not endpoint or not on_responses:
            if not self.model_exists(body.model):
                raise HTTPException(404, "Model not found")
            msg = "Given model not support this endpoint.\n"
            if endpoint:
                supported_endpoints = []
                if endpoint.endpoint.on_messages:
                    supported_endpoints.append("/v1/messages")
                if endpoint.endpoint.on_chat_completion:
                    supported_endpoints.append("/v1/chat/completions")
                if endpoint.endpoint.on_completion:
                    supported_endpoints.append("/v1/completions")

                msg = msg + "Supported endpoints:\n" + ", ".join(supported_endpoints)

            raise HTTPException(400, msg)

        async def func() -> StarletteResponse:
            return await on_responses(body, request)

        return await self.with_usage(endpoint, func, self.config.is_log_payloads_enabled())

    async def execute_chat_completion(
        self,
        body: ChatCompletionRequest,
        request: Request | None = None,
        registration_id: RegistrationId | None = None,
    ) -> StarletteResponse:
        """Process chat completion request."""
        if self.config.is_log_payloads_enabled():
            logger.info(f"DUMP REQUEST PAYLOAD /v1/chat/completions {body.model_dump_json(exclude_none=True)}")  # noqa: G004

        endpoint = self.chat_completion_endpoints.get_model(
            body.model,
            filter=lambda x: x.on_chat_completion is not None,
            registration_id=registration_id,
        )
        on_chat_completion = endpoint.endpoint.on_chat_completion if endpoint else None
        if not endpoint or not on_chat_completion:
            if not self.model_exists(body.model):
                raise HTTPException(404, "Model not found")
            msg = "Given model not support this endpoint.\n"
            if endpoint:
                supported_endpoints = []
                if endpoint.endpoint.on_messages:
                    supported_endpoints.append("/v1/messages")
                if endpoint.endpoint.on_responses:
                    supported_endpoints.append("/v1/responses")
                if endpoint.endpoint.on_completion:
                    supported_endpoints.append("/v1/completions")

                msg = msg + "Supported endpoints:\n" + ", ".join(supported_endpoints)

            raise HTTPException(400, msg)

        async def func() -> StarletteResponse:
            return await on_chat_completion(body, request)

        return await self.with_usage(endpoint, func, self.config.is_log_payloads_enabled())

    async def execute_completion(
        self,
        body: CompletionLegacyRequest,
        request: Request | None = None,
        registration_id: RegistrationId | None = None,
    ) -> StarletteResponse:
        """Process completion request."""
        endpoint = self.chat_completion_endpoints.get_model(
            body.model,
            filter=lambda x: x.on_completion is not None,
            registration_id=registration_id,
        )
        on_completion = endpoint.endpoint.on_completion if endpoint else None
        if not endpoint or not on_completion:
            if not self.model_exists(body.model):
                raise HTTPException(404, "Model not found")
            msg = "Given model not support this endpoint.\n"
            if endpoint:
                supported_endpoints = []
                if endpoint.endpoint.on_messages:
                    supported_endpoints.append("/v1/messages")
                if endpoint.endpoint.on_responses:
                    supported_endpoints.append("/v1/responses")
                if endpoint.endpoint.on_chat_completion:
                    supported_endpoints.append("/v1/chat/completions")

                msg = msg + "Supported endpoints:\n" + ", ".join(supported_endpoints)

            raise HTTPException(400, msg)

        async def func() -> StarletteResponse:
            return await on_completion(body, request)

        return await self.with_usage(endpoint, func)

    async def execute_ollama_chat(
        self,
        body: OllamaChatRequest,
        request: Request | None = None,
        registration_id: RegistrationId | None = None,
    ) -> StarletteResponse:
        """Process ollama chat."""
        endpoint = self.chat_completion_endpoints.get_model(
            body.model,
            filter=lambda x: x.on_ollama_chat is not None,
            registration_id=registration_id,
        )
        on_ollama_chat = endpoint.endpoint.on_ollama_chat if endpoint else None
        if not endpoint or not on_ollama_chat:
            if not self.model_exists(body.model):
                raise HTTPException(404, "Model not found")
            msg = "Given model not support this endpoint.\n"
            if endpoint:
                supported_endpoints = []
                if endpoint.endpoint.on_messages:
                    supported_endpoints.append("/v1/messages")
                if endpoint.endpoint.on_responses:
                    supported_endpoints.append("/v1/responses")
                if endpoint.endpoint.on_chat_completion:
                    supported_endpoints.append("/v1/chat/completions")
                if endpoint.endpoint.on_completion:
                    supported_endpoints.append("/v1/completions")

                msg = msg + "Supported endpoints:\n" + ", ".join(supported_endpoints)

            raise HTTPException(400, msg)

        async def func() -> StarletteResponse:
            return await on_ollama_chat(body, request)

        return await self.with_usage(endpoint, func)

    async def execute_embeddings(
        self,
        body: EmbeddingRequest,
        request: Request | None = None,
        registration_id: RegistrationId | None = None,
    ) -> StarletteResponse:
        """Process embeddings request."""
        endpoint = self.embeddings_endpoints.get_model(body.model, registration_id=registration_id)
        if not endpoint:
            if not self.model_exists(body.model):
                raise HTTPException(404, "Model not found")
            raise HTTPException(400, "Given model is not supported")

        async def func() -> StarletteResponse:
            return await endpoint.endpoint.on_request(body, request)

        return await self.with_usage(endpoint, func)

    async def execute_images_generations(
        self,
        body: ImagesRequest,
        request: Request | None = None,
        registration_id: RegistrationId | None = None,
    ) -> StarletteResponse:
        """Process images generations request."""
        if self.config.is_log_payloads_enabled():
            logger.info(f"DUMP REQUEST PAYLOAD /v1/images/generations {body.model_dump_json(exclude_none=True)}")  # noqa: G004
        endpoint = self.images_generations_endpoints.get_model(body.model, registration_id=registration_id)
        if not endpoint:
            if not self.model_exists(body.model):
                raise HTTPException(404, "Model not found")
            raise HTTPException(400, "Given model is not supported")

        async def func() -> StarletteResponse:
            return await endpoint.endpoint.on_request(body, request)

        return await self.with_usage(endpoint, func)

    async def execute_audio_speech(
        self,
        body: CreateSpeechRequest,
        request: Request | None = None,
        registration_id: RegistrationId | None = None,
    ) -> StarletteResponse:
        """Process audio speech request."""
        if self.config.is_log_payloads_enabled():
            logger.info(f"DUMP REQUEST PAYLOAD /v1/audio/speech {body.model_dump_json(exclude_none=True)}")  # noqa: G004
        endpoint = self.audio_speech_endpoints.get_model(body.model, registration_id=registration_id)
        if not endpoint:
            if not self.model_exists(body.model):
                raise HTTPException(404, "Model not found")
            raise HTTPException(400, "Given model is not supported")

        async def func() -> StarletteResponse:
            return await endpoint.endpoint.on_request(body, request)

        return await self.with_usage(endpoint, func)

    async def execute_audio_transcriptions(
        self,
        body: CreateTranscriptionRequest,
        request: Request | None = None,
        registration_id: RegistrationId | None = None,
    ) -> StarletteResponse:
        """Process audio transcriptions request."""
        endpoint = self.audio_transcriptions_endpoints.get_model(body.model, registration_id=registration_id)
        if not endpoint:
            if not self.model_exists(body.model):
                raise HTTPException(404, "Model not found")
            raise HTTPException(400, "Given model is not supported")

        async def func() -> StarletteResponse:
            return await endpoint.endpoint.on_request(body, request)

        return await self.with_usage(endpoint, func)

    async def execute_rerank(
        self,
        body: RerankRequest,
        request: Request | None = None,
        registration_id: RegistrationId | None = None,
    ) -> StarletteResponse:
        """Process rerank request."""
        if self.config.is_log_payloads_enabled():
            logger.info(f"DUMP REQUEST PAYLOAD /v1/rerank {body.model_dump_json(exclude_none=True)}")  # noqa: G004
        endpoint = self.rerank_endpoints.get_model(body.model, registration_id=registration_id)
        if not endpoint:
            if not self.model_exists(body.model):
                raise HTTPException(404, "Model not found")
            raise HTTPException(400, "Given model is not supported")

        async def func() -> StarletteResponse:
            return await endpoint.endpoint.on_request(body, request)

        return await self.with_usage(endpoint, func)

    async def execute_custom_endpoints(self, url: str, request: Request) -> StarletteResponse:
        """Process custom endpoint request."""
        endpoint = self.custom_endpoints.get_model(url.split("/")[0])
        if not endpoint:
            raise HTTPException(400, "Given url is not supported")

        async def func() -> StarletteResponse:
            return await endpoint.endpoint.on_request(request)

        return await self.with_usage(endpoint, func)

    async def execute_mcp_endpoints(self, url: str, request: Request) -> StarletteResponse:
        """Process mcp endpoint request."""
        endpoint = self.mcp_endpoints.get_model(url.split("/")[0])
        if not endpoint:
            raise HTTPException(400, "Given url is not supported")

        async def func() -> StarletteResponse:
            return await endpoint.endpoint.on_request(request)

        return await self.with_usage(endpoint, func)

    async def with_usage(self, model: RegisteredModel[Any], func: Callable[[], Awaitable[T]], log_payload: bool = False) -> T:
        """With usage."""
        if model.origin != "local":
            return await func()
        self._add_usage(model)
        self.metrics_registry.requests_in_flight.labels(model_type=model.type).inc()
        start_time = time.perf_counter()

        try:
            resp = await func()
        except Exception as e:
            duration = time.perf_counter() - start_time
            self.metrics_registry.request_duration.labels(model_name=model.name, model_type=model.type).observe(duration)
            self.metrics_registry.request_total.labels(model_name=model.name, model_type=model.type, status="error").inc()
            self.metrics_registry.request_errors.labels(model_name=model.name, error_type=_classify_error(e)).inc()
            self.metrics_registry.requests_in_flight.labels(model_type=model.type).dec()
            self._remove_usage(model)
            raise

        if isinstance(resp, StreamingResponse):

            async def create_generator() -> AsyncGenerator[Any]:
                """Add usage for response."""
                chunks = list[bytes]()
                try:
                    async for chunk in resp.body_iterator:
                        if log_payload and isinstance(chunk, bytes):
                            chunks.append(chunk)
                        yield chunk
                finally:
                    if log_payload:
                        logger.info(f"DUMP RESPONSE PAYLOAD {b''.join(chunks).decode('utf-8')}")  # noqa: G004
                    duration = time.perf_counter() - start_time
                    self.metrics_registry.request_duration.labels(model_name=model.name, model_type=model.type).observe(duration)
                    self.metrics_registry.request_total.labels(model_name=model.name, model_type=model.type, status="success").inc()
                    self.metrics_registry.requests_in_flight.labels(model_type=model.type).dec()
                    self._remove_usage(model)

            return StreamingResponse(create_generator(), media_type=resp.media_type, status_code=resp.status_code, headers=resp.headers)  # pyright: ignore[reportReturnType]
        duration = time.perf_counter() - start_time
        self.metrics_registry.request_duration.labels(model_name=model.name, model_type=model.type).observe(duration)
        self.metrics_registry.request_total.labels(model_name=model.name, model_type=model.type, status="success").inc()
        self.metrics_registry.requests_in_flight.labels(model_type=model.type).dec()
        self._remove_usage(model)
        return resp

    def _add_usage(self, model: RegisteredModel[Any]) -> None:
        model.usage += 1
        self._refresh_usage(model)

    def _remove_usage(self, model: RegisteredModel[Any]) -> None:
        model.usage -= 1
        self._refresh_usage(model)

    def _refresh_usage(self, model: RegisteredModel[Any]) -> None:
        logger.debug(f"Model usage origin={model.origin} id={model.id} name={model.name} usage={model.usage}")  # noqa: G004
        self.parent_infra.send_usage(UsageChangeRequest(id=model.id, usage=model.usage))

    async def test_model(self, registration_id: RegistrationId) -> JsonSerializable:
        """Test model of given id."""
        entry = self.registry.get(registration_id, None)
        if not entry:
            raise HTTPException(400, "Model not installed")
        return await self.model_tester.test_model(entry)


def _classify_error(e: Exception) -> str:
    """Classify an exception into an error type for metrics."""
    if isinstance(e, asyncio.TimeoutError):
        return "timeout"
    if isinstance(e, aiohttp.ClientError):
        return "connection_error"
    if isinstance(e, HttpClientError):
        return "http_error"
    return "model_error"


async def post_json(data: BaseModel, options: ProxyOptions, request: Request | None = None) -> StreamingResponse:
    """Make HTTP POST request sending data as JSON."""
    raw = data.model_dump(exclude_none=True)
    if options.remove_model:
        del raw["model"]
    if options.rewrite_model_to:
        raw["model"] = options.rewrite_model_to
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
