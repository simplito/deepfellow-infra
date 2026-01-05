# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Endpoint registry holds callbacks for given endpoints and models."""

import logging
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import TYPE_CHECKING, Any, NamedTuple, TypeVar
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
    Model,
    ModelId,
    ModelProps,
    ModelType,
)
from server.models.common import JsonSerializable, StarletteResponse
from server.utils.core import Utils
from server.websockets.models import RegistrationId, UsageChangeRequest
from server.websockets.parent_infra import ParentInfra

if TYPE_CHECKING:
    from server.model_tester import ModelTester

logger = logging.getLogger("uvicorn.error")
T = TypeVar("T")

type EndpointCallback[T] = Callable[[T, Request | None], Awaitable[StarletteResponse]]
type CustomEndpointCallback = Callable[[Request], Awaitable[StarletteResponse]]


class SimpleEndpoint[T](NamedTuple):
    on_request: EndpointCallback[T]


class CustomEndpoint(NamedTuple):
    on_request: CustomEndpointCallback


class ChatCompletionEndpoint:
    on_chat_completion: EndpointCallback[ChatCompletionRequest] | None = None
    on_completion: EndpointCallback[CompletionLegacyRequest] | None = None


class RegisteredModel[T](BaseModel):
    id: RegistrationId
    name: ModelId
    origin: str
    props: ModelProps
    type: ModelType
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
        parent_infra: ParentInfra,
    ):
        self.registry = registry
        self.parent_infra = parent_infra
        self.models = dict[ModelId, dict[RegistrationId, RegisteredModel[T]]]()

    def add_model(
        self, model_id: ModelId, props: ModelProps, endpoint: T, type: ModelType, options: RegistrationOptions | None
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

    def get_models(self) -> list[ApiModel]:
        """List models from registry."""
        res = list[ApiModel]()
        for model_id, map in self.models.items():
            private = False
            for item in map.values():
                if item.props.private:
                    private = True
            res.append(ApiModel(id=model_id, object="model", created=0, owned_by="unknown", props=ModelProps(private=private)))
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
                        type=item.type,
                        props=item.props,
                        usage=item.usage,
                    )
                )
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
    def __init__(
        self,
        config: AppSettings,
        parent_infra: ParentInfra,
        model_tester: "ModelTester",
    ):
        self.config = config
        self.parent_infra = parent_infra
        self.parent_infra.endpoint_registry = self
        self.model_tester = model_tester
        self.registry = dict[RegistrationId, RegistryEntry]()
        self.chat_completion_endpoints = Endpoint[ChatCompletionEndpoint](self.registry, self.parent_infra)
        self.embeddings_endpoints = Endpoint[SimpleEndpoint[EmbeddingRequest]](self.registry, self.parent_infra)
        self.audio_speech_endpoints = Endpoint[SimpleEndpoint[CreateSpeechRequest]](self.registry, self.parent_infra)
        self.audio_transcriptions_endpoints = Endpoint[SimpleEndpoint[CreateTranscriptionRequest]](self.registry, self.parent_infra)
        self.custom_endpoints = Endpoint[CustomEndpoint](self.registry, self.parent_infra)
        self.images_generations_endpoints = Endpoint[SimpleEndpoint[ImagesRequest]](self.registry, self.parent_infra)

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
        models.extend(self.custom_endpoints.list_models())
        return models

    def register_chat_completion(
        self,
        model: str,
        props: ModelProps,
        endpoint: ChatCompletionEndpoint,
        registration_options: RegistrationOptions | None,
    ) -> RegistrationId:
        """Register chat completion endpoint for given model."""
        model_type: ModelType = "llm"
        if not endpoint.on_chat_completion and endpoint.on_completion:
            model_type = "llm-only-v1"
        if endpoint.on_chat_completion and not endpoint.on_completion:
            model_type = "llm-only-v2"
        return self.chat_completion_endpoints.add_model(model, props, endpoint, model_type, registration_options)

    def register_chat_completion_as_proxy(
        self,
        model: str,
        props: ModelProps,
        chat_completions: ProxyOptions | None,
        completions: ProxyOptions | None,
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
            full_url = Utils.join_url(options.url, request.path_params["full_path"].split("/", 1)[-1])
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
        type: ModelType,
        props: ModelProps,
        url: str,
        api_key: str,
        registration_options: RegistrationOptions | None,
    ) -> None:
        """Register proxy."""
        if type == "llm":
            self.register_chat_completion_as_proxy(
                model=model_id,
                props=props,
                chat_completions=ProxyOptions(
                    url=urljoin(url, "v1/chat/completions"),
                    headers={"Authorization": f"Bearer {api_key}"},
                ),
                completions=ProxyOptions(
                    url=urljoin(url, "v1/completions"),
                    headers={"Authorization": f"Bearer {api_key}"},
                ),
                registration_options=registration_options,
            )
        elif type == "llm-only-v1":
            self.register_chat_completion_as_proxy(
                model=model_id,
                props=props,
                chat_completions=None,
                completions=ProxyOptions(
                    url=urljoin(url, "v1/completions"),
                    headers={"Authorization": f"Bearer {api_key}"},
                ),
                registration_options=registration_options,
            )
        elif type == "llm-only-v2":
            self.register_chat_completion_as_proxy(
                model=model_id,
                props=props,
                chat_completions=ProxyOptions(
                    url=urljoin(url, "v1/chat/completions"),
                    headers={"Authorization": f"Bearer {api_key}"},
                ),
                completions=None,
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
        else:
            logger.warning(f"Cannot register proxy with model_type={type}")  # noqa: G004

    def has_chat_completion_model(self, model: str) -> bool:
        """Check whether the chat completion model is registered."""
        endpoint = self.chat_completion_endpoints.get_model(model, lambda x: x.on_chat_completion is not None)
        return endpoint is not None and endpoint.endpoint.on_chat_completion is not None

    def has_completion_model(self, model: str) -> bool:
        """Check whether the completion model is registered."""
        endpoint = self.chat_completion_endpoints.get_model(model, lambda x: x.on_completion is not None)
        return endpoint is not None and endpoint.endpoint.on_completion is not None

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
            msg = (
                "Given model is only supported in legacy /v1/completions"
                if endpoint and endpoint.endpoint.on_completion
                else "Given model is not supported"
            )
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
            msg = (
                "Given model is only supported in /v1/chat/completions"
                if endpoint and endpoint.endpoint.on_chat_completion
                else "Given model is not supported"
            )
            raise HTTPException(400, msg)

        async def func() -> StarletteResponse:
            return await on_completion(body, request)

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

    async def with_usage(self, model: RegisteredModel[Any], func: Callable[[], Awaitable[T]], log_payload: bool = False) -> T:
        """With usage."""
        if model.origin != "local":
            return await func()
        self._add_usage(model)
        try:
            resp = await func()
        except Exception:
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
                    self._remove_usage(model)

            return StreamingResponse(create_generator(), media_type=resp.media_type, status_code=resp.status_code, headers=resp.headers)  # pyright: ignore[reportReturnType]

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
    # NOTE: Uncomment to debug http request
    # logger.info(f"Making HTTP request to: {url}")
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
