# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""API endpoints for openai compatibile endpoints."""

import logging
from typing import Annotated

from fastapi import APIRouter, Body, Depends, Form, Path, Request

from server.core.dependencies import auth_server, get_endpoint_registry
from server.endpointregistry import EndpointRegistry
from server.models.api import (
    ApiModel,
    ApiModels,
    ChatCompletionRequest,
    CompletionLegacyRequest,
    CreateSpeechRequest,
    CreateTranscriptionRequest,
    EmbeddingRequest,
    ImagesRequest,
    MessagesRequest,
    MessagesResponse,
    ResponsesRequest,
    ResponsesResponse,
)
from server.models.common import StarletteResponse

logger = logging.getLogger("uvicorn.error")


router = APIRouter(prefix="", tags=["AI Endpoints"])


@router.get("/v1/models")
async def on_models(
    _: Annotated[str, Depends(auth_server)],
    endpoint_registry: Annotated[EndpointRegistry, Depends(get_endpoint_registry)],
) -> ApiModels:
    """Process models request."""
    return endpoint_registry.get_models()


@router.get("/v1/models/{model_id:path}")
async def on_model(
    _: Annotated[str, Depends(auth_server)],
    model_id: Annotated[str, Path(description="The ID of the model to use.")],
    endpoint_registry: Annotated[EndpointRegistry, Depends(get_endpoint_registry)],
) -> ApiModel:
    """Process model request."""
    return endpoint_registry.get_model(model_id)


@router.post("/v1/messages")
async def on_messages(
    request: Request,
    body: Annotated[MessagesRequest, Body(...)],
    _: Annotated[str, Depends(auth_server)],
    endpoint_registry: Annotated[EndpointRegistry, Depends(get_endpoint_registry)],
) -> MessagesResponse | StarletteResponse:
    """Process messages request."""
    return await endpoint_registry.execute_messages(body, request)


@router.post("/v1/responses")
async def on_responses(
    request: Request,
    body: Annotated[ResponsesRequest, Body(...)],
    _: Annotated[str, Depends(auth_server)],
    endpoint_registry: Annotated[EndpointRegistry, Depends(get_endpoint_registry)],
) -> ResponsesResponse | StarletteResponse:
    """Process responses request."""
    return await endpoint_registry.execute_responses(body, request)


@router.post("/v1/chat/completions")
async def on_chat_completions(
    request: Request,
    body: Annotated[ChatCompletionRequest, Body(...)],
    _: Annotated[str, Depends(auth_server)],
    endpoint_registry: Annotated[EndpointRegistry, Depends(get_endpoint_registry)],
) -> StarletteResponse:
    """Process chat completions request."""
    return await endpoint_registry.execute_chat_completion(body, request)


@router.post("/v1/completions")
async def on_completions(
    request: Request,
    body: Annotated[CompletionLegacyRequest, Body(...)],
    _: Annotated[str, Depends(auth_server)],
    endpoint_registry: Annotated[EndpointRegistry, Depends(get_endpoint_registry)],
) -> StarletteResponse:
    """Process completions request."""
    return await endpoint_registry.execute_completion(body, request)


@router.post("/v1/embeddings")
async def on_embeddings(
    request: Request,
    body: Annotated[EmbeddingRequest, Body(...)],
    _: Annotated[str, Depends(auth_server)],
    endpoint_registry: Annotated[EndpointRegistry, Depends(get_endpoint_registry)],
) -> StarletteResponse:
    """Process embeddings request."""
    return await endpoint_registry.execute_embeddings(body, request)


@router.post("/v1/audio/speech")
async def on_audio_speech(
    request: Request,
    body: Annotated[CreateSpeechRequest, Body(...)],
    _: Annotated[str, Depends(auth_server)],
    endpoint_registry: Annotated[EndpointRegistry, Depends(get_endpoint_registry)],
) -> StarletteResponse:
    """Process audio speech request."""
    return await endpoint_registry.execute_audio_speech(body, request)


@router.post("/v1/audio/transcriptions")
async def on_audio_transcriptions(
    request: Request,
    body: Annotated[CreateTranscriptionRequest, Form(media_type="multipart/form-data")],
    _: Annotated[str, Depends(auth_server)],
    endpoint_registry: Annotated[EndpointRegistry, Depends(get_endpoint_registry)],
) -> StarletteResponse:
    """Process audio translation request."""
    return await endpoint_registry.execute_audio_transcriptions(body, request)


@router.post("/v1/images/generations")
async def on_images_generations(
    request: Request,
    body: Annotated[ImagesRequest, Body(...)],
    _: Annotated[str, Depends(auth_server)],
    endpoint_registry: Annotated[EndpointRegistry, Depends(get_endpoint_registry)],
) -> StarletteResponse:
    """Process images genenerations request."""
    return await endpoint_registry.execute_images_generations(body, request)


# Replacement from router.api_route which is kind bugged in swagger.
# Also FastAPI creators doesn't recommend using api_route method.
@router.get("/custom/{full_path:path}")
@router.post("/custom/{full_path:path}")
@router.put("/custom/{full_path:path}")
@router.delete("/custom/{full_path:path}")
@router.patch("/custom/{full_path:path}")
@router.head("/custom/{full_path:path}")
@router.options("/custom/{full_path:path}")
async def on_custom_endpoint(
    request: Request,
    _: Annotated[str, Depends(auth_server)],
    endpoint_registry: Annotated[EndpointRegistry, Depends(get_endpoint_registry)],
    full_path: str,
) -> StarletteResponse:
    """Process custom endpoint request."""
    return await endpoint_registry.execute_custom_endpoints(full_path, request)
