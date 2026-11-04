# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""API endpoints for openai compatibile endpoints."""

from typing import Annotated

from fastapi import APIRouter, Body, Depends, Form, Path, Query, Request, Response

from server.config import AppSettings
from server.core.dependencies import auth_server, get_config, get_endpoint_registry
from server.endpointregistry import EndpointRegistry
from server.models.api import (
    ApiModel,
    ApiModelCompatible,
    ApiModels,
    ApiModelsCompatible,
    ChatCompletionRequest,
    CompletionLegacyRequest,
    CreateSpeechRequest,
    CreateTranscriptionRequest,
    EmbeddingRequest,
    ImagesRequest,
    MessagesRequest,
    MessagesResponse,
    OllamaChatRequest,
    RerankRequest,
    ResponsesRequest,
    ResponsesResponse,
    SystemOneRequest,
    SystemOneResponse,
)
from server.models.common import StarletteResponse
from server.utils.request_limits import apply_default_generation_limits, enforce_generation_limits
from server.utils.validators import SafePath

router = APIRouter(prefix="", tags=["AI Endpoints"])


@router.get("/v1/models")
async def on_models(
    _: Annotated[str, Depends(auth_server)],
    endpoint_registry: Annotated[EndpointRegistry, Depends(get_endpoint_registry)],
    additional_data: Annotated[bool, Query()] = False,
) -> ApiModels | ApiModelsCompatible:
    """Process models request.

    Set `additional_data=true` to also get `props` (context window, type, endpoints, and in
    `alias_of` the models an alias points to) plus `usage` (in-flight requests; a snapshot, not
    refreshed, for models received from a parent node) and `capacity` per model — omitted by
    default for strict OpenAI-compatibility, so plain OpenAI SDK clients never see it.
    """
    models = endpoint_registry.get_models()
    return models if additional_data else ApiModelsCompatible(**models.model_dump())


@router.get(
    "/v1/models/{model_id:path}/tokenizer",
    response_class=Response,
    responses={
        200: {
            "content": {"application/zip": {}},
            "headers": {
                "X-Max-Tokens": {"description": "Most tokens the model embeds, the smallest over its backends; absent when unknown"}
            },
        },
        400: {"description": "The model is not an embedding model"},
        404: {"description": "`model_not_found`, or `tokenizer_not_available` when infra has no tokenizer for the model"},
        503: {"description": "`tokenizer_fetch_failed` when a backend that may have one failed, or the model is still loading; retry"},
    },
)
async def on_model_tokenizer(
    _: Annotated[str, Depends(auth_server)],
    model_id: Annotated[str, Path(description="The ID of an embedding model.")],
    endpoint_registry: Annotated[EndpointRegistry, Depends(get_endpoint_registry)],
) -> Response:
    """Return an installed embedding model's tokenizer as a zip (`tokenizer.json`, `tokenizer_config.json`) for doc-chunker."""
    archive, max_tokens = await endpoint_registry.get_embeddings_tokenizer(model_id)
    headers = {"Content-Disposition": 'attachment; filename="tokenizer.zip"'}
    if max_tokens is not None:
        headers["X-Max-Tokens"] = str(max_tokens)
    return Response(content=archive, media_type="application/zip", headers=headers)


@router.get("/v1/models/{model_id:path}")
async def on_model(
    _: Annotated[str, Depends(auth_server)],
    model_id: Annotated[str, Path(description="The ID of the model to use.")],
    endpoint_registry: Annotated[EndpointRegistry, Depends(get_endpoint_registry)],
    additional_data: Annotated[bool, Query()] = False,
) -> ApiModel | ApiModelCompatible:
    """Process model request.

    Set `additional_data=true` to also get `props` (context window, type, endpoints, and in
    `alias_of` the models an alias points to) plus `usage` (in-flight requests; a snapshot, not
    refreshed, for models received from a parent node) and `capacity` — omitted by default for
    strict OpenAI-compatibility, so plain OpenAI SDK clients never see it.
    """
    model = endpoint_registry.get_model(model_id)
    return model if additional_data else ApiModelCompatible(**model.model_dump())


@router.post("/v1/messages")
async def on_messages(
    request: Request,
    body: Annotated[MessagesRequest, Body(...)],
    _: Annotated[str, Depends(auth_server)],
    endpoint_registry: Annotated[EndpointRegistry, Depends(get_endpoint_registry)],
    config: Annotated[AppSettings, Depends(get_config)],
) -> MessagesResponse | StarletteResponse:
    """Process messages request."""
    enforce_generation_limits(body, config)
    return await endpoint_registry.execute_messages(body, request)


@router.post("/v1/responses")
async def on_responses(
    request: Request,
    body: Annotated[ResponsesRequest, Body(...)],
    _: Annotated[str, Depends(auth_server)],
    endpoint_registry: Annotated[EndpointRegistry, Depends(get_endpoint_registry)],
    config: Annotated[AppSettings, Depends(get_config)],
) -> ResponsesResponse | StarletteResponse:
    """Process responses request."""
    enforce_generation_limits(body, config)
    return await endpoint_registry.execute_responses(body, request)


@router.post("/v1/chat/completions")
async def on_chat_completions(
    request: Request,
    body: Annotated[ChatCompletionRequest, Body(...)],
    _: Annotated[str, Depends(auth_server)],
    endpoint_registry: Annotated[EndpointRegistry, Depends(get_endpoint_registry)],
    config: Annotated[AppSettings, Depends(get_config)],
) -> StarletteResponse:
    """Process chat completions request."""
    enforce_generation_limits(body, config)
    return await endpoint_registry.execute_chat_completion(body, request)


@router.post("/v1/completions")
async def on_completions(
    request: Request,
    body: Annotated[CompletionLegacyRequest, Body(...)],
    _: Annotated[str, Depends(auth_server)],
    endpoint_registry: Annotated[EndpointRegistry, Depends(get_endpoint_registry)],
    config: Annotated[AppSettings, Depends(get_config)],
) -> StarletteResponse:
    """Process completions request."""
    enforce_generation_limits(body, config)
    return await endpoint_registry.execute_completion(body, request)


@router.post("/api/chat")
async def on_ollama_chat(
    request: Request,
    body: Annotated[OllamaChatRequest, Body(...)],
    _: Annotated[str, Depends(auth_server)],
    endpoint_registry: Annotated[EndpointRegistry, Depends(get_endpoint_registry)],
    config: Annotated[AppSettings, Depends(get_config)],
) -> StarletteResponse:
    """Process completions request."""
    enforce_generation_limits(body, config)
    apply_default_generation_limits(body, config)
    return await endpoint_registry.execute_ollama_chat(body, request)


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


@router.post("/v1/rerank", response_model_exclude_unset=True)
async def on_rerank(
    request: Request,
    body: Annotated[RerankRequest, Body(...)],
    _: Annotated[str, Depends(auth_server)],
    endpoint_registry: Annotated[EndpointRegistry, Depends(get_endpoint_registry)],
) -> StarletteResponse:
    """Process rerank request."""
    return await endpoint_registry.execute_rerank(body, request)


@router.post("/v1/systemone")
async def on_systemone(
    request: Request,
    body: Annotated[SystemOneRequest, Body(...)],
    _: Annotated[str, Depends(auth_server)],
    endpoint_registry: Annotated[EndpointRegistry, Depends(get_endpoint_registry)],
) -> SystemOneResponse | StarletteResponse:
    """Answer typed choice, yes/no (noul) and score questions with a decision model."""
    return await endpoint_registry.execute_systemone(body, request)


@router.get("/custom")
async def get_custom_prefixes(
    request: Request,  # noqa: ARG001
    _: Annotated[str, Depends(auth_server)],
    endpoint_registry: Annotated[EndpointRegistry, Depends(get_endpoint_registry)],
) -> list[str]:
    """Process custom endpoint request."""
    return [model.props.prefix for model in endpoint_registry.custom_endpoints.list_models() if model.props.prefix]


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
    full_path: SafePath,
) -> StarletteResponse:
    """Process custom endpoint request."""
    return await endpoint_registry.execute_custom_endpoints(full_path, request)


@router.get("/mcp")
async def get_mcp_prefixes(
    request: Request,  # noqa: ARG001
    _: Annotated[str, Depends(auth_server)],
    endpoint_registry: Annotated[EndpointRegistry, Depends(get_endpoint_registry)],
) -> list[str]:
    """Process custom endpoint request."""
    return [model.props.prefix for model in endpoint_registry.mcp_endpoints.list_models() if model.props.prefix]


# Replacement from router.api_route which is kind bugged in swagger.
# Also FastAPI creators doesn't recommend using api_route method.
@router.get("/mcp/{full_path:path}")
@router.post("/mcp/{full_path:path}")
@router.put("/mcp/{full_path:path}")
@router.delete("/mcp/{full_path:path}")
@router.patch("/mcp/{full_path:path}")
@router.head("/mcp/{full_path:path}")
@router.options("/mcp/{full_path:path}")
async def on_mcp_endpoint(
    request: Request,
    _: Annotated[str, Depends(auth_server)],
    endpoint_registry: Annotated[EndpointRegistry, Depends(get_endpoint_registry)],
    full_path: SafePath,
) -> StarletteResponse:
    """Process mcp endpoint request."""
    return await endpoint_registry.execute_mcp_endpoints(full_path, request)
