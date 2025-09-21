"""API endpoints for openai compatibile endpoints."""

import logging
from typing import Annotated

from fastapi import APIRouter, Body, Depends, Form, HTTPException, Path, Request
from fastapi.responses import JSONResponse

from server.core.dependencies import auth_server, get_endpoint_registry
from server.endpointregistry import EndpointRegistry, ProxyOptions, post_form, post_json
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
from server.models.common import StarletteResponse
from server.websockets.utils import auth_header, get_lazy_infra, get_proxy_url

logger = logging.getLogger("uvicorn.error")


router = APIRouter(prefix="", tags=["AI Endpoints"])


@router.get("/v1/models")
async def on_models(
    _: Annotated[str, Depends(auth_server)],
    endpoint_registry: Annotated[EndpointRegistry, Depends(get_endpoint_registry)],
) -> ChatCompletionModels:
    """Process models request."""
    return endpoint_registry.get_chat_completions_models()


@router.get("/v1/models/{model_id}")
async def on_model(
    _: Annotated[str, Depends(auth_server)],
    model_id: Annotated[str, Path(..., description="The ID of the model to use.")],
    endpoint_registry: Annotated[EndpointRegistry, Depends(get_endpoint_registry)],
) -> ChatCompletionModel:
    """Process model request."""
    return endpoint_registry.get_chat_completions_model(model_id)


@router.post("/v1/chat/completions")
async def on_chat_completions(
    request: Request,
    body: Annotated[ChatCompletionRequest, Body(...)],
    _: Annotated[str, Depends(auth_server)],
    endpoint_registry: Annotated[EndpointRegistry, Depends(get_endpoint_registry)],
) -> StarletteResponse:
    """Process chat completions request."""
    model = body.model
    url, key, is_inside = get_lazy_infra(request, model)
    if not url:
        raise HTTPException(400, "Given model is not found")

    if not is_inside:
        return await post_json(body, ProxyOptions(url=get_proxy_url(url, request), headers=auth_header(key)), request)

    if not endpoint_registry.has_chat_completion_model(model):
        raise HTTPException(400, "Given model is not found")
    return await endpoint_registry.execute_chat_completion(request, model, body)


@router.post("/v1/completions")
async def on_completions(
    request: Request,
    body: Annotated[CompletionLegacyRequest, Body(...)],
    _: Annotated[str, Depends(auth_server)],
    endpoint_registry: Annotated[EndpointRegistry, Depends(get_endpoint_registry)],
) -> StarletteResponse:
    """Process completions request."""
    model = body.model
    url, key, is_inside = get_lazy_infra(request, model)
    if not url:
        raise HTTPException(400, "Given model is not found")

    if not is_inside:
        return await post_json(body, ProxyOptions(url=get_proxy_url(url, request), headers=auth_header(key)), request)

    return await endpoint_registry.execute_completion(request, model, body)


@router.post("/v1/embeddings")
async def on_embeddings(
    request: Request,
    body: Annotated[EmbeddingRequest, Body(...)],
    _: Annotated[str, Depends(auth_server)],
    endpoint_registry: Annotated[EndpointRegistry, Depends(get_endpoint_registry)],
) -> StarletteResponse:
    """Process embeddings request."""
    model = body.model
    url, key, is_inside = get_lazy_infra(request, model)
    if not url:
        raise HTTPException(400, "Given model is not found")

    if not is_inside:
        return await post_json(body, ProxyOptions(url=get_proxy_url(url, request), headers=auth_header(key)), request)

    return await endpoint_registry.execute_embeddings(request, model, body)


@router.post("/v1/audio/speech")
async def on_audio_speech(
    request: Request,
    body: Annotated[CreateSpeechRequest, Body(...)],
    _: Annotated[str, Depends(auth_server)],
    endpoint_registry: Annotated[EndpointRegistry, Depends(get_endpoint_registry)],
) -> StarletteResponse:
    """Process audio speech request."""
    model = body.model
    url, key, is_inside = get_lazy_infra(request, model)
    if not url:
        raise HTTPException(400, "Given model is not found")

    if not is_inside:
        return await post_json(body, ProxyOptions(url=get_proxy_url(url, request), headers=auth_header(key)), request)

    return await endpoint_registry.execute_audio_speech(request, model, body)


@router.post("/v1/audio/transcriptions")
async def on_audio_transcriptions(
    request: Request,
    body: Annotated[CreateTranscriptionRequest, Form(..., media_type="multipart/form-data")],
    _: Annotated[str, Depends(auth_server)],
    endpoint_registry: Annotated[EndpointRegistry, Depends(get_endpoint_registry)],
) -> StarletteResponse:
    """Process audio translation request."""
    model = body.model
    url, key, is_inside = get_lazy_infra(request, model)
    if not url:
        raise HTTPException(400, "Given model is not found")

    if not is_inside:
        return await post_form(body, ProxyOptions(url=get_proxy_url(url, request), headers=auth_header(key)), request)

    return await endpoint_registry.execute_audio_transcriptions(request, model, body)


@router.post("/v1/images/generations")
async def on_images_generations(
    request: Request,
    body: Annotated[ImagesRequest, Body(...)],
    _: Annotated[str, Depends(auth_server)],
    endpoint_registry: Annotated[EndpointRegistry, Depends(get_endpoint_registry)],
) -> StarletteResponse:
    """Process images genenerations request."""
    model = body.model
    url, key, is_inside = get_lazy_infra(request, model)
    if not url:
        raise HTTPException(400, "Given model is not found")

    if not is_inside:
        return await post_json(body, ProxyOptions(url=get_proxy_url(url, request), headers=auth_header(key)), request)

    return await endpoint_registry.execute_images_generations(request, model, body)


@router.post("/custom/{full_path:path}")
async def on_custom_endpoint(
    request: Request,
    _: Annotated[str, Depends(auth_server)],
    endpoint_registry: Annotated[EndpointRegistry, Depends(get_endpoint_registry)],
) -> StarletteResponse:
    """Process custom endpoint request."""
    url_path = request.url.path[7:]
    if not endpoint_registry.has_custom_endpoint(url_path):
        return JSONResponse(content="404 Not found, given url is no registered", status_code=404)
    return await endpoint_registry.execute_custom_endpoints(request, url_path)
