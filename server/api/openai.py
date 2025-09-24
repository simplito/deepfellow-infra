"""API endpoints for openai compatibile endpoints."""

import logging
from typing import Annotated

from fastapi import APIRouter, Body, Depends, Form, HTTPException, Path, Request
from fastapi.responses import JSONResponse

from server.core.dependencies import auth_server, get_endpoint_registry, get_load_balancer
from server.endpointregistry import EndpointRegistry, post_form, post_json
from server.models.api import (
    ApiModel,
    ApiModels,
    ChatCompletionRequest,
    CompletionLegacyRequest,
    CreateSpeechRequest,
    CreateTranscriptionRequest,
    EmbeddingRequest,
    ImagesRequest,
)
from server.models.common import StarletteResponse
from server.websockets.loadbalancer import LoadBalancer

logger = logging.getLogger("uvicorn.error")


router = APIRouter(prefix="", tags=["AI Endpoints"])


@router.get("/v1/models")
async def on_models(
    _: Annotated[str, Depends(auth_server)],
    endpoint_registry: Annotated[EndpointRegistry, Depends(get_endpoint_registry)],
) -> ApiModels:
    """Process models request."""
    return endpoint_registry.get_models()


@router.get("/v1/models/{model_id}")
async def on_model(
    _: Annotated[str, Depends(auth_server)],
    model_id: Annotated[str, Path(..., description="The ID of the model to use.")],
    endpoint_registry: Annotated[EndpointRegistry, Depends(get_endpoint_registry)],
) -> ApiModel:
    """Process model request."""
    return endpoint_registry.get_model(model_id)


@router.post("/v1/chat/completions")
async def on_chat_completions(
    request: Request,
    body: Annotated[ChatCompletionRequest, Body(...)],
    _: Annotated[str, Depends(auth_server)],
    endpoint_registry: Annotated[EndpointRegistry, Depends(get_endpoint_registry)],
    load_balancer: Annotated[LoadBalancer, Depends(get_load_balancer)],
) -> StarletteResponse:
    """Process chat completions request."""
    info = load_balancer.get_lazy_infra(body.model)
    if info is None:
        raise HTTPException(400, "Given model is not found")

    if info == "internal":
        return await endpoint_registry.execute_chat_completion(request, body.model, body)

    return await post_json(body, info.get_proxy_options(request.url.path), request)


@router.post("/v1/completions")
async def on_completions(
    request: Request,
    body: Annotated[CompletionLegacyRequest, Body(...)],
    _: Annotated[str, Depends(auth_server)],
    endpoint_registry: Annotated[EndpointRegistry, Depends(get_endpoint_registry)],
    load_balancer: Annotated[LoadBalancer, Depends(get_load_balancer)],
) -> StarletteResponse:
    """Process completions request."""
    info = load_balancer.get_lazy_infra(body.model)
    if info is None:
        raise HTTPException(400, "Given model is not found")

    if info == "internal":
        return await endpoint_registry.execute_completion(request, body.model, body)

    return await post_json(body, info.get_proxy_options(request.url.path), request)


@router.post("/v1/embeddings")
async def on_embeddings(
    request: Request,
    body: Annotated[EmbeddingRequest, Body(...)],
    _: Annotated[str, Depends(auth_server)],
    endpoint_registry: Annotated[EndpointRegistry, Depends(get_endpoint_registry)],
    load_balancer: Annotated[LoadBalancer, Depends(get_load_balancer)],
) -> StarletteResponse:
    """Process embeddings request."""
    info = load_balancer.get_lazy_infra(body.model)
    if info is None:
        raise HTTPException(400, "Given model is not found")

    if info == "internal":
        return await endpoint_registry.execute_embeddings(request, body.model, body)

    return await post_json(body, info.get_proxy_options(request.url.path), request)


@router.post("/v1/audio/speech")
async def on_audio_speech(
    request: Request,
    body: Annotated[CreateSpeechRequest, Body(...)],
    _: Annotated[str, Depends(auth_server)],
    endpoint_registry: Annotated[EndpointRegistry, Depends(get_endpoint_registry)],
    load_balancer: Annotated[LoadBalancer, Depends(get_load_balancer)],
) -> StarletteResponse:
    """Process audio speech request."""
    info = load_balancer.get_lazy_infra(body.model)
    if info is None:
        raise HTTPException(400, "Given model is not found")

    if info == "internal":
        return await endpoint_registry.execute_audio_speech(request, body.model, body)

    return await post_json(body, info.get_proxy_options(request.url.path), request)


@router.post("/v1/audio/transcriptions")
async def on_audio_transcriptions(
    request: Request,
    body: Annotated[CreateTranscriptionRequest, Form(..., media_type="multipart/form-data")],
    _: Annotated[str, Depends(auth_server)],
    endpoint_registry: Annotated[EndpointRegistry, Depends(get_endpoint_registry)],
    load_balancer: Annotated[LoadBalancer, Depends(get_load_balancer)],
) -> StarletteResponse:
    """Process audio translation request."""
    info = load_balancer.get_lazy_infra(body.model)
    if info is None:
        raise HTTPException(400, "Given model is not found")

    if info == "internal":
        return await endpoint_registry.execute_audio_transcriptions(request, body.model, body)

    return await post_form(body, info.get_proxy_options(request.url.path), request)


@router.post("/v1/images/generations")
async def on_images_generations(
    request: Request,
    body: Annotated[ImagesRequest, Body(...)],
    _: Annotated[str, Depends(auth_server)],
    endpoint_registry: Annotated[EndpointRegistry, Depends(get_endpoint_registry)],
    load_balancer: Annotated[LoadBalancer, Depends(get_load_balancer)],
) -> StarletteResponse:
    """Process images genenerations request."""
    info = load_balancer.get_lazy_infra(body.model)
    if info is None:
        raise HTTPException(400, "Given model is not found")

    if info == "internal":
        return await endpoint_registry.execute_images_generations(request, body.model, body)

    return await post_json(body, info.get_proxy_options(request.url.path), request)


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
