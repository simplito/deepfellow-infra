"""API endpoints for openai compatibile endpoints."""

import logging
from typing import Annotated

from aiohttp import FormData
from fastapi import APIRouter, Body, Depends, Form, HTTPException, Path, Request
from fastapi.responses import JSONResponse

from server.core.dependencies import auth_server, get_endpoint_registry
from server.endpointregistry import EndpointRegistry, ProxyOptions, proxy
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
    # url, key, is_inside = get_lazy_infra(request, model)
    # if not url:
    #     raise HTTPException(400, "Given model is not found")

    # if not is_inside:
    #     return await proxy(body.model_dump(), ProxyOptions(url=get_proxy_url(url, request)), request, auth_header(key))
    # TODO revert changes
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
        return await proxy(body.model_dump(), ProxyOptions(url=get_proxy_url(url, request)), request, auth_header(key))

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
        return await proxy(body.model_dump(), ProxyOptions(url=get_proxy_url(url, request)), request, auth_header(key))

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
        return await proxy(body.model_dump(), ProxyOptions(url=get_proxy_url(url, request)), request, auth_header(key))

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

        return await proxy(body.model_dump(), ProxyOptions(url=get_proxy_url(url, request), form=True), request, auth_header(key))

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
        return await proxy(body.model_dump(), ProxyOptions(url=get_proxy_url(url, request)), request, auth_header(key))

    return await endpoint_registry.execute_images_generations(request, model, body)


@router.post("/custom/{full_path:path}")
async def on_custom_endpoint(
    request: Request,
    _: Annotated[str, Depends(auth_server)],
    endpoint_registry: Annotated[EndpointRegistry, Depends(get_endpoint_registry)],
) -> StarletteResponse:
    """Process custom endpoint request."""
    if not endpoint_registry.has_custom_endpoint(request.url.path):
        return JSONResponse(content="404 Not found", status_code=404)
    return await endpoint_registry.execute_custom_endpoints(request, request.url.path)
