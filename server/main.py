"""Main app module."""

from typing import Annotated

from fastapi import Body, Depends, FastAPI, Form, HTTPException, Path, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from server.api import models, services
from server.core.dependencies import auth_server, get_endpoint_registry
from server.endpointregistry import EndpointRegistry
from server.lifecycle import lifespan
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

app = FastAPI(lifespan=lifespan)
app.include_router(services.router)
app.include_router(models.router)


@app.get("/v1/models")
async def on_models(
    _: Annotated[str, Depends(auth_server)],
    endpoint_registry: Annotated[EndpointRegistry, Depends(get_endpoint_registry)],
) -> ChatCompletionModels:
    """Process models request."""
    return endpoint_registry.get_chat_completions_models()


@app.get("/v1/models/{model_id}")
async def on_model(
    _: Annotated[str, Depends(auth_server)],
    model_id: Annotated[str, Path(..., description="The ID of the model to use.")],
    endpoint_registry: Annotated[EndpointRegistry, Depends(get_endpoint_registry)],
) -> ChatCompletionModel:
    """Process model request."""
    return endpoint_registry.get_chat_completions_model(model_id)


@app.post("/v1/chat/completions")
async def on_chat_completions(
    request: Request,
    body: Annotated[ChatCompletionRequest, Body(...)],
    _: Annotated[str, Depends(auth_server)],
    endpoint_registry: Annotated[EndpointRegistry, Depends(get_endpoint_registry)],
) -> StarletteResponse:
    """Process chat completions request."""
    if not endpoint_registry.has_chat_completion_model(body.model):
        return JSONResponse(content={"error": "Given model is not supported"}, status_code=400)
    return await endpoint_registry.execute_chat_completion(body, request)


@app.post("/v1/completions")
async def on_completions(
    request: Request,
    body: Annotated[CompletionLegacyRequest, Body(...)],
    _: Annotated[str, Depends(auth_server)],
    endpoint_registry: Annotated[EndpointRegistry, Depends(get_endpoint_registry)],
) -> StarletteResponse:
    """Process completions request."""
    if not endpoint_registry.has_completion_model(body.model):
        return JSONResponse(content={"error": "Given model is not supported"}, status_code=400)
    return await endpoint_registry.execute_completion(body, request)


@app.post("/v1/embeddings")
async def on_embeddings(
    request: Request,
    body: Annotated[EmbeddingRequest, Body(...)],
    _: Annotated[str, Depends(auth_server)],
    endpoint_registry: Annotated[EndpointRegistry, Depends(get_endpoint_registry)],
) -> StarletteResponse:
    """Process embeddings request."""
    if not endpoint_registry.has_embeddings_model(body.model):
        return JSONResponse(content={"error": "Given model is not supported"}, status_code=400)
    return await endpoint_registry.execute_embeddings(body, request)


@app.post("/v1/audio/speech")
async def on_audio_speech(
    request: Request,
    body: Annotated[CreateSpeechRequest, Body(...)],
    _: Annotated[str, Depends(auth_server)],
    endpoint_registry: Annotated[EndpointRegistry, Depends(get_endpoint_registry)],
) -> StarletteResponse:
    """Process audio speech request."""
    if not endpoint_registry.has_audio_speech_model(body.model):
        return JSONResponse(content={"error": "Given model is not supported"}, status_code=400)
    return await endpoint_registry.execute_audio_speech(body, request)


@app.post("/v1/audio/transcriptions")
async def on_audio_transcriptions(
    request: Request,
    form: Annotated[CreateTranscriptionRequest, Form(..., media_type="multipart/form-data")],
    _: Annotated[str, Depends(auth_server)],
    endpoint_registry: Annotated[EndpointRegistry, Depends(get_endpoint_registry)],
) -> StarletteResponse:
    """Process audio translation request."""
    if not endpoint_registry.has_audio_transcriptions_model(form.model):
        return JSONResponse(content={"error": "Given model is not supported"}, status_code=400)
    return await endpoint_registry.execute_audio_transcriptions(form, request)


@app.post("/v1/images/generations")
async def on_images_generations(
    request: Request,
    body: Annotated[ImagesRequest, Body(...)],
    _: Annotated[str, Depends(auth_server)],
    endpoint_registry: Annotated[EndpointRegistry, Depends(get_endpoint_registry)],
) -> StarletteResponse:
    """Process images genenerations request."""
    if not endpoint_registry.has_image_generations_model(body.model):
        raise HTTPException(400, "Given model is not supported")
    return await endpoint_registry.execute_images_generations(body, request)


@app.api_route(
    "/custom/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
)
async def on_custom_endpoint(
    request: Request,
    _: Annotated[str, Depends(auth_server)],
    endpoint_registry: Annotated[EndpointRegistry, Depends(get_endpoint_registry)],
) -> StarletteResponse:
    """Process custom endpoint request."""
    url = request.url.path[7:]
    if not endpoint_registry.has_custom_endpoint(url):
        return JSONResponse(content="404 Not found", status_code=404)
    return await endpoint_registry.execute_custom_endpoints(url, request)


app.mount("/", StaticFiles(directory="static", html=True), name="static")
