"""Main app module."""

from typing import Annotated, Any, cast

from fastapi import Depends, FastAPI, HTTPException, Path, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from server.api import models, services
from server.core.dependencies import get_endpoint_registry
from server.dependecies import auth_server
from server.endpointregistry import EndpointRegistry
from server.lifecycle import lifespan

app = FastAPI(lifespan=lifespan)
app.include_router(services.router)
app.include_router(models.router)


@app.get("/v1/models")
async def on_models(
    _: Annotated[str, Depends(auth_server)],
    endpoint_registry: Annotated[EndpointRegistry, Depends(get_endpoint_registry)],
) -> Any:  # noqa: ANN401
    """Process models request."""
    return {"object": "list", "data": endpoint_registry.get_chat_completions_models()}


@app.get("/v1/models/{model_id}")
async def on_model(
    _: Annotated[str, Depends(auth_server)],
    model_id: Annotated[str, Path(..., description="The ID of the model to use.")],
    endpoint_registry: Annotated[EndpointRegistry, Depends(get_endpoint_registry)],
) -> Any:  # noqa: ANN401
    """Process models request."""
    return endpoint_registry.get_chat_completions_model(model_id)


@app.post("/v1/chat/completions")
async def on_chat_complete(
    request: Request,
    body: dict,
    _: Annotated[str, Depends(auth_server)],
    endpoint_registry: Annotated[EndpointRegistry, Depends(get_endpoint_registry)],
) -> Any:  # noqa: ANN401
    """Process chat completions request."""
    if not endpoint_registry.has_chat_completion_model(body["model"]):
        return JSONResponse(content={"error": "Given model is not supported"}, status_code=400)
    return await endpoint_registry.execute_chat_completion(body, request)


@app.post("/v1/embeddings")
async def on_embeddings(
    request: Request,
    _: Annotated[str, Depends(auth_server)],
    endpoint_registry: Annotated[EndpointRegistry, Depends(get_endpoint_registry)],
) -> Any:  # noqa: ANN401
    """Process embeddings request."""
    body = await _read_json(request)
    if not endpoint_registry.has_embeddings_model(body["model"]):
        return JSONResponse(content={"error": "Given model is not supported"}, status_code=400)
    return await endpoint_registry.execute_embeddings(body, request)


@app.post("/v1/audio/speech")
async def on_audio_speech(
    request: Request,
    _: Annotated[str, Depends(auth_server)],
    endpoint_registry: Annotated[EndpointRegistry, Depends(get_endpoint_registry)],
) -> Any:  # noqa: ANN401
    """Process audio speech request."""
    body = await _read_json(request)
    if not endpoint_registry.has_audio_speech_model(body["model"]):
        return JSONResponse(content={"error": "Given model is not supported"}, status_code=400)
    return await endpoint_registry.execute_audio_speech(body, request)


@app.post("/v1/audio/transcriptions")
async def on_audio_translation(
    request: Request,
    _: Annotated[str, Depends(auth_server)],
    endpoint_registry: Annotated[EndpointRegistry, Depends(get_endpoint_registry)],
) -> Any:  # noqa: ANN401
    """Process audio translation request."""
    body = dict(await request.form())
    if not endpoint_registry.has_audio_transcriptions_model(cast("str", body["model"])):
        return JSONResponse(content={"error": "Given model is not supported"}, status_code=400)
    return await endpoint_registry.execute_audio_transcriptions(body, request)


@app.post("/v1/images/generations")
async def on_image_generation(
    request: Request,
    _: Annotated[str, Depends(auth_server)],
    endpoint_registry: Annotated[EndpointRegistry, Depends(get_endpoint_registry)],
) -> Any:  # noqa: ANN401
    """Process images genenerations request."""
    body = await _read_json(request)
    if not endpoint_registry.has_image_generations_model(body["model"]):
        raise HTTPException(400, "Given model is not supported")
    return await endpoint_registry.execute_images_generations(body, request)


@app.post("/{full_path:path}")
async def on_custom_endpoint(
    request: Request,
    _: Annotated[str, Depends(auth_server)],
    endpoint_registry: Annotated[EndpointRegistry, Depends(get_endpoint_registry)],
) -> Any:  # noqa: ANN401
    """Process custom endpoint request."""
    body = await _read_json(request)
    if not endpoint_registry.has_custom_endpoint(request.url.path):
        return JSONResponse(content="404 Not found", status_code=404)
    return await endpoint_registry.execute_custom_endpoints(request.url.path, body, request)


app.mount("/", StaticFiles(directory="static", html=True), name="static")


async def _read_json(request: Request) -> dict:
    try:
        return await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid json") from None
