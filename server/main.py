"""Main app module."""

from typing import Annotated, Any, cast

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from server.dependecies import auth_server
from server.endpointregistry import EndpointRegistry
from server.lifecycle import lifespan

app = FastAPI(lifespan=lifespan)


@app.get("/")
async def home() -> Response:
    """Return hello world label."""
    return Response("Hello world")


@app.post("/admin")
async def admin(request: Request) -> JSONResponse:
    """Process administration calls."""
    body = await _read_json(request)
    res = await app.state.context.run(body["args"])
    return JSONResponse(content=res)


@app.post("/v1/chat/completions")
async def on_chat_complete(request: Request, body: dict, _: Annotated[str, Depends(auth_server)]) -> Any:  # noqa: ANN401
    """Process chat completions request."""
    endpoint_registry = _get_endpoint()
    if not endpoint_registry.has_chat_completion_model(body["model"]):
        return JSONResponse(content={"error": "Given model is not supported"}, status_code=400)
    return endpoint_registry.execute_chat_completion(body, request)


@app.post("/v1/audio/speech")
async def on_audio_speech(request: Request, _: Annotated[str, Depends(auth_server)]) -> Any:  # noqa: ANN401
    """Process audio speech request."""
    body = await _read_json(request)
    endpoint_registry = _get_endpoint()
    if not endpoint_registry.has_audio_speech_model(body["model"]):
        return JSONResponse(content={"error": "Given model is not supported"}, status_code=400)
    return await endpoint_registry.execute_audio_speech(body, request)


@app.post("/v1/audio/transcriptions")
async def on_audio_translation(request: Request, _: Annotated[str, Depends(auth_server)]) -> Any:  # noqa: ANN401
    """Process audio translation request."""
    body = dict(await request.form())
    endpoint_registry = _get_endpoint()
    if not endpoint_registry.has_audio_transcriptions_model(cast("str", body["model"])):
        return JSONResponse(content={"error": "Given model is not supported"}, status_code=400)
    return await endpoint_registry.execute_audio_transcriptions(body, request)


@app.post("/v1/images/generations")
async def on_image_generation(request: Request, _: Annotated[str, Depends(auth_server)]) -> Any:  # noqa: ANN401
    """Process images genenerations request."""
    body = await _read_json(request)
    endpoint_registry = _get_endpoint()
    if not endpoint_registry.has_image_generations_model(body["model"]):
        raise HTTPException(400, "Given model is not supported")
    return await endpoint_registry.execute_images_generations(body, request)


@app.post("/{full_path:path}")
async def on_custom_endpoint(request: Request, _: Annotated[str, Depends(auth_server)]) -> Any:  # noqa: ANN401
    """Process custom endpoint request."""
    body = await _read_json(request)
    endpoint_registry = _get_endpoint()
    if not endpoint_registry.has_custom_endpoint(request.url.path):
        return JSONResponse(content="404 Not found", status_code=404)
    return endpoint_registry.execute_custom_endpoints(request.url.path, body, request)


def _get_endpoint() -> EndpointRegistry:
    return app.state.endpoint_registry


async def _read_json(request: Request) -> dict:
    try:
        return await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid json") from None
