#from mcp.server.lowlevel.server import Server as MCPServer
#from mcp.types import Tool as MCPTool
from typing import Any
#from mcp.server.sse import SseServerTransport
from fastapi import FastAPI, Request, Response, HTTPException
from starlette.routing import Route, Scope, Receive, Send
from contextlib import asynccontextmanager
from server.config import Config
from server.endpointregistry import EndpointRegistry
from server.applicationcontext import ApplicationContext
from server.serviceprovider import ServiceProvider
from fastapi.responses import JSONResponse

from server.stable_diffusion import ImagesRequest, ImagesResponse

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.config = Config()
    app.state.endpoint_registry = EndpointRegistry()
    app.state.service_provider = ServiceProvider()
    app.state.context = ApplicationContext(app.state.endpoint_registry, app.state.config, app.state.service_provider)
    await app.state.context.load()
    yield
    # tu możesz posprzątać, jeśli trzeba

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def home(request: Request):
    return Response("Hello world")

@app.post("/admin")
async def admin(request: Request):
    body = await read_json(request)
    res = await app.state.context.run(body["args"])
    return JSONResponse(content=res)

@app.post("/v1/chat/completions")
async def on_chat_complete(request: Request):
    body = await read_json(request)
    if not app.state.endpoint_registry.has_chat_completion_model(body["model"]):
        return JSONResponse(content={"error": "Given model is not supported"}, status_code=400)
    return app.state.endpoint_registry.execute_chat_completion(body, request)

@app.post("/v1/audio/speech")
async def on_audio_speech(request: Request):
    body = await read_json(request)
    if not app.state.endpoint_registry.has_audio_speech_model(body["model"]):
        return JSONResponse(content={"error": "Given model is not supported"}, status_code=400)
    return await app.state.endpoint_registry.execute_audio_speech(body, request)

@app.post("/v1/images/generations")
async def on_image_generation(request: Request, body: ImagesRequest) -> ImagesResponse:
    if not app.state.endpoint_registry.has_image_generations_model(body.model):
        raise HTTPException(400, "Given model is not supported")
    return await app.state.endpoint_registry.execute_images_generations(body, request)

@app.post("/{full_path:path}")
async def on_custom_endpoint(request: Request):
    print(f"{request.url.path=}")
    body = await read_json(request)
    if not app.state.endpoint_registry.has_custom_endpoint(request.url.path):
        return JSONResponse(content="404 Not found", status_code=404)
    return app.state.endpoint_registry.execute_custom_endpoints(request.url.path, body, request)

async def read_json(request: Request):
    try:
        return await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid json")
