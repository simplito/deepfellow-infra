"""Main app module."""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from server.api import models, openai, services
from server.lifecycle import lifespan
from server.websockets import api as websocket

app = FastAPI(lifespan=lifespan)
app.include_router(services.router)
app.include_router(models.router)
app.include_router(websocket.router)
app.include_router(openai.router)

app.mount("/", StaticFiles(directory="static", html=True), name="static")
