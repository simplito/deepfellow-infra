"""Lifecycle."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from server.applicationcontext import ApplicationContext
from server.config import AppSettings, ConfigError
from server.endpointregistry import EndpointRegistry
from server.serviceprovider import ServiceProvider


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """FastAPI Lifespan.

    In here:
    You can define logic (code) that should be executed before the application starts up.
    This means that this code will be executed once, before the application starts receiving requests.
    """
    try:
        app.state.config = AppSettings()  # type: ignore
    except Exception as e:
        raise ConfigError("Config error. Have you created the .env file?") from e

    app.state.endpoint_registry = EndpointRegistry()
    app.state.service_provider = ServiceProvider()
    app.state.context = ApplicationContext(app.state.endpoint_registry, app.state.config, app.state.service_provider)
    await app.state.context.load()
    yield
