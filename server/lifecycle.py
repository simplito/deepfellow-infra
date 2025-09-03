"""Lifecycle."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from server.applicationcontext import ApplicationContext
from server.config import AppSettings, ConfigError
from server.endpointregistry import EndpointRegistry
from server.serviceprovider import ServiceProvider
from server.services.coqui_service import CoquiService
from server.services.custom_service import CustomService
from server.services.llamapcpp_service import LLamacppService
from server.services.ollama_service import OllamaService
from server.services.speches_ai_service import SpeachesAIService
from server.services.stable_diffusion_service import StableDiffusionService
from server.services.vllm_service import VllmService
from server.services_manager import ServicesManager


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
    app.state.services_manager = ServicesManager()
    app.state.context = ApplicationContext(
        app.state.endpoint_registry, app.state.config, app.state.service_provider, app.state.services_manager
    )
    app.state.services_manager.register_service(OllamaService(app.state.context, app.state.endpoint_registry, app.state.service_provider))
    app.state.services_manager.register_service(
        SpeachesAIService(app.state.context, app.state.endpoint_registry, app.state.service_provider)
    )
    app.state.services_manager.register_service(
        StableDiffusionService(app.state.context, app.state.endpoint_registry, app.state.service_provider)
    )
    app.state.services_manager.register_service(LLamacppService(app.state.context, app.state.endpoint_registry, app.state.service_provider))
    app.state.services_manager.register_service(VllmService(app.state.context, app.state.endpoint_registry, app.state.service_provider))
    app.state.services_manager.register_service(CustomService(app.state.context, app.state.endpoint_registry, app.state.service_provider))
    app.state.services_manager.register_service(CoquiService(app.state.context, app.state.endpoint_registry, app.state.service_provider))

    await app.state.context.load()
    yield
