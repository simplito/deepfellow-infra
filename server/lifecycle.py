"""Lifecycle."""

import asyncio
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
from server.services.openai_service import OpenAIService
from server.services.sindri_service import SindriService
from server.services.speches_ai_service import SpeachesAIService
from server.services.stable_diffusion_service import StableDiffusionService
from server.services.vllm_service import VllmService
from server.services_manager import ServicesManager
from server.websockets.models import InfraInfo
from server.websockets.subinfra import ExternalInfraWsManager, InternalInfraWsManager
from server.websockets.utils import create_infra_uri


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """FastAPI Lifespan.

    In here:
    You can define logic (code) that should be executed before the application starts up.
    This means that this code will be executed once, before the application starts receiving requests.
    """
    # Definitions
    try:
        app.state.config = config = AppSettings()  # type: ignore
    except Exception as e:
        raise ConfigError("Config error. Have you created the .env file?") from e

    app.state.internal_ws_manager = InternalInfraWsManager()
    app.state.external_ws_manager = external_ws_manager = ExternalInfraWsManager(create_infra_uri(config.parent_infra), app.state)
    app.state.endpoint_registry = endpoint_registry = EndpointRegistry()
    app.state.service_provider = service_provider = ServiceProvider()
    app.state.services_manager = services_manager = ServicesManager(external_ws_manager)
    app.state.context = context = ApplicationContext(endpoint_registry, config, service_provider, services_manager)
    app.state.tasks = tasks = set[asyncio.Task[None]]()
    app.state.models_list = dict[str, dict[str, list[str]]]()  # url | type | model
    app.state.models_usage = dict[str, dict[str, int]]()  # model | url | usage
    app.state.infra_infos = list[InfraInfo]()

    # Register services
    services_manager.register_service(OllamaService(context, endpoint_registry, service_provider))
    services_manager.register_service(SpeachesAIService(context, endpoint_registry, service_provider))
    services_manager.register_service(StableDiffusionService(context, endpoint_registry, service_provider))
    services_manager.register_service(LLamacppService(context, endpoint_registry, service_provider))
    services_manager.register_service(VllmService(context, endpoint_registry, service_provider))
    services_manager.register_service(CustomService(context, endpoint_registry, service_provider))
    services_manager.register_service(CoquiService(context, endpoint_registry, service_provider))
    services_manager.register_service(SindriService(context, endpoint_registry, service_provider))
    services_manager.register_service(OpenAIService(context, endpoint_registry, service_provider))

    # Load functions
    await context.load()
    tasks.add(asyncio.create_task(external_ws_manager.run()))

    yield
