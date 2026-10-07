# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Lifecycle."""

import asyncio
import logging
import os
import re
import subprocess
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from server.applicationcontext import ApplicationContext
from server.config import ConfigError, get_main_dir, load_config
from server.docker import create_docker_service
from server.dynamic_config import load_or_init as load_dynamic_config
from server.endpointregistry import EndpointRegistry
from server.metrics import MetricsService
from server.metrics_registry import MetricsRegistry
from server.model_tester import ModelTester
from server.portservice import PortService
from server.serviceprovider import ServiceProvider
from server.services.claude_service import ClaudeService
from server.services.coqui_service import CoquiService
from server.services.custom_service import CustomService
from server.services.deepseek_service import DeepSeekService
from server.services.docker_model_runner_service import DockerModelRunnerService
from server.services.googleai_service import GoogleAIService
from server.services.kimi_service import KimiService
from server.services.llamacpp_service import LLamacppService
from server.services.mcp_service import McpService
from server.services.model_catalog_refresh import probe_static_dir_writable, set_catalog_refresh_supported
from server.services.ollama_cloud_service import OllamaCloudService
from server.services.ollama_external_service import OllamaExternalService
from server.services.ollama_service import OllamaService
from server.services.openai_service import OpenAIService
from server.services.rerank_service import RerankService
from server.services.sglang_service import SglangService
from server.services.speaches_ai_service import SpeachesAIService
from server.services.stable_diffusion_service import StableDiffusionService
from server.services.vllm_service import VllmService
from server.services_manager import ServicesManager
from server.task_manager import TaskManager
from server.utils.exceptions import AppStartError
from server.utils.hardware import Hardware
from server.utils.model_downloader import ModelDownloader
from server.utils.tracing import OtlpLoggingManager, tracer
from server.websockets.infra_websocket_server import InfraWebsocketServer
from server.websockets.parent_infra import ParentInfra
from server.websockets.parent_infra_group import ParentInfraGroup

logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """FastAPI Lifespan.

    In here:
    You can define logic (code) that should be executed before the application starts up.
    This means that this code will be executed once, before the application starts receiving requests.
    """
    # Definitions
    try:
        try:
            app.state.config = config = load_config()
            app.state.config_lock = asyncio.Lock()
            load_dynamic_config(config)
        except ConfigError as e:
            raise AppStartError(str(e))  # noqa: B904
        except Exception as e:
            raise AppStartError("Config error. Check your .env file and config.json.") from e

        tracer.config = config
        app.state.otlp_logging = otlp_logging = OtlpLoggingManager()
        if config.otel_logging_enabled:
            otlp_logging.setup(config)

        if not config.external_only and app.state.config.docker_subnet:
            check_subnet(app.state.config.docker_subnet)

        app.state.hardware = hardware = Hardware()
        await hardware.init_async()

        set_catalog_refresh_supported(probe_static_dir_writable(get_main_dir() / "static"))

        app.state.metrics_registry = metrics_registry = MetricsRegistry()
        app.state.task_manager = task_manager = TaskManager()
        app.state.service_provider = service_provider = ServiceProvider(config)
        parents = [ParentInfra(config, task_manager, config.connect_to_mesh_url)] if config.connect_to_mesh_url else []
        app.state.parent_infra = parent_infra = ParentInfraGroup(parents)
        app.state.services_manager = services_manager = ServicesManager(external_only=config.external_only)
        app.state.model_tester = model_tester = ModelTester()
        app.state.endpoint_registry = endpoint_registry = EndpointRegistry(config, parent_infra, model_tester, metrics_registry)
        app.state.infra_websocket_server = infra_websocket_server = InfraWebsocketServer(config, parent_infra, endpoint_registry)
        app.state.metrics_service = MetricsService(
            endpoint_registry, config, hardware, services_manager, metrics_registry, infra_websocket_server
        )
        app.state.context = context = ApplicationContext(endpoint_registry, config, service_provider, services_manager)
        app.state.port_service = port_service = PortService()
        app.state.docker_service = docker_service = await create_docker_service(port_service, config)
        app.state.model_downloader = model_downloader = ModelDownloader(app.state.config)

        model_input = (config, endpoint_registry, service_provider, model_downloader, docker_service, hardware)

        # Register services
        if config.external_only:
            # Only socket-free services may run here.
            services_manager.register_service(OpenAIService(*model_input))
            assert_external_only_registration(services_manager)
        else:
            services_manager.register_service(ClaudeService(*model_input))
            services_manager.register_service(CoquiService(*model_input))
            services_manager.register_service(CustomService(*model_input))
            services_manager.register_service(DeepSeekService(*model_input))
            services_manager.register_service(DockerModelRunnerService(*model_input))
            services_manager.register_service(GoogleAIService(*model_input))
            services_manager.register_service(KimiService(*model_input))
            services_manager.register_service(LLamacppService(*model_input))
            services_manager.register_service(McpService(*model_input))
            services_manager.register_service(OllamaCloudService(*model_input))
            services_manager.register_service(OllamaExternalService(*model_input))
            services_manager.register_service(OllamaService(*model_input))
            services_manager.register_service(OpenAIService(*model_input))
            services_manager.register_service(RerankService(*model_input))
            services_manager.register_service(SglangService(*model_input))
            services_manager.register_service(SpeachesAIService(*model_input))
            services_manager.register_service(StableDiffusionService(*model_input))
            services_manager.register_service(VllmService(*model_input))

        # Load functions
        await context.load_services()
        task_manager.add_task(parent_infra.run())
    except AppStartError as e:
        logger.error(str(e))  # noqa: TRY400
        os._exit(1)
    yield
    # Shutdown: stop all services gracefully if enabled
    logger.debug("Shutdown all containers gracefully")
    await services_manager.drain_warning_tasks()
    if config.is_stop_containers_on_shutdown_enabled():
        await services_manager.stop_all_services()
    await tracer.shutdown()
    await otlp_logging.teardown()


def assert_external_only_registration(services_manager: ServicesManager) -> None:
    """Fail startup if the external-only registration and its allowlist disagree.

    `ServicesManager.EXTERNAL_ONLY_ALLOWED_SERVICE_TYPES` is the single declaration of which
    service types may run without Docker, but the registration above is a separate list. If a
    type were added to one and not the other the mismatch would surface only as confusing HTTP
    semantics — a 404 for a type that is supposed to answer a deterministic 409, or the reverse.

    Raises:
        AppStartError: If the registered types differ from the allowlist.
    """
    registered = frozenset(services_manager.services)
    allowed = ServicesManager.EXTERNAL_ONLY_ALLOWED_SERVICE_TYPES
    if registered != allowed:
        msg = (
            "external-only registration does not match EXTERNAL_ONLY_ALLOWED_SERVICE_TYPES: "
            f"registered={sorted(registered)}, allowed={sorted(allowed)}"
        )
        raise AppStartError(msg)


def check_subnet(subnet: str) -> None:
    """Check if a Docker network exists and is active.

    Args:
        subnet: The name of the Docker network to check.

    Returns:
        None

    Raises:
        AppStartError: If subnet name contains invalid characters or it does not exist.
        subprocess.CalledProcessError: If docker command fails unexpectedly.
    """
    if not re.match(r"^[a-zA-Z0-9._-]+$", subnet):
        msg = f"Invalid subnet name: {subnet}"
        raise AppStartError(msg)

    try:
        subprocess.run(
            ["docker", "network", "inspect", subnet],  # NOTE: list = safe
            capture_output=True,
            check=True,
            timeout=10,
        )
    except subprocess.CalledProcessError as e:
        if e.returncode == 1:
            msg = f"Given docker network does not exist ({subnet}). Create it and start again."
            raise AppStartError(msg) from e
        raise
    except subprocess.TimeoutExpired as e:
        timeout_code = 124
        raise subprocess.CalledProcessError(timeout_code, ["docker", "network", "inspect", subnet]) from e
