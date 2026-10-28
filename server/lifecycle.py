# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Lifecycle."""

import asyncio
import os
import re
import subprocess
from collections.abc import AsyncGenerator, Awaitable, Callable
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
from server.services.custom_service import CustomService
from server.services.deepseek_service import DeepSeekService
from server.services.docker_model_runner_service import DockerModelRunnerService
from server.services.finetune_service import FinetuneService
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
from server.utils.logger import logger
from server.utils.model_downloader import ModelDownloader
from server.utils.sentry import sentry
from server.utils.tracing import OtlpLoggingManager, tracer
from server.websockets.infra_websocket_server import InfraWebsocketServer
from server.websockets.parent_infra import ParentInfra
from server.websockets.parent_infra_group import ParentInfraGroup


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
        try:
            await sentry.reconfigure(config)
        except Exception:
            # Error tracking must never block startup; the next config update retries it.
            logger.warning("Sentry error tracking could not be set up and is off", exc_info=True)
        app.state.otlp_logging = otlp_logging = OtlpLoggingManager()
        if config.otel_logging_enabled:
            otlp_logging.setup(config)

        if app.state.config.docker_subnet:
            check_subnet(app.state.config.docker_subnet)

        app.state.hardware = hardware = Hardware()
        await hardware.init_async()

        set_catalog_refresh_supported(probe_static_dir_writable(get_main_dir() / "static"))

        app.state.metrics_registry = metrics_registry = MetricsRegistry()
        app.state.task_manager = task_manager = TaskManager()
        app.state.service_provider = service_provider = ServiceProvider(config)
        parents = [ParentInfra(config, task_manager, config.connect_to_mesh_url)] if config.connect_to_mesh_url else []
        app.state.parent_infra = parent_infra = ParentInfraGroup(
            parents, configured_mesh_url=config.connect_to_mesh_url, configured_mesh_key=config.connect_to_mesh_key
        )
        app.state.services_manager = services_manager = ServicesManager()
        sentry.register_secret_source(services_manager.secret_values)
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

        model_input = (config, endpoint_registry, service_provider, model_downloader, docker_service, hardware, services_manager)

        # Register services
        services_manager.register_service(ClaudeService(*model_input))
        services_manager.register_service(CustomService(*model_input))
        services_manager.register_service(DeepSeekService(*model_input))
        services_manager.register_service(DockerModelRunnerService(*model_input))
        services_manager.register_service(FinetuneService(*model_input))
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

        await services_manager.resolve_platform_compatibility()

        # Load functions
        await context.load_services()
        task_manager.add_task_safe(parent_infra.run(), "lifecycle.parent_infra")
    except AppStartError as e:
        logger.error(str(e))
        os._exit(1)
    yield
    # Shutdown: stop all services gracefully if enabled
    logger.debug("Shutdown all containers gracefully")
    shutdown_steps: list[tuple[str, Callable[[], Awaitable[None]]]] = [("drain warning tasks", services_manager.drain_warning_tasks)]
    if config.is_stop_containers_on_shutdown_enabled():
        shutdown_steps.append(("stop services", services_manager.stop_all_services))
    shutdown_steps += [("tracer", tracer.close), ("OTLP logging", otlp_logging.close), ("Sentry", sentry.shutdown)]
    # Each step is isolated so one failure doesn't skip the teardown of the remaining services.
    for name, step in shutdown_steps:
        try:
            await step()
        except Exception:
            logger.exception("Shutdown step '%s' failed, continuing", name)


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
