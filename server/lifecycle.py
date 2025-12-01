# DeepFellow Software Framework.
# Copyright © 2025 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Lifecycle."""

import logging
import os
import re
import subprocess
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from server.applicationcontext import ApplicationContext
from server.config import ConfigError, load_config
from server.docker import create_docker_service
from server.endpointregistry import EndpointRegistry
from server.portservice import PortService
from server.serviceprovider import ServiceProvider
from server.services.coqui_service import CoquiService
from server.services.custom_service import CustomService
from server.services.googleai_service import GoogleAIService
from server.services.llamapcpp_service import LLamacppService
from server.services.ollama_external_service import OllamaExternalService
from server.services.ollama_service import OllamaService
from server.services.openai_service import OpenAIService
from server.services.sindri_service import SindriService
from server.services.speches_ai_service import SpeachesAIService
from server.services.stable_diffusion_service import StableDiffusionService
from server.services.vllm_service import VllmService
from server.services_manager import ServicesManager
from server.task_manager import TaskManager
from server.utils.exceptions import AppStartError
from server.utils.model_downloader import ModelDownloader
from server.websockets.infra_websocket_server import InfraWebsocketServer
from server.websockets.parent_infra import ParentInfra

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
        except ConfigError as e:
            raise AppStartError(str(e))  # noqa: B904
        except Exception as e:
            raise AppStartError("Config error. Have you created the .env file?") from e

        if app.state.config.docker_subnet:
            check_subnet(app.state.config.docker_subnet)

        app.state.task_manager = task_manager = TaskManager()
        app.state.service_provider = service_provider = ServiceProvider(config)
        app.state.parent_infra = parent_infra = ParentInfra(config, task_manager)
        app.state.services_manager = services_manager = ServicesManager()
        app.state.endpoint_registry = endpoint_registry = EndpointRegistry(config, parent_infra)
        app.state.infra_websocket_server = InfraWebsocketServer(config, parent_infra, endpoint_registry)
        app.state.context = context = ApplicationContext(endpoint_registry, config, service_provider, services_manager)
        app.state.port_service = port_service = PortService()
        app.state.docker_service = docker_service = await create_docker_service(port_service, config)
        app.state.model_downloader = model_downloader = ModelDownloader(app.state.config)

        model_input = (config, endpoint_registry, service_provider, model_downloader, docker_service)

        # Register services
        services_manager.register_service(OllamaService(*model_input))
        services_manager.register_service(OllamaExternalService(*model_input))
        services_manager.register_service(SpeachesAIService(*model_input))
        services_manager.register_service(StableDiffusionService(*model_input))
        services_manager.register_service(LLamacppService(*model_input))
        services_manager.register_service(VllmService(*model_input))
        services_manager.register_service(CustomService(*model_input))
        services_manager.register_service(CoquiService(*model_input))
        services_manager.register_service(SindriService(*model_input))
        services_manager.register_service(OpenAIService(*model_input))
        services_manager.register_service(GoogleAIService(*model_input))

        # Load functions
        await context.load_services()
        task_manager.add_task(parent_infra.run())
    except AppStartError as e:
        logger.error(str(e))  # noqa: TRY400
        os._exit(1)
    yield


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
