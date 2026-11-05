# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Core FastAPI dependencies for the application."""

import asyncio
from typing import Annotated, Any

from fastapi import Depends, HTTPException, Request, WebSocket
from fastapi.security import HTTPAuthorizationCredentials, HTTPBasic, HTTPBasicCredentials, HTTPBearer

from server.config import AppSettings
from server.endpointregistry import EndpointRegistry
from server.metrics import MetricsService
from server.serviceprovider import ServiceProvider
from server.services_manager import ServicesManager
from server.task_manager import TaskManager
from server.utils.core import secrets_equal
from server.utils.hardware import Hardware
from server.utils.model_downloader import ModelDownloader
from server.utils.tracing import OtlpLoggingManager
from server.websockets.infra_websocket_server import InfraWebsocketServer
from server.websockets.parent_infra_group import ParentInfraGroup

oauth2_scheme = HTTPBearer()
basic_security = HTTPBasic()


def get_dependency(request: Request | WebSocket, name: str) -> Any:  # noqa: ANN401
    """Get dependency by given name from application state."""
    dep = getattr(request.app.state, name, None)
    if dep is None:
        msg = f"Dependency `{name}` not found in application state. Ensure it's set during app lifespan."
        raise RuntimeError(msg)
    return dep


def get_endpoint_registry(request: Request) -> EndpointRegistry:
    """Get EndpointRegistry instance from application state."""
    return get_dependency(request, "endpoint_registry")


def get_services_manager(request: Request) -> ServicesManager:
    """Get ServicesManager instance from application state."""
    return get_dependency(request, "services_manager")


def get_service_provider(request: Request) -> ServiceProvider:
    """Get ServiceProvider instance from application state."""
    return get_dependency(request, "service_provider")


def get_config(request: Request) -> AppSettings:
    """Get AppSettings from application state."""
    return get_dependency(request, "config")


def get_config_lock(request: Request) -> asyncio.Lock:
    """Get the config.json read-merge-write lock from application state."""
    return get_dependency(request, "config_lock")


def get_infra_websocket_server(request: Request) -> InfraWebsocketServer:
    """Get InfraWebsocketServer from application state."""
    return get_dependency(request, "infra_websocket_server")


def get_parent_infra(request: Request) -> ParentInfraGroup:
    """Get parent infra from application state."""
    return get_dependency(request, "parent_infra")


def get_hardware(request: Request) -> Hardware:
    """Get Hardware instance from application state."""
    return get_dependency(request, "hardware")


def get_metrics_service(request: Request) -> MetricsService:
    """Get MetricsService instance from application state."""
    return get_dependency(request, "metrics_service")


def get_task_manager(request: Request) -> TaskManager:
    """Get TaskManager instance from application state."""
    return get_dependency(request, "task_manager")


def get_otlp_logging(request: Request) -> OtlpLoggingManager:
    """Get OtlpLoggingManager instance from application state."""
    return get_dependency(request, "otlp_logging")


def get_model_downloader(request: Request) -> ModelDownloader:
    """Get ModelDownloader instance from application state."""
    return get_dependency(request, "model_downloader")


def auth_server(
    api_key: Annotated[HTTPAuthorizationCredentials, Depends(oauth2_scheme)],
    config: Annotated[AppSettings, Depends(get_config)],
) -> str:
    """Authenticate an server key."""
    if secrets_equal(api_key.credentials, config.infra_api_key.get_secret_value()):
        return api_key.credentials

    raise HTTPException(status_code=401, detail="Unauthorized")


def auth_admin(
    api_key: Annotated[HTTPAuthorizationCredentials, Depends(oauth2_scheme)],
    config: Annotated[AppSettings, Depends(get_config)],
) -> str:
    """Authenticate administrator."""
    if secrets_equal(api_key.credentials, config.infra_admin_api_key.get_secret_value()):
        return api_key.credentials

    raise HTTPException(status_code=401, detail="Unauthorized")


def require_services_config_loaded(
    service_provider: Annotated[ServiceProvider, Depends(get_service_provider)],
    _: Annotated[str, Depends(auth_admin)],
) -> None:
    """Reject the request with 503 when services.json could not be read at startup.

    Depends on `auth_admin` itself, so the error, which names the file path, never reaches an unauthenticated caller.
    """
    if service_provider.load_error is not None:
        raise HTTPException(status_code=503, detail=service_provider.load_error)


async def auth_metrics(
    request: Request,
    auth: Annotated[HTTPBasicCredentials, Depends(basic_security)],
) -> None:
    """Authenticate using HTTP Basic Auth."""
    config: AppSettings = request.app.state.config
    # Evaluate both comparisons so the response time doesn't reveal whether the username matched.
    username_ok = secrets_equal(auth.username, config.metrics_username)
    password_ok = secrets_equal(auth.password, config.metrics_password.get_secret_value())
    if username_ok and password_ok:
        return

    raise HTTPException(401, "Not Authenticated.")
