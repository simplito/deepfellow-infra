"""Core FastAPI dependencies for the application."""

from typing import TYPE_CHECKING, Annotated

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from server.applicationcontext import ApplicationContext
from server.endpointregistry import EndpointRegistry
from server.serviceprovider import ServiceProvider
from server.services_manager import ServicesManager
from server.websockets.manager import ExternalWebsocketManager

if TYPE_CHECKING:
    from server.config import AppSettings

oauth2_scheme = HTTPBearer()


def get_application_context(request: Request) -> ApplicationContext:
    """Get the ApplicationContext instance from application state."""
    application_context: ApplicationContext | None = getattr(request.app.state, "context", None)
    if not application_context:
        raise RuntimeError("ApplicationContext not found in application state. Ensure it's set during app lifespan.")

    return application_context


def get_endpoint_registry(request: Request) -> EndpointRegistry:
    """Get the EndpointRegistry instance from application state."""
    endpoint_registry: EndpointRegistry | None = getattr(request.app.state, "endpoint_registry", None)
    if not endpoint_registry:
        raise RuntimeError("EndpointRegistry not found in application state. Ensure it's set during app lifespan.")

    return endpoint_registry


def get_service_provider(request: Request) -> ServiceProvider:
    """Get the ServiceProvider instance from application state."""
    service_provider: ServiceProvider | None = getattr(request.app.state, "context", None)
    if not service_provider:
        raise RuntimeError("ServiceProvider not found in application state. Ensure it's set during app lifespan.")

    return service_provider


def get_services_manager(request: Request) -> ServicesManager:
    """Get the ServicesManager instance from application state."""
    services_manager: ServicesManager | None = getattr(request.app.state, "services_manager", None)
    if not services_manager:
        raise RuntimeError("ServicesManager not found in application state. Ensure it's set during app lifespan.")

    return services_manager


def get_external_ws_manager(request: Request) -> ExternalWebsocketManager:
    """Get external websocket manager from application state."""
    return request.app.state.external_ws_manager


def auth_server(request: Request, api_key: Annotated[HTTPAuthorizationCredentials, Depends(oauth2_scheme)]) -> str:
    """Authenticate an server key."""
    config: AppSettings = request.app.state.config
    if api_key.credentials == config.api_key.get_secret_value():
        return api_key.credentials

    raise HTTPException(status_code=401, detail="Unauthorized")


def auth_admin(request: Request, api_key: Annotated[HTTPAuthorizationCredentials, Depends(oauth2_scheme)]) -> str:
    """Authenticate administrator."""
    config: AppSettings = request.app.state.config
    if api_key.credentials == config.admin_api_key.get_secret_value():
        return api_key.credentials

    raise HTTPException(status_code=401, detail="Unauthorized")
