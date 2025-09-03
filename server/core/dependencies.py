"""Core FastAPI dependencies for the application."""

from fastapi import Request

from server.applicationcontext import ApplicationContext
from server.endpointregistry import EndpointRegistry
from server.serviceprovider import ServiceProvider
from server.services_manager import ServicesManager


def get_application_context(request: Request) -> ApplicationContext:
    """Get the ApplicationContext instance from application state."""
    application_context = getattr(request.app.state, "context", None)
    if not application_context:
        raise RuntimeError("ApplicationContext not found in application state. Ensure it's set during app lifespan.")

    return application_context


def get_endpoint_registry(request: Request) -> EndpointRegistry:
    """Get the EndpointRegistry instance from application state."""
    endpoint_registry = getattr(request.app.state, "endpoint_registry", None)
    if not endpoint_registry:
        raise RuntimeError("EndpointRegistry not found in application state. Ensure it's set during app lifespan.")

    return endpoint_registry


def get_service_provider(request: Request) -> ServiceProvider:
    """Get the ServiceProvider instance from application state."""
    service_provider = getattr(request.app.state, "context", None)
    if not service_provider:
        raise RuntimeError("ServiceProvider not found in application state. Ensure it's set during app lifespan.")

    return service_provider


def get_services_manager(request: Request) -> ServicesManager:
    """Get the ServicesManager instance from application state."""
    services_manager = getattr(request.app.state, "services_manager", None)
    if not services_manager:
        raise RuntimeError("ServicesManager not found in application state. Ensure it's set during app lifespan.")

    return services_manager
