"""Services API."""

from typing import Annotated

from fastapi import APIRouter, Body, Depends, Path, Query

from server.core.dependencies import get_services_manager
from server.dependecies import auth_admin
from server.models.services import (
    InstallServiceIn,
    InstallServiceOut,
    ListAllModelsFilters,
    ListAllModelsOut,
    ListServicesFilters,
    ListServicesOut,
    RetrieveServiceOut,
    UninstallServiceIn,
    UninstallServiceOut,
)
from server.services_manager import ServicesManager

router = APIRouter(prefix="/admin/services", tags=["Services"])


@router.post(
    "/{service_id}",
    summary="Install the service.",
)
async def install_service(
    model: Annotated[InstallServiceIn, Body()],
    service_id: Annotated[str, Path(..., description="The ID of the service to install")],
    services_manager: Annotated[ServicesManager, Depends(get_services_manager)],
    _: Annotated[str, Depends(auth_admin)],
) -> InstallServiceOut:
    """Install the service."""
    await services_manager.install_service(service_id, model)
    return InstallServiceOut(status="OK")


@router.delete(
    "/{service_id}",
    summary="Uninstall the service.",
)
async def uninstall_service(
    model: Annotated[UninstallServiceIn, Body()],
    service_id: Annotated[str, Path(..., description="The ID of the service to uninstall")],
    services_manager: Annotated[ServicesManager, Depends(get_services_manager)],
    _: Annotated[str, Depends(auth_admin)],
) -> UninstallServiceOut:
    """Uninstall the service."""
    await services_manager.uninstall_service(service_id, model)
    return UninstallServiceOut(status="OK")


@router.get(
    "/models",
    summary="List models among all services.",
)
async def list_models(
    filters: Annotated[ListAllModelsFilters, Query()],
    services_manager: Annotated[ServicesManager, Depends(get_services_manager)],
    _: Annotated[str, Depends(auth_admin)],
) -> ListAllModelsOut:
    """List models among all services."""
    return await services_manager.list_models_from_all_services(filters)


@router.get(
    "/{service_id}",
    summary="Retrieve the service.",
)
async def retrieve_service(
    service_id: Annotated[str, Path(..., description="The ID of the service to retrieve")],
    services_manager: Annotated[ServicesManager, Depends(get_services_manager)],
    _: Annotated[str, Depends(auth_admin)],
) -> RetrieveServiceOut:
    """Retrieve the service."""
    return await services_manager.get_service(service_id)


@router.get(
    "",
    summary="List services.",
)
async def list_service(
    filters: Annotated[ListServicesFilters, Query()],
    services_manager: Annotated[ServicesManager, Depends(get_services_manager)],
    _: Annotated[str, Depends(auth_admin)],
) -> ListServicesOut:
    """List services."""
    return await services_manager.list_services(filters)
