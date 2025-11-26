# DeepFellow Software Framework.
# Copyright © 2025 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Services API."""

from typing import Annotated

from fastapi import APIRouter, Body, Depends, Path, Query
from fastapi.responses import JSONResponse, Response

from server.core.dependencies import auth_admin, get_services_manager
from server.models.services import (
    InstallServiceIn,
    ListAllModelsFilters,
    ListAllModelsOut,
    ListServicesFilters,
    ListServicesOut,
    OptionalModelIdQuery,
    RestartDockerContainerOut,
    RetrieveDockerComposeFileOut,
    RetrieveDockerLogsOut,
    RetrieveServiceOut,
    UninstallServiceIn,
    UninstallServiceOut,
)
from server.services_manager import ServicesManager
from server.utils.core import convert_promise_with_progress_to_fastapi_response

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
) -> Response:
    """Install the service."""
    promise = await services_manager.install_service(service_id, model)
    if model.stream:
        return await convert_promise_with_progress_to_fastapi_response(promise)
    result = await promise.wait()
    return JSONResponse(result.model_dump())


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


@router.get(
    "/{service_id}/docker/logs",
    summary="Retrieve docker logs.",
)
async def retrieve_docker_logs(
    service_id: Annotated[str, Path(..., description="The ID of the service")],
    query: Annotated[OptionalModelIdQuery, Query()],
    services_manager: Annotated[ServicesManager, Depends(get_services_manager)],
    _: Annotated[str, Depends(auth_admin)],
) -> RetrieveDockerLogsOut:
    """Retrieve docker logs."""
    logs = await services_manager.get_docker_logs(service_id, query.model_id)
    return RetrieveDockerLogsOut(logs=logs)


@router.get(
    "/{service_id}/docker/compose",
    summary="Retrieve docker compose file.",
)
async def retrieve_docker_compose_file(
    service_id: Annotated[str, Path(..., description="The ID of the service")],
    query: Annotated[OptionalModelIdQuery, Query()],
    services_manager: Annotated[ServicesManager, Depends(get_services_manager)],
    _: Annotated[str, Depends(auth_admin)],
) -> RetrieveDockerComposeFileOut:
    """Retrieve docker logs."""
    compose_file = await services_manager.get_docker_compose_file(service_id, query.model_id)
    return RetrieveDockerComposeFileOut(compose_file=compose_file)


@router.post(
    "/{service_id}/docker/restart",
    summary="Restart docker container.",
)
async def restart_docker_container(
    service_id: Annotated[str, Path(..., description="The ID of the service")],
    query: Annotated[OptionalModelIdQuery, Query()],
    services_manager: Annotated[ServicesManager, Depends(get_services_manager)],
    _: Annotated[str, Depends(auth_admin)],
) -> RestartDockerContainerOut:
    """Retrieve docker logs."""
    await services_manager.restart_docker(service_id, query.model_id)
    return RestartDockerContainerOut(status="OK")
