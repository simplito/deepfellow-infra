# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Models API."""

import logging
from typing import Annotated

from fastapi import APIRouter, Body, Depends, Path, Query
from fastapi.responses import JSONResponse, Response

from server.core.dependencies import auth_admin, get_services_manager
from server.models.models import (
    AddCustomModelIn,
    AddCustomModelOut,
    InstallModelIn,
    ListModelsFilters,
    ListModelsOut,
    ModelIdQuery,
    RemoveCustomModelOut,
    RetrieveModelOut,
    UninstallModelIn,
    UninstallModelOut,
)
from server.services_manager import ServicesManager
from server.utils.core import convert_promise_with_progress_to_fastapi_response

logger = logging.getLogger("uvicorn.error")

router = APIRouter(prefix="/admin/services/{service_id}/models", tags=["Services"])


@router.post(
    "/_",
    summary="Install the model from the service.",
)
async def install_model(
    model: Annotated[InstallModelIn, Body()],
    service_id: Annotated[str, Path(description="The ID of the service to use.")],
    query: Annotated[ModelIdQuery, Query()],
    services_manager: Annotated[ServicesManager, Depends(get_services_manager)],
    _: Annotated[str, Depends(auth_admin)],
) -> Response:
    """Install the model from the service."""
    msg = f"{service_id} model installing."
    logger.debug(msg)
    promise = await services_manager.install_model_in_service(service_id, query.model_id, model)
    if model.stream:
        return await convert_promise_with_progress_to_fastapi_response(promise)
    result = await promise.wait()
    return JSONResponse(result.model_dump())


@router.get(
    "/progress",
    summary="Get progress of installing model.",
)
async def get_install_progress_model(
    service_id: Annotated[str, Path(description="The ID of the service to use.")],
    query: Annotated[ModelIdQuery, Query()],
    services_manager: Annotated[ServicesManager, Depends(get_services_manager)],
    _: Annotated[str, Depends(auth_admin)],
) -> Response:
    """Get progress of installing model."""
    promise = await services_manager.get_model_install_progress(service_id, query.model_id)
    return await convert_promise_with_progress_to_fastapi_response(promise)


@router.delete(
    "/_",
    summary="Uninstall the model from the service.",
)
async def uninstall_model(
    model: Annotated[UninstallModelIn, Body()],
    service_id: Annotated[str, Path(description="The ID of the service to use.")],
    query: Annotated[ModelIdQuery, Query()],
    services_manager: Annotated[ServicesManager, Depends(get_services_manager)],
    _: Annotated[str, Depends(auth_admin)],
) -> UninstallModelOut:
    """Uninstall the model from the service."""
    msg = f"{service_id} model uninstalling."
    logger.debug(msg)
    await services_manager.uninstall_model_from_service(service_id, query.model_id, model)
    msg = f"{service_id} model uninstalled."
    logger.info(msg)
    return UninstallModelOut(status="OK")


@router.get(
    "/_",
    summary="Retrieve the model from the service.",
)
async def retrieve_model(
    service_id: Annotated[str, Path(description="The ID of the service to use.")],
    query: Annotated[ModelIdQuery, Query()],
    services_manager: Annotated[ServicesManager, Depends(get_services_manager)],
    _: Annotated[str, Depends(auth_admin)],
) -> RetrieveModelOut:
    """Retrieve the model from the service."""
    return await services_manager.get_model_from_service(service_id, query.model_id)


@router.get(
    "",
    summary="List models in the service.",
)
async def list_model(
    filters: Annotated[ListModelsFilters, Query()],
    service_id: Annotated[str, Path(description="The ID of the service to use.")],
    services_manager: Annotated[ServicesManager, Depends(get_services_manager)],
    _: Annotated[str, Depends(auth_admin)],
) -> ListModelsOut:
    """List models in the service."""
    return await services_manager.list_models_from_service(service_id, filters)


@router.post(
    "/custom",
    summary="Add custom model.",
)
async def add_custom_model(
    model: Annotated[AddCustomModelIn, Body()],
    service_id: Annotated[str, Path(description="The ID of the service to use.")],
    services_manager: Annotated[ServicesManager, Depends(get_services_manager)],
    _: Annotated[str, Depends(auth_admin)],
) -> AddCustomModelOut:
    """List models in the service."""
    custom_model_id = await services_manager.add_custom_model(service_id, model)
    return AddCustomModelOut(custom_model_id=custom_model_id)


@router.delete(
    "/custom/{custom_model_id}",
    summary="Remove custom model.",
)
async def remove_custom_model(
    service_id: Annotated[str, Path(description="The ID of the service to use.")],
    custom_model_id: Annotated[str, Path(description="The ID of the custom model to delete.")],
    services_manager: Annotated[ServicesManager, Depends(get_services_manager)],
    _: Annotated[str, Depends(auth_admin)],
) -> RemoveCustomModelOut:
    """List models in the service."""
    await services_manager.remove_custom_model(service_id, custom_model_id)
    return RemoveCustomModelOut(status="OK")
