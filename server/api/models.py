# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Models API."""

from typing import Annotated

from fastapi import APIRouter, Body, Depends, Path, Query, Request
from fastapi.responses import JSONResponse, Response

from server.core.dependencies import auth_admin, get_services_manager, require_services_config_loaded
from server.models.models import (
    AddCustomModelIn,
    AddCustomModelOut,
    CancelModelInstallOut,
    DuplicateSpecOut,
    EditModelOut,
    InstallModelIn,
    ListModelsFilters,
    ListModelsOut,
    ModelIdQuery,
    RemoveCustomModelOut,
    RetrieveModelOut,
    SyncModelsOut,
    UninstallModelIn,
    UninstallModelOut,
    UpdateCustomModelOut,
)
from server.services_manager import ServicesManager
from server.utils.core import convert_promise_with_progress_to_fastapi_response
from server.utils.logger import logger
from server.utils.tracing import tracer

router = APIRouter(prefix="/admin/services/{service_id}/models", tags=["Services"])


@router.post(
    "/_",
    summary="Install the model from the service.",
    dependencies=[Depends(require_services_config_loaded)],
)
@tracer.trace_request()
async def install_model(
    request: Request,  # noqa: ARG001 needed for tracer
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


@router.post(
    "/cancel",
    summary="Cancel an in-progress model install.",
)
@tracer.trace_request()
async def cancel_model_install(
    request: Request,  # noqa: ARG001 needed for tracer
    service_id: Annotated[str, Path(description="The ID of the service to use.")],
    query: Annotated[ModelIdQuery, Query()],
    services_manager: Annotated[ServicesManager, Depends(get_services_manager)],
    _: Annotated[str, Depends(auth_admin)],
) -> CancelModelInstallOut:
    """Cancel an in-progress model install, stopping the underlying Docker image pull."""
    msg = f"{service_id} model install cancelling."
    logger.debug(msg)
    await services_manager.cancel_model_install(service_id, query.model_id)
    msg = f"{service_id} model install cancelled."
    logger.info(msg)
    return CancelModelInstallOut(status="OK")


@router.delete(
    "/_",
    summary="Uninstall the model from the service.",
    dependencies=[Depends(require_services_config_loaded)],
)
@tracer.trace_request()
async def uninstall_model(
    request: Request,  # noqa: ARG001 needed for tracer
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
    dependencies=[Depends(require_services_config_loaded)],
)
@tracer.trace_request()
async def add_custom_model(
    request: Request,  # noqa: ARG001 needed for tracer
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
    dependencies=[Depends(require_services_config_loaded)],
)
@tracer.trace_request()
async def remove_custom_model(
    request: Request,  # noqa: ARG001 needed for tracer
    service_id: Annotated[str, Path(description="The ID of the service to use.")],
    custom_model_id: Annotated[str, Path(description="The ID of the custom model to delete.")],
    services_manager: Annotated[ServicesManager, Depends(get_services_manager)],
    _: Annotated[str, Depends(auth_admin)],
) -> RemoveCustomModelOut:
    """Remove custom model from the service."""
    await services_manager.remove_custom_model(service_id, custom_model_id)
    return RemoveCustomModelOut(status="OK")


@router.put(
    "/custom/{custom_model_id}",
    summary="Update custom model.",
    dependencies=[Depends(require_services_config_loaded)],
)
@tracer.trace_request()
async def update_custom_model(
    request: Request,  # noqa: ARG001 needed for tracer
    model: Annotated[AddCustomModelIn, Body()],
    service_id: Annotated[str, Path(description="The ID of the service to use.")],
    custom_model_id: Annotated[str, Path(description="The ID of the custom model to update.")],
    services_manager: Annotated[ServicesManager, Depends(get_services_manager)],
    _: Annotated[str, Depends(auth_admin)],
) -> UpdateCustomModelOut:
    """Update a custom model definition."""
    await services_manager.update_custom_model(service_id, custom_model_id, model)
    return UpdateCustomModelOut(status="OK")


@router.post(
    "/custom/{custom_model_id}/edit",
    summary="Edit a custom model, uninstalling and reinstalling it automatically if it's installed.",
    dependencies=[Depends(require_services_config_loaded)],
)
@tracer.trace_request()
async def edit_custom_model(
    request: Request,  # noqa: ARG001 needed for tracer
    model: Annotated[AddCustomModelIn, Body()],
    service_id: Annotated[str, Path(description="The ID of the service to use.")],
    custom_model_id: Annotated[str, Path(description="The ID of the custom model to edit.")],
    services_manager: Annotated[ServicesManager, Depends(get_services_manager)],
    _: Annotated[str, Depends(auth_admin)],
) -> EditModelOut:
    """Edit a custom model's definition without requiring a manual uninstall first.

    Unlike PUT .../custom/{custom_model_id}, which rejects the request while the model is installed,
    this uninstalls it, applies the new definition, and reinstalls it with its previous install options.
    """
    promise = await services_manager.edit_model(service_id, custom_model_id, model)
    if promise is None:
        return EditModelOut(status="OK", reinstalled=False)
    await promise.wait()
    return EditModelOut(status="OK", reinstalled=True)


@router.post(
    "/_/edit",
    summary="Edit a model's install-time options only (prefix/envs/headers), reinstalling it if needed.",
    dependencies=[Depends(require_services_config_loaded)],
)
@tracer.trace_request()
async def edit_model_options(
    request: Request,  # noqa: ARG001 needed for tracer
    model: Annotated[InstallModelIn, Body()],
    service_id: Annotated[str, Path(description="The ID of the service to use.")],
    query: Annotated[ModelIdQuery, Query()],
    services_manager: Annotated[ServicesManager, Depends(get_services_manager)],
    _: Annotated[str, Depends(auth_admin)],
) -> EditModelOut:
    """Edit a model's install-time options only.

    Options are e.g. prefix, envs, headers, with no persisted custom model definition involved - for
    catalog models. Uninstalls the model if installed, then reinstalls it with the new options.
    """
    was_installed, promise = await services_manager.edit_model_install_options(service_id, query.model_id, model)
    await promise.wait()
    return EditModelOut(status="OK", reinstalled=was_installed)


@router.get(
    "/_/duplicate-spec",
    summary="Get a full add-model spec synthesized from an existing model, for duplication.",
)
async def get_duplicate_spec(
    service_id: Annotated[str, Path(description="The ID of the service to use.")],
    query: Annotated[ModelIdQuery, Query()],
    services_manager: Annotated[ServicesManager, Depends(get_services_manager)],
    _: Annotated[str, Depends(auth_admin)],
) -> DuplicateSpecOut:
    """Return a spec for duplicating the given model.

    Its stored definition if custom-backed, or one synthesized from the live registered model if it's
    a catalog model. Submit the result (with a new id) to the existing add-custom-model endpoint to
    create the duplicate.
    """
    spec = await services_manager.get_duplicate_spec(service_id, query.model_id)
    return DuplicateSpecOut(spec=spec)


@router.post(
    "/sync",
    summary="Trigger immediate model sync.",
    dependencies=[Depends(require_services_config_loaded)],
)
@tracer.trace_request()
async def sync_models(
    request: Request,  # noqa: ARG001 needed for tracer
    service_id: Annotated[str, Path(description="The ID of the service to sync.")],
    services_manager: Annotated[ServicesManager, Depends(get_services_manager)],
    _: Annotated[str, Depends(auth_admin)],
) -> SyncModelsOut:
    """Trigger an immediate model sync for the service instance."""
    await services_manager.sync_models_in_service(service_id)
    return SyncModelsOut(status="OK")
