# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Warnings API."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path

from server.core.dependencies import auth_admin, get_service_provider, require_services_config_loaded
from server.models.warnings import ListWarningsOut, ServiceWarning
from server.serviceprovider import ServiceProvider

router = APIRouter(prefix="/admin/warnings", tags=["Warnings"])


@router.get("", summary="List recorded warnings.")
async def list_warnings(
    service_provider: Annotated[ServiceProvider, Depends(get_service_provider)],
    _: Annotated[str, Depends(auth_admin)],
) -> ListWarningsOut:
    """List recorded warnings."""
    if service_provider.load_error is not None:
        return ListWarningsOut(list=[], load_error=service_provider.load_error)
    warnings = await service_provider.list_warnings()
    parsed = [ServiceWarning.model_validate(w) for w in warnings]
    parsed.sort(key=lambda w: w.created_at, reverse=True)
    return ListWarningsOut(list=parsed)


@router.delete("", summary="Dismiss all warnings.", dependencies=[Depends(require_services_config_loaded)])
async def dismiss_all_warnings(
    service_provider: Annotated[ServiceProvider, Depends(get_service_provider)],
    _: Annotated[str, Depends(auth_admin)],
) -> None:
    """Dismiss all recorded warnings."""
    await service_provider.dismiss_all_warnings()


@router.delete("/{warning_id}", summary="Dismiss a warning.", dependencies=[Depends(require_services_config_loaded)])
async def dismiss_warning(
    warning_id: Annotated[str, Path(description="Id of the warning to dismiss.")],
    service_provider: Annotated[ServiceProvider, Depends(get_service_provider)],
    _: Annotated[str, Depends(auth_admin)],
) -> None:
    """Dismiss a warning."""
    removed = await service_provider.dismiss_warning(warning_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Warning not found")
