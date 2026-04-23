# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Infra settings API."""

from typing import Annotated

from fastapi import APIRouter, Body, Depends

from server.core.dependencies import auth_admin, get_service_provider
from server.models.services import InfraSettingsOut, UpdateInfraSettingsIn
from server.serviceprovider import ServiceProvider

router = APIRouter(prefix="/admin/settings", tags=["Settings"])


@router.get("", summary="Retrieve infra settings.")
async def get_settings(
    service_provider: Annotated[ServiceProvider, Depends(get_service_provider)],
    _: Annotated[str, Depends(auth_admin)],
) -> InfraSettingsOut:
    """Retrieve infra settings."""
    cloud_enabled = await service_provider.get_cloud_enabled()
    return InfraSettingsOut(cloud_enabled=cloud_enabled)


@router.put("", summary="Update infra settings.")
async def update_settings(
    model: Annotated[UpdateInfraSettingsIn, Body()],
    service_provider: Annotated[ServiceProvider, Depends(get_service_provider)],
    _: Annotated[str, Depends(auth_admin)],
) -> InfraSettingsOut:
    """Update infra settings."""
    await service_provider.set_cloud_enabled(model.cloud_enabled)
    return InfraSettingsOut(cloud_enabled=model.cloud_enabled)
