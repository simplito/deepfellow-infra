# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Mesh API."""

from typing import Annotated

from fastapi import APIRouter, Body, Depends, HTTPException

from server.core.dependencies import auth_admin, get_infra_websocket_server, get_parent_infra
from server.models.mesh import CheckMeshConnection, ShowMeshInfoOut
from server.websockets.infra_websocket_server import InfraWebsocketServer
from server.websockets.parent_infra import ParentInfra

router = APIRouter(prefix="/admin/mesh", tags=["Services"])


@router.get(
    "/info",
    summary="Show info about the mesh.",
)
async def show_mesh_info(
    infra_websocket_server: Annotated[InfraWebsocketServer, Depends(get_infra_websocket_server)],
    _: Annotated[str, Depends(auth_admin)],
) -> ShowMeshInfoOut:
    """Show info about the mesh."""
    return ShowMeshInfoOut(info=infra_websocket_server.get_mesh_info())


@router.post(
    "/check",
    summary="Check mesh connection.",
)
async def check_mesh_connection(
    model: Annotated[CheckMeshConnection, Body()],
    parent_infra: Annotated[ParentInfra, Depends(get_parent_infra)],
) -> str:
    """Check mesh connection."""
    if not parent_infra.check_subinfra_connection(model):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return "OK"
