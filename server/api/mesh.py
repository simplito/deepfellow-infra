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

from fastapi import APIRouter, Depends

from server.core.dependencies import auth_admin, get_infra_websocket_server
from server.models.mesh import ShowMeshInfoOut
from server.websockets.infra_websocket_server import InfraWebsocketServer

router = APIRouter(prefix="/admin/mesh", tags=["Services"])


@router.get(
    "/info",
    summary="Show info about the mesh.",
)
async def show_mesh_info(
    infra_websocket_server: Annotated[InfraWebsocketServer, Depends(get_infra_websocket_server)],
    _: Annotated[str, Depends(auth_admin)],
) -> ShowMeshInfoOut:
    """Install the service."""
    return ShowMeshInfoOut(info=infra_websocket_server.get_mesh_info())
