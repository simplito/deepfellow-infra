# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Webscoket endpoints."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, WebSocket

from server.websockets.dependencies import get_infra_websocket_server
from server.websockets.infra_websocket_server import InfraWebsocketServer

logger = logging.getLogger("uvicorn.error")


router = APIRouter()


@router.websocket("/ws")
async def ws(
    ws: WebSocket,
    infra_websocket_server: Annotated[InfraWebsocketServer, Depends(get_infra_websocket_server)],
) -> None:
    """Websocket endpoint."""
    await infra_websocket_server.connect(ws)
