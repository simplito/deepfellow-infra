"""Webscoket endpoints."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, WebSocket

from server.core.dependencies import get_infra_websocket_server
from server.websockets.dependencies import ws_auth
from server.websockets.infra_websocket_server import InfraWebsocketServer

logger = logging.getLogger("uvicorn.error")


router = APIRouter()


@router.websocket("/ws")
async def ws(
    ws: WebSocket,
    infra_websocket_server: Annotated[InfraWebsocketServer, Depends(get_infra_websocket_server)],
    _: Annotated[str, Depends(ws_auth)],
) -> None:
    """Websocket endpoint to list active datalinks and inform about connection and disconnection of."""
    await infra_websocket_server.loop(ws)
