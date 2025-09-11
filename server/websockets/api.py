"""Webscoket endpoints."""

import logging
from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Depends, WebSocket

from server.websockets.dependencies import ws_auth

if TYPE_CHECKING:
    from server.websockets.subinfra import InternalInfraWsManager

logger = logging.getLogger("uvicorn.error")


router = APIRouter()


@router.websocket("/ws")
async def ws(ws: WebSocket, _: Annotated[str, Depends(ws_auth)]) -> None:
    """Websocket endpoint to list active datalinks and inform about connection and disconnection of."""
    ws_manager: InternalInfraWsManager = ws.app.state.internal_ws_manager
    await ws_manager.loop(ws)
