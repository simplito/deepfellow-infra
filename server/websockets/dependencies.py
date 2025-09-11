"""Additional dependencies for websocket endpoints."""

import logging
from typing import TYPE_CHECKING, Annotated

from fastapi import Query, WebSocket

from .exceptions import AuthError

if TYPE_CHECKING:
    from server.config import AppSettings

logger = logging.getLogger("uvicorn.error")


async def ws_auth(ws: WebSocket, key: Annotated[str | None, Query()] = None) -> str:
    """Auth websocket by api key."""
    config: AppSettings = ws.app.state.config
    if key and key == config.infra_api_key.get_secret_value():
        return key

    raise AuthError
