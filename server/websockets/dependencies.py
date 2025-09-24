"""Additional dependencies for websocket endpoints."""

import logging
from typing import Annotated, Any

from fastapi import Depends, Query, WebSocket

from server.config import AppSettings

from .exceptions import AuthError

logger = logging.getLogger("uvicorn.error")


def get_dependency(ws: WebSocket, name: str) -> Any:  # noqa: ANN401
    """Get dependency by given name from application state."""
    dep = getattr(ws.app.state, name, None)
    if dep is None:
        msg = f"Dependency `{name}` not found in application state. Ensure it's set during app lifespan."
        raise RuntimeError(msg)
    return dep


def get_config(ws: WebSocket) -> AppSettings:
    """Get external websocket manager from application state."""
    return get_dependency(ws, "config")


async def ws_auth(
    config: Annotated[AppSettings, Depends(get_config)],
    key: Annotated[str | None, Query()] = None,
) -> str:
    """Auth websocket by api key."""
    if key and key == config.infra_api_key.get_secret_value():
        return key

    raise AuthError
