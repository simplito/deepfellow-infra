# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Additional dependencies for websocket endpoints."""

import logging
from typing import Any

from fastapi import WebSocket

from server.config import AppSettings
from server.websockets.infra_websocket_server import InfraWebsocketServer

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


def get_infra_websocket_server(websocket: WebSocket) -> InfraWebsocketServer:
    """Get InfraWebsocketServer from application state."""
    return get_dependency(websocket, "infra_websocket_server")
