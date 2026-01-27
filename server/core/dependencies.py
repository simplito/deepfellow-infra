# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Core FastAPI dependencies for the application."""

from typing import Annotated, Any

from fastapi import Depends, HTTPException, Request, WebSocket
from fastapi.security import HTTPAuthorizationCredentials, HTTPBasic, HTTPBasicCredentials, HTTPBearer

from server.config import AppSettings
from server.endpointregistry import EndpointRegistry
from server.metrics import MetricsService
from server.services_manager import ServicesManager
from server.websockets.infra_websocket_server import InfraWebsocketServer
from server.websockets.parent_infra import ParentInfra

oauth2_scheme = HTTPBearer()
basic_security = HTTPBasic()


def get_dependency(request: Request | WebSocket, name: str) -> Any:  # noqa: ANN401
    """Get dependency by given name from application state."""
    dep = getattr(request.app.state, name, None)
    if dep is None:
        msg = f"Dependency `{name}` not found in application state. Ensure it's set during app lifespan."
        raise RuntimeError(msg)
    return dep


def get_endpoint_registry(request: Request) -> EndpointRegistry:
    """Get EndpointRegistry instance from application state."""
    return get_dependency(request, "endpoint_registry")


def get_services_manager(request: Request) -> ServicesManager:
    """Get ServicesManager instance from application state."""
    return get_dependency(request, "services_manager")


def get_config(request: Request) -> AppSettings:
    """Get AppSettings from application state."""
    return get_dependency(request, "config")


def get_infra_websocket_server(request: Request) -> InfraWebsocketServer:
    """Get InfraWebsocketServer from application state."""
    return get_dependency(request, "infra_websocket_server")


def get_parent_infra(request: Request) -> ParentInfra:
    """Get parent infra from application state."""
    return get_dependency(request, "parent_infra")


def get_metrics_service(request: Request) -> MetricsService:
    """Get MetricsService instance from application state."""
    return get_dependency(request, "metrics_service")


def auth_server(
    api_key: Annotated[HTTPAuthorizationCredentials, Depends(oauth2_scheme)],
    config: Annotated[AppSettings, Depends(get_config)],
) -> str:
    """Authenticate an server key."""
    if api_key.credentials == config.infra_api_key.get_secret_value():
        return api_key.credentials

    raise HTTPException(status_code=401, detail="Unauthorized")


def auth_admin(
    api_key: Annotated[HTTPAuthorizationCredentials, Depends(oauth2_scheme)],
    config: Annotated[AppSettings, Depends(get_config)],
) -> str:
    """Authenticate administrator."""
    if api_key.credentials == config.infra_admin_api_key.get_secret_value():
        return api_key.credentials

    raise HTTPException(status_code=401, detail="Unauthorized")


async def auth_metrics(
    request: Request,
    auth: Annotated[HTTPBasicCredentials, Depends(basic_security)],
) -> None:
    """Authenticate using HTTP Basic Auth."""
    config: AppSettings = request.app.state.config
    if (auth.username == config.metrics_username) and (auth.password == config.metrics_password):
        return

    raise HTTPException(401, "Not Authenticated.")
