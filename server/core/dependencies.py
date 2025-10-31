# DeepFellow Software Framework.
# Copyright © 2025 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Core FastAPI dependencies for the application."""

from typing import Annotated, Any

from fastapi import Depends, HTTPException, Request, WebSocket
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from server.config import AppSettings
from server.endpointregistry import EndpointRegistry
from server.services_manager import ServicesManager

oauth2_scheme = HTTPBearer()


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
