# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for server/core/dependencies.py."""

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBasicCredentials

from server.core.dependencies import (
    auth_admin,
    auth_metrics,
    auth_server,
    get_config,
    get_dependency,
    get_endpoint_registry,
    get_hardware,
    get_infra_websocket_server,
    get_metrics_service,
    get_parent_infra,
    get_service_provider,
    get_services_manager,
)


def _make_request(state_attrs: dict[str, object] | None = None) -> MagicMock:
    request = MagicMock()
    request.app.state = MagicMock(spec=[])
    for name, value in (state_attrs or {}).items():
        setattr(request.app.state, name, value)
    return request


def _bearer(credentials: str) -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=credentials)


def _config(api_key: str = "server-key", admin_key: str = "admin-key") -> MagicMock:
    cfg = MagicMock()
    cfg.infra_api_key.get_secret_value.return_value = api_key
    cfg.infra_admin_api_key.get_secret_value.return_value = admin_key
    return cfg


def test_get_dependency_returns_existing_dependency() -> None:
    sentinel = object()

    req = _make_request({"endpoint_registry": sentinel})

    assert get_dependency(req, "endpoint_registry") is sentinel


def test_get_dependency_raises_runtime_error_when_missing() -> None:
    req = _make_request()

    with pytest.raises(RuntimeError, match="not found in application state"):
        get_dependency(req, "endpoint_registry")


@pytest.mark.parametrize(
    ("state_key", "getter"),
    [
        ("endpoint_registry", get_endpoint_registry),
        ("services_manager", get_services_manager),
        ("service_provider", get_service_provider),
        ("config", get_config),
        ("infra_websocket_server", get_infra_websocket_server),
        ("parent_infra", get_parent_infra),
        ("metrics_service", get_metrics_service),
        ("hardware", get_hardware),
    ],
)
def test_getter_returns_state_value(state_key: str, getter: object) -> None:
    sentinel = object()

    req = _make_request({state_key: sentinel})

    assert getter(req) is sentinel  # type: ignore[operator]


def test_auth_server_returns_credentials_when_valid() -> None:
    result = auth_server(_bearer("server-key"), _config(api_key="server-key"))

    assert result == "server-key"


def test_auth_server_raises_401_when_invalid() -> None:
    with pytest.raises(HTTPException) as exc_info:
        auth_server(_bearer("wrong-key"), _config(api_key="server-key"))

    assert exc_info.value.status_code == 401


def test_auth_admin_returns_credentials_when_valid() -> None:
    result = auth_admin(_bearer("admin-key"), _config(admin_key="admin-key"))

    assert result == "admin-key"


def test_auth_admin_raises_401_when_invalid() -> None:
    with pytest.raises(HTTPException) as exc_info:
        auth_admin(_bearer("wrong-key"), _config(admin_key="admin-key"))

    assert exc_info.value.status_code == 401


@pytest.fixture
def metrics_request() -> MagicMock:
    request = MagicMock()
    request.app.state.config.metrics_username = "user"
    request.app.state.config.metrics_password = "pass"
    return request


@pytest.mark.asyncio
async def test_auth_metrics_passes_when_credentials_match(metrics_request: MagicMock) -> None:
    creds = HTTPBasicCredentials(username="user", password="pass")

    await auth_metrics(metrics_request, creds)  # should not raise


@pytest.mark.parametrize(
    ("username", "password"),
    [
        ("wrong", "pass"),
        ("user", "wrong"),
    ],
)
@pytest.mark.asyncio
async def test_auth_metrics_raises_401_for_invalid_credentials(username: str, password: str, metrics_request: MagicMock) -> None:
    creds = HTTPBasicCredentials(username=username, password=password)

    with pytest.raises(HTTPException) as exc_info:
        await auth_metrics(metrics_request, creds)

    assert exc_info.value.status_code == 401
