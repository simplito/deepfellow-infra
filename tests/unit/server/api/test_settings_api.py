# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for server/api/settings.py endpoints."""

from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock, call

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from server.api.settings import router
from server.core.dependencies import auth_admin, get_service_provider


@pytest.fixture
def service_provider() -> MagicMock:
    sp = MagicMock()
    sp.get_cloud_enabled = AsyncMock(return_value=False)
    sp.set_cloud_enabled = AsyncMock(return_value=None)
    return sp


@pytest.fixture
def client(service_provider: MagicMock) -> Generator[TestClient]:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[auth_admin] = lambda: "test-key"
    app.dependency_overrides[get_service_provider] = lambda: service_provider
    with TestClient(app) as c:
        yield c


def test_get_settings_200(client: TestClient, auth_header: dict[str, str]) -> None:
    resp = client.get("/admin/settings", headers=auth_header)

    assert resp.status_code == 200


def test_get_settings_returns_cloud_enabled_false(client: TestClient, auth_header: dict[str, str]) -> None:
    resp = client.get("/admin/settings", headers=auth_header)

    assert resp.json()["cloud_enabled"] is False


def test_get_settings_returns_cloud_enabled_true(service_provider: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    service_provider.get_cloud_enabled = AsyncMock(return_value=True)

    resp = client.get("/admin/settings", headers=auth_header)

    assert resp.json()["cloud_enabled"] is True


def test_get_settings_calls_get_cloud_enabled(service_provider: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    client.get("/admin/settings", headers=auth_header)

    assert service_provider.get_cloud_enabled.call_count == 1


def test_update_settings_200(client: TestClient, auth_header: dict[str, str]) -> None:
    resp = client.put("/admin/settings", json={"cloud_enabled": True}, headers=auth_header)

    assert resp.status_code == 200


def test_update_settings_returns_updated_value(client: TestClient, auth_header: dict[str, str]) -> None:
    resp = client.put("/admin/settings", json={"cloud_enabled": True}, headers=auth_header)

    assert resp.json()["cloud_enabled"] is True


def test_update_settings_calls_set_cloud_enabled(service_provider: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    client.put("/admin/settings", json={"cloud_enabled": True}, headers=auth_header)

    assert service_provider.set_cloud_enabled.await_count == 1
    assert service_provider.set_cloud_enabled.await_args == call(True)
