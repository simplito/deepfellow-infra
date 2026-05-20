# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for server/api/mesh.py endpoints."""

from unittest.mock import MagicMock

from fastapi import FastAPI
from starlette.testclient import TestClient

from server.api.mesh import router
from server.core.dependencies import auth_admin, get_infra_websocket_server, get_parent_infra
from server.models.mesh import MeshInfo

API_KEY = "test-key"
AUTH_HEADER = {"Authorization": f"Bearer {API_KEY}"}


def _make_mesh_info() -> MeshInfo:
    return MeshInfo(connections=[])


def _make_infra_ws_server(mesh_info: MeshInfo | None = None) -> MagicMock:
    svc = MagicMock()
    svc.get_mesh_info.return_value = mesh_info or _make_mesh_info()
    return svc


def _make_parent_infra(connected: bool = True) -> MagicMock:
    infra = MagicMock()
    infra.check_subinfra_connection = MagicMock(return_value=connected)
    return infra


def _make_app(
    ws_server: MagicMock | None = None,
    parent_infra: MagicMock | None = None,
) -> FastAPI:
    if ws_server is None:
        ws_server = _make_infra_ws_server()
    if parent_infra is None:
        parent_infra = _make_parent_infra()
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[auth_admin] = lambda: API_KEY
    app.dependency_overrides[get_infra_websocket_server] = lambda: ws_server
    app.dependency_overrides[get_parent_infra] = lambda: parent_infra
    return app


def test_show_mesh_info_200() -> None:
    with TestClient(_make_app()) as client:
        resp = client.get("/admin/mesh/info", headers=AUTH_HEADER)

    assert resp.status_code == 200


def test_show_mesh_info_returns_mesh_info() -> None:
    info = _make_mesh_info()
    ws = _make_infra_ws_server(mesh_info=info)

    with TestClient(_make_app(ws_server=ws)) as client:
        resp = client.get("/admin/mesh/info", headers=AUTH_HEADER)

    assert resp.json()["info"]["connections"] == []


def test_show_mesh_info_calls_get_mesh_info() -> None:
    ws = _make_infra_ws_server()

    with TestClient(_make_app(ws_server=ws)) as client:
        client.get("/admin/mesh/info", headers=AUTH_HEADER)

    assert ws.get_mesh_info.call_count == 1


def test_check_mesh_connection_200() -> None:
    parent = _make_parent_infra(connected=True)

    with TestClient(_make_app(parent_infra=parent)) as client:
        resp = client.post("/admin/mesh/check", json={"infra_api_key": "tok", "connection_verifier": "v"}, headers=AUTH_HEADER)

    assert resp.status_code == 200
    assert resp.json() == "OK"


def test_check_mesh_connection_401() -> None:
    parent = _make_parent_infra(connected=False)

    with TestClient(_make_app(parent_infra=parent), raise_server_exceptions=False) as client:
        resp = client.post("/admin/mesh/check", json={"infra_api_key": "tok", "connection_verifier": "v"}, headers=AUTH_HEADER)

    assert resp.status_code == 401
