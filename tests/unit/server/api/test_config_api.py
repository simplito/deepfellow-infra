# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for server/api/config.py endpoints."""

from collections.abc import Generator
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from pydantic import SecretStr
from starlette.testclient import TestClient

from server.api.config import router
from server.config import AppSettings
from server.core.dependencies import auth_admin, get_config


@pytest.fixture
def config() -> MagicMock:
    mock = MagicMock(spec=AppSettings)
    mock.name = "test"
    mock.infra_url = "http://localhost:8086"
    mock.infra_admin_api_key = SecretStr("admin-secret")
    mock.mesh_key = SecretStr("mesh-secret")
    mock.infra_api_key = SecretStr("api-secret")
    mock.connect_to_mesh_key = SecretStr("")
    mock.docker_subnet = ""
    mock.storage_dir = ""
    mock.storage_services_dir = ""
    mock.hugging_face_token = SecretStr("")
    mock.civitai_token = SecretStr("")
    mock.adapter_registry_url = ""
    mock.adapter_registry_secret = SecretStr("")
    mock.log_payloads = ""
    mock.container_name_prefix = ""
    mock.compose_prefix = "df_"
    mock.stop_containers_on_shutdown = ""
    mock.metrics_username = ""
    mock.metrics_password = SecretStr("")
    mock.connect_to_mesh_url = ""
    mock.mcp_sse_session_ttl_seconds = 300
    mock.mcp_sse_max_sessions = 128
    mock.otel_exporter_otlp_endpoint = "http://localhost:4317"
    mock.otel_tracing_enabled = False
    mock.otel_logging_enabled = False
    return mock


@pytest.fixture
def client(config: MagicMock) -> Generator[TestClient]:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[auth_admin] = lambda: "test-key"
    app.dependency_overrides[get_config] = lambda: config
    with TestClient(app) as c:
        yield c


def test_get_config_200(client: TestClient, auth_header: dict[str, str]) -> None:
    resp = client.get("/admin/config", headers=auth_header)

    assert resp.status_code == 200


def test_get_config_returns_entries_list(client: TestClient, auth_header: dict[str, str]) -> None:
    resp = client.get("/admin/config", headers=auth_header)

    assert "entries" in resp.json()
    assert isinstance(resp.json()["entries"], list)


def test_get_config_plain_field_is_readable(client: TestClient, auth_header: dict[str, str]) -> None:
    resp = client.get("/admin/config", headers=auth_header)

    entries = {e["key"]: e for e in resp.json()["entries"]}
    assert entries["DF_NAME"]["value"] == "test"
    assert entries["DF_NAME"]["is_secret"] is False


def test_get_config_secret_field_is_masked(client: TestClient, auth_header: dict[str, str]) -> None:
    resp = client.get("/admin/config", headers=auth_header)

    entries = {e["key"]: e for e in resp.json()["entries"]}
    assert entries["DF_INFRA_ADMIN_API_KEY"]["value"] == "••••••••"
    assert entries["DF_INFRA_ADMIN_API_KEY"]["is_secret"] is True


def test_get_config_does_not_expose_secret_plaintext(client: TestClient, auth_header: dict[str, str]) -> None:
    resp = client.get("/admin/config", headers=auth_header)

    body_text = resp.text
    assert "admin-secret" not in body_text
    assert "mesh-secret" not in body_text
    assert "api-secret" not in body_text


def test_get_config_uses_df_prefix_for_keys(client: TestClient, auth_header: dict[str, str]) -> None:
    resp = client.get("/admin/config", headers=auth_header)

    keys = [e["key"] for e in resp.json()["entries"]]
    assert all(k.startswith("DF_") for k in keys)


def test_reveal_returns_plaintext_secret(client: TestClient, auth_header: dict[str, str]) -> None:
    resp = client.get("/admin/config/DF_INFRA_ADMIN_API_KEY/reveal", headers=auth_header)

    assert resp.status_code == 200
    assert resp.json()["value"] == "admin-secret"
    assert resp.json()["key"] == "DF_INFRA_ADMIN_API_KEY"


def test_reveal_is_case_insensitive(client: TestClient, auth_header: dict[str, str]) -> None:
    resp = client.get("/admin/config/df_infra_admin_api_key/reveal", headers=auth_header)

    assert resp.status_code == 200
    assert resp.json()["value"] == "admin-secret"


def test_reveal_unknown_key_returns_404(client: TestClient, auth_header: dict[str, str]) -> None:
    resp = client.get("/admin/config/DF_NONEXISTENT_KEY/reveal", headers=auth_header)

    assert resp.status_code == 404


def test_reveal_key_without_df_prefix_returns_404(client: TestClient, auth_header: dict[str, str]) -> None:
    resp = client.get("/admin/config/INFRA_ADMIN_API_KEY/reveal", headers=auth_header)

    assert resp.status_code == 404


def test_reveal_non_secret_key_returns_400(client: TestClient, auth_header: dict[str, str]) -> None:
    resp = client.get("/admin/config/DF_NAME/reveal", headers=auth_header)

    assert resp.status_code == 400


def test_reveal_returns_500_when_secret_field_has_wrong_runtime_type(config: MagicMock, auth_header: dict[str, str]) -> None:
    config.infra_admin_api_key = "not-a-secret-str"

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[auth_admin] = lambda: "test-key"
    app.dependency_overrides[get_config] = lambda: config
    with TestClient(app) as c:
        resp = c.get("/admin/config/DF_INFRA_ADMIN_API_KEY/reveal", headers=auth_header)

    assert resp.status_code == 500


def test_reveal_returns_empty_string_when_secret_field_is_none(config: MagicMock, auth_header: dict[str, str]) -> None:
    config.infra_admin_api_key = None

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[auth_admin] = lambda: "test-key"
    app.dependency_overrides[get_config] = lambda: config
    with TestClient(app) as c:
        resp = c.get("/admin/config/DF_INFRA_ADMIN_API_KEY/reveal", headers=auth_header)

    assert resp.status_code == 200
    assert resp.json()["value"] == ""
