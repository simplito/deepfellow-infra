# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Tests for server/api/config.py endpoints."""

import asyncio
from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr
from starlette.testclient import TestClient

from server.api.config import router
from server.config import AppSettings
from server.core.dependencies import (
    auth_admin,
    get_config,
    get_config_lock,
    get_infra_websocket_server,
    get_model_downloader,
    get_otlp_logging,
    get_parent_infra,
    get_task_manager,
)


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
    mock.external_only = False
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
    mock.share_models_downstream = True
    mock.docker_hub_token = ""
    mock.mcp_sse_session_ttl_seconds = 300
    mock.mcp_sse_max_sessions = 128
    mock.standard_proxy_timeout_seconds = 300
    mock.otel_exporter_otlp_endpoint = "http://localhost:4317"
    mock.otel_tracing_enabled = False
    mock.otel_logging_enabled = False
    mock.ollama_kv_cache_type = "f16"
    mock.ollama_num_parallel = 1
    mock.ollama_vram_overhead_factor = 1.0
    return mock


@pytest.fixture
def parent_infra() -> MagicMock:
    mock = MagicMock()
    mock.reconfigure = AsyncMock()
    return mock


@pytest.fixture
def task_manager() -> MagicMock:
    return MagicMock()


@pytest.fixture
def config_lock() -> asyncio.Lock:
    return asyncio.Lock()


@pytest.fixture
def otlp_logging() -> MagicMock:
    mock = MagicMock()
    mock.reconfigure = AsyncMock()
    return mock


@pytest.fixture
def infra_websocket_server() -> MagicMock:
    return MagicMock()


@pytest.fixture
def model_downloader() -> MagicMock:
    return MagicMock()


@pytest.fixture
def client(
    config: MagicMock,
    parent_infra: MagicMock,
    task_manager: MagicMock,
    config_lock: asyncio.Lock,
    otlp_logging: MagicMock,
    infra_websocket_server: MagicMock,
    model_downloader: MagicMock,
) -> Generator[TestClient]:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[auth_admin] = lambda: "test-key"
    app.dependency_overrides[get_config] = lambda: config
    app.dependency_overrides[get_parent_infra] = lambda: parent_infra
    app.dependency_overrides[get_task_manager] = lambda: task_manager
    app.dependency_overrides[get_config_lock] = lambda: config_lock
    app.dependency_overrides[get_otlp_logging] = lambda: otlp_logging
    app.dependency_overrides[get_infra_websocket_server] = lambda: infra_websocket_server
    app.dependency_overrides[get_model_downloader] = lambda: model_downloader
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


def test_update_config_unknown_field_returns_400(client: TestClient, auth_header: dict[str, str]) -> None:
    resp = client.put("/admin/config", json={"not_a_field": "x"}, headers=auth_header)

    assert resp.status_code == 400
    assert "not_a_field" in resp.json()["detail"]


def test_update_config_invalid_value_returns_400(client: TestClient, auth_header: dict[str, str]) -> None:
    with patch("server.api.config.dynamic_config.persist_settings") as mock_persist:
        resp = client.put("/admin/config", json={"mcp_sse_max_sessions": "not-an-int"}, headers=auth_header)

    assert resp.status_code == 400
    assert mock_persist.call_count == 0


def test_update_config_persists_and_applies_change(client: TestClient, config: MagicMock, auth_header: dict[str, str]) -> None:
    with patch("server.api.config.dynamic_config.persist_settings") as mock_persist:
        resp = client.put("/admin/config", json={"metrics_username": "newname"}, headers=auth_header)

    assert resp.status_code == 200
    assert mock_persist.call_count == 1
    assert mock_persist.call_args[0][1].metrics_username == "newname"
    assert config.metrics_username == "newname"

    entries = {e["key"]: e for e in resp.json()["entries"]}
    assert entries["DF_METRICS_USERNAME"]["value"] == "newname"


def test_update_config_mesh_url_change_triggers_parent_reconfigure(
    client: TestClient,
    config: MagicMock,
    parent_infra: MagicMock,
    task_manager: MagicMock,
    otlp_logging: MagicMock,
    auth_header: dict[str, str],
) -> None:
    with patch("server.api.config.dynamic_config.persist_settings"):
        resp = client.put("/admin/config", json={"connect_to_mesh_url": "ws://new.example"}, headers=auth_header)

    assert resp.status_code == 200
    parent_infra.reconfigure.assert_awaited_once_with(config, task_manager)
    otlp_logging.reconfigure.assert_not_awaited()


def test_update_config_otel_change_triggers_otlp_reconfigure(
    client: TestClient, config: MagicMock, parent_infra: MagicMock, otlp_logging: MagicMock, auth_header: dict[str, str]
) -> None:
    with patch("server.api.config.dynamic_config.persist_settings"):
        resp = client.put("/admin/config", json={"otel_logging_enabled": True}, headers=auth_header)

    assert resp.status_code == 200
    otlp_logging.reconfigure.assert_awaited_once_with(config)
    parent_infra.reconfigure.assert_not_awaited()


def test_update_config_otel_tracing_disabled_shuts_down_tracer(client: TestClient, config: MagicMock, auth_header: dict[str, str]) -> None:
    config.otel_tracing_enabled = True
    with patch("server.api.config.dynamic_config.persist_settings"), patch("server.api.config.tracer") as mock_tracer:
        mock_tracer.shutdown = AsyncMock()
        resp = client.put("/admin/config", json={"otel_tracing_enabled": False}, headers=auth_header)

    assert resp.status_code == 200
    assert mock_tracer.shutdown.call_count == 1


def test_update_config_otel_tracing_still_enabled_does_not_shut_down_tracer(
    client: TestClient, config: MagicMock, auth_header: dict[str, str]
) -> None:
    config.otel_tracing_enabled = True
    with patch("server.api.config.dynamic_config.persist_settings"), patch("server.api.config.tracer") as mock_tracer:
        mock_tracer.shutdown = AsyncMock()
        resp = client.put("/admin/config", json={"otel_exporter_otlp_endpoint": "http://new:4317"}, headers=auth_header)

    assert resp.status_code == 200
    assert mock_tracer.shutdown.call_count == 0


def test_update_config_share_models_downstream_change_broadcasts_ancestors(
    client: TestClient, infra_websocket_server: MagicMock, auth_header: dict[str, str]
) -> None:
    with patch("server.api.config.dynamic_config.persist_settings"):
        resp = client.put("/admin/config", json={"share_models_downstream": False}, headers=auth_header)

    assert resp.status_code == 200
    infra_websocket_server.broadcast_ancestors_to_children.assert_called_once()


def test_update_config_infra_url_change_broadcasts_ancestors(
    client: TestClient, infra_websocket_server: MagicMock, auth_header: dict[str, str]
) -> None:
    with patch("server.api.config.dynamic_config.persist_settings"):
        resp = client.put("/admin/config", json={"infra_url": "http://new-url:8086"}, headers=auth_header)

    assert resp.status_code == 200
    infra_websocket_server.broadcast_ancestors_to_children.assert_called_once()


def test_update_config_name_change_broadcasts_ancestors(
    client: TestClient, infra_websocket_server: MagicMock, auth_header: dict[str, str]
) -> None:
    with patch("server.api.config.dynamic_config.persist_settings"):
        resp = client.put("/admin/config", json={"name": "renamed"}, headers=auth_header)

    assert resp.status_code == 200
    infra_websocket_server.broadcast_ancestors_to_children.assert_called_once()


def test_update_config_no_relevant_change_skips_reconfigure(
    client: TestClient,
    parent_infra: MagicMock,
    otlp_logging: MagicMock,
    model_downloader: MagicMock,
    auth_header: dict[str, str],
) -> None:
    with patch("server.api.config.dynamic_config.persist_settings"):
        resp = client.put("/admin/config", json={"metrics_username": "newname"}, headers=auth_header)

    assert resp.status_code == 200
    parent_infra.reconfigure.assert_not_awaited()
    otlp_logging.reconfigure.assert_not_awaited()
    assert model_downloader.create_downloaders.call_count == 0


def test_update_config_civitai_token_change_refreshes_downloaders(
    client: TestClient, config: MagicMock, model_downloader: MagicMock, auth_header: dict[str, str]
) -> None:
    with patch("server.api.config.dynamic_config.persist_settings"):
        resp = client.put("/admin/config", json={"civitai_token": "new-civitai-token"}, headers=auth_header)

    assert resp.status_code == 200
    model_downloader.create_downloaders.assert_called_once_with(config)


def test_update_config_hugging_face_token_change_refreshes_downloaders(
    client: TestClient, config: MagicMock, model_downloader: MagicMock, auth_header: dict[str, str]
) -> None:
    with patch("server.api.config.dynamic_config.persist_settings"):
        resp = client.put("/admin/config", json={"hugging_face_token": "new-hf-token"}, headers=auth_header)

    assert resp.status_code == 200
    model_downloader.create_downloaders.assert_called_once_with(config)


def test_update_config_adapter_registry_change_refreshes_downloaders(
    client: TestClient, config: MagicMock, model_downloader: MagicMock, auth_header: dict[str, str]
) -> None:
    with patch("server.api.config.dynamic_config.persist_settings"):
        resp = client.put("/admin/config", json={"adapter_registry_url": "http://new-registry"}, headers=auth_header)

    assert resp.status_code == 200
    model_downloader.create_downloaders.assert_called_once_with(config)


def test_update_config_reconfigure_failure_returns_502(
    client: TestClient, config: MagicMock, parent_infra: MagicMock, auth_header: dict[str, str]
) -> None:
    parent_infra.reconfigure.side_effect = RuntimeError("mesh unreachable")

    with patch("server.api.config.dynamic_config.persist_settings") as mock_persist:
        resp = client.put("/admin/config", json={"connect_to_mesh_url": "ws://new.example"}, headers=auth_header)

    assert resp.status_code == 502
    # The write already happened — the failure is in applying it live, not persisting it.
    assert mock_persist.call_count == 1
    assert config.connect_to_mesh_url == "ws://new.example"


def test_update_config_persist_failure_returns_502(client: TestClient, config: MagicMock, auth_header: dict[str, str]) -> None:
    with patch("server.api.config.dynamic_config.persist_settings", side_effect=OSError("disk full")):
        resp = client.put("/admin/config", json={"metrics_username": "newname"}, headers=auth_header)

    assert resp.status_code == 502
    # apply_dynamic_settings must not run when the write itself failed — the disk is the source
    # of truth, and it doesn't have this value.
    assert config.metrics_username != "newname"


@pytest.mark.asyncio
async def test_update_config_serializes_concurrent_writes(
    config: MagicMock, parent_infra: MagicMock, task_manager: MagicMock, auth_header: dict[str, str]
) -> None:
    """Two concurrent PUTs must not interleave their read-merge-write critical sections.

    `parent_infra.reconfigure()` is the one genuine `await` point inside the lock's scope, so it's
    used here to force a task switch mid-critical-section: while the mesh-change request is
    suspended awaiting a slow `reconfigure()`, a second request racing in should block on the lock
    rather than running its own persist in between.
    """
    persist_count = 0
    # Snapshotted persist_count at the moment reconfigure starts/ends. If the lock holds, the
    # metrics-only request can't sneak its persist in during the mesh request's reconfigure await,
    # so these two snapshots must be equal (both requests use counters, not field content, so the
    # assertion doesn't depend on which of the two concurrent requests happens to run first).
    counts_at_reconfigure: list[int] = []

    async def slow_reconfigure(*_args: object, **_kwargs: object) -> None:
        counts_at_reconfigure.append(persist_count)
        await asyncio.sleep(0.05)
        counts_at_reconfigure.append(persist_count)

    parent_infra.reconfigure = AsyncMock(side_effect=slow_reconfigure)

    shared_lock = asyncio.Lock()
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[auth_admin] = lambda: "test-key"
    app.dependency_overrides[get_config] = lambda: config
    app.dependency_overrides[get_parent_infra] = lambda: parent_infra
    app.dependency_overrides[get_task_manager] = lambda: task_manager
    app.dependency_overrides[get_config_lock] = lambda: shared_lock
    app.dependency_overrides[get_otlp_logging] = lambda: MagicMock()
    app.dependency_overrides[get_infra_websocket_server] = lambda: MagicMock()
    app.dependency_overrides[get_model_downloader] = lambda: MagicMock()

    def record_persist(_config: object, _settings: object) -> None:
        nonlocal persist_count
        persist_count += 1

    with patch("server.api.config.dynamic_config.persist_settings", side_effect=record_persist):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            await asyncio.gather(
                ac.put("/admin/config", json={"connect_to_mesh_url": "ws://new.example"}, headers=auth_header),
                ac.put("/admin/config", json={"metrics_username": "concurrent-name"}, headers=auth_header),
            )

    # asyncio doesn't guarantee which of the two concurrent requests runs first, so the snapshot
    # value itself (1 or 2) isn't fixed — but it must not *change* while reconfigure is in flight,
    # since that would mean the other request's persist snuck in during the lock's critical section.
    assert persist_count == 2
    assert counts_at_reconfigure[0] == counts_at_reconfigure[1]


def test_update_config_preserves_unset_secret_field(client: TestClient, config: MagicMock, auth_header: dict[str, str]) -> None:
    config.mesh_key = SecretStr("existing-mesh-secret")

    with patch("server.api.config.dynamic_config.persist_settings") as mock_persist:
        resp = client.put("/admin/config", json={"metrics_username": "newname"}, headers=auth_header)

    assert resp.status_code == 200
    assert mock_persist.call_args[0][1].mesh_key.get_secret_value() == "existing-mesh-secret"
