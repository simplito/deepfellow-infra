# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Tests for server/api/config.py endpoints."""

import asyncio
import json
import logging
from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr
from starlette.testclient import TestClient

from server.api.config import SENTRY_TUNNEL_MAX_BYTES, router
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
from server.error_handlers import register_exception_handlers
from server.utils.sentry import SentryManager


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
    mock.adapter_registry_secret = SecretStr("")
    mock.log_payloads = ""
    mock.container_name_prefix = ""
    mock.container_name_suffix = ""
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
    mock.max_generated_tokens = 131072
    mock.max_context_tokens = 1048576
    mock.max_choices = 16
    mock.max_top_k = 1000
    mock.otel_exporter_otlp_endpoint = "http://localhost:4317"
    mock.otel_tracing_enabled = False
    mock.otel_logging_enabled = False
    mock.sentry_enabled = False
    mock.sentry_dsn = SecretStr("")
    mock.sentry_environment = ""
    mock.ollama_vram_overhead_factor = 1.0
    return mock


@pytest.fixture
def parent_infra() -> MagicMock:
    mock = MagicMock()
    mock.reconfigure = AsyncMock()
    # Must match `config`'s initial values, so `mesh_changed` starts out "unchanged".
    mock.configured_mesh_url = ""
    mock.configured_mesh_key = SecretStr("")
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
    # Must match `config`'s initial values, so `otel_logging_changed` starts out "unchanged".
    mock.configured_enabled = False
    mock.configured_endpoint = "http://localhost:4317"
    return mock


@pytest.fixture(autouse=True)
def sentry_manager(config: MagicMock) -> Generator[SentryManager]:
    manager = SentryManager()
    # Start from `config`'s Sentry settings already applied, as after startup, so `sentry_changed` starts out "unchanged".
    asyncio.run(manager.reconfigure(config))
    with patch("server.api.config.sentry", manager):
        yield manager


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
    register_exception_handlers(app)
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
    assert "not_a_field" in resp.json()["error"]["message"]


def test_update_config_invalid_value_returns_400(client: TestClient, auth_header: dict[str, str]) -> None:
    with patch("server.api.config.dynamic_config.persist_settings") as mock_persist:
        resp = client.put("/admin/config", json={"mcp_sse_max_sessions": "not-an-int"}, headers=auth_header)

    assert resp.status_code == 400
    assert mock_persist.call_count == 0


def test_update_config_invalid_value_error_names_field_and_problem(client: TestClient, auth_header: dict[str, str]) -> None:
    with patch("server.api.config.dynamic_config.persist_settings"):
        resp = client.put("/admin/config", json={"container_name_suffix": "Bad Suffix"}, headers=auth_header)

    assert resp.status_code == 400
    assert resp.json() == {
        "error": {
            "message": "container_name_suffix: String should match pattern '^[a-z0-9_-]*$'",
            "type": "invalid_request_error",
            "param": "container_name_suffix",
            "code": None,
        }
    }


def test_update_config_several_invalid_values_are_joined_in_one_message(client: TestClient, auth_header: dict[str, str]) -> None:
    with patch("server.api.config.dynamic_config.persist_settings"):
        resp = client.put("/admin/config", json={"container_name_suffix": "Bad Suffix", "max_choices": 0}, headers=auth_header)

    assert resp.status_code == 400
    error = resp.json()["error"]
    assert error["message"] == (
        "container_name_suffix: String should match pattern '^[a-z0-9_-]*$'; max_choices: Input should be greater than or equal to 1"
    )
    assert error["param"] == "container_name_suffix"


@pytest.mark.parametrize("suffix", ["bad/name", "a" * 33, "_Prod"])
def test_update_config_rejects_container_name_suffix_docker_disallows(client: TestClient, auth_header: dict[str, str], suffix: str) -> None:
    with patch("server.api.config.dynamic_config.persist_settings") as mock_persist:
        resp = client.put("/admin/config", json={"container_name_suffix": suffix}, headers=auth_header)

    assert resp.status_code == 400
    assert mock_persist.call_count == 0


def test_update_config_rejects_new_container_name_suffix_while_legacy_prefix_set(
    client: TestClient, config: MagicMock, auth_header: dict[str, str]
) -> None:
    """The legacy prefix wins in get_docker_container_name(), so the suffix would be silently ignored."""
    config.container_name_prefix = "df_"

    with patch("server.api.config.dynamic_config.persist_settings") as mock_persist:
        resp = client.put("/admin/config", json={"container_name_suffix": "_myinstance"}, headers=auth_header)

    assert resp.status_code == 400
    assert "DF_CONTAINER_NAME_PREFIX" in resp.json()["error"]["message"]
    assert mock_persist.call_count == 0


@pytest.mark.parametrize(("current_suffix", "new_suffix"), [("_mine", ""), ("_mine", "_mine")])
def test_update_config_allows_clearing_or_resending_suffix_while_legacy_prefix_set(
    client: TestClient, config: MagicMock, auth_header: dict[str, str], current_suffix: str, new_suffix: str
) -> None:
    config.container_name_prefix = "df_"
    config.container_name_suffix = current_suffix

    with patch("server.api.config.dynamic_config.persist_settings") as mock_persist:
        resp = client.put("/admin/config", json={"container_name_suffix": new_suffix}, headers=auth_header)

    assert resp.status_code == 200
    assert mock_persist.call_count == 1


def test_update_config_container_name_suffix_persists_and_applies_change(
    client: TestClient, config: MagicMock, auth_header: dict[str, str]
) -> None:
    with patch("server.api.config.dynamic_config.persist_settings") as mock_persist:
        resp = client.put("/admin/config", json={"container_name_suffix": "_myinstance"}, headers=auth_header)

    assert resp.status_code == 200
    assert mock_persist.call_count == 1
    assert mock_persist.call_args[0][1].container_name_suffix == "_myinstance"
    assert config.container_name_suffix == "_myinstance"

    entries = {e["key"]: e for e in resp.json()["entries"]}
    assert entries["DF_CONTAINER_NAME_SUFFIX"]["value"] == "_myinstance"


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
        mock_tracer.is_active = True
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


def test_update_config_adapter_registry_secret_change_does_not_refresh_downloaders(
    client: TestClient, model_downloader: MagicMock, auth_header: dict[str, str]
) -> None:
    with patch("server.api.config.dynamic_config.persist_settings"):
        resp = client.put("/admin/config", json={"adapter_registry_secret": "new-secret"}, headers=auth_header)

    assert resp.status_code == 200
    model_downloader.create_downloaders.assert_not_called()


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
    config: MagicMock, parent_infra: MagicMock, task_manager: MagicMock, otlp_logging: MagicMock, auth_header: dict[str, str]
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
    app.dependency_overrides[get_otlp_logging] = lambda: otlp_logging
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


def test_update_config_mesh_reconfigure_retry_after_failure_re_attempts_reconfigure(
    client: TestClient, parent_infra: MagicMock, auth_header: dict[str, str]
) -> None:
    """A retry must not silently skip a mesh reconfigure that failed."""
    parent_infra.reconfigure = AsyncMock(side_effect=[RuntimeError("mesh unreachable"), None])

    with patch("server.api.config.dynamic_config.persist_settings"):
        first = client.put("/admin/config", json={"connect_to_mesh_url": "ws://new.example"}, headers=auth_header)
        second = client.put("/admin/config", json={"connect_to_mesh_url": "ws://new.example"}, headers=auth_header)

    assert first.status_code == 502
    assert second.status_code == 200
    assert parent_infra.reconfigure.await_count == 2


def test_update_config_otel_logging_reconfigure_retry_after_failure_re_attempts_reconfigure(
    client: TestClient, otlp_logging: MagicMock, auth_header: dict[str, str]
) -> None:
    """A retry must not silently skip an OTEL logging reconfigure that failed."""
    otlp_logging.reconfigure = AsyncMock(side_effect=[RuntimeError("otlp collector unreachable"), None])

    with patch("server.api.config.dynamic_config.persist_settings"):
        first = client.put("/admin/config", json={"otel_logging_enabled": True}, headers=auth_header)
        second = client.put("/admin/config", json={"otel_logging_enabled": True}, headers=auth_header)

    assert first.status_code == 502
    assert second.status_code == 200
    assert otlp_logging.reconfigure.await_count == 2


def test_update_config_tracer_shutdown_retried_when_still_active_after_prior_failure(
    client: TestClient, config: MagicMock, auth_header: dict[str, str]
) -> None:
    """Tracer shutdown is checked against live provider state, not a flag."""
    config.otel_tracing_enabled = False
    with patch("server.api.config.dynamic_config.persist_settings"), patch("server.api.config.tracer") as mock_tracer:
        mock_tracer.shutdown = AsyncMock()
        mock_tracer.is_active = True  # a live provider left over from a prior, partially-failed attempt
        resp = client.put("/admin/config", json={"metrics_username": "newname"}, headers=auth_header)

    assert resp.status_code == 200
    assert mock_tracer.shutdown.call_count == 1


def test_update_config_unrelated_change_skips_reconfigure_but_broadcasts_ancestors(
    client: TestClient,
    parent_infra: MagicMock,
    otlp_logging: MagicMock,
    infra_websocket_server: MagicMock,
    auth_header: dict[str, str],
) -> None:
    with patch("server.api.config.dynamic_config.persist_settings"):
        resp = client.put("/admin/config", json={"hugging_face_token": "new-hf-token"}, headers=auth_header)

    assert resp.status_code == 200
    parent_infra.reconfigure.assert_not_awaited()
    assert otlp_logging.reconfigure.call_count == 0
    infra_websocket_server.broadcast_ancestors_to_children.assert_called_once()


# --- sentry settings ---

SENTRY_DSN = "https://publickey@sentry.example.com/42"


def test_get_config_lists_sentry_settings_with_masked_dsn(client: TestClient, config: MagicMock, auth_header: dict[str, str]) -> None:
    config.sentry_dsn = SecretStr(SENTRY_DSN)

    resp = client.get("/admin/config", headers=auth_header)

    entries = {e["key"]: e for e in resp.json()["entries"]}
    assert entries["DF_SENTRY_ENABLED"]["value"] == "False"
    assert entries["DF_SENTRY_ENABLED"]["is_editable"] is True
    assert entries["DF_SENTRY_DSN"]["value"] == "••••••••"
    assert entries["DF_SENTRY_DSN"]["is_secret"] is True
    assert entries["DF_SENTRY_ENVIRONMENT"]["value"] == ""
    assert SENTRY_DSN not in resp.text


def test_reveal_returns_sentry_dsn(client: TestClient, config: MagicMock, auth_header: dict[str, str]) -> None:
    config.sentry_dsn = SecretStr(SENTRY_DSN)

    resp = client.get("/admin/config/DF_SENTRY_DSN/reveal", headers=auth_header)

    assert resp.status_code == 200
    assert resp.json()["value"] == SENTRY_DSN


@pytest.mark.parametrize("dsn", ["not-a-dsn", "https://sentry.example.com/42", "https://key@sentry.example.com:notaport/42"])
@patch("server.api.config.dynamic_config.persist_settings")
def test_update_config_rejects_malformed_sentry_dsn(
    mock_persist_settings: MagicMock, client: TestClient, config: MagicMock, auth_header: dict[str, str], dsn: str
) -> None:
    resp = client.put("/admin/config", json={"sentry_dsn": dsn}, headers=auth_header)

    assert resp.status_code == 400
    assert "Invalid Sentry DSN" in resp.json()["error"]["message"]
    assert mock_persist_settings.call_count == 0
    assert config.sentry_dsn.get_secret_value() == ""


@patch("server.utils.sentry.sentry_sdk.init")
@patch("server.api.config.dynamic_config.persist_settings")
def test_update_config_accepts_empty_sentry_dsn(
    mock_persist_settings: MagicMock, mock_init: MagicMock, client: TestClient, config: MagicMock, auth_header: dict[str, str]
) -> None:
    config.sentry_dsn = SecretStr(SENTRY_DSN)

    resp = client.put("/admin/config", json={"sentry_enabled": True, "sentry_dsn": ""}, headers=auth_header)

    assert resp.status_code == 200
    assert mock_persist_settings.call_count == 1
    assert config.sentry_dsn.get_secret_value() == ""
    assert mock_init.call_count == 0


@patch("server.utils.sentry.sentry_sdk.init")
@patch("server.api.config.dynamic_config.persist_settings")
def test_update_config_ignores_stored_malformed_sentry_dsn_for_unrelated_change(
    mock_persist_settings: MagicMock,
    mock_init: MagicMock,
    client: TestClient,
    config: MagicMock,
    sentry_manager: SentryManager,
    auth_header: dict[str, str],
) -> None:
    config.sentry_enabled = True
    config.sentry_dsn = SecretStr("not-a-dsn")
    asyncio.run(sentry_manager.reconfigure(config))

    resp = client.put("/admin/config", json={"metrics_username": "newname"}, headers=auth_header)

    assert resp.status_code == 200
    assert mock_persist_settings.call_count == 1
    assert mock_init.call_count == 0


@patch("server.utils.sentry.sentry_sdk.init")
@patch("server.api.config.dynamic_config.persist_settings")
def test_update_config_sentry_change_reconfigures_sentry(
    mock_persist_settings: MagicMock, mock_init: MagicMock, client: TestClient, auth_header: dict[str, str]
) -> None:
    resp = client.put("/admin/config", json={"sentry_enabled": True, "sentry_dsn": SENTRY_DSN}, headers=auth_header)

    assert resp.status_code == 200
    assert mock_init.call_count == 1
    assert mock_init.call_args.kwargs["dsn"] == SENTRY_DSN


@patch("server.utils.sentry.sentry_sdk.init")
@patch("server.api.config.dynamic_config.persist_settings")
def test_update_config_unrelated_change_skips_sentry_reconfigure(
    mock_persist_settings: MagicMock, mock_init: MagicMock, client: TestClient, auth_header: dict[str, str]
) -> None:
    client.put("/admin/config", json={"sentry_enabled": True, "sentry_dsn": SENTRY_DSN}, headers=auth_header)

    resp = client.put("/admin/config", json={"metrics_username": "newname"}, headers=auth_header)

    assert resp.status_code == 200
    assert mock_init.call_count == 1


@patch("server.utils.sentry.sentry_sdk.init")
@patch("server.api.config.dynamic_config.persist_settings")
def test_update_config_retries_failed_sentry_reconfigure_on_next_update(
    mock_persist_settings: MagicMock, mock_init: MagicMock, client: TestClient, auth_header: dict[str, str]
) -> None:
    mock_init.side_effect = [RuntimeError("init failed"), None]

    first = client.put("/admin/config", json={"sentry_enabled": True, "sentry_dsn": SENTRY_DSN}, headers=auth_header)
    second = client.put("/admin/config", json={"metrics_username": "newname"}, headers=auth_header)

    assert first.status_code == 502
    assert second.status_code == 200
    assert mock_init.call_count == 2


@pytest.mark.parametrize(
    "updates",
    [{"sentry_enabled": True}, {"sentry_environment": "staging"}],
)
@patch("server.api.config.dynamic_config.persist_settings")
def test_update_config_rejects_enabling_sentry_over_stored_malformed_dsn(
    mock_persist_settings: MagicMock, client: TestClient, config: MagicMock, auth_header: dict[str, str], updates: dict[str, object]
) -> None:
    config.sentry_enabled = "sentry_environment" in updates
    config.sentry_dsn = SecretStr("not-a-dsn")

    resp = client.put("/admin/config", json=updates, headers=auth_header)

    assert resp.status_code == 400
    assert "Invalid Sentry DSN" in resp.json()["error"]["message"]
    assert mock_persist_settings.call_count == 0


@patch("server.utils.sentry.sentry_sdk.init")
@patch("server.api.config.dynamic_config.persist_settings")
def test_update_config_allows_disabling_sentry_with_stored_malformed_dsn(
    mock_persist_settings: MagicMock, mock_init: MagicMock, client: TestClient, config: MagicMock, auth_header: dict[str, str]
) -> None:
    config.sentry_enabled = True
    config.sentry_dsn = SecretStr("not-a-dsn")

    resp = client.put("/admin/config", json={"sentry_enabled": False}, headers=auth_header)

    assert resp.status_code == 200
    assert mock_persist_settings.call_count == 1
    assert config.sentry_enabled is False


@patch("server.utils.sentry.sentry_sdk.init")
@patch("server.api.config.dynamic_config.persist_settings")
def test_update_config_allows_enabling_sentry_without_dsn(
    mock_persist_settings: MagicMock, mock_init: MagicMock, client: TestClient, config: MagicMock, auth_header: dict[str, str]
) -> None:
    resp = client.put("/admin/config", json={"sentry_enabled": True}, headers=auth_header)

    assert resp.status_code == 200
    assert mock_persist_settings.call_count == 1
    assert mock_init.call_count == 0


def test_update_config_rejects_non_positive_generation_limit(client: TestClient, auth_header: dict[str, str]) -> None:
    response = client.put("/admin/config", json={"max_generated_tokens": 0}, headers=auth_header)

    assert response.status_code == 400


def test_update_config_applies_generation_limit(client: TestClient, config: MagicMock, auth_header: dict[str, str]) -> None:
    response = client.put("/admin/config", json={"max_choices": 4}, headers=auth_header)

    assert response.status_code == 200
    assert config.max_choices == 4


VALID_DSN = "https://public-key@o1.ingest.sentry.io/123"


@pytest.fixture
def public_client(config: MagicMock) -> Generator[TestClient]:
    # No `auth_admin` override: the endpoint must answer without any credentials.
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_config] = lambda: config
    with TestClient(app) as c:
        yield c


def _set_sentry(config: MagicMock, *, enabled: bool, dsn: str, environment: str = "staging") -> None:
    config.sentry_enabled = enabled
    config.sentry_dsn = SecretStr(dsn)
    config.sentry_environment = environment


@patch("server.utils.sentry.get_infra_version", return_value="1.2.3")
def test_sentry_browser_config_returns_dsn_without_auth_when_enabled(
    mock_get_infra_version: MagicMock, public_client: TestClient, config: MagicMock
) -> None:
    _set_sentry(config, enabled=True, dsn=VALID_DSN)

    resp = public_client.get("/admin/config/sentry")

    assert resp.status_code == 200
    assert resp.json() == {"enabled": True, "dsn": VALID_DSN, "environment": "staging", "release": "1.2.3"}
    assert mock_get_infra_version.call_count == 1


@patch("server.utils.sentry.get_infra_version", return_value="1.2.3")
def test_sentry_browser_config_hides_dsn_when_disabled(
    mock_get_infra_version: MagicMock, public_client: TestClient, config: MagicMock
) -> None:
    _set_sentry(config, enabled=False, dsn=VALID_DSN)

    resp = public_client.get("/admin/config/sentry")

    assert resp.status_code == 200
    assert resp.json() == {"enabled": False, "dsn": None, "environment": None, "release": None}
    assert VALID_DSN not in resp.text
    assert mock_get_infra_version.call_count == 0


@patch("server.utils.sentry.get_infra_version", return_value="1.2.3")
def test_sentry_browser_config_is_off_with_empty_dsn(
    mock_get_infra_version: MagicMock, public_client: TestClient, config: MagicMock
) -> None:
    _set_sentry(config, enabled=True, dsn="")

    resp = public_client.get("/admin/config/sentry")

    assert resp.status_code == 200
    assert resp.json() == {"enabled": False, "dsn": None, "environment": None, "release": None}
    assert mock_get_infra_version.call_count == 0


@patch("server.utils.sentry.get_infra_version", return_value="1.2.3")
def test_sentry_browser_config_is_off_with_malformed_dsn(
    mock_get_infra_version: MagicMock, public_client: TestClient, config: MagicMock
) -> None:
    _set_sentry(config, enabled=True, dsn="not-a-dsn")

    resp = public_client.get("/admin/config/sentry")

    assert resp.status_code == 200
    assert resp.json() == {"enabled": False, "dsn": None, "environment": None, "release": None}
    assert "not-a-dsn" not in resp.text
    assert mock_get_infra_version.call_count == 0


@patch("server.utils.sentry.get_infra_version", return_value="1.2.3")
def test_sentry_browser_config_defaults_environment_to_production(
    mock_get_infra_version: MagicMock, public_client: TestClient, config: MagicMock
) -> None:
    _set_sentry(config, enabled=True, dsn=VALID_DSN, environment="")

    resp = public_client.get("/admin/config/sentry")

    assert resp.status_code == 200
    assert resp.json()["environment"] == "production"
    assert mock_get_infra_version.call_count == 1


# --- sentry tunnel ---

TUNNEL_PATH = "/admin/config/sentry/tunnel"
VALID_ENVELOPE_URL = "https://o1.ingest.sentry.io/api/123/envelope/"


def _envelope(header: object) -> bytes:
    return json.dumps(header).encode() + b'\n{"type":"event"}\n{"message":"boom"}'


def _upstream_session(status: int = 200, headers: dict[str, str] | None = None) -> MagicMock:
    """Build a mock for aiohttp.ClientSession whose `post` answers with `status` and `headers`."""
    upstream = MagicMock()
    upstream.status = status
    upstream.headers = headers or {}
    request_cm = MagicMock()
    request_cm.__aenter__ = AsyncMock(return_value=upstream)
    request_cm.__aexit__ = AsyncMock(return_value=False)
    session = MagicMock()
    session.post.return_value = request_cm
    session_cm = MagicMock()
    session_cm.__aenter__ = AsyncMock(return_value=session)
    session_cm.__aexit__ = AsyncMock(return_value=False)
    return MagicMock(return_value=session_cm)


def _posted(session_factory: MagicMock) -> MagicMock:
    return session_factory.return_value.__aenter__.return_value.post


def test_sentry_tunnel_forwards_envelope_to_configured_sentry_without_auth(public_client: TestClient, config: MagicMock) -> None:
    _set_sentry(config, enabled=True, dsn=VALID_DSN)
    envelope = _envelope({"dsn": VALID_DSN, "event_id": "abc"})
    session_factory = _upstream_session(429, {"X-Sentry-Rate-Limits": "60:error:key", "Retry-After": "60", "X-Other": "x"})

    with patch("server.api.config.aiohttp.ClientSession", session_factory):
        resp = public_client.post(TUNNEL_PATH, content=envelope)

    assert resp.status_code == 429
    assert resp.headers["X-Sentry-Rate-Limits"] == "60:error:key"
    assert resp.headers["Retry-After"] == "60"
    assert "X-Other" not in resp.headers
    _posted(session_factory).assert_called_once_with(
        VALID_ENVELOPE_URL, data=envelope, headers={"Content-Type": "application/x-sentry-envelope"}, allow_redirects=False
    )


def test_sentry_tunnel_builds_url_from_configured_dsn_with_port_and_path(public_client: TestClient, config: MagicMock) -> None:
    dsn = "https://key@sentry.example.com:9000/sub/42"
    _set_sentry(config, enabled=True, dsn=dsn)
    session_factory = _upstream_session()

    with patch("server.api.config.aiohttp.ClientSession", session_factory):
        resp = public_client.post(TUNNEL_PATH, content=_envelope({"dsn": dsn}))

    assert resp.status_code == 200
    assert _posted(session_factory).call_args.args == ("https://sentry.example.com:9000/sub/api/42/envelope/",)


@pytest.mark.parametrize(
    "envelope",
    [
        _envelope({"event_id": "abc"}),
        _envelope({"dsn": 42}),
        _envelope(["not", "an", "object"]),
        _envelope({"dsn": "not-a-dsn"}),
        _envelope({"dsn": "https://public-key@evil.example.com/123"}),
        _envelope({"dsn": "https://public-key@o1.ingest.sentry.io/999"}),
        _envelope({"dsn": "https://other-key@o1.ingest.sentry.io/123"}),
        _envelope({"dsn": "http://public-key@o1.ingest.sentry.io/123"}),
        _envelope({"dsn": "https://public-key@o1.ingest.sentry.io:8443/123"}),
        b"not json\n{}",
        b"\xff\xfe\n{}",
        b"",
    ],
)
def test_sentry_tunnel_rejects_envelope_not_for_configured_dsn(public_client: TestClient, config: MagicMock, envelope: bytes) -> None:
    _set_sentry(config, enabled=True, dsn=VALID_DSN)
    session_factory = _upstream_session()

    with patch("server.api.config.aiohttp.ClientSession", session_factory):
        resp = public_client.post(TUNNEL_PATH, content=envelope)

    assert resp.status_code == 400
    session_factory.assert_not_called()


@pytest.mark.parametrize(("enabled", "dsn"), [(False, VALID_DSN), (True, ""), (True, "not-a-dsn")])
def test_sentry_tunnel_is_not_found_while_sentry_is_off(public_client: TestClient, config: MagicMock, enabled: bool, dsn: str) -> None:
    _set_sentry(config, enabled=enabled, dsn=dsn)
    session_factory = _upstream_session()

    with patch("server.api.config.aiohttp.ClientSession", session_factory):
        resp = public_client.post(TUNNEL_PATH, content=_envelope({"dsn": VALID_DSN}))

    assert resp.status_code == 404
    session_factory.assert_not_called()


@pytest.mark.parametrize("chunked", [False, True])
def test_sentry_tunnel_rejects_envelope_over_size_limit(public_client: TestClient, config: MagicMock, chunked: bool) -> None:
    _set_sentry(config, enabled=True, dsn=VALID_DSN)
    envelope = _envelope({"dsn": VALID_DSN}) + b"x" * SENTRY_TUNNEL_MAX_BYTES
    session_factory = _upstream_session()

    with patch("server.api.config.aiohttp.ClientSession", session_factory):
        # A generator body is sent chunked, without Content-Length, so only the streamed byte count can stop it.
        content = iter([envelope[i : i + 65536] for i in range(0, len(envelope), 65536)]) if chunked else envelope
        resp = public_client.post(TUNNEL_PATH, content=content)

    assert resp.status_code == 413
    session_factory.assert_not_called()


@pytest.mark.parametrize("error", [aiohttp.ClientConnectionError("refused"), TimeoutError()])
def test_sentry_tunnel_answers_bad_gateway_when_sentry_unreachable(
    public_client: TestClient, config: MagicMock, caplog: pytest.LogCaptureFixture, error: Exception
) -> None:
    _set_sentry(config, enabled=True, dsn=VALID_DSN)
    session_factory = _upstream_session()
    _posted(session_factory).return_value.__aenter__.side_effect = error

    with patch("server.api.config.aiohttp.ClientSession", session_factory), caplog.at_level(logging.WARNING):
        resp = public_client.post(TUNNEL_PATH, content=_envelope({"dsn": VALID_DSN}))

    assert resp.status_code == 502
    assert [record.levelno for record in caplog.records] == [logging.WARNING]


def test_sentry_tunnel_answers_too_many_requests_while_all_slots_busy(public_client: TestClient, config: MagicMock) -> None:
    _set_sentry(config, enabled=True, dsn=VALID_DSN)
    session_factory = _upstream_session()

    with (
        patch("server.api.config._sentry_tunnel_slots", MagicMock(locked=MagicMock(return_value=True))),
        patch("server.api.config.aiohttp.ClientSession", session_factory),
    ):
        resp = public_client.post(TUNNEL_PATH, content=_envelope({"dsn": VALID_DSN}))

    assert resp.status_code == 429
    assert resp.headers["Retry-After"] == "60"
    session_factory.assert_not_called()
