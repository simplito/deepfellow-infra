# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Tests for server/api/models.py endpoints."""

from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from starlette.testclient import TestClient

from server.api.models import router
from server.core.dependencies import auth_admin, get_config, get_service_provider, get_services_manager
from server.error_handlers import register_exception_handlers
from server.models.models import (
    InstallModelOut,
    ListModelsOut,
    ModelSpecification,
    RetrieveModelOut,
)

SERVICE_ID = "ollama"
MODEL_ID = "llama3"
ADD_CUSTOM_BODY = {"spec": {"fields": []}}


@pytest.fixture
def retrieve_model_out() -> RetrieveModelOut:
    return RetrieveModelOut(
        id=MODEL_ID,
        service=SERVICE_ID,
        type="llm",
        installed=True,
        downloaded=True,
        size="4GB",
        spec=ModelSpecification(fields=[]),
        has_docker=False,
    )


@pytest.fixture
def services_manager(retrieve_model_out: RetrieveModelOut) -> MagicMock:
    manager = MagicMock()
    promise = MagicMock()
    promise.wait = AsyncMock(return_value=InstallModelOut(status="OK", details="done"))
    manager.install_model_in_service = AsyncMock(return_value=promise)
    manager.get_model_install_progress = AsyncMock(return_value=promise)
    manager.uninstall_model_from_service = AsyncMock(return_value=None)
    manager.get_model_from_service = AsyncMock(return_value=retrieve_model_out)
    manager.list_models_from_service = AsyncMock(return_value=ListModelsOut(list=[]))
    manager.add_custom_model = AsyncMock(return_value="custom-123")
    manager.remove_custom_model = AsyncMock(return_value=None)
    manager.update_custom_model = AsyncMock(return_value=None)
    manager.sync_models_in_service = AsyncMock(return_value=None)
    manager.cancel_model_install = AsyncMock(return_value=None)
    manager.edit_model = AsyncMock(return_value=promise)
    manager.edit_model_install_options = AsyncMock(return_value=(True, promise))
    manager.get_duplicate_spec = AsyncMock(return_value={"id": MODEL_ID, "image": "img:latest"})
    return manager


@pytest.fixture
def service_provider() -> MagicMock:
    provider = MagicMock()
    provider.load_error = None
    return provider


@pytest.fixture
def client(services_manager: MagicMock, service_provider: MagicMock) -> Generator[TestClient]:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(router)
    app.dependency_overrides[auth_admin] = lambda: "test-key"
    app.dependency_overrides[get_services_manager] = lambda: services_manager
    app.dependency_overrides[get_service_provider] = lambda: service_provider
    with TestClient(app) as c:
        yield c


INSTALL_BODY = {"stream": False, "ignore_warnings": False}


def test_install_model_returns_200(client: TestClient, auth_header: dict[str, str]) -> None:
    resp = client.post(f"/admin/services/{SERVICE_ID}/models/_", json=INSTALL_BODY, params={"model_id": MODEL_ID}, headers=auth_header)

    assert resp.status_code == 200


def test_install_model_calls_manager(services_manager: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    client.post(f"/admin/services/{SERVICE_ID}/models/_", json=INSTALL_BODY, params={"model_id": MODEL_ID}, headers=auth_header)

    assert services_manager.install_model_in_service.call_count == 1


@pytest.mark.parametrize(
    ("service_id", "model_id", "arg_idx", "expected"),
    [
        ("my-svc", MODEL_ID, 0, "my-svc"),
        (SERVICE_ID, "my-model", 1, "my-model"),
    ],
)
def test_install_model_passes_ids(
    service_id: str,
    model_id: str,
    arg_idx: int,
    expected: str,
    services_manager: MagicMock,
    client: TestClient,
    auth_header: dict[str, str],
) -> None:
    client.post(f"/admin/services/{service_id}/models/_", json=INSTALL_BODY, params={"model_id": model_id}, headers=auth_header)

    assert services_manager.install_model_in_service.call_args.args[arg_idx] == expected


def test_install_model_returns_ok_status(client: TestClient, auth_header: dict[str, str]) -> None:
    resp = client.post(f"/admin/services/{SERVICE_ID}/models/_", json=INSTALL_BODY, params={"model_id": MODEL_ID}, headers=auth_header)

    assert resp.json()["status"] == "OK"


def test_install_model_stream_returns_streaming_response(client: TestClient, auth_header: dict[str, str]) -> None:
    streaming = StreamingResponse(iter([b"data"]), media_type="text/event-stream")

    with patch(
        "server.api.models.convert_promise_with_progress_to_fastapi_response",
        new=AsyncMock(return_value=streaming),
    ):
        resp = client.post(
            f"/admin/services/{SERVICE_ID}/models/_",
            json={"stream": True},
            params={"model_id": MODEL_ID},
            headers=auth_header,
        )

    assert resp.status_code == 200


def test_get_install_progress_returns_200(client: TestClient, auth_header: dict[str, str]) -> None:
    streaming = StreamingResponse(iter([b"data"]), media_type="text/event-stream")

    with patch(
        "server.api.models.convert_promise_with_progress_to_fastapi_response",
        new=AsyncMock(return_value=streaming),
    ):
        resp = client.get(f"/admin/services/{SERVICE_ID}/models/progress", params={"model_id": MODEL_ID}, headers=auth_header)

    assert resp.status_code == 200


def test_get_install_progress_calls_manager(services_manager: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    streaming = StreamingResponse(iter([b"data"]), media_type="text/event-stream")

    with patch(
        "server.api.models.convert_promise_with_progress_to_fastapi_response",
        new=AsyncMock(return_value=streaming),
    ):
        client.get("/admin/services/my-svc/models/progress", params={"model_id": MODEL_ID}, headers=auth_header)

    assert services_manager.get_model_install_progress.await_count == 1
    assert services_manager.get_model_install_progress.await_args == call("my-svc", MODEL_ID)


def test_cancel_model_install_returns_ok_status(client: TestClient, auth_header: dict[str, str]) -> None:
    resp = client.post(f"/admin/services/{SERVICE_ID}/models/cancel", params={"model_id": MODEL_ID}, headers=auth_header)

    assert resp.status_code == 200
    assert resp.json()["status"] == "OK"


def test_cancel_model_install_calls_manager(services_manager: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    client.post("/admin/services/my-svc/models/cancel", params={"model_id": MODEL_ID}, headers=auth_header)

    assert services_manager.cancel_model_install.await_count == 1
    assert services_manager.cancel_model_install.await_args == call("my-svc", MODEL_ID)


UNINSTALL_BODY = {"purge": False}


def test_uninstall_model_returns_200(client: TestClient, auth_header: dict[str, str]) -> None:
    resp = client.request(
        "DELETE", f"/admin/services/{SERVICE_ID}/models/_", json=UNINSTALL_BODY, params={"model_id": MODEL_ID}, headers=auth_header
    )

    assert resp.status_code == 200


def test_uninstall_model_calls_manager(services_manager: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    client.request(
        "DELETE", f"/admin/services/{SERVICE_ID}/models/_", json=UNINSTALL_BODY, params={"model_id": MODEL_ID}, headers=auth_header
    )

    assert services_manager.uninstall_model_from_service.call_count == 1


@pytest.mark.parametrize(
    ("service_id", "model_id", "arg_idx", "expected"),
    [
        ("my-svc", MODEL_ID, 0, "my-svc"),
        (SERVICE_ID, "my-model", 1, "my-model"),
    ],
)
def test_uninstall_model_passes_ids(
    service_id: str,
    model_id: str,
    arg_idx: int,
    expected: str,
    services_manager: MagicMock,
    client: TestClient,
    auth_header: dict[str, str],
) -> None:
    client.request(
        "DELETE", f"/admin/services/{service_id}/models/_", json=UNINSTALL_BODY, params={"model_id": model_id}, headers=auth_header
    )

    assert services_manager.uninstall_model_from_service.call_args.args[arg_idx] == expected


def test_uninstall_model_passes_purge_flag(services_manager: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    client.request(
        "DELETE", f"/admin/services/{SERVICE_ID}/models/_", json={"purge": True}, params={"model_id": MODEL_ID}, headers=auth_header
    )

    body_arg = services_manager.uninstall_model_from_service.call_args.args[2]
    assert body_arg.purge is True


def test_uninstall_model_returns_ok_status(client: TestClient, auth_header: dict[str, str]) -> None:
    resp = client.request(
        "DELETE", f"/admin/services/{SERVICE_ID}/models/_", json=UNINSTALL_BODY, params={"model_id": MODEL_ID}, headers=auth_header
    )

    assert resp.json()["status"] == "OK"


def test_retrieve_model_returns_200(client: TestClient, auth_header: dict[str, str]) -> None:
    resp = client.get(f"/admin/services/{SERVICE_ID}/models/_", params={"model_id": MODEL_ID}, headers=auth_header)

    assert resp.status_code == 200


def test_retrieve_model_calls_manager(services_manager: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    client.get(f"/admin/services/{SERVICE_ID}/models/_", params={"model_id": MODEL_ID}, headers=auth_header)

    assert services_manager.get_model_from_service.await_count == 1
    assert services_manager.get_model_from_service.await_args == call(SERVICE_ID, MODEL_ID)


def test_retrieve_model_passes_service_id(services_manager: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    client.get("/admin/services/my-svc/models/_", params={"model_id": MODEL_ID}, headers=auth_header)

    assert services_manager.get_model_from_service.call_args.args[0] == "my-svc"


def test_retrieve_model_returns_model_data(
    retrieve_model_out: RetrieveModelOut,
    services_manager: MagicMock,
    client: TestClient,
    auth_header: dict[str, str],
) -> None:
    services_manager.get_model_from_service = AsyncMock(return_value=retrieve_model_out)

    resp = client.get(f"/admin/services/{SERVICE_ID}/models/_", params={"model_id": MODEL_ID}, headers=auth_header)

    assert resp.json()["id"] == "llama3"


def test_list_models_returns_200(client: TestClient, auth_header: dict[str, str]) -> None:
    resp = client.get(f"/admin/services/{SERVICE_ID}/models", headers=auth_header)
    assert resp.status_code == 200


def test_list_models_calls_manager(services_manager: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    client.get(f"/admin/services/{SERVICE_ID}/models", headers=auth_header)

    services_manager.list_models_from_service.assert_awaited_once()


def test_list_models_passes_service_id(services_manager: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    client.get("/admin/services/my-svc/models", headers=auth_header)

    assert services_manager.list_models_from_service.call_args.args[0] == "my-svc"


def test_list_models_returns_empty_list_by_default(client: TestClient, auth_header: dict[str, str]) -> None:
    resp = client.get(f"/admin/services/{SERVICE_ID}/models", headers=auth_header)

    assert resp.json()["list"] == []


def test_list_models_passes_installed_filter(services_manager: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    client.get(f"/admin/services/{SERVICE_ID}/models?installed=true", headers=auth_header)
    filters = services_manager.list_models_from_service.call_args.args[1]

    assert filters.installed is True


def test_add_custom_model_returns_200(client: TestClient, auth_header: dict[str, str]) -> None:
    resp = client.post(f"/admin/services/{SERVICE_ID}/models/custom", json=ADD_CUSTOM_BODY, headers=auth_header)

    assert resp.status_code == 200


def test_add_custom_model_calls_manager(services_manager: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    client.post(f"/admin/services/{SERVICE_ID}/models/custom", json=ADD_CUSTOM_BODY, headers=auth_header)

    services_manager.add_custom_model.assert_awaited_once()


def test_add_custom_model_passes_service_id(services_manager: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    client.post("/admin/services/my-svc/models/custom", json=ADD_CUSTOM_BODY, headers=auth_header)

    assert services_manager.add_custom_model.call_args.args[0] == "my-svc"


def test_add_custom_model_returns_custom_model_id(services_manager: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    services_manager.add_custom_model = AsyncMock(return_value="my-custom-id")

    resp = client.post(f"/admin/services/{SERVICE_ID}/models/custom", json=ADD_CUSTOM_BODY, headers=auth_header)

    assert resp.json()["custom_model_id"] == "my-custom-id"


def test_remove_custom_model_returns_200(client: TestClient, auth_header: dict[str, str]) -> None:
    resp = client.request("DELETE", f"/admin/services/{SERVICE_ID}/models/custom/cust-1", headers=auth_header)

    assert resp.status_code == 200


def test_remove_custom_model_calls_manager(services_manager: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    client.request("DELETE", f"/admin/services/{SERVICE_ID}/models/custom/cust-1", headers=auth_header)

    assert services_manager.remove_custom_model.call_count == 1


@pytest.mark.parametrize(
    ("service_id", "custom_model_id", "arg_idx", "expected"),
    [
        ("my-svc", "cust-1", 0, "my-svc"),
        (SERVICE_ID, "my-custom-id", 1, "my-custom-id"),
    ],
)
def test_remove_custom_model_passes_ids(
    service_id: str,
    custom_model_id: str,
    arg_idx: int,
    expected: str,
    services_manager: MagicMock,
    client: TestClient,
    auth_header: dict[str, str],
) -> None:
    client.request("DELETE", f"/admin/services/{service_id}/models/custom/{custom_model_id}", headers=auth_header)

    assert services_manager.remove_custom_model.call_args.args[arg_idx] == expected


def test_remove_custom_model_returns_ok_status(client: TestClient, auth_header: dict[str, str]) -> None:
    resp = client.request("DELETE", f"/admin/services/{SERVICE_ID}/models/custom/cust-1", headers=auth_header)

    assert resp.json()["status"] == "OK"


def test_update_custom_model_returns_200(client: TestClient, auth_header: dict[str, str]) -> None:
    resp = client.put(f"/admin/services/{SERVICE_ID}/models/custom/cust-1", json={"spec": {"name": "x"}}, headers=auth_header)

    assert resp.status_code == 200


def test_update_custom_model_calls_manager(services_manager: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    client.put(f"/admin/services/{SERVICE_ID}/models/custom/cust-1", json={"spec": {"name": "x"}}, headers=auth_header)

    services_manager.update_custom_model.assert_awaited_once()


def test_update_custom_model_passes_ids(services_manager: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    client.put("/admin/services/my-svc/models/custom/my-id", json={"spec": {"name": "x"}}, headers=auth_header)

    assert services_manager.update_custom_model.call_args.args[0] == "my-svc"
    assert services_manager.update_custom_model.call_args.args[1] == "my-id"


def test_update_custom_model_returns_ok_status(client: TestClient, auth_header: dict[str, str]) -> None:
    resp = client.put(f"/admin/services/{SERVICE_ID}/models/custom/cust-1", json={"spec": {"name": "x"}}, headers=auth_header)

    assert resp.json()["status"] == "OK"


def test_edit_custom_model_returns_200(client: TestClient, auth_header: dict[str, str]) -> None:
    resp = client.post(f"/admin/services/{SERVICE_ID}/models/custom/cust-1/edit", json={"spec": {"name": "x"}}, headers=auth_header)

    assert resp.status_code == 200


def test_edit_custom_model_calls_manager_with_ids(services_manager: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    client.post("/admin/services/my-svc/models/custom/my-id/edit", json={"spec": {"name": "x"}}, headers=auth_header)

    assert services_manager.edit_model.call_args.args[0] == "my-svc"
    assert services_manager.edit_model.call_args.args[1] == "my-id"


def test_edit_custom_model_reports_reinstalled_when_promise_returned(
    services_manager: MagicMock, client: TestClient, auth_header: dict[str, str]
) -> None:
    resp = client.post(f"/admin/services/{SERVICE_ID}/models/custom/cust-1/edit", json={"spec": {"name": "x"}}, headers=auth_header)

    assert resp.json() == {"status": "OK", "reinstalled": True}


def test_edit_custom_model_reports_not_reinstalled_when_not_previously_installed(
    services_manager: MagicMock, client: TestClient, auth_header: dict[str, str]
) -> None:
    services_manager.edit_model = AsyncMock(return_value=None)

    resp = client.post(f"/admin/services/{SERVICE_ID}/models/custom/cust-1/edit", json={"spec": {"name": "x"}}, headers=auth_header)

    assert resp.json() == {"status": "OK", "reinstalled": False}


def test_edit_model_options_returns_200(client: TestClient, auth_header: dict[str, str]) -> None:
    resp = client.post(f"/admin/services/{SERVICE_ID}/models/_/edit", json=INSTALL_BODY, params={"model_id": MODEL_ID}, headers=auth_header)

    assert resp.status_code == 200
    assert resp.json() == {"status": "OK", "reinstalled": True}


def test_edit_model_options_reports_not_reinstalled_when_not_previously_installed(
    services_manager: MagicMock, client: TestClient, auth_header: dict[str, str]
) -> None:
    """edit_model_install_options always installs, even for a model that wasn't installed before - the
    response must reflect that as reinstalled=False rather than hardcoding True regardless."""
    promise = MagicMock()
    promise.wait = AsyncMock(return_value=InstallModelOut(status="OK", details="done"))
    services_manager.edit_model_install_options = AsyncMock(return_value=(False, promise))

    resp = client.post(f"/admin/services/{SERVICE_ID}/models/_/edit", json=INSTALL_BODY, params={"model_id": MODEL_ID}, headers=auth_header)

    assert resp.json() == {"status": "OK", "reinstalled": False}


def test_edit_model_options_calls_manager_with_ids(services_manager: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    client.post("/admin/services/my-svc/models/_/edit", json=INSTALL_BODY, params={"model_id": "my-model"}, headers=auth_header)

    assert services_manager.edit_model_install_options.call_args.args[0] == "my-svc"
    assert services_manager.edit_model_install_options.call_args.args[1] == "my-model"


def test_get_duplicate_spec_returns_200(client: TestClient, auth_header: dict[str, str]) -> None:
    resp = client.get(f"/admin/services/{SERVICE_ID}/models/_/duplicate-spec", params={"model_id": MODEL_ID}, headers=auth_header)

    assert resp.status_code == 200
    assert resp.json() == {"spec": {"id": MODEL_ID, "image": "img:latest"}}


def test_get_duplicate_spec_calls_manager_with_ids(services_manager: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    client.get("/admin/services/my-svc/models/_/duplicate-spec", params={"model_id": "my-model"}, headers=auth_header)

    assert services_manager.get_duplicate_spec.call_args.args[0] == "my-svc"
    assert services_manager.get_duplicate_spec.call_args.args[1] == "my-model"


def test_get_duplicate_spec_not_found_propagates_404(services_manager: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    services_manager.get_duplicate_spec = AsyncMock(side_effect=HTTPException(400, "Model not found"))

    resp = client.get(f"/admin/services/{SERVICE_ID}/models/_/duplicate-spec", params={"model_id": "ghost"}, headers=auth_header)

    assert resp.status_code == 400


def test_update_custom_model_route_still_rejects_when_installed(
    services_manager: MagicMock, client: TestClient, auth_header: dict[str, str]
) -> None:
    """The existing PUT route's contract is unchanged by the new edit route - it still delegates
    straight to `update_custom_model`, which is where the installed-guard lives."""
    services_manager.update_custom_model = AsyncMock(side_effect=HTTPException(400, "Cannot update installed server. Uninstall first."))

    resp = client.put(f"/admin/services/{SERVICE_ID}/models/custom/cust-1", json={"spec": {"name": "x"}}, headers=auth_header)

    assert resp.status_code == 400
    assert "Uninstall first" in resp.json()["error"]["message"]


def test_sync_models_returns_200(client: TestClient, auth_header: dict[str, str]) -> None:
    resp = client.post(f"/admin/services/{SERVICE_ID}/models/sync", headers=auth_header)

    assert resp.status_code == 200


def test_sync_models_calls_manager(services_manager: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    client.post(f"/admin/services/{SERVICE_ID}/models/sync", headers=auth_header)

    assert services_manager.sync_models_in_service.call_count == 1
    assert services_manager.sync_models_in_service.call_args.args[0] == SERVICE_ID


def test_sync_models_returns_ok_status(client: TestClient, auth_header: dict[str, str]) -> None:
    resp = client.post(f"/admin/services/{SERVICE_ID}/models/sync", headers=auth_header)

    assert resp.json()["status"] == "OK"


MODELS_PATH = f"/admin/services/{SERVICE_ID}/models"
MODEL_ID_PARAMS = {"model_id": MODEL_ID}
CUSTOM_SPEC_BODY = {"spec": {"name": "x"}}

BLOCKED_WHEN_CONFIG_NOT_LOADED = [
    ("POST", f"{MODELS_PATH}/_", INSTALL_BODY, MODEL_ID_PARAMS, "install_model_in_service"),
    ("DELETE", f"{MODELS_PATH}/_", UNINSTALL_BODY, MODEL_ID_PARAMS, "uninstall_model_from_service"),
    ("POST", f"{MODELS_PATH}/custom", ADD_CUSTOM_BODY, None, "add_custom_model"),
    ("DELETE", f"{MODELS_PATH}/custom/cust-1", None, None, "remove_custom_model"),
    ("PUT", f"{MODELS_PATH}/custom/cust-1", CUSTOM_SPEC_BODY, None, "update_custom_model"),
    ("POST", f"{MODELS_PATH}/custom/cust-1/edit", CUSTOM_SPEC_BODY, None, "edit_model"),
    ("POST", f"{MODELS_PATH}/_/edit", INSTALL_BODY, MODEL_ID_PARAMS, "edit_model_install_options"),
    ("POST", f"{MODELS_PATH}/sync", None, None, "sync_models_in_service"),
]


@pytest.mark.parametrize(("method", "path", "body", "params", "manager_method"), BLOCKED_WHEN_CONFIG_NOT_LOADED)
def test_model_change_503_when_services_config_unreadable(
    method: str,
    path: str,
    body: dict[str, object] | None,
    params: dict[str, str] | None,
    manager_method: str,
    services_manager: MagicMock,
    service_provider: MagicMock,
    client: TestClient,
    auth_header: dict[str, str],
) -> None:
    service_provider.load_error = "Could not read the services configuration"

    resp = client.request(method, path, json=body, params=params, headers=auth_header)

    assert resp.status_code == 503
    assert resp.json() == {
        "error": {"message": "Could not read the services configuration", "type": "api_error", "param": None, "code": None}
    }
    getattr(services_manager, manager_method).assert_not_awaited()


def test_cancel_model_install_allowed_when_services_config_unreadable(
    services_manager: MagicMock, service_provider: MagicMock, client: TestClient, auth_header: dict[str, str]
) -> None:
    service_provider.load_error = "Could not read the services configuration"

    resp = client.post(f"{MODELS_PATH}/cancel", params=MODEL_ID_PARAMS, headers=auth_header)

    assert resp.status_code == 200
    services_manager.cancel_model_install.assert_awaited_once_with(SERVICE_ID, MODEL_ID)


@pytest.mark.parametrize(("method", "path", "body", "params", "manager_method"), BLOCKED_WHEN_CONFIG_NOT_LOADED)
def test_model_change_load_error_hidden_without_auth(
    method: str, path: str, body: dict[str, object] | None, params: dict[str, str] | None, manager_method: str
) -> None:
    fake_config = MagicMock()
    fake_config.infra_admin_api_key.get_secret_value.return_value = "correct-key"
    services_manager = MagicMock()
    provider = MagicMock()
    provider.load_error = "Could not read the services configuration"
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_services_manager] = lambda: services_manager
    app.dependency_overrides[get_service_provider] = lambda: provider
    app.dependency_overrides[get_config] = lambda: fake_config

    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.request(method, path, json=body, params=params, headers={"Authorization": "Bearer wrong-key"})

    assert resp.status_code == 401
    assert "services configuration" not in resp.text
    getattr(services_manager, manager_method).assert_not_called()
