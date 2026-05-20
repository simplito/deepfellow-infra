# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for server/api/services.py endpoints."""

from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from starlette.testclient import TestClient

from server.api.services import router
from server.core.dependencies import auth_admin, get_config, get_endpoint_registry, get_services_manager
from server.models.services import (
    InstallServiceOut,
    ListAllModelsOut,
    ListServicesOut,
    RetrieveServiceOut,
    ServiceSpecification,
)

INSTALL_BODY = {"stream": False, "ignore_warnings": False, "spec": {"model": "llama3"}}


def _make_retrieve_service_out(service_id: str = "ollama") -> RetrieveServiceOut:
    return RetrieveServiceOut(
        id=service_id,
        type="llm",
        instance="local",
        description="Test service",
        installed=True,
        downloaded=True,
        spec=ServiceSpecification(fields=[]),
        size="5GB",
        custom_model_spec=None,
        has_docker=True,
        is_cloud=False,
    )


@pytest.fixture
def services_manager() -> MagicMock:
    manager = MagicMock()
    promise = MagicMock()
    promise.wait = AsyncMock(return_value=InstallServiceOut(status="OK"))
    manager.install_service = AsyncMock(return_value=promise)
    manager.uninstall_service = AsyncMock(return_value=None)
    manager.get_service = AsyncMock(return_value=_make_retrieve_service_out())
    manager.get_service_install_progress = AsyncMock(return_value=promise)
    manager.list_services = AsyncMock(return_value=ListServicesOut(list=[]))
    manager.list_models_from_all_services = AsyncMock(return_value=ListAllModelsOut(list=[]))
    manager.get_docker_logs = AsyncMock(return_value="some logs")
    manager.get_docker_compose_file = AsyncMock(return_value="version: '3'")
    manager.restart_docker = AsyncMock(return_value=None)
    return manager


@pytest.fixture
def endpoint_registry() -> MagicMock:
    registry = MagicMock()
    registry.test_model = AsyncMock(return_value={"ok": True})
    return registry


@pytest.fixture
def client(services_manager: MagicMock, endpoint_registry: MagicMock) -> Generator[TestClient]:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[auth_admin] = lambda: "test-key"
    app.dependency_overrides[get_services_manager] = lambda: services_manager
    app.dependency_overrides[get_endpoint_registry] = lambda: endpoint_registry
    with TestClient(app) as c:
        yield c


def test_auth_required_returns_401_with_bad_token() -> None:
    fake_config = MagicMock()
    fake_config.infra_admin_api_key.get_secret_value.return_value = "correct-key"
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_services_manager] = lambda: MagicMock()
    app.dependency_overrides[get_endpoint_registry] = lambda: MagicMock()
    app.dependency_overrides[get_config] = lambda: fake_config

    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.get("/admin/services", headers={"Authorization": "Bearer wrong-key"})

    assert resp.status_code in (401, 403)


def test_install_service_200(client: TestClient, auth_header: dict[str, str]) -> None:
    resp = client.post("/admin/services/ollama", json=INSTALL_BODY, headers=auth_header)

    assert resp.status_code == 200


def test_install_service_calls_manager(services_manager: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    client.post("/admin/services/ollama", json=INSTALL_BODY, headers=auth_header)

    assert services_manager.install_service.call_count == 1


def test_install_service_passes_service_id(services_manager: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    client.post("/admin/services/my-svc", json=INSTALL_BODY, headers=auth_header)

    assert services_manager.install_service.call_args.args[0] == "my-svc"


def test_install_service_returns_ok_status(client: TestClient, auth_header: dict[str, str]) -> None:
    resp = client.post("/admin/services/ollama", json=INSTALL_BODY, headers=auth_header)

    assert resp.json()["status"] == "OK"


def test_install_service_stream_returns_streaming_response(
    services_manager: MagicMock, client: TestClient, auth_header: dict[str, str]
) -> None:
    promise = MagicMock()
    promise.wait = AsyncMock(return_value=InstallServiceOut(status="OK"))
    services_manager.install_service = AsyncMock(return_value=promise)
    streaming = StreamingResponse(iter([b"data"]), media_type="text/event-stream")

    with patch(
        "server.api.services.convert_promise_with_progress_to_fastapi_response",
        new=AsyncMock(return_value=streaming),
    ):
        resp = client.post(
            "/admin/services/ollama",
            json={"stream": True, "ignore_warnings": False, "spec": {}},
            headers=auth_header,
        )
    assert resp.status_code == 200


UNINSTALL_BODY = {"purge": False}


def test_uninstall_service_returns_200(client: TestClient, auth_header: dict[str, str]) -> None:
    resp = client.request("DELETE", "/admin/services/ollama", json=UNINSTALL_BODY, headers=auth_header)

    assert resp.status_code == 200


def test_uninstall_service_calls_manager(services_manager: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    client.request("DELETE", "/admin/services/ollama", json=UNINSTALL_BODY, headers=auth_header)

    assert services_manager.uninstall_service.call_count == 1


def test_uninstall_service_passes_service_id(services_manager: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    client.request("DELETE", "/admin/services/my-svc", json=UNINSTALL_BODY, headers=auth_header)

    assert services_manager.uninstall_service.call_args.args[0] == "my-svc"


def test_uninstall_service_passes_purge_flag(services_manager: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    client.request("DELETE", "/admin/services/my-svc", json={"purge": True}, headers=auth_header)

    body_arg = services_manager.uninstall_service.call_args.args[1]
    assert body_arg.purge is True


def test_uninstall_service_returns_ok_status(client: TestClient, auth_header: dict[str, str]) -> None:
    resp = client.request("DELETE", "/admin/services/ollama", json=UNINSTALL_BODY, headers=auth_header)

    assert resp.json()["status"] == "OK"


def test_list_models_returns_200(client: TestClient, auth_header: dict[str, str]) -> None:
    resp = client.get("/admin/services/models", headers=auth_header)
    assert resp.status_code == 200


def test_list_models_calls_manager(services_manager: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    client.get("/admin/services/models", headers=auth_header)

    assert services_manager.list_models_from_all_services.call_count == 1


def test_list_models_returns_empty_list_by_default(client: TestClient, auth_header: dict[str, str]) -> None:
    resp = client.get("/admin/services/models", headers=auth_header)

    assert resp.json()["list"] == []


def test_list_models_passes_filters(services_manager: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    client.get("/admin/services/models?installed=true&service_id=ollama", headers=auth_header)

    filters = services_manager.list_models_from_all_services.call_args.args[0]
    assert filters.installed is True
    assert filters.service_id == "ollama"


def test_test_model_returns_200(client: TestClient, auth_header: dict[str, str]) -> None:
    resp = client.get("/admin/services/model/test/reg-123", headers=auth_header)

    assert resp.status_code == 200


def test_test_model_calls_registry(endpoint_registry: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    client.get("/admin/services/model/test/reg-123", headers=auth_header)

    assert endpoint_registry.test_model.call_count == 1
    assert endpoint_registry.test_model.call_args[0] == ("reg-123",)


def test_test_model_returns_registry_result(endpoint_registry: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    endpoint_registry.test_model = AsyncMock(return_value={"result": "pass"})

    resp = client.get("/admin/services/model/test/reg-123", headers=auth_header)

    assert resp.json() == {"result": "pass"}


def test_list_services_returns_200(client: TestClient, auth_header: dict[str, str]) -> None:
    resp = client.get("/admin/services", headers=auth_header)

    assert resp.status_code == 200


def test_list_services_calls_manager(services_manager: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    client.get("/admin/services", headers=auth_header)

    assert services_manager.list_services.call_count == 1


def test_list_services_returns_empty_list_by_default(client: TestClient, auth_header: dict[str, str]) -> None:
    resp = client.get("/admin/services", headers=auth_header)

    assert resp.json()["list"] == []


def test_list_services_passes_installed_filter(services_manager: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    client.get("/admin/services?installed=true", headers=auth_header)

    filters = services_manager.list_services.call_args.args[0]
    assert filters.installed is True


def test_retrieve_service_returns_200(client: TestClient, auth_header: dict[str, str]) -> None:
    resp = client.get("/admin/services/ollama", headers=auth_header)

    assert resp.status_code == 200


def test_retrieve_service_calls_manager_with_service_id(
    services_manager: MagicMock, client: TestClient, auth_header: dict[str, str]
) -> None:
    client.get("/admin/services/ollama", headers=auth_header)

    assert services_manager.get_service.call_count == 1
    assert services_manager.get_service.call_args[0] == ("ollama",)


def test_retrieve_service_returns_service_data(services_manager: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    services_manager.get_service = AsyncMock(return_value=_make_retrieve_service_out("ollama"))

    resp = client.get("/admin/services/ollama", headers=auth_header)

    assert resp.json()["id"] == "ollama"


def test_get_install_progress_returns_200(client: TestClient, auth_header: dict[str, str]) -> None:
    streaming = StreamingResponse(iter([b"data"]), media_type="text/event-stream")

    with patch(
        "server.api.services.convert_promise_with_progress_to_fastapi_response",
        new=AsyncMock(return_value=streaming),
    ):
        resp = client.get("/admin/services/ollama/progress", headers=auth_header)

    assert resp.status_code == 200


def test_get_install_progress_calls_manager(services_manager: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    streaming = StreamingResponse(iter([b"data"]), media_type="text/event-stream")

    with patch(
        "server.api.services.convert_promise_with_progress_to_fastapi_response",
        new=AsyncMock(return_value=streaming),
    ):
        client.get("/admin/services/my-svc/progress", headers=auth_header)

    assert services_manager.get_service_install_progress.await_count == 1
    assert services_manager.get_service_install_progress.await_args == call("my-svc")


def test_retrieve_docker_logs_returns_200(client: TestClient, auth_header: dict[str, str]) -> None:
    resp = client.get("/admin/services/ollama/docker/logs", headers=auth_header)

    assert resp.status_code == 200


@pytest.mark.parametrize(
    ("query", "expected_model_id"),
    [
        ("", None),
        ("?model_id=llama3", "llama3"),
    ],
)
def test_retrieve_docker_logs_passes_model_id(
    query: str, expected_model_id: str | None, services_manager: MagicMock, client: TestClient, auth_header: dict[str, str]
) -> None:
    client.get(f"/admin/services/ollama/docker/logs{query}", headers=auth_header)

    assert services_manager.get_docker_logs.await_count == 1
    assert services_manager.get_docker_logs.await_args == call("ollama", expected_model_id)


def test_retrieve_docker_logs_returns_logs_in_body(services_manager: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    services_manager.get_docker_logs = AsyncMock(return_value="container log output")

    resp = client.get("/admin/services/ollama/docker/logs", headers=auth_header)

    assert resp.json()["logs"] == "container log output"


def test_retrieve_docker_compose_returns_200(client: TestClient, auth_header: dict[str, str]) -> None:
    resp = client.get("/admin/services/ollama/docker/compose", headers=auth_header)

    assert resp.status_code == 200


@pytest.mark.parametrize(
    ("query", "expected_model_id"),
    [
        ("", None),
        ("?model_id=llama3", "llama3"),
    ],
)
def test_retrieve_docker_compose_passes_model_id(
    query: str, expected_model_id: str | None, services_manager: MagicMock, client: TestClient, auth_header: dict[str, str]
) -> None:
    client.get(f"/admin/services/ollama/docker/compose{query}", headers=auth_header)

    assert services_manager.get_docker_compose_file.await_count == 1
    assert services_manager.get_docker_compose_file.await_args == call("ollama", expected_model_id)


def test_retrieve_docker_compose_returns_file_in_body(services_manager: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    services_manager.get_docker_compose_file = AsyncMock(return_value="version: '3'\nservices: {}")

    resp = client.get("/admin/services/ollama/docker/compose", headers=auth_header)

    assert resp.json()["compose_file"] == "version: '3'\nservices: {}"


def test_restart_docker_returns_200(client: TestClient, auth_header: dict[str, str]) -> None:
    resp = client.post("/admin/services/ollama/docker/restart", headers=auth_header)

    assert resp.status_code == 200


@pytest.mark.parametrize(
    ("service_id", "query", "expected_model_id"),
    [
        ("ollama", "", None),
        ("my-svc", "?model_id=llama3", "llama3"),
    ],
)
def test_restart_docker_passes_model_id(
    service_id: str,
    query: str,
    expected_model_id: str | None,
    services_manager: MagicMock,
    client: TestClient,
    auth_header: dict[str, str],
) -> None:
    client.post(f"/admin/services/{service_id}/docker/restart{query}", headers=auth_header)

    assert services_manager.restart_docker.await_count == 1
    assert services_manager.restart_docker.await_args == call(service_id, expected_model_id)


def test_restart_docker_returns_ok_status(client: TestClient, auth_header: dict[str, str]) -> None:
    resp = client.post("/admin/services/ollama/docker/restart", headers=auth_header)

    assert resp.json()["status"] == "OK"
