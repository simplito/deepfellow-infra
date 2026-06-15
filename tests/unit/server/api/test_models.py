# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for server/api/models.py endpoints."""

from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from starlette.testclient import TestClient

from server.api.models import router
from server.core.dependencies import auth_admin, get_services_manager
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
    manager.sync_models_in_service = AsyncMock(return_value=None)
    manager.cancel_model_install = AsyncMock(return_value=None)
    return manager


@pytest.fixture
def client(services_manager: MagicMock) -> Generator[TestClient]:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[auth_admin] = lambda: "test-key"
    app.dependency_overrides[get_services_manager] = lambda: services_manager
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
