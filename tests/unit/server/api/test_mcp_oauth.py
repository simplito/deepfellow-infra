# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Tests for the MCP OAuth admin and callback endpoints."""

from unittest.mock import AsyncMock, MagicMock, call

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from server.api.mcp_oauth import callback_router, router
from server.core.dependencies import auth_admin, get_services_manager
from server.error_handlers import register_exception_handlers
from server.services.mcp_service import McpOAuthStatusOut, McpService

SERVICE_ID = "mcp"
MODEL_ID = "my-model"


@pytest.fixture
def mcp_service() -> MagicMock:
    service = MagicMock(spec=McpService)
    service.start_oauth_flow = AsyncMock(return_value="https://auth.example/authorize?client_id=abc")
    service.get_oauth_status = MagicMock(return_value=McpOAuthStatusOut(enabled=True, status="not_started"))
    service.complete_oauth_callback = AsyncMock(return_value="Authorization complete. You can close this tab.")
    return service


@pytest.fixture
def services_manager(mcp_service: MagicMock) -> MagicMock:
    manager = MagicMock()
    manager.split_service_type_and_instance = MagicMock(return_value=(SERVICE_ID, "default"))
    manager.services = {SERVICE_ID: mcp_service}
    return manager


@pytest.fixture
def client(services_manager: MagicMock) -> TestClient:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(router)
    app.include_router(callback_router)
    app.dependency_overrides[get_services_manager] = lambda: services_manager
    app.dependency_overrides[auth_admin] = lambda: "admin"
    return TestClient(app)


def test_start_mcp_oauth_returns_authorize_url(client: TestClient, mcp_service: MagicMock) -> None:
    response = client.post(f"/admin/services/{SERVICE_ID}/models/{MODEL_ID}/oauth/start")

    assert response.status_code == 200
    assert response.json() == {"authorize_url": "https://auth.example/authorize?client_id=abc"}


def test_start_mcp_oauth_forwards_instance_and_model_id(client: TestClient, services_manager: MagicMock, mcp_service: MagicMock) -> None:
    services_manager.split_service_type_and_instance = MagicMock(return_value=(SERVICE_ID, "custom-instance"))

    client.post(f"/admin/services/{SERVICE_ID}/models/{MODEL_ID}/oauth/start")

    assert mcp_service.start_oauth_flow.call_count == 1
    assert mcp_service.start_oauth_flow.call_args == call("custom-instance", MODEL_ID)


def test_start_mcp_oauth_service_not_mcp_returns_404(client: TestClient, services_manager: MagicMock) -> None:
    services_manager.services = {SERVICE_ID: MagicMock()}

    response = client.post(f"/admin/services/{SERVICE_ID}/models/{MODEL_ID}/oauth/start")

    assert response.status_code == 404
    assert response.json()["error"]["message"] == f"Service {SERVICE_ID} is not an MCP service"


def test_start_mcp_oauth_unknown_service_returns_404(client: TestClient, services_manager: MagicMock) -> None:
    services_manager.services = {}

    response = client.post(f"/admin/services/{SERVICE_ID}/models/{MODEL_ID}/oauth/start")

    assert response.status_code == 404


def test_get_mcp_oauth_status_returns_status(client: TestClient) -> None:
    response = client.get(f"/admin/services/{SERVICE_ID}/models/{MODEL_ID}/oauth/status")

    assert response.status_code == 200
    assert response.json() == {
        "enabled": True,
        "status": "not_started",
        "has_client_id": False,
        "has_client_secret": False,
        "expires_at": None,
        "last_error": None,
    }


def test_get_mcp_oauth_status_forwards_instance_and_model_id(
    client: TestClient, services_manager: MagicMock, mcp_service: MagicMock
) -> None:
    services_manager.split_service_type_and_instance = MagicMock(return_value=(SERVICE_ID, "custom-instance"))

    client.get(f"/admin/services/{SERVICE_ID}/models/{MODEL_ID}/oauth/status")

    assert mcp_service.get_oauth_status.call_count == 1
    assert mcp_service.get_oauth_status.call_args == call("custom-instance", MODEL_ID)


def test_get_mcp_oauth_status_service_not_mcp_returns_404(client: TestClient, services_manager: MagicMock) -> None:
    services_manager.services = {SERVICE_ID: MagicMock()}

    response = client.get(f"/admin/services/{SERVICE_ID}/models/{MODEL_ID}/oauth/status")

    assert response.status_code == 404


def test_get_mcp_oauth_status_never_exposes_secrets(client: TestClient, mcp_service: MagicMock) -> None:
    mcp_service.get_oauth_status = MagicMock(
        return_value=McpOAuthStatusOut(enabled=True, status="authorized", has_client_id=True, has_client_secret=True)
    )

    response = client.get(f"/admin/services/{SERVICE_ID}/models/{MODEL_ID}/oauth/status")

    assert "client_secret" not in response.json()
    assert "access_token" not in response.json()
    assert "refresh_token" not in response.json()


def test_mcp_oauth_callback_completes_flow(client: TestClient, mcp_service: MagicMock) -> None:
    response = client.get("/mcp-oauth/callback", params={"state": "abc123", "code": "auth-code"})

    assert response.status_code == 200
    assert mcp_service.complete_oauth_callback.call_count == 1
    assert mcp_service.complete_oauth_callback.call_args == call("abc123", "auth-code", None)
    assert "Authorization complete. You can close this tab." in response.text


def test_mcp_oauth_callback_forwards_error_param(client: TestClient, mcp_service: MagicMock) -> None:
    client.get("/mcp-oauth/callback", params={"state": "abc123", "error": "access_denied"})

    assert mcp_service.complete_oauth_callback.call_count == 1
    assert mcp_service.complete_oauth_callback.call_args == call("abc123", None, "access_denied")


def test_mcp_oauth_callback_missing_state_returns_invalid_message(client: TestClient, mcp_service: MagicMock) -> None:
    response = client.get("/mcp-oauth/callback", params={"code": "auth-code"})

    assert response.status_code == 200
    assert mcp_service.complete_oauth_callback.call_count == 0
    assert "Invalid OAuth callback: missing or unrecognized state." in response.text


def test_mcp_oauth_callback_no_mcp_service_registered_returns_invalid_message(
    client: TestClient, services_manager: MagicMock, mcp_service: MagicMock
) -> None:
    services_manager.services = {}

    response = client.get("/mcp-oauth/callback", params={"state": "abc123"})

    assert response.status_code == 200
    assert mcp_service.complete_oauth_callback.call_count == 0
    assert "Invalid OAuth callback: missing or unrecognized state." in response.text


def test_mcp_oauth_callback_service_not_mcp_returns_invalid_message(client: TestClient, services_manager: MagicMock) -> None:
    services_manager.services = {SERVICE_ID: MagicMock()}

    response = client.get("/mcp-oauth/callback", params={"state": "abc123"})

    assert response.status_code == 200
    assert "Invalid OAuth callback: missing or unrecognized state." in response.text


def test_mcp_oauth_callback_escapes_html_in_message(client: TestClient, mcp_service: MagicMock) -> None:
    mcp_service.complete_oauth_callback = AsyncMock(return_value="<script>alert(1)</script>")

    response = client.get("/mcp-oauth/callback", params={"state": "abc123"})

    assert "<script>alert(1)</script>" not in response.text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in response.text


def test_mcp_oauth_callback_no_auth_required(mcp_service: MagicMock) -> None:
    app = FastAPI()
    app.include_router(callback_router)
    app.dependency_overrides[get_services_manager] = lambda: MagicMock(services={SERVICE_ID: mcp_service})
    no_auth_client = TestClient(app)

    response = no_auth_client.get("/mcp-oauth/callback", params={"state": "abc123"}, headers={})

    assert response.status_code == 200
