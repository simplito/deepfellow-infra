# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Tests for the MCP JSON config conversion endpoint."""

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from server.api.mcp import router
from server.core.dependencies import auth_admin
from server.error_handlers import register_exception_handlers


@pytest.fixture
def app() -> FastAPI:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(router)
    app.dependency_overrides[auth_admin] = lambda: "admin"
    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


def test_convert_config_returns_proxy_parameters(client: TestClient) -> None:
    response = client.post(
        "/admin/mcp/convert-config",
        json={"config": {"mcpServers": {"deepwiki": {"serverUrl": "https://mcp.deepwiki.com/mcp"}}}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["kind"] == "proxy"
    assert body["name"] == "deepwiki"
    assert body["server_url"] == "https://mcp.deepwiki.com/mcp"
    assert body["transport"] == "streamable_http"


def test_convert_config_returns_stdio_parameters(client: TestClient) -> None:
    response = client.post(
        "/admin/mcp/convert-config",
        json={"config": {"mcpServers": {"fs": {"command": "npx", "args": ["-y", "server-filesystem"]}}}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["kind"] == "user"
    assert body["command"] == "npx -y server-filesystem"
    assert body["variant"] == "node-headless"


def test_convert_config_invalid_input_returns_400(client: TestClient) -> None:
    response = client.post("/admin/mcp/convert-config", json={"config": {"mcpServers": {}}})

    assert response.status_code == 400
    assert "No servers found" in response.json()["error"]["message"]


def test_convert_config_requires_auth(app: FastAPI, client: TestClient) -> None:
    app.dependency_overrides.pop(auth_admin, None)

    response = client.post(
        "/admin/mcp/convert-config",
        json={"config": {"mcpServers": {"deepwiki": {"serverUrl": "https://mcp.deepwiki.com/mcp"}}}},
    )

    assert response.status_code in (401, 403)
