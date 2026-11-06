# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Tests for server/api/warnings.py endpoints."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from server.api.warnings import router
from server.core.dependencies import auth_admin, get_config, get_service_provider
from server.error_handlers import register_exception_handlers

API_KEY = "test-key"
AUTH_HEADER = {"Authorization": f"Bearer {API_KEY}"}


def _make_service_provider(warnings: list[dict[str, str]] | None = None) -> MagicMock:
    provider = MagicMock()
    provider.load_error = None
    provider.list_warnings = AsyncMock(return_value=warnings or [])
    provider.dismiss_warning = AsyncMock(return_value=True)
    provider.dismiss_all_warnings = AsyncMock(return_value=None)
    return provider


def _make_app(service_provider: MagicMock | None = None) -> FastAPI:
    if service_provider is None:
        service_provider = _make_service_provider()
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(router)
    app.dependency_overrides[auth_admin] = lambda: API_KEY
    app.dependency_overrides[get_service_provider] = lambda: service_provider
    return app


def test_list_warnings_200_empty() -> None:
    with TestClient(_make_app()) as client:
        resp = client.get("/admin/warnings", headers=AUTH_HEADER)

    assert resp.status_code == 200
    assert resp.json() == {"list": [], "load_error": None}


def test_list_warnings_returns_entries() -> None:
    entry = {
        "id": "w1",
        "created_at": "2026-08-10T00:00:00+00:00",
        "service_id": "ollama",
        "instance": "default",
        "model_id": "m1",
        "message": "boom",
    }
    provider = _make_service_provider(warnings=[entry])

    with TestClient(_make_app(service_provider=provider)) as client:
        resp = client.get("/admin/warnings", headers=AUTH_HEADER)

    assert resp.status_code == 200
    assert resp.json() == {"list": [{**entry, "created_at": "2026-08-10T00:00:00Z"}], "load_error": None}


def test_list_warnings_sorted_newest_first() -> None:
    older = {
        "id": "w1",
        "created_at": "2026-08-10T00:00:00+00:00",
        "service_id": "ollama",
        "instance": "default",
        "model_id": "m1",
        "message": "older",
    }
    newer = {**older, "id": "w2", "created_at": "2026-08-11T00:00:00+00:00", "message": "newer"}
    provider = _make_service_provider(warnings=[older, newer])

    with TestClient(_make_app(service_provider=provider)) as client:
        resp = client.get("/admin/warnings", headers=AUTH_HEADER)

    assert resp.status_code == 200
    assert [w["id"] for w in resp.json()["list"]] == ["w2", "w1"]


def test_list_warnings_requires_auth() -> None:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_service_provider] = lambda: _make_service_provider()

    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.get("/admin/warnings")

    assert resp.status_code in (401, 403)


def test_dismiss_warning_calls_provider() -> None:
    provider = _make_service_provider()

    with TestClient(_make_app(service_provider=provider)) as client:
        resp = client.delete("/admin/warnings/w1", headers=AUTH_HEADER)

    assert resp.status_code == 200
    provider.dismiss_warning.assert_awaited_once_with("w1")


def test_dismiss_all_warnings_calls_provider() -> None:
    provider = _make_service_provider()

    with TestClient(_make_app(service_provider=provider)) as client:
        resp = client.delete("/admin/warnings", headers=AUTH_HEADER)

    assert resp.status_code == 200
    provider.dismiss_all_warnings.assert_awaited_once_with()


def test_list_warnings_returns_load_error_when_services_config_unreadable() -> None:
    provider = _make_service_provider()
    provider.load_error = "Could not read the services configuration"

    with TestClient(_make_app(service_provider=provider)) as client:
        resp = client.get("/admin/warnings", headers=AUTH_HEADER)

    assert resp.status_code == 200
    assert resp.json() == {"list": [], "load_error": "Could not read the services configuration"}
    provider.list_warnings.assert_not_awaited()


@pytest.mark.parametrize("path", ["/admin/warnings", "/admin/warnings/w1"])
def test_dismiss_warnings_503_when_services_config_unreadable(path: str) -> None:
    provider = _make_service_provider()
    provider.load_error = "Could not read the services configuration"

    with TestClient(_make_app(service_provider=provider)) as client:
        resp = client.delete(path, headers=AUTH_HEADER)

    assert resp.status_code == 503
    assert resp.json() == {
        "error": {"message": "Could not read the services configuration", "type": "api_error", "param": None, "code": None}
    }
    provider.dismiss_warning.assert_not_awaited()
    provider.dismiss_all_warnings.assert_not_awaited()


@pytest.mark.parametrize("method", ["GET", "DELETE"])
def test_warnings_load_error_hidden_without_auth(method: str) -> None:
    fake_config = MagicMock()
    fake_config.infra_admin_api_key.get_secret_value.return_value = "correct-key"
    provider = _make_service_provider()
    provider.load_error = "Could not read the services configuration"
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_service_provider] = lambda: provider
    app.dependency_overrides[get_config] = lambda: fake_config

    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.request(method, "/admin/warnings", headers={"Authorization": "Bearer wrong-key"})

    assert resp.status_code == 401
    assert "services configuration" not in resp.text


def test_dismiss_warning_unknown_id_404() -> None:
    provider = _make_service_provider()
    provider.dismiss_warning = AsyncMock(return_value=False)

    with TestClient(_make_app(service_provider=provider)) as client:
        resp = client.delete("/admin/warnings/does-not-exist", headers=AUTH_HEADER)

    assert resp.status_code == 404
