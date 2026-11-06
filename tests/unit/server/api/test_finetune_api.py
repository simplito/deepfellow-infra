# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Tests for the Finetune admin API."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from server.api.finetune import router
from server.core.dependencies import auth_admin, get_services_manager
from server.error_handlers import register_exception_handlers
from server.services.finetune_service import FinetuneAdapterListOut, FinetuneAdapterOut, FinetuneService

SERVICE_ID = "finetune"


@pytest.fixture
def finetune_service() -> MagicMock:
    service = MagicMock(spec=FinetuneService)
    service.list_adapters = AsyncMock(
        return_value=FinetuneAdapterListOut(
            data=[
                FinetuneAdapterOut(
                    id="pirate-0c571d9a",
                    created=1777548607,
                    base_model="meta-llama/llama-3.1-8b-instruct",
                    formats=["safetensors", "gguf"],
                    lora={"rank": 16, "alpha": 32},
                )
            ]
        )
    )
    return service


@pytest.fixture
def services_manager(finetune_service: MagicMock) -> MagicMock:
    manager = MagicMock()
    manager.split_service_type_and_instance = MagicMock(return_value=(SERVICE_ID, "default"))
    manager.services = {SERVICE_ID: finetune_service}
    return manager


@pytest.fixture
def client(services_manager: MagicMock) -> TestClient:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(router)
    app.dependency_overrides[get_services_manager] = lambda: services_manager
    app.dependency_overrides[auth_admin] = lambda: "admin"
    return TestClient(app)


def test_list_adapters_returns_data(client: TestClient, finetune_service: MagicMock) -> None:
    response = client.get(f"/admin/services/{SERVICE_ID}/adapters")

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert len(body["data"]) == 1
    assert body["data"][0]["id"] == "pirate-0c571d9a"
    assert body["data"][0]["base_model"] == "meta-llama/llama-3.1-8b-instruct"
    assert finetune_service.list_adapters.call_count == 1
    assert finetune_service.list_adapters.call_args.args == ("default",)


def test_list_adapters_forwards_instance(client: TestClient, services_manager: MagicMock, finetune_service: MagicMock) -> None:
    services_manager.split_service_type_and_instance = MagicMock(return_value=(SERVICE_ID, "custom-instance"))

    client.get(f"/admin/services/{SERVICE_ID}/adapters")

    assert finetune_service.list_adapters.call_args.args == ("custom-instance",)


def test_list_adapters_service_not_finetune_returns_404(client: TestClient, services_manager: MagicMock) -> None:
    services_manager.services = {SERVICE_ID: MagicMock()}

    response = client.get(f"/admin/services/{SERVICE_ID}/adapters")

    assert response.status_code == 404
    assert response.json()["error"]["message"] == f"Service {SERVICE_ID} is not a Finetune service"


def test_list_adapters_unknown_service_returns_404(client: TestClient, services_manager: MagicMock) -> None:
    services_manager.services = {}

    response = client.get(f"/admin/services/{SERVICE_ID}/adapters")

    assert response.status_code == 404
