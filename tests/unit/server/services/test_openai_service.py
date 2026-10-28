# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from server.models.models import InstallModelIn, ListModelsFilters
from server.models.services import InstallServiceIn
from server.services.base2_service import InstanceConfig, ModelConfig
from server.services.openai_service import OpenAIService
from server.services.remote_service import DefaultRemoteServiceOptions, InstalledInfo


@pytest.fixture
def deps() -> dict[str, Any]:
    return {
        "config": MagicMock(),
        "endpoint_registry": MagicMock(),
        "service_provider": MagicMock(),
        "model_downloader": MagicMock(),
        "docker_service": MagicMock(),
        "hardware": MagicMock(gpus=[]),
        "services_manager": MagicMock(),
    }


@pytest.fixture
def svc(deps: dict[str, Any]) -> OpenAIService:
    return OpenAIService(**deps)


def _make_installed(api_key: str = "test-key") -> InstalledInfo[DefaultRemoteServiceOptions]:
    return InstalledInfo(
        models={},
        options=InstallServiceIn(spec={"api_url": "https://api.openai.com", "api_key": api_key}),
        parsed_options=DefaultRemoteServiceOptions(api_url="https://api.openai.com", api_key=api_key),
    )


def _mock_models_response(body: object, status: int = 200) -> MagicMock:
    text = body if isinstance(body, str) else json.dumps(body)
    mock_resp = AsyncMock()
    mock_resp.status = status
    mock_resp.text = AsyncMock(return_value=text)
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)
    return mock_resp


def _mock_session(mock_resp: MagicMock) -> MagicMock:
    mock_session = AsyncMock()
    mock_session.get = MagicMock(return_value=mock_resp)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    return mock_session


@pytest.mark.asyncio
async def test_fetch_live_models_returns_entries_on_success(svc: OpenAIService) -> None:
    svc.instances_info["default"].installed = _make_installed()
    body = {"object": "list", "data": [{"id": "gpt-4o", "object": "model"}, {"id": "some-new-model", "object": "model"}]}
    mock_session = _mock_session(_mock_models_response(body))

    with patch("server.services.openai_service.aiohttp.ClientSession", return_value=mock_session):
        result = await svc._fetch_live_models("default")  # pyright: ignore[reportPrivateUsage]

    assert result is not None
    ids = {entry.id for entry in result}
    assert ids == {"gpt-4o", "some-new-model"}
    for entry in result:
        assert entry.type is None
        assert entry.messages is None
        assert entry.responses is None
        assert entry.completions is None
        assert entry.legacy_completions is None
        assert entry.context_length is None
        assert entry.max_context_length is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("api_url", "expected_url"),
    [
        ("http://ollama:11434", "http://ollama:11434/v1/models"),
        ("http://ollama:11434/v1", "http://ollama:11434/v1/models"),
        ("https://api.openai.com", "https://api.openai.com/v1/models"),
    ],
)
async def test_fetch_live_models_requests_models_endpoint_regardless_of_v1_suffix(
    svc: OpenAIService, api_url: str, expected_url: str
) -> None:
    svc.instances_info["default"].installed = InstalledInfo(
        models={},
        options=InstallServiceIn(spec={"api_url": api_url, "api_key": "test-key"}),
        parsed_options=DefaultRemoteServiceOptions(api_url=api_url, api_key="test-key"),
    )
    body = {"object": "list", "data": [{"id": "some-model", "object": "model"}]}
    mock_session = _mock_session(_mock_models_response(body))

    with patch("server.services.openai_service.aiohttp.ClientSession", return_value=mock_session):
        result = await svc._fetch_live_models("default")  # pyright: ignore[reportPrivateUsage]

    assert result is not None
    mock_session.get.assert_called_once()
    called_url = mock_session.get.call_args.args[0]
    assert called_url == expected_url


@pytest.mark.asyncio
async def test_fetch_live_models_returns_none_on_non_200(svc: OpenAIService) -> None:
    svc.instances_info["default"].installed = _make_installed()
    mock_session = _mock_session(_mock_models_response({}, status=500))

    with patch("server.services.openai_service.aiohttp.ClientSession", return_value=mock_session):
        result = await svc._fetch_live_models("default")  # pyright: ignore[reportPrivateUsage]

    assert result is None


@pytest.mark.asyncio
async def test_fetch_live_models_returns_none_on_request_exception(svc: OpenAIService) -> None:
    svc.instances_info["default"].installed = _make_installed()
    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.get = MagicMock(side_effect=ConnectionError("network error"))

    with patch("server.services.openai_service.aiohttp.ClientSession", return_value=mock_session):
        result = await svc._fetch_live_models("default")  # pyright: ignore[reportPrivateUsage]

    assert result is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        "not json",
        {"object": "list"},
        {"object": "list", "data": "not-a-list"},
        {"object": "list", "data": [{"object": "model"}]},
    ],
)
async def test_fetch_live_models_returns_none_on_malformed_body(svc: OpenAIService, body: object) -> None:
    svc.instances_info["default"].installed = _make_installed()
    mock_session = _mock_session(_mock_models_response(body))

    with patch("server.services.openai_service.aiohttp.ClientSession", return_value=mock_session):
        result = await svc._fetch_live_models("default")  # pyright: ignore[reportPrivateUsage]

    assert result is None


@pytest.mark.asyncio
async def test_fetch_live_models_treats_null_data_as_empty_catalog(svc: OpenAIService) -> None:
    """Ollama's OpenAI-compatible /v1/models reports zero local models as {"data": null}."""
    svc.instances_info["default"].installed = _make_installed()
    body = {"object": "list", "data": None}
    mock_session = _mock_session(_mock_models_response(body))

    with patch("server.services.openai_service.aiohttp.ClientSession", return_value=mock_session):
        result = await svc._fetch_live_models("default")  # pyright: ignore[reportPrivateUsage]

    assert result == []


@pytest.mark.asyncio
async def test_refresh_catalog_resolves_known_id_and_flags_unknown_id(svc: OpenAIService) -> None:
    svc.instances_info["default"].installed = _make_installed()
    body = {"data": [{"id": "gpt-4o"}, {"id": "some-brand-new-model"}]}
    mock_session = _mock_session(_mock_models_response(body))

    with patch("server.services.openai_service.aiohttp.ClientSession", return_value=mock_session):
        await svc.refresh_catalog()

    out = await svc.list_models("default", ListModelsFilters())
    by_id = {model.id: model for model in out.list}

    assert by_id["gpt-4o"].capabilities_resolved is True
    assert by_id["some-brand-new-model"].capabilities_resolved is False


@pytest.mark.asyncio
async def test_fetch_live_models_returns_none_when_instance_not_installed(svc: OpenAIService) -> None:
    result = await svc._fetch_live_models("default")  # pyright: ignore[reportPrivateUsage]

    assert result is None


@pytest.mark.asyncio
async def test_install_instance_refreshes_catalog_automatically(svc: OpenAIService, deps: dict[str, Any]) -> None:
    deps["service_provider"].dismiss_warnings_matching_any = AsyncMock()
    deps["service_provider"].save_service_config = AsyncMock()
    body = {"data": [{"id": "llama3.2:latest"}]}
    mock_session = _mock_session(_mock_models_response(body))

    with patch("server.services.openai_service.aiohttp.ClientSession", return_value=mock_session):
        promise = await svc.install_instance("default", InstallServiceIn(spec={"api_url": "http://localhost:11434", "api_key": ""}))
        await promise.wait()

    out = await svc.list_models("default", ListModelsFilters())
    ids = {model.id for model in out.list}
    assert "llama3.2:latest" in ids


@pytest.mark.asyncio
async def test_load_instance_preserves_persisted_model_missing_from_live_listing(svc: OpenAIService, deps: dict[str, Any]) -> None:
    """A persisted, previously-installed model must still load successfully on restart even when the
    live catalog refresh that runs right after (`_after_install`) no longer reports it -- it should be
    reloaded against the static overlay and then marked stale, not dropped from the catalog before
    `load_model` ever gets a chance to reinstall it (see the `run_after_install` ordering fix).
    """
    deps["service_provider"].dismiss_warnings_matching_any = AsyncMock()
    deps["service_provider"].dismiss_warnings_matching = AsyncMock()
    deps["service_provider"].save_service_config = AsyncMock()
    deps["endpoint_registry"].register_chat_completion_as_proxy = MagicMock(return_value="reg-1")
    body = {"data": []}  # live listing no longer reports gpt-4o
    mock_session = _mock_session(_mock_models_response(body))

    instance_data = InstanceConfig(
        options=InstallServiceIn(spec={"api_url": "https://api.openai.com", "api_key": "test-key"}),
        models=[ModelConfig(model_id="gpt-4o", options=InstallModelIn(spec={}))],
    )

    with patch("server.services.openai_service.aiohttp.ClientSession", return_value=mock_session):
        await svc.load_instance("default", instance_data)

    installed = svc.get_instance_installed_info("default")
    assert "gpt-4o" in installed.models  # reloaded successfully, not rejected as "Model not found"

    out = await svc.list_models("default", ListModelsFilters())
    by_id = {model.id: model for model in out.list}
    assert by_id["gpt-4o"].stale is True


def test_get_secret_values_returns_installed_and_configured_api_keys(svc: OpenAIService) -> None:
    svc.instances_info["default"].installed = _make_installed(api_key="sk-live")
    svc.instances_info["default"].config = InstanceConfig(options=InstallServiceIn(spec={"api_key": "sk-saved"}))

    result = svc.get_secret_values()

    assert sorted(result) == ["sk-live", "sk-saved"]


def test_get_secret_values_follows_updated_api_key(svc: OpenAIService) -> None:
    svc.instances_info["default"].installed = _make_installed(api_key="sk-old")
    svc.get_secret_values()

    svc.instances_info["default"].installed = _make_installed(api_key="sk-new")

    assert svc.get_secret_values() == ["sk-new"]


def test_get_secret_values_skips_empty_key_and_non_password_options(svc: OpenAIService) -> None:
    svc.instances_info["default"].installed = _make_installed(api_key="")

    assert svc.get_secret_values() == []


def test_get_secret_values_empty_when_not_installed(svc: OpenAIService) -> None:
    assert svc.get_secret_values() == []
