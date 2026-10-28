# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from fastapi import HTTPException

from server.models.models import InstallModelIn, ListModelsFilters
from server.models.services import InstallServiceIn
from server.services.base2_service import CustomModel, InstanceConfig, ModelConfig
from server.services.ollama_cloud_service import MODELS_TTL, OllamaCloudService
from server.services.remote_service import DefaultRemoteServiceOptions, InstalledInfo, ModelInstalledInfo, RemoteModel, RemoteService


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
def svc(deps: dict[str, Any]) -> OllamaCloudService:
    return OllamaCloudService(**deps)


def _make_installed(api_key: str = "test-key") -> InstalledInfo[DefaultRemoteServiceOptions]:
    return InstalledInfo(
        models={},
        options=InstallServiceIn(spec={"api_url": "https://ollama.com", "api_key": api_key}),
        parsed_options=DefaultRemoteServiceOptions(api_url="https://ollama.com", api_key=api_key),
    )


def _mock_tags_response(models: list[str], status: int = 200, capabilities: dict[str, list[str]] | None = None) -> MagicMock:
    caps = capabilities or {}
    body = json.dumps({"models": [{"name": m, "capabilities": caps[m]} if m in caps else {"name": m} for m in models]})
    mock_resp = AsyncMock()
    mock_resp.status = status
    mock_resp.text = AsyncMock(return_value=body)
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)
    return mock_resp


def _mock_session(mock_resp: MagicMock) -> MagicMock:
    mock_session = AsyncMock()
    mock_session.get = MagicMock(return_value=mock_resp)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    return mock_session


def test_get_type(svc: OllamaCloudService) -> None:
    assert svc.get_type() == "ollama-cloud"


def test_is_cloud(svc: OllamaCloudService) -> None:
    assert svc.is_cloud is True
    assert svc.is_cloud_service() is True


def test_get_description(svc: OllamaCloudService) -> None:
    assert "ollama" in svc.get_description().lower()


def test_get_default_url(svc: OllamaCloudService) -> None:
    assert "ollama.com" in svc.get_default_url()


def test_models_registry_is_empty(svc: OllamaCloudService) -> None:
    assert svc.get_models_registry().models == {}


def test_default_models_dict_is_empty(svc: OllamaCloudService) -> None:
    assert svc.models["default"] == {}


@pytest.mark.asyncio
async def test_fetch_models_returns_dict_on_success(svc: OllamaCloudService) -> None:
    svc.instances_info["default"].installed = _make_installed()
    mock_resp = _mock_tags_response(["llama3:latest", "mistral:7b"])
    mock_session = _mock_session(mock_resp)

    with patch("server.services.ollama_cloud_service.aiohttp.ClientSession", return_value=mock_session):
        result = await svc._fetch_models_from_api("default")  # pyright: ignore[reportPrivateUsage]

    assert result is not None
    assert "llama3:latest" in result
    assert "mistral:7b" in result
    assert result["llama3:latest"].type == "llm"


@pytest.mark.asyncio
@patch("server.services.ollama_cloud_service.aiohttp.ClientSession")
async def test_fetch_models_from_api_types_models_by_capabilities(mock_client_session: MagicMock, svc: OllamaCloudService) -> None:
    svc.instances_info["default"].installed = _make_installed()
    mock_client_session.return_value = _mock_session(
        _mock_tags_response(
            ["clef:27b", "gpt-oss:120b", "no-caps", "legacy"],
            capabilities={"clef:27b": ["decision", "vision"], "gpt-oss:120b": ["completion", "tools", "thinking"], "no-caps": []},
        )
    )

    result = await svc._fetch_models_from_api("default")  # pyright: ignore[reportPrivateUsage]

    assert result is not None
    assert result["clef:27b"].type == "decision"
    assert result["gpt-oss:120b"].type == "llm"
    assert result["no-caps"].type == "llm"
    assert result["legacy"].type == "llm"


@pytest.mark.asyncio
async def test_fetch_models_returns_none_on_http_error(svc: OllamaCloudService) -> None:
    svc.instances_info["default"].installed = _make_installed()
    mock_resp = _mock_tags_response([], status=500)
    mock_session = _mock_session(mock_resp)

    with patch("server.services.ollama_cloud_service.aiohttp.ClientSession", return_value=mock_session):
        result = await svc._fetch_models_from_api("default")  # pyright: ignore[reportPrivateUsage]

    assert result is None


@pytest.mark.asyncio
async def test_fetch_models_returns_none_on_empty_list(svc: OllamaCloudService) -> None:
    svc.instances_info["default"].installed = _make_installed()
    mock_resp = _mock_tags_response([])
    mock_session = _mock_session(mock_resp)

    with patch("server.services.ollama_cloud_service.aiohttp.ClientSession", return_value=mock_session):
        result = await svc._fetch_models_from_api("default")  # pyright: ignore[reportPrivateUsage]

    assert result is None


@pytest.mark.asyncio
async def test_fetch_models_returns_none_on_exception(svc: OllamaCloudService) -> None:
    svc.instances_info["default"].installed = _make_installed()
    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.get = MagicMock(side_effect=Exception("network error"))

    with patch("server.services.ollama_cloud_service.aiohttp.ClientSession", return_value=mock_session):
        result = await svc._fetch_models_from_api("default")  # pyright: ignore[reportPrivateUsage]

    assert result is None


@pytest.mark.asyncio
async def test_list_models_fetches_on_first_call(svc: OllamaCloudService) -> None:
    svc.instances_info["default"].installed = _make_installed()
    mock_resp = _mock_tags_response(["llama3:latest"])
    mock_session = _mock_session(mock_resp)

    with patch("server.services.ollama_cloud_service.aiohttp.ClientSession", return_value=mock_session):
        await svc.list_models("default", ListModelsFilters())

    assert mock_session.get.call_count == 1


@pytest.mark.asyncio
async def test_list_models_uses_cache_within_ttl(svc: OllamaCloudService) -> None:
    svc.instances_info["default"].installed = _make_installed()
    mock_resp = _mock_tags_response(["llama3:latest"])
    mock_session = _mock_session(mock_resp)

    with patch("server.services.ollama_cloud_service.aiohttp.ClientSession", return_value=mock_session):
        await svc.list_models("default", ListModelsFilters())
        await svc.list_models("default", ListModelsFilters())

    assert mock_session.get.call_count == 1


@pytest.mark.asyncio
async def test_list_models_refetches_after_ttl(svc: OllamaCloudService) -> None:
    svc.instances_info["default"].installed = _make_installed()
    mock_resp = _mock_tags_response(["llama3:latest"])
    mock_session = _mock_session(mock_resp)

    with patch("server.services.ollama_cloud_service.aiohttp.ClientSession", return_value=mock_session):
        await svc.list_models("default", ListModelsFilters())
        svc._models_cache_time -= MODELS_TTL + 1  # pyright: ignore[reportPrivateUsage]
        await svc.list_models("default", ListModelsFilters())

    assert mock_session.get.call_count == 2


@pytest.mark.asyncio
async def test_cleanup_unregisters_model_removed_from_ollama(svc: OllamaCloudService, deps: dict[str, Any]) -> None:
    installed = _make_installed()
    installed.models["old-model"] = ModelInstalledInfo(
        id="old-model",
        registered_name="old-model",
        type="llm",
        options=InstallModelIn(spec={}),
        completions=True,
        legacy_completions=True,
        registration_id="reg-old",
    )
    svc.instances_info["default"].installed = installed
    svc.models["default"] = {"old-model": RemoteModel(type="llm")}

    mock_resp = _mock_tags_response(["new-model"])
    mock_session = _mock_session(mock_resp)

    with patch("server.services.ollama_cloud_service.aiohttp.ClientSession", return_value=mock_session):
        svc._models_cache_time -= MODELS_TTL + 1  # pyright: ignore[reportPrivateUsage]
        await svc.list_models("default", ListModelsFilters())

    assert deps["endpoint_registry"].unregister_chat_completion.call_count == 1
    assert "old-model" not in svc.instances_info["default"].installed.models  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_cleanup_skipped_when_api_returns_empty(svc: OllamaCloudService, deps: dict[str, Any]) -> None:
    installed = _make_installed()
    installed.models["existing-model"] = ModelInstalledInfo(
        id="existing-model",
        registered_name="existing-model",
        type="llm",
        options=InstallModelIn(spec={}),
        completions=True,
        legacy_completions=True,
        registration_id="reg-1",
    )
    svc.instances_info["default"].installed = installed
    svc.models["default"] = {"existing-model": RemoteModel(type="llm")}

    mock_resp = _mock_tags_response([])
    mock_session = _mock_session(mock_resp)

    with patch("server.services.ollama_cloud_service.aiohttp.ClientSession", return_value=mock_session):
        svc._models_cache_time -= MODELS_TTL + 1  # pyright: ignore[reportPrivateUsage]
        await svc.list_models("default", ListModelsFilters())

    assert deps["endpoint_registry"].unregister_chat_completion.call_count == 0
    assert "existing-model" in svc.instances_info["default"].installed.models  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_cleanup_removes_model_with_unknown_type_without_unregistering(svc: OllamaCloudService, deps: dict[str, Any]) -> None:
    installed = _make_installed()
    installed.models["old-model"] = ModelInstalledInfo(
        id="old-model",
        registered_name="old-model",
        type="unknown-type",
        options=InstallModelIn(spec={}),
        completions=True,
        legacy_completions=True,
        registration_id="reg-old",
    )
    svc.instances_info["default"].installed = installed
    svc.models["default"] = {"old-model": RemoteModel(type="llm")}

    mock_resp = _mock_tags_response(["new-model"])
    mock_session = _mock_session(mock_resp)

    with patch("server.services.ollama_cloud_service.aiohttp.ClientSession", return_value=mock_session):
        svc._models_cache_time -= MODELS_TTL + 1  # pyright: ignore[reportPrivateUsage]
        await svc.list_models("default", ListModelsFilters())

    assert deps["endpoint_registry"].unregister_chat_completion.call_count == 0
    assert "old-model" not in svc.instances_info["default"].installed.models  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_install_model_force_refreshes_when_not_in_cache(svc: OllamaCloudService) -> None:
    svc.instances_info["default"].installed = _make_installed()
    svc.models["default"] = {}

    mock_resp = _mock_tags_response(["new-cloud-model"])
    mock_session = _mock_session(mock_resp)

    with patch("server.services.ollama_cloud_service.aiohttp.ClientSession", return_value=mock_session):
        promise = await svc._install_model("default", "new-cloud-model", InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert mock_session.get.call_count == 1
    assert "new-cloud-model" in svc.models["default"]


@pytest.mark.asyncio
async def test_install_model_skips_refresh_when_model_already_in_cache(svc: OllamaCloudService) -> None:
    svc.instances_info["default"].installed = _make_installed()
    svc.models["default"] = {"cached-model": RemoteModel(type="llm")}

    mock_resp = _mock_tags_response(["cached-model"])
    mock_session = _mock_session(mock_resp)

    with patch("server.services.ollama_cloud_service.aiohttp.ClientSession", return_value=mock_session):
        promise = await svc._install_model("default", "cached-model", InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert mock_session.get.call_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model_type", "unregister_method"),
    [
        ("tts", "unregister_audio_speech"),
        ("stt", "unregister_audio_transcriptions"),
        ("txt2img", "unregister_image_generations"),
        ("embedding", "unregister_embeddings"),
        ("decision", "unregister_systemone"),
    ],
)
async def test_cleanup_unregisters_non_llm_model_types(
    svc: OllamaCloudService, deps: dict[str, Any], model_type: str, unregister_method: str
) -> None:
    installed = _make_installed()
    installed.models["old-model"] = ModelInstalledInfo(
        id="old-model",
        registered_name="old-model",
        type=model_type,
        options=InstallModelIn(spec={}),
        completions=True,
        legacy_completions=True,
        registration_id="reg-old",
    )
    svc.instances_info["default"].installed = installed
    svc.models["default"] = {"old-model": RemoteModel(type=model_type)}  # type: ignore[arg-type]

    mock_resp = _mock_tags_response(["new-model"])
    mock_session = _mock_session(mock_resp)

    with patch("server.services.ollama_cloud_service.aiohttp.ClientSession", return_value=mock_session):
        svc._models_cache_time -= MODELS_TTL + 1  # pyright: ignore[reportPrivateUsage]
        await svc.list_models("default", ListModelsFilters())

    assert getattr(deps["endpoint_registry"], unregister_method).call_count == 1
    assert "old-model" not in svc.instances_info["default"].installed.models  # type: ignore[union-attr]


@pytest.mark.asyncio
@patch("server.services.ollama_cloud_service.aiohttp.ClientSession")
async def test_load_model_registers_decision_model_on_systemone(
    mock_client_session: MagicMock, svc: OllamaCloudService, deps: dict[str, Any]
) -> None:
    svc.instances_info["default"].installed = _make_installed()
    mock_client_session.return_value = _mock_session(_mock_tags_response(["clef:27b"], capabilities={"clef:27b": ["decision", "vision"]}))
    deps["service_provider"].dismiss_warnings_matching = AsyncMock()

    await svc.load_model("default", ModelConfig(model_id="clef:27b", options=InstallModelIn(spec={})))

    registry = deps["endpoint_registry"]
    assert registry.register_systemone_as_proxy.call_count == 1
    assert registry.register_chat_completion_as_proxy.call_count == 0
    assert svc.instances_info["default"].installed.models["clef:27b"].type == "decision"  # type: ignore[union-attr]


def test_get_custom_model_spec_returns_none(svc: OllamaCloudService) -> None:
    result = svc.get_custom_model_spec()

    assert result is None


def test_add_custom_model_raises(svc: OllamaCloudService) -> None:
    svc.instances_info["default"].installed = _make_installed()
    model = CustomModel(id="c-1", data={"id": "clef-flash", "type": "decision"})

    with pytest.raises(HTTPException) as exc:
        svc._add_custom_model("default", model)  # pyright: ignore[reportPrivateUsage]

    assert exc.value.status_code == 400
    assert "clef-flash" not in svc.models.get("default", {})


@pytest.mark.asyncio
@patch.object(RemoteService, "load_instance", new_callable=AsyncMock)
async def test_load_instance_drops_custom_models(mock_load_instance: AsyncMock, svc: OllamaCloudService) -> None:
    options = InstallServiceIn(spec={"api_url": "https://ollama.com", "api_key": "k"})
    instance_data = InstanceConfig(options=options, custom=[CustomModel(id="c-1", data={"id": "clef-flash", "type": "decision"})])

    await svc.load_instance("default", instance_data)

    assert mock_load_instance.call_count == 1
    assert mock_load_instance.call_args == call("default", InstanceConfig(options=options, custom=None))


@pytest.mark.asyncio
async def test_install_model_raises_when_not_found_after_force_refresh(svc: OllamaCloudService) -> None:
    svc.instances_info["default"].installed = _make_installed()
    svc.models["default"] = {}

    mock_resp = _mock_tags_response(["other-model"])
    mock_session = _mock_session(mock_resp)

    with patch("server.services.ollama_cloud_service.aiohttp.ClientSession", return_value=mock_session):  # noqa: SIM117
        with pytest.raises(HTTPException) as exc:
            await svc._install_model("default", "nonexistent", InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]

    assert exc.value.status_code == 400
