# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

import asyncio
import itertools
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from server.models.models import InstallModelIn, ListModelsFilters, UninstallModelIn
from server.models.services import InstallServiceIn, UninstallServiceIn
from server.services.base2_service import Base2Service, CustomModel, Instance, InstanceConfig
from server.services.ollama_external_service import (
    DownloadedInfo,
    InstalledInfo,
    ModelInstalledInfo,
    OllamaExternalOptions,
    OllamaExternalService,
    OllamaModel,
    _const,  # pyright: ignore[reportPrivateUsage]
    _read_models,  # pyright: ignore[reportPrivateUsage]
)
from server.utils.core import FetchResult, Stream, StreamChunkProgress


def _make_installed_info(base_url: str = "http://localhost:11434") -> InstalledInfo:
    return InstalledInfo(
        models={},
        options=InstallServiceIn(spec={"url": base_url}),
        parsed_options=OllamaExternalOptions(url=base_url),
        base_url=base_url,
    )


def _make_model_installed_info(model_id: str = "my-llm", registration_id: str = "reg-1") -> ModelInstalledInfo:
    return ModelInstalledInfo(
        id=model_id,
        registered_name=model_id,
        type="llm",
        options=InstallModelIn(),
        registration_id=registration_id,
    )


@pytest.fixture
def deps() -> dict[str, Any]:
    service_provider = MagicMock()
    service_provider.save_service_config = AsyncMock()
    return {
        "config": MagicMock(),
        "endpoint_registry": MagicMock(),
        "service_provider": service_provider,
        "model_downloader": MagicMock(),
        "docker_service": MagicMock(),
        "hardware": MagicMock(gpus=[]),
        "services_manager": MagicMock(),
    }


@pytest.fixture
def svc(deps: dict[str, Any]) -> OllamaExternalService:
    return OllamaExternalService(**deps)


@pytest.fixture(autouse=True)
def embedding_token_limit() -> Iterator[AsyncMock]:
    """Keep installs and syncs from measuring against a real Ollama; tests set `return_value` to a measured limit."""
    with patch(
        "server.services.ollama_external_service.OllamaEmbeddingTokenLimits.resolve", new_callable=AsyncMock, return_value=None
    ) as mock:
        yield mock


@pytest.fixture
def installed(svc: OllamaExternalService) -> InstalledInfo:
    info = _make_installed_info()
    svc.instances_info["default"].installed = info
    svc.support_responses = True
    svc.support_messages = True
    svc.support_systemone = True
    svc.ollama_version = "0.40.8"
    return info


def test_get_type(svc: OllamaExternalService) -> None:
    assert svc.get_type() == "ollama-external"


def test_get_description_not_empty(svc: OllamaExternalService) -> None:
    assert svc.get_description()


def test_service_has_docker_false(svc: OllamaExternalService) -> None:
    assert svc.service_has_docker() is False


def test_default_instance_created_on_init(svc: OllamaExternalService) -> None:
    assert "default" in svc.instances_info


def test_default_models_loaded_on_init(svc: OllamaExternalService) -> None:
    assert "default" in svc.models
    assert len(svc.models["default"]) > 0


def test_add_custom_llm_model(svc: OllamaExternalService) -> None:
    custom = CustomModel(id="c-1", data={"id": "my-llm", "size": "2GB", "type": "llm"})

    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    assert "my-llm" in svc.models["default"]
    assert svc.models["default"]["my-llm"].type == "llm"


def test_add_custom_embedding_model(svc: OllamaExternalService) -> None:
    custom = CustomModel(id="c-2", data={"id": "my-emb", "size": "500MB", "type": "embedding"})

    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    assert svc.models["default"]["my-emb"].type == "embedding"


def test_add_custom_model_duplicate_raises(svc: OllamaExternalService) -> None:
    custom = CustomModel(id="c-3", data={"id": "dup-model", "size": "1GB", "type": "llm"})

    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(HTTPException) as exc_info:
        svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400


def test_get_spec_contains_url_field(svc: OllamaExternalService) -> None:
    spec = svc.get_spec()

    field_names = [f.name for f in spec.fields]
    assert "url" in field_names


def test_get_size_returns_empty_string(svc: OllamaExternalService) -> None:
    assert svc.get_size() == ""


def test_get_model_spec_has_rename_and_alive_time(svc: OllamaExternalService) -> None:
    spec = svc.get_model_spec()

    names = {f.name for f in spec.fields}
    assert {"rename", "alive_time"} <= names


def test_get_custom_model_spec_has_required_fields(svc: OllamaExternalService) -> None:
    spec = svc.get_custom_model_spec()

    assert spec is not None
    names = {f.name for f in spec.fields}
    assert {"id", "size", "type"} <= names


def test_model_installed_info_get_info() -> None:
    model = _make_model_installed_info(registration_id="reg-42")

    info = model.get_info()

    assert info.registration_id == "reg-42"


def test_load_download_info_returns_downloaded_info(svc: OllamaExternalService) -> None:
    result = svc._load_download_info({})  # pyright: ignore[reportPrivateUsage]

    assert isinstance(result, DownloadedInfo)


def test_generate_instance_config_with_no_info(svc: OllamaExternalService) -> None:
    config = svc._generate_instance_config("default", None, None)  # pyright: ignore[reportPrivateUsage]

    assert config.options is None
    assert config.models == []


def test_generate_instance_config_with_info(svc: OllamaExternalService) -> None:
    info = _make_installed_info()
    info.models["my-llm"] = ModelInstalledInfo(
        id="my-llm",
        registered_name="my-llm",
        type="llm",
        options=InstallModelIn(),
        registration_id="reg-1",
    )

    config = svc._generate_instance_config("default", info, None)  # pyright: ignore[reportPrivateUsage]

    assert config.options == info.options
    assert len(config.models or []) == 1


def test_get_installed_info_when_not_installed(svc: OllamaExternalService) -> None:
    svc.instances_info["default"].installed = None

    with patch.object(svc, "_get_service_installed_info", return_value=False) as mock:  # pyright: ignore[reportPrivateUsage]
        result = svc.get_installed_info("default")

    assert mock.call_count == 1
    assert mock.call_args == call("default")
    assert result is False


def test_get_installed_info_when_installed(svc: OllamaExternalService) -> None:
    info = _make_installed_info()
    svc.instances_info["default"].installed = info

    result = svc.get_installed_info("default")

    assert result == info.options.spec


def test_remove_custom_model_deletes_entry(svc: OllamaExternalService) -> None:
    custom = CustomModel(id="c-5", data={"id": "to-remove", "size": "1GB", "type": "llm"})
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]
    svc._remove_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]
    assert "to-remove" not in svc.models["default"]


def test_remove_custom_model_raises_400_when_in_use(svc: OllamaExternalService) -> None:
    custom = CustomModel(id="c-6", data={"id": "in-use", "size": "1GB", "type": "llm"})
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]
    installed = _make_installed_info()
    installed.models["in-use"] = ModelInstalledInfo(
        id="in-use",
        registered_name="in-use",
        type="llm",
        options=InstallModelIn(),
        registration_id="",
    )
    svc.instances_info["default"].installed = installed

    with pytest.raises(HTTPException) as exc_info:
        svc._remove_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_stop_instance_does_nothing(svc: OllamaExternalService) -> None:
    await svc.stop_instance("default")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("version_str", "expected_responses", "expected_messages"),
    [
        ("0.15.0", True, True),
        ("0.14.1", False, True),
        ("0.14.0", False, True),
        ("0.13.0", False, False),
    ],
)
async def test_install_instance_version_flags(
    svc: OllamaExternalService,
    version_str: str,
    expected_responses: bool,
    expected_messages: bool,
) -> None:
    options = InstallServiceIn(spec={"url": "http://localhost:11434"})
    with patch("server.services.ollama_external_service.fetch_from", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = FetchResult(status_code=200, data=json.dumps({"version": version_str}))

        promise = await svc._install_instance("default", options)  # pyright: ignore[reportPrivateUsage]
        result = await promise.wait()

    assert isinstance(result, InstalledInfo)
    assert svc.support_responses is expected_responses
    assert svc.support_messages is expected_messages


@pytest.mark.asyncio
async def test_install_instance_bad_status_raises_400(svc: OllamaExternalService) -> None:
    options = InstallServiceIn(spec={"url": "http://localhost:11434"})
    with patch("server.services.ollama_external_service.fetch_from", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = FetchResult(status_code=503, data="")

        promise = await svc._install_instance("default", options)  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(HTTPException) as exc_info:
            await promise.wait()

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_install_instance_invalid_json_raises_500(svc: OllamaExternalService) -> None:
    options = InstallServiceIn(spec={"url": "http://localhost:11434"})
    with patch("server.services.ollama_external_service.fetch_from", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = FetchResult(status_code=200, data="not-json")

        promise = await svc._install_instance("default", options)  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(HTTPException) as exc_info:
            await promise.wait()

    assert exc_info.value.status_code == 500


@pytest.mark.asyncio
async def test_install_instance_connection_error_raises_400(svc: OllamaExternalService) -> None:
    options = InstallServiceIn(spec={"url": "http://localhost:11434"})

    with patch("server.services.ollama_external_service.fetch_from", side_effect=ConnectionError("refused")):
        promise = await svc._install_instance("default", options)  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(HTTPException) as exc_info:
            await promise.wait()

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_uninstall_instance_unregisters_llm_and_embedding(svc: OllamaExternalService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    installed.models["test-llm"] = ModelInstalledInfo(
        id="test-llm", registered_name="test-llm", type="llm", options=InstallModelIn(), registration_id="reg-llm"
    )
    installed.models["test-emb"] = ModelInstalledInfo(
        id="test-emb", registered_name="test-emb", type="embedding", options=InstallModelIn(), registration_id="reg-emb"
    )
    svc.instances_info["default"].installed = installed

    with patch.object(svc, "_uninstall_model", new_callable=AsyncMock):  # pyright: ignore[reportPrivateUsage]
        await svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert deps["endpoint_registry"].unregister_chat_completion.call_count == 1
    assert deps["endpoint_registry"].unregister_chat_completion.call_args == call("test-llm", "reg-llm")
    assert deps["endpoint_registry"].unregister_embeddings.call_count == 1
    assert deps["endpoint_registry"].unregister_embeddings.call_args == call("test-emb", "reg-emb")


@pytest.mark.asyncio
async def test_uninstall_instance_sets_installed_to_none(svc: OllamaExternalService) -> None:
    svc.instances_info["default"].installed = _make_installed_info()

    with patch.object(svc, "_uninstall_model", new_callable=AsyncMock):  # pyright: ignore[reportPrivateUsage]
        await svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert svc.instances_info["default"].installed is None


@pytest.mark.asyncio
async def test_uninstall_instance_purge_resets_default_instance(svc: OllamaExternalService) -> None:
    svc.instances_info["default"].installed = _make_installed_info()

    with (
        patch.object(svc, "_uninstall_model", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_clear_working_dir", new_callable=AsyncMock) as mock_clear,  # pyright: ignore[reportPrivateUsage]
    ):
        await svc._uninstall_instance("default", UninstallServiceIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert svc.service_downloaded is False
    assert mock_clear.call_count == 1
    assert svc.instances_info["default"].installed is None


@pytest.mark.asyncio
async def test_uninstall_instance_purge_deletes_non_default_instance(svc: OllamaExternalService) -> None:
    svc.instances_info["extra"] = Instance(None, None, {}, InstanceConfig())
    svc.instances_info["extra"].installed = _make_installed_info()
    svc.models["extra"] = {}

    with (
        patch.object(svc, "_uninstall_model", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_clear_working_dir", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
    ):
        await svc._uninstall_instance("extra", UninstallServiceIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert "extra" not in svc.instances_info


@pytest.mark.asyncio
async def test_uninstall_instance_purge_with_other_instance_installed_does_not_clear_working_dir(
    svc: OllamaExternalService,
) -> None:
    svc.instances_info["extra"] = Instance(None, None, {}, InstanceConfig())
    svc.instances_info["extra"].installed = _make_installed_info()
    svc.instances_info["default"].installed = _make_installed_info()
    svc.models["extra"] = {}

    with (
        patch.object(svc, "_uninstall_model", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_clear_working_dir", new_callable=AsyncMock) as mock_clear,  # pyright: ignore[reportPrivateUsage]
    ):
        await svc._uninstall_instance("default", UninstallServiceIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert mock_clear.call_count == 0
    assert svc.instances_info["default"].installed is None
    assert "extra" in svc.instances_info


@pytest.mark.asyncio
async def test_list_models_returns_all_for_valid_instance(svc: OllamaExternalService, installed: InstalledInfo) -> None:
    result = await svc.list_models("default", ListModelsFilters())

    assert len(result.list) == len(svc.models["default"])


@pytest.mark.asyncio
async def test_list_models_raises_404_for_unknown_instance(svc: OllamaExternalService) -> None:
    with pytest.raises(HTTPException) as exc_info:
        await svc.list_models("nonexistent", ListModelsFilters())

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_list_models_filters_installed_only(svc: OllamaExternalService, installed: InstalledInfo) -> None:
    some_id = next(iter(svc.models["default"]))
    installed.models[some_id] = ModelInstalledInfo(
        id=some_id, registered_name=some_id, type="llm", options=InstallModelIn(), registration_id="reg-1"
    )

    result = await svc.list_models("default", ListModelsFilters(installed=True))

    assert len(result.list) >= 1
    assert all(bool(m.installed) for m in result.list)


@pytest.mark.asyncio
async def test_get_model_returns_model_out(svc: OllamaExternalService, installed: InstalledInfo) -> None:
    model_id = next(iter(svc.models["default"]))

    result = await svc.get_model("default", model_id)

    assert result.id == model_id


@pytest.mark.asyncio
async def test_get_model_returns_installed_info_when_model_installed(svc: OllamaExternalService, installed: InstalledInfo) -> None:
    model_id = next(iter(svc.models["default"]))
    installed.models[model_id] = ModelInstalledInfo(
        id=model_id, registered_name=model_id, type="llm", options=InstallModelIn(), registration_id="reg-1"
    )

    result = await svc.get_model("default", model_id)

    assert result.installed is not None


@pytest.mark.asyncio
async def test_get_model_raises_400_for_unknown_model(svc: OllamaExternalService, installed: InstalledInfo) -> None:
    with pytest.raises(HTTPException) as exc_info:
        await svc.get_model("default", "nonexistent-model")

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_download_model_emits_progress_and_completes(svc: OllamaExternalService) -> None:
    stream = MagicMock()
    model = OllamaModel(id="llama3", size="4GB", type="llm")

    async def mock_stream(*args: object, **kwargs: object):  # type: ignore[misc]
        yield FetchResult(status_code=200, data=json.dumps({"status": "success"}))

    with patch("server.services.ollama_external_service.stream_fetch_from", side_effect=mock_stream):
        await svc._download_model(stream, model, "llama3", "http://localhost:11434")  # pyright: ignore[reportPrivateUsage]

    assert stream.emit.call_count >= 2


@pytest.mark.asyncio
async def test_download_model_raises_400_on_error_in_data(svc: OllamaExternalService) -> None:
    stream = MagicMock()
    model = OllamaModel(id="bad-model", size="1GB", type="llm")

    async def mock_stream(*args: object, **kwargs: object):  # type: ignore[misc]
        yield FetchResult(status_code=200, data=json.dumps({"error": "model not found"}))

    with (
        patch("server.services.ollama_external_service.stream_fetch_from", side_effect=mock_stream),
        pytest.raises(HTTPException) as exc_info,
    ):
        await svc._download_model(stream, model, "bad-model", "http://localhost:11434")  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_download_model_raises_400_on_bad_status(svc: OllamaExternalService) -> None:
    stream = MagicMock()
    model = OllamaModel(id="bad-model", size="1GB", type="llm")

    async def mock_stream(*args: object, **kwargs: object):  # type: ignore[misc]
        yield FetchResult(status_code=404, data=json.dumps({"status": "not found"}))

    with (
        patch("server.services.ollama_external_service.stream_fetch_from", side_effect=mock_stream),
        pytest.raises(HTTPException) as exc_info,
    ):
        await svc._download_model(stream, model, "bad-model", "http://localhost:11434")  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_download_model_raises_on_missing_hf_prefix(svc: OllamaExternalService) -> None:
    stream = MagicMock()
    model = OllamaModel(id="bartowski/Qwen3-0.6B-GGUF", size="1GB", type="llm")

    async def mock_stream(*args: object, **kwargs: object):  # type: ignore[misc]
        yield FetchResult(status_code=200, data=json.dumps({"status": "pulling manifest"}))
        yield FetchResult(status_code=200, data=json.dumps({"error": "pull model manifest: file does not exist"}))

    with (
        patch("server.services.ollama_external_service.stream_fetch_from", side_effect=mock_stream),
        pytest.raises(HTTPException) as exc_info,
    ):
        await svc._download_model(stream, model, "bartowski/Qwen3-0.6B-GGUF", "http://localhost:11434")  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400
    assert "hf.co/" in exc_info.value.detail


@pytest.mark.asyncio
async def test_download_model_raises_on_non_gguf_repo(svc: OllamaExternalService) -> None:
    stream = MagicMock()
    model = OllamaModel(id="hf.co/Qwen/Qwen3-0.6B", size="1GB", type="llm")

    async def mock_stream(*args: object, **kwargs: object):  # type: ignore[misc]
        yield FetchResult(status_code=200, data=json.dumps({"status": "pulling manifest"}))
        yield FetchResult(
            status_code=200,
            data=json.dumps({"error": 'pull model manifest: 400: {"error":"Repository is not GGUF or is not compatible with llama.cpp"}'}),
        )

    with (
        patch("server.services.ollama_external_service.stream_fetch_from", side_effect=mock_stream),
        pytest.raises(HTTPException) as exc_info,
    ):
        await svc._download_model(stream, model, "hf.co/Qwen/Qwen3-0.6B", "http://localhost:11434")  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400
    assert "GGUF" in exc_info.value.detail


@pytest.mark.asyncio
async def test_download_model_raises_on_gated_model(svc: OllamaExternalService) -> None:
    stream = MagicMock()
    model = OllamaModel(id="hf.co/meta-llama/Llama-3.2-1B", size="1GB", type="llm")

    async def mock_stream(*args: object, **kwargs: object):  # type: ignore[misc]
        yield FetchResult(status_code=200, data=json.dumps({"status": "pulling manifest"}))
        yield FetchResult(
            status_code=200,
            data=json.dumps({"error": 'pull model manifest: realm host "huggingface.co" does not match original host "hf.co"'}),
        )

    with (
        patch("server.services.ollama_external_service.stream_fetch_from", side_effect=mock_stream),
        pytest.raises(HTTPException) as exc_info,
    ):
        await svc._download_model(stream, model, "hf.co/meta-llama/Llama-3.2-1B", "http://localhost:11434")  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400
    assert "HuggingFace" in exc_info.value.detail


@pytest.mark.asyncio
async def test_download_model_raises_with_raw_error_for_unknown_errors(svc: OllamaExternalService) -> None:
    stream = MagicMock()
    model = OllamaModel(id="llama3", size="1GB", type="llm")

    async def mock_stream(*args: object, **kwargs: object):  # type: ignore[misc]
        yield FetchResult(status_code=200, data=json.dumps({"error": "some unexpected ollama error"}))

    with (
        patch("server.services.ollama_external_service.stream_fetch_from", side_effect=mock_stream),
        pytest.raises(HTTPException) as exc_info,
    ):
        await svc._download_model(stream, model, "llama3", "http://localhost:11434")  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400
    assert "some unexpected ollama error" in exc_info.value.detail


@pytest.mark.asyncio
async def test_download_model_or_set_progress_starts_download(svc: OllamaExternalService) -> None:
    stream = MagicMock()
    model = OllamaModel(id="llama3", size="1GB", type="llm")

    with patch.object(svc, "_download_model", new_callable=AsyncMock) as mock_dl:  # pyright: ignore[reportPrivateUsage]
        await svc._download_model_or_set_progress(stream, model, "llama3", "http://localhost:11434")  # pyright: ignore[reportPrivateUsage]

    assert mock_dl.call_count == 1
    assert "llama3" not in svc.models_download_progress


@pytest.mark.asyncio
async def test_download_model_or_set_progress_forwards_existing_stream(svc: OllamaExternalService) -> None:
    existing: Stream[StreamChunkProgress] = Stream()  # type: ignore[type-arg]
    chunk = StreamChunkProgress(type="progress", stage="download", value=0.5, data={})
    existing.emit(chunk)
    existing.close()
    svc.models_download_progress["llama3"] = existing  # type: ignore[assignment]
    output = MagicMock()
    model = OllamaModel(id="llama3", size="1GB", type="llm")

    await svc._download_model_or_set_progress(output, model, "llama3", "http://localhost:11434")  # pyright: ignore[reportPrivateUsage]

    assert output.emit.call_count == 1
    assert output.emit.call_args == call(chunk)


@pytest.mark.asyncio
async def test_download_model_or_set_progress_cleans_up_on_failure(svc: OllamaExternalService) -> None:
    stream = MagicMock()
    model = OllamaModel(id="llama3", size="1GB", type="llm")

    with (
        patch.object(svc, "_download_model", new_callable=AsyncMock, side_effect=HTTPException(400, "Model not available")),
        pytest.raises(HTTPException),
    ):
        await svc._download_model_or_set_progress(stream, model, "llama3", "http://localhost:11434")  # pyright: ignore[reportPrivateUsage]

    assert "llama3" not in svc.models_download_progress


@pytest.mark.asyncio
async def test_download_model_or_set_progress_retries_download_after_failure(svc: OllamaExternalService) -> None:
    stream1 = MagicMock()
    stream2 = MagicMock()
    model = OllamaModel(id="llama3", size="1GB", type="llm")

    mock_dl = AsyncMock(side_effect=[HTTPException(400, "Model not available"), None])
    with patch.object(svc, "_download_model", mock_dl):  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(HTTPException):
            await svc._download_model_or_set_progress(stream1, model, "llama3", "http://localhost:11434")  # pyright: ignore[reportPrivateUsage]
        await svc._download_model_or_set_progress(stream2, model, "llama3", "http://localhost:11434")  # pyright: ignore[reportPrivateUsage]

    assert mock_dl.call_count == 2


@pytest.mark.asyncio
async def test_install_model_returns_already_installed(svc: OllamaExternalService, installed: InstalledInfo) -> None:
    model_id = next(iter(svc.models["default"]))
    installed.models[model_id] = ModelInstalledInfo(
        id=model_id, registered_name=model_id, type="llm", options=InstallModelIn(), registration_id=""
    )

    promise = await svc._install_model("default", model_id, InstallModelIn())  # pyright: ignore[reportPrivateUsage]
    result = await promise.wait()

    assert result.status == "OK"
    assert "Already installed" in result.details


@pytest.mark.asyncio
async def test_install_model_raises_400_for_unknown_model(svc: OllamaExternalService, installed: InstalledInfo) -> None:
    with pytest.raises(HTTPException) as exc_info:
        await svc._install_model("default", "nonexistent", InstallModelIn())  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_install_model_registers_llm_endpoint(svc: OllamaExternalService, installed: InstalledInfo, deps: dict[str, Any]) -> None:
    svc.models["default"]["test-llm"] = OllamaModel(id="test-llm", size="1GB", type="llm")

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_sync_models_from_external_ollama", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
    ):
        promise = await svc._install_model("default", "test-llm", InstallModelIn())  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert deps["endpoint_registry"].register_chat_completion_as_proxy.call_count == 1


@pytest.mark.asyncio
async def test_install_model_registers_embedding_endpoint(
    svc: OllamaExternalService, installed: InstalledInfo, deps: dict[str, Any]
) -> None:
    svc.models["default"]["test-emb"] = OllamaModel(id="test-emb", size="500MB", type="embedding")

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_sync_models_from_external_ollama", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
    ):
        promise = await svc._install_model("default", "test-emb", InstallModelIn())  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert deps["endpoint_registry"].register_embeddings_as_proxy.call_count == 1


@pytest.mark.asyncio
async def test_install_model_embedding_reports_token_limit_ollama_applies(
    svc: OllamaExternalService, installed: InstalledInfo, deps: dict[str, Any], embedding_token_limit: AsyncMock
) -> None:
    svc.models["default"]["bge-m3"] = OllamaModel(id="bge-m3", size="1GB", type="embedding")
    embedding_token_limit.return_value = 2048

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_sync_models_from_external_ollama", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_fetch_show_info", new_callable=AsyncMock, return_value=(8192, None, None, None)),  # pyright: ignore[reportPrivateUsage]
    ):
        promise = await svc._install_model("default", "bge-m3", InstallModelIn())  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    props = deps["endpoint_registry"].register_embeddings_as_proxy.call_args.kwargs["props"]
    assert props.max_context_window == 2048
    assert props.context_window == 2048
    assert embedding_token_limit.call_args == call("default", "http://localhost:11434", "bge-m3")
    tokenizer = deps["endpoint_registry"].register_embeddings_as_proxy.call_args.kwargs["tokenizer"]
    assert tokenizer.func == svc._tokenizers.archive  # pyright: ignore[reportPrivateUsage]
    assert tokenizer.args == ("http://localhost:11434", "bge-m3")


@pytest.mark.asyncio
async def test_install_model_llm_does_not_measure_embedding_limit(
    svc: OllamaExternalService, installed: InstalledInfo, embedding_token_limit: AsyncMock
) -> None:
    svc.models["default"]["test-llm"] = OllamaModel(id="test-llm", size="1GB", type="llm")

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_sync_models_from_external_ollama", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_fetch_show_info", new_callable=AsyncMock, return_value=(8192, None, None, None)),  # pyright: ignore[reportPrivateUsage]
    ):
        promise = await svc._install_model("default", "test-llm", InstallModelIn())  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert embedding_token_limit.call_count == 0


@pytest.mark.asyncio
async def test_sync_models_embedding_reports_token_limit_ollama_applies(
    svc: OllamaExternalService, installed: InstalledInfo, deps: dict[str, Any], embedding_token_limit: AsyncMock
) -> None:
    svc.models["default"]["paraphrase-multilingual"] = OllamaModel(id="paraphrase-multilingual", size="500MB", type="embedding")
    embedding_token_limit.return_value = 128
    tags = json.dumps({"models": [{"name": "paraphrase-multilingual:latest", "size": 1024}]})

    with (
        patch(
            "server.services.ollama_external_service.fetch_from",
            new_callable=AsyncMock,
            return_value=FetchResult(status_code=200, data=tags),
        ),
        patch.object(svc, "_fetch_show_info", new_callable=AsyncMock, return_value=(512, None, None, None)),  # pyright: ignore[reportPrivateUsage]
    ):
        await svc._sync_models_from_external_ollama("default", installed, is_initial_sync=True)  # pyright: ignore[reportPrivateUsage]

    props = deps["endpoint_registry"].register_embeddings_as_proxy.call_args.kwargs["props"]
    assert props.max_context_window == 128
    assert props.context_window == 128
    tokenizer = deps["endpoint_registry"].register_embeddings_as_proxy.call_args.kwargs["tokenizer"]
    assert tokenizer.args == ("http://localhost:11434", "paraphrase-multilingual")


@pytest.mark.asyncio
async def test_sync_models_does_not_measure_a_decision_model(
    svc: OllamaExternalService, installed: InstalledInfo, embedding_token_limit: AsyncMock
) -> None:
    svc.models["default"]["laya"] = OllamaModel(id="laya", size="500MB", type="embedding")
    tags = json.dumps({"models": [{"name": "laya:latest", "size": 1024}]})

    with (
        patch(
            "server.services.ollama_external_service.fetch_from",
            new_callable=AsyncMock,
            return_value=FetchResult(status_code=200, data=tags),
        ),
        patch.object(svc, "_fetch_show_info", new_callable=AsyncMock, return_value=(512, None, None, True)),  # pyright: ignore[reportPrivateUsage]
    ):
        await svc._sync_models_from_external_ollama("default", installed, is_initial_sync=True)  # pyright: ignore[reportPrivateUsage]

    assert embedding_token_limit.call_count == 0


async def _until_awaited(mock: AsyncMock, count: int) -> None:
    for _ in range(100):
        if mock.await_count >= count:
            return
        await asyncio.sleep(0)
    pytest.fail(f"expected {count} awaits, got {mock.await_count}")


@pytest.mark.asyncio
async def test_sync_during_install_probe_leaves_one_registration(
    svc: OllamaExternalService, installed: InstalledInfo, deps: dict[str, Any], embedding_token_limit: AsyncMock
) -> None:
    svc.models["default"]["bge-m3"] = OllamaModel(id="bge-m3", size="1GB", type="embedding")
    tags = json.dumps({"models": [{"name": "bge-m3:latest", "size": 1024}]})
    probe_done = asyncio.Event()

    async def slow_probe(*_args: object, **_kwargs: object) -> int:
        await probe_done.wait()
        return 2048

    embedding_token_limit.side_effect = slow_probe

    with (
        patch(
            "server.services.ollama_external_service.fetch_from",
            new_callable=AsyncMock,
            return_value=FetchResult(status_code=200, data=tags),
        ),
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_fetch_show_info", new_callable=AsyncMock, return_value=(8192, None, None, None)),  # pyright: ignore[reportPrivateUsage]
    ):
        promise = await svc._install_model("default", "bge-m3", InstallModelIn())  # pyright: ignore[reportPrivateUsage]
        await _until_awaited(embedding_token_limit, 1)
        sync = asyncio.create_task(svc._sync_models_from_external_ollama("default", installed))  # pyright: ignore[reportPrivateUsage]
        await asyncio.sleep(0)
        probe_done.set()
        await asyncio.gather(promise.wait(), sync)

    assert deps["endpoint_registry"].register_embeddings_as_proxy.call_count == 1
    assert embedding_token_limit.await_count == 1


@pytest.mark.asyncio
async def test_concurrent_syncs_leave_one_registration(
    svc: OllamaExternalService, installed: InstalledInfo, deps: dict[str, Any], embedding_token_limit: AsyncMock
) -> None:
    svc.models["default"]["bge-m3"] = OllamaModel(id="bge-m3", size="1GB", type="embedding")
    tags = json.dumps({"models": [{"name": "bge-m3:latest", "size": 1024}]})
    probes_done = asyncio.Event()

    async def slow_probe(*_args: object, **_kwargs: object) -> int:
        await probes_done.wait()
        return 2048

    embedding_token_limit.side_effect = slow_probe

    with (
        patch(
            "server.services.ollama_external_service.fetch_from",
            new_callable=AsyncMock,
            return_value=FetchResult(status_code=200, data=tags),
        ),
        patch.object(svc, "_fetch_show_info", new_callable=AsyncMock, return_value=(8192, None, None, None)),  # pyright: ignore[reportPrivateUsage]
    ):
        syncs = [asyncio.create_task(svc._sync_models_from_external_ollama("default", installed)) for _ in range(2)]  # pyright: ignore[reportPrivateUsage]
        await _until_awaited(embedding_token_limit, 2)
        probes_done.set()
        await asyncio.gather(*syncs)

    assert deps["endpoint_registry"].register_embeddings_as_proxy.call_count == 1
    assert installed.models["bge-m3"].registration_id == deps["endpoint_registry"].register_embeddings_as_proxy.return_value


def _register_embedding(installed: InstalledInfo, model_id: str = "bge-m3", registration_id: str = "reg-emb") -> ModelInstalledInfo:
    model = _make_model_installed_info(model_id, registration_id)
    model.type = "embedding"
    installed.models[model_id] = model
    return model


@pytest.mark.asyncio
async def test_refresh_measures_a_stale_embedding_model_and_updates_its_registration(
    svc: OllamaExternalService, installed: InstalledInfo, deps: dict[str, Any], embedding_token_limit: AsyncMock
) -> None:
    _register_embedding(installed)
    embedding_token_limit.return_value = 2048
    fresh = MagicMock(side_effect=[False, True])

    with (
        patch.object(svc._embedding_token_limits, "has_fresh_measurement", fresh),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_fetch_show_info", new_callable=AsyncMock, return_value=(8192, None, None, None)),  # pyright: ignore[reportPrivateUsage]
    ):
        await svc._refresh_embedding_token_limits("default", installed)  # pyright: ignore[reportPrivateUsage]

    assert embedding_token_limit.call_args == call("default", "http://localhost:11434", "bge-m3")
    assert deps["endpoint_registry"].update_context_window.call_args == call("reg-emb", context_window=2048, max_context_window=2048)


@pytest.mark.asyncio
async def test_refresh_keeps_the_registration_when_measuring_fails_again(
    svc: OllamaExternalService, installed: InstalledInfo, deps: dict[str, Any], embedding_token_limit: AsyncMock
) -> None:
    _register_embedding(installed)
    embedding_token_limit.return_value = 8192

    with (
        patch.object(svc._embedding_token_limits, "has_fresh_measurement", return_value=False),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_fetch_show_info", new_callable=AsyncMock, return_value=(8192, None, None, None)),  # pyright: ignore[reportPrivateUsage]
    ):
        await svc._refresh_embedding_token_limits("default", installed)  # pyright: ignore[reportPrivateUsage]

    assert embedding_token_limit.call_count == 1
    assert deps["endpoint_registry"].update_context_window.call_count == 0


@pytest.mark.asyncio
async def test_refresh_leaves_fresh_and_non_embedding_models_alone(
    svc: OllamaExternalService, installed: InstalledInfo, deps: dict[str, Any], embedding_token_limit: AsyncMock
) -> None:
    _register_embedding(installed)
    installed.models["my-llm"] = _make_model_installed_info("my-llm")

    with patch.object(svc._embedding_token_limits, "has_fresh_measurement", return_value=True) as fresh:  # pyright: ignore[reportPrivateUsage]
        await svc._refresh_embedding_token_limits("default", installed)  # pyright: ignore[reportPrivateUsage]

    assert fresh.call_args_list == [call("default", "bge-m3")]
    assert embedding_token_limit.call_count == 0
    assert deps["endpoint_registry"].update_context_window.call_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("uninstalled", [True, False])
async def test_refresh_skips_a_model_without_a_registration(
    svc: OllamaExternalService, installed: InstalledInfo, deps: dict[str, Any], embedding_token_limit: AsyncMock, uninstalled: bool
) -> None:
    _register_embedding(installed, registration_id="")
    embedding_token_limit.return_value = 2048

    async def show(*_args: object) -> tuple[int, None, None, None]:
        if uninstalled:
            installed.models.pop("bge-m3")
        return 8192, None, None, None

    with (
        patch.object(svc._embedding_token_limits, "has_fresh_measurement", MagicMock(side_effect=[False, True])),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_fetch_show_info", side_effect=show),  # pyright: ignore[reportPrivateUsage]
    ):
        await svc._refresh_embedding_token_limits("default", installed)  # pyright: ignore[reportPrivateUsage]

    assert deps["endpoint_registry"].update_context_window.call_count == 0


@pytest.mark.asyncio
async def test_install_model_uses_rename_as_registered_name(svc: OllamaExternalService, installed: InstalledInfo) -> None:
    svc.models["default"]["test-llm"] = OllamaModel(id="test-llm", size="1GB", type="llm")

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_sync_models_from_external_ollama", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
    ):
        promise = await svc._install_model("default", "test-llm", InstallModelIn(spec={"rename": "my-alias"}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert installed.models["test-llm"].registered_name == "my-alias"


@pytest.mark.asyncio
async def test_install_model_calls_generate_when_alive_time_set(svc: OllamaExternalService, installed: InstalledInfo) -> None:
    svc.models["default"]["test-llm"] = OllamaModel(id="test-llm", size="1GB", type="llm")
    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_sync_models_from_external_ollama", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_fetch_show_info", new_callable=AsyncMock, return_value=(None, None, None, None)),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.ollama_external_service.fetch_from", new_callable=AsyncMock) as mock_fetch,
    ):
        mock_fetch.return_value = FetchResult(status_code=200, data="")

        promise = await svc._install_model("default", "test-llm", InstallModelIn(spec={"alive_time": "5m"}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert mock_fetch.call_count == 1
    assert "keep_alive" in mock_fetch.call_args[0][2]


@pytest.mark.asyncio
async def test_install_model_marks_as_downloaded(svc: OllamaExternalService, installed: InstalledInfo) -> None:
    svc.models["default"]["test-llm"] = OllamaModel(id="test-llm", size="1GB", type="llm")

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_sync_models_from_external_ollama", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
    ):
        promise = await svc._install_model("default", "test-llm", InstallModelIn())  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert "test-llm" in svc.models_downloaded


@pytest.mark.asyncio
async def test_uninstall_model_unregisters_llm(svc: OllamaExternalService, installed: InstalledInfo, deps: dict[str, Any]) -> None:
    installed.models["test-llm"] = ModelInstalledInfo(
        id="test-llm", registered_name="test-llm", type="llm", options=InstallModelIn(), registration_id="reg-1"
    )

    with (
        patch.object(svc, "_sync_models_from_external_ollama", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_save", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
    ):
        await svc._uninstall_model("default", "test-llm", UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert "test-llm" not in installed.models
    assert deps["endpoint_registry"].unregister_chat_completion.call_count == 1
    assert deps["endpoint_registry"].unregister_chat_completion.call_args == call("test-llm", "reg-1")


@pytest.mark.asyncio
async def test_uninstall_model_unregisters_embedding(svc: OllamaExternalService, installed: InstalledInfo, deps: dict[str, Any]) -> None:
    installed.models["test-emb"] = ModelInstalledInfo(
        id="test-emb", registered_name="test-emb", type="embedding", options=InstallModelIn(), registration_id="reg-emb"
    )

    with (
        patch.object(svc, "_sync_models_from_external_ollama", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_save", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
    ):
        await svc._uninstall_model("default", "test-emb", UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert deps["endpoint_registry"].unregister_embeddings.call_count == 1
    assert deps["endpoint_registry"].unregister_embeddings.call_args == call("test-emb", "reg-emb")


@pytest.mark.asyncio
async def test_uninstall_model_purges_downloaded_data(svc: OllamaExternalService, installed: InstalledInfo) -> None:
    installed.models["test-llm"] = ModelInstalledInfo(
        id="test-llm", registered_name="test-llm", type="llm", options=InstallModelIn(), registration_id=""
    )
    svc.models_downloaded["test-llm"] = DownloadedInfo()
    with (
        patch.object(svc, "_sync_models_from_external_ollama", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.ollama_external_service.fetch_from", new_callable=AsyncMock) as mock_fetch,
    ):
        mock_fetch.return_value = FetchResult(status_code=200, data="")

        await svc._uninstall_model("default", "test-llm", UninstallModelIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert mock_fetch.call_count == 1
    assert "test-llm" not in svc.models_downloaded


@pytest.mark.asyncio
async def test_uninstall_model_no_purge_skips_delete(svc: OllamaExternalService, installed: InstalledInfo) -> None:
    installed.models["test-llm"] = ModelInstalledInfo(
        id="test-llm", registered_name="test-llm", type="llm", options=InstallModelIn(), registration_id=""
    )
    svc.models_downloaded["test-llm"] = DownloadedInfo()

    with (
        patch.object(svc, "_sync_models_from_external_ollama", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.ollama_external_service.fetch_from", new_callable=AsyncMock) as mock_fetch,
    ):
        await svc._uninstall_model("default", "test-llm", UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert mock_fetch.call_count == 0
    assert "test-llm" in svc.models_downloaded


@pytest.mark.asyncio
async def test_uninstall_model_does_nothing_for_unknown_model(
    svc: OllamaExternalService, installed: InstalledInfo, deps: dict[str, Any]
) -> None:
    with (
        patch.object(svc, "_sync_models_from_external_ollama", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_save", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
    ):
        await svc._uninstall_model("default", "unknown", UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]
    assert deps["endpoint_registry"].unregister_chat_completion.call_count == 0


def test_add_custom_model_creates_dict_for_new_instance(svc: OllamaExternalService) -> None:
    svc.instances_info["inst2"] = Instance(None, None, {}, InstanceConfig())
    custom = CustomModel(id="custom-1", data={"id": "custom-llm", "size": "1GB", "type": "llm"})

    svc._add_custom_model("inst2", custom)  # pyright: ignore[reportPrivateUsage]

    assert "custom-llm" in svc.models["inst2"]


@pytest.mark.asyncio
async def test_install_instance_loads_default_models_for_new_instance(svc: OllamaExternalService) -> None:
    svc.instances_info["inst2"] = Instance(None, None, {}, InstanceConfig())
    options = InstallServiceIn(spec={"url": "http://localhost:11434"})

    async def mock_fetch(*args: object, **kwargs: object):  # type: ignore[misc]
        yield FetchResult(status_code=200, data=json.dumps({"version": "0.1"}))

    with (
        patch(
            "server.services.ollama_external_service.fetch_from",
            new_callable=AsyncMock,
            return_value=FetchResult(status_code=200, data=json.dumps({"version": "0.1"})),
        ),
        patch.object(svc, "load_default_models") as mock_load,
    ):
        promise = await svc._install_instance("inst2", options)  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert mock_load.call_count == 1
    assert mock_load.call_args == call("inst2")


@pytest.mark.asyncio
async def test_list_models_skips_instances_not_in_filter(svc: OllamaExternalService, installed: InstalledInfo) -> None:
    svc.instances_info["inst2"] = Instance(None, None, {}, InstanceConfig())
    svc.models["inst2"] = {"extra-model": OllamaModel(id="extra-model", size="1GB", type="llm")}

    result = await svc.list_models("default", ListModelsFilters())

    assert not any(m.id == "extra-model" for m in result.list)


@pytest.mark.asyncio
async def test_download_model_tracks_completed_field_progress(svc: OllamaExternalService) -> None:
    stream = MagicMock()
    model = OllamaModel(id="llama3", size="4GB", type="llm")
    records = [
        {"completed": 100, "digest": "sha256:abc"},
        {"completed": 200, "digest": "sha256:abc"},
        {"status": "success"},
    ]

    async def mock_stream(*args: object, **kwargs: object):  # type: ignore[misc]
        yield FetchResult(status_code=200, data="\n".join(json.dumps(r) for r in records))

    with patch("server.services.ollama_external_service.stream_fetch_from", side_effect=mock_stream):
        await svc._download_model(stream, model, "llama3", "http://localhost:11434")  # pyright: ignore[reportPrivateUsage]

    assert stream.emit.call_count >= 2


@pytest.mark.asyncio
async def test_download_model_or_set_progress_breaks_on_non_download_chunk(svc: OllamaExternalService) -> None:
    existing: Stream[StreamChunkProgress] = Stream()  # type: ignore[type-arg]
    finish_chunk: StreamChunkProgress = StreamChunkProgress(type="finish", stage="download", value=1.0, data={})  # type: ignore[arg-type]
    existing.emit(finish_chunk)
    existing.close()
    svc.models_download_progress["llama3"] = existing  # type: ignore[assignment]
    output = MagicMock()
    model = OllamaModel(id="llama3", size="1GB", type="llm")

    await svc._download_model_or_set_progress(output, model, "llama3", "http://localhost:11434")  # pyright: ignore[reportPrivateUsage]

    assert output.emit.call_count == 0


@pytest.mark.asyncio
async def test_uninstall_instance_when_not_installed_does_not_unregister(svc: OllamaExternalService, deps: dict[str, Any]) -> None:
    svc.instances_info["default"].installed = None

    await svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert deps["endpoint_registry"].unregister_chat_completion.call_count == 0
    assert deps["endpoint_registry"].unregister_embeddings.call_count == 0
    assert svc.instances_info["default"].installed is None


@pytest.mark.asyncio
async def test_uninstall_instance_skips_uninstall_model_when_in_other_instance(svc: OllamaExternalService) -> None:
    installed = _make_installed_info()
    installed.models["test-llm"] = ModelInstalledInfo(
        id="test-llm", registered_name="test-llm", type="llm", options=InstallModelIn(), registration_id="reg-1"
    )
    svc.instances_info["default"].installed = installed
    svc.instances_info["inst2"] = Instance(None, None, {}, InstanceConfig())
    svc.instances_info["inst2"].installed = _make_installed_info()
    svc.instances_info["inst2"].installed.models["test-llm"] = ModelInstalledInfo(
        id="test-llm", registered_name="test-llm", type="llm", options=InstallModelIn(), registration_id="reg-2"
    )

    with patch.object(svc, "_uninstall_model", new_callable=AsyncMock) as mock_uninstall:  # pyright: ignore[reportPrivateUsage]
        await svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert mock_uninstall.call_count == 0


@pytest.mark.asyncio
async def test_download_model_zero_size_skips_progress_tracking(svc: OllamaExternalService) -> None:
    stream = MagicMock()
    model = OllamaModel(id="llama3", size="0", type="llm")

    async def mock_stream(*args: object, **kwargs: object):  # type: ignore[misc]
        yield FetchResult(status_code=200, data=json.dumps({"completed": 100, "digest": "sha256:abc"}))

    with patch("server.services.ollama_external_service.stream_fetch_from", side_effect=mock_stream):
        await svc._download_model(stream, model, "llama3", "http://localhost:11434")  # pyright: ignore[reportPrivateUsage]

    assert stream.emit.call_count >= 2


@pytest.mark.asyncio
async def test_download_model_record_with_no_completed_or_success_emits_progress(svc: OllamaExternalService) -> None:
    stream = MagicMock()
    model = OllamaModel(id="llama3", size="4GB", type="llm")
    records = [
        {"status": "pulling manifest"},
    ]

    async def mock_stream(*args: object, **kwargs: object):  # type: ignore[misc]
        yield FetchResult(status_code=200, data="\n".join(json.dumps(r) for r in records))

    with patch("server.services.ollama_external_service.stream_fetch_from", side_effect=mock_stream):
        await svc._download_model(stream, model, "llama3", "http://localhost:11434")  # pyright: ignore[reportPrivateUsage]

    assert stream.emit.call_count >= 2


@pytest.mark.asyncio
async def test_resolve_custom_model_size_returns_formatted_size(svc: OllamaExternalService, deps: dict[str, Any]) -> None:
    with patch("server.services.ollama_external_service.fetch_ollama_ref_bytes", new=AsyncMock(return_value=1024**3)):
        result = await svc._resolve_custom_model_size({"id": "llama3"})  # pyright: ignore[reportPrivateUsage]

    assert result == "1.0 GB"


@pytest.mark.asyncio
async def test_resolve_custom_model_size_returns_none_on_exception(svc: OllamaExternalService, deps: dict[str, Any]) -> None:
    with patch("server.services.ollama_external_service.fetch_ollama_ref_bytes", new=AsyncMock(side_effect=Exception("fail"))):
        result = await svc._resolve_custom_model_size({"id": "llama3"})  # pyright: ignore[reportPrivateUsage]

    assert result is None


@pytest.mark.asyncio
async def test_resolve_custom_model_size_returns_none_when_bytes_is_none(svc: OllamaExternalService, deps: dict[str, Any]) -> None:
    with patch("server.services.ollama_external_service.fetch_ollama_ref_bytes", new=AsyncMock(return_value=None)):
        result = await svc._resolve_custom_model_size({"id": "llama3"})  # pyright: ignore[reportPrivateUsage]

    assert result is None


# ---------------------------------------------------------------------------
# sync_interval validation
# ---------------------------------------------------------------------------


def test_ollama_external_options_rejects_sync_interval_below_10() -> None:
    with pytest.raises(ValidationError):
        OllamaExternalOptions(url="http://localhost:11434", sync_interval=5)


def test_ollama_external_options_accepts_sync_interval_of_10() -> None:
    opts = OllamaExternalOptions(url="http://localhost:11434", sync_interval=10)
    assert opts.sync_interval == 10


# ---------------------------------------------------------------------------
# sync after install / uninstall
# ---------------------------------------------------------------------------

TAGS_RESPONSE = json.dumps({"models": [{"name": "new-model:latest", "size": 1024 * 1024 * 1024, "details": {"family": "llm"}}]})


@pytest.mark.asyncio
async def test_install_model_triggers_sync_for_new_models(svc: OllamaExternalService, installed: InstalledInfo) -> None:
    svc.models["default"]["test-llm"] = OllamaModel(id="test-llm", size="1GB", type="llm")

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_fetch_show_info", new_callable=AsyncMock, return_value=(None, None, None, None)),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.ollama_external_service.fetch_from", new_callable=AsyncMock) as mock_fetch,
        patch.object(svc, "_save", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
    ):
        mock_fetch.return_value = FetchResult(status_code=200, data=TAGS_RESPONSE)
        promise = await svc._install_model("default", "test-llm", InstallModelIn())  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert "new-model" in installed.models


@pytest.mark.asyncio
async def test_uninstall_model_no_purge_triggers_sync(svc: OllamaExternalService, installed: InstalledInfo) -> None:
    installed.models["test-llm"] = ModelInstalledInfo(
        id="test-llm", registered_name="test-llm", type="llm", options=InstallModelIn(), registration_id=""
    )
    svc.models_downloaded["test-llm"] = DownloadedInfo()

    with (
        patch("server.services.ollama_external_service.fetch_from", new_callable=AsyncMock) as mock_fetch,
        patch.object(svc, "_save", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
    ):
        mock_fetch.return_value = FetchResult(status_code=200, data=json.dumps({"models": []}))
        await svc._uninstall_model("default", "test-llm", UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    # fetch_from called once for /api/tags (no purge, no /api/delete)
    assert mock_fetch.call_count == 1
    assert "/api/tags" in mock_fetch.call_args[0][0]


@pytest.mark.asyncio
async def test_uninstall_model_purge_triggers_sync(svc: OllamaExternalService, installed: InstalledInfo) -> None:
    installed.models["test-llm"] = ModelInstalledInfo(
        id="test-llm", registered_name="test-llm", type="llm", options=InstallModelIn(), registration_id=""
    )
    svc.models_downloaded["test-llm"] = DownloadedInfo()

    delete_result = FetchResult(status_code=200, data="")
    tags_result = FetchResult(status_code=200, data=json.dumps({"models": []}))

    with (
        patch("server.services.ollama_external_service.fetch_from", new_callable=AsyncMock) as mock_fetch,
        patch.object(svc, "_save", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
    ):
        mock_fetch.side_effect = [delete_result, tags_result]
        await svc._uninstall_model("default", "test-llm", UninstallModelIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    # first call: /api/delete, second call: /api/tags
    calls = [c[0][0] for c in mock_fetch.call_args_list]
    assert any("/api/delete" in url for url in calls)
    assert any("/api/tags" in url for url in calls)


@pytest.mark.asyncio
async def test_sync_models_public_method_updates_model_list(svc: OllamaExternalService, installed: InstalledInfo) -> None:
    with (
        patch("server.services.ollama_external_service.fetch_from", new_callable=AsyncMock) as mock_fetch,
        patch.object(svc, "_save", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
    ):
        mock_fetch.return_value = FetchResult(
            status_code=200,
            data=json.dumps({"models": [{"name": "synced-model", "size": 512, "details": {"family": "llm"}}]}),
        )
        await svc.sync_models("default")

    assert "synced-model" in installed.models


@pytest.mark.asyncio
async def test_sync_models_on_base_service_is_noop(svc: OllamaExternalService) -> None:
    # base implementation is a no-op — should return without raising
    await Base2Service.sync_models(svc, "default")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_load_instance_returns_early_when_no_options(svc: OllamaExternalService) -> None:
    instance_data = InstanceConfig(options=None)

    with patch.object(svc, "install_instance", new_callable=AsyncMock) as mock_install:
        await svc.load_instance("default", instance_data)

    assert mock_install.call_count == 0


@pytest.mark.asyncio
async def test_load_instance_installs_and_sets_config_models(svc: OllamaExternalService) -> None:
    options = InstallServiceIn(spec={"url": "http://localhost:11434"})
    instance_data = InstanceConfig(options=options)
    installed_info = _make_installed_info()
    installed_info.models["synced-llm"] = ModelInstalledInfo(
        id="synced-llm", registered_name="synced-llm", type="llm", options=InstallModelIn(), registration_id="reg-x"
    )

    async def mock_install(instance: str, opts: object, *args: object, **kwargs: object) -> MagicMock:
        svc.instances_info[instance].installed = installed_info
        promise = MagicMock()
        promise.wait = AsyncMock(return_value=installed_info)
        return promise

    with (
        patch.object(svc, "install_instance", side_effect=mock_install),
        patch.object(svc, "_save", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
    ):
        await svc.load_instance("default", instance_data)

    assert svc.instances_info["default"].config.models is not None
    assert len(svc.instances_info["default"].config.models) == 1
    assert svc.instances_info["default"].config.models[0].model_id == "synced-llm"


@pytest.mark.asyncio
async def test_load_instance_adds_custom_models(svc: OllamaExternalService) -> None:
    options = InstallServiceIn(spec={"url": "http://localhost:11434"})
    custom = CustomModel(id="c-1", data={"id": "my-custom", "size": "1GB", "type": "llm"})
    instance_data = InstanceConfig(options=options, custom=[custom])

    async def mock_install(instance: str, opts: object, *args: object, **kwargs: object) -> MagicMock:
        promise = MagicMock()
        promise.wait = AsyncMock(return_value=None)
        return promise

    with (
        patch.object(svc, "install_instance", side_effect=mock_install),
        patch.object(svc, "_save", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
    ):
        await svc.load_instance("default", instance_data)

    assert "my-custom" in svc.models["default"]


def test_determine_model_type_returns_type_from_instance_models(svc: OllamaExternalService) -> None:
    svc.models["default"]["custom-emb"] = OllamaModel(id="custom-emb", size="1GB", type="embedding")

    result = svc._determine_model_type("custom-emb", "default")  # pyright: ignore[reportPrivateUsage]

    assert result == "embedding"


def test_determine_model_type_falls_back_to_const_when_instance_unknown(svc: OllamaExternalService) -> None:
    const_model_id = next(iter(_const.models))
    expected_type = _const.models[const_model_id].type

    result = svc._determine_model_type(const_model_id, "nonexistent-instance")  # pyright: ignore[reportPrivateUsage]

    assert result == expected_type


@pytest.mark.asyncio
async def test_download_model_interleaved_digests_progress_never_decreases(svc: OllamaExternalService) -> None:
    # Regression test for issue #464 root cause 2:
    # Ollama streams progress for multiple layers in parallel, interleaving digest records.
    # The old single-(last_diggest, last_value) tracker treated every digest switch as a
    # fresh start (increment = full completed value), overcounting bytes and saturating the bar
    # prematurely — then a later correction caused a visible regression.
    stream = MagicMock()
    model = OllamaModel(id="llama3", size="3GB", type="llm")
    records = [
        {"status": "pulling", "digest": "sha256:aaa", "completed": 1_000_000_000},  # layer A: 1 GB
        {"status": "pulling", "digest": "sha256:bbb", "completed": 500_000_000},  # layer B: 0.5 GB
        {"status": "pulling", "digest": "sha256:aaa", "completed": 2_000_000_000},  # layer A delta: +1 GB
        {"status": "pulling", "digest": "sha256:bbb", "completed": 1_000_000_000},  # layer B delta: +0.5 GB
        {"status": "success"},
    ]
    data = "\n".join(json.dumps(r) for r in records)

    async def mock_stream(*args: object, **kwargs: object):  # type: ignore[misc]
        yield FetchResult(status_code=200, data=data)

    with patch("server.services.ollama_external_service.stream_fetch_from", side_effect=mock_stream):
        await svc._download_model(stream, model, "llama3", "http://localhost:11434")  # pyright: ignore[reportPrivateUsage]

    progress_values = [
        call.args[0]["value"]
        for call in stream.emit.call_args_list
        if isinstance(call.args[0], dict) and call.args[0].get("type") == "progress"
    ]
    assert progress_values, "no progress events emitted"
    for prev, curr in itertools.pairwise(progress_values):
        assert curr >= prev, f"progress went backward: {prev} → {curr}"
    assert all(v <= 1.0 for v in progress_values)


@pytest.mark.asyncio
async def test_download_model_duplicate_completed_value_not_double_counted(svc: OllamaExternalService) -> None:
    # Covers the increment == 0 branch: same completed value sent twice for a digest must not
    # add anything to progress (increment = max(0, value - last_values[digest]) == 0).
    stream = MagicMock()
    model = OllamaModel(id="llama3", size="1GB", type="llm")
    records = [
        {"status": "pulling", "digest": "sha256:aaa", "completed": 500_000_000},
        {"status": "pulling", "digest": "sha256:aaa", "completed": 500_000_000},  # duplicate — increment=0
        {"status": "success"},
    ]
    data = "\n".join(json.dumps(r) for r in records)

    async def mock_stream(*args: object, **kwargs: object):  # type: ignore[misc]
        yield FetchResult(status_code=200, data=data)

    with patch("server.services.ollama_external_service.stream_fetch_from", side_effect=mock_stream):
        await svc._download_model(stream, model, "llama3", "http://localhost:11434")  # pyright: ignore[reportPrivateUsage]

    progress_values = [
        call.args[0]["value"]
        for call in stream.emit.call_args_list
        if isinstance(call.args[0], dict) and call.args[0].get("type") == "progress"
    ]
    assert progress_values, "no progress events emitted"
    # Both records for sha256:aaa report the same completed value — second must not advance progress.
    for prev, curr in itertools.pairwise(progress_values):
        assert curr >= prev, f"progress went backward: {prev} → {curr}"


@pytest.mark.asyncio
async def test_download_model_interleaved_digests_correct_total(svc: OllamaExternalService) -> None:
    # Each digest's contribution should be counted exactly once — no overcounting.
    stream = MagicMock()
    model = OllamaModel(id="llama3", size="3GB", type="llm")
    records = [
        {"status": "pulling", "digest": "sha256:aaa", "completed": 1_000_000_000},
        {"status": "pulling", "digest": "sha256:bbb", "completed": 500_000_000},
        {"status": "pulling", "digest": "sha256:aaa", "completed": 2_000_000_000},
        {"status": "pulling", "digest": "sha256:bbb", "completed": 1_000_000_000},
    ]
    data = "\n".join(json.dumps(r) for r in records)

    async def mock_stream(*args: object, **kwargs: object):  # type: ignore[misc]
        yield FetchResult(status_code=200, data=data)

    with patch("server.services.ollama_external_service.stream_fetch_from", side_effect=mock_stream):
        await svc._download_model(stream, model, "llama3", "http://localhost:11434")  # pyright: ignore[reportPrivateUsage]

    # With overcounting the bar would saturate at 1.0 after just the first two records.
    # There should be at least one intermediate event below 1.0.
    progress_values = [
        call.args[0]["value"]
        for call in stream.emit.call_args_list
        if isinstance(call.args[0], dict) and call.args[0].get("type") == "progress" and call.args[0]["value"] < 1.0
    ]
    assert progress_values, "progress jumped straight to 1.0 — overcounting likely"


@pytest.mark.asyncio
async def test_start_sync_task_cancels_existing_task(svc: OllamaExternalService) -> None:
    async def never_ends() -> None:
        await asyncio.sleep(9999)

    first_task = asyncio.create_task(never_ends())
    svc._sync_tasks["default"] = first_task  # pyright: ignore[reportPrivateUsage]

    with patch("asyncio.sleep", new_callable=AsyncMock):
        svc._start_sync_task("default", 1)  # pyright: ignore[reportPrivateUsage]

    await asyncio.sleep(0)
    assert first_task.cancelled()


@pytest.mark.asyncio
async def test_start_sync_task_loop_breaks_when_not_installed(svc: OllamaExternalService) -> None:
    with patch("asyncio.sleep", new_callable=AsyncMock):
        svc._start_sync_task("default", 1)  # pyright: ignore[reportPrivateUsage]
        task = svc._sync_tasks["default"]  # pyright: ignore[reportPrivateUsage]

    await task


@pytest.mark.asyncio
async def test_start_sync_task_loop_runs_sync_and_breaks(svc: OllamaExternalService, installed: InstalledInfo) -> None:
    call_count = 0

    async def mock_sleep(_: object) -> None:
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            svc.instances_info["default"].installed = None

    with (
        patch("asyncio.sleep", side_effect=mock_sleep),
        patch.object(svc, "_sync_models_from_external_ollama", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_refresh_embedding_token_limits", new_callable=AsyncMock) as refresh,  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_save", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
    ):
        svc._start_sync_task("default", 1)  # pyright: ignore[reportPrivateUsage]
        await svc._sync_tasks["default"]  # pyright: ignore[reportPrivateUsage]

    assert refresh.call_args == call("default", installed)


@pytest.mark.asyncio
async def test_start_sync_task_loop_swallows_exception(svc: OllamaExternalService, installed: InstalledInfo) -> None:
    call_count = 0

    async def mock_sleep(_: object) -> None:
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            svc.instances_info["default"].installed = None

    with (
        patch("asyncio.sleep", side_effect=mock_sleep),
        patch.object(
            svc,
            "_sync_models_from_external_ollama",  # pyright: ignore[reportPrivateUsage]
            new_callable=AsyncMock,
            side_effect=Exception("sync error"),
        ),
        patch.object(svc, "_save", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
    ):
        svc._start_sync_task("default", 1)  # pyright: ignore[reportPrivateUsage]
        await svc._sync_tasks["default"]  # pyright: ignore[reportPrivateUsage]


def test_register_synced_model_embedding_registers_proxy(
    svc: OllamaExternalService, installed: InstalledInfo, deps: dict[str, Any]
) -> None:
    svc.models["default"]["emb-from-ollama"] = OllamaModel(id="emb-from-ollama", size="500MB", type="embedding")

    svc._register_synced_model("default", installed, "emb-from-ollama", 512 * 1024 * 1024)  # pyright: ignore[reportPrivateUsage]

    assert deps["endpoint_registry"].register_embeddings_as_proxy.call_count == 1
    assert "emb-from-ollama" in installed.models
    assert installed.models["emb-from-ollama"].type == "embedding"


def test_remove_stale_models_unregisters_embedding(svc: OllamaExternalService, installed: InstalledInfo, deps: dict[str, Any]) -> None:
    installed.models["stale-emb"] = ModelInstalledInfo(
        id="stale-emb", registered_name="stale-emb", type="embedding", options=InstallModelIn(), registration_id="reg-stale"
    )

    svc._remove_stale_models(installed, {"stale-emb"})  # pyright: ignore[reportPrivateUsage]

    assert deps["endpoint_registry"].unregister_embeddings.call_count == 1
    assert deps["endpoint_registry"].unregister_embeddings.call_args == call("stale-emb", "reg-stale")
    assert "stale-emb" not in installed.models


def test_remove_stale_models_skips_unregister_when_model_not_in_installed(
    svc: OllamaExternalService, installed: InstalledInfo, deps: dict[str, Any]
) -> None:
    svc.models_downloaded["phantom"] = DownloadedInfo()

    svc._remove_stale_models(installed, {"phantom"})  # pyright: ignore[reportPrivateUsage]

    assert deps["endpoint_registry"].unregister_chat_completion.call_count == 0
    assert deps["endpoint_registry"].unregister_embeddings.call_count == 0
    assert "phantom" not in svc.models_downloaded


@pytest.mark.asyncio
async def test_sync_models_returns_early_on_non_200(svc: OllamaExternalService, installed: InstalledInfo) -> None:
    with patch("server.services.ollama_external_service.fetch_from", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = FetchResult(status_code=503, data="")
        await svc._sync_models_from_external_ollama("default", installed)  # pyright: ignore[reportPrivateUsage]

    assert len(installed.models) == 0


@pytest.mark.asyncio
async def test_sync_models_skips_already_installed_model(
    svc: OllamaExternalService, installed: InstalledInfo, deps: dict[str, Any]
) -> None:
    installed.models["already-installed"] = ModelInstalledInfo(
        id="already-installed", registered_name="already-installed", type="llm", options=InstallModelIn(), registration_id="reg-x"
    )
    tags = json.dumps({"models": [{"name": "already-installed:latest", "size": 1024}]})

    with patch("server.services.ollama_external_service.fetch_from", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = FetchResult(status_code=200, data=tags)
        await svc._sync_models_from_external_ollama("default", installed)  # pyright: ignore[reportPrivateUsage]

    assert deps["endpoint_registry"].register_chat_completion_as_proxy.call_count == 0


@pytest.mark.asyncio
async def test_sync_models_skips_already_downloaded_on_non_initial_sync(
    svc: OllamaExternalService, installed: InstalledInfo, deps: dict[str, Any]
) -> None:
    svc.models_downloaded["pre-downloaded"] = DownloadedInfo()
    tags = json.dumps({"models": [{"name": "pre-downloaded:latest", "size": 1024}]})

    with patch("server.services.ollama_external_service.fetch_from", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = FetchResult(status_code=200, data=tags)
        await svc._sync_models_from_external_ollama("default", installed, is_initial_sync=False)  # pyright: ignore[reportPrivateUsage]

    assert deps["endpoint_registry"].register_chat_completion_as_proxy.call_count == 0


SHOW_RESPONSE = json.dumps(
    {
        "model_info": {
            "general.architecture": "llama",
            "llama.context_length": 131072,
            "llama.embedding_length": 4096,
        }
    }
)


@pytest.mark.asyncio
async def test_fetch_show_info_returns_context_length(svc: OllamaExternalService) -> None:
    with patch("server.services.ollama_external_service.fetch_from", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = FetchResult(status_code=200, data=SHOW_RESPONSE)
        context_length, supports_vision, supports_audio, _ = await svc._fetch_show_info(  # pyright: ignore[reportPrivateUsage]
            "http://localhost:11434", "llama3"
        )

    assert context_length == 131072
    assert supports_vision is None
    assert supports_audio is None


@pytest.mark.asyncio
async def test_fetch_show_info_returns_none_on_bad_status(svc: OllamaExternalService) -> None:
    with patch("server.services.ollama_external_service.fetch_from", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = FetchResult(status_code=404, data="")
        result = await svc._fetch_show_info("http://localhost:11434", "unknown")  # pyright: ignore[reportPrivateUsage]

    assert result == (None, None, None, None)


@pytest.mark.asyncio
async def test_fetch_show_info_returns_none_on_exception(svc: OllamaExternalService) -> None:
    with patch("server.services.ollama_external_service.fetch_from", side_effect=ConnectionError("refused")):
        result = await svc._fetch_show_info("http://localhost:11434", "llama3")  # pyright: ignore[reportPrivateUsage]

    assert result == (None, None, None, None)


@pytest.mark.asyncio
async def test_fetch_show_info_returns_none_context_length_when_field_missing(svc: OllamaExternalService) -> None:
    data = json.dumps({"model_info": {"general.architecture": "llama"}})
    with patch("server.services.ollama_external_service.fetch_from", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = FetchResult(status_code=200, data=data)
        context_length, _, _, _ = await svc._fetch_show_info("http://localhost:11434", "llama3")  # pyright: ignore[reportPrivateUsage]

    assert context_length is None


@pytest.mark.asyncio
async def test_fetch_show_info_parses_capabilities(svc: OllamaExternalService) -> None:
    data = json.dumps({"model_info": {"general.architecture": "llama"}, "capabilities": ["completion", "vision"]})
    with patch("server.services.ollama_external_service.fetch_from", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = FetchResult(status_code=200, data=data)
        _, supports_vision, supports_audio, _ = await svc._fetch_show_info("http://localhost:11434", "llama3")  # pyright: ignore[reportPrivateUsage]

    assert supports_vision is True
    assert supports_audio is None


def test_register_synced_model_passes_context_length_to_props(
    svc: OllamaExternalService, installed: InstalledInfo, deps: dict[str, Any]
) -> None:
    svc.models["default"]["ctx-llm"] = OllamaModel(id="ctx-llm", size="4GB", type="llm")

    svc._register_synced_model("default", installed, "ctx-llm", 0, context_length=8192)  # pyright: ignore[reportPrivateUsage]

    call_kwargs = deps["endpoint_registry"].register_chat_completion_as_proxy.call_args.kwargs
    assert call_kwargs["props"].context_window == 8192
    assert call_kwargs["props"].max_context_window == 8192


def test_register_synced_model_none_context_length_leaves_props_none(
    svc: OllamaExternalService, installed: InstalledInfo, deps: dict[str, Any]
) -> None:
    svc.models["default"]["no-ctx-llm"] = OllamaModel(id="no-ctx-llm", size="4GB", type="llm")

    svc._register_synced_model("default", installed, "no-ctx-llm", 0)  # pyright: ignore[reportPrivateUsage]

    call_kwargs = deps["endpoint_registry"].register_chat_completion_as_proxy.call_args.kwargs
    assert call_kwargs["props"].context_window is None
    assert call_kwargs["props"].max_context_window is None


@pytest.mark.asyncio
async def test_sync_models_fetches_and_passes_context_length(
    svc: OllamaExternalService, installed: InstalledInfo, deps: dict[str, Any]
) -> None:
    tags = json.dumps({"models": [{"name": "ctx-model:latest", "size": 1024}]})

    with (
        patch("server.services.ollama_external_service.fetch_from", new_callable=AsyncMock) as mock_fetch,
        patch.object(svc, "_fetch_show_info", new_callable=AsyncMock, return_value=(32768, None, None, None)) as mock_ctx,  # pyright: ignore[reportPrivateUsage]
    ):
        mock_fetch.return_value = FetchResult(status_code=200, data=tags)
        await svc._sync_models_from_external_ollama("default", installed)  # pyright: ignore[reportPrivateUsage]

    mock_ctx.assert_called_once_with(installed.base_url, "ctx-model")
    call_kwargs = deps["endpoint_registry"].register_chat_completion_as_proxy.call_args.kwargs
    assert call_kwargs["props"].context_window == 32768
    assert call_kwargs["props"].max_context_window == 32768


@pytest.mark.asyncio
async def test_sync_models_retries_capabilities_for_already_installed_model_with_unknown_values(
    svc: OllamaExternalService, installed: InstalledInfo, deps: dict[str, Any]
) -> None:
    """A model already registered with both capabilities unresolved must be retried, not skipped forever."""
    installed.models["vlm"] = ModelInstalledInfo(
        id="vlm",
        registered_name="vlm",
        type="llm",
        options=InstallModelIn(),
        registration_id="reg-vlm",
        supports_vision=None,
        supports_audio=None,
    )
    tags = json.dumps({"models": [{"name": "vlm:latest", "size": 1024}]})

    with (
        patch("server.services.ollama_external_service.fetch_from", new_callable=AsyncMock) as mock_fetch,
        patch.object(svc, "_fetch_show_info", new_callable=AsyncMock, return_value=(None, True, None, None)) as mock_show,  # pyright: ignore[reportPrivateUsage]
    ):
        mock_fetch.return_value = FetchResult(status_code=200, data=tags)
        await svc._sync_models_from_external_ollama("default", installed)  # pyright: ignore[reportPrivateUsage]

    mock_show.assert_called_once_with(installed.base_url, "vlm")
    assert installed.models["vlm"].supports_vision is True
    assert installed.models["vlm"].supports_audio is None
    deps["endpoint_registry"].update_modality_support.assert_called_once_with("reg-vlm", supports_vision=True, supports_audio=None)
    # Retrying capabilities must not re-register the already-installed model.
    assert deps["endpoint_registry"].register_chat_completion_as_proxy.call_count == 0


@pytest.mark.asyncio
async def test_sync_models_skips_capability_retry_when_already_resolved(
    svc: OllamaExternalService, installed: InstalledInfo, deps: dict[str, Any]
) -> None:
    """At least one resolved value means don't keep re-hitting the network on every sync."""
    installed.models["known"] = ModelInstalledInfo(
        id="known",
        registered_name="known",
        type="llm",
        options=InstallModelIn(),
        registration_id="reg-known",
        supports_vision=False,
        supports_audio=None,
    )
    tags = json.dumps({"models": [{"name": "known:latest", "size": 1024}]})

    with (
        patch("server.services.ollama_external_service.fetch_from", new_callable=AsyncMock) as mock_fetch,
        patch.object(svc, "_fetch_show_info", new_callable=AsyncMock) as mock_show,  # pyright: ignore[reportPrivateUsage]
    ):
        mock_fetch.return_value = FetchResult(status_code=200, data=tags)
        await svc._sync_models_from_external_ollama("default", installed)  # pyright: ignore[reportPrivateUsage]

    mock_show.assert_not_called()
    deps["endpoint_registry"].update_modality_support.assert_not_called()


@pytest.mark.asyncio
async def test_sync_models_retries_capabilities_without_registration_id(
    svc: OllamaExternalService, installed: InstalledInfo, deps: dict[str, Any]
) -> None:
    """A model without a registration_id (not yet registered in the endpoint registry) still gets its
    capability fields updated locally, but must not be pushed to the endpoint registry."""
    installed.models["unregistered"] = ModelInstalledInfo(
        id="unregistered",
        registered_name="unregistered",
        type="llm",
        options=InstallModelIn(),
        registration_id="",
        supports_vision=None,
        supports_audio=None,
    )
    tags = json.dumps({"models": [{"name": "unregistered:latest", "size": 1024}]})

    with (
        patch("server.services.ollama_external_service.fetch_from", new_callable=AsyncMock) as mock_fetch,
        patch.object(svc, "_fetch_show_info", new_callable=AsyncMock, return_value=(None, True, None, None)) as mock_show,  # pyright: ignore[reportPrivateUsage]
    ):
        mock_fetch.return_value = FetchResult(status_code=200, data=tags)
        await svc._sync_models_from_external_ollama("default", installed)  # pyright: ignore[reportPrivateUsage]

    mock_show.assert_called_once_with(installed.base_url, "unregistered")
    assert installed.models["unregistered"].supports_vision is True
    assert installed.models["unregistered"].supports_audio is None
    deps["endpoint_registry"].update_modality_support.assert_not_called()


def test_register_synced_model_forwards_modality_support(
    svc: OllamaExternalService, installed: InstalledInfo, deps: dict[str, Any]
) -> None:
    svc.models["default"]["vlm"] = OllamaModel(id="vlm", size="4GB", type="llm")

    svc._register_synced_model(  # pyright: ignore[reportPrivateUsage]
        "default", installed, "vlm", 0, supports_vision=True, supports_audio=None
    )

    call_kwargs = deps["endpoint_registry"].register_chat_completion_as_proxy.call_args.kwargs
    assert call_kwargs["props"].supports_vision is True
    assert call_kwargs["props"].supports_audio is None


@pytest.mark.asyncio
async def test_install_model_passes_context_length_to_props(
    svc: OllamaExternalService, installed: InstalledInfo, deps: dict[str, Any]
) -> None:
    svc.models["default"]["ctx-llm"] = OllamaModel(id="ctx-llm", size="4GB", type="llm")

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_sync_models_from_external_ollama", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_fetch_show_info", new_callable=AsyncMock, return_value=(65536, None, None, None)),  # pyright: ignore[reportPrivateUsage]
    ):
        promise = await svc._install_model("default", "ctx-llm", InstallModelIn())  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    call_kwargs = deps["endpoint_registry"].register_chat_completion_as_proxy.call_args.kwargs
    assert call_kwargs["props"].context_window == 65536
    assert call_kwargs["props"].max_context_window == 65536


@pytest.mark.asyncio
async def test_uninstall_instance_cancels_active_sync_task(svc: OllamaExternalService) -> None:
    svc.instances_info["default"].installed = _make_installed_info()

    async def never_ends() -> None:
        await asyncio.sleep(9999)

    task = asyncio.create_task(never_ends())
    svc._sync_tasks["default"] = task  # pyright: ignore[reportPrivateUsage]

    with patch.object(svc, "_uninstall_model", new_callable=AsyncMock):  # pyright: ignore[reportPrivateUsage]
        await svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    await asyncio.sleep(0)
    assert task.cancelled()


@pytest.mark.asyncio
async def test_install_model_registration_failure_rolls_back_model_without_unregister(
    svc: OllamaExternalService, installed: InstalledInfo, deps: dict[str, Any]
) -> None:
    svc.models["default"]["test-llm"] = OllamaModel(id="test-llm", size="1GB", type="llm")
    deps["endpoint_registry"].register_chat_completion_as_proxy.side_effect = RuntimeError("registry down")

    with patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock):  # pyright: ignore[reportPrivateUsage]
        promise = await svc._install_model("default", "test-llm", InstallModelIn())  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(RuntimeError):
            await promise.wait()

    assert "test-llm" not in installed.models
    assert deps["endpoint_registry"].unregister_chat_completion.call_count == 0


@pytest.mark.asyncio
async def test_install_model_post_registration_failure_unregisters_and_rolls_back(
    svc: OllamaExternalService, installed: InstalledInfo, deps: dict[str, Any]
) -> None:
    svc.models["default"]["test-llm"] = OllamaModel(id="test-llm", size="1GB", type="llm")
    deps["endpoint_registry"].register_chat_completion_as_proxy.return_value = "reg-llm"

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_sync_models_from_external_ollama", new_callable=AsyncMock, side_effect=RuntimeError("sync failed")),  # pyright: ignore[reportPrivateUsage]
    ):
        promise = await svc._install_model("default", "test-llm", InstallModelIn())  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(RuntimeError):
            await promise.wait()

    assert "test-llm" not in installed.models
    assert deps["endpoint_registry"].unregister_chat_completion.call_count == 1
    assert "default" not in svc._sync_tasks  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_install_model_post_registration_failure_unregisters_embedding_and_rolls_back(
    svc: OllamaExternalService, installed: InstalledInfo, deps: dict[str, Any]
) -> None:
    svc.models["default"]["test-emb"] = OllamaModel(id="test-emb", size="500MB", type="embedding")
    deps["endpoint_registry"].register_embeddings_as_proxy.return_value = "reg-emb"

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_sync_models_from_external_ollama", new_callable=AsyncMock, side_effect=RuntimeError("sync failed")),  # pyright: ignore[reportPrivateUsage]
    ):
        promise = await svc._install_model("default", "test-emb", InstallModelIn())  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(RuntimeError):
            await promise.wait()

    assert "test-emb" not in installed.models
    assert deps["endpoint_registry"].unregister_embeddings.call_count == 1


@pytest.mark.asyncio
async def test_install_model_registration_failure_skips_rollback_when_already_removed(
    svc: OllamaExternalService, installed: InstalledInfo, deps: dict[str, Any]
) -> None:
    svc.models["default"]["test-llm"] = OllamaModel(id="test-llm", size="1GB", type="llm")

    def side_effect(*args: object, **kwargs: object) -> None:
        installed.models.pop("test-llm", None)
        raise RuntimeError("registry down")

    deps["endpoint_registry"].register_chat_completion_as_proxy.side_effect = side_effect

    with patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock):  # pyright: ignore[reportPrivateUsage]
        promise = await svc._install_model("default", "test-llm", InstallModelIn())  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(RuntimeError):
            await promise.wait()

    assert "test-llm" not in installed.models


@pytest.mark.asyncio
async def test_sync_warm_state_marks_loaded_model_warm(installed: InstalledInfo, svc: OllamaExternalService, deps: dict[str, Any]) -> None:
    installed.models["test-llm"] = ModelInstalledInfo(
        id="test-llm", registered_name="test-llm", type="llm", options=InstallModelIn(spec={}), registration_id="reg-1"
    )

    with patch("server.services.ollama_external_service.fetch_from", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = FetchResult(status_code=200, data=json.dumps({"models": [{"name": "test-llm"}]}))
        await svc._sync_warm_state(installed)  # pyright: ignore[reportPrivateUsage]

    deps["endpoint_registry"].update_warm.assert_called_once_with("reg-1", True)


@pytest.mark.asyncio
async def test_sync_warm_state_marks_evicted_model_cold(installed: InstalledInfo, svc: OllamaExternalService, deps: dict[str, Any]) -> None:
    installed.models["test-llm"] = ModelInstalledInfo(
        id="test-llm", registered_name="test-llm", type="llm", options=InstallModelIn(spec={}), registration_id="reg-1"
    )

    with patch("server.services.ollama_external_service.fetch_from", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = FetchResult(status_code=200, data=json.dumps({"models": []}))
        await svc._sync_warm_state(installed)  # pyright: ignore[reportPrivateUsage]

    deps["endpoint_registry"].update_warm.assert_called_once_with("reg-1", False)


@pytest.mark.asyncio
async def test_sync_warm_state_skips_model_without_registration_id(
    installed: InstalledInfo, svc: OllamaExternalService, deps: dict[str, Any]
) -> None:
    installed.models["test-llm"] = ModelInstalledInfo(
        id="test-llm", registered_name="test-llm", type="llm", options=InstallModelIn(spec={}), registration_id=""
    )

    with patch("server.services.ollama_external_service.fetch_from", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = FetchResult(status_code=200, data=json.dumps({"models": [{"name": "test-llm"}]}))
        await svc._sync_warm_state(installed)  # pyright: ignore[reportPrivateUsage]

    deps["endpoint_registry"].update_warm.assert_not_called()


@pytest.mark.asyncio
async def test_sync_warm_state_no_op_when_poll_fails(installed: InstalledInfo, svc: OllamaExternalService, deps: dict[str, Any]) -> None:
    installed.models["test-llm"] = ModelInstalledInfo(
        id="test-llm", registered_name="test-llm", type="llm", options=InstallModelIn(spec={}), registration_id="reg-1"
    )

    with patch("server.services.ollama_external_service.fetch_from", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = FetchResult(status_code=503, data="")
        await svc._sync_warm_state(installed)  # pyright: ignore[reportPrivateUsage]

    deps["endpoint_registry"].update_warm.assert_not_called()


@pytest.mark.asyncio
async def test_get_loaded_model_names_returns_none_on_exception(svc: OllamaExternalService) -> None:
    with patch("server.services.ollama_external_service.fetch_from", side_effect=ConnectionError("refused")):
        result = await svc._get_loaded_model_names("http://localhost:11434")  # pyright: ignore[reportPrivateUsage]

    assert result is None


@pytest.mark.asyncio
async def test_start_warmth_poll_task_restarts_existing_task(svc: OllamaExternalService, installed: InstalledInfo) -> None:
    svc._start_warmth_poll_task("default")  # pyright: ignore[reportPrivateUsage]
    first_task = svc._warmth_poll_tasks["default"]  # pyright: ignore[reportPrivateUsage]

    svc._start_warmth_poll_task("default")  # pyright: ignore[reportPrivateUsage]
    second_task = svc._warmth_poll_tasks["default"]  # pyright: ignore[reportPrivateUsage]

    await asyncio.gather(first_task, return_exceptions=True)
    assert first_task is not second_task
    assert first_task.cancelled()
    assert svc._warmth_poll_tasks["default"] is second_task  # pyright: ignore[reportPrivateUsage]

    svc._stop_warmth_poll_task("default")  # pyright: ignore[reportPrivateUsage]
    await asyncio.gather(second_task, return_exceptions=True)
    assert second_task.cancelled()
    assert "default" not in svc._warmth_poll_tasks  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_warmth_poll_loop_ticks_and_stops_once_instance_uninstalled(svc: OllamaExternalService, installed: InstalledInfo) -> None:
    calls = 0

    async def fake_sync(_info: InstalledInfo) -> None:
        nonlocal calls
        calls += 1
        svc.instances_info["default"].installed = None

    with (
        patch("server.services.ollama_external_service.asyncio.sleep", new_callable=AsyncMock),
        patch.object(svc, "_sync_warm_state", new=AsyncMock(side_effect=fake_sync)),  # pyright: ignore[reportPrivateUsage]
    ):
        svc._start_warmth_poll_task("default")  # pyright: ignore[reportPrivateUsage]
        task = svc._warmth_poll_tasks["default"]  # pyright: ignore[reportPrivateUsage]
        await asyncio.wait_for(task, timeout=1)

    assert calls == 1


@pytest.mark.asyncio
async def test_warmth_poll_loop_breaks_when_not_installed(svc: OllamaExternalService) -> None:
    with patch("server.services.ollama_external_service.asyncio.sleep", new_callable=AsyncMock):
        svc._start_warmth_poll_task("default")  # pyright: ignore[reportPrivateUsage]
        task = svc._warmth_poll_tasks["default"]  # pyright: ignore[reportPrivateUsage]
        await asyncio.wait_for(task, timeout=1)


@pytest.mark.asyncio
async def test_warmth_poll_loop_logs_and_continues_on_tick_exception(svc: OllamaExternalService, installed: InstalledInfo) -> None:
    calls = 0

    async def fake_sync(_info: InstalledInfo) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("boom")
        svc.instances_info["default"].installed = None

    with (
        patch("server.services.ollama_external_service.asyncio.sleep", new_callable=AsyncMock),
        patch.object(svc, "_sync_warm_state", new=AsyncMock(side_effect=fake_sync)),  # pyright: ignore[reportPrivateUsage]
    ):
        svc._start_warmth_poll_task("default")  # pyright: ignore[reportPrivateUsage]
        task = svc._warmth_poll_tasks["default"]  # pyright: ignore[reportPrivateUsage]
        await asyncio.wait_for(task, timeout=1)

    assert calls == 2


@pytest.mark.asyncio
async def test_warmth_poll_task_logs_error_on_unexpected_death(svc: OllamaExternalService, installed: InstalledInfo) -> None:
    with (
        patch("server.services.ollama_external_service.asyncio.sleep", new=AsyncMock(side_effect=RuntimeError("boom"))),
        patch("server.services.ollama_external_service.logger") as mock_logger,
    ):
        svc._start_warmth_poll_task("default")  # pyright: ignore[reportPrivateUsage]
        task = svc._warmth_poll_tasks["default"]  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(RuntimeError):
            await task

    assert mock_logger.error.call_count == 1
    assert "default" not in svc._warmth_poll_tasks  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_stop_instance_stops_warmth_poll_task(svc: OllamaExternalService) -> None:
    async def never_ends() -> None:
        await asyncio.sleep(9999)

    task = asyncio.create_task(never_ends())
    svc._warmth_poll_tasks["default"] = task  # pyright: ignore[reportPrivateUsage]

    await svc.stop_instance("default")

    await asyncio.sleep(0)
    assert task.cancelled()
    assert "default" not in svc._warmth_poll_tasks  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_uninstall_instance_cancels_active_warmth_poll_task(svc: OllamaExternalService) -> None:
    svc.instances_info["default"].installed = _make_installed_info()

    async def never_ends() -> None:
        await asyncio.sleep(9999)

    task = asyncio.create_task(never_ends())
    svc._warmth_poll_tasks["default"] = task  # pyright: ignore[reportPrivateUsage]

    with patch.object(svc, "_uninstall_model", new_callable=AsyncMock):  # pyright: ignore[reportPrivateUsage]
        await svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    await asyncio.sleep(0)
    assert task.cancelled()
    assert "default" not in svc._warmth_poll_tasks  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_install_instance_starts_warmth_poll_task(svc: OllamaExternalService) -> None:
    with patch("server.services.ollama_external_service.fetch_from", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = FetchResult(status_code=200, data=json.dumps({"version": "0.5.0"}))
        with (
            patch.object(svc, "_sync_models_from_external_ollama", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
            patch.object(svc, "_start_warmth_poll_task") as mock_start,  # pyright: ignore[reportPrivateUsage]
        ):
            promise = await svc.install_instance("default", InstallServiceIn(spec={"url": "http://localhost:11434"}), None)
            await promise.wait()

    mock_start.assert_called_once_with("default")


def _make_decision_installed_info(model_id: str = "nimble") -> ModelInstalledInfo:
    return ModelInstalledInfo(id=model_id, registered_name=model_id, type="decision", options=InstallModelIn(), registration_id="reg-dec")


def _write_ollama_catalog(tmp_path: Path, catalog: dict[str, Any]) -> None:
    (tmp_path / "static").mkdir()
    (tmp_path / "static" / "ollama-min.json").write_text(json.dumps(catalog), encoding="utf-8")


@patch("server.services.ollama_external_service.get_main_dir")
def test_read_models_loads_decision_category(mock_get_main_dir: MagicMock, tmp_path: Path) -> None:
    _write_ollama_catalog(tmp_path, {"llms": [], "embeddings": [], "decision": [{"name": "tev1:0.8b", "size": "834MB"}]})
    mock_get_main_dir.return_value = tmp_path

    models = _read_models()

    assert models["tev1:0.8b"].type == "decision"


@patch("server.services.ollama_external_service.get_main_dir")
def test_read_models_accepts_catalog_without_decision_category(mock_get_main_dir: MagicMock, tmp_path: Path) -> None:
    _write_ollama_catalog(tmp_path, {"llms": [{"name": "llama3.2", "size": "2GB"}], "embeddings": []})
    mock_get_main_dir.return_value = tmp_path

    models = _read_models()

    assert {model_id: model.type for model_id, model in models.items()} == {"llama3.2": "llm"}


def test_read_models_types_bundled_nimble_as_decision() -> None:
    assert _const.models["nimble"].type == "decision"


@pytest.mark.asyncio
@pytest.mark.parametrize(("capabilities", "expected"), [(["decision"], True), (["completion", "tools"], False)])
@patch("server.services.ollama_external_service.fetch_from", new_callable=AsyncMock)
async def test_fetch_show_info_reports_decision_capability(
    mock_fetch_from: AsyncMock, svc: OllamaExternalService, capabilities: list[str], expected: bool
) -> None:
    data = json.dumps({"model_info": {"general.architecture": "qwen35"}, "capabilities": capabilities})
    mock_fetch_from.return_value = FetchResult(status_code=200, data=data)

    *_, supports_decision = await svc._fetch_show_info("http://localhost:11434", "m")  # pyright: ignore[reportPrivateUsage]

    assert supports_decision is expected


@pytest.mark.asyncio
@patch("server.services.ollama_external_service.fetch_from", new_callable=AsyncMock)
async def test_fetch_show_info_leaves_decision_unknown_without_capabilities(mock_fetch_from: AsyncMock, svc: OllamaExternalService) -> None:
    mock_fetch_from.return_value = FetchResult(status_code=200, data=SHOW_RESPONSE)

    *_, supports_decision = await svc._fetch_show_info("http://localhost:11434", "llama3")  # pyright: ignore[reportPrivateUsage]

    assert supports_decision is None


@pytest.mark.asyncio
@patch.object(OllamaExternalService, "_fetch_show_info", new_callable=AsyncMock)
@patch("server.services.ollama_external_service.fetch_from", new_callable=AsyncMock)
async def test_sync_models_from_external_ollama_registers_decision_capability_model_under_systemone(
    mock_fetch_from: AsyncMock, mock_fetch_show_info: AsyncMock, svc: OllamaExternalService, installed: InstalledInfo, deps: dict[str, Any]
) -> None:
    mock_fetch_from.return_value = FetchResult(
        status_code=200, data=json.dumps({"models": [{"name": "brand-new-decider:latest", "size": 1024}]})
    )
    mock_fetch_show_info.return_value = (262144, False, None, True)

    await svc._sync_models_from_external_ollama("default", installed)  # pyright: ignore[reportPrivateUsage]

    registry = deps["endpoint_registry"]
    assert registry.register_systemone_as_proxy.call_count == 1
    kwargs = registry.register_systemone_as_proxy.call_args.kwargs
    assert kwargs["model"] == "brand-new-decider"
    assert kwargs["props"].type == "decision"
    assert kwargs["props"].endpoints == ["/v1/systemone"]
    assert kwargs["props"].context_window == 262144
    assert kwargs["options"].url == f"{installed.base_url}/v1/systemone"
    assert registry.register_chat_completion_as_proxy.call_count == 0
    assert installed.models["brand-new-decider"].type == "decision"


@pytest.mark.asyncio
@patch.object(OllamaExternalService, "_fetch_show_info", new_callable=AsyncMock)
@patch("server.services.ollama_external_service.fetch_from", new_callable=AsyncMock)
async def test_sync_models_from_external_ollama_registers_catalog_decision_model_when_capabilities_unknown(
    mock_fetch_from: AsyncMock, mock_fetch_show_info: AsyncMock, svc: OllamaExternalService, installed: InstalledInfo, deps: dict[str, Any]
) -> None:
    mock_fetch_from.return_value = FetchResult(status_code=200, data=json.dumps({"models": [{"name": "nimble:latest", "size": 1024}]}))
    mock_fetch_show_info.return_value = (None, None, None, None)

    await svc._sync_models_from_external_ollama("default", installed)  # pyright: ignore[reportPrivateUsage]

    assert deps["endpoint_registry"].register_systemone_as_proxy.call_count == 1
    assert deps["endpoint_registry"].register_systemone_as_proxy.call_args.kwargs["model"] == "nimble"


@pytest.mark.asyncio
@patch.object(OllamaExternalService, "_fetch_show_info", new_callable=AsyncMock)
@patch("server.services.ollama_external_service.fetch_from", new_callable=AsyncMock)
async def test_sync_models_from_external_ollama_skips_decision_model_on_ollama_older_than_0_35(
    mock_fetch_from: AsyncMock, mock_fetch_show_info: AsyncMock, svc: OllamaExternalService, installed: InstalledInfo, deps: dict[str, Any]
) -> None:
    svc.support_systemone = False
    svc.ollama_version = "0.34.1"
    mock_fetch_from.return_value = FetchResult(status_code=200, data=json.dumps({"models": [{"name": "nimble:latest", "size": 1024}]}))
    mock_fetch_show_info.return_value = (None, None, None, None)

    await svc._sync_models_from_external_ollama("default", installed)  # pyright: ignore[reportPrivateUsage]

    registry = deps["endpoint_registry"]
    assert registry.register_systemone_as_proxy.call_count == 0
    assert registry.register_chat_completion_as_proxy.call_count == 0
    assert "nimble" not in installed.models


@pytest.mark.asyncio
async def test_install_model_rejects_decision_model_on_ollama_older_than_0_35(
    svc: OllamaExternalService, installed: InstalledInfo, deps: dict[str, Any]
) -> None:
    svc.support_systemone = False
    svc.ollama_version = "0.34.1"

    with pytest.raises(HTTPException) as exc_info:
        await svc._install_model("default", "nimble", InstallModelIn())  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Decision models need Ollama 0.35.0 or newer; this Ollama is 0.34.1."
    assert ("default", "nimble") not in svc._installing  # pyright: ignore[reportPrivateUsage]
    assert "nimble" not in installed.models
    assert deps["endpoint_registry"].register_systemone_as_proxy.call_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(("version_str", "expected"), [("0.35.0", True), ("0.34.1", False)])
async def test_install_instance_sets_systemone_support_from_version(svc: OllamaExternalService, version_str: str, expected: bool) -> None:
    options = InstallServiceIn(spec={"url": "http://localhost:11434"})
    with patch("server.services.ollama_external_service.fetch_from", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = FetchResult(status_code=200, data=json.dumps({"version": version_str}))

        promise = await svc._install_instance("default", options)  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert svc.support_systemone is expected
    assert svc.ollama_version == version_str


def test_determine_model_type_custom_type_wins_over_decision_capability(svc: OllamaExternalService) -> None:
    svc._add_custom_model("default", CustomModel(id="c-1", data={"id": "my-model", "size": "1GB", "type": "llm"}))  # pyright: ignore[reportPrivateUsage]

    result = svc._determine_model_type("my-model", "default", supports_decision=True)  # pyright: ignore[reportPrivateUsage]

    assert result == "llm"


@pytest.mark.parametrize(
    ("model_id", "supports_decision", "expected"),
    [
        ("llama3.2", True, "decision"),
        ("unknown-model", True, "decision"),
        ("unknown-model", False, "llm"),
        ("nimble", None, "decision"),
    ],
)
def test_determine_model_type_resolves_decision_from_capability_or_catalog(
    svc: OllamaExternalService, model_id: str, supports_decision: bool | None, expected: str
) -> None:
    result = svc._determine_model_type(model_id, "default", supports_decision=supports_decision)  # pyright: ignore[reportPrivateUsage]

    assert result == expected


@pytest.mark.asyncio
@patch.object(OllamaExternalService, "_fetch_show_info", new_callable=AsyncMock)
@patch.object(OllamaExternalService, "_sync_models_from_external_ollama", new_callable=AsyncMock)
@patch.object(OllamaExternalService, "_download_model_or_set_progress", new_callable=AsyncMock)
async def test_install_model_registers_decision_endpoint(
    mock_download_model_or_set_progress: AsyncMock,
    mock_sync_models_from_external_ollama: AsyncMock,
    mock_fetch_show_info: AsyncMock,
    svc: OllamaExternalService,
    installed: InstalledInfo,
    deps: dict[str, Any],
) -> None:
    mock_fetch_show_info.return_value = (262144, False, None, True)

    promise = await svc._install_model("default", "nimble", InstallModelIn())  # pyright: ignore[reportPrivateUsage]
    await promise.wait()

    registry = deps["endpoint_registry"]
    assert registry.register_systemone_as_proxy.call_count == 1
    kwargs = registry.register_systemone_as_proxy.call_args.kwargs
    assert kwargs["model"] == "nimble"
    assert kwargs["props"].endpoints == ["/v1/systemone"]
    assert kwargs["options"].url == f"{installed.base_url}/v1/systemone"
    assert registry.register_chat_completion_as_proxy.call_count == 0
    assert mock_download_model_or_set_progress.call_count == 1
    assert mock_sync_models_from_external_ollama.call_count == 1


@pytest.mark.asyncio
@patch.object(OllamaExternalService, "_fetch_show_info", new_callable=AsyncMock)
@patch.object(OllamaExternalService, "_sync_models_from_external_ollama", new_callable=AsyncMock)
@patch.object(OllamaExternalService, "_download_model_or_set_progress", new_callable=AsyncMock)
async def test_install_model_post_registration_failure_unregisters_decision_and_rolls_back(
    mock_download_model_or_set_progress: AsyncMock,
    mock_sync_models_from_external_ollama: AsyncMock,
    mock_fetch_show_info: AsyncMock,
    svc: OllamaExternalService,
    installed: InstalledInfo,
    deps: dict[str, Any],
) -> None:
    deps["endpoint_registry"].register_systemone_as_proxy.return_value = "reg-dec"
    mock_sync_models_from_external_ollama.side_effect = RuntimeError("sync failed")
    mock_fetch_show_info.return_value = (None, None, None, None)

    promise = await svc._install_model("default", "nimble", InstallModelIn())  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(RuntimeError):
        await promise.wait()

    assert "nimble" not in installed.models
    assert deps["endpoint_registry"].unregister_systemone.call_count == 1
    assert deps["endpoint_registry"].unregister_systemone.call_args == call("nimble", "reg-dec")


@pytest.mark.asyncio
@patch.object(OllamaExternalService, "_save", new_callable=AsyncMock)
@patch.object(OllamaExternalService, "_sync_models_from_external_ollama", new_callable=AsyncMock)
async def test_uninstall_model_unregisters_decision(
    mock_sync_models_from_external_ollama: AsyncMock,
    mock_save: AsyncMock,
    svc: OllamaExternalService,
    installed: InstalledInfo,
    deps: dict[str, Any],
) -> None:
    installed.models["nimble"] = _make_decision_installed_info()

    await svc._uninstall_model("default", "nimble", UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert deps["endpoint_registry"].unregister_systemone.call_count == 1
    assert deps["endpoint_registry"].unregister_systemone.call_args == call("nimble", "reg-dec")
    assert mock_sync_models_from_external_ollama.call_count == 1
    assert mock_save.call_count == 1


@pytest.mark.asyncio
@patch.object(OllamaExternalService, "_uninstall_model", new_callable=AsyncMock)
async def test_uninstall_instance_unregisters_decision(
    mock_uninstall_model: AsyncMock, svc: OllamaExternalService, deps: dict[str, Any]
) -> None:
    installed = _make_installed_info()
    installed.models["nimble"] = _make_decision_installed_info()
    svc.instances_info["default"].installed = installed

    await svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert deps["endpoint_registry"].unregister_systemone.call_count == 1
    assert deps["endpoint_registry"].unregister_systemone.call_args == call("nimble", "reg-dec")
    assert mock_uninstall_model.call_count == 1


def test_remove_stale_models_unregisters_decision(svc: OllamaExternalService, installed: InstalledInfo, deps: dict[str, Any]) -> None:
    installed.models["nimble"] = _make_decision_installed_info()

    svc._remove_stale_models(installed, {"nimble"})  # pyright: ignore[reportPrivateUsage]

    assert deps["endpoint_registry"].unregister_systemone.call_count == 1
    assert deps["endpoint_registry"].unregister_systemone.call_args == call("nimble", "reg-dec")


def test_get_custom_model_spec_offers_decision_type(svc: OllamaExternalService) -> None:
    spec = svc.get_custom_model_spec()

    assert spec is not None
    type_field = next(field for field in spec.fields if field.name == "type")
    assert "decision" in (type_field.values or [])
