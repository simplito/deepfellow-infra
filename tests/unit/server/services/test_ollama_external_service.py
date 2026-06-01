# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import itertools
import json
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
)
from server.utils.core import FetchResult, Stream, StreamChunkProgress


def _make_installed_info(base_url: str = "http://localhost:11434") -> InstalledInfo:
    return InstalledInfo(
        models={},
        options=InstallServiceIn(spec={"url": base_url}),
        parsed_options=OllamaExternalOptions(url=base_url),
        base_url=base_url,
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
    }


@pytest.fixture
def svc(deps: dict[str, Any]) -> OllamaExternalService:
    return OllamaExternalService(**deps)


@pytest.fixture
def installed(svc: OllamaExternalService) -> InstalledInfo:
    info = _make_installed_info()
    svc.instances_info["default"].installed = info
    svc.support_responses = True
    svc.support_messages = True
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


def test_get_model_spec_has_alias_and_alive_time(svc: OllamaExternalService) -> None:
    spec = svc.get_model_spec()

    names = {f.name for f in spec.fields}
    assert {"alias", "alive_time"} <= names


def test_get_custom_model_spec_has_required_fields(svc: OllamaExternalService) -> None:
    spec = svc.get_custom_model_spec()

    assert spec is not None
    names = {f.name for f in spec.fields}
    assert {"id", "size", "type"} <= names


def test_model_installed_info_get_info() -> None:
    model = ModelInstalledInfo(
        id="my-llm",
        registered_name="my-llm",
        type="llm",
        options=InstallModelIn(),
        registration_id="reg-42",
    )

    info = model.get_info()

    assert info.registration_id == "reg-42"


def test_load_download_info_returns_downloaded_info(svc: OllamaExternalService) -> None:
    result = svc._load_download_info({})  # pyright: ignore[reportPrivateUsage]

    assert isinstance(result, DownloadedInfo)


def test_generate_instance_config_with_no_info(svc: OllamaExternalService) -> None:
    config = svc._generate_instance_config(None, None)  # pyright: ignore[reportPrivateUsage]

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

    config = svc._generate_instance_config(info, None)  # pyright: ignore[reportPrivateUsage]

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


@pytest.mark.parametrize(
    ("support_responses", "support_messages", "expected"),
    [
        (False, False, ["/v1/completions", "/v1/chat/completions"]),
        (False, True, ["/v1/completions", "/v1/chat/completions", "/v1/messages"]),
        (True, True, ["/v1/completions", "/v1/chat/completions", "/v1/responses", "/v1/messages"]),
    ],
)
def test_get_supported_endpoints(
    svc: OllamaExternalService,
    support_responses: bool,
    support_messages: bool,
    expected: list[str],
) -> None:
    svc.support_responses = support_responses
    svc.support_messages = support_messages

    result = svc.get_supported_endpoints()

    assert result == expected


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

    with patch.object(svc, "_uninstall_model", new_callable=AsyncMock):  # pyright: ignore[reportPrivateUsage]
        await svc._uninstall_instance("extra", UninstallServiceIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert "extra" not in svc.instances_info


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
    # Regression test for issue #458:
    # If _download_model raises, the entry must be removed from models_download_progress
    # so that a subsequent install attempt re-downloads instead of following the dead stream.
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
    # Regression test for issue #458:
    # After a failed first attempt, the second call must invoke _download_model again
    # (not silently replay the dead stream), ensuring the model is actually fetched.
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
async def test_install_model_uses_alias_as_registered_name(svc: OllamaExternalService, installed: InstalledInfo) -> None:
    svc.models["default"]["test-llm"] = OllamaModel(id="test-llm", size="1GB", type="llm")

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_sync_models_from_external_ollama", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
    ):
        promise = await svc._install_model("default", "test-llm", InstallModelIn(spec={"alias": "my-alias"}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert installed.models["test-llm"].registered_name == "my-alias"


@pytest.mark.asyncio
async def test_install_model_calls_generate_when_alive_time_set(svc: OllamaExternalService, installed: InstalledInfo) -> None:
    svc.models["default"]["test-llm"] = OllamaModel(id="test-llm", size="1GB", type="llm")
    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_sync_models_from_external_ollama", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
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

    fetch_results = [
        FetchResult(status_code=200, data=TAGS_RESPONSE),
    ]

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.ollama_external_service.fetch_from", new_callable=AsyncMock) as mock_fetch,
        patch.object(svc, "_save", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
    ):
        mock_fetch.side_effect = fetch_results
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
        patch.object(svc, "_save", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
    ):
        svc._start_sync_task("default", 1)  # pyright: ignore[reportPrivateUsage]
        await svc._sync_tasks["default"]  # pyright: ignore[reportPrivateUsage]


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
    assert "default" not in svc._sync_tasks  # pyright: ignore[reportPrivateUsage]
