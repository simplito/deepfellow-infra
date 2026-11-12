# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

import json
import logging
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from server.models.models import InstallModelIn, ListModelsFilters, UninstallModelIn
from server.models.services import InstallServiceIn, UninstallServiceIn
from server.services.base2_service import Instance, InstanceConfig, ModelConfig
from server.services.docker_model_runner_service import (
    DmrModel,
    DmrRegistry,
    DmrRegistryEntry,
    DockerModelRunnerService,
    DownloadedInfo,
    InstalledInfo,
    ModelInstalledInfo,
    _const_models,  # pyright: ignore[reportPrivateUsage]
    _read_models,  # pyright: ignore[reportPrivateUsage]
)
from server.utils.core import FetchResult


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
def svc(deps: dict[str, Any]) -> DockerModelRunnerService:
    return DockerModelRunnerService(**deps)


def _make_installed_info(base_url: str = "http://localhost:12434") -> InstalledInfo:
    return InstalledInfo(
        models={},
        options=InstallServiceIn(spec={}),
        parsed_options=MagicMock(url=base_url, backend="none"),
        base_url=base_url,
        backend="none",
    )


def _make_model_installed_info(model_id: str = "ai/llama3.2", model_type: str = "llm", reg_id: str = "reg-1") -> ModelInstalledInfo:
    return ModelInstalledInfo(
        id=model_id,
        registered_name=model_id,
        type=model_type,
        options=InstallModelIn(),
        registration_id=reg_id,
    )


# ─── Simple accessors ────────────────────────────────────────────────────────


def test_get_type(svc: DockerModelRunnerService) -> None:
    assert svc.get_type() == "docker-model-runner"


def test_get_description(svc: DockerModelRunnerService) -> None:
    assert "Docker Model Runner" in svc.get_description()


def test_get_size(svc: DockerModelRunnerService) -> None:
    assert svc.get_size() == ""


def test_get_spec_returns_url_and_backend_fields(svc: DockerModelRunnerService) -> None:
    spec = svc.get_spec()
    names = {f.name for f in spec.fields}
    assert {"url", "backend"} <= names


def test_get_spec_darwin_arm_selects_vllm_metal(svc: DockerModelRunnerService) -> None:
    with (
        patch("platform.system", return_value="Darwin"),
        patch("platform.machine", return_value="arm64"),
    ):
        spec = svc.get_spec()
    backend_field = next(f for f in spec.fields if f.name == "backend")
    assert backend_field.default == "vllm-metal"


def test_get_spec_linux_selects_vllm(svc: DockerModelRunnerService) -> None:
    with (
        patch("platform.system", return_value="Linux"),
        patch("platform.machine", return_value="x86_64"),
    ):
        spec = svc.get_spec()
    backend_field = next(f for f in spec.fields if f.name == "backend")
    assert backend_field.default == "vllm"


def test_get_spec_other_platform_selects_none(svc: DockerModelRunnerService) -> None:
    with (
        patch("platform.system", return_value="Windows"),
        patch("platform.machine", return_value="AMD64"),
    ):
        spec = svc.get_spec()
    backend_field = next(f for f in spec.fields if f.name == "backend")
    assert backend_field.default == "none"


def test_get_model_spec(svc: DockerModelRunnerService) -> None:
    spec = svc.get_model_spec()
    names = {f.name for f in spec.fields}
    assert "rename" in names


def test_get_custom_model_spec_returns_none(svc: DockerModelRunnerService) -> None:
    assert svc.get_custom_model_spec() is None


def test_service_has_docker_returns_false(svc: DockerModelRunnerService) -> None:
    assert svc.service_has_docker() is False


@pytest.mark.asyncio
async def test_stop_instance_is_noop(svc: DockerModelRunnerService) -> None:
    await svc.stop_instance("default")


def test_load_download_info_returns_downloaded_info(svc: DockerModelRunnerService) -> None:
    result = svc._load_download_info({})  # pyright: ignore[reportPrivateUsage]
    assert isinstance(result, DownloadedInfo)


# ─── ModelInstalledInfo ───────────────────────────────────────────────────────


def test_model_installed_info_get_info() -> None:
    info = _make_model_installed_info(reg_id="reg-abc")
    result = info.get_info()
    assert result.registration_id == "reg-abc"


# ─── DmrRegistry / Pydantic models ───────────────────────────────────────────


def test_dmr_registry_parses_llms_and_embeddings() -> None:
    registry = DmrRegistry(
        llms=[DmrRegistryEntry(name="ai/llama3.2", size="2.0 GB")],
        embeddings=[DmrRegistryEntry(name="ai/mxbai-embed-large", size="0.7 GB")],
    )
    assert len(registry.llms) == 1
    assert len(registry.embeddings) == 1


def test_const_models_loaded_from_json() -> None:
    assert len(_const_models) > 0
    model = next(iter(_const_models.values()))
    assert isinstance(model, DmrModel)
    assert model.type in ("llm", "embedding")


def test_read_models_includes_embeddings(tmp_path: Path) -> None:
    static_dir = tmp_path / "static"
    static_dir.mkdir()
    fake_json = {
        "llms": [{"name": "ai/llama3.2", "size": "2.0 GB"}],
        "embeddings": [{"name": "ai/mxbai-embed-large", "size": "0.7 GB"}],
    }
    (static_dir / "docker-model-runner-min.json").write_text(json.dumps(fake_json))

    with patch("server.services.docker_model_runner_service.get_main_dir", return_value=tmp_path):
        result = _read_models()

    assert "ai/mxbai-embed-large" in result
    assert result["ai/mxbai-embed-large"].type == "embedding"
    assert result["ai/llama3.2"].type == "llm"


# ─── get_installed_info ───────────────────────────────────────────────────────


def test_get_installed_info_not_installed(svc: DockerModelRunnerService) -> None:
    result = svc.get_installed_info("default")
    assert result is not None


def test_get_installed_info_installed(svc: DockerModelRunnerService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    result = svc.get_installed_info("default")
    assert result is not None


# ─── _generate_instance_config ────────────────────────────────────────────────


def test_generate_instance_config_none_info(svc: DockerModelRunnerService) -> None:
    config = svc._generate_instance_config("default", None, None)  # pyright: ignore[reportPrivateUsage]
    assert config.options is None
    assert config.models == []


def test_generate_instance_config_with_info(svc: DockerModelRunnerService) -> None:
    installed = _make_installed_info()
    installed.models["ai/llama3.2"] = _make_model_installed_info()
    config = svc._generate_instance_config("default", installed, None)  # pyright: ignore[reportPrivateUsage]
    assert config.options is not None
    assert config.models is not None
    assert len(config.models) == 1


# ─── _determine_model_type ────────────────────────────────────────────────────


def test_determine_model_type_in_instance_models(svc: DockerModelRunnerService) -> None:
    svc.models["default"]["ai/custom-emb"] = DmrModel(id="ai/custom-emb", size="1 GB", type="embedding")
    result = svc._determine_model_type("ai/custom-emb", "default")  # pyright: ignore[reportPrivateUsage]
    assert result == "embedding"


def test_determine_model_type_in_const_models(svc: DockerModelRunnerService) -> None:
    model_id = next(iter(_const_models))
    result = svc._determine_model_type(model_id, "nonexistent-instance")  # pyright: ignore[reportPrivateUsage]
    assert result == _const_models[model_id].type


def test_determine_model_type_fallback_to_llm(svc: DockerModelRunnerService) -> None:
    result = svc._determine_model_type("ai/totally-unknown-model", "nonexistent")  # pyright: ignore[reportPrivateUsage]
    assert result == "llm"


# ─── _register_model ──────────────────────────────────────────────────────────


def test_register_model_llm(svc: DockerModelRunnerService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    deps["endpoint_registry"].register_chat_completion_as_proxy.return_value = "reg-llm"

    svc._register_model("default", installed, "ai/llama3.2", 2_000_000_000)  # pyright: ignore[reportPrivateUsage]

    deps["endpoint_registry"].register_chat_completion_as_proxy.assert_called_once()
    assert "ai/llama3.2" in installed.models
    assert installed.models["ai/llama3.2"].type == "llm"


def test_register_model_embedding(svc: DockerModelRunnerService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    emb_id = "ai/mxbai-embed-large"
    svc.models["default"][emb_id] = DmrModel(id=emb_id, size="0.7 GB", type="embedding")
    deps["endpoint_registry"].register_embeddings_as_proxy.return_value = "reg-emb"

    svc._register_model("default", installed, emb_id, 0)  # pyright: ignore[reportPrivateUsage]

    deps["endpoint_registry"].register_embeddings_as_proxy.assert_called_once()
    assert installed.models[emb_id].type == "embedding"


def test_register_model_zero_size_gives_empty_size_str(svc: DockerModelRunnerService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    deps["endpoint_registry"].register_chat_completion_as_proxy.return_value = "reg-x"
    svc._register_model("default", installed, "ai/llama3.2", 0)  # pyright: ignore[reportPrivateUsage]
    assert svc.models["default"]["ai/llama3.2"].size == ""


# ─── _remove_stale_models ─────────────────────────────────────────────────────


def test_remove_stale_models_llm(svc: DockerModelRunnerService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    installed.models["ai/llama3.2"] = _make_model_installed_info("ai/llama3.2", "llm", "reg-1")
    svc.models_downloaded["ai/llama3.2"] = DownloadedInfo()

    svc._remove_stale_models(installed, {"ai/llama3.2"})  # pyright: ignore[reportPrivateUsage]

    deps["endpoint_registry"].unregister_chat_completion.assert_called_once()
    assert "ai/llama3.2" not in installed.models
    assert "ai/llama3.2" not in svc.models_downloaded


def test_remove_stale_models_embedding(svc: DockerModelRunnerService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    installed.models["ai/mxbai-embed-large"] = _make_model_installed_info("ai/mxbai-embed-large", "embedding", "reg-2")

    svc._remove_stale_models(installed, {"ai/mxbai-embed-large"})  # pyright: ignore[reportPrivateUsage]

    deps["endpoint_registry"].unregister_embeddings.assert_called_once()


def test_remove_stale_models_missing_model_is_noop(svc: DockerModelRunnerService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    svc._remove_stale_models(installed, {"ai/nonexistent"})  # pyright: ignore[reportPrivateUsage]
    deps["endpoint_registry"].unregister_chat_completion.assert_not_called()
    deps["endpoint_registry"].unregister_embeddings.assert_not_called()


# ─── _sync_models ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@patch("server.services.docker_model_runner_service.fetch_from", new_callable=AsyncMock)
async def test_sync_models_non_200_returns_early(mock_fetch: AsyncMock, svc: DockerModelRunnerService) -> None:
    installed = _make_installed_info()
    mock_fetch.return_value = FetchResult(status_code=503, data="{}")

    await svc._sync_models("default", installed)  # pyright: ignore[reportPrivateUsage]

    assert len(installed.models) == 0


@pytest.mark.asyncio
@patch("server.services.docker_model_runner_service.fetch_from", new_callable=AsyncMock)
async def test_sync_models_adds_new_models(mock_fetch: AsyncMock, svc: DockerModelRunnerService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    deps["endpoint_registry"].register_chat_completion_as_proxy.return_value = "reg-1"
    mock_fetch.return_value = FetchResult(
        status_code=200,
        data=json.dumps({"data": [{"id": "ai/llama3.2", "size": 2_000_000_000}]}),
    )

    await svc._sync_models("default", installed, is_initial=True)  # pyright: ignore[reportPrivateUsage]

    assert "ai/llama3.2" in installed.models


@pytest.mark.asyncio
@patch("server.services.docker_model_runner_service.fetch_from", new_callable=AsyncMock)
async def test_sync_models_skips_already_installed(mock_fetch: AsyncMock, svc: DockerModelRunnerService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    installed.models["ai/llama3.2"] = _make_model_installed_info()
    svc.instances_info["default"].installed = installed
    deps["endpoint_registry"].register_chat_completion_as_proxy.return_value = "reg-1"
    mock_fetch.return_value = FetchResult(
        status_code=200,
        data=json.dumps({"data": [{"id": "ai/llama3.2", "size": 0}]}),
    )

    await svc._sync_models("default", installed)  # pyright: ignore[reportPrivateUsage]

    deps["endpoint_registry"].register_chat_completion_as_proxy.assert_not_called()


@pytest.mark.asyncio
@patch("server.services.docker_model_runner_service.fetch_from", new_callable=AsyncMock)
async def test_sync_models_skips_downloaded_when_not_initial(mock_fetch: AsyncMock, svc: DockerModelRunnerService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    svc.models_downloaded["ai/llama3.2"] = DownloadedInfo()
    mock_fetch.return_value = FetchResult(
        status_code=200,
        data=json.dumps({"data": [{"id": "ai/llama3.2", "size": 0}]}),
    )

    await svc._sync_models("default", installed, is_initial=False)  # pyright: ignore[reportPrivateUsage]

    assert "ai/llama3.2" not in installed.models


@pytest.mark.asyncio
@patch("server.services.docker_model_runner_service.fetch_from", new_callable=AsyncMock)
async def test_sync_models_normalizes_docker_io_prefix_and_tag(
    mock_fetch: AsyncMock, svc: DockerModelRunnerService, deps: dict[str, Any]
) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"] = Instance(
        installed=None,
        installing=None,
        installing_model_progress={},
        config=InstanceConfig(models=[ModelConfig(model_id="ai/llama3.2", options=InstallModelIn())]),
    )
    deps["endpoint_registry"].register_chat_completion_as_proxy.return_value = "reg-1"
    mock_fetch.return_value = FetchResult(
        status_code=200,
        data=json.dumps({"data": [{"id": "docker.io/ai/llama3.2:latest", "size": 2_000_000_000}]}),
    )

    await svc._sync_models("default", installed, is_initial=True)  # pyright: ignore[reportPrivateUsage]

    assert "ai/llama3.2" in installed.models
    assert "docker.io/ai/llama3.2:latest" not in installed.models


@pytest.mark.asyncio
@patch("server.services.docker_model_runner_service.fetch_from", new_callable=AsyncMock)
async def test_sync_models_removes_stale(mock_fetch: AsyncMock, svc: DockerModelRunnerService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    installed.models["ai/old-model"] = _make_model_installed_info("ai/old-model", "llm", "old-reg")
    svc.instances_info["default"].installed = installed
    mock_fetch.return_value = FetchResult(status_code=200, data=json.dumps({"data": []}))

    await svc._sync_models("default", installed)  # pyright: ignore[reportPrivateUsage]

    assert "ai/old-model" not in installed.models


@pytest.mark.asyncio
@patch("server.services.docker_model_runner_service.fetch_from", new_callable=AsyncMock)
async def test_sync_models_restores_rename_on_initial_sync(
    mock_fetch: AsyncMock, svc: DockerModelRunnerService, deps: dict[str, Any]
) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"] = Instance(
        installed=None,
        installing=None,
        installing_model_progress={},
        config=InstanceConfig(models=[ModelConfig(model_id="ai/llama3.2", options=InstallModelIn(spec={"rename": "my-alias"}))]),
    )
    deps["endpoint_registry"].register_chat_completion_as_proxy.return_value = "reg-1"
    mock_fetch.return_value = FetchResult(
        status_code=200,
        data=json.dumps({"data": [{"id": "ai/llama3.2", "size": 2_000_000_000}]}),
    )

    await svc._sync_models("default", installed, is_initial=True)  # pyright: ignore[reportPrivateUsage]

    assert installed.models["ai/llama3.2"].registered_name == "my-alias"
    assert deps["endpoint_registry"].register_chat_completion_as_proxy.call_args.kwargs["model"] == "my-alias"


@pytest.mark.asyncio
@patch("server.services.docker_model_runner_service.fetch_from", new_callable=AsyncMock)
async def test_sync_models_initial_skips_non_purged_uninstall(
    mock_fetch: AsyncMock, svc: DockerModelRunnerService, deps: dict[str, Any]
) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    # Model was previously downloaded, then uninstalled without purge: it's still tracked in
    # models_downloaded and still reported by DMR's live model list, but no longer part of the
    # persisted instance config.
    svc.models_downloaded["ai/llama3.2"] = DownloadedInfo()
    mock_fetch.return_value = FetchResult(
        status_code=200,
        data=json.dumps({"data": [{"id": "ai/llama3.2", "size": 2_000_000_000}]}),
    )

    await svc._sync_models("default", installed, is_initial=True)  # pyright: ignore[reportPrivateUsage]

    assert "ai/llama3.2" not in installed.models
    deps["endpoint_registry"].register_chat_completion_as_proxy.assert_not_called()


@pytest.mark.asyncio
@patch("server.services.docker_model_runner_service.fetch_from", new_callable=AsyncMock)
async def test_sync_models_not_initial_registers_undownloaded_model(
    mock_fetch: AsyncMock, svc: DockerModelRunnerService, deps: dict[str, Any]
) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    deps["endpoint_registry"].register_chat_completion_as_proxy.return_value = "reg-1"
    mock_fetch.return_value = FetchResult(
        status_code=200,
        data=json.dumps({"data": [{"id": "ai/llama3.2", "size": 2_000_000_000}]}),
    )

    await svc._sync_models("default", installed, is_initial=False)  # pyright: ignore[reportPrivateUsage]

    assert "ai/llama3.2" in installed.models


# ─── _install_instance ────────────────────────────────────────────────────────


@pytest.mark.asyncio
@patch("server.services.docker_model_runner_service.fetch_from", new_callable=AsyncMock)
async def test_install_instance_raises_on_non_200(mock_fetch: AsyncMock, svc: DockerModelRunnerService) -> None:
    mock_fetch.return_value = FetchResult(status_code=503, data="{}")

    promise = await svc._install_instance("default", InstallServiceIn(spec={"url": "http://localhost:12434", "backend": "none"}))  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(HTTPException) as exc_info:
        await promise.wait()

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
@patch("server.services.docker_model_runner_service.fetch_from", new_callable=AsyncMock)
async def test_install_instance_raises_on_connection_exception(mock_fetch: AsyncMock, svc: DockerModelRunnerService) -> None:
    mock_fetch.side_effect = RuntimeError("connection refused")

    promise = await svc._install_instance("default", InstallServiceIn(spec={"url": "http://localhost:12434", "backend": "none"}))  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(HTTPException) as exc_info:
        await promise.wait()

    assert exc_info.value.status_code == 400
    assert "connection refused" in exc_info.value.detail


@pytest.mark.asyncio
@patch("server.services.docker_model_runner_service.fetch_from", new_callable=AsyncMock)
async def test_install_instance_loads_default_models_when_missing(mock_fetch: AsyncMock, svc: DockerModelRunnerService) -> None:
    # Clear models for instance so the branch on line 321 runs
    del svc.models["default"]
    mock_fetch.return_value = FetchResult(status_code=200, data=json.dumps({"data": []}))

    with patch.object(svc, "_save", new_callable=AsyncMock):
        promise = await svc._install_instance("default", InstallServiceIn(spec={"url": "http://localhost:12434", "backend": "none"}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert "default" in svc.models


@pytest.mark.asyncio
@patch("server.services.docker_model_runner_service.fetch_from", new_callable=AsyncMock)
async def test_install_instance_success_no_backend(mock_fetch: AsyncMock, svc: DockerModelRunnerService) -> None:
    mock_fetch.return_value = FetchResult(status_code=200, data=json.dumps({"data": []}))

    with patch.object(svc, "_save", new_callable=AsyncMock):
        promise = await svc._install_instance("default", InstallServiceIn(spec={"url": "http://localhost:12434", "backend": "none"}))  # pyright: ignore[reportPrivateUsage]
        result = await promise.wait()

    assert isinstance(result, InstalledInfo)
    assert result.backend == "none"


@pytest.mark.asyncio
@patch("server.services.docker_model_runner_service.fetch_from", new_callable=AsyncMock)
async def test_install_instance_success_with_backend(mock_fetch: AsyncMock, svc: DockerModelRunnerService) -> None:
    mock_fetch.return_value = FetchResult(status_code=200, data=json.dumps({"data": []}))
    mock_result = MagicMock()
    mock_result.exit_code = 0

    with (
        patch("server.services.docker_model_runner_service.Utils.run_command", new_callable=AsyncMock, return_value=mock_result),
        patch.object(svc, "_save", new_callable=AsyncMock),
    ):
        promise = await svc._install_instance("default", InstallServiceIn(spec={"url": "http://localhost:12434", "backend": "vllm"}))  # pyright: ignore[reportPrivateUsage]
        result = await promise.wait()

    assert result.backend == "vllm"


@pytest.mark.asyncio
@patch("server.services.docker_model_runner_service.fetch_from", new_callable=AsyncMock)
async def test_install_instance_backend_failure_raises_500(mock_fetch: AsyncMock, svc: DockerModelRunnerService) -> None:
    mock_fetch.return_value = FetchResult(status_code=200, data=json.dumps({"data": []}))
    mock_result = MagicMock()
    mock_result.exit_code = 1
    mock_result.stderr = "backend error"

    with patch("server.services.docker_model_runner_service.Utils.run_command", new_callable=AsyncMock, return_value=mock_result):
        promise = await svc._install_instance("default", InstallServiceIn(spec={"url": "http://localhost:12434", "backend": "vllm"}))  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(HTTPException) as exc_info:
            await promise.wait()

    assert exc_info.value.status_code == 500


# ─── _uninstall_instance ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_uninstall_instance_not_installed_is_noop(svc: DockerModelRunnerService, deps: dict[str, Any]) -> None:
    # installed is None — the `if installed:` branch is False
    assert svc.instances_info["default"].installed is None

    with patch.object(svc, "_save", new_callable=AsyncMock):
        await svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    deps["endpoint_registry"].unregister_chat_completion.assert_not_called()
    deps["endpoint_registry"].unregister_embeddings.assert_not_called()


@pytest.mark.asyncio
async def test_uninstall_instance_without_purge_keeps_instance(svc: DockerModelRunnerService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    installed.models["ai/llama3.2"] = _make_model_installed_info("ai/llama3.2", "llm")
    installed.models["ai/emb"] = _make_model_installed_info("ai/emb", "embedding")
    svc.instances_info["default"].installed = installed

    with patch.object(svc, "_save", new_callable=AsyncMock):
        await svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert svc.instances_info["default"].installed is None
    deps["endpoint_registry"].unregister_chat_completion.assert_called_once()
    deps["endpoint_registry"].unregister_embeddings.assert_called_once()


@pytest.mark.asyncio
async def test_uninstall_instance_with_purge_default_resets_instance(svc: DockerModelRunnerService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    with (
        patch.object(svc, "_save", new_callable=AsyncMock),
        patch.object(svc, "_clear_working_dir", new_callable=AsyncMock),
    ):
        await svc._uninstall_instance("default", UninstallServiceIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert "default" in svc.instances_info
    assert svc.instances_info["default"].installed is None


@pytest.mark.asyncio
async def test_uninstall_instance_with_purge_non_default_deletes_instance(svc: DockerModelRunnerService) -> None:
    installed = _make_installed_info()
    svc.instances_info["gpu-1"] = Instance(None, None, {}, InstanceConfig())
    svc.instances_info["gpu-1"].installed = installed
    svc.models["gpu-1"] = {}

    with patch.object(svc, "_save", new_callable=AsyncMock):
        await svc._uninstall_instance("gpu-1", UninstallServiceIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert "gpu-1" not in svc.instances_info


# ─── list_models ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_models_raises_404_for_unknown_instance(svc: DockerModelRunnerService) -> None:
    with pytest.raises(HTTPException) as exc_info:
        await svc.list_models("nonexistent", ListModelsFilters())

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_list_models_returns_all(svc: DockerModelRunnerService) -> None:
    svc.instances_info["default"].installed = _make_installed_info()
    result = await svc.list_models(None, ListModelsFilters())
    assert len(result.list) > 0


@pytest.mark.asyncio
async def test_list_models_filter_installed_only(svc: DockerModelRunnerService) -> None:
    installed = _make_installed_info()
    installed.models["ai/llama3.2"] = _make_model_installed_info()
    svc.instances_info["default"].installed = installed

    result = await svc.list_models(None, ListModelsFilters(installed=True))

    assert all(m.installed is not None for m in result.list)


@pytest.mark.asyncio
async def test_list_models_filter_not_installed(svc: DockerModelRunnerService) -> None:
    svc.instances_info["default"].installed = _make_installed_info()
    result = await svc.list_models(None, ListModelsFilters(installed=False))
    assert all(not m.installed for m in result.list)


@pytest.mark.asyncio
async def test_list_models_single_instance_string(svc: DockerModelRunnerService) -> None:
    svc.instances_info["default"].installed = _make_installed_info()
    result = await svc.list_models("default", ListModelsFilters())
    assert len(result.list) > 0


@pytest.mark.asyncio
async def test_list_models_skips_instances_not_in_requested_list(svc: DockerModelRunnerService) -> None:
    # Add a second instance to self.models that is not requested
    svc.models["gpu-1"] = {"ai/llama3.2": DmrModel(id="ai/llama3.2", size="2 GB", type="llm")}
    svc.instances_info["default"].installed = _make_installed_info()

    result = await svc.list_models("default", ListModelsFilters())

    # gpu-1 models should not appear in the result (continue branch hit)
    assert not any("gpu-1" in str(m.service) for m in result.list)


# ─── get_model ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_model_not_found_raises_400(svc: DockerModelRunnerService) -> None:
    svc.instances_info["default"].installed = _make_installed_info()
    with pytest.raises(HTTPException) as exc_info:
        await svc.get_model("default", "nonexistent/model")

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_get_model_not_installed(svc: DockerModelRunnerService) -> None:
    model_id = next(iter(svc.models["default"]))
    svc.instances_info["default"].installed = _make_installed_info()
    result = await svc.get_model("default", model_id)
    assert result.id == model_id
    assert not result.installed


@pytest.mark.asyncio
async def test_get_model_installed(svc: DockerModelRunnerService) -> None:
    model_id = next(iter(svc.models["default"]))
    installed = _make_installed_info()
    installed.models[model_id] = _make_model_installed_info(model_id)
    svc.instances_info["default"].installed = installed

    result = await svc.get_model("default", model_id)

    assert result.id == model_id
    assert result.installed is not None


# ─── _pull_model ──────────────────────────────────────────────────────────────


def _dmr_layer_line(layer_id: str, current: int, size: int, total: int) -> str:
    return json.dumps(
        {
            "type": "progress",
            "message": f"Downloaded: {current} B",
            "total": total,
            "layer": {"id": layer_id, "size": size, "current": current},
            "mode": "pull",
        }
    )


def _dmr_success_line() -> str:
    return json.dumps({"type": "success", "message": "Model pulled successfully", "mode": "pull"})


@pytest.mark.asyncio
async def test_pull_model_raises_on_bad_status_code(svc: DockerModelRunnerService) -> None:
    stream = MagicMock()
    model = DmrModel(id="ai/llama3.2", size="2.0 GB", type="llm")

    async def mock_stream(*args: object, **kwargs: object):  # type: ignore[misc]
        yield FetchResult(status_code=500, data="{}")

    with (
        patch("server.services.docker_model_runner_service.stream_fetch_from", side_effect=mock_stream),
        pytest.raises(HTTPException) as exc_info,
    ):
        await svc._pull_model(stream, model, "ai/llama3.2", "http://localhost:12434")  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 500


@pytest.mark.asyncio
async def test_pull_model_raises_on_error_in_json(svc: DockerModelRunnerService) -> None:
    stream = MagicMock()
    model = DmrModel(id="ai/llama3.2", size="2.0 GB", type="llm")

    async def mock_stream(*args: object, **kwargs: object):  # type: ignore[misc]
        yield FetchResult(status_code=200, data=json.dumps({"error": "pull failed"}))

    with (
        patch("server.services.docker_model_runner_service.stream_fetch_from", side_effect=mock_stream),
        pytest.raises(HTTPException) as exc_info,
    ):
        await svc._pull_model(stream, model, "ai/llama3.2", "http://localhost:12434")  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 500


@pytest.mark.asyncio
async def test_pull_model_emits_progress_and_completes(svc: DockerModelRunnerService) -> None:
    stream = MagicMock()
    model = DmrModel(id="ai/llama3.2", size="2.0 GB", type="llm")

    async def mock_stream(*args: object, **kwargs: object):  # type: ignore[misc]
        yield FetchResult(status_code=200, data=_dmr_layer_line("sha256:aaa", 500, 1000, 1000) + "\n")
        yield FetchResult(status_code=200, data=_dmr_success_line())

    with patch("server.services.docker_model_runner_service.stream_fetch_from", side_effect=mock_stream):
        await svc._pull_model(stream, model, "ai/llama3.2", "http://localhost:12434")  # pyright: ignore[reportPrivateUsage]

    assert stream.emit.call_count >= 2


@pytest.mark.asyncio
async def test_pull_model_sends_from_field(svc: DockerModelRunnerService) -> None:
    stream = MagicMock()
    model = DmrModel(id="ai/llama3.2", size="2.0 GB", type="llm")

    async def mock_stream(*args: object, **kwargs: object):  # type: ignore[misc]
        yield FetchResult(status_code=200, data=json.dumps({"status": "success"}))

    with patch("server.services.docker_model_runner_service.stream_fetch_from", side_effect=mock_stream) as mock_fetch:
        await svc._pull_model(stream, model, "ai/llama3.2", "http://localhost:12434")  # pyright: ignore[reportPrivateUsage]

    assert mock_fetch.call_args.args[2] == {"from": "ai/llama3.2"}


@pytest.mark.asyncio
async def test_pull_model_skips_invalid_json_lines(svc: DockerModelRunnerService) -> None:
    stream = MagicMock()
    model = DmrModel(id="ai/llama3.2", size="2.0 GB", type="llm")

    async def mock_stream(*args: object, **kwargs: object):  # type: ignore[misc]
        yield FetchResult(status_code=200, data="not-json\n" + json.dumps({"status": "done"}))

    with patch("server.services.docker_model_runner_service.stream_fetch_from", side_effect=mock_stream):
        await svc._pull_model(stream, model, "ai/llama3.2", "http://localhost:12434")  # pyright: ignore[reportPrivateUsage]

    stream.emit.assert_called()


@pytest.mark.asyncio
async def test_pull_model_skips_empty_lines(svc: DockerModelRunnerService) -> None:
    stream = MagicMock()
    model = DmrModel(id="ai/llama3.2", size="2.0 GB", type="llm")

    async def mock_stream(*args: object, **kwargs: object):  # type: ignore[misc]
        yield FetchResult(status_code=200, data="\n\n" + json.dumps({"status": "complete"}))

    with patch("server.services.docker_model_runner_service.stream_fetch_from", side_effect=mock_stream):
        await svc._pull_model(stream, model, "ai/llama3.2", "http://localhost:12434")  # pyright: ignore[reportPrivateUsage]

    stream.emit.assert_called()


@pytest.mark.asyncio
async def test_pull_model_unknown_json_record_no_progress_emitted(svc: DockerModelRunnerService) -> None:
    # Record has neither completed/total nor known status — elif branch is False
    stream = MagicMock()
    model = DmrModel(id="ai/llama3.2", size="2.0 GB", type="llm")

    async def mock_stream(*args: object, **kwargs: object):  # type: ignore[misc]
        yield FetchResult(status_code=200, data=json.dumps({"some_other_field": "value"}))

    with patch("server.services.docker_model_runner_service.stream_fetch_from", side_effect=mock_stream):
        await svc._pull_model(stream, model, "ai/llama3.2", "http://localhost:12434")  # pyright: ignore[reportPrivateUsage]

    # Only start/end progress emitted (value 0 and 1), no intermediate 0.0-0.99
    values = [c.args[0]["value"] for c in stream.emit.call_args_list]
    assert set(values) == {0, 1}


@pytest.mark.asyncio
async def test_pull_model_reassembles_json_split_across_chunks(svc: DockerModelRunnerService) -> None:
    stream = MagicMock()
    model = DmrModel(id="ai/llama3.2", size="2.0 GB", type="llm")
    full_line = json.dumps({"completed": 500, "total": 1000}) + "\n"
    split_at = len(full_line) // 2

    async def mock_stream(*args: object, **kwargs: object):  # type: ignore[misc]
        yield FetchResult(status_code=200, data=full_line[:split_at])
        yield FetchResult(status_code=200, data=full_line[split_at:])

    with patch("server.services.docker_model_runner_service.stream_fetch_from", side_effect=mock_stream):
        await svc._pull_model(stream, model, "ai/llama3.2", "http://localhost:12434")  # pyright: ignore[reportPrivateUsage]

    values = [c.args[0]["value"] for c in stream.emit.call_args_list]
    assert 0.5 in values


@pytest.mark.asyncio
async def test_pull_model_handles_final_line_without_trailing_newline(svc: DockerModelRunnerService) -> None:
    stream = MagicMock()
    model = DmrModel(id="ai/llama3.2", size="2.0 GB", type="llm")

    async def mock_stream(*args: object, **kwargs: object):  # type: ignore[misc]
        yield FetchResult(status_code=200, data=json.dumps({"status": "success"}))

    with patch("server.services.docker_model_runner_service.stream_fetch_from", side_effect=mock_stream):
        await svc._pull_model(stream, model, "ai/llama3.2", "http://localhost:12434")  # pyright: ignore[reportPrivateUsage]

    values = [c.args[0]["value"] for c in stream.emit.call_args_list]
    assert 0.99 in values


@pytest.mark.asyncio
async def test_pull_model_aggregates_progress_across_layers(svc: DockerModelRunnerService) -> None:
    stream = MagicMock()
    model = DmrModel(id="ai/llama3.2", size="1000B", type="llm")

    async def mock_stream(*args: object, **kwargs: object):  # type: ignore[misc]
        yield FetchResult(status_code=200, data=_dmr_layer_line("sha256:aaa", 300, 300, 1000) + "\n")
        yield FetchResult(status_code=200, data=_dmr_layer_line("sha256:bbb", 400, 400, 1000) + "\n")
        yield FetchResult(status_code=200, data=_dmr_layer_line("sha256:aaa", 600, 300, 1000))

    with patch("server.services.docker_model_runner_service.stream_fetch_from", side_effect=mock_stream):
        await svc._pull_model(stream, model, "ai/llama3.2", "http://localhost:12434")  # pyright: ignore[reportPrivateUsage]

    values = [c.args[0]["value"] for c in stream.emit.call_args_list]
    # After layer B's first line: layer A at 300 + layer B at 400 = 700/1000, not just layer B's own 400/1000 = 0.4.
    assert 0.7 in values


@pytest.mark.asyncio
async def test_pull_model_stale_layer_progress_not_double_counted(svc: DockerModelRunnerService) -> None:
    stream = MagicMock()
    model = DmrModel(id="ai/llama3.2", size="1000B", type="llm")

    async def mock_stream(*args: object, **kwargs: object):  # type: ignore[misc]
        yield FetchResult(status_code=200, data=_dmr_layer_line("sha256:aaa", 600, 600, 1000) + "\n")
        # Same layer reports an unchanged (stale/duplicate) current value — must not add again.
        yield FetchResult(status_code=200, data=_dmr_layer_line("sha256:aaa", 600, 600, 1000))

    with patch("server.services.docker_model_runner_service.stream_fetch_from", side_effect=mock_stream):
        await svc._pull_model(stream, model, "ai/llama3.2", "http://localhost:12434")  # pyright: ignore[reportPrivateUsage]

    values = [c.args[0]["value"] for c in stream.emit.call_args_list]
    # If the stale record were double-counted, percentage would exceed 0.6 (e.g. 0.99/1.0 clipped).
    assert 0.6 in values
    assert 1.2 not in values


@pytest.mark.asyncio
async def test_pull_model_parses_html_escaped_layer_line(svc: DockerModelRunnerService) -> None:
    # Some DMR builds emit the progress stream HTML-escaped — this is a verbatim line from such a build.
    stream = MagicMock()
    model = DmrModel(id="ai/smollm2", size="1000B", type="llm")
    escaped_line = (
        "{&#34;type&#34;:&#34;progress&#34;,&#34;message&#34;:&#34;Downloaded: 0.01 MB&#34;,&#34;total&#34;:1000,"
        "&#34;layer&#34;:{&#34;id&#34;:&#34;sha256:bf6f20a6&#34;,&#34;size&#34;:1000,&#34;current&#34;:600},&#34;mode&#34;:&#34;pull&#34;}"
    )

    async def mock_stream(*args: object, **kwargs: object):  # type: ignore[misc]
        yield FetchResult(status_code=200, data=escaped_line)

    with patch("server.services.docker_model_runner_service.stream_fetch_from", side_effect=mock_stream):
        await svc._pull_model(stream, model, "ai/smollm2", "http://localhost:12434")  # pyright: ignore[reportPrivateUsage]

    values = [c.args[0]["value"] for c in stream.emit.call_args_list]
    assert 0.6 in values


@pytest.mark.asyncio
async def test_pull_model_raises_on_html_escaped_error_line(svc: DockerModelRunnerService) -> None:
    stream = MagicMock()
    model = DmrModel(id="ai/smollm2", size="1000B", type="llm")

    async def mock_stream(*args: object, **kwargs: object):  # type: ignore[misc]
        yield FetchResult(status_code=200, data="{&#34;error&#34;:&#34;pull failed&#34;}")

    with (
        patch("server.services.docker_model_runner_service.stream_fetch_from", side_effect=mock_stream),
        pytest.raises(HTTPException) as exc_info,
    ):
        await svc._pull_model(stream, model, "ai/smollm2", "http://localhost:12434")  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 500


@pytest.mark.asyncio
async def test_pull_model_skips_line_still_unparseable_after_unescaping(svc: DockerModelRunnerService) -> None:
    # Unescaping changes the line but the result is still not valid JSON — the line is skipped, not raised on.
    stream = MagicMock()
    model = DmrModel(id="ai/llama3.2", size="1000B", type="llm")

    async def mock_stream(*args: object, **kwargs: object):  # type: ignore[misc]
        yield FetchResult(status_code=200, data="&#34;truncated")

    with patch("server.services.docker_model_runner_service.stream_fetch_from", side_effect=mock_stream):
        await svc._pull_model(stream, model, "ai/llama3.2", "http://localhost:12434")  # pyright: ignore[reportPrivateUsage]

    values = [c.args[0]["value"] for c in stream.emit.call_args_list]
    assert set(values) == {0, 1}


@pytest.mark.asyncio
async def test_pull_model_warns_once_for_many_unparseable_lines(svc: DockerModelRunnerService, caplog: pytest.LogCaptureFixture) -> None:
    stream = MagicMock()
    model = DmrModel(id="ai/llama3.2", size="1000B", type="llm")

    async def mock_stream(*args: object, **kwargs: object):  # type: ignore[misc]
        yield FetchResult(status_code=200, data="not-json\n" * 20)

    with caplog.at_level(logging.WARNING), patch("server.services.docker_model_runner_service.stream_fetch_from", side_effect=mock_stream):
        await svc._pull_model(stream, model, "ai/llama3.2", "http://localhost:12434")  # pyright: ignore[reportPrivateUsage]

    per_line_warnings = [r for r in caplog.records if "unparseable line:" in r.message]
    summary_warnings = [r for r in caplog.records if "unparseable line(s) while pulling" in r.message]
    assert len(per_line_warnings) == 1
    assert len(summary_warnings) == 1
    assert "20 unparseable line(s)" in summary_warnings[0].message


# ─── _install_model ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_install_model_already_installed_returns_ok(svc: DockerModelRunnerService) -> None:
    model_id = next(iter(svc.models["default"]))
    installed = _make_installed_info()
    installed.models[model_id] = _make_model_installed_info(model_id)
    svc.instances_info["default"].installed = installed

    promise = await svc._install_model("default", model_id, InstallModelIn())  # pyright: ignore[reportPrivateUsage]
    result = await promise.wait()

    assert result.status == "OK"
    assert "Already installed" in result.details


@pytest.mark.asyncio
async def test_install_model_not_found_raises_400(svc: DockerModelRunnerService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    with pytest.raises(HTTPException) as exc_info:
        await svc._install_model("default", "nonexistent/model", InstallModelIn())  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_install_model_mlx_blocked_on_non_darwin_host(svc: DockerModelRunnerService) -> None:
    mlx_id = next(mid for mid in svc.models["default"] if mid.startswith("mlx-community/"))
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    with (
        patch("platform.system", return_value="Linux"),
        patch("platform.machine", return_value="x86_64"),
        pytest.raises(HTTPException) as exc_info,
    ):
        await svc._install_model("default", mlx_id, InstallModelIn())  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_install_model_mlx_blocked_on_darwin_intel_host(svc: DockerModelRunnerService) -> None:
    mlx_id = next(mid for mid in svc.models["default"] if mid.startswith("mlx-community/"))
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    with (
        patch("platform.system", return_value="Darwin"),
        patch("platform.machine", return_value="x86_64"),
        pytest.raises(HTTPException) as exc_info,
    ):
        await svc._install_model("default", mlx_id, InstallModelIn())  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_install_model_mlx_allowed_on_darwin_arm(svc: DockerModelRunnerService, deps: dict[str, Any]) -> None:
    mlx_id = next(mid for mid in svc.models["default"] if mid.startswith("mlx-community/"))
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    deps["endpoint_registry"].register_chat_completion_as_proxy.return_value = "reg-mlx"

    with (
        patch("platform.system", return_value="Darwin"),
        patch("platform.machine", return_value="arm64"),
        patch.object(svc, "_pull_model", new_callable=AsyncMock),
        patch.object(svc, "_save", new_callable=AsyncMock),
    ):
        promise = await svc._install_model("default", mlx_id, InstallModelIn())  # pyright: ignore[reportPrivateUsage]
        result = await promise.wait()

    assert result.status == "OK"


@pytest.mark.asyncio
async def test_install_model_llm_registers_chat_completion(svc: DockerModelRunnerService, deps: dict[str, Any]) -> None:
    model_id = next(mid for mid, m in svc.models["default"].items() if m.type == "llm")
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    deps["endpoint_registry"].register_chat_completion_as_proxy.return_value = "reg-1"

    with (
        patch.object(svc, "_pull_model", new_callable=AsyncMock),
        patch.object(svc, "_save", new_callable=AsyncMock),
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn())  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    deps["endpoint_registry"].register_chat_completion_as_proxy.assert_called_once()
    assert model_id in installed.models


@pytest.mark.asyncio
async def test_install_model_embedding_registers_embeddings(svc: DockerModelRunnerService, deps: dict[str, Any]) -> None:
    emb_id = "ai/mxbai-embed-large"
    svc.models["default"][emb_id] = DmrModel(id=emb_id, size="0.7 GB", type="embedding")
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    deps["endpoint_registry"].register_embeddings_as_proxy.return_value = "reg-emb"

    with (
        patch.object(svc, "_pull_model", new_callable=AsyncMock),
        patch.object(svc, "_save", new_callable=AsyncMock),
    ):
        promise = await svc._install_model("default", emb_id, InstallModelIn())  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    deps["endpoint_registry"].register_embeddings_as_proxy.assert_called_once()


@pytest.mark.asyncio
async def test_install_model_rename_is_used_as_registered_name(svc: DockerModelRunnerService, deps: dict[str, Any]) -> None:
    model_id = next(mid for mid, m in svc.models["default"].items() if m.type == "llm")
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    deps["endpoint_registry"].register_chat_completion_as_proxy.return_value = "reg-alias"

    with (
        patch.object(svc, "_pull_model", new_callable=AsyncMock),
        patch.object(svc, "_save", new_callable=AsyncMock),
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn(spec={"rename": "my-alias"}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    call_kwargs = deps["endpoint_registry"].register_chat_completion_as_proxy.call_args
    assert call_kwargs.kwargs["model"] == "my-alias"


# ─── _uninstall_model ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_uninstall_model_llm_unregisters(svc: DockerModelRunnerService, deps: dict[str, Any]) -> None:
    model_id = "ai/llama3.2"
    installed = _make_installed_info()
    installed.models[model_id] = _make_model_installed_info(model_id, "llm", "reg-1")
    svc.instances_info["default"].installed = installed

    with patch.object(svc, "_save", new_callable=AsyncMock):
        await svc._uninstall_model("default", model_id, UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    deps["endpoint_registry"].unregister_chat_completion.assert_called_once()
    assert model_id not in installed.models


@pytest.mark.asyncio
async def test_uninstall_model_embedding_unregisters(svc: DockerModelRunnerService, deps: dict[str, Any]) -> None:
    model_id = "ai/mxbai-embed-large"
    installed = _make_installed_info()
    installed.models[model_id] = _make_model_installed_info(model_id, "embedding", "reg-2")
    svc.instances_info["default"].installed = installed

    with patch.object(svc, "_save", new_callable=AsyncMock):
        await svc._uninstall_model("default", model_id, UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    deps["endpoint_registry"].unregister_embeddings.assert_called_once()


@pytest.mark.asyncio
@patch("server.services.docker_model_runner_service.fetch_from", new_callable=AsyncMock)
async def test_uninstall_model_with_purge_calls_delete(mock_fetch: AsyncMock, svc: DockerModelRunnerService) -> None:
    model_id = "ai/llama3.2"
    installed = _make_installed_info()
    installed.models[model_id] = _make_model_installed_info(model_id, "llm", "reg-1")
    svc.instances_info["default"].installed = installed
    svc.models_downloaded[model_id] = DownloadedInfo()
    mock_fetch.return_value = FetchResult(status_code=200, data="{}")

    with patch.object(svc, "_save", new_callable=AsyncMock):
        await svc._uninstall_model("default", model_id, UninstallModelIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    mock_fetch.assert_called_once()
    assert model_id not in svc.models_downloaded


@pytest.mark.asyncio
async def test_uninstall_model_not_in_installed_is_noop(svc: DockerModelRunnerService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    with patch.object(svc, "_save", new_callable=AsyncMock):
        await svc._uninstall_model("default", "ai/nonexistent", UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    deps["endpoint_registry"].unregister_chat_completion.assert_not_called()


# ─── load_instance ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_load_instance_no_options_returns_early(svc: DockerModelRunnerService) -> None:
    await svc.load_instance("default", InstanceConfig())

    assert svc.instances_info["default"].installed is None


@pytest.mark.asyncio
async def test_load_instance_installs_and_saves(svc: DockerModelRunnerService) -> None:
    installed = _make_installed_info()
    options = InstallServiceIn(spec={"url": "http://localhost:12434", "backend": "none"})
    config = InstanceConfig(options=options)

    with (
        patch.object(svc, "install_instance", new_callable=AsyncMock) as mock_install,
        patch.object(svc, "_save", new_callable=AsyncMock),
    ):
        promise_mock = AsyncMock()
        promise_mock.wait = AsyncMock(return_value=installed)
        mock_install.return_value = promise_mock
        svc.instances_info["default"].installed = installed

        await svc.load_instance("default", config)

    mock_install.assert_called_once()


@pytest.mark.asyncio
async def test_load_instance_installed_is_none_after_promise(svc: DockerModelRunnerService) -> None:
    # After install_instance, installed stays None — the `if installed:` branch is False
    options = InstallServiceIn(spec={"url": "http://localhost:12434", "backend": "none"})
    config = InstanceConfig(options=options)

    with (
        patch.object(svc, "install_instance", new_callable=AsyncMock) as mock_install,
        patch.object(svc, "_save", new_callable=AsyncMock),
    ):
        promise_mock = AsyncMock()
        promise_mock.wait = AsyncMock(return_value=None)
        mock_install.return_value = promise_mock
        # instances_info["default"].installed stays None

        await svc.load_instance("default", config)

    mock_install.assert_called_once()
    assert svc.instances_info["default"].installed is None


@pytest.mark.asyncio
async def test_pull_model_emits_record_byte_counts_in_progress_data(svc: DockerModelRunnerService) -> None:
    stream = MagicMock()
    model = DmrModel(id="ai/llama3.2", size="2.0 GB", type="llm")

    async def mock_stream(*args: object, **kwargs: object):  # type: ignore[misc]
        yield FetchResult(status_code=200, data=json.dumps({"completed": 500, "total": 1000}) + "\n")

    with patch("server.services.docker_model_runner_service.stream_fetch_from", side_effect=mock_stream):
        await svc._pull_model(stream, model, "ai/llama3.2", "http://localhost:12434")  # pyright: ignore[reportPrivateUsage]

    chunk = next(c.args[0] for c in stream.emit.call_args_list if c.args[0]["value"] == 0.5)
    assert chunk["data"] == {"completed": 500, "total": 1000}


@pytest.mark.asyncio
async def test_pull_model_sends_record_byte_counts_as_integers(svc: DockerModelRunnerService) -> None:
    stream = MagicMock()
    model = DmrModel(id="ai/llama3.2", size="2.0 GB", type="llm")

    async def mock_stream(*args: object, **kwargs: object):  # type: ignore[misc]
        yield FetchResult(status_code=200, data=json.dumps({"completed": 500.0, "total": 1000.0}) + "\n")

    with patch("server.services.docker_model_runner_service.stream_fetch_from", side_effect=mock_stream):
        await svc._pull_model(stream, model, "ai/llama3.2", "http://localhost:12434")  # pyright: ignore[reportPrivateUsage]

    chunk = next(c.args[0] for c in stream.emit.call_args_list if c.args[0]["value"] == 0.5)
    assert {key: type(value) for key, value in chunk["data"].items()} == {"completed": int, "total": int}
