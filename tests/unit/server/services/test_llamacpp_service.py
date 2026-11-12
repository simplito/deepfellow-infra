# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

import asyncio
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from scripts.get_llamacpp_models import DegradedFetchError
from server.docker import ContainerStatus
from server.models.models import InstallModelIn, ListModelsFilters, UninstallModelIn
from server.models.registries import LlamacppRegistryEntry
from server.models.services import InstallServiceIn, UninstallServiceIn
from server.services.base2_service import CustomModel, Instance, InstanceConfig, ModelConfig
from server.services.llamacpp_service import (
    DownloadedInfo,
    InstalledInfo,
    LlamacppModel,
    LLamacppOptions,
    LLamacppService,
    ModelInstalledInfo,
    _const,  # pyright: ignore[reportPrivateUsage]
    _read_models,  # pyright: ignore[reportPrivateUsage]
)
from server.services.model_catalog_refresh import huggingface_refresh_guard
from server.utils.core import DownloadedPacket, PreDownloadPacket, Stream, StreamChunk, StreamChunkProgress, SuccessDownloadPacket
from server.utils.docker_image_version import newest_docker_tag
from server.utils.hardware import CpuInfo, GpuInfo, IntelGpuInfo, NvidiaGpuInfo
from tests.unit.server.real_docker_tags import real_tags


@pytest.fixture
def deps() -> dict[str, Any]:
    hw = MagicMock()
    hw.gpus = []
    hw.nvidia_gpus = []
    hw.intel_gpus = []
    return {
        "config": MagicMock(),
        "endpoint_registry": MagicMock(),
        "service_provider": MagicMock(),
        "model_downloader": MagicMock(),
        "docker_service": MagicMock(),
        "hardware": hw,
        "services_manager": MagicMock(),
    }


@pytest.fixture
def svc(deps: dict[str, Any]) -> LLamacppService:
    return LLamacppService(**deps)


def _make_installed_info(svc: LLamacppService, instance: str = "default") -> InstalledInfo:
    return InstalledInfo(
        models={},
        options=InstallServiceIn(spec={}),
        parsed_options=LLamacppOptions(),
    )


def _make_model_installed_info(model_id: str = "test-model", registration_id: str = "reg-1") -> ModelInstalledInfo:
    docker = MagicMock()
    docker.name = f"df-llamacpp-{model_id}"
    docker.container_name = f"df-llamacpp-{model_id}"
    return ModelInstalledInfo(
        id=model_id,
        registered_name=model_id,
        options=InstallModelIn(spec={}),
        docker=docker,
        model_path=Path("/tmp/model.gguf"),
        container_host="localhost",
        container_port=8080,
        docker_exposed_port=8080,
        registration_id=registration_id,
        base_url="http://localhost:8080",
    )


def _setup_install_mocks(svc: LLamacppService, deps: dict[str, Any]) -> InstalledInfo:
    installed = _make_installed_info(svc)
    svc.instances_info["default"].installed = installed
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=(8080, True, False))
    deps["docker_service"].get_docker_subnet.return_value = None
    deps["docker_service"].get_docker_container_name.return_value = "container"
    deps["docker_service"].get_container_host.return_value = "localhost"
    deps["docker_service"].get_container_port.return_value = 8080
    deps["docker_service"].stop_docker = AsyncMock()
    return installed


# --- Identity (existing) ---


def test_get_type(svc: LLamacppService) -> None:
    assert svc.get_type() == "llamacpp"


def test_get_description_not_empty(svc: LLamacppService) -> None:
    assert svc.get_description()


def test_service_has_docker(svc: LLamacppService) -> None:
    assert svc.service_has_docker() is False


def test_default_models_loaded(svc: LLamacppService) -> None:
    assert "default" in svc.models
    assert len(svc.models["default"]) > 0


def test_read_models_loads_registry_file(tmp_path: Path) -> None:
    static_dir = tmp_path / "static"
    static_dir.mkdir()
    (static_dir / "llamacpp-min.json").write_text(
        json.dumps(
            {
                "llms": [
                    {"name": "org/model-q4-k-m", "url": "https://huggingface.co/org/model/resolve/main/model-Q4_K_M.gguf", "size": "4GB"},
                    {
                        "name": "org/other-model",
                        "url": "https://huggingface.co/org/other/resolve/main/other.gguf",
                        "size": "1GB",
                        "jinja": True,
                    },
                ]
            }
        )
    )

    with patch("server.services.llamacpp_service.get_main_dir", return_value=tmp_path):
        models = _read_models()

    assert models["org/model-q4-k-m"] == LlamacppModel(
        url="https://huggingface.co/org/model/resolve/main/model-Q4_K_M.gguf", size="4GB", jinja=False
    )
    assert models["org/other-model"] == LlamacppModel(
        url="https://huggingface.co/org/other/resolve/main/other.gguf", size="1GB", jinja=True
    )


def test_read_models_raises_validationerror_naming_malformed_entry(tmp_path: Path) -> None:
    static_dir = tmp_path / "static"
    static_dir.mkdir()
    (static_dir / "llamacpp-min.json").write_text(json.dumps({"llms": [{"name": "org/broken-model", "size": "1GB"}]}))

    with (
        patch("server.services.llamacpp_service.get_main_dir", return_value=tmp_path),
        pytest.raises(ValidationError, match="url"),
    ):
        _read_models()


def test_const_has_cpu_image() -> None:
    assert "cpu" in _const.images
    assert _const.images["cpu"].name


def test_const_has_gpu_image() -> None:
    assert "gpu" in _const.images
    assert _const.images["gpu"].name


def test_const_has_vulkan_image() -> None:
    assert "vulkan" in _const.images


@pytest.mark.parametrize(
    ("hardware_list", "expected_image_key"),
    [
        ([MagicMock(spec=NvidiaGpuInfo)], "gpu"),
        ([MagicMock(spec=IntelGpuInfo)], "vulkan"),
        ([MagicMock(spec=CpuInfo)], "cpu"),
        ([], "cpu"),
    ],
)
def test_get_image_returns_correct_image(svc: LLamacppService, hardware_list: list[GpuInfo], expected_image_key: str) -> None:
    image = svc._get_image(hardware_list)  # pyright: ignore[reportPrivateUsage]

    assert image == _const.images[expected_image_key]  # pyright: ignore[reportArgumentType]


def test_get_spec_has_hardware_field(svc: LLamacppService) -> None:
    spec = svc.get_spec()

    field_names = [f.name for f in spec.fields]
    assert "hardware" in field_names


def test_get_model_spec_has_rename_field(svc: LLamacppService) -> None:
    spec = svc.get_model_spec()

    field_names = [f.name for f in spec.fields]
    assert "rename" in field_names


def test_get_model_spec_has_max_model_length_field(svc: LLamacppService) -> None:
    spec = svc.get_model_spec()

    field_names = [f.name for f in spec.fields]
    assert "max_model_length" in field_names


def test_get_custom_model_spec_not_none(svc: LLamacppService) -> None:
    assert svc.get_custom_model_spec() is not None


def test_model_installed_info_get_info() -> None:
    model = _make_model_installed_info(registration_id="reg-42")

    info = model.get_info()

    assert info.registration_id == "reg-42"


def test_get_size_cpu_only(deps: dict[str, Any]) -> None:
    deps["hardware"].nvidia_gpus = []
    deps["hardware"].intel_gpus = []
    svc = LLamacppService(**deps)

    sizes = svc.get_size()

    assert "cpu" in sizes
    assert "gpu" not in sizes
    assert "vulkan" not in sizes


def test_get_size_with_nvidia_gpu(deps: dict[str, Any]) -> None:
    deps["hardware"].nvidia_gpus = [MagicMock()]
    svc = LLamacppService(**deps)

    sizes = svc.get_size()

    assert "cpu" in sizes
    assert "gpu" in sizes


def test_get_size_with_intel_gpu(deps: dict[str, Any]) -> None:
    deps["hardware"].intel_gpus = [MagicMock()]
    svc = LLamacppService(**deps)

    sizes = svc.get_size()

    assert "cpu" in sizes
    assert "vulkan" in sizes


def test_get_installed_info_returns_spec_when_installed(svc: LLamacppService) -> None:
    installed = _make_installed_info(svc)
    installed.options.spec["hardware"] = True
    svc.instances_info["default"].installed = installed

    result = svc.get_installed_info("default")

    assert result == installed.options.spec


def test_get_installed_info_calls_helper_when_none(svc: LLamacppService) -> None:
    svc.instances_info["default"].installed = None

    with patch.object(svc, "_get_service_installed_info", return_value=False) as mock:  # pyright: ignore[reportPrivateUsage]
        result = svc.get_installed_info("default")

    assert mock.call_count == 1
    assert result is False


def test_generate_instance_config_none_info(svc: LLamacppService) -> None:
    config = svc._generate_instance_config("default", None, None)  # pyright: ignore[reportPrivateUsage]

    assert config.options is None
    assert config.models == []


def test_generate_instance_config_persists_resolved_container_name(svc: LLamacppService) -> None:
    """Resolved container name round-trips into the persisted ModelConfig, so a suffix change can't rename it."""
    info = _make_installed_info(svc)
    info.models["m1"] = _make_model_installed_info("m1")
    info.models["m1"].docker.container_name = "pinned-container-name"

    config = svc._generate_instance_config("default", info, None)  # pyright: ignore[reportPrivateUsage]

    assert (config.models or [])[0].container_name == "pinned-container-name"


def test_generate_instance_config_with_info(svc: LLamacppService) -> None:
    info = _make_installed_info(svc)
    info.models["m1"] = _make_model_installed_info("m1")

    config = svc._generate_instance_config("default", info, None)  # pyright: ignore[reportPrivateUsage]

    assert config.options == info.options
    assert len(config.models or []) == 1


def test_generate_instance_config_embeds_definition_when_in_registry(svc: LLamacppService) -> None:
    info = _make_installed_info(svc)
    info.models["m1"] = _make_model_installed_info("m1")
    svc.models["default"]["m1"] = LlamacppModel(url="https://example.com/m1.gguf", size="1 GB")

    config = svc._generate_instance_config("default", info, None)  # pyright: ignore[reportPrivateUsage]

    assert (config.models or [])[0].definition == {
        "url": "https://example.com/m1.gguf",
        "size": "1 GB",
        "custom": None,
        "jinja": False,
        "supports_vision": None,
    }


def test_generate_instance_config_omits_definition_when_missing_from_registry(svc: LLamacppService) -> None:
    info = _make_installed_info(svc)
    info.models["ghost-model"] = _make_model_installed_info("ghost-model")

    config = svc._generate_instance_config("default", info, None)  # pyright: ignore[reportPrivateUsage]

    assert (config.models or [])[0].definition is None


def test_restore_model_definition_reinstates_model(svc: LLamacppService) -> None:
    definition = {"url": "https://example.com/ghost.gguf", "size": "2 GB"}

    svc._restore_model_definition("default", "ghost-model", definition)  # pyright: ignore[reportPrivateUsage]

    assert svc.models["default"]["ghost-model"] == LlamacppModel(url="https://example.com/ghost.gguf", size="2 GB")


def test_load_download_info_returns_dataclass(svc: LLamacppService) -> None:
    result = svc._load_download_info({"model_path": "/tmp/x.gguf"})  # pyright: ignore[reportPrivateUsage]

    assert isinstance(result, DownloadedInfo)
    assert result.model_path == "/tmp/x.gguf"


def test_get_docker_compose_file_path_raises_400_no_model_id(svc: LLamacppService) -> None:
    installed = _make_installed_info(svc)
    svc.instances_info["default"].installed = installed

    with pytest.raises(HTTPException) as exc_info:
        svc.get_docker_compose_file_path("default", None)

    assert exc_info.value.status_code == 400


def test_get_docker_compose_file_path_raises_400_model_not_installed(svc: LLamacppService) -> None:
    installed = _make_installed_info(svc)
    svc.instances_info["default"].installed = installed

    with pytest.raises(HTTPException) as exc_info:
        svc.get_docker_compose_file_path("default", "some-model")

    assert exc_info.value.status_code == 400


def test_get_docker_compose_file_path_returns_path_for_installed_model(svc: LLamacppService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info(svc)
    installed.models["test-model"] = _make_model_installed_info("test-model")
    svc.instances_info["default"].installed = installed
    expected = Path("/some/path/docker-compose.yml")
    deps["docker_service"].get_docker_compose_file_path.return_value = expected

    result = svc.get_docker_compose_file_path("default", "test-model")

    assert result == expected


def test_add_custom_model_registers_entry(svc: LLamacppService) -> None:
    custom = CustomModel(id="c-1", data={"id": "my-gguf", "url": "https://example.com/model.gguf", "size": "2GB"})

    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    assert "my-gguf" in svc.models["default"]
    assert svc.models["default"]["my-gguf"].custom == "c-1"


def test_add_custom_model_duplicate_raises_400(svc: LLamacppService) -> None:
    custom = CustomModel(id="c-2", data={"id": "dup", "url": "https://example.com/model.gguf", "size": "1GB"})
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    with pytest.raises(HTTPException) as exc_info:
        svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400


def test_add_custom_model_creates_dict_for_new_instance(svc: LLamacppService) -> None:
    svc.instances_info["extra"] = Instance(None, None, {}, InstanceConfig())
    custom = CustomModel(id="c-3", data={"id": "new-model", "url": "https://example.com/model.gguf", "size": "1GB"})

    svc._add_custom_model("extra", custom)  # pyright: ignore[reportPrivateUsage]

    assert "new-model" in svc.models["extra"]


def test_remove_custom_model_deletes_entry(svc: LLamacppService) -> None:
    custom = CustomModel(id="c-4", data={"id": "to-remove", "url": "https://example.com/model.gguf", "size": "1GB"})
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    svc._remove_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    assert "to-remove" not in svc.models["default"]


def test_remove_custom_model_raises_400_when_in_use(svc: LLamacppService) -> None:
    custom = CustomModel(id="c-5", data={"id": "in-use", "url": "https://example.com/model.gguf", "size": "1GB"})
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]
    installed = _make_installed_info(svc)
    installed.models["in-use"] = _make_model_installed_info("in-use")
    svc.instances_info["default"].installed = installed

    with pytest.raises(HTTPException) as exc_info:
        svc._remove_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_list_models_raises_404_for_unknown_instance(svc: LLamacppService) -> None:
    with pytest.raises(HTTPException) as exc_info:
        await svc.list_models("nonexistent", ListModelsFilters())

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_list_models_returns_all_models_no_filter(svc: LLamacppService) -> None:
    installed = _make_installed_info(svc)
    svc.instances_info["default"].installed = installed

    result = await svc.list_models("default", ListModelsFilters())

    assert len(result.list) == len(svc.models["default"])


@pytest.mark.asyncio
async def test_list_models_filters_installed_only(svc: LLamacppService) -> None:
    installed = _make_installed_info(svc)
    model_id = next(iter(svc.models["default"]))
    installed.models[model_id] = _make_model_installed_info(model_id)
    svc.instances_info["default"].installed = installed

    result = await svc.list_models("default", ListModelsFilters(installed=True))

    assert len(result.list) >= 1
    assert all(bool(m.installed) for m in result.list)


@pytest.mark.asyncio
async def test_get_model_raises_400_for_unknown_model_id(svc: LLamacppService) -> None:
    installed = _make_installed_info(svc)
    svc.instances_info["default"].installed = installed

    with pytest.raises(HTTPException) as exc_info:
        await svc.get_model("default", "nonexistent-model")

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_get_model_returns_correct_model(svc: LLamacppService) -> None:
    installed = _make_installed_info(svc)
    svc.instances_info["default"].installed = installed
    model_id = next(iter(svc.models["default"]))

    result = await svc.get_model("default", model_id)

    assert result.id == model_id


@pytest.mark.asyncio
async def test_download_model_emits_initial_and_final_progress(svc: LLamacppService, deps: dict[str, Any], tmp_path: Path) -> None:
    model = LlamacppModel(url="https://example.com/model.gguf", size="1GB")
    stream = MagicMock()
    local_path = tmp_path / "model.gguf"
    local_path.write_bytes(b"data")

    async def mock_download(*args: object):  # type: ignore[misc]
        yield SuccessDownloadPacket(local_path=local_path, filename="model.gguf")

    deps["model_downloader"].download = mock_download

    with patch.object(svc, "_get_working_dir", return_value=tmp_path):  # pyright: ignore[reportPrivateUsage]
        await svc._download_model(stream, model)  # pyright: ignore[reportPrivateUsage]

    assert stream.emit.call_count >= 2


@pytest.mark.asyncio
async def test_download_model_updates_progress_on_downloaded_packet(svc: LLamacppService, deps: dict[str, Any], tmp_path: Path) -> None:
    model = LlamacppModel(url="https://example.com/model.gguf", size="100MB")
    stream = MagicMock()

    async def mock_download(*args: object):  # type: ignore[misc]
        yield DownloadedPacket(downloaded_bytes_size=1024 * 1024)

    deps["model_downloader"].download = mock_download

    with patch.object(svc, "_get_working_dir", return_value=tmp_path):  # pyright: ignore[reportPrivateUsage]
        await svc._download_model(stream, model)  # pyright: ignore[reportPrivateUsage]

    progress_calls = [call for call in stream.emit.call_args_list if isinstance(call.args[0], dict) and call.args[0].get("value", 0) > 0]
    assert len(progress_calls) >= 1


@pytest.mark.asyncio
async def test_download_model_or_set_progress_starts_new_download(svc: LLamacppService) -> None:
    stream = MagicMock()
    model = LlamacppModel(url="https://example.com/model.gguf", size="1GB")
    svc.models["default"]["my-model"] = model

    with patch.object(svc, "_download_model", new_callable=AsyncMock, return_value=(Path("/tmp/model.gguf"), "model.gguf")) as mock_dl:
        await svc._download_model_or_set_progress(stream, model, "my-model")  # pyright: ignore[reportPrivateUsage]

    assert mock_dl.call_count == 1
    assert "my-model" not in svc.models_download_progress


@pytest.mark.asyncio
async def test_download_model_or_set_progress_cleans_up_on_failure(svc: LLamacppService) -> None:
    stream = MagicMock()
    model = LlamacppModel(url="https://example.com/model.gguf", size="1GB")
    svc.models["default"]["my-model"] = model

    with (
        patch.object(svc, "_download_model", new_callable=AsyncMock, side_effect=HTTPException(400, "boom")),  # pyright: ignore[reportPrivateUsage]
        pytest.raises(HTTPException),
    ):
        await svc._download_model_or_set_progress(stream, model, "my-model")  # pyright: ignore[reportPrivateUsage]

    assert "my-model" not in svc.models_download_progress


@pytest.mark.asyncio
async def test_download_model_or_set_progress_retries_download_after_failure(svc: LLamacppService) -> None:
    stream1 = MagicMock()
    stream2 = MagicMock()
    model = LlamacppModel(url="https://example.com/model.gguf", size="1GB")
    svc.models["default"]["my-model"] = model

    mock_dl = AsyncMock(side_effect=[HTTPException(400, "boom"), (Path("/tmp/model.gguf"), "model.gguf")])
    with patch.object(svc, "_download_model", mock_dl):  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(HTTPException):
            await svc._download_model_or_set_progress(stream1, model, "my-model")  # pyright: ignore[reportPrivateUsage]
        await svc._download_model_or_set_progress(stream2, model, "my-model")  # pyright: ignore[reportPrivateUsage]

    assert mock_dl.call_count == 2


@pytest.mark.asyncio
async def test_download_model_or_set_progress_forwards_existing_stream(svc: LLamacppService) -> None:
    existing_stream: Stream[StreamChunk] = Stream()  # type: ignore[type-arg]
    chunk = StreamChunkProgress(type="progress", stage="download", value=0.5, data={})
    existing_stream.emit(chunk)
    existing_stream.close()
    model = LlamacppModel(url="https://example.com/model.gguf", size="1GB")
    svc.models_download_progress["my-model"] = existing_stream  # type: ignore[assignment]
    output_stream = MagicMock()

    await svc._download_model_or_set_progress(output_stream, model, "my-model")  # pyright: ignore[reportPrivateUsage]

    assert output_stream.emit.call_count == 1
    args, _ = output_stream.emit.call_args
    assert args[0] == chunk


@pytest.mark.asyncio
async def test_install_model_returns_already_installed(svc: LLamacppService) -> None:
    installed = _make_installed_info(svc)
    model_id = next(iter(svc.models["default"]))
    installed.models[model_id] = _make_model_installed_info(model_id)
    svc.instances_info["default"].installed = installed

    promise = await svc._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
    result = await promise.wait()

    assert result.status == "OK"
    assert "Already installed" in result.details


@pytest.mark.asyncio
async def test_install_model_raises_400_for_unknown_model(svc: LLamacppService) -> None:
    installed = _make_installed_info(svc)
    svc.instances_info["default"].installed = installed

    with pytest.raises(HTTPException) as exc_info:
        await svc._install_model("default", "nonexistent-model", InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_install_model_calls_docker_install(svc: LLamacppService, deps: dict[str, Any], tmp_path: Path) -> None:
    _setup_install_mocks(svc, deps)
    model_id = next(iter(svc.models["default"]))

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=(tmp_path / "model.gguf", "model.gguf")),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.llamacpp_service.get_gguf_context_window", new_callable=AsyncMock, return_value=4096),
        patch("server.services.llamacpp_service.get_base_url", return_value="http://localhost:8080"),
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert deps["docker_service"].install_and_run_docker.call_count == 1


@pytest.mark.asyncio
async def test_install_model_uses_container_name_override_when_given(svc: LLamacppService, deps: dict[str, Any], tmp_path: Path) -> None:
    _setup_install_mocks(svc, deps)
    model_id = next(iter(svc.models["default"]))

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=(tmp_path / "model.gguf", "model.gguf")),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.llamacpp_service.get_gguf_context_window", new_callable=AsyncMock, return_value=4096),
        patch("server.services.llamacpp_service.get_base_url", return_value="http://localhost:8080"),
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn(spec={}), "pinned-name")  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    docker_options = deps["docker_service"].install_and_run_docker.call_args.args[0]
    assert docker_options.container_name == "pinned-name"
    deps["docker_service"].get_docker_container_name.assert_not_called()


@pytest.mark.asyncio
async def test_install_model_falls_back_to_generated_container_name(svc: LLamacppService, deps: dict[str, Any], tmp_path: Path) -> None:
    _setup_install_mocks(svc, deps)
    model_id = next(iter(svc.models["default"]))

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=(tmp_path / "model.gguf", "model.gguf")),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.llamacpp_service.get_gguf_context_window", new_callable=AsyncMock, return_value=4096),
        patch("server.services.llamacpp_service.get_base_url", return_value="http://localhost:8080"),
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    docker_options = deps["docker_service"].install_and_run_docker.call_args.args[0]
    assert docker_options.container_name == deps["docker_service"].get_docker_container_name.return_value


@pytest.mark.asyncio
async def test_install_model_registers_endpoint(svc: LLamacppService, deps: dict[str, Any], tmp_path: Path) -> None:
    _setup_install_mocks(svc, deps)
    model_id = next(iter(svc.models["default"]))

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=(tmp_path / "model.gguf", "model.gguf")),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.llamacpp_service.get_gguf_context_window", new_callable=AsyncMock, return_value=4096),
        patch("server.services.llamacpp_service.get_base_url", return_value="http://localhost:8080"),
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert deps["endpoint_registry"].register_chat_completion_as_proxy.call_count == 1


@pytest.mark.asyncio
async def test_install_model_supports_vision_defaults_to_none(svc: LLamacppService, deps: dict[str, Any], tmp_path: Path) -> None:
    """A catalog entry with no confirmed capability must register with `supports_vision=None`.

    `LlamacppModel` has no `has_mmproj` field - the catalog's mmproj presence hint never reaches
    installed-model registration - so this pins that an unresolved catalog value never turns into
    a wrong "vision works" guess here.
    """
    _setup_install_mocks(svc, deps)
    model_id = "vision-candidate"
    svc.models["default"][model_id] = LlamacppModel(url="https://example.com/model.gguf", size="1 GB", supports_vision=None)

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=(tmp_path / "model.gguf", "model.gguf")),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.llamacpp_service.get_gguf_context_window", new_callable=AsyncMock, return_value=4096),
        patch("server.services.llamacpp_service.get_base_url", return_value="http://localhost:8080"),
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    call_kwargs = deps["endpoint_registry"].register_chat_completion_as_proxy.call_args.kwargs
    assert call_kwargs["props"].supports_vision is None


@pytest.mark.asyncio
async def test_install_model_forwards_confirmed_supports_vision(svc: LLamacppService, deps: dict[str, Any], tmp_path: Path) -> None:
    """A catalog entry with a confirmed `supports_vision` value is still forwarded as-is."""
    _setup_install_mocks(svc, deps)
    model_id = "known-vision-model"
    svc.models["default"][model_id] = LlamacppModel(url="https://example.com/model.gguf", size="1 GB", supports_vision=True)

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=(tmp_path / "model.gguf", "model.gguf")),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.llamacpp_service.get_gguf_context_window", new_callable=AsyncMock, return_value=4096),
        patch("server.services.llamacpp_service.get_base_url", return_value="http://localhost:8080"),
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    call_kwargs = deps["endpoint_registry"].register_chat_completion_as_proxy.call_args.kwargs
    assert call_kwargs["props"].supports_vision is True


@pytest.mark.asyncio
async def test_install_model_registers_capacity_from_num_parallel(svc: LLamacppService, deps: dict[str, Any], tmp_path: Path) -> None:
    installed = _setup_install_mocks(svc, deps)
    installed.parsed_options.num_parallel = 4
    model_id = next(iter(svc.models["default"]))

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=(tmp_path / "model.gguf", "model.gguf")),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.llamacpp_service.get_gguf_context_window", new_callable=AsyncMock, return_value=4096),
        patch("server.services.llamacpp_service.get_base_url", return_value="http://localhost:8080"),
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    call_kwargs = deps["endpoint_registry"].register_chat_completion_as_proxy.call_args.kwargs
    assert call_kwargs["registration_options"].capacity_state == 4


@pytest.mark.asyncio
async def test_install_model_records_model_as_downloaded(svc: LLamacppService, deps: dict[str, Any], tmp_path: Path) -> None:
    _setup_install_mocks(svc, deps)
    model_id = next(iter(svc.models["default"]))

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=(tmp_path / "model.gguf", "model.gguf")),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.llamacpp_service.get_gguf_context_window", new_callable=AsyncMock, return_value=4096),
        patch("server.services.llamacpp_service.get_base_url", return_value="http://localhost:8080"),
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert model_id in svc.models_downloaded


@pytest.mark.asyncio
async def test_install_model_uses_rename_as_registered_name(svc: LLamacppService, deps: dict[str, Any], tmp_path: Path) -> None:
    installed = _setup_install_mocks(svc, deps)
    model_id = next(iter(svc.models["default"]))
    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=(tmp_path / "model.gguf", "model.gguf")),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.llamacpp_service.get_gguf_context_window", new_callable=AsyncMock, return_value=4096),
        patch("server.services.llamacpp_service.get_base_url", return_value="http://localhost:8080"),
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn(spec={"rename": "my-alias"}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert installed.models[model_id].registered_name == "my-alias"


@pytest.mark.asyncio
async def test_install_model_appends_ctx_size_when_max_model_length_set(svc: LLamacppService, deps: dict[str, Any], tmp_path: Path) -> None:
    _setup_install_mocks(svc, deps)
    model_id = next(iter(svc.models["default"]))
    captured: list[object] = []

    async def capture_docker(opts: object) -> tuple[int, bool, bool]:
        captured.append(opts)
        return (8080, True, False)

    deps["docker_service"].install_and_run_docker = capture_docker

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=(tmp_path / "model.gguf", "model.gguf")),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.llamacpp_service.get_gguf_context_window", new_callable=AsyncMock, return_value=4096),
        patch("server.services.llamacpp_service.get_base_url", return_value="http://localhost:8080"),
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn(spec={"max_model_length": 4096}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert len(captured) == 1
    assert "--ctx-size" in captured[0].command  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_install_model_appends_jinja_flag_for_jinja_model(svc: LLamacppService, deps: dict[str, Any], tmp_path: Path) -> None:
    _setup_install_mocks(svc, deps)
    jinja_model_id = "org/jinja-model"
    svc.models["default"][jinja_model_id] = LlamacppModel(url="https://example.com/model.gguf", size="1GB", jinja=True)
    captured: list[object] = []

    async def capture_docker(opts: object) -> tuple[int, bool, bool]:
        captured.append(opts)
        return (8080, True, False)

    deps["docker_service"].install_and_run_docker = capture_docker

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=(tmp_path / "model.gguf", "model.gguf")),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.llamacpp_service.get_gguf_context_window", new_callable=AsyncMock, return_value=4096),
        patch("server.services.llamacpp_service.get_base_url", return_value="http://localhost:8080"),
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
    ):
        promise = await svc._install_model("default", jinja_model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert len(captured) == 1
    assert "--jinja" in captured[0].command  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_uninstall_model_unregisters_endpoint_and_removes_from_info(svc: LLamacppService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info(svc)
    installed.models["test-model"] = _make_model_installed_info("test-model", "reg-1")
    svc.instances_info["default"].installed = installed
    deps["docker_service"].uninstall_docker = AsyncMock()

    await svc._uninstall_model("default", "test-model", UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert "test-model" not in installed.models
    assert deps["endpoint_registry"].unregister_chat_completion.call_count == 1
    assert deps["endpoint_registry"].unregister_chat_completion.call_args[0] == ("test-model", "reg-1")


@pytest.mark.asyncio
async def test_uninstall_model_purges_file_and_downloaded_entry(svc: LLamacppService, deps: dict[str, Any], tmp_path: Path) -> None:
    installed = _make_installed_info(svc)
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"data")
    model_info = _make_model_installed_info("test-model")
    model_info.model_path = model_file
    installed.models["test-model"] = model_info
    svc.instances_info["default"].installed = installed
    svc.models_downloaded["test-model"] = DownloadedInfo(model_path=str(model_file))
    deps["docker_service"].uninstall_docker = AsyncMock()

    await svc._uninstall_model("default", "test-model", UninstallModelIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert not model_file.exists()
    assert "test-model" not in svc.models_downloaded


@pytest.mark.asyncio
async def test_uninstall_model_ignores_unknown_model_id(svc: LLamacppService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info(svc)
    svc.instances_info["default"].installed = installed
    deps["docker_service"].uninstall_docker = AsyncMock()

    await svc._uninstall_model("default", "unknown-model", UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert deps["endpoint_registry"].unregister_chat_completion.call_count == 0
    assert deps["docker_service"].uninstall_docker.call_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_num_parallel", [0, -1])
async def test_install_instance_clamps_non_positive_num_parallel(svc: LLamacppService, bad_num_parallel: int) -> None:
    """A non-positive `num_parallel` must be clamped to 1 at instance setup rather than rejected -
    such a value may already be persisted in an existing config.json from before `num_parallel`
    gained validation, and rejecting it would leave the instance's models unable to load at all."""
    options = InstallServiceIn(spec={"hardware": False, "num_parallel": bad_num_parallel})

    with (
        patch.object(svc, "_download_image_or_set_progress", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_verify_docker_image", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
    ):
        promise = await svc._install_instance("default", options)  # pyright: ignore[reportPrivateUsage]
        installed = await promise.wait()

    assert installed.parsed_options.num_parallel == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_num_parallel", [0, -1])
async def test_update_instance_clamps_non_positive_num_parallel(svc: LLamacppService, deps: dict[str, Any], bad_num_parallel: int) -> None:
    """A non-positive `num_parallel` on `update_instance` must be clamped to 1 rather than
    rejected, consistent with `_install_instance`'s handling of the same value."""
    svc.instances_info["default"].installed = _make_installed_info(svc)
    options = InstallServiceIn(spec={"hardware": False, "num_parallel": bad_num_parallel})
    deps["service_provider"].save_service_config = AsyncMock()

    with (
        patch.object(svc, "_uninstall_instance", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_download_image_or_set_progress", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_verify_docker_image", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
    ):
        promise = await svc.update_instance("default", options)
        await promise.wait()

    installed = svc.instances_info["default"].installed
    assert installed is not None
    assert installed.parsed_options.num_parallel == 1


@pytest.mark.asyncio
async def test_install_instance_returns_installed_info(svc: LLamacppService) -> None:
    options = InstallServiceIn(spec={"hardware": False})

    with (
        patch.object(svc, "_download_image_or_set_progress", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_verify_docker_image", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
    ):
        promise = await svc._install_instance("default", options)  # pyright: ignore[reportPrivateUsage]
        result = await promise.wait()

    assert isinstance(result, InstalledInfo)
    assert result.models == {}


@pytest.mark.asyncio
async def test_uninstall_instance_calls_uninstall_model_for_each_model(svc: LLamacppService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info(svc)
    installed.models["m1"] = _make_model_installed_info("m1")
    installed.models["m2"] = _make_model_installed_info("m2")
    svc.instances_info["default"].installed = installed
    deps["docker_service"].uninstall_docker = AsyncMock()

    with patch.object(svc, "_uninstall_model", new_callable=AsyncMock) as mock_uninstall:  # pyright: ignore[reportPrivateUsage]
        await svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert mock_uninstall.call_count == 2


@pytest.mark.asyncio
async def test_uninstall_instance_purge_clears_service_state(svc: LLamacppService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info(svc)
    svc.instances_info["default"].installed = installed
    deps["docker_service"].remove_image = AsyncMock()
    deps["docker_service"].uninstall_docker = AsyncMock()

    with (
        patch.object(svc, "_uninstall_model", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_clear_working_dir", new_callable=AsyncMock) as mock_clear,  # pyright: ignore[reportPrivateUsage]
    ):
        await svc._uninstall_instance("default", UninstallServiceIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert svc.service_downloaded is False
    assert mock_clear.call_count == 1


@pytest.mark.asyncio
async def test_uninstall_instance_purge_with_other_instance_installed_skips_cleanup(svc: LLamacppService, deps: dict[str, Any]) -> None:
    svc.instances_info["extra"] = Instance(None, None, {}, InstanceConfig())
    svc.instances_info["extra"].installed = _make_installed_info(svc, "extra")
    svc.instances_info["default"].installed = _make_installed_info(svc)
    deps["docker_service"].remove_image = AsyncMock()

    with (
        patch.object(svc, "_uninstall_model", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_clear_working_dir", new_callable=AsyncMock) as mock_clear,  # pyright: ignore[reportPrivateUsage]
    ):
        await svc._uninstall_instance("default", UninstallServiceIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert deps["docker_service"].remove_image.call_count == 0
    assert mock_clear.call_count == 0
    assert svc.instances_info["default"].installed is None
    assert "extra" in svc.instances_info


@pytest.mark.asyncio
async def test_uninstall_instance_purge_removes_per_model_docker_images(svc: LLamacppService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info(svc)
    model_info = _make_model_installed_info("test-model")
    model_info.docker.image = "llamacpp-model-image:tag"
    installed.models["test-model"] = model_info
    svc.instances_info["default"].installed = installed
    deps["docker_service"].remove_image = AsyncMock()

    with (
        patch.object(svc, "_uninstall_model", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_clear_working_dir", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
    ):
        await svc._uninstall_instance("default", UninstallServiceIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    removed_images = {c.args[0] for c in deps["docker_service"].remove_image.call_args_list}
    assert "llamacpp-model-image:tag" in removed_images


@pytest.mark.asyncio
async def test_stop_instance_does_nothing_when_not_installed(svc: LLamacppService) -> None:
    svc.instances_info["default"].installed = None

    with patch.object(svc, "_stop_dockers_parallel", new_callable=AsyncMock) as mock_stop:  # pyright: ignore[reportPrivateUsage]
        await svc.stop_instance("default")

    assert mock_stop.call_count == 0


@pytest.mark.asyncio
async def test_stop_instance_stops_all_containers(svc: LLamacppService) -> None:
    installed = _make_installed_info(svc)
    installed.models["m1"] = _make_model_installed_info("m1")
    installed.models["m2"] = _make_model_installed_info("m2")
    svc.instances_info["default"].installed = installed

    with patch.object(svc, "_stop_dockers_parallel", new_callable=AsyncMock) as mock_stop:  # pyright: ignore[reportPrivateUsage]
        await svc.stop_instance("default")

    assert mock_stop.call_count == 1

    called_dockers = mock_stop.call_args[0][0]

    assert len(called_dockers) == 2


@pytest.mark.asyncio
async def test_install_instance_loads_default_models_for_new_instance(svc: LLamacppService) -> None:
    svc.instances_info["inst2"] = Instance(None, None, {}, InstanceConfig())
    options = InstallServiceIn(spec={"hardware": False})

    with (
        patch.object(svc, "_download_image_or_set_progress", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_verify_docker_image", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
        patch.object(svc, "load_default_models") as mock_load,
    ):
        promise = await svc._install_instance("inst2", options)  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert mock_load.call_count == 1
    assert mock_load.call_args == call("inst2")


@pytest.mark.asyncio
async def test_install_instance_adds_hardware_key_when_missing(svc: LLamacppService, deps: dict[str, Any]) -> None:
    options = InstallServiceIn(spec={})
    deps["docker_service"].has_gpu_support = False

    with (
        patch.object(svc, "_download_image_or_set_progress", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_verify_docker_image", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
    ):
        await svc._install_instance("default", options)  # pyright: ignore[reportPrivateUsage]

    assert "hardware" in options.spec


@pytest.mark.asyncio
async def test_uninstall_instance_purge_removes_non_default_instance(svc: LLamacppService, deps: dict[str, Any]) -> None:
    svc.instances_info["inst2"] = Instance(None, None, {}, InstanceConfig())
    installed = _make_installed_info(svc)
    svc.instances_info["inst2"].installed = installed
    deps["docker_service"].remove_image = AsyncMock()

    with (
        patch.object(svc, "_uninstall_model", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_clear_working_dir", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
    ):
        await svc._uninstall_instance("inst2", UninstallServiceIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert "inst2" not in svc.instances_info


@pytest.mark.asyncio
async def test_list_models_skips_instances_not_in_filter(svc: LLamacppService) -> None:
    svc.instances_info["inst2"] = Instance(None, None, {}, InstanceConfig())
    svc.models["inst2"] = {"extra-model": LlamacppModel(url="http://x.com/x.gguf", size="1GB")}
    installed = _make_installed_info(svc)
    svc.instances_info["default"].installed = installed

    result = await svc.list_models("default", ListModelsFilters())

    model_ids = [m.id for m in result.list]
    assert not any(m == "extra-model" for m in model_ids)


@pytest.mark.asyncio
async def test_download_model_handles_pre_download_packet_with_size(svc: LLamacppService, deps: dict[str, Any], tmp_path: Path) -> None:
    model = LlamacppModel(url="https://example.com/model.gguf", size="100MB")
    stream = MagicMock()

    async def mock_download(*args: object):  # type: ignore[misc]
        yield PreDownloadPacket(file_bytes_size=50 * 1024 * 1024)
        yield SuccessDownloadPacket(local_path=tmp_path / "model.gguf", filename="model.gguf")

    deps["model_downloader"].download = mock_download

    with patch.object(svc, "_get_working_dir", return_value=tmp_path):  # pyright: ignore[reportPrivateUsage]
        _local_path, filename = await svc._download_model(stream, model)  # pyright: ignore[reportPrivateUsage]

    assert filename == "model.gguf"


@pytest.mark.asyncio
async def test_download_model_or_set_progress_forwards_chunk_with_data(svc: LLamacppService) -> None:
    existing_stream: Stream[StreamChunk] = Stream()  # type: ignore[type-arg]
    chunk = StreamChunkProgress(
        type="progress", stage="download", value=0.9, data={"local_model_path": "/tmp/model.gguf", "filename": "model.gguf"}
    )
    existing_stream.emit(chunk)
    existing_stream.close()
    model = LlamacppModel(url="https://example.com/model.gguf", size="1GB")
    svc.models_download_progress["my-model"] = existing_stream  # type: ignore[assignment]
    output_stream = MagicMock()

    result_path, result_name = await svc._download_model_or_set_progress(output_stream, model, "my-model")  # pyright: ignore[reportPrivateUsage]

    assert result_name == "model.gguf"
    assert result_path is not None


@pytest.mark.asyncio
async def test_download_model_or_set_progress_breaks_on_non_download_chunk(svc: LLamacppService) -> None:
    existing_stream: Stream[StreamChunk] = Stream()  # type: ignore[type-arg]
    finish_chunk: StreamChunk = {"type": "finish", "status": "ok"}  # type: ignore[assignment]
    existing_stream.emit(finish_chunk)
    existing_stream.close()
    model = LlamacppModel(url="https://example.com/model.gguf", size="1GB")
    svc.models_download_progress["my-model"] = existing_stream  # type: ignore[assignment]
    output_stream = MagicMock()

    await svc._download_model_or_set_progress(output_stream, model, "my-model")  # pyright: ignore[reportPrivateUsage]

    assert output_stream.emit.call_count == 0


@pytest.mark.asyncio
async def test_install_model_raises_400_when_local_path_missing(svc: LLamacppService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info(svc)
    svc.instances_info["default"].installed = installed
    model_id = next(iter(svc.models["default"]))

    with patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=(None, "")):  # pyright: ignore[reportPrivateUsage]
        promise = await svc._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(HTTPException) as exc_info:
            await promise.wait()

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_uninstall_instance_when_not_installed_sets_installed_to_none(svc: LLamacppService) -> None:
    svc.instances_info["default"].installed = None

    await svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert svc.instances_info["default"].installed is None


@pytest.mark.asyncio
async def test_uninstall_instance_skips_model_installed_in_other_instance(svc: LLamacppService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info(svc)
    installed.models["shared-model"] = _make_model_installed_info("shared-model")
    svc.instances_info["default"].installed = installed
    deps["docker_service"].uninstall_docker = AsyncMock()

    with (
        patch.object(svc, "is_model_installed_in_other_instance", return_value=True),
        patch.object(svc, "_uninstall_model", new_callable=AsyncMock) as mock_uninstall,  # pyright: ignore[reportPrivateUsage]
    ):
        await svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert mock_uninstall.call_count == 0


@pytest.mark.asyncio
async def test_download_model_handles_pre_download_packet_with_zero_size(
    svc: LLamacppService, deps: dict[str, Any], tmp_path: Path
) -> None:
    model = LlamacppModel(url="https://example.com/model.gguf", size="100MB")
    stream = MagicMock()

    async def mock_download(*args: object):  # type: ignore[misc]
        yield PreDownloadPacket(file_bytes_size=0)
        yield SuccessDownloadPacket(local_path=tmp_path / "model.gguf", filename="model.gguf")

    deps["model_downloader"].download = mock_download

    with patch.object(svc, "_get_working_dir", return_value=tmp_path):  # pyright: ignore[reportPrivateUsage]
        _, filename = await svc._download_model(stream, model)  # pyright: ignore[reportPrivateUsage]

    assert filename == "model.gguf"


@pytest.mark.asyncio
async def test_download_model_handles_downloaded_packet_with_zero_bytes(svc: LLamacppService, deps: dict[str, Any], tmp_path: Path) -> None:
    model = LlamacppModel(url="https://example.com/model.gguf", size="100MB")
    stream = MagicMock()

    async def mock_download(*args: object):  # type: ignore[misc]
        yield DownloadedPacket(downloaded_bytes_size=0)
        yield SuccessDownloadPacket(local_path=tmp_path / "model.gguf", filename="model.gguf")

    deps["model_downloader"].download = mock_download

    with patch.object(svc, "_get_working_dir", return_value=tmp_path):  # pyright: ignore[reportPrivateUsage]
        _, filename = await svc._download_model(stream, model)  # pyright: ignore[reportPrivateUsage]

    assert filename == "model.gguf"


@pytest.mark.asyncio
async def test_uninstall_model_purge_skips_unlink_when_no_model_path(svc: LLamacppService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info(svc)
    model_info = _make_model_installed_info("test-model")
    installed.models["test-model"] = model_info
    svc.instances_info["default"].installed = installed
    svc.models_downloaded["test-model"] = DownloadedInfo(model_path="")
    deps["docker_service"].uninstall_docker = AsyncMock()

    await svc._uninstall_model("default", "test-model", UninstallModelIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert "test-model" not in svc.models_downloaded


@pytest.mark.asyncio
async def test_resolve_custom_model_size_returns_formatted_size(svc: LLamacppService, deps: dict[str, Any]) -> None:
    with patch("server.services.llamacpp_service.fetch_file_size_from_url", new=AsyncMock(return_value="3.0 GB")):
        result = await svc._resolve_custom_model_size({"url": "https://example.com/model.gguf"})  # pyright: ignore[reportPrivateUsage]

    assert result == "3.0 GB"


@pytest.mark.asyncio
async def test_resolve_custom_model_size_returns_none_on_exception(svc: LLamacppService, deps: dict[str, Any]) -> None:
    with patch("server.services.llamacpp_service.fetch_file_size_from_url", new=AsyncMock(side_effect=Exception("fail"))):
        result = await svc._resolve_custom_model_size({"url": "https://example.com/model.gguf"})  # pyright: ignore[reportPrivateUsage]

    assert result is None


@pytest.mark.asyncio
async def test_get_cached_vram_estimate_cache_hit(svc: LLamacppService) -> None:
    svc._vram_cache[("default", "test-model")] = 3.0  # pyright: ignore[reportPrivateUsage]

    result = await svc._get_cached_vram_estimate("default", "test-model", is_loaded=True)  # pyright: ignore[reportPrivateUsage]

    assert result == 3.0


@pytest.mark.asyncio
async def test_get_cached_vram_estimate_from_logs_non_none(svc: LLamacppService) -> None:
    with patch.object(svc, "_get_vram_from_logs", new_callable=AsyncMock, return_value=4.5):  # pyright: ignore[reportPrivateUsage]
        result = await svc._get_cached_vram_estimate("default", "test-model", is_loaded=True)  # pyright: ignore[reportPrivateUsage]

    assert result == 4.5
    assert svc._vram_cache[("default", "test-model")] == 4.5  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_get_cached_vram_estimate_stores_in_cache_when_estimate_available(svc: LLamacppService) -> None:
    with (
        patch.object(svc, "_get_vram_from_logs", new_callable=AsyncMock, return_value=None),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_get_vram_estimate", new_callable=AsyncMock, return_value=2.0),  # pyright: ignore[reportPrivateUsage]
    ):
        result = await svc._get_cached_vram_estimate("default", "test-model", is_loaded=True)  # pyright: ignore[reportPrivateUsage]

    assert result == 2.0
    assert svc._vram_cache[("default", "test-model")] == 2.0  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_install_model_appends_kv_cache_type_when_not_f16(svc: LLamacppService, deps: dict[str, Any], tmp_path: Path) -> None:
    installed = _make_installed_info(svc)
    installed.parsed_options.kv_cache_type = "q8_0"
    svc.instances_info["default"].installed = installed
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=(8080, True, False))
    deps["docker_service"].get_docker_subnet.return_value = None
    deps["docker_service"].get_docker_container_name.return_value = "container"
    deps["docker_service"].get_container_host.return_value = "localhost"
    deps["docker_service"].get_container_port.return_value = 8080
    captured: list[object] = []

    async def capture_docker(opts: object) -> tuple[int, bool, bool]:
        captured.append(opts)
        return (8080, True, False)

    deps["docker_service"].install_and_run_docker = capture_docker
    model_id = next(iter(svc.models["default"]))

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=(tmp_path / "model.gguf", "model.gguf")),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.llamacpp_service.get_gguf_context_window", new_callable=AsyncMock, return_value=4096),
        patch("server.services.llamacpp_service.get_base_url", return_value="http://localhost:8080"),
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert len(captured) == 1
    assert "--cache-type-k q8_0" in captured[0].command  # type: ignore[union-attr]
    assert "--cache-type-v q8_0" in captured[0].command  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_install_model_appends_parallel_flag_when_num_parallel_gt_1(
    svc: LLamacppService, deps: dict[str, Any], tmp_path: Path
) -> None:
    installed = _make_installed_info(svc)
    installed.parsed_options.num_parallel = 4
    svc.instances_info["default"].installed = installed
    captured: list[object] = []

    async def capture_docker(opts: object) -> tuple[int, bool, bool]:
        captured.append(opts)
        return (8080, True, False)

    deps["docker_service"].install_and_run_docker = capture_docker
    deps["docker_service"].get_docker_subnet.return_value = None
    deps["docker_service"].get_docker_container_name.return_value = "container"
    deps["docker_service"].get_container_host.return_value = "localhost"
    deps["docker_service"].get_container_port.return_value = 8080
    model_id = next(iter(svc.models["default"]))

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=(tmp_path / "model.gguf", "model.gguf")),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.llamacpp_service.get_gguf_context_window", new_callable=AsyncMock, return_value=4096),
        patch("server.services.llamacpp_service.get_base_url", return_value="http://localhost:8080"),
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert len(captured) == 1
    assert "--parallel 4" in captured[0].command  # type: ignore[union-attr]


def test_get_image_with_version_overrides_tag(svc: LLamacppService) -> None:
    image = svc._get_image(hardware=[], image_version="server-cuda-b9999")  # pyright: ignore[reportPrivateUsage]
    base = _const.images["cpu"].name.split(":")[0]
    assert image.name == f"{base}:server-cuda-b9999"
    assert image.size == _const.images["cpu"].size


@pytest.mark.asyncio
async def test_get_docker_tags_returns_filtered_tags(svc: LLamacppService) -> None:
    mock_client = AsyncMock()
    mock_client.get_tags = AsyncMock(return_value=["server-cuda-b7836", "server-b7836", "server-vulkan-b7836"])
    with (
        patch("server.services.llamacpp_service.registry_for", return_value=mock_client),
        patch("server.services.llamacpp_service.image_without_registry_prefix", return_value="ggml-org/llama.cpp"),
    ):
        tags = await svc.get_docker_tags("GPU")
    assert "server-cuda-b7836" in tags
    assert "server-b7836" not in tags


@pytest.mark.asyncio
async def test_get_docker_tags_cpu_hardware_returns_plain_tags(svc: LLamacppService) -> None:
    mock_client = AsyncMock()
    mock_client.get_tags = AsyncMock(return_value=["server-cuda-b7836", "server-b7836", "server-vulkan-b7836"])
    with (
        patch("server.services.llamacpp_service.registry_for", return_value=mock_client),
        patch("server.services.llamacpp_service.image_without_registry_prefix", return_value="ggml-org/llama.cpp"),
    ):
        tags = await svc.get_docker_tags("cpu")
    assert "server-b7836" in tags
    assert "server-cuda-b7836" not in tags
    assert "server-vulkan-b7836" not in tags


def test_get_default_docker_tag_gpu(svc: LLamacppService) -> None:
    assert svc.get_default_docker_tag("GPU") == _const.images["gpu"].name.split(":")[1]


def test_get_default_docker_tag_cpu(svc: LLamacppService) -> None:
    assert svc.get_default_docker_tag("cpu") == _const.images["cpu"].name.split(":")[1]


def test_get_default_docker_tag_vulkan(svc: LLamacppService) -> None:
    assert svc.get_default_docker_tag("intel") == _const.images["vulkan"].name.split(":")[1]


def test_get_default_docker_tag_defaults_to_gpu_when_no_hardware(svc: LLamacppService) -> None:
    assert svc.get_default_docker_tag(None) == _const.images["gpu"].name.split(":")[1]


def test_get_image_with_nvidia_gpu_and_version_overrides_tag(svc: LLamacppService) -> None:
    image = svc._get_image(hardware=[MagicMock(spec=NvidiaGpuInfo)], image_version="server-cuda-b9999")  # pyright: ignore[reportPrivateUsage]
    base = _const.images["gpu"].name.split(":")[0]
    assert image.name == f"{base}:server-cuda-b9999"
    assert image.size == _const.images["gpu"].size


def test_get_image_with_intel_gpu_and_version_overrides_tag(svc: LLamacppService) -> None:
    image = svc._get_image(hardware=[MagicMock(spec=IntelGpuInfo)], image_version="server-vulkan-b9999")  # pyright: ignore[reportPrivateUsage]
    base = _const.images["vulkan"].name.split(":")[0]
    assert image.name == f"{base}:server-vulkan-b9999"
    assert image.size == _const.images["vulkan"].size


def test_get_default_docker_image_name_no_gpu_returns_cpu_image(svc: LLamacppService) -> None:
    assert svc.get_default_docker_image_name() == _const.images["cpu"].name


def test_get_default_docker_image_name_with_nvidia_gpu_returns_gpu_image(deps: dict[str, Any]) -> None:
    deps["hardware"].gpus = [MagicMock(spec=NvidiaGpuInfo)]
    svc = LLamacppService(**deps)

    assert svc.get_default_docker_image_name() == _const.images["gpu"].name


@pytest.mark.asyncio
async def test_install_model_registration_failure_rolls_back_model(svc: LLamacppService, deps: dict[str, Any], tmp_path: Path) -> None:
    installed = _setup_install_mocks(svc, deps)
    model_id = next(iter(svc.models["default"]))
    deps["endpoint_registry"].register_chat_completion_as_proxy.side_effect = RuntimeError("registry down")

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=(tmp_path / "model.gguf", "model.gguf")),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.llamacpp_service.get_gguf_context_window", new_callable=AsyncMock, return_value=4096),
        patch("server.services.llamacpp_service.get_base_url", return_value="http://localhost:8080"),
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(RuntimeError):
            await promise.wait()

    assert model_id not in installed.models
    assert deps["docker_service"].stop_docker.call_count == 1


@pytest.mark.asyncio
async def test_install_model_registration_failure_skips_rollback_when_already_removed(
    svc: LLamacppService, deps: dict[str, Any], tmp_path: Path
) -> None:
    installed = _setup_install_mocks(svc, deps)
    model_id = next(iter(svc.models["default"]))

    def side_effect(*args: object, **kwargs: object) -> None:
        installed.models.pop(model_id, None)
        raise RuntimeError("registry down")

    deps["endpoint_registry"].register_chat_completion_as_proxy.side_effect = side_effect

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=(tmp_path / "model.gguf", "model.gguf")),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.llamacpp_service.get_gguf_context_window", new_callable=AsyncMock, return_value=4096),
        patch("server.services.llamacpp_service.get_base_url", return_value="http://localhost:8080"),
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(RuntimeError):
            await promise.wait()

    assert model_id not in installed.models


def _status(exists: bool = True, state: str = "running", health: str = "healthy", restart_count: int = 0) -> ContainerStatus:
    return ContainerStatus(exists=exists, state=state, health=health, restart_count=restart_count)


@pytest.mark.asyncio
async def test_reconcile_instance_models_running_healthy_does_not_release(svc: LLamacppService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info(svc)
    installed.models["m1"] = _make_model_installed_info(model_id="m1")
    svc._vram_cache[("default", "m1")] = 4.2  # pyright: ignore[reportPrivateUsage]
    deps["docker_service"].get_container_status = AsyncMock(return_value=_status(state="running", health="healthy"))

    await svc._reconcile_instance_models("default", installed)  # pyright: ignore[reportPrivateUsage]

    assert "m1" in installed.models
    assert ("default", "m1") in svc._vram_cache  # pyright: ignore[reportPrivateUsage]
    deps["endpoint_registry"].unregister_chat_completion.assert_not_called()


@pytest.mark.asyncio
async def test_reconcile_instance_models_recovers_within_threshold_resets_counter(svc: LLamacppService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info(svc)
    installed.models["m1"] = _make_model_installed_info(model_id="m1")
    deps["docker_service"].get_container_status = AsyncMock(return_value=_status(state="exited", health=""))

    await svc._reconcile_instance_models("default", installed)  # pyright: ignore[reportPrivateUsage]
    await svc._reconcile_instance_models("default", installed)  # pyright: ignore[reportPrivateUsage]
    assert svc._crash_poll_state[("default", "m1")] == 2  # pyright: ignore[reportPrivateUsage]

    deps["docker_service"].get_container_status = AsyncMock(return_value=_status(state="running", health="healthy"))
    await svc._reconcile_instance_models("default", installed)  # pyright: ignore[reportPrivateUsage]

    assert ("default", "m1") not in svc._crash_poll_state  # pyright: ignore[reportPrivateUsage]
    assert "m1" in installed.models


@pytest.mark.asyncio
async def test_reconcile_instance_models_releases_after_threshold_bad_polls(svc: LLamacppService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info(svc)
    model_info = _make_model_installed_info(model_id="m1", registration_id="reg-1")
    installed.models["m1"] = model_info
    svc._vram_cache[("default", "m1")] = 4.2  # pyright: ignore[reportPrivateUsage]
    deps["docker_service"].get_container_status = AsyncMock(return_value=_status(exists=False, state="", health=""))
    deps["docker_service"].uninstall_docker = AsyncMock()

    with patch.object(svc, "_save", new_callable=AsyncMock) as mock_save:
        for _ in range(3):
            await svc._reconcile_instance_models("default", installed)  # pyright: ignore[reportPrivateUsage]

    assert "m1" not in installed.models
    assert ("default", "m1") not in svc._vram_cache  # pyright: ignore[reportPrivateUsage]
    assert ("default", "m1") not in svc._crash_poll_state  # pyright: ignore[reportPrivateUsage]
    deps["endpoint_registry"].unregister_chat_completion.assert_called_once_with("m1", "reg-1")
    deps["docker_service"].uninstall_docker.assert_called_once_with(model_info.docker)
    mock_save.assert_awaited_once()


@pytest.mark.asyncio
async def test_reconcile_instance_models_unhealthy_running_container_counts_as_bad(svc: LLamacppService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info(svc)
    installed.models["m1"] = _make_model_installed_info(model_id="m1", registration_id="reg-1")
    deps["docker_service"].get_container_status = AsyncMock(return_value=_status(state="running", health="unhealthy"))
    deps["docker_service"].uninstall_docker = AsyncMock()

    with patch.object(svc, "_save", new_callable=AsyncMock):
        for _ in range(3):
            await svc._reconcile_instance_models("default", installed)  # pyright: ignore[reportPrivateUsage]

    assert "m1" not in installed.models
    deps["endpoint_registry"].unregister_chat_completion.assert_called_once_with("m1", "reg-1")


@pytest.mark.asyncio
async def test_reconcile_instance_models_keeps_bookkeeping_when_teardown_fails_and_retries(
    svc: LLamacppService, deps: dict[str, Any]
) -> None:
    installed = _make_installed_info(svc)
    model_info = _make_model_installed_info(model_id="m1", registration_id="reg-1")
    installed.models["m1"] = model_info
    deps["docker_service"].get_container_status = AsyncMock(return_value=_status(exists=False, state="", health=""))
    deps["docker_service"].uninstall_docker = AsyncMock(side_effect=RuntimeError("docker daemon busy"))

    for _ in range(3):
        await svc._reconcile_instance_models("default", installed)  # pyright: ignore[reportPrivateUsage]

    assert "m1" in installed.models
    deps["endpoint_registry"].unregister_chat_completion.assert_not_called()

    deps["docker_service"].uninstall_docker = AsyncMock()
    with patch.object(svc, "_save", new_callable=AsyncMock) as mock_save:
        await svc._reconcile_instance_models("default", installed)  # pyright: ignore[reportPrivateUsage]

    assert "m1" not in installed.models
    deps["endpoint_registry"].unregister_chat_completion.assert_called_once_with("m1", "reg-1")
    mock_save.assert_awaited_once()


@pytest.mark.asyncio
async def test_reconcile_instance_models_skips_models_currently_installing(svc: LLamacppService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info(svc)
    installed.models["m1"] = _make_model_installed_info(model_id="m1")
    svc._installing.add(("default", "m1"))  # pyright: ignore[reportPrivateUsage]
    deps["docker_service"].get_container_status = AsyncMock(return_value=_status(exists=False, state="", health=""))

    await svc._reconcile_instance_models("default", installed)  # pyright: ignore[reportPrivateUsage]

    deps["docker_service"].get_container_status.assert_not_called()
    assert "m1" in installed.models


@pytest.mark.asyncio
async def test_reconcile_instance_models_skips_model_without_container_name(svc: LLamacppService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info(svc)
    installed.models["m1"] = _make_model_installed_info(model_id="m1")
    svc._crash_poll_state[("default", "m1")] = 2  # pyright: ignore[reportPrivateUsage]
    deps["docker_service"].get_container_status = AsyncMock()

    with patch.object(svc, "_reconcile_container_name", return_value=None):
        await svc._reconcile_instance_models("default", installed)  # pyright: ignore[reportPrivateUsage]

    deps["docker_service"].get_container_status.assert_not_called()
    assert ("default", "m1") not in svc._crash_poll_state  # pyright: ignore[reportPrivateUsage]
    assert "m1" in installed.models


@pytest.mark.asyncio
async def test_stop_instance_cancels_reconciliation_task(svc: LLamacppService) -> None:
    svc.instances_info["default"].installed = _make_installed_info(svc)
    svc._start_reconciliation_task("default")  # pyright: ignore[reportPrivateUsage]
    assert "default" in svc._reconciliation_tasks  # pyright: ignore[reportPrivateUsage]

    await svc.stop_instance("default")

    assert "default" not in svc._reconciliation_tasks  # pyright: ignore[reportPrivateUsage]


@pytest.fixture(autouse=True)
def _reset_catalog_refresh_guard() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    huggingface_refresh_guard.release()
    yield
    huggingface_refresh_guard.release()


def test_get_catalog_refresh_unavailable_reason_none_when_supported(svc: LLamacppService) -> None:
    with patch("server.services.llamacpp_service.model_catalog_refresh.catalog_refresh_supported", True):
        assert svc.get_catalog_refresh_unavailable_reason() is None


def test_get_catalog_refresh_unavailable_reason_set_when_unsupported(svc: LLamacppService) -> None:
    with patch("server.services.llamacpp_service.model_catalog_refresh.catalog_refresh_supported", False):
        assert svc.get_catalog_refresh_unavailable_reason() is not None


@pytest.mark.asyncio
async def test_refresh_catalog_writes_registry_and_reloads_const(svc: LLamacppService, tmp_path: Path) -> None:
    (tmp_path / "static").mkdir()
    (tmp_path / "static" / "llamacpp-min.json").write_text(
        json.dumps({"llms": [{"name": "old-model", "url": "https://old", "size": "1GB", "jinja": True}]}), encoding="utf-8"
    )

    fetched = [LlamacppRegistryEntry(name="new-model", url="https://old", size="2GB")]

    original_models = dict(_const.models)
    try:
        with (
            patch("server.services.llamacpp_service.model_catalog_refresh.catalog_refresh_supported", True),
            patch("server.services.llamacpp_service.get_main_dir", return_value=tmp_path),
            patch("server.services.llamacpp_service.fetch_llamacpp_entries", new=AsyncMock(return_value=fetched)),
        ):
            promise = await svc.refresh_catalog()
            result = await promise.wait()

        assert result.added == 1
        assert "new-model" in _const.models
        assert "old-model" not in _const.models
        written = json.loads((tmp_path / "static" / "llamacpp-min.json").read_text(encoding="utf-8"))
        # jinja flag carried forward because the URL matches the previous entry
        assert written["llms"] == [
            {"name": "new-model", "url": "https://old", "size": "2GB", "jinja": True, "supports_vision": None, "has_mmproj": None}
        ]
        assert set(svc.models["default"].keys()) == set(_const.models.keys())
    finally:
        _const.models = original_models


@pytest.mark.asyncio
async def test_refresh_catalog_raises_when_unsupported(svc: LLamacppService) -> None:
    with (
        patch("server.services.llamacpp_service.model_catalog_refresh.catalog_refresh_supported", False),
        pytest.raises(HTTPException) as exc_info,
    ):
        await svc.refresh_catalog()
    assert exc_info.value.status_code == 405


@pytest.mark.asyncio
async def test_refresh_catalog_rejects_concurrent_refresh(svc: LLamacppService) -> None:
    huggingface_refresh_guard.acquire("vllm")
    try:
        with (
            patch("server.services.llamacpp_service.model_catalog_refresh.catalog_refresh_supported", True),
            pytest.raises(HTTPException) as exc_info,
        ):
            await svc.refresh_catalog()
        assert exc_info.value.status_code == 429
    finally:
        huggingface_refresh_guard.release()


@pytest.mark.asyncio
async def test_refresh_catalog_leaves_catalog_unchanged_on_degraded_fetch(svc: LLamacppService, tmp_path: Path) -> None:
    (tmp_path / "static").mkdir()
    original_content = json.dumps({"llms": [{"name": "old-model", "url": "https://old", "size": "1GB"}]})
    (tmp_path / "static" / "llamacpp-min.json").write_text(original_content, encoding="utf-8")

    original_models = dict(_const.models)
    try:
        with (
            patch("server.services.llamacpp_service.model_catalog_refresh.catalog_refresh_supported", True),
            patch("server.services.llamacpp_service.get_main_dir", return_value=tmp_path),
            patch(
                "server.services.llamacpp_service.fetch_llamacpp_entries",
                new=AsyncMock(side_effect=DegradedFetchError("too many failures")),
            ),
        ):
            promise = await svc.refresh_catalog()
            with pytest.raises(HTTPException) as exc_info:
                await promise.wait()
            assert exc_info.value.status_code == 502

        assert (tmp_path / "static" / "llamacpp-min.json").read_text(encoding="utf-8") == original_content
        assert _const.models == original_models
    finally:
        _const.models = original_models


@pytest.mark.asyncio
async def test_refresh_catalog_raises_502_when_catalog_shrinks_too_much(svc: LLamacppService, tmp_path: Path) -> None:
    (tmp_path / "static").mkdir()
    original_content = json.dumps({"llms": [{"name": f"old-model-{i}", "url": "https://old", "size": "1GB"} for i in range(10)]})
    (tmp_path / "static" / "llamacpp-min.json").write_text(original_content, encoding="utf-8")

    fetched = [LlamacppRegistryEntry(name="new-model", url="https://new", size="2GB")]

    original_models = dict(_const.models)
    try:
        with (
            patch("server.services.llamacpp_service.model_catalog_refresh.catalog_refresh_supported", True),
            patch("server.services.llamacpp_service.get_main_dir", return_value=tmp_path),
            patch("server.services.llamacpp_service.fetch_llamacpp_entries", new=AsyncMock(return_value=fetched)),
        ):
            promise = await svc.refresh_catalog()
            with pytest.raises(HTTPException) as exc_info:
                await promise.wait()
            assert exc_info.value.status_code == 502

        assert (tmp_path / "static" / "llamacpp-min.json").read_text(encoding="utf-8") == original_content
        assert _const.models == original_models
    finally:
        _const.models = original_models


@pytest.mark.asyncio
async def test_refresh_catalog_restores_installed_model_dropped_from_catalog(svc: LLamacppService, tmp_path: Path) -> None:
    (tmp_path / "static").mkdir()
    (tmp_path / "static" / "llamacpp-min.json").write_text(
        json.dumps({"llms": [{"name": "old-model", "url": "https://old", "size": "1GB"}]}), encoding="utf-8"
    )

    fetched = [LlamacppRegistryEntry(name="new-model", url="https://new", size="2GB")]

    installed = _make_installed_info(svc)
    installed.models["old-model"] = _make_model_installed_info("old-model")
    svc.instances_info["default"].installed = installed
    svc.instances_info["default"].config = InstanceConfig(
        models=[
            ModelConfig(
                model_id="old-model", options=InstallModelIn(spec={}), definition={"url": "https://old", "size": "1GB", "jinja": False}
            )
        ]
    )

    original_models = dict(_const.models)
    try:
        with (
            patch("server.services.llamacpp_service.model_catalog_refresh.catalog_refresh_supported", True),
            patch("server.services.llamacpp_service.get_main_dir", return_value=tmp_path),
            patch("server.services.llamacpp_service.fetch_llamacpp_entries", new=AsyncMock(return_value=fetched)),
        ):
            promise = await svc.refresh_catalog()
            await promise.wait()

        assert "old-model" not in _const.models
        assert svc.models["default"]["old-model"] == LlamacppModel(url="https://old", size="1GB", jinja=False)
    finally:
        _const.models = original_models


@pytest.mark.asyncio
async def test_refresh_catalog_drops_installed_model_with_no_persisted_definition(svc: LLamacppService, tmp_path: Path) -> None:
    (tmp_path / "static").mkdir()
    (tmp_path / "static" / "llamacpp-min.json").write_text(
        json.dumps({"llms": [{"name": "old-model", "url": "https://old", "size": "1GB"}]}), encoding="utf-8"
    )

    fetched = [LlamacppRegistryEntry(name="new-model", url="https://new", size="2GB")]

    installed = _make_installed_info(svc)
    installed.models["old-model"] = _make_model_installed_info("old-model")
    svc.instances_info["default"].installed = installed

    original_models = dict(_const.models)
    try:
        with (
            patch("server.services.llamacpp_service.model_catalog_refresh.catalog_refresh_supported", True),
            patch("server.services.llamacpp_service.get_main_dir", return_value=tmp_path),
            patch("server.services.llamacpp_service.fetch_llamacpp_entries", new=AsyncMock(return_value=fetched)),
        ):
            promise = await svc.refresh_catalog()
            await promise.wait()

        assert "old-model" not in svc.models["default"]
    finally:
        _const.models = original_models


@pytest.mark.asyncio
async def test_sync_models_triggers_refresh_without_awaiting_completion(svc: LLamacppService) -> None:
    with patch.object(svc, "refresh_catalog", new=AsyncMock()) as mock_refresh:
        await svc.sync_models("default")
    mock_refresh.assert_awaited_once()


@pytest.mark.asyncio
async def test_sync_models_suppresses_unsupported_error(svc: LLamacppService) -> None:
    with patch("server.services.llamacpp_service.model_catalog_refresh.catalog_refresh_supported", False):
        await svc.sync_models("default")  # must not raise


@pytest.mark.asyncio
async def test_sync_models_logs_warning_when_background_refresh_fails(svc: LLamacppService) -> None:
    class _FailingPromise:
        async def wait(self) -> None:
            raise RuntimeError("boom")

    with (
        patch.object(svc, "refresh_catalog", new=AsyncMock(return_value=_FailingPromise())),
        patch("server.services.llamacpp_service.logger") as mock_logger,
    ):
        await svc.sync_models("default")
        await asyncio.sleep(0)  # let the fire-and-forget task run

    mock_logger.warning.assert_called_once()


@pytest.mark.parametrize(
    ("hardware", "installed", "expected"),
    [
        ("GPU", "server-cuda12-b5966", "server-cuda12-b11580"),
        ("vulkan", "server-vulkan-b5966", "server-vulkan-b11580"),
        ("cpu", "server-intel-b5966", "server-intel-b11580"),
        ("cpu", "server-b5966", "server-b11580"),
    ],
)
def test_newest_tag_in_the_real_ghcr_list(svc: LLamacppService, hardware: str, installed: str, expected: str) -> None:
    tags = svc.filter_docker_tags(real_tags("llamacpp"), hardware)
    assert newest_docker_tag(tags, installed) == expected


@pytest.mark.asyncio
async def test_download_model_emits_byte_counts_in_progress_data(svc: LLamacppService, deps: dict[str, Any], tmp_path: Path) -> None:
    model = LlamacppModel(url="https://example.com/model.gguf", size="100MB")
    stream = MagicMock()

    async def mock_download(*args: object):  # type: ignore[misc]
        yield DownloadedPacket(downloaded_bytes_size=1_000_000)
        yield SuccessDownloadPacket(local_path=tmp_path / "model.gguf", filename="model.gguf")

    deps["model_downloader"].download = mock_download

    with patch.object(svc, "_get_working_dir", return_value=tmp_path):  # pyright: ignore[reportPrivateUsage]
        await svc._download_model(stream, model)  # pyright: ignore[reportPrivateUsage]

    assert stream.emit.call_args_list[1].args[0]["data"] == {"completed": 1_000_000, "total": 100_000_000}
    assert stream.emit.call_args_list[-1].args[0]["data"] == {"local_model_path": str(tmp_path / "model.gguf"), "filename": "model.gguf"}


@pytest.mark.asyncio
async def test_download_model_omits_path_from_final_chunk_when_downloader_returns_none(
    svc: LLamacppService, deps: dict[str, Any], tmp_path: Path
) -> None:
    model = LlamacppModel(url="https://example.com/model.gguf", size="100MB")
    stream = MagicMock()

    async def mock_download(*args: object):  # type: ignore[misc]
        yield DownloadedPacket(downloaded_bytes_size=1_000_000)

    deps["model_downloader"].download = mock_download

    with patch.object(svc, "_get_working_dir", return_value=tmp_path):  # pyright: ignore[reportPrivateUsage]
        result = await svc._download_model(stream, model)  # pyright: ignore[reportPrivateUsage]

    assert result == (None, "")
    assert stream.emit.call_args_list[-1].args[0]["data"] == {}


@pytest.mark.asyncio
async def test_download_model_or_set_progress_returns_no_path_when_watched_download_fails_after_byte_chunks(
    svc: LLamacppService,
) -> None:
    existing_stream: Stream[StreamChunk] = Stream()  # type: ignore[type-arg]
    existing_stream.emit(StreamChunkProgress(type="progress", stage="download", value=0.1, data={"completed": 100, "total": 1000}))
    existing_stream.close()
    model = LlamacppModel(url="https://example.com/model.gguf", size="1GB")
    svc.models_download_progress["my-model"] = existing_stream  # type: ignore[assignment]
    output_stream = MagicMock()

    result = await svc._download_model_or_set_progress(output_stream, model, "my-model")  # pyright: ignore[reportPrivateUsage]

    assert result == (None, "")


@pytest.mark.asyncio
async def test_download_model_ends_with_progress_one_and_the_downloaded_file(
    svc: LLamacppService, deps: dict[str, Any], tmp_path: Path
) -> None:
    model = LlamacppModel(url="https://example.com/model.gguf", size="100MB")
    stream = MagicMock()

    async def mock_download(*args: object):  # type: ignore[misc]
        yield DownloadedPacket(downloaded_bytes_size=90_000_000)
        yield SuccessDownloadPacket(local_path=tmp_path / "model.gguf", filename="model.gguf")

    deps["model_downloader"].download = mock_download

    with patch.object(svc, "_get_working_dir", return_value=tmp_path):  # pyright: ignore[reportPrivateUsage]
        await svc._download_model(stream, model)  # pyright: ignore[reportPrivateUsage]

    values = [emit_call.args[0]["value"] for emit_call in stream.emit.call_args_list]
    assert values == pytest.approx([0, 0.9, 1])
