# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from fastapi import HTTPException

from server.models.models import InstallModelIn, ListModelsFilters, UninstallModelIn
from server.models.services import InstallServiceIn, UninstallServiceIn
from server.services.base2_service import CustomModel, Instance, InstanceConfig
from server.services.llamacpp_service import (
    DownloadedInfo,
    InstalledInfo,
    LlamacppModel,
    LLamacppOptions,
    LLamacppService,
    ModelInstalledInfo,
    _const,  # pyright: ignore[reportPrivateUsage]
)
from server.utils.core import DownloadedPacket, PreDownloadPacket, Stream, StreamChunk, StreamChunkProgress, SuccessDownloadPacket
from server.utils.hardware import CpuInfo, GpuInfo, IntelGpuInfo, NvidiaGpuInfo


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
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=8080)
    deps["docker_service"].get_docker_subnet.return_value = None
    deps["docker_service"].get_docker_container_name.return_value = "container"
    deps["docker_service"].get_container_host.return_value = "localhost"
    deps["docker_service"].get_container_port.return_value = 8080
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


def test_get_model_spec_has_alias_field(svc: LLamacppService) -> None:
    spec = svc.get_model_spec()

    field_names = [f.name for f in spec.fields]
    assert "alias" in field_names


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
    config = svc._generate_instance_config(None, None)  # pyright: ignore[reportPrivateUsage]

    assert config.options is None
    assert config.models == []


def test_generate_instance_config_with_info(svc: LLamacppService) -> None:
    info = _make_installed_info(svc)
    info.models["m1"] = _make_model_installed_info("m1")

    config = svc._generate_instance_config(info, None)  # pyright: ignore[reportPrivateUsage]

    assert config.options == info.options
    assert len(config.models or []) == 1


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
async def test_install_model_uses_alias_as_registered_name(svc: LLamacppService, deps: dict[str, Any], tmp_path: Path) -> None:
    installed = _setup_install_mocks(svc, deps)
    model_id = next(iter(svc.models["default"]))
    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=(tmp_path / "model.gguf", "model.gguf")),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.llamacpp_service.get_gguf_context_window", new_callable=AsyncMock, return_value=4096),
        patch("server.services.llamacpp_service.get_base_url", return_value="http://localhost:8080"),
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn(spec={"alias": "my-alias"}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert installed.models[model_id].registered_name == "my-alias"


@pytest.mark.asyncio
async def test_install_model_appends_ctx_size_when_max_model_length_set(svc: LLamacppService, deps: dict[str, Any], tmp_path: Path) -> None:
    _setup_install_mocks(svc, deps)
    model_id = next(iter(svc.models["default"]))
    captured: list[object] = []

    async def capture_docker(opts: object) -> int:
        captured.append(opts)
        return 8080

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
    jinja_model_id = "speakleash/Bielik-11B-v2.5-Instruct"
    assert svc.models["default"][jinja_model_id].jinja is True
    captured: list[object] = []

    async def capture_docker(opts: object) -> int:
        captured.append(opts)
        return 8080

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
