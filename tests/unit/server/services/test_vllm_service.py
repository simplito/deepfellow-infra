# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from fastapi import HTTPException

from server.models.models import InstallModelIn, ListModelsFilters, UninstallModelIn
from server.models.services import InstallServiceIn, UninstallServiceIn
from server.services.base2_service import CustomModel, Instance, InstanceConfig
from server.services.vllm_service import (
    DownloadedInfo,
    InstalledInfo,
    ModelInstalledInfo,
    VllmModel,
    VllmModelOptions,
    VllmOptions,
    VllmService,
    _const,  # pyright: ignore[reportPrivateUsage]
)
from server.utils.core import DownloadedPacket, PreDownloadPacket, Stream, StreamChunk, StreamChunkProgress, SuccessDownloadPacket
from server.utils.hardware import NvidiaGpuInfo


@pytest.fixture
def deps() -> dict[str, Any]:
    hw = MagicMock()
    hw.cpu.avx512 = True
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
def svc(deps: dict[str, Any]) -> VllmService:
    return VllmService(**deps)


def _make_installed_info(hardware: str | bool | None = False) -> InstalledInfo:
    return InstalledInfo(
        models={},
        options=InstallServiceIn(spec={}),
        parsed_options=VllmOptions(hardware=hardware),
    )


def _make_model_installed_info(
    model_id: str = "test-model",
    registration_id: str = "reg-1",
    gpu_memory_utilization: float | None = None,
    model_type: str = "llm",
) -> ModelInstalledInfo:
    docker = MagicMock()
    docker.name = f"df-vllm-{model_id}"
    return ModelInstalledInfo(
        id=model_id,
        registered_name=model_id,
        options=InstallModelIn(spec={}),
        docker=docker,
        container_host="localhost",
        container_port=8000,
        docker_exposed_port=8000,
        registration_id=registration_id,
        model_path=Path("/tmp/model"),
        base_url="http://localhost:8000",
        gpu_memory_utilization=gpu_memory_utilization,
        model_type=model_type,  # type: ignore[arg-type]
    )


def _setup_install_mocks(svc: VllmService, deps: dict[str, Any], hardware: str | bool | None = False) -> InstalledInfo:
    installed = _make_installed_info(hardware=hardware)
    svc.instances_info["default"].installed = installed
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=8000)
    deps["docker_service"].get_docker_subnet.return_value = None
    deps["docker_service"].get_docker_container_name.return_value = "container"
    deps["docker_service"].get_container_host.return_value = "localhost"
    deps["docker_service"].get_container_port.return_value = 8000
    return installed


def test_get_type(svc: VllmService) -> None:
    assert svc.get_type() == "vllm"


def test_get_description_not_empty(svc: VllmService) -> None:
    assert svc.get_description()


def test_default_models_loaded(svc: VllmService) -> None:
    assert "default" in svc.models
    assert len(svc.models["default"]) > 0


def test_get_size_cpu_only(deps: dict[str, Any]) -> None:
    deps["hardware"].cpu.avx512 = True
    deps["hardware"].gpus = []
    svc = VllmService(**deps)

    sizes = svc.get_size()

    assert "cpu" in sizes
    assert "gpu" not in sizes


def test_get_size_with_gpu(deps: dict[str, Any]) -> None:
    nvidia = MagicMock(spec=NvidiaGpuInfo)
    deps["hardware"].gpus = [nvidia]
    svc = VllmService(**deps)

    sizes = svc.get_size()

    assert "gpu" in sizes


def test_get_spec_has_hardware_field(svc: VllmService) -> None:
    spec = svc.get_spec()

    field_names = [f.name for f in spec.fields]
    assert "hardware" in field_names


def test_const_has_cpu_and_gpu_images() -> None:
    assert "cpu" in _const.images
    assert "gpu" in _const.images
    assert _const.images["cpu"].name
    assert _const.images["gpu"].name


def test_get_model_spec_baseline_fields(svc: VllmService) -> None:
    installed = _make_installed_info(hardware=False)
    svc.instances_info["default"].installed = installed

    spec = svc.get_model_spec("default", "llm")

    field_names = [f.name for f in spec.fields]
    for expected in ("alias", "max_model_length", "quantization", "extra_args", "extra_envs"):
        assert expected in field_names


def test_get_model_spec_adds_gpu_memory_utilization_for_gpu_hardware(svc: VllmService) -> None:
    svc.instances_info["default"].config.options = InstallServiceIn(spec={"hardware": True})

    spec = svc.get_model_spec("default", "llm")

    field_names = [f.name for f in spec.fields]
    assert "gpu_memory_utilization" in field_names


def test_get_model_spec_omits_gpu_memory_utilization_for_cpu_hardware(svc: VllmService) -> None:
    spec = svc.get_model_spec("default", "llm")

    field_names = [f.name for f in spec.fields]
    assert "gpu_memory_utilization" not in field_names


def test_get_model_spec_reranker_default_extra_args(svc: VllmService) -> None:
    spec = svc.get_model_spec("default", "reranker")

    extra_args_field = next(f for f in spec.fields if f.name == "extra_args")
    assert extra_args_field.default is not None
    assert "--trust-remote-code" in (extra_args_field.default or "")


def test_get_custom_model_spec_not_none(svc: VllmService) -> None:
    result = svc.get_custom_model_spec()

    assert result is not None
    field_names = [f.name for f in result.fields]
    assert "id" in field_names
    assert "hf_id" in field_names
    assert "size" in field_names


def test_model_installed_info_get_info() -> None:
    model = _make_model_installed_info(registration_id="reg-42")

    info = model.get_info()

    assert info.registration_id == "reg-42"


def test_get_installed_info_returns_spec_when_installed(svc: VllmService) -> None:
    installed = _make_installed_info()
    installed.options.spec["hardware"] = False
    svc.instances_info["default"].installed = installed

    result = svc.get_installed_info("default")

    assert result == installed.options.spec


def test_get_installed_info_delegates_when_not_installed(svc: VllmService) -> None:
    svc.instances_info["default"].installed = None

    with patch.object(svc, "_get_service_installed_info", return_value=False) as mock:  # pyright: ignore[reportPrivateUsage]
        result = svc.get_installed_info("default")

    assert mock.call_count == 1
    assert mock.call_args == call("default")
    assert result is False


def test_generate_instance_config_none_info(svc: VllmService) -> None:
    config = svc._generate_instance_config(None, None)  # pyright: ignore[reportPrivateUsage]

    assert config.options is None
    assert config.models == []


def test_generate_instance_config_with_info(svc: VllmService) -> None:
    info = _make_installed_info()
    info.models["m1"] = _make_model_installed_info("m1")

    config = svc._generate_instance_config(info, None)  # pyright: ignore[reportPrivateUsage]

    assert config.options == info.options
    assert len(config.models or []) == 1


def test_load_download_info(svc: VllmService) -> None:
    result = svc._load_download_info({"model_path": "/tmp/x"})  # pyright: ignore[reportPrivateUsage]

    assert isinstance(result, DownloadedInfo)
    assert result.model_path == "/tmp/x"


def test_is_given_hardware_support_gpu_raises_400_no_avx512_no_gpu(deps: dict[str, Any]) -> None:
    deps["hardware"].cpu.avx512 = False
    deps["hardware"].gpus = []
    svc = VllmService(**deps)

    with pytest.raises(HTTPException) as exc_info:
        svc.is_given_hardware_support_gpu(None)

    assert exc_info.value.status_code == 400


def test_get_specified_hardware_parts_raises_400_no_avx512_no_gpu(deps: dict[str, Any]) -> None:
    deps["hardware"].cpu.avx512 = False
    deps["hardware"].gpus = []
    svc = VllmService(**deps)

    with pytest.raises(HTTPException) as exc_info:
        svc.get_specified_hardware_parts(None)

    assert exc_info.value.status_code == 400


def test_is_given_hardware_support_gpu_with_gpu_does_not_raise(deps: dict[str, Any]) -> None:
    nvidia = MagicMock(spec=NvidiaGpuInfo)
    deps["hardware"].gpus = [nvidia]
    svc = VllmService(**deps)

    result = svc.is_given_hardware_support_gpu(True)

    assert result is True


def test_get_docker_compose_file_path_raises_400_no_model_id(svc: VllmService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    with pytest.raises(HTTPException) as exc_info:
        svc.get_docker_compose_file_path("default", None)

    assert exc_info.value.status_code == 400


def test_get_docker_compose_file_path_raises_400_model_not_installed(svc: VllmService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    with pytest.raises(HTTPException) as exc_info:
        svc.get_docker_compose_file_path("default", "some-model")

    assert exc_info.value.status_code == 400


def test_get_docker_compose_file_path_returns_path_for_installed_model(svc: VllmService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    installed.models["test-model"] = _make_model_installed_info("test-model")
    svc.instances_info["default"].installed = installed
    expected = Path("/some/path/docker-compose.yml")
    deps["docker_service"].get_docker_compose_file_path.return_value = expected

    result = svc.get_docker_compose_file_path("default", "test-model")

    assert result == expected


def test_add_custom_model_registers_entry(svc: VllmService) -> None:
    custom = CustomModel(id="c-1", data={"id": "my-model", "hf_id": "google/gemma-3-270m-it", "size": "1GB"})

    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    assert "my-model" in svc.models["default"]
    assert svc.models["default"]["my-model"].custom == "c-1"
    assert svc.models["default"]["my-model"].hf_id == "google/gemma-3-270m-it"


def test_add_custom_model_duplicate_raises_400(svc: VllmService) -> None:
    custom = CustomModel(id="c-2", data={"id": "dup", "hf_id": "google/model", "size": "1GB"})

    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    with pytest.raises(HTTPException) as exc_info:
        svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]
    assert exc_info.value.status_code == 400


def test_add_custom_model_creates_dict_for_new_instance(svc: VllmService) -> None:
    svc.instances_info["extra"] = Instance(None, None, {}, InstanceConfig())
    custom = CustomModel(id="c-3", data={"id": "new-model", "hf_id": "google/model", "size": "1GB"})

    svc._add_custom_model("extra", custom)  # pyright: ignore[reportPrivateUsage]

    assert "new-model" in svc.models["extra"]


def test_remove_custom_model_deletes_entry(svc: VllmService) -> None:
    custom = CustomModel(id="c-4", data={"id": "to-remove", "hf_id": "google/model", "size": "1GB"})
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]
    svc.instances_info["default"].installed = _make_installed_info()

    svc._remove_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    assert "to-remove" not in svc.models["default"]


def test_remove_custom_model_raises_400_when_in_use(svc: VllmService) -> None:
    custom = CustomModel(id="c-5", data={"id": "in-use", "hf_id": "google/model", "size": "1GB"})
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]
    installed = _make_installed_info()
    installed.models["in-use"] = _make_model_installed_info("in-use")
    svc.instances_info["default"].installed = installed

    with pytest.raises(HTTPException) as exc_info:
        svc._remove_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_list_models_raises_404_for_unknown_instance(svc: VllmService) -> None:
    with pytest.raises(HTTPException) as exc_info:
        await svc.list_models("nonexistent", ListModelsFilters())

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_list_models_returns_all_models_no_filter(svc: VllmService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    result = await svc.list_models("default", ListModelsFilters())

    assert len(result.list) == len(svc.models["default"])


@pytest.mark.asyncio
async def test_list_models_filters_installed_only(svc: VllmService) -> None:
    installed = _make_installed_info()
    model_id = next(iter(svc.models["default"]))
    installed.models[model_id] = _make_model_installed_info(model_id)
    svc.instances_info["default"].installed = installed

    result = await svc.list_models("default", ListModelsFilters(installed=True))

    assert len(result.list) >= 1
    assert all(bool(m.installed) for m in result.list)


@pytest.mark.asyncio
async def test_get_model_raises_400_for_unknown_model_id(svc: VllmService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    with pytest.raises(HTTPException) as exc_info:
        await svc.get_model("default", "nonexistent-model")

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_get_model_returns_correct_model(svc: VllmService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    model_id = next(iter(svc.models["default"]))

    result = await svc.get_model("default", model_id)

    assert result.id == model_id


@pytest.mark.asyncio
async def test_download_model_emits_initial_and_final_progress(svc: VllmService, deps: dict[str, Any], tmp_path: Path) -> None:
    model = VllmModel(hf_id="google/test-model", size="1GB")
    stream = MagicMock()
    local_path = tmp_path / "model-files"
    local_path.mkdir()

    async def mock_download(*args: object):  # type: ignore[misc]
        yield SuccessDownloadPacket(local_path=local_path, filename="model-files")

    deps["model_downloader"].download = mock_download

    await svc._download_model(stream, "google/test-model", model, tmp_path)  # pyright: ignore[reportPrivateUsage]

    assert stream.emit.call_count >= 2


@pytest.mark.asyncio
async def test_download_model_updates_progress_on_downloaded_packet(svc: VllmService, deps: dict[str, Any], tmp_path: Path) -> None:
    model = VllmModel(hf_id="google/test-model", size="100MB")
    stream = MagicMock()
    local_path = tmp_path / "model-files"
    local_path.mkdir()

    async def mock_download(*args: object):  # type: ignore[misc]
        yield DownloadedPacket(downloaded_bytes_size=1024 * 1024)
        yield SuccessDownloadPacket(local_path=local_path, filename="model-files")

    deps["model_downloader"].download = mock_download

    await svc._download_model(stream, "google/test-model", model, tmp_path)  # pyright: ignore[reportPrivateUsage]

    progress_calls = [call for call in stream.emit.call_args_list if hasattr(call.args[0], "get") and call.args[0].get("value", 0) > 0]
    assert len(progress_calls) >= 1


@pytest.mark.asyncio
async def test_download_model_or_set_progress_starts_new_download(svc: VllmService, tmp_path: Path) -> None:
    stream = MagicMock()
    model_id = "google/test-model"
    model = VllmModel(hf_id=model_id, size="1GB")
    svc.models["default"][model_id] = model

    with patch.object(  # pyright: ignore[reportPrivateUsage]
        svc, "_download_model", new_callable=AsyncMock, return_value=tmp_path / "model"
    ) as mock_dl:
        await svc._download_model_or_set_progress(stream, model_id, model, tmp_path)  # pyright: ignore[reportPrivateUsage]

    assert mock_dl.call_count == 1
    assert model_id not in svc.models_download_progress


@pytest.mark.asyncio
async def test_download_model_or_set_progress_cleans_up_on_failure(svc: VllmService, tmp_path: Path) -> None:
    stream = MagicMock()
    model_id = "google/test-model"
    model = VllmModel(hf_id=model_id, size="1GB")
    svc.models["default"][model_id] = model

    with (
        patch.object(svc, "_download_model", new_callable=AsyncMock, side_effect=HTTPException(400, "boom")),  # pyright: ignore[reportPrivateUsage]
        pytest.raises(HTTPException),
    ):
        await svc._download_model_or_set_progress(stream, model_id, model, tmp_path)  # pyright: ignore[reportPrivateUsage]

    assert model_id not in svc.models_download_progress


@pytest.mark.asyncio
async def test_download_model_or_set_progress_retries_download_after_failure(svc: VllmService, tmp_path: Path) -> None:
    stream1 = MagicMock()
    stream2 = MagicMock()
    model_id = "google/test-model"
    model = VllmModel(hf_id=model_id, size="1GB")
    svc.models["default"][model_id] = model

    mock_dl = AsyncMock(side_effect=[HTTPException(400, "boom"), tmp_path / "model"])
    with patch.object(svc, "_download_model", mock_dl):  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(HTTPException):
            await svc._download_model_or_set_progress(stream1, model_id, model, tmp_path)  # pyright: ignore[reportPrivateUsage]
        await svc._download_model_or_set_progress(stream2, model_id, model, tmp_path)  # pyright: ignore[reportPrivateUsage]

    assert mock_dl.call_count == 2


@pytest.mark.asyncio
async def test_download_model_or_set_progress_forwards_existing_stream(svc: VllmService, tmp_path: Path) -> None:
    existing_stream: Stream[StreamChunk] = Stream()  # type: ignore[type-arg]
    chunk = StreamChunkProgress(type="progress", stage="download", value=0.5, data={"local_model_path": str(tmp_path / "model")})
    existing_stream.emit(chunk)
    existing_stream.close()
    model_id = "google/test-model"
    model = VllmModel(hf_id=model_id, size="1GB")
    svc.models_download_progress[model_id] = existing_stream  # type: ignore[assignment]
    output_stream = MagicMock()

    await svc._download_model_or_set_progress(output_stream, model_id, model, tmp_path)  # pyright: ignore[reportPrivateUsage]

    assert output_stream.emit.call_count == 1
    assert output_stream.emit.call_args == call(chunk)


@pytest.mark.asyncio
async def test_get_gpu_memory_utilization_increments_total(svc: VllmService) -> None:
    svc.gpu_memory_utilization = 0.0
    model = VllmModel(hf_id="google/test", size="1GB")
    opts = VllmModelOptions(gpu_memory_utilization=0.5)

    result = await svc._get_gpu_memory_utilization(opts, model)  # pyright: ignore[reportPrivateUsage]

    assert result == 0.5
    assert svc.gpu_memory_utilization == 0.5


@pytest.mark.asyncio
async def test_get_gpu_memory_utilization_raises_422_when_sum_exceeds_1(svc: VllmService) -> None:
    svc.gpu_memory_utilization = 0.8
    model = VllmModel(hf_id="google/test", size="1GB")
    opts = VllmModelOptions(gpu_memory_utilization=0.5)

    with pytest.raises(HTTPException) as exc_info:
        await svc._get_gpu_memory_utilization(opts, model)  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 422


@pytest.mark.asyncio
async def test_get_quantization_returns_valid_value(svc: VllmService) -> None:
    model = VllmModel(hf_id="google/test", size="1GB")
    opts = VllmModelOptions(quantization="fp8")

    result = await svc._get_quantization(opts, model)  # pyright: ignore[reportPrivateUsage]

    assert result == "fp8"


@pytest.mark.asyncio
async def test_get_quantization_raises_422_on_invalid_characters(svc: VllmService) -> None:
    model = VllmModel(hf_id="google/test", size="1GB")
    opts = VllmModelOptions(quantization=None)
    opts.quantization = "fp8!@#"  # bypass pydantic

    with pytest.raises(HTTPException) as exc_info:
        await svc._get_quantization(opts, model)  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 422


@pytest.mark.asyncio
async def test_get_quantization_returns_none_when_not_set(svc: VllmService) -> None:
    model = VllmModel(hf_id="google/test", size="1GB")
    opts = VllmModelOptions()

    result = await svc._get_quantization(opts, model)  # pyright: ignore[reportPrivateUsage]

    assert result is None


@pytest.mark.asyncio
async def test_get_max_model_length_returns_value_for_integer_float(svc: VllmService) -> None:
    model = VllmModel(hf_id="google/test", size="1GB")
    opts = VllmModelOptions.model_construct(max_model_length=4096.0)

    result = await svc._get_max_model_length(opts, model)  # pyright: ignore[reportPrivateUsage]

    assert result == 4096.0


def test_register_model_endpoint_llm_calls_chat_completion_proxy(svc: VllmService, deps: dict[str, Any]) -> None:
    model = VllmModel(hf_id="google/test", size="1GB", model_type="llm")
    model_info = _make_model_installed_info("google/test", model_type="llm")

    svc._register_model_endpoint(  # pyright: ignore[reportPrivateUsage]
        model_info=model_info,
        model=model,
        registered_name="google/test",
        model_id="google/test",
        context_window=4096,
        max_context_window=4096,
    )

    assert deps["endpoint_registry"].register_chat_completion_as_proxy.call_count == 1
    assert deps["endpoint_registry"].register_rerank_as_proxy.call_count == 0


def test_register_model_endpoint_reranker_calls_rerank_proxy(svc: VllmService, deps: dict[str, Any]) -> None:
    model = VllmModel(hf_id="google/reranker", size="1GB", model_type="reranker")
    model_info = _make_model_installed_info("google/reranker", model_type="reranker")

    svc._register_model_endpoint(  # pyright: ignore[reportPrivateUsage]
        model_info=model_info,
        model=model,
        registered_name="google/reranker",
        model_id="google/reranker",
        context_window=None,
        max_context_window=None,
    )

    assert deps["endpoint_registry"].register_rerank_as_proxy.call_count == 1
    assert deps["endpoint_registry"].register_chat_completion_as_proxy.call_count == 0


def test_get_image_true_returns_gpu_image(svc: VllmService) -> None:
    image = svc._get_image(True)  # pyright: ignore[reportPrivateUsage]

    assert image == _const.images["gpu"]


def test_get_image_false_returns_cpu_image(svc: VllmService) -> None:
    image = svc._get_image(False)  # pyright: ignore[reportPrivateUsage]

    assert image == _const.images["cpu"]


@pytest.mark.asyncio
async def test_install_model_returns_already_installed(svc: VllmService) -> None:
    installed = _make_installed_info()
    model_id = next(iter(svc.models["default"]))
    installed.models[model_id] = _make_model_installed_info(model_id)
    svc.instances_info["default"].installed = installed

    promise = await svc._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]

    result = await promise.wait()
    assert result.status == "OK"
    assert "Already installed" in result.details


@pytest.mark.asyncio
async def test_install_model_raises_400_for_unknown_model(svc: VllmService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    with pytest.raises(HTTPException) as exc_info:
        await svc._install_model("default", "nonexistent-model", InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_install_model_calls_docker_install(svc: VllmService, deps: dict[str, Any], tmp_path: Path) -> None:
    _setup_install_mocks(svc, deps)
    model_id = next(iter(svc.models["default"]))

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.vllm_service.get_model_dir_context_window", new_callable=AsyncMock, return_value=4096),
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert deps["docker_service"].install_and_run_docker.call_count == 1


@pytest.mark.asyncio
async def test_install_model_registers_chat_completion_endpoint(svc: VllmService, deps: dict[str, Any], tmp_path: Path) -> None:
    _setup_install_mocks(svc, deps)
    model_id = next(iter(svc.models["default"]))

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.vllm_service.get_model_dir_context_window", new_callable=AsyncMock, return_value=4096),
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert deps["endpoint_registry"].register_chat_completion_as_proxy.call_count == 1


@pytest.mark.asyncio
async def test_install_model_records_model_as_downloaded(svc: VllmService, deps: dict[str, Any], tmp_path: Path) -> None:
    _setup_install_mocks(svc, deps)
    model_id = next(iter(svc.models["default"]))

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.vllm_service.get_model_dir_context_window", new_callable=AsyncMock, return_value=4096),
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert model_id in svc.models_downloaded


@pytest.mark.asyncio
async def test_install_model_uses_alias_as_registered_name(svc: VllmService, deps: dict[str, Any], tmp_path: Path) -> None:
    installed = _setup_install_mocks(svc, deps)
    model_id = next(iter(svc.models["default"]))

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.vllm_service.get_model_dir_context_window", new_callable=AsyncMock, return_value=4096),
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn(spec={"alias": "my-alias"}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert installed.models[model_id].registered_name == "my-alias"


@pytest.mark.asyncio
async def test_install_model_registers_reranker_endpoint_for_reranker_model(svc: VllmService, deps: dict[str, Any], tmp_path: Path) -> None:
    _setup_install_mocks(svc, deps)
    reranker_id = "reranker-model"
    svc.models["default"][reranker_id] = VllmModel(hf_id=reranker_id, size="1GB", model_type="reranker")

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
    ):
        promise = await svc._install_model("default", reranker_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert deps["endpoint_registry"].register_rerank_as_proxy.call_count == 1
    assert deps["endpoint_registry"].register_chat_completion_as_proxy.call_count == 0


@pytest.mark.asyncio
async def test_install_model_docker_failure_decrements_gpu_memory(svc: VllmService, deps: dict[str, Any], tmp_path: Path) -> None:
    nvidia = MagicMock(spec=NvidiaGpuInfo)
    deps["hardware"].gpus = [nvidia]
    svc2 = VllmService(**deps)
    installed = _make_installed_info(hardware=True)
    svc2.instances_info["default"].installed = installed
    deps["docker_service"].install_and_run_docker = AsyncMock(side_effect=RuntimeError("docker failed"))
    deps["docker_service"].get_docker_subnet.return_value = None
    deps["docker_service"].get_docker_container_name.return_value = "container"
    deps["docker_service"].stop_docker = AsyncMock()
    model_id = next(iter(svc2.models["default"]))

    with (
        patch.object(svc2, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc2, "get_specified_hardware_parts", return_value=[nvidia]),
        patch.object(svc2, "is_given_hardware_support_gpu", return_value=True),
    ):
        promise = await svc2._install_model("default", model_id, InstallModelIn(spec={"gpu_memory_utilization": 0.5}))  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(RuntimeError):
            await promise.wait()

    assert svc2.gpu_memory_utilization == 0.0


@pytest.mark.asyncio
async def test_uninstall_model_removes_from_installed_and_unregisters_endpoint(svc: VllmService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    installed.models["test-model"] = _make_model_installed_info("test-model", "reg-1")
    svc.instances_info["default"].installed = installed
    deps["docker_service"].uninstall_docker = AsyncMock()

    await svc._uninstall_model("default", "test-model", UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert "test-model" not in installed.models
    assert deps["endpoint_registry"].unregister_chat_completion.call_count == 1
    assert deps["endpoint_registry"].unregister_chat_completion.call_args == call("test-model", "reg-1")


@pytest.mark.asyncio
async def test_uninstall_model_decrements_gpu_memory_utilization(svc: VllmService, deps: dict[str, Any]) -> None:
    svc.gpu_memory_utilization = 0.5
    installed = _make_installed_info()
    installed.models["test-model"] = _make_model_installed_info("test-model", gpu_memory_utilization=0.5)
    svc.instances_info["default"].installed = installed
    deps["docker_service"].uninstall_docker = AsyncMock()

    await svc._uninstall_model("default", "test-model", UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert svc.gpu_memory_utilization == 0.0


@pytest.mark.asyncio
async def test_uninstall_model_floors_gpu_memory_at_zero(svc: VllmService, deps: dict[str, Any]) -> None:
    svc.gpu_memory_utilization = 0.1
    installed = _make_installed_info()
    installed.models["test-model"] = _make_model_installed_info("test-model", gpu_memory_utilization=0.5)
    svc.instances_info["default"].installed = installed
    deps["docker_service"].uninstall_docker = AsyncMock()

    await svc._uninstall_model("default", "test-model", UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert svc.gpu_memory_utilization == 0.0


@pytest.mark.asyncio
async def test_uninstall_model_purges_files_and_downloaded_entry(svc: VllmService, deps: dict[str, Any], tmp_path: Path) -> None:
    installed = _make_installed_info()
    model_dir = tmp_path / "model-dir"
    model_dir.mkdir()
    model_info = _make_model_installed_info("test-model")
    model_info.model_path = model_dir
    installed.models["test-model"] = model_info
    svc.instances_info["default"].installed = installed
    svc.models_downloaded["test-model"] = DownloadedInfo(model_path=str(model_dir))
    deps["docker_service"].uninstall_docker = AsyncMock()

    await svc._uninstall_model("default", "test-model", UninstallModelIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert not model_dir.exists()
    assert "test-model" not in svc.models_downloaded


@pytest.mark.asyncio
async def test_uninstall_model_reranker_calls_unregister_rerank(svc: VllmService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    installed.models["reranker-model"] = _make_model_installed_info("reranker-model", "reg-rerank", model_type="reranker")
    svc.instances_info["default"].installed = installed
    deps["docker_service"].uninstall_docker = AsyncMock()

    await svc._uninstall_model("default", "reranker-model", UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert deps["endpoint_registry"].unregister_rerank.call_count == 1
    assert deps["endpoint_registry"].unregister_rerank.call_args == call("reranker-model", "reg-rerank")
    assert deps["endpoint_registry"].unregister_chat_completion.call_count == 0


@pytest.mark.asyncio
async def test_uninstall_model_ignores_unknown_model_id(svc: VllmService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    deps["docker_service"].uninstall_docker = AsyncMock()

    await svc._uninstall_model("default", "unknown-model", UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert deps["endpoint_registry"].unregister_chat_completion.call_count == 0
    assert deps["docker_service"].uninstall_docker.call_count == 0


@pytest.mark.asyncio
async def test_install_instance_returns_installed_info_with_empty_models(svc: VllmService) -> None:
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
async def test_uninstall_instance_calls_uninstall_model_for_each_model(svc: VllmService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    installed.models["m1"] = _make_model_installed_info("m1")
    installed.models["m2"] = _make_model_installed_info("m2")
    svc.instances_info["default"].installed = installed
    deps["docker_service"].uninstall_docker = AsyncMock()

    with patch.object(svc, "_uninstall_model", new_callable=AsyncMock) as mock_uninstall:  # pyright: ignore[reportPrivateUsage]
        await svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert mock_uninstall.call_count == 2


@pytest.mark.asyncio
async def test_uninstall_instance_purge_clears_service_state(svc: VllmService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
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
async def test_stop_instance_does_nothing_when_not_installed(svc: VllmService) -> None:
    svc.instances_info["default"].installed = None

    with patch.object(svc, "_stop_dockers_parallel", new_callable=AsyncMock) as mock_stop:  # pyright: ignore[reportPrivateUsage]
        await svc.stop_instance("default")

    assert mock_stop.call_count == 0


@pytest.mark.asyncio
async def test_stop_instance_stops_all_containers(svc: VllmService) -> None:
    installed = _make_installed_info()
    installed.models["m1"] = _make_model_installed_info("m1")
    installed.models["m2"] = _make_model_installed_info("m2")
    svc.instances_info["default"].installed = installed

    with patch.object(svc, "_stop_dockers_parallel", new_callable=AsyncMock) as mock_stop:  # pyright: ignore[reportPrivateUsage]
        await svc.stop_instance("default")

    assert mock_stop.call_count == 1
    called_dockers = mock_stop.call_args[0][0]
    assert len(called_dockers) == 2


def test_get_specified_hardware_parts_delegates_to_super_when_hardware_ok(svc: VllmService) -> None:
    svc.hardware.cpu.avx512 = True

    with patch("server.services.base2_service.Base2Service.get_specified_hardware_parts", return_value=[]) as mock_super:
        result = svc.get_specified_hardware_parts(False)

    assert mock_super.call_count == 1
    assert result == []


def test_get_model_spec_adds_hardware_key_when_missing(svc: VllmService, deps: dict[str, Any]) -> None:
    deps["docker_service"].has_gpu_support = False
    svc.instances_info["default"].config = InstanceConfig(options=InstallServiceIn(spec={}))

    result = svc.get_model_spec("default", "llm")

    assert result is not None


@pytest.mark.asyncio
async def test_install_instance_loads_default_models_for_new_instance(svc: VllmService, deps: dict[str, Any]) -> None:
    svc.instances_info["inst2"] = Instance(None, None, {}, InstanceConfig())
    options = InstallServiceIn(spec={"hardware": False})
    svc.hardware.cpu.avx512 = True

    with (
        patch.object(svc, "_download_image_or_set_progress", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_verify_docker_image", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "load_default_models") as mock_load,
    ):
        promise = await svc._install_instance("inst2", options)  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert mock_load.call_count == 1
    assert mock_load.call_args == call("inst2")


@pytest.mark.asyncio
async def test_install_instance_adds_hardware_key_when_missing(svc: VllmService, deps: dict[str, Any]) -> None:
    options = InstallServiceIn(spec={})
    deps["docker_service"].has_gpu_support = False
    svc.hardware.cpu.avx512 = True

    with (
        patch.object(svc, "_download_image_or_set_progress", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_verify_docker_image", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
    ):
        await svc._install_instance("default", options)  # pyright: ignore[reportPrivateUsage]

    assert "hardware" in options.spec


@pytest.mark.asyncio
async def test_uninstall_instance_purge_removes_non_default_instance(svc: VllmService, deps: dict[str, Any]) -> None:
    svc.instances_info["inst2"] = Instance(None, None, {}, InstanceConfig())
    installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}), parsed_options=VllmOptions())
    svc.instances_info["inst2"].installed = installed
    deps["docker_service"].remove_image = AsyncMock()

    with (
        patch.object(svc, "_uninstall_model", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_clear_working_dir", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
    ):
        await svc._uninstall_instance("inst2", UninstallServiceIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert "inst2" not in svc.instances_info


@pytest.mark.asyncio
async def test_list_models_skips_instances_not_in_filter(svc: VllmService) -> None:
    svc.instances_info["inst2"] = Instance(None, None, {}, InstanceConfig())
    svc.models["inst2"] = {"extra-model": VllmModel(hf_id="extra/model", size="1GB")}
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}), parsed_options=VllmOptions())

    result = await svc.list_models("default", ListModelsFilters())

    assert not any(m.id == "extra-model" for m in result.list)


@pytest.mark.asyncio
async def test_download_model_handles_pre_download_packet_with_size(svc: VllmService, deps: dict[str, Any], tmp_path: Path) -> None:
    model = VllmModel(hf_id="google/test", size="100MB")
    stream = MagicMock()

    async def mock_download(*args: object):  # type: ignore[misc]
        yield PreDownloadPacket(file_bytes_size=50 * 1024 * 1024)
        yield SuccessDownloadPacket(local_path=tmp_path / "model", filename="model")

    deps["model_downloader"].download = mock_download

    with patch.object(svc, "_get_working_dir", return_value=tmp_path):  # pyright: ignore[reportPrivateUsage]
        local_path = await svc._download_model(stream, "google/test", model, tmp_path)  # pyright: ignore[reportPrivateUsage]

    assert local_path is not None


@pytest.mark.asyncio
async def test_download_model_or_set_progress_forwards_chunk_with_data(svc: VllmService, tmp_path: Path) -> None:
    existing_stream: Stream[StreamChunk] = Stream()  # type: ignore[type-arg]
    chunk = StreamChunkProgress(type="progress", stage="download", value=0.9, data={"local_model_path": str(tmp_path / "model")})
    existing_stream.emit(chunk)
    existing_stream.close()
    model = VllmModel(hf_id="google/test", size="1GB")
    svc.models_download_progress["google/test"] = existing_stream  # type: ignore[assignment]
    output_stream = MagicMock()

    result_path = await svc._download_model_or_set_progress(output_stream, "google/test", model, tmp_path)  # pyright: ignore[reportPrivateUsage]

    assert result_path == tmp_path / "model"


@pytest.mark.asyncio
async def test_download_model_or_set_progress_breaks_on_non_download_chunk(svc: VllmService, tmp_path: Path) -> None:
    existing_stream: Stream[StreamChunk] = Stream()  # type: ignore[type-arg]
    finish_chunk: StreamChunk = {"type": "finish", "status": "ok"}  # type: ignore[assignment]
    progress_chunk = StreamChunkProgress(type="progress", stage="download", value=1.0, data={"local_model_path": str(tmp_path / "model")})
    existing_stream.emit(progress_chunk)
    existing_stream.emit(finish_chunk)
    existing_stream.close()
    model = VllmModel(hf_id="google/test", size="1GB")
    svc.models_download_progress["google/test"] = existing_stream  # type: ignore[assignment]
    output_stream = MagicMock()

    result_path = await svc._download_model_or_set_progress(output_stream, "google/test", model, tmp_path)  # pyright: ignore[reportPrivateUsage]

    assert result_path is not None


@pytest.mark.asyncio
async def test_get_max_model_length_raises_422_for_non_integer_float(svc: VllmService) -> None:
    model = VllmModel(hf_id="google/test", size="1GB")
    opts = VllmModelOptions.model_construct(max_model_length=4096.5)

    with pytest.raises(HTTPException) as exc_info:
        await svc._get_max_model_length(opts, model)  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 422


def test_build_vllm_command_use_gpu_with_model_length(svc: VllmService, tmp_path: Path) -> None:
    opts = VllmModelOptions.model_construct(extra_args={})

    cmd = svc._build_vllm_command(  # pyright: ignore[reportPrivateUsage]
        docker_model_path=tmp_path / "model",
        model_id="google/test",
        opts=opts,
        quantization=None,
        gpu_memory_utilization=None,
        user_model_length=4096,
        use_gpu=True,
    )

    assert "--max-model-len" in cmd
    assert "4096" in cmd


def test_build_vllm_command_no_gpu_no_model_length_adds_disable_sliding_window(svc: VllmService, tmp_path: Path) -> None:
    opts = VllmModelOptions.model_construct(extra_args={})

    cmd = svc._build_vllm_command(  # pyright: ignore[reportPrivateUsage]
        docker_model_path=tmp_path / "model",
        model_id="google/test",
        opts=opts,
        quantization=None,
        gpu_memory_utilization=None,
        user_model_length=None,
        use_gpu=False,
    )

    assert "--disable-sliding-window" in cmd


def test_build_vllm_command_extra_args_with_value(svc: VllmService, tmp_path: Path) -> None:
    opts = VllmModelOptions.model_construct(extra_args={"--dtype": "float16"})

    cmd = svc._build_vllm_command(  # pyright: ignore[reportPrivateUsage]
        docker_model_path=tmp_path / "model",
        model_id="google/test",
        opts=opts,
        quantization=None,
        gpu_memory_utilization=None,
        user_model_length=None,
        use_gpu=True,
    )

    assert "--dtype" in cmd
    assert "float16" in cmd


def test_build_vllm_command_extra_args_flag_only(svc: VllmService, tmp_path: Path) -> None:
    opts = VllmModelOptions.model_construct(extra_args={"--enforce-eager": None})

    cmd = svc._build_vllm_command(  # pyright: ignore[reportPrivateUsage]
        docker_model_path=tmp_path / "model",
        model_id="google/test",
        opts=opts,
        quantization=None,
        gpu_memory_utilization=None,
        user_model_length=None,
        use_gpu=True,
    )

    assert "--enforce-eager" in cmd


def test_build_vllm_command_no_gpu_with_model_length(svc: VllmService, tmp_path: Path) -> None:
    opts = VllmModelOptions.model_construct(extra_args={})

    cmd = svc._build_vllm_command(  # pyright: ignore[reportPrivateUsage]
        docker_model_path=tmp_path / "model",
        model_id="google/test",
        opts=opts,
        quantization=None,
        gpu_memory_utilization=None,
        user_model_length=2048,
        use_gpu=False,
    )

    assert "--max-model-len" in cmd
    assert "2048" in cmd


@pytest.mark.asyncio
async def test_download_model_raises_500_when_no_success_packet(svc: VllmService, deps: dict[str, Any], tmp_path: Path) -> None:
    model = VllmModel(hf_id="google/test", size="100MB")
    stream = MagicMock()

    async def mock_download(*args: object):  # type: ignore[misc]
        yield DownloadedPacket(downloaded_bytes_size=1024)

    deps["model_downloader"].download = mock_download

    with (
        patch.object(svc, "_get_working_dir", return_value=tmp_path),  # pyright: ignore[reportPrivateUsage]
        pytest.raises(HTTPException) as exc_info,
    ):
        await svc._download_model(stream, "google/test", model, tmp_path)  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 500


@pytest.mark.asyncio
async def test_download_model_or_set_progress_raises_500_after_break_with_no_path(svc: VllmService, tmp_path: Path) -> None:
    existing_stream: Stream[StreamChunk] = Stream()  # type: ignore[type-arg]
    finish_chunk: StreamChunk = {"type": "finish", "status": "ok"}  # type: ignore[assignment]
    existing_stream.emit(finish_chunk)
    existing_stream.close()
    model = VllmModel(hf_id="google/test", size="1GB")
    svc.models_download_progress["google/test"] = existing_stream  # type: ignore[assignment]
    output_stream = MagicMock()

    with pytest.raises(HTTPException) as exc_info:
        await svc._download_model_or_set_progress(output_stream, "google/test", model, tmp_path)  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 500


def test_get_size_no_cpu_without_avx512(deps: dict[str, Any]) -> None:
    deps["hardware"].cpu.avx512 = False
    nvidia = MagicMock(spec=NvidiaGpuInfo)
    deps["hardware"].gpus = [nvidia]
    svc = VllmService(**deps)

    sizes = svc.get_size()

    assert "cpu" not in sizes
    assert "gpu" in sizes


@pytest.mark.asyncio
async def test_uninstall_instance_does_nothing_when_not_installed(svc: VllmService, deps: dict[str, Any]) -> None:
    svc.instances_info["default"].installed = None

    with patch.object(svc, "_uninstall_model", new_callable=AsyncMock) as mock_uninstall:  # pyright: ignore[reportPrivateUsage]
        await svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert mock_uninstall.call_count == 0
    assert svc.instances_info["default"].installed is None


@pytest.mark.asyncio
async def test_uninstall_instance_skips_model_installed_in_other_instance(svc: VllmService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    installed.models["m1"] = _make_model_installed_info("m1")
    installed.models["m2"] = _make_model_installed_info("m2")
    svc.instances_info["default"].installed = installed
    deps["docker_service"].uninstall_docker = AsyncMock()

    def mock_is_in_other(instance: str, model_id: str) -> bool:
        return model_id == "m1"

    with (
        patch.object(svc, "is_model_installed_in_other_instance", side_effect=mock_is_in_other),
        patch.object(svc, "_uninstall_model", new_callable=AsyncMock) as mock_uninstall,  # pyright: ignore[reportPrivateUsage]
    ):
        await svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert mock_uninstall.call_count == 1
    called_ids = [call.args[1] for call in mock_uninstall.call_args_list]
    assert "m1" not in called_ids
    assert "m2" in called_ids


@pytest.mark.asyncio
async def test_download_model_handles_pre_download_packet_without_file_size(svc: VllmService, deps: dict[str, Any], tmp_path: Path) -> None:
    model = VllmModel(hf_id="google/test", size="100MB")
    stream = MagicMock()
    local_path = tmp_path / "model-files"
    local_path.mkdir()

    async def mock_download(*args: object):  # type: ignore[misc]
        yield PreDownloadPacket(file_bytes_size=None)  # pyright: ignore[reportArgumentType]
        yield SuccessDownloadPacket(local_path=local_path, filename="model-files")

    deps["model_downloader"].download = mock_download
    result = await svc._download_model(stream, "google/test", model, tmp_path)  # pyright: ignore[reportPrivateUsage]

    assert result == local_path


@pytest.mark.asyncio
async def test_download_model_ignores_zero_bytes_downloaded_packet(svc: VllmService, deps: dict[str, Any], tmp_path: Path) -> None:
    model = VllmModel(hf_id="google/test", size="100MB")
    stream = MagicMock()
    local_path = tmp_path / "model-files"
    local_path.mkdir()

    async def mock_download(*args: object):  # type: ignore[misc]
        yield DownloadedPacket(downloaded_bytes_size=0)
        yield SuccessDownloadPacket(local_path=local_path, filename="model-files")

    deps["model_downloader"].download = mock_download
    result = await svc._download_model(stream, "google/test", model, tmp_path)  # pyright: ignore[reportPrivateUsage]

    assert result == local_path


@pytest.mark.asyncio
async def test_download_model_or_set_progress_emits_chunk_with_empty_data(svc: VllmService, tmp_path: Path) -> None:
    existing_stream: Stream[StreamChunk] = Stream()  # type: ignore[type-arg]
    chunk_empty_data = StreamChunkProgress(type="progress", stage="download", value=0.3, data={})
    chunk_with_data = StreamChunkProgress(
        type="progress",
        stage="download",
        value=1.0,
        data={"local_model_path": str(tmp_path / "model")},
    )
    existing_stream.emit(chunk_empty_data)
    existing_stream.emit(chunk_with_data)
    existing_stream.close()

    model_id = "google/test"
    model = VllmModel(hf_id=model_id, size="1GB")
    svc.models_download_progress[model_id] = existing_stream  # type: ignore[assignment]
    output_stream = MagicMock()

    result_path = await svc._download_model_or_set_progress(output_stream, model_id, model, tmp_path)  # pyright: ignore[reportPrivateUsage]

    assert result_path == tmp_path / "model"
    assert output_stream.emit.call_count == 2


@pytest.mark.asyncio
async def test_install_model_docker_failure_no_decrement_when_model_gpu_utilization_zero(
    svc: VllmService, deps: dict[str, Any], tmp_path: Path
) -> None:
    nvidia = MagicMock(spec=NvidiaGpuInfo)
    deps["hardware"].gpus = [nvidia]
    svc2 = VllmService(**deps)
    installed = _make_installed_info(hardware=True)
    svc2.instances_info["default"].installed = installed
    deps["docker_service"].install_and_run_docker = AsyncMock(side_effect=RuntimeError("docker failed"))
    deps["docker_service"].get_docker_subnet.return_value = None
    deps["docker_service"].get_docker_container_name.return_value = "container"
    deps["docker_service"].stop_docker = AsyncMock()

    model_id = "zero-gpu-model"
    svc2.models["default"][model_id] = VllmModel(hf_id=model_id, size="1GB", gpu_memory_utilization=0.0)

    with (
        patch.object(svc2, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc2, "get_specified_hardware_parts", return_value=[nvidia]),
        patch.object(svc2, "is_given_hardware_support_gpu", return_value=True),
    ):
        promise = await svc2._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(RuntimeError):
            await promise.wait()

    assert svc2.gpu_memory_utilization > 0


@pytest.mark.asyncio
async def test_install_model_docker_failure_no_floor_when_utilization_stays_positive(
    svc: VllmService, deps: dict[str, Any], tmp_path: Path
) -> None:
    nvidia = MagicMock(spec=NvidiaGpuInfo)
    deps["hardware"].gpus = [nvidia]
    svc2 = VllmService(**deps)
    svc2.gpu_memory_utilization = 0.5
    installed = _make_installed_info(hardware=True)
    svc2.instances_info["default"].installed = installed
    deps["docker_service"].install_and_run_docker = AsyncMock(side_effect=RuntimeError("docker failed"))
    deps["docker_service"].get_docker_subnet.return_value = None
    deps["docker_service"].get_docker_container_name.return_value = "container"
    deps["docker_service"].stop_docker = AsyncMock()

    model_id = "small-gpu-model"
    svc2.models["default"][model_id] = VllmModel(hf_id=model_id, size="1GB", gpu_memory_utilization=0.3)

    with (
        patch.object(svc2, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc2, "get_specified_hardware_parts", return_value=[nvidia]),
        patch.object(svc2, "is_given_hardware_support_gpu", return_value=True),
    ):
        promise = await svc2._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(RuntimeError):
            await promise.wait()

    assert svc2.gpu_memory_utilization == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_uninstall_model_purge_with_none_model_path_removes_downloaded_entry(svc: VllmService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    installed.models["test-model"] = _make_model_installed_info("test-model")
    svc.instances_info["default"].installed = installed
    svc.models_downloaded["test-model"] = DownloadedInfo(model_path=None)
    deps["docker_service"].uninstall_docker = AsyncMock()

    await svc._uninstall_model("default", "test-model", UninstallModelIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert "test-model" not in svc.models_downloaded


@pytest.mark.asyncio
async def test_resolve_custom_model_size_returns_formatted_size(svc: VllmService, deps: dict[str, Any]) -> None:
    with patch("server.services.vllm_service.fetch_huggingface_model_size", new=AsyncMock(return_value="4.0 GB")):
        result = await svc._resolve_custom_model_size({"hf_id": "google/gemma-3-270m-it"})  # pyright: ignore[reportPrivateUsage]

    assert result == "4.0 GB"


@pytest.mark.asyncio
async def test_resolve_custom_model_size_returns_none_on_exception(svc: VllmService, deps: dict[str, Any]) -> None:
    with patch("server.services.vllm_service.fetch_huggingface_model_size", new=AsyncMock(side_effect=Exception("fail"))):
        result = await svc._resolve_custom_model_size({"hf_id": "google/gemma"})  # pyright: ignore[reportPrivateUsage]

    assert result is None


@pytest.mark.asyncio
async def test_get_docker_logs_cache_hit(svc: VllmService) -> None:
    svc._log_cache["my-container"] = (time.monotonic(), "cached logs")  # pyright: ignore[reportPrivateUsage]

    result = await svc._get_docker_logs("my-container")  # pyright: ignore[reportPrivateUsage]

    assert result == "cached logs"


@pytest.mark.asyncio
async def test_get_docker_logs_fetches_and_caches(svc: VllmService) -> None:
    mock_result = MagicMock()
    mock_result.stdout = "stdout logs"
    mock_result.stderr = "stderr logs"

    with patch("server.services.base2_service.Utils.run_command", new_callable=AsyncMock, return_value=mock_result):
        result = await svc._get_docker_logs("my-container")  # pyright: ignore[reportPrivateUsage]

    assert result == "stdout logsstderr logs"
    assert "my-container" in svc._log_cache  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Total (incl. non-KV cache overhead): 8.50 GiB\n", 8.5),
        ("Model weights: 6.00 GiB\nKV Cache: 2.00 GiB\n", 8.0),
        ("model weights take 5.00 GiB\nKV cache memory: 1.50 GiB\n", 6.5),
        ("Model weights: 4.00 GiB\n", 4.0),
        ("KV cache: 3.00 GiB\n", 3.0),
        ("no useful info here", None),
    ],
)
def test_parse_vllm_vram_gb(raw: str, expected: float | None) -> None:
    result = VllmService._parse_vllm_vram_gb(raw)  # pyright: ignore[reportPrivateUsage]

    assert result == expected


@pytest.mark.asyncio
async def test_get_vram_from_logs_returns_parsed_value(svc: VllmService) -> None:
    installed = _make_installed_info()
    model_info = _make_model_installed_info("test-model")
    model_info.docker.container_name = "vllm-container"
    installed.models["test-model"] = model_info
    svc.instances_info["default"].installed = installed

    with patch.object(svc, "_get_docker_logs", new_callable=AsyncMock, return_value="Total (incl. non-KV cache overhead): 7.00 GiB"):  # pyright: ignore[reportPrivateUsage]
        result = await svc._get_vram_from_logs("default", "test-model")  # pyright: ignore[reportPrivateUsage]

    assert result == 7.0


@pytest.mark.asyncio
async def test_get_vram_from_logs_returns_none_when_no_container(svc: VllmService) -> None:
    installed = _make_installed_info()
    model_info = _make_model_installed_info("test-model")
    model_info.docker.container_name = ""
    installed.models["test-model"] = model_info
    svc.instances_info["default"].installed = installed

    result = await svc._get_vram_from_logs("default", "test-model")  # pyright: ignore[reportPrivateUsage]

    assert result is None


def test_get_vram_estimate_returns_value(svc: VllmService) -> None:
    svc.hardware.total_vram_gb = 24.0  # pyright: ignore[reportAttributeAccessIssue]
    model_info = _make_model_installed_info("test-model", gpu_memory_utilization=0.5)

    result = svc._get_vram_estimate(model_info)  # pyright: ignore[reportPrivateUsage]

    assert result == 12.0


def test_get_vram_estimate_returns_none_when_no_utilization(svc: VllmService) -> None:
    model_info = _make_model_installed_info("test-model", gpu_memory_utilization=None)

    result = svc._get_vram_estimate(model_info)  # pyright: ignore[reportPrivateUsage]

    assert result is None


def test_get_vram_estimate_returns_none_when_total_vram_zero(svc: VllmService) -> None:
    svc.hardware.total_vram_gb = 0.0  # pyright: ignore[reportAttributeAccessIssue]
    model_info = _make_model_installed_info("test-model", gpu_memory_utilization=0.5)

    result = svc._get_vram_estimate(model_info)  # pyright: ignore[reportPrivateUsage]

    assert result is None


@pytest.mark.asyncio
async def test_get_cached_vram_estimate_cache_hit(svc: VllmService) -> None:
    svc._vram_cache[("default", "test-model")] = 4.0  # pyright: ignore[reportPrivateUsage]

    result = await svc._get_cached_vram_estimate("default", "test-model", is_loaded=True)  # pyright: ignore[reportPrivateUsage]

    assert result == 4.0


@pytest.mark.asyncio
async def test_get_cached_vram_estimate_from_logs_non_none(svc: VllmService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    with patch.object(svc, "_get_vram_from_logs", new_callable=AsyncMock, return_value=5.5):  # pyright: ignore[reportPrivateUsage]
        result = await svc._get_cached_vram_estimate("default", "test-model", is_loaded=True)  # pyright: ignore[reportPrivateUsage]

    assert result == 5.5
    assert svc._vram_cache[("default", "test-model")] == 5.5  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_get_cached_vram_estimate_model_not_in_installed(svc: VllmService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed  # no models in installed

    with patch.object(svc, "_get_vram_from_logs", new_callable=AsyncMock, return_value=None):  # pyright: ignore[reportPrivateUsage]
        result = await svc._get_cached_vram_estimate("default", "missing-model", is_loaded=True)  # pyright: ignore[reportPrivateUsage]

    assert result is None
    assert ("default", "missing-model") not in svc._vram_cache  # pyright: ignore[reportPrivateUsage]
