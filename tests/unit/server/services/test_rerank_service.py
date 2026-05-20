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

from server.models.models import InstallModelIn, ListModelsFilters, ModelInfo, UninstallModelIn
from server.models.services import InstallServiceIn, UninstallServiceIn
from server.services.base2_service import CustomModel, Instance, InstanceConfig
from server.services.rerank_service import (
    DownloadedInfo,
    InstalledInfo,
    ModelInstalledInfo,
    RerankModel,
    RerankModelOptions,
    RerankOptions,
    RerankService,
    _const,  # pyright: ignore[reportPrivateUsage]
)
from server.utils.core import DownloadedPacket, PreDownloadPacket, Stream, StreamChunkProgress
from server.utils.hardware import NvidiaGpuInfo


def _make_installed_info(instance: str = "default") -> InstalledInfo:
    docker = MagicMock()
    docker.name = f"rerank-{instance}"
    return InstalledInfo(
        docker=docker,
        models={},
        options=InstallServiceIn(spec={}),
        parsed_options=RerankOptions(),
        container_host="localhost",
        container_port=8089,
        docker_exposed_port=8089,
        base_url="http://localhost:8089",
    )


@pytest.fixture
def deps() -> dict[str, Any]:
    return {
        "config": MagicMock(),
        "endpoint_registry": MagicMock(),
        "service_provider": MagicMock(),
        "model_downloader": MagicMock(),
        "docker_service": MagicMock(),
        "hardware": MagicMock(gpus=[], nvidia_gpus=[]),
    }


@pytest.fixture
def svc(deps: dict[str, Any]) -> RerankService:
    return RerankService(**deps)


def test_get_type(svc: RerankService) -> None:
    assert svc.get_type() == "rerank"


def test_get_description_not_empty(svc: RerankService) -> None:
    assert svc.get_description()


def test_service_has_docker(svc: RerankService) -> None:
    assert svc.service_has_docker() is True


def test_is_not_cloud_service(svc: RerankService) -> None:
    assert svc.is_cloud_service() is False


def test_const_has_gpu_and_cpu_images() -> None:
    assert "gpu" in _const.images
    assert "cpu" in _const.images


def test_const_gpu_image_contains_cuda() -> None:
    assert "cuda" in _const.images["gpu"].name.lower()


def test_const_models_not_empty() -> None:
    assert len(_const.models) > 0


def test_const_models_all_rerank_type() -> None:
    for model in _const.models.values():
        assert model.type == "rerank"


def test_get_spec_has_hardware_field(svc: RerankService) -> None:
    spec = svc.get_spec()

    field_names = [f.name for f in spec.fields]
    assert "hardware" in field_names


def test_get_spec_has_keep_alive_field(svc: RerankService) -> None:
    spec = svc.get_spec()

    field_names = [f.name for f in spec.fields]
    assert "keep_alive" in field_names


def test_get_spec_keep_alive_is_optional(svc: RerankService) -> None:
    spec = svc.get_spec()

    field = next(f for f in spec.fields if f.name == "keep_alive")
    assert field.required is False


def test_get_model_spec_has_alias_alive_time_preload(svc: RerankService) -> None:
    spec = svc.get_model_spec()

    field_names = {f.name for f in spec.fields}
    assert {"alias", "alive_time", "preload"} <= field_names


def test_get_model_spec_all_fields_optional(svc: RerankService) -> None:
    spec = svc.get_model_spec()

    for field in spec.fields:
        assert field.required is False, f"Field {field.name!r} should be optional"


def test_get_custom_model_spec_not_none(svc: RerankService) -> None:
    assert svc.get_custom_model_spec() is not None


def test_get_custom_model_spec_has_id_hf_id_size(svc: RerankService) -> None:
    spec = svc.get_custom_model_spec()

    assert spec is not None
    field_names = {f.name for f in spec.fields}
    assert {"id", "hf_id", "size"} <= field_names


def test_rerank_options_defaults() -> None:
    opts = RerankOptions()

    assert opts.hardware is None
    assert opts.keep_alive is None


def test_rerank_model_options_defaults() -> None:
    opts = RerankModelOptions()

    assert opts.alias is None
    assert opts.alive_time is None
    assert opts.preload is None


def test_get_image_with_nvidia_gpu_returns_gpu_image(svc: RerankService) -> None:
    nvidia_gpu = MagicMock(spec=NvidiaGpuInfo)

    image = svc._get_image([nvidia_gpu])  # pyright: ignore[reportPrivateUsage]

    assert image == _const.images["gpu"]


def test_get_image_no_gpu_returns_cpu_image(svc: RerankService) -> None:
    image = svc._get_image([])  # pyright: ignore[reportPrivateUsage]

    assert image == _const.images["cpu"]


def test_get_image_non_nvidia_returns_cpu_image(svc: RerankService) -> None:
    non_nvidia = MagicMock()

    image = svc._get_image([non_nvidia])  # pyright: ignore[reportPrivateUsage]

    assert image == _const.images["cpu"]


def test_default_models_loaded_on_init(svc: RerankService) -> None:
    assert "default" in svc.models
    assert len(svc.models["default"]) > 0


def test_default_models_match_const(svc: RerankService) -> None:
    for model_id in _const.models:
        assert model_id in svc.models["default"]


def test_model_installed_info_get_info() -> None:
    model = ModelInstalledInfo(
        id="cross-encoder/ms-marco-MiniLM-L6-v2",
        registered_name="cross-encoder/ms-marco-MiniLM-L6-v2",
        type="rerank",
        options=InstallModelIn(spec={}),
        registration_id="reg-42",
    )

    info = model.get_info()

    assert info.registration_id == "reg-42"


# --- get_size ---


def test_get_size_without_gpu_has_cpu_key(svc: RerankService) -> None:
    sizes = svc.get_size()

    assert "cpu" in sizes


def test_get_size_with_nvidia_gpu_has_gpu_key(deps: dict[str, Any]) -> None:
    nvidia_gpu = MagicMock(spec=NvidiaGpuInfo)
    deps["hardware"].nvidia_gpus = [nvidia_gpu]
    svc = RerankService(**deps)

    sizes = svc.get_size()

    assert "gpu" in sizes


def test_supported_gpus_only_nvidia(deps: dict[str, Any]) -> None:
    nvidia = MagicMock(spec=NvidiaGpuInfo)
    other = MagicMock()
    deps["hardware"].gpus = [nvidia, other]

    svc = RerankService(**deps)

    assert svc._supported_gpus == [nvidia]  # pyright: ignore[reportPrivateUsage]


def test_generate_instance_config_no_info(svc: RerankService) -> None:
    config = svc._generate_instance_config(None, None)  # pyright: ignore[reportPrivateUsage]

    assert config.options is None
    assert config.models == []


def test_generate_instance_config_with_info(svc: RerankService) -> None:
    info = _make_installed_info()
    model_id = "BAAI/bge-reranker-base"
    info.models[model_id] = ModelInstalledInfo(
        id=model_id,
        registered_name=model_id,
        type="rerank",
        options=InstallModelIn(spec={}),
        registration_id="reg-1",
    )

    config = svc._generate_instance_config(info, None)  # pyright: ignore[reportPrivateUsage]

    assert config.options == info.options
    assert len(config.models or []) == 1


def test_generate_instance_config_with_custom(svc: RerankService) -> None:
    info = _make_installed_info()
    custom = [CustomModel(id="c-1", data={})]

    config = svc._generate_instance_config(info, custom)  # pyright: ignore[reportPrivateUsage]

    assert config.custom is not None


def test_load_download_info_returns_downloaded_info(svc: RerankService) -> None:
    result = svc._load_download_info({"model_path": "/some/path"})  # pyright: ignore[reportPrivateUsage]

    assert isinstance(result, DownloadedInfo)
    assert result.model_path == "/some/path"


def test_load_download_info_none_model_path(svc: RerankService) -> None:
    result = svc._load_download_info({"model_path": None})  # pyright: ignore[reportPrivateUsage]

    assert result.model_path is None


def test_get_installed_info_returns_spec_when_installed(svc: RerankService) -> None:
    installed = _make_installed_info()
    installed.options.spec["hardware"] = True
    svc.instances_info["default"].installed = installed

    result = svc.get_installed_info("default")

    assert result == installed.options.spec


def test_get_installed_info_calls_get_service_installed_info_when_none(svc: RerankService) -> None:
    svc.instances_info["default"].installed = None

    with patch.object(svc, "_get_service_installed_info", return_value=False) as mock:  # pyright: ignore[reportPrivateUsage]
        result = svc.get_installed_info("default")

    assert mock.call_count == 1
    assert mock.call_args == call("default")
    assert result is False


def test_get_docker_compose_file_path_raises_400_with_model_id(svc: RerankService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    with pytest.raises(HTTPException) as exc_info:
        svc.get_docker_compose_file_path("default", "some-model")

    assert exc_info.value.status_code == 400


def test_get_docker_compose_file_path_returns_path_without_model_id(svc: RerankService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    expected = Path("/some/path/docker-compose.yml")
    deps["docker_service"].get_docker_compose_file_path.return_value = expected

    result = svc.get_docker_compose_file_path("default", None)

    assert result == expected


@pytest.mark.asyncio
async def test_stop_instance_does_nothing_when_not_installed(svc: RerankService) -> None:
    svc.instances_info["default"].installed = None

    with patch.object(svc, "_stop_docker", new_callable=AsyncMock) as mock_stop:  # pyright: ignore[reportPrivateUsage]
        await svc.stop_instance("default")

    assert mock_stop.call_count == 0


@pytest.mark.asyncio
async def test_stop_instance_calls_stop_docker_when_installed(svc: RerankService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    with patch.object(svc, "_stop_docker", new_callable=AsyncMock) as mock_stop:  # pyright: ignore[reportPrivateUsage]
        await svc.stop_instance("default")

    assert mock_stop.call_count == 1
    assert mock_stop.call_args == call(installed.docker)


@pytest.mark.asyncio
async def test_list_models_returns_all_models_for_valid_instance(svc: RerankService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    result = await svc.list_models("default", ListModelsFilters())

    assert len(result.list) == len(svc.models["default"])


@pytest.mark.asyncio
async def test_list_models_raises_404_for_unknown_instance(svc: RerankService) -> None:
    with pytest.raises(HTTPException) as exc_info:
        await svc.list_models("nonexistent", ListModelsFilters())

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_list_models_filters_to_installed_only(svc: RerankService) -> None:
    installed = _make_installed_info()
    model_id = next(iter(svc.models["default"]))
    installed.models[model_id] = ModelInstalledInfo(
        id=model_id,
        registered_name=model_id,
        type="rerank",
        options=InstallModelIn(spec={}),
        registration_id="reg-1",
    )
    svc.instances_info["default"].installed = installed

    result = await svc.list_models("default", ListModelsFilters(installed=True))

    assert len(result.list) >= 1
    assert all(bool(m.installed) for m in result.list)


@pytest.mark.asyncio
async def test_list_models_filters_to_not_installed(svc: RerankService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    result = await svc.list_models("default", ListModelsFilters(installed=False))

    assert all(not m.installed for m in result.list)


@pytest.mark.asyncio
async def test_list_models_with_list_of_instances(svc: RerankService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    result = await svc.list_models(["default"], ListModelsFilters())

    assert len(result.list) > 0


@pytest.mark.asyncio
async def test_list_models_with_none_uses_all_instances(svc: RerankService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    result = await svc.list_models(None, ListModelsFilters())

    assert len(result.list) > 0


@pytest.mark.asyncio
async def test_get_model_raises_400_for_unknown_model_id(svc: RerankService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    with pytest.raises(HTTPException) as exc_info:
        await svc.get_model("default", "nonexistent-model")

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_get_model_returns_retrieve_model_out_for_known_model(svc: RerankService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    model_id = next(iter(_const.models))

    result = await svc.get_model("default", model_id)

    assert result.id == model_id


@pytest.mark.asyncio
async def test_get_model_installed_model_returns_get_info(svc: RerankService) -> None:
    installed = _make_installed_info()
    model_id = next(iter(_const.models))
    installed.models[model_id] = ModelInstalledInfo(
        id=model_id,
        registered_name=model_id,
        type="rerank",
        options=InstallModelIn(spec={}),
        registration_id="reg-42",
    )
    svc.instances_info["default"].installed = installed

    result = await svc.get_model("default", model_id)

    assert result.installed is not None
    assert isinstance(result.installed, ModelInfo)
    assert result.installed.registration_id == "reg-42"


@pytest.mark.asyncio
async def test_download_model_emits_progress(svc: RerankService) -> None:
    model_id = "cross-encoder/ms-marco-MiniLM-L6-v2"
    model = _const.models[model_id]
    stream = MagicMock()

    async def mock_download(*args: object, **kwargs: object):  # type: ignore[misc]
        yield DownloadedPacket(downloaded_bytes_size=1000)
        yield PreDownloadPacket(file_bytes_size=5000)

    svc.model_downloader.hugging_face_repo_with_blobs_downloader.download.return_value = mock_download()  # pyright: ignore[reportAttributeAccessIssue]

    await svc._download_model(stream, model, model_id, Path("/tmp"))  # pyright: ignore[reportPrivateUsage]

    assert stream.emit.call_count >= 2


@pytest.mark.asyncio
async def test_download_model_emits_final_progress_1(svc: RerankService) -> None:
    model_id = "cross-encoder/ms-marco-MiniLM-L6-v2"
    model = _const.models[model_id]
    stream = MagicMock()

    async def mock_download(*args: object, **kwargs: object):  # type: ignore[misc]
        return
        yield  # make it an async generator

    svc.model_downloader.hugging_face_repo_with_blobs_downloader.download.return_value = mock_download()  # pyright: ignore[reportAttributeAccessIssue]

    await svc._download_model(stream, model, model_id, Path("/tmp"))  # pyright: ignore[reportPrivateUsage]

    last_call = stream.emit.call_args_list[-1]
    chunk = last_call.args[0]
    assert chunk["value"] == 1


@pytest.mark.asyncio
async def test_download_model_or_set_progress_starts_download(svc: RerankService) -> None:
    model_id = "cross-encoder/ms-marco-MiniLM-L6-v2"
    model = _const.models[model_id]
    stream = MagicMock()

    with patch.object(svc, "_download_model", new_callable=AsyncMock) as mock_dl:  # pyright: ignore[reportPrivateUsage]
        await svc._download_model_or_set_progress(stream, model, model_id, Path("/tmp"))  # pyright: ignore[reportPrivateUsage]

    assert mock_dl.call_count == 1
    assert model_id not in svc.models_download_progress


@pytest.mark.asyncio
async def test_download_model_or_set_progress_forwards_when_in_progress(svc: RerankService) -> None:
    model_id = "cross-encoder/ms-marco-MiniLM-L6-v2"
    model = _const.models[model_id]
    existing_stream: Stream[StreamChunkProgress] = Stream()  # type: ignore[type-arg]
    chunk = StreamChunkProgress(type="progress", stage="download", value=0.5, data={})
    existing_stream.emit(chunk)
    existing_stream.close()
    svc.models_download_progress[model_id] = existing_stream  # type: ignore[assignment]
    output_stream = MagicMock()

    await svc._download_model_or_set_progress(output_stream, model, model_id, Path("/tmp"))  # pyright: ignore[reportPrivateUsage]

    assert output_stream.emit.call_count == 1
    assert output_stream.emit.call_args == call(chunk)


@pytest.mark.asyncio
async def test_install_model_returns_already_installed_when_model_in_info(svc: RerankService) -> None:
    installed = _make_installed_info()
    model_id = next(iter(_const.models))
    installed.models[model_id] = ModelInstalledInfo(
        id=model_id,
        registered_name=model_id,
        type="rerank",
        options=InstallModelIn(spec={}),
        registration_id="",
    )
    svc.instances_info["default"].installed = installed

    promise = await svc._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
    result = await promise.wait()

    assert result.status == "OK"
    assert "Already installed" in result.details


@pytest.mark.asyncio
async def test_install_model_raises_400_for_unknown_model(svc: RerankService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    with pytest.raises(HTTPException) as exc_info:
        await svc._install_model("default", "nonexistent-model", InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_install_model_registers_rerank_endpoint(svc: RerankService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    model_id = next(iter(_const.models))

    with patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock):  # pyright: ignore[reportPrivateUsage]
        promise = await svc._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert deps["endpoint_registry"].register_rerank_as_proxy.call_count == 1


@pytest.mark.asyncio
async def test_install_model_marks_model_as_downloaded(svc: RerankService, tmp_path: Path) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    model_id = next(iter(_const.models))

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_get_working_dir", return_value=tmp_path),  # pyright: ignore[reportPrivateUsage]
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert model_id in svc.models_downloaded


@pytest.mark.asyncio
async def test_install_model_uses_alias_as_registered_name(svc: RerankService, tmp_path: Path) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    model_id = next(iter(_const.models))

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_get_working_dir", return_value=tmp_path),  # pyright: ignore[reportPrivateUsage]
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn(spec={"alias": "my-alias"}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert installed.models[model_id].registered_name == "my-alias"


@pytest.mark.asyncio
async def test_install_model_creates_models_dict_for_new_instance(svc: RerankService, tmp_path: Path) -> None:
    svc.instances_info["extra"] = Instance(None, None, {}, InstanceConfig())
    installed = _make_installed_info("extra")
    svc.instances_info["extra"].installed = installed
    svc.load_default_models("extra")
    model_id = next(iter(_const.models))

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_get_working_dir", return_value=tmp_path),  # pyright: ignore[reportPrivateUsage]
    ):
        promise = await svc._install_model("extra", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert "extra" in svc.models


@pytest.mark.asyncio
async def test_install_model_calls_fetch_for_alive_time(svc: RerankService, tmp_path: Path) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    model_id = next(iter(_const.models))

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_get_working_dir", return_value=tmp_path),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.rerank_service.fetch_from", new_callable=AsyncMock) as mock_fetch,
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn(spec={"alive_time": 60}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()
    calls = [str(c.args[0]) for c in mock_fetch.call_args_list]

    assert any("configure" in url for url in calls)


@pytest.mark.asyncio
async def test_install_model_calls_fetch_for_preload(svc: RerankService, tmp_path: Path) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    model_id = next(iter(_const.models))

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_get_working_dir", return_value=tmp_path),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.rerank_service.fetch_from", new_callable=AsyncMock) as mock_fetch,
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn(spec={"preload": True}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()
    calls = [str(c.args[0]) for c in mock_fetch.call_args_list]

    assert any("load" in url for url in calls)


@pytest.mark.asyncio
async def test_uninstall_model_unregisters_rerank_and_removes_from_info(svc: RerankService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    model_id = next(iter(_const.models))
    installed.models[model_id] = ModelInstalledInfo(
        id=model_id,
        registered_name=model_id,
        type="rerank",
        options=InstallModelIn(spec={}),
        registration_id="reg-1",
    )
    svc.instances_info["default"].installed = installed

    with patch("server.services.rerank_service.fetch_from", new_callable=AsyncMock):
        await svc._uninstall_model("default", model_id, UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert model_id not in installed.models
    assert deps["endpoint_registry"].unregister_rerank.call_count == 1
    assert deps["endpoint_registry"].unregister_rerank.call_args == call(model_id, "reg-1")


@pytest.mark.asyncio
async def test_uninstall_model_purges_model_data(svc: RerankService, tmp_path: Path) -> None:
    installed = _make_installed_info()
    model_id = next(iter(_const.models))
    model_dir = tmp_path / "model_data"
    model_dir.mkdir()
    installed.models[model_id] = ModelInstalledInfo(
        id=model_id,
        registered_name=model_id,
        type="rerank",
        options=InstallModelIn(spec={}),
        registration_id="",
    )
    svc.instances_info["default"].installed = installed
    svc.models_downloaded[model_id] = DownloadedInfo(model_path=str(model_dir))

    with patch("server.services.rerank_service.fetch_from", new_callable=AsyncMock):
        await svc._uninstall_model("default", model_id, UninstallModelIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert model_id not in svc.models_downloaded
    assert not model_dir.exists()


@pytest.mark.asyncio
async def test_uninstall_model_does_nothing_for_unknown_model_id(svc: RerankService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    with patch("server.services.rerank_service.fetch_from", new_callable=AsyncMock):
        await svc._uninstall_model("default", "unknown-model", UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert deps["endpoint_registry"].unregister_rerank.call_count == 0


@pytest.mark.asyncio
async def test_uninstall_model_purge_with_none_model_path_does_not_raise(svc: RerankService) -> None:
    installed = _make_installed_info()
    model_id = next(iter(_const.models))
    installed.models[model_id] = ModelInstalledInfo(
        id=model_id,
        registered_name=model_id,
        type="rerank",
        options=InstallModelIn(spec={}),
        registration_id="",
    )
    svc.instances_info["default"].installed = installed
    svc.models_downloaded[model_id] = DownloadedInfo(model_path=None)

    with patch("server.services.rerank_service.fetch_from", new_callable=AsyncMock):
        await svc._uninstall_model("default", model_id, UninstallModelIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert model_id not in svc.models_downloaded


@pytest.mark.asyncio
async def test_install_instance_calls_docker_and_returns_installed_info(svc: RerankService, deps: dict[str, Any]) -> None:
    deps["docker_service"].get_docker_subnet.return_value = None
    deps["docker_service"].get_docker_container_name.return_value = "df-rerank"
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=8089)
    deps["docker_service"].get_container_host.return_value = "localhost"
    deps["docker_service"].get_container_port.return_value = 8089
    options = InstallServiceIn(spec={})

    with (
        patch.object(svc, "_download_image_or_set_progress", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_verify_docker_image", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
        patch("server.services.rerank_service.get_base_url", return_value="http://localhost:8089"),
    ):
        promise = await svc._install_instance("default", options)  # pyright: ignore[reportPrivateUsage]
        result = await promise.wait()

    assert isinstance(result, InstalledInfo)
    assert deps["docker_service"].install_and_run_docker.call_count == 1


@pytest.mark.asyncio
async def test_install_instance_sets_keep_alive_env_var(svc: RerankService, deps: dict[str, Any]) -> None:
    deps["docker_service"].get_docker_subnet.return_value = None
    deps["docker_service"].get_docker_container_name.return_value = "df-rerank"
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=8089)
    deps["docker_service"].get_container_host.return_value = "localhost"
    deps["docker_service"].get_container_port.return_value = 8089
    options = InstallServiceIn(spec={"keep_alive": 60})
    captured: list[Any] = []

    async def capture(docker_opts: Any) -> int:
        captured.append(docker_opts)
        return 8089

    deps["docker_service"].install_and_run_docker = capture

    with (
        patch.object(svc, "_download_image_or_set_progress", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_verify_docker_image", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
        patch("server.services.rerank_service.get_base_url", return_value="http://localhost:8089"),
    ):
        promise = await svc._install_instance("default", options)  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert captured
    assert captured[0].env_vars.get("DF_RERANK_KEEP_ALIVE") == "60"


@pytest.mark.asyncio
async def test_install_instance_loads_default_models_when_missing(svc: RerankService, deps: dict[str, Any]) -> None:
    svc.instances_info["extra"] = Instance(None, None, {}, InstanceConfig())
    del svc.models["default"]
    deps["docker_service"].get_docker_subnet.return_value = None
    deps["docker_service"].get_docker_container_name.return_value = "df-rerank-extra"
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=8089)
    deps["docker_service"].get_container_host.return_value = "localhost"
    deps["docker_service"].get_container_port.return_value = 8089
    options = InstallServiceIn(spec={})

    with (
        patch.object(svc, "_download_image_or_set_progress", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_verify_docker_image", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
        patch("server.services.rerank_service.get_base_url", return_value="http://localhost:8089"),
    ):
        promise = await svc._install_instance("default", options)  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert "default" in svc.models


@pytest.mark.asyncio
async def test_uninstall_instance_unregisters_rerank_model(svc: RerankService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    model_id = next(iter(_const.models))
    installed.models[model_id] = ModelInstalledInfo(
        id=model_id,
        registered_name=model_id,
        type="rerank",
        options=InstallModelIn(spec={}),
        registration_id="reg-1",
    )
    svc.instances_info["default"].installed = installed
    deps["docker_service"].uninstall_docker = AsyncMock()

    with patch.object(svc, "_uninstall_model", new_callable=AsyncMock):  # pyright: ignore[reportPrivateUsage]
        await svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert deps["endpoint_registry"].unregister_rerank.call_count == 1
    assert deps["endpoint_registry"].unregister_rerank.call_args == call(model_id, "reg-1")
    assert deps["docker_service"].uninstall_docker.call_count == 1


@pytest.mark.asyncio
async def test_uninstall_instance_purges_working_dir_on_single_instance(svc: RerankService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    deps["docker_service"].uninstall_docker = AsyncMock()
    deps["docker_service"].remove_image = AsyncMock()

    with (
        patch.object(svc, "_uninstall_model", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_clear_working_dir", new_callable=AsyncMock) as mock_clear,  # pyright: ignore[reportPrivateUsage]
    ):
        await svc._uninstall_instance("default", UninstallServiceIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert svc.service_downloaded is False
    assert mock_clear.call_count == 1


@pytest.mark.asyncio
async def test_uninstall_instance_resets_default_instance_to_empty(svc: RerankService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    deps["docker_service"].uninstall_docker = AsyncMock()
    deps["docker_service"].remove_image = AsyncMock()

    with (
        patch.object(svc, "_uninstall_model", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_clear_working_dir", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
    ):
        await svc._uninstall_instance("default", UninstallServiceIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert "default" in svc.instances_info
    assert svc.instances_info["default"].installed is None


@pytest.mark.asyncio
async def test_uninstall_instance_removes_non_default_instance_on_purge(svc: RerankService, deps: dict[str, Any]) -> None:
    svc.instances_info["gpu-1"] = Instance(None, None, {}, InstanceConfig())
    installed = _make_installed_info("gpu-1")
    svc.instances_info["gpu-1"].installed = installed
    svc.load_default_models("gpu-1")
    deps["docker_service"].uninstall_docker = AsyncMock()
    deps["docker_service"].remove_image = AsyncMock()

    with (
        patch.object(svc, "_uninstall_model", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_clear_working_dir", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
    ):
        await svc._uninstall_instance("gpu-1", UninstallServiceIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert "gpu-1" not in svc.instances_info


@pytest.mark.asyncio
async def test_uninstall_instance_sets_installed_to_none_when_not_installed(svc: RerankService, deps: dict[str, Any]) -> None:
    svc.instances_info["default"].installed = None

    await svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert svc.instances_info["default"].installed is None


@pytest.mark.asyncio
async def test_list_models_skips_instances_not_in_filter(svc: RerankService) -> None:
    svc.instances_info["extra"] = Instance(None, None, {}, InstanceConfig())
    svc.instances_info["extra"].installed = _make_installed_info("extra")
    svc.load_default_models("extra")

    svc.instances_info["default"].installed = _make_installed_info()

    result = await svc.list_models("default", ListModelsFilters())

    service_ids = {m.service for m in result.list}
    assert all("extra" not in s for s in service_ids)


@pytest.mark.asyncio
async def test_get_model_initialises_models_dict_when_missing(svc: RerankService) -> None:
    svc.instances_info["default"].installed = _make_installed_info()
    svc.models.pop("default", None)

    with pytest.raises(HTTPException) as exc_info:
        await svc.get_model("default", "nonexistent-model")

    assert exc_info.value.status_code == 400
    assert "default" in svc.models


@pytest.mark.asyncio
async def test_download_model_or_set_progress_breaks_on_non_download_chunk(svc: RerankService) -> None:
    model_id = next(iter(_const.models))
    existing_stream: Stream[Any] = Stream()
    existing_stream.emit({"type": "finish", "status": "ok", "details": {}})
    existing_stream.close()

    svc.models_download_progress[model_id] = existing_stream

    out_stream: Stream[Any] = Stream()
    model = next(iter(svc.models["default"].values()))

    await svc._download_model_or_set_progress(out_stream, model, model_id, Path("/tmp"))  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_install_model_initialises_models_dict_for_new_instance_already_installed(svc: RerankService) -> None:
    installed = _make_installed_info("new-inst")
    svc.instances_info["new-inst"] = Instance(None, None, {}, InstanceConfig())
    svc.instances_info["new-inst"].installed = installed
    model_id = next(iter(_const.models))
    installed.models[model_id] = ModelInstalledInfo(
        id=model_id,
        registered_name=model_id,
        type="rerank",
        options=InstallModelIn(spec={}),
        registration_id="reg-1",
    )

    # Instance has no models loaded so the branch triggers, then returns early (already installed)
    promise = await svc._install_model("new-inst", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]

    assert "new-inst" in svc.models
    result = await promise.wait()
    assert result.status == "OK"


@pytest.mark.asyncio
async def test_uninstall_instance_skips_unregister_for_non_rerank_model(svc: RerankService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    model_id = next(iter(_const.models))
    installed.models[model_id] = ModelInstalledInfo(
        id=model_id,
        registered_name=model_id,
        type="embed",  # not "rerank"
        options=InstallModelIn(spec={}),
        registration_id="reg-1",
    )
    svc.instances_info["default"].installed = installed
    deps["docker_service"].uninstall_docker = AsyncMock()

    with patch.object(svc, "_uninstall_model", new_callable=AsyncMock):  # pyright: ignore[reportPrivateUsage]
        await svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert deps["endpoint_registry"].unregister_rerank.call_count == 0


@pytest.mark.asyncio
async def test_uninstall_instance_skips_uninstall_model_when_shared_across_instances(svc: RerankService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    model_id = next(iter(_const.models))
    installed.models[model_id] = ModelInstalledInfo(
        id=model_id,
        registered_name=model_id,
        type="rerank",
        options=InstallModelIn(spec={}),
        registration_id="reg-1",
    )
    svc.instances_info["default"].installed = installed
    deps["docker_service"].uninstall_docker = AsyncMock()

    with (
        patch.object(svc, "is_model_installed_in_other_instance", return_value=True),
        patch.object(svc, "_uninstall_model", new_callable=AsyncMock) as mock_uninstall,  # pyright: ignore[reportPrivateUsage]
    ):
        await svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert mock_uninstall.call_count == 0


@pytest.mark.asyncio
async def test_download_model_ignores_unknown_packet_type(svc: RerankService) -> None:
    model_id = next(iter(_const.models))
    model = _const.models[model_id]
    stream = MagicMock()

    async def mock_download(*args: object, **kwargs: object):  # type: ignore[misc]
        yield object()  # neither DownloadedPacket nor PreDownloadPacket

    svc.model_downloader.hugging_face_repo_with_blobs_downloader.download.return_value = mock_download()  # pyright: ignore[reportAttributeAccessIssue]

    await svc._download_model(stream, model, model_id, Path("/tmp"))  # pyright: ignore[reportPrivateUsage]

    last_call = stream.emit.call_args_list[-1]
    assert last_call.args[0]["value"] == 1


@pytest.mark.asyncio
async def test_download_model_pre_download_packet_with_zero_size(svc: RerankService) -> None:
    model_id = next(iter(_const.models))
    model = _const.models[model_id]
    stream = MagicMock()

    async def mock_download(*args: object, **kwargs: object):  # type: ignore[misc]
        yield PreDownloadPacket(file_bytes_size=0)

    svc.model_downloader.hugging_face_repo_with_blobs_downloader.download.return_value = mock_download()  # pyright: ignore[reportAttributeAccessIssue]

    await svc._download_model(stream, model, model_id, Path("/tmp"))  # pyright: ignore[reportPrivateUsage]

    last_call = stream.emit.call_args_list[-1]
    assert last_call.args[0]["value"] == 1


@pytest.mark.asyncio
async def test_install_model_skips_rerank_registration_for_non_rerank_type(
    svc: RerankService, deps: dict[str, Any], tmp_path: Path
) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    model_id = next(iter(_const.models))

    svc.models["default"][model_id] = RerankModel(type="embed", size="88MB")

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_get_working_dir", return_value=tmp_path),  # pyright: ignore[reportPrivateUsage]
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert deps["endpoint_registry"].register_rerank_as_proxy.call_count == 0


@pytest.mark.asyncio
async def test_uninstall_model_skips_unregister_for_non_rerank_type(svc: RerankService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    model_id = next(iter(_const.models))
    installed.models[model_id] = ModelInstalledInfo(
        id=model_id,
        registered_name=model_id,
        type="embed",  # not "rerank"
        options=InstallModelIn(spec={}),
        registration_id="reg-1",
    )
    svc.instances_info["default"].installed = installed

    with patch("server.services.rerank_service.fetch_from", new_callable=AsyncMock):
        await svc._uninstall_model("default", model_id, UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert deps["endpoint_registry"].unregister_rerank.call_count == 0
    assert model_id not in installed.models


@pytest.mark.asyncio
async def test_resolve_custom_model_size_returns_formatted_size(svc: RerankService, deps: dict[str, Any]) -> None:
    with patch("server.services.rerank_service.fetch_huggingface_model_size", new=AsyncMock(return_value="2.0 GB")):
        result = await svc._resolve_custom_model_size({"hf_id": "cross-encoder/ms-marco-MiniLM-L-6-v2"})  # pyright: ignore[reportPrivateUsage]

    assert result == "2.0 GB"


@pytest.mark.asyncio
async def test_resolve_custom_model_size_returns_none_on_exception(svc: RerankService, deps: dict[str, Any]) -> None:
    with patch("server.services.rerank_service.fetch_huggingface_model_size", new=AsyncMock(side_effect=Exception("fail"))):
        result = await svc._resolve_custom_model_size({"hf_id": "google/model"})  # pyright: ignore[reportPrivateUsage]

    assert result is None
