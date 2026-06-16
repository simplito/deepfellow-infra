# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any
from unittest.mock import AsyncMock, MagicMock, call

import pytest
from fastapi import HTTPException

from server.models.models import AddCustomModelIn, InstallModelIn, ListModelsFilters, ModelInfo, UninstallModelIn
from server.models.services import InstallServiceIn, UninstallServiceIn
from server.services.base2_service import CustomModel, Instance, InstanceConfig
from server.services.custom_service import (
    CustomService,
    DownloadedInfo,
    InstalledInfo,
    ModelInstalledInfo,
    _const,  # pyright: ignore[reportPrivateUsage]
    create_bge_m3_model,
    create_doc_chunker_model,
    create_finetune_model,
    create_lemmatizer_model,
)
from server.utils.hardware import NvidiaGpuInfo

_CUSTOM_MODEL_DATA: dict[str, Any] = {
    "id": "my-custom",
    "default_prefix": "my-custom",
    "size": "1GB",
    "image": "test/image:latest",
    "image_port": 8000,
}


@pytest.fixture
def deps() -> dict[str, Any]:
    docker_svc = MagicMock()
    docker_svc.get_docker_subnet.return_value = "172.20.0.0/16"
    docker_svc.get_docker_container_name.side_effect = lambda name: f"df-{name}"  # pyright: ignore[reportUnknownLambdaType]
    return {
        "config": MagicMock(),
        "endpoint_registry": MagicMock(),
        "service_provider": MagicMock(),
        "model_downloader": MagicMock(),
        "docker_service": docker_svc,
        "hardware": MagicMock(gpus=[]),
    }


@pytest.fixture
def svc(deps: dict[str, Any]) -> CustomService:
    return CustomService(**deps)


def test_get_type(svc: CustomService) -> None:
    assert svc.get_type() == "custom"


def test_get_description_not_empty(svc: CustomService) -> None:
    assert svc.get_description()


def test_get_size_empty(svc: CustomService) -> None:
    assert svc.get_size() == ""


def test_get_spec_returns_empty_fields(svc: CustomService) -> None:
    spec = svc.get_spec()
    assert spec.fields == []


def test_default_models_loaded_on_init(svc: CustomService) -> None:
    assert "default" in svc.models
    assert len(svc.models["default"]) > 0


def test_const_models_not_empty() -> None:
    assert len(_const.models) > 0


def test_service_has_docker(svc: CustomService) -> None:
    assert svc.service_has_docker() is False


def test_get_custom_model_spec_not_none(svc: CustomService) -> None:
    assert svc.get_custom_model_spec() is not None


def test_get_default_model_spec_contains_prefix_field(svc: CustomService) -> None:
    spec = svc.get_default_model_spec("my-prefix")
    field_names = [f.name for f in spec.fields]
    assert "prefix" in field_names


def test_model_installed_info_get_info() -> None:
    options = InstallModelIn(spec={"prefix": "test"})
    model_info = ModelInstalledInfo(
        id="test-model",
        options=options,
        docker_options=MagicMock(),
        container_host="127.0.0.1",
        container_port=8080,
        docker_exposed_port=8080,
        registration_id="reg-123",
        prefix="test",
        base_url="http://127.0.0.1:8080",
    )

    info = model_info.get_info()

    assert info.spec == options.spec
    assert info.registration_id == "reg-123"


@pytest.mark.asyncio
async def test_stop_instance_no_installed_is_noop(svc: CustomService, deps: dict[str, Any]) -> None:
    deps["docker_service"].stop_docker = AsyncMock()

    await svc.stop_instance("default")

    assert deps["docker_service"].stop_docker.call_count == 0


@pytest.mark.asyncio
async def test_stop_instance_with_installed_stops_containers(svc: CustomService, deps: dict[str, Any]) -> None:
    docker_opts = MagicMock()
    mock_model = MagicMock()
    mock_model.docker_options = docker_opts
    svc.instances_info["default"].installed = InstalledInfo(
        models={"lemmatizer": mock_model},
        options=InstallServiceIn(spec={}),
    )
    deps["docker_service"].stop_docker = AsyncMock()

    await svc.stop_instance("default")

    assert deps["docker_service"].stop_docker.call_count == 1
    assert deps["docker_service"].stop_docker.call_args == call(docker_opts)


def test_get_installed_info_when_not_installed_returns_false(svc: CustomService) -> None:
    result = svc.get_installed_info("default")
    assert result is False


def test_get_installed_info_when_installed_returns_spec(svc: CustomService) -> None:
    options = InstallServiceIn(spec={"key": "val"})
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=options)

    result = svc.get_installed_info("default")

    assert result == {"key": "val"}


def test_generate_instance_config_without_info(svc: CustomService) -> None:
    config = svc._generate_instance_config(None, None)  # pyright: ignore[reportPrivateUsage]

    assert config.options is None
    assert config.models == []


def test_generate_instance_config_with_info(svc: CustomService) -> None:
    mock_model_info = MagicMock()
    mock_model_info.id = "lemmatizer"
    mock_model_info.options = InstallModelIn(spec={})
    installed = InstalledInfo(models={"lemmatizer": mock_model_info}, options=InstallServiceIn(spec={}))

    config = svc._generate_instance_config(installed, None)  # pyright: ignore[reportPrivateUsage]

    assert config.options == installed.options
    assert config.models is not None
    assert len(config.models) == 1
    assert config.models[0].model_id == "lemmatizer"


def test_load_download_info(svc: CustomService) -> None:
    info = svc._load_download_info({"image": "test/image:latest"})  # pyright: ignore[reportPrivateUsage]

    assert info.image == "test/image:latest"


@pytest.mark.asyncio
async def test_install_instance_returns_installed_info(svc: CustomService) -> None:
    options = InstallServiceIn(spec={})

    promise = await svc._install_instance("default", options)  # pyright: ignore[reportPrivateUsage]

    result = await promise.wait()
    assert isinstance(result, InstalledInfo)
    assert svc.service_downloaded is True


@pytest.mark.asyncio
async def test_uninstall_instance_without_purge_clears_installed(svc: CustomService) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))

    await svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert svc.instances_info["default"].installed is None


@pytest.mark.asyncio
async def test_uninstall_instance_with_purge_resets_service(svc: CustomService) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    svc._clear_working_dir = AsyncMock()  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage]

    await svc._uninstall_instance("default", UninstallServiceIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert svc.service_downloaded is False
    assert svc._clear_working_dir.call_count == 1  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage]


@pytest.mark.asyncio
async def test_uninstall_instance_with_models_uninstalls_each(svc: CustomService, deps: dict[str, Any]) -> None:
    mock_model = MagicMock()
    mock_model.id = "lemmatizer"
    mock_model.prefix = "lemmatizer"
    mock_model.registration_id = "reg-1"
    mock_model.docker_options = MagicMock()
    deps["docker_service"].uninstall_docker = AsyncMock()
    svc.instances_info["default"].installed = InstalledInfo(
        models={"lemmatizer": mock_model},
        options=InstallServiceIn(spec={}),
    )

    await svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert svc.instances_info["default"].installed is None
    assert deps["docker_service"].uninstall_docker.call_count == 1


def test_get_docker_compose_file_path_no_model_id_raises(svc: CustomService) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))

    with pytest.raises(HTTPException) as exc:
        svc.get_docker_compose_file_path("default", None)

    assert exc.value.status_code == 400


def test_get_docker_compose_file_path_model_not_found_raises(svc: CustomService) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))

    with pytest.raises(HTTPException) as exc:
        svc.get_docker_compose_file_path("default", "nonexistent")

    assert exc.value.status_code == 400


def test_get_docker_compose_file_path_success(svc: CustomService, deps: dict[str, Any]) -> None:
    expected = MagicMock()
    deps["docker_service"].get_docker_compose_file_path.return_value = expected
    svc.instances_info["default"].installed = InstalledInfo(
        models={"lemmatizer": MagicMock()},
        options=InstallServiceIn(spec={}),
    )

    result = svc.get_docker_compose_file_path("default", "lemmatizer")

    assert result == expected


def test_add_custom_model_adds_to_models(svc: CustomService) -> None:
    model = CustomModel(id="cm-1", data=_CUSTOM_MODEL_DATA)

    svc._add_custom_model("default", model)  # pyright: ignore[reportPrivateUsage]

    assert "my-custom" in svc.models["default"]


def test_add_custom_model_duplicate_raises(svc: CustomService) -> None:
    svc._add_custom_model("default", CustomModel(id="cm-1", data=_CUSTOM_MODEL_DATA))  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(HTTPException) as exc:
        svc._add_custom_model("default", CustomModel(id="cm-2", data=_CUSTOM_MODEL_DATA))  # pyright: ignore[reportPrivateUsage]

    assert exc.value.status_code == 400


def test_remove_custom_model_when_in_use_raises(svc: CustomService) -> None:
    model = CustomModel(id="cm-1", data=_CUSTOM_MODEL_DATA)
    svc._add_custom_model("default", model)  # pyright: ignore[reportPrivateUsage]
    svc.instances_info["default"].installed = InstalledInfo(
        models={"my-custom": MagicMock()},
        options=InstallServiceIn(spec={}),
    )

    with pytest.raises(HTTPException) as exc:
        svc._remove_custom_model("default", model)  # pyright: ignore[reportPrivateUsage]

    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_update_custom_model_not_found_raises_404(svc: CustomService) -> None:
    with pytest.raises(HTTPException) as exc:
        await svc.update_custom_model("default", "nonexistent-id", AddCustomModelIn(spec={"name": "x"}))

    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_update_custom_model_default_raises_400(svc: CustomService) -> None:
    model = CustomModel(id="cm-1", data=_CUSTOM_MODEL_DATA)
    svc.instances_info["default"].config.custom = [model]

    with pytest.raises(HTTPException) as exc:
        await svc.update_custom_model("default", "cm-1", AddCustomModelIn(spec=_CUSTOM_MODEL_DATA))

    assert exc.value.status_code == 400


def test_remove_custom_model_success(svc: CustomService) -> None:
    model = CustomModel(id="cm-1", data=_CUSTOM_MODEL_DATA)
    svc._add_custom_model("default", model)  # pyright: ignore[reportPrivateUsage]

    svc._remove_custom_model("default", model)  # pyright: ignore[reportPrivateUsage]

    assert "my-custom" not in svc.models["default"]


@pytest.mark.asyncio
async def test_list_models_unknown_instance_raises(svc: CustomService) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))

    with pytest.raises(HTTPException) as exc:
        await svc.list_models("ghost", ListModelsFilters())

    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_list_models_returns_all_when_no_filter(svc: CustomService) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))

    result = await svc.list_models("default", ListModelsFilters(installed=None))

    assert len(result.list) > 0


@pytest.mark.asyncio
async def test_list_models_filter_not_installed(svc: CustomService) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))

    result = await svc.list_models("default", ListModelsFilters(installed=False))

    assert len(result.list) > 0
    assert all(item.installed is False for item in result.list)


@pytest.mark.asyncio
async def test_list_models_filter_installed_returns_only_installed(svc: CustomService) -> None:
    mock_installed = MagicMock()
    mock_installed.get_info.return_value = ModelInfo(spec={"prefix": "lemmatizer"}, registration_id="r1")
    svc.instances_info["default"].installed = InstalledInfo(
        models={"lemmatizer": mock_installed},
        options=InstallServiceIn(spec={}),
    )

    result = await svc.list_models("default", ListModelsFilters(installed=True))

    assert len(result.list) == 1
    assert result.list[0].id == "lemmatizer"


@pytest.mark.asyncio
async def test_list_models_none_instance_iterates_all(svc: CustomService) -> None:
    svc.instances_info["gpu-1"] = Instance(None, None, {}, InstanceConfig())
    svc.load_default_models("gpu-1")
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    svc.instances_info["gpu-1"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))

    result = await svc.list_models(None, ListModelsFilters(installed=None))

    assert len(result.list) > 0


@pytest.mark.asyncio
async def test_get_model_not_found_raises(svc: CustomService) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))

    with pytest.raises(HTTPException) as exc:
        await svc.get_model("default", "ghost")

    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_get_model_not_installed_returns_false(svc: CustomService) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))

    result = await svc.get_model("default", "lemmatizer")

    assert result.id == "lemmatizer"
    assert result.installed is False


@pytest.mark.asyncio
async def test_get_model_installed_returns_model_info(svc: CustomService) -> None:
    model_info = ModelInfo(spec={"prefix": "lemmatizer"}, registration_id="r1")
    mock_installed = MagicMock()
    mock_installed.get_info.return_value = model_info
    svc.instances_info["default"].installed = InstalledInfo(
        models={"lemmatizer": mock_installed},
        options=InstallServiceIn(spec={}),
    )

    result = await svc.get_model("default", "lemmatizer")

    assert result.installed == model_info


@pytest.mark.asyncio
async def test_install_model_already_installed_returns_ok(svc: CustomService) -> None:
    svc.instances_info["default"].installed = InstalledInfo(
        models={"lemmatizer": MagicMock()},
        options=InstallServiceIn(spec={}),
    )

    promise = await svc._install_model("default", "lemmatizer", InstallModelIn())  # pyright: ignore[reportPrivateUsage]

    result = await promise.wait()
    assert result.status == "OK"
    assert "Already installed" in result.details


@pytest.mark.asyncio
async def test_install_model_not_found_raises(svc: CustomService) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))

    with pytest.raises(HTTPException) as exc:
        await svc._install_model("default", "ghost", InstallModelIn())  # pyright: ignore[reportPrivateUsage]

    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_install_model_success(svc: CustomService, deps: dict[str, Any]) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    deps["docker_service"].get_image_warnings = AsyncMock(return_value=[])
    deps["docker_service"].is_docker_image_pulled = AsyncMock(return_value=True)
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=8090)
    deps["docker_service"].get_container_host.return_value = "172.20.0.1"
    deps["docker_service"].get_container_port.return_value = 8090
    deps["endpoint_registry"].register_custom_endpoint_as_proxy.return_value = "reg-1"

    promise = await svc._install_model(  # pyright: ignore[reportPrivateUsage]
        "default", "lemmatizer", InstallModelIn(spec={"prefix": "lemmatizer"})
    )
    result = await promise.wait()

    assert result.status == "OK"
    assert "lemmatizer" in svc.instances_info["default"].installed.models  # pyright: ignore[reportOptionalMemberAccess]


@pytest.mark.asyncio
async def test_uninstall_model_removes_from_installed(svc: CustomService, deps: dict[str, Any]) -> None:
    mock_model = MagicMock()
    mock_model.prefix = "lemmatizer"
    mock_model.registration_id = "r1"
    deps["docker_service"].uninstall_docker = AsyncMock()
    svc.instances_info["default"].installed = InstalledInfo(
        models={"lemmatizer": mock_model},
        options=InstallServiceIn(spec={}),
    )

    await svc._uninstall_model("default", "lemmatizer", UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert "lemmatizer" not in svc.instances_info["default"].installed.models  # pyright: ignore[reportOptionalMemberAccess]
    assert deps["endpoint_registry"].unregister_custom_endpoint.call_count == 1


@pytest.mark.asyncio
async def test_uninstall_model_with_purge_removes_downloaded(svc: CustomService, deps: dict[str, Any]) -> None:
    mock_model = MagicMock()
    mock_model.prefix = "lemmatizer"
    mock_model.registration_id = "r1"
    deps["docker_service"].uninstall_docker = AsyncMock()
    deps["docker_service"].remove_image = AsyncMock()
    svc.models_downloaded["lemmatizer"] = DownloadedInfo(image="lemmatizer:latest")
    svc.instances_info["default"].installed = InstalledInfo(
        models={"lemmatizer": mock_model},
        options=InstallServiceIn(spec={}),
    )

    await svc._uninstall_model("default", "lemmatizer", UninstallModelIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert "lemmatizer" not in svc.models_downloaded
    assert deps["docker_service"].remove_image.call_count == 1
    assert deps["docker_service"].remove_image.call_args == call("lemmatizer:latest")


def test_create_lemmatizer_model_cpu_image(svc: CustomService) -> None:
    model = create_lemmatizer_model(svc, "172.20.0.0/16")

    assert callable(model.options)
    docker_opts = model.options({"hardware": "CPU"})  # pyright: ignore[reportCallIssue]
    assert "cpu" in docker_opts.image


def test_create_lemmatizer_model_gpu_image(deps: dict[str, Any]) -> None:
    gpu = NvidiaGpuInfo(name="RTX 4090", vram="24GB", id=0)
    deps["hardware"] = MagicMock(gpus=[gpu], cpu=MagicMock())
    svc_gpu = CustomService(**deps)

    model = create_lemmatizer_model(svc_gpu, "172.20.0.0/16")

    docker_opts = model.options({"hardware": "GPU"})  # pyright: ignore[reportCallIssue]
    assert "cuda" in docker_opts.image


def test_create_bge_m3_model_cpu_image(svc: CustomService) -> None:
    model = create_bge_m3_model(svc, "172.20.0.0/16")

    assert callable(model.options)
    docker_opts = model.options({"hardware": "CPU"})  # pyright: ignore[reportCallIssue]
    assert "cpu" in docker_opts.image


def test_create_bge_m3_model_gpu_image(deps: dict[str, Any]) -> None:
    gpu = NvidiaGpuInfo(name="RTX 4090", vram="24GB", id=0)
    deps["hardware"] = MagicMock(gpus=[gpu], cpu=MagicMock())
    svc_gpu = CustomService(**deps)

    model = create_bge_m3_model(svc_gpu, "172.20.0.0/16")

    docker_opts = model.options({"hardware": "GPU"})  # pyright: ignore[reportCallIssue]
    assert "cuda" in docker_opts.image


def test_create_doc_chunker_model_cpu_image(svc: CustomService) -> None:
    model = create_doc_chunker_model(svc, "172.20.0.0/16")

    assert callable(model.options)
    docker_opts = model.options({"hardware": "CPU"})  # pyright: ignore[reportCallIssue]
    assert "cpu" in docker_opts.image


def test_create_doc_chunker_model_gpu_image(deps: dict[str, Any]) -> None:
    gpu = NvidiaGpuInfo(name="RTX 4090", vram="24GB", id=0)
    deps["hardware"] = MagicMock(gpus=[gpu], cpu=MagicMock())
    svc_gpu = CustomService(**deps)

    model = create_doc_chunker_model(svc_gpu, "172.20.0.0/16")

    docker_opts = model.options({"hardware": "GPU"})  # pyright: ignore[reportCallIssue]
    assert "gpu" in docker_opts.image


def test_create_finetune_model_returns_docker_options(svc: CustomService) -> None:
    model = create_finetune_model(svc, "172.20.0.0/16")

    assert callable(model.options)
    docker_opts = model.options({})  # pyright: ignore[reportCallIssue]
    assert docker_opts.image_port == 8333


@pytest.mark.asyncio
async def test_install_instance_loads_models_for_new_instance(svc: CustomService) -> None:
    # Remove models for "default" to trigger the load branch
    del svc.models["default"]
    options = InstallServiceIn(spec={})
    promise = await svc._install_instance("default", options)  # pyright: ignore[reportPrivateUsage]
    result = await promise.wait()
    assert isinstance(result, InstalledInfo)
    assert "default" in svc.models


@pytest.mark.asyncio
async def test_uninstall_instance_with_purge_deletes_non_default_instance(svc: CustomService) -> None:
    svc.instances_info["gpu-1"] = Instance(None, None, {}, InstanceConfig())
    svc.instances_info["gpu-1"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    svc._clear_working_dir = AsyncMock()  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage]

    await svc._uninstall_instance("gpu-1", UninstallServiceIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert "gpu-1" not in svc.instances_info


def test_add_custom_model_creates_instance_dict_when_missing(svc: CustomService) -> None:
    del svc.models["default"]
    model = CustomModel(id="cm-1", data=_CUSTOM_MODEL_DATA)

    svc._add_custom_model("default", model)  # pyright: ignore[reportPrivateUsage]

    assert "my-custom" in svc.models["default"]


@pytest.mark.asyncio
async def test_get_model_initialises_empty_models_for_instance(svc: CustomService) -> None:
    del svc.models["default"]
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))

    with pytest.raises(HTTPException):
        await svc.get_model("default", "lemmatizer")

    assert "default" in svc.models


@pytest.mark.asyncio
async def test_list_models_skips_instance_not_in_requested(svc: CustomService) -> None:
    svc.instances_info["gpu-1"] = Instance(None, None, {}, InstanceConfig())
    svc.load_default_models("gpu-1")
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    svc.instances_info["gpu-1"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))

    result_default = await svc.list_models("default", ListModelsFilters(installed=None))
    result_gpu = await svc.list_models("gpu-1", ListModelsFilters(installed=None))

    # Each instance returns only its own models
    assert len(result_default.list) > 0
    assert len(result_gpu.list) > 0
    assert len(result_default.list) == len(result_gpu.list)


@pytest.mark.asyncio
async def test_install_model_uses_default_prefix_when_no_spec(svc: CustomService, deps: dict[str, Any]) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    deps["docker_service"].get_image_warnings = AsyncMock(return_value=[])
    deps["docker_service"].is_docker_image_pulled = AsyncMock(return_value=True)
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=8090)
    deps["docker_service"].get_container_host.return_value = "172.20.0.1"
    deps["docker_service"].get_container_port.return_value = 8090
    deps["endpoint_registry"].register_custom_endpoint_as_proxy.return_value = "reg-1"

    # No spec at all → should use default_prefix "lemmatizer"
    promise = await svc._install_model("default", "lemmatizer", InstallModelIn())  # pyright: ignore[reportPrivateUsage]

    result = await promise.wait()
    assert result.status == "OK"


@pytest.mark.asyncio
async def test_install_model_uses_default_prefix_when_prefix_missing(svc: CustomService, deps: dict[str, Any]) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    deps["docker_service"].get_image_warnings = AsyncMock(return_value=[])
    deps["docker_service"].is_docker_image_pulled = AsyncMock(return_value=True)
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=8090)
    deps["docker_service"].get_container_host.return_value = "172.20.0.1"
    deps["docker_service"].get_container_port.return_value = 8090
    deps["endpoint_registry"].register_custom_endpoint_as_proxy.return_value = "reg-1"

    # spec without "prefix" → should set default_prefix
    promise = await svc._install_model(  # pyright: ignore[reportPrivateUsage]
        "default", "lemmatizer", InstallModelIn(spec={"hardware": "CPU"})
    )

    result = await promise.wait()
    assert result.status == "OK"


@pytest.mark.asyncio
async def test_install_model_initialises_models_for_instance(svc: CustomService, deps: dict[str, Any]) -> None:
    del svc.models["default"]
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    deps["docker_service"].get_image_warnings = AsyncMock(return_value=[])

    with pytest.raises(HTTPException):
        await svc._install_model("default", "ghost", InstallModelIn())  # pyright: ignore[reportPrivateUsage]

    assert "default" in svc.models


@pytest.mark.asyncio
async def test_uninstall_instance_when_not_installed_is_noop(svc: CustomService) -> None:
    # Branch 239->244: installed is None, skip the if block entirely
    assert svc.instances_info["default"].installed is None

    await svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert svc.instances_info["default"].installed is None


@pytest.mark.asyncio
async def test_uninstall_instance_skips_model_shared_with_other_instance(svc: CustomService) -> None:
    # Branch 241->240: is_model_installed_in_other_instance returns True,
    # so the inner if condition is False and the for loop continues without calling _uninstall_model
    mock_model = MagicMock()
    mock_model.id = "lemmatizer"
    mock_model.prefix = "lemmatizer"
    mock_model.registration_id = "reg-1"
    mock_model.docker_options = MagicMock()

    shared_installed = InstalledInfo(models={"lemmatizer": mock_model}, options=InstallServiceIn(spec={}))
    svc.instances_info["default"].installed = shared_installed

    svc.instances_info["gpu-1"] = Instance(None, None, {}, InstanceConfig())
    svc.instances_info["gpu-1"].installed = InstalledInfo(
        models={"lemmatizer": mock_model},
        options=InstallServiceIn(spec={}),
    )

    svc._uninstall_model = AsyncMock()  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage]

    await svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert svc._uninstall_model.call_count == 0  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage]


@pytest.mark.asyncio
async def test_uninstall_model_when_not_in_installed_is_noop(svc: CustomService, deps: dict[str, Any]) -> None:
    # Branch 430->436: model_id not in info.models, skip the if block and go to purge check
    deps["docker_service"].uninstall_docker = AsyncMock()
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))

    await svc._uninstall_model("default", "lemmatizer", UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert deps["docker_service"].uninstall_docker.call_count == 0
    assert deps["endpoint_registry"].unregister_custom_endpoint.call_count == 0


@pytest.mark.asyncio
async def test_resolve_custom_model_size_returns_formatted_size(svc: CustomService, deps: dict[str, Any]) -> None:
    deps["docker_service"].get_docker_image_size = AsyncMock(return_value=1024**2)

    result = await svc._resolve_custom_model_size({"image": "myimage:latest"})  # pyright: ignore[reportPrivateUsage]

    assert result == "1.0 MB"


@pytest.mark.asyncio
async def test_resolve_custom_model_size_returns_none_when_image_size_is_none(svc: CustomService, deps: dict[str, Any]) -> None:
    deps["docker_service"].get_docker_image_size = AsyncMock(return_value=None)

    result = await svc._resolve_custom_model_size({"image": "myimage:latest"})  # pyright: ignore[reportPrivateUsage]

    assert result is None


@pytest.mark.asyncio
async def test_resolve_custom_model_size_returns_none_on_exception(svc: CustomService, deps: dict[str, Any]) -> None:
    deps["docker_service"].get_docker_image_size = AsyncMock(side_effect=Exception("fail"))

    result = await svc._resolve_custom_model_size({"image": "myimage:latest"})  # pyright: ignore[reportPrivateUsage]

    assert result is None
