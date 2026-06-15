# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from fastapi import HTTPException

from server.docker import DockerImage
from server.models.models import (
    AddCustomModelIn,
    CustomModelSpecification,
    InstallModelIn,
    InstallModelOut,
    ListModelsFilters,
    ListModelsOut,
    RetrieveModelOut,
    UninstallModelIn,
)
from server.models.services import (
    InstallServiceIn,
    InstallServiceOut,
    RetrieveServiceOut,
    ServiceSpecification,
    UninstallServiceIn,
)
from server.serviceprovider import ServiceRawConfig
from server.services.base2_service import (
    Base2Service,
    CustomModel,
    InstallingInstance,
    InstallingModel,
    InstanceConfig,
    ModelConfig,
)
from server.services.base_service import BaseService
from server.utils.core import PromiseWithProgress, Stream, StreamChunk, StreamChunkProgress
from server.utils.hardware import NvidiaGpuInfo


class _BaseImpl(BaseService):
    """Minimal concrete BaseService for testing abstract method contracts."""

    instances_info: dict[str, Any] = {}

    def get_type(self) -> str:
        return "test-service"

    def get_description(self) -> str:
        return "A test service."

    def get_size(self) -> str:
        return "small"

    def get_spec(self) -> ServiceSpecification:
        return ServiceSpecification(fields=[])

    def get_custom_model_spec(self) -> CustomModelSpecification | None:
        return None

    def get_instance_install_progress(self, instance: str) -> Any:
        raise NotImplementedError

    def get_model_install_progress(self, instance: str, model: str) -> Any:
        raise NotImplementedError

    def is_installed(self, instance: str) -> bool:
        return False

    def get_installed_info(self, instance: str) -> bool:
        return False

    def get_downloaded(self) -> bool:
        return False

    async def load_service(self, config: ServiceRawConfig) -> None:
        pass

    async def install_instance(self, instance: str, options: InstallServiceIn) -> Any:
        raise NotImplementedError

    async def uninstall_instance(self, instance: str, options: UninstallServiceIn) -> None:
        pass

    async def list_models(self, input_instance: Any, filters: ListModelsFilters) -> ListModelsOut:
        return ListModelsOut(list=[])

    async def get_model(self, instance: str, model_id: str) -> RetrieveModelOut:
        raise NotImplementedError

    async def install_model(self, instance: str, model_id: str, options: InstallModelIn) -> Any:
        raise NotImplementedError

    async def uninstall_model(self, instance: str, model_id: str, options: UninstallModelIn) -> None:
        pass

    async def add_custom_model(self, instance: str, options: Any) -> Any:
        raise NotImplementedError

    async def remove_custom_model(self, instance: str, custom_model_id: Any) -> None:
        pass

    async def get_docker_logs(self, instance: str, model_id: str | None) -> str:
        return ""

    async def get_docker_compose_file(self, instance: str, model_id: str | None) -> str:
        return ""

    async def restart_docker(self, instance: str, model_id: str | None) -> None:
        pass

    async def stop_instance(self, instance: str) -> None:
        pass


class _Base2ImplWithCustom(Base2Service):  # pyright: ignore[reportMissingTypeArgument]
    """Base2Service that supports custom models."""

    _custom_store: dict[str, CustomModel]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._custom_store = {}

    def get_type(self) -> str:
        return "test-b2-custom"

    def get_description(self) -> str:
        return "Test base2 service with custom models."

    def get_size(self) -> str:
        return ""

    def get_spec(self) -> ServiceSpecification:
        return ServiceSpecification(fields=[])

    def get_custom_model_spec(self) -> CustomModelSpecification | None:
        return None

    def get_installed_info(self, instance: str) -> bool:
        return self.is_installed(instance)

    def _load_download_info(self, data: dict[str, Any]) -> dict[str, Any]:
        return data

    def _generate_instance_config(self, info: Any, custom: Any) -> InstanceConfig:
        return InstanceConfig()

    async def _install_instance(self, instance: str, options: InstallServiceIn) -> Any:
        raise NotImplementedError

    async def _uninstall_instance(self, instance: str, options: UninstallServiceIn) -> None:
        pass

    async def _install_model(self, instance: str, model_id: str, options: InstallModelIn) -> Any:
        raise NotImplementedError

    async def _uninstall_model(self, instance: str, model_id: str, options: UninstallModelIn) -> None:
        pass

    async def list_models(self, input_instance: Any, filters: ListModelsFilters) -> ListModelsOut:
        return ListModelsOut(list=[])

    async def get_model(self, instance: str, model_id: str) -> RetrieveModelOut:
        raise NotImplementedError

    async def stop_instance(self, instance: str) -> None:
        pass

    def _add_custom_model(self, instance: str, model: CustomModel) -> None:
        self._custom_store[model.id] = model

    def _remove_custom_model(self, instance: str, model: CustomModel) -> None:
        del self._custom_store[model.id]


class _Base2Impl(Base2Service):  # pyright: ignore[reportMissingTypeArgument]
    """Minimal concrete Base2Service for testing."""

    def get_type(self) -> str:
        return "test-b2"

    def get_description(self) -> str:
        return "Test base2 service."

    def get_size(self) -> str:
        return ""

    def get_spec(self) -> ServiceSpecification:
        return ServiceSpecification(fields=[])

    def get_custom_model_spec(self) -> CustomModelSpecification | None:
        return None

    def get_installed_info(self, instance: str) -> bool:
        return self.is_installed(instance)

    def _load_download_info(self, data: dict[str, Any]) -> dict[str, Any]:
        return data

    def _generate_instance_config(self, info: Any, custom: Any) -> InstanceConfig:
        return InstanceConfig()

    async def _install_instance(self, instance: str, options: InstallServiceIn) -> Any:
        raise NotImplementedError

    async def _uninstall_instance(self, instance: str, options: UninstallServiceIn) -> None:
        pass

    async def _install_model(self, instance: str, model_id: str, options: InstallModelIn) -> Any:
        raise NotImplementedError

    async def _uninstall_model(self, instance: str, model_id: str, options: UninstallModelIn) -> None:
        pass

    async def list_models(self, input_instance: Any, filters: ListModelsFilters) -> ListModelsOut:
        return ListModelsOut(list=[])

    async def get_model(self, instance: str, model_id: str) -> RetrieveModelOut:
        raise NotImplementedError

    async def stop_instance(self, instance: str) -> None:
        pass


@pytest.fixture
def base_svc() -> _BaseImpl:
    return _BaseImpl()


@pytest.fixture
def base2_deps() -> dict[str, Any]:
    return {
        "config": MagicMock(),
        "endpoint_registry": MagicMock(),
        "service_provider": MagicMock(),
        "model_downloader": MagicMock(),
        "docker_service": MagicMock(),
        "hardware": MagicMock(),
    }


@pytest.fixture
def base2_svc(base2_deps: dict[str, Any]) -> _Base2Impl:
    return _Base2Impl(**base2_deps)


@pytest.fixture
def custom_svc(base2_deps: dict[str, Any]) -> _Base2ImplWithCustom:
    base2_deps["service_provider"].save_service_config = AsyncMock()
    return _Base2ImplWithCustom(**base2_deps)


@pytest.mark.parametrize(
    ("instance", "expected"),
    [
        ("default", "test-service"),
        ("gpu-1", "test-service|gpu-1"),
    ],
)
def test_get_id(base_svc: _BaseImpl, instance: str, expected: str) -> None:
    assert base_svc.get_id(instance) == expected


@pytest.mark.parametrize(
    ("instance", "expected"),
    [
        ("default", "test-service"),
        ("gpu-1", "test-service-gpu-1"),
    ],
)
def test_get_service_id(base_svc: _BaseImpl, instance: str, expected: str) -> None:
    assert base_svc.get_service_id(instance) == expected


def test_service_has_docker_default_false(base_svc: _BaseImpl) -> None:
    assert base_svc.service_has_docker() is False


def test_is_cloud_service_default_false(base_svc: _BaseImpl) -> None:
    assert base_svc.is_cloud_service() is False


@pytest.mark.asyncio
async def test_cancel_model_install_default_raises_405(base_svc: _BaseImpl) -> None:
    with pytest.raises(HTTPException) as exc_info:
        await base_svc.cancel_model_install("default", "m1")

    assert exc_info.value.status_code == 405


def test_is_cloud_service_true_when_class_attr_set() -> None:
    class CloudSvc(_BaseImpl):
        is_cloud = True

    assert CloudSvc().is_cloud_service() is True


def test_get_info_default_instance(base_svc: _BaseImpl) -> None:
    result = base_svc.get_info("default")

    assert isinstance(result, RetrieveServiceOut)
    assert result.id == "test-service"
    assert result.type == "test-service"
    assert result.instance == "default"
    assert result.description == "A test service."
    assert result.size == "small"
    assert result.has_docker is False
    assert result.is_cloud is False
    assert result.custom_model_spec is None


def test_get_info_non_default_instance(base_svc: _BaseImpl) -> None:
    result = base_svc.get_info("gpu-1")

    assert result.id == "test-service|gpu-1"
    assert result.instance == "gpu-1"


def test_get_info_cloud_service() -> None:
    class CloudSvc(_BaseImpl):
        is_cloud = True

    result = CloudSvc().get_info("default")
    assert result.is_cloud is True


def test_base2_init_creates_default_instance(base2_svc: _Base2Impl) -> None:
    assert "default" in base2_svc.instances_info


def test_base2_init_service_downloaded_false(base2_svc: _Base2Impl) -> None:
    assert base2_svc.service_downloaded is False


def test_base2_init_models_downloaded_empty(base2_svc: _Base2Impl) -> None:
    assert base2_svc.models_downloaded == {}


def test_base2_is_installed_false_when_not_installed(base2_svc: _Base2Impl) -> None:
    assert base2_svc.is_installed("default") is False


def test_base2_is_installed_true_when_installed(base2_svc: _Base2Impl) -> None:
    base2_svc.instances_info["default"].installed = object()

    assert base2_svc.is_installed("default") is True


def test_base2_get_downloaded_reflects_flag(base2_svc: _Base2Impl) -> None:
    assert base2_svc.get_downloaded() is False

    base2_svc.service_downloaded = True

    assert base2_svc.get_downloaded() is True


def test_base2_check_instance_exists_raises_for_missing(base2_svc: _Base2Impl) -> None:
    with pytest.raises(HTTPException) as exc_info:
        base2_svc.check_instance_exists("nonexistent")

    assert exc_info.value.status_code == 404


def test_base2_check_instance_exists_passes_for_default(base2_svc: _Base2Impl) -> None:
    base2_svc.check_instance_exists("default")


@pytest.mark.parametrize(
    ("hardware_support", "expected"),
    [
        (False, False),
        ("CPU", False),
        (True, True),
    ],
)
def test_is_given_hardware_support_gpu(base2_svc: _Base2Impl, hardware_support: Any, expected: bool) -> None:
    assert base2_svc.is_given_hardware_support_gpu(hardware_support) is expected


def test_is_given_hardware_support_gpu_none_no_gpus_returns_false(base2_deps: dict[str, Any]) -> None:
    base2_deps["hardware"].gpus = []

    svc = _Base2Impl(**base2_deps)

    assert svc.is_given_hardware_support_gpu(None) is False


def test_is_given_hardware_support_gpu_string_gpu_raises_when_no_gpus(base2_deps: dict[str, Any]) -> None:
    base2_deps["hardware"].gpus = []
    svc = _Base2Impl(**base2_deps)

    with pytest.raises(HTTPException) as exc_info:
        svc.is_given_hardware_support_gpu("GPU")

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_installing_model_init_creates_task() -> None:
    value = InstallModelOut(status="OK", details="done")
    promise: PromiseWithProgress[InstallModelOut, StreamChunk] = PromiseWithProgress(value=value)
    promise.progress.close()

    installing = InstallingModel(promise=promise)

    assert installing.promise is promise
    assert installing.task is not None
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_installing_instance_init_creates_task() -> None:
    value = InstallServiceOut(status="OK")
    promise: PromiseWithProgress[InstallServiceOut, StreamChunk] = PromiseWithProgress(value=value)
    promise.progress.close()

    installing = InstallingInstance(promise=promise)

    assert installing.promise is promise
    assert installing.task is not None
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_installing_model_stores_last_chunk_from_progress() -> None:
    chunk: StreamChunk = {"type": "progress", "stage": "download", "percentage": 0.5}  # pyright: ignore[reportAssignmentType]
    value = InstallModelOut(status="OK", details="done")
    promise: PromiseWithProgress[InstallModelOut, StreamChunk] = PromiseWithProgress(value=value)
    promise.progress.emit(chunk)
    promise.progress.close()

    installing = InstallingModel(promise=promise)
    await asyncio.sleep(0)

    assert installing.last_chunk == chunk


@pytest.mark.asyncio
async def test_installing_instance_stores_last_chunk_from_progress() -> None:
    chunk: StreamChunk = {"type": "progress", "stage": "download", "percentage": 0.8}  # pyright: ignore[reportAssignmentType]
    value = InstallServiceOut(status="OK")
    promise: PromiseWithProgress[InstallServiceOut, StreamChunk] = PromiseWithProgress(value=value)
    promise.progress.emit(chunk)
    promise.progress.close()

    installing = InstallingInstance(promise=promise)
    await asyncio.sleep(0)

    assert installing.last_chunk == chunk


def test_get_instance_info_raises_404_for_missing(base2_svc: _Base2Impl) -> None:
    with pytest.raises(HTTPException) as exc_info:
        base2_svc.get_instance_info("nonexistent")

    assert exc_info.value.status_code == 404


def test_get_instance_info_returns_instance(base2_svc: _Base2Impl) -> None:
    result = base2_svc.get_instance_info("default")

    assert result is base2_svc.instances_info["default"]


def test_get_model_install_progress_raises_when_not_installing(base2_svc: _Base2Impl) -> None:
    with pytest.raises(HTTPException) as exc_info:
        base2_svc.get_model_install_progress("default", "unknown-model")

    assert exc_info.value.status_code == 404


def test_get_model_install_progress_returns_promise(base2_svc: _Base2Impl) -> None:
    mock_promise = MagicMock()
    mock_installing = MagicMock()
    mock_installing.promise = mock_promise

    base2_svc.instances_info["default"].installing_model_progress["m1"] = mock_installing

    assert base2_svc.get_model_install_progress("default", "m1") is mock_promise


@pytest.mark.asyncio
async def test_cancel_model_install_raises_when_not_installing(base2_svc: _Base2Impl) -> None:
    with pytest.raises(HTTPException) as exc_info:
        await base2_svc.cancel_model_install("default", "unknown-model")

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_cancel_model_install_cancels_tasks_and_clears_tracking(base2_svc: _Base2Impl, base2_deps: dict[str, Any]) -> None:
    base2_deps["service_provider"].save_service_config = AsyncMock()
    started = asyncio.Event()
    release = asyncio.Event()

    async def func(_stream: Stream[StreamChunk]) -> InstallModelOut:
        started.set()
        await release.wait()
        return InstallModelOut(status="OK", details="done")

    promise: PromiseWithProgress[InstallModelOut, StreamChunk] = PromiseWithProgress(func=func)

    with patch.object(base2_svc, "_install_model", new=AsyncMock(return_value=promise)):
        returned_promise = await base2_svc.install_model("default", "m1", InstallModelIn())

    await started.wait()
    installing = base2_svc.instances_info["default"].installing_model_progress["m1"]

    await base2_svc.cancel_model_install("default", "m1")

    assert "m1" not in base2_svc.instances_info["default"].installing_model_progress
    assert promise.task.cancelled()
    assert installing.task.cancelled()
    assert promise.progress._closed  # pyright: ignore[reportPrivateUsage]
    assert promise._future.cancelled()  # pyright: ignore[reportPrivateUsage]
    assert returned_promise.task.done()
    assert returned_promise._future.cancelled()  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_download_image_progress_cleared_on_cancel(base2_svc: _Base2Impl) -> None:
    image = DockerImage(name="example/image:latest", size="1GB")
    stream: Stream[StreamChunk] = Stream()

    async def fake_pull(_image: DockerImage, _stream: Stream[StreamChunk]) -> None:
        raise asyncio.CancelledError

    with patch.object(base2_svc, "_docker_pull", new=fake_pull), pytest.raises(asyncio.CancelledError):
        await base2_svc._download_image_or_set_progress(stream, image)  # pyright: ignore[reportPrivateUsage]

    assert image.name not in base2_svc.images_download_progress


def test_get_instance_install_progress_raises_when_not_installing(base2_svc: _Base2Impl) -> None:
    with pytest.raises(HTTPException) as exc_info:
        base2_svc.get_instance_install_progress("default")

    assert exc_info.value.status_code == 404


def test_get_instance_install_progress_returns_promise(base2_svc: _Base2Impl) -> None:
    mock_promise = MagicMock()
    mock_installing = MagicMock()
    mock_installing.promise = mock_promise

    base2_svc.instances_info["default"].installing = mock_installing

    assert base2_svc.get_instance_install_progress("default") is mock_promise


@pytest.mark.asyncio
async def test_load_model_happy_path(base2_svc: _Base2Impl) -> None:
    value = InstallModelOut(status="OK", details="done")
    promise: PromiseWithProgress[InstallModelOut, StreamChunk] = PromiseWithProgress(value=value)
    model = ModelConfig(model_id="m1", options=InstallModelIn())

    with patch.object(base2_svc, "_install_model", new=AsyncMock(return_value=promise)):
        await base2_svc.load_model("default", model)


@pytest.mark.asyncio
async def test_load_model_exception_is_caught(base2_svc: _Base2Impl) -> None:
    model = ModelConfig(model_id="m1", options=InstallModelIn())

    with patch.object(base2_svc, "_install_model", new=AsyncMock(side_effect=RuntimeError("fail"))):
        await base2_svc.load_model("default", model)


@pytest.mark.asyncio
async def test_load_instance_skips_when_no_options(base2_svc: _Base2Impl) -> None:
    await base2_svc.load_instance("default", InstanceConfig(options=None))

    assert base2_svc.instances_info["default"].installed is None


@pytest.mark.asyncio
async def test_load_instance_installs_service(base2_svc: _Base2Impl, base2_deps: dict[str, Any]) -> None:
    base2_deps["service_provider"].save_service_config = AsyncMock()
    installed_value: dict[str, Any] = {}
    promise: PromiseWithProgress[Any, StreamChunk] = PromiseWithProgress(value=installed_value)
    instance_data = InstanceConfig(options=InstallServiceIn(spec={}), models=[], custom=None)

    with patch.object(base2_svc, "_install_instance", new=AsyncMock(return_value=promise)):
        await base2_svc.load_instance("default", instance_data)

    assert base2_svc.instances_info["default"].installed is installed_value


@pytest.mark.asyncio
async def test_load_instance_with_custom_models(custom_svc: _Base2ImplWithCustom, base2_deps: dict[str, Any]) -> None:
    base2_deps["service_provider"].save_service_config = AsyncMock()
    installed_value: dict[str, Any] = {}
    promise: PromiseWithProgress[Any, StreamChunk] = PromiseWithProgress(value=installed_value)
    custom = CustomModel(id="cm-test", data={"name": "my-model"})
    instance_data = InstanceConfig(options=InstallServiceIn(spec={}), models=[], custom=[custom])

    with patch.object(custom_svc, "_install_instance", new=AsyncMock(return_value=promise)):
        await custom_svc.load_instance("default", instance_data)

    assert "cm-test" in custom_svc._custom_store  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_install_instance_on_error_clears_installing(base2_svc: _Base2Impl, base2_deps: dict[str, Any]) -> None:
    async def fail_func(stream: Any) -> Any:
        raise RuntimeError("install failed")

    fail_promise: PromiseWithProgress[Any, StreamChunk] = PromiseWithProgress(func=fail_func)

    with patch.object(base2_svc, "_install_instance", new=AsyncMock(return_value=fail_promise)):
        result_promise = await base2_svc.install_instance("default", InstallServiceIn(spec={}))
        with pytest.raises(RuntimeError):
            await result_promise.wait()

    assert base2_svc.instances_info["default"].installing is None


@pytest.mark.asyncio
async def test_load_service_sets_downloads_and_flag(base2_svc: _Base2Impl) -> None:
    config: ServiceRawConfig = {"downloaded": {"m1": {"key": "val"}}, "service_downloaded": True}

    await base2_svc.load_service(config)

    assert base2_svc.service_downloaded is True
    assert "m1" in base2_svc.models_downloaded


@pytest.mark.asyncio
async def test_load_service_loads_instances(base2_svc: _Base2Impl, base2_deps: dict[str, Any]) -> None:
    base2_deps["service_provider"].save_service_config = AsyncMock()
    installed_value: dict[str, Any] = {}
    promise: PromiseWithProgress[Any, StreamChunk] = PromiseWithProgress(value=installed_value)
    config: ServiceRawConfig = {
        "downloaded": None,
        "service_downloaded": False,
        "instances": {"default": {"options": {"stream": False, "ignore_warnings": False, "spec": {}}, "models": [], "custom": None}},
    }

    with patch.object(base2_svc, "_install_instance", new=AsyncMock(return_value=promise)):
        await base2_svc.load_service(config)

    assert base2_svc.instances_info["default"].installed is installed_value


@pytest.mark.asyncio
async def test_save_calls_service_provider(base2_svc: _Base2Impl, base2_deps: dict[str, Any]) -> None:
    base2_deps["service_provider"].save_service_config = AsyncMock()

    await base2_svc._save()  # pyright: ignore[reportPrivateUsage]

    assert base2_deps["service_provider"].save_service_config.call_count == 1


def test_service_config_returns_config(base2_svc: _Base2Impl) -> None:
    instances = {"default": InstanceConfig()}

    cfg = base2_svc.service_config(instances)

    assert cfg.instances == instances
    assert cfg.downloaded == {}
    assert cfg.service_downloaded is False


@pytest.mark.asyncio
async def test_install_instance_raises_if_already_installed(base2_svc: _Base2Impl) -> None:
    base2_svc.instances_info["default"].installed = object()

    with pytest.raises(HTTPException) as exc_info:
        await base2_svc.install_instance("default", InstallServiceIn(spec={}))

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_install_instance_raises_if_already_installing(base2_svc: _Base2Impl) -> None:
    base2_svc.instances_info["default"].installing = MagicMock()

    with pytest.raises(HTTPException) as exc_info:
        await base2_svc.install_instance("default", InstallServiceIn(spec={}))

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_install_instance_happy_path(base2_svc: _Base2Impl, base2_deps: dict[str, Any]) -> None:
    base2_deps["service_provider"].save_service_config = AsyncMock()
    installed_value: dict[str, Any] = {}
    promise: PromiseWithProgress[Any, StreamChunk] = PromiseWithProgress(value=installed_value)

    with patch.object(base2_svc, "_install_instance", new=AsyncMock(return_value=promise)):
        result = await base2_svc.install_instance("default", InstallServiceIn(spec={}))
        await result.wait()

    assert base2_svc.instances_info["default"].installed is installed_value
    assert base2_svc.instances_info["default"].installing is None


@pytest.mark.asyncio
async def test_uninstall_instance_calls_uninstall_and_save(base2_svc: _Base2Impl, base2_deps: dict[str, Any]) -> None:
    base2_deps["service_provider"].save_service_config = AsyncMock()
    options = UninstallServiceIn(purge=False)

    with patch.object(base2_svc, "_uninstall_instance", new=AsyncMock()) as mock_uninstall:
        await base2_svc.uninstall_instance("default", options)

    assert mock_uninstall.call_count == 1
    assert mock_uninstall.call_args == call("default", options)
    assert base2_deps["service_provider"].save_service_config.call_count == 1


@pytest.mark.asyncio
async def test_add_custom_model_base_raises_400(base2_svc: _Base2Impl) -> None:
    with pytest.raises(HTTPException) as exc_info:
        await base2_svc.add_custom_model("default", AddCustomModelIn(spec={"name": "test"}))

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_add_custom_model_happy_path(custom_svc: _Base2ImplWithCustom) -> None:
    model_id = await custom_svc.add_custom_model("default", AddCustomModelIn(spec={"name": "test"}))

    assert model_id in custom_svc._custom_store  # pyright: ignore[reportPrivateUsage]
    assert custom_svc.instances_info["default"].config.custom is not None


@pytest.mark.asyncio
async def test_add_custom_model_sets_size_unknown_when_no_size_and_no_resolver(custom_svc: _Base2ImplWithCustom) -> None:
    # When spec has no "size" and _resolve_custom_model_size returns None → "unknown"
    model_id = await custom_svc.add_custom_model("default", AddCustomModelIn(spec={"name": "test"}))

    stored = custom_svc._custom_store[model_id]  # pyright: ignore[reportPrivateUsage]

    assert stored.data["size"] == "unknown"


@pytest.mark.asyncio
async def test_add_custom_model_uses_resolved_size(custom_svc: _Base2ImplWithCustom) -> None:
    # When _resolve_custom_model_size returns a string, it is stored
    with patch.object(custom_svc, "_resolve_custom_model_size", new=AsyncMock(return_value="1.5 GB")):
        model_id = await custom_svc.add_custom_model("default", AddCustomModelIn(spec={"name": "test"}))

    stored = custom_svc._custom_store[model_id]  # pyright: ignore[reportPrivateUsage]

    assert stored.data["size"] == "1.5 GB"


@pytest.mark.asyncio
async def test_add_custom_model_preserves_existing_size(custom_svc: _Base2ImplWithCustom) -> None:
    # When spec already has "size", _resolve_custom_model_size is NOT called
    with patch.object(custom_svc, "_resolve_custom_model_size", new=AsyncMock(return_value="99 GB")) as mock_resolve:
        model_id = await custom_svc.add_custom_model("default", AddCustomModelIn(spec={"name": "test", "size": "2 GB"}))

    stored = custom_svc._custom_store[model_id]  # pyright: ignore[reportPrivateUsage]

    assert stored.data["size"] == "2 GB"
    mock_resolve.assert_not_called()


@pytest.mark.asyncio
async def test_remove_custom_model_base_raises_400(base2_svc: _Base2Impl) -> None:
    base2_svc.instances_info["default"].config.custom = [CustomModel(id="cm1", data={})]

    with pytest.raises(HTTPException) as exc_info:
        await base2_svc.remove_custom_model("default", "cm1")

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_remove_custom_model_happy_path(custom_svc: _Base2ImplWithCustom) -> None:
    model_id = await custom_svc.add_custom_model("default", AddCustomModelIn(spec={"x": 1}))
    await custom_svc.remove_custom_model("default", model_id)
    assert model_id not in custom_svc._custom_store  # pyright: ignore[reportPrivateUsage]

    config_custom = custom_svc.instances_info["default"].config.custom or []

    assert all(m.id != model_id for m in config_custom)


@pytest.mark.asyncio
async def test_install_model_happy_path(base2_svc: _Base2Impl, base2_deps: dict[str, Any]) -> None:
    base2_deps["service_provider"].save_service_config = AsyncMock()
    model_out = InstallModelOut(status="OK", details="done")
    promise: PromiseWithProgress[InstallModelOut, StreamChunk] = PromiseWithProgress(value=model_out)

    with patch.object(base2_svc, "_install_model", new=AsyncMock(return_value=promise)):
        result_promise = await base2_svc.install_model("default", "m1", InstallModelIn())
        result = await result_promise.wait()

    assert result.status == "OK"
    assert "m1" not in base2_svc.instances_info["default"].installing_model_progress


@pytest.mark.asyncio
async def test_install_model_on_error_cleans_up(base2_svc: _Base2Impl) -> None:
    async def fail_func(stream: Any) -> Any:
        raise RuntimeError("fail")

    fail_promise: PromiseWithProgress[InstallModelOut, StreamChunk] = PromiseWithProgress(func=fail_func)

    with patch.object(base2_svc, "_install_model", new=AsyncMock(return_value=fail_promise)):
        result_promise = await base2_svc.install_model("default", "m1", InstallModelIn())
        with pytest.raises(RuntimeError):
            await result_promise.wait()

    assert "m1" not in base2_svc.instances_info["default"].installing_model_progress


@pytest.mark.asyncio
async def test_uninstall_model_calls_uninstall_and_save(base2_svc: _Base2Impl, base2_deps: dict[str, Any]) -> None:
    base2_deps["service_provider"].save_service_config = AsyncMock()
    options = UninstallModelIn()

    with patch.object(base2_svc, "_uninstall_model", new=AsyncMock()) as mock_uninstall:
        await base2_svc.uninstall_model("default", "m1", options)

    assert mock_uninstall.call_count == 1
    assert mock_uninstall.call_args == call("default", "m1", options)


@pytest.mark.asyncio
async def test_get_docker_logs_calls_docker_service(base2_svc: _Base2Impl, base2_deps: dict[str, Any]) -> None:
    base2_deps["docker_service"].get_docker_compose_logs = AsyncMock(return_value="logs")

    with patch.object(base2_svc, "get_docker_compose_file_path", return_value=Path("/tmp/dc.yml")):
        result = await base2_svc.get_docker_logs("default", None)

    assert result == "logs"


def test_get_model_installed_info_returns_false_when_not_in_progress(base2_svc: _Base2Impl) -> None:
    assert base2_svc._get_model_installed_info("default", "m1") is False  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize(
    ("last_chunk", "expected_stage", "expected_value"),
    [
        ({"type": "progress", "stage": "download", "value": 0.5}, "download", 0.5),
        ({"type": "finish"}, "install", 1),
        (None, "download", 0),
    ],
)
def test_get_model_installed_info_chunk(
    base2_svc: _Base2Impl,
    last_chunk: dict[str, object] | None,
    expected_stage: str,
    expected_value: float,
) -> None:
    mock_installing = MagicMock()
    mock_installing.last_chunk = last_chunk
    base2_svc.instances_info["default"].installing_model_progress["m1"] = mock_installing

    result = base2_svc._get_model_installed_info("default", "m1")  # pyright: ignore[reportPrivateUsage]

    assert result.stage == expected_stage  # type: ignore[union-attr]
    assert result.value == expected_value  # type: ignore[union-attr]


def test_get_service_installed_info_returns_false_when_not_installing(base2_svc: _Base2Impl) -> None:
    assert base2_svc._get_service_installed_info("default") is False  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize(
    ("last_chunk", "expected_stage", "expected_value"),
    [
        ({"type": "progress", "stage": "download", "value": 0.3}, "download", 0.3),
        ({"type": "finish"}, "install", 1),
        (None, "download", 0),
    ],
)
def test_get_service_installed_info_chunk(
    base2_svc: _Base2Impl,
    last_chunk: dict[str, object] | None,
    expected_stage: str,
    expected_value: float,
) -> None:
    mock_installing = MagicMock()
    mock_installing.last_chunk = last_chunk
    base2_svc.instances_info["default"].installing = mock_installing

    result = base2_svc._get_service_installed_info("default")  # pyright: ignore[reportPrivateUsage]

    assert result.stage == expected_stage  # type: ignore[union-attr]
    assert result.value == expected_value  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_get_docker_compose_file_reads_file(base2_svc: _Base2Impl) -> None:
    with (
        patch.object(base2_svc, "get_docker_compose_file_path", return_value=Path("/tmp/dc.yml")),
        patch("server.services.base2_service.Utils.read_file", new=AsyncMock(return_value="content")),
    ):
        result = await base2_svc.get_docker_compose_file("default", None)

    assert result == "content"


@pytest.mark.asyncio
async def test_restart_docker_calls_docker_service(base2_svc: _Base2Impl, base2_deps: dict[str, Any]) -> None:
    base2_deps["docker_service"].restart_docker_compose = AsyncMock()

    with patch.object(base2_svc, "get_docker_compose_file_path", return_value=Path("/tmp/dc.yml")):
        await base2_svc.restart_docker("default", None)

    assert base2_deps["docker_service"].restart_docker_compose.call_count == 1
    assert base2_deps["docker_service"].restart_docker_compose.call_args == call(Path("/tmp/dc.yml"))


def test_get_docker_compose_file_path_raises_when_not_installed(base2_svc: _Base2Impl) -> None:
    with pytest.raises(HTTPException) as exc_info:
        base2_svc.get_docker_compose_file_path("default", None)

    assert exc_info.value.status_code == 400


def test_get_docker_compose_file_path_raises_no_docker_when_installed(base2_svc: _Base2Impl) -> None:
    base2_svc.instances_info["default"].installed = object()

    with pytest.raises(HTTPException) as exc_info:
        base2_svc.get_docker_compose_file_path("default", None)

    assert exc_info.value.status_code == 400


def test_get_service_dir_creates_dir_if_not_exists(base2_svc: _Base2Impl, base2_deps: dict[str, Any], tmp_path: Path) -> None:
    base2_deps["config"].get_storage_services_dir.return_value = tmp_path

    result = base2_svc._get_service_dir("my-svc")  # pyright: ignore[reportPrivateUsage]

    assert result.is_dir()
    assert result == tmp_path / "my-svc"


@pytest.mark.asyncio
async def test_clear_working_dir_removes_directory(base2_svc: _Base2Impl, tmp_path: Path) -> None:
    working_dir = tmp_path / "test-b2"
    working_dir.mkdir()

    with patch.object(base2_svc, "_get_working_dir", return_value=working_dir):  # pyright: ignore[reportPrivateUsage]
        await base2_svc._clear_working_dir()  # pyright: ignore[reportPrivateUsage]

    assert not working_dir.exists()


def test_get_instance_installed_info_raises_when_not_installed(base2_svc: _Base2Impl) -> None:
    with pytest.raises(HTTPException) as exc_info:
        base2_svc.get_instance_installed_info("default")

    assert exc_info.value.status_code == 400


def test_get_instance_installed_info_returns_value_when_installed(base2_svc: _Base2Impl) -> None:
    sentinel = object()

    base2_svc.instances_info["default"].installed = sentinel

    assert base2_svc.get_instance_installed_info("default") is sentinel


def test_get_hugging_face_token(base2_svc: _Base2Impl, base2_deps: dict[str, Any]) -> None:
    base2_deps["config"].hugging_face_token = "hf-abc"

    assert base2_svc.get_hugging_face_token() == "hf-abc"


def test_get_civitai_token(base2_svc: _Base2Impl, base2_deps: dict[str, Any]) -> None:
    base2_deps["config"].civitai_token = "civ-xyz"

    assert base2_svc.get_civitai_token() == "civ-xyz"


def test_has_gpu_for_spec_true(base2_svc: _Base2Impl, base2_deps: dict[str, Any]) -> None:
    base2_deps["docker_service"].has_gpu_support = True

    assert base2_svc._has_gpu_for_spec() == "true"  # pyright: ignore[reportPrivateUsage]


def test_has_gpu_for_spec_false(base2_svc: _Base2Impl, base2_deps: dict[str, Any]) -> None:
    base2_deps["docker_service"].has_gpu_support = False

    assert base2_svc._has_gpu_for_spec() == "false"  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_download_image_or_set_progress_forwards_existing_stream(base2_svc: _Base2Impl) -> None:
    existing_stream: Stream[StreamChunk] = Stream()
    chunk = StreamChunkProgress(type="progress", stage="download", value=0.5, data={})
    existing_stream.emit(chunk)
    existing_stream.close()
    image = DockerImage(name="test-image", size="1GB")
    base2_svc.images_download_progress[image.name] = existing_stream  # type: ignore[assignment]
    output_stream = MagicMock()

    await base2_svc._download_image_or_set_progress(output_stream, image)  # pyright: ignore[reportPrivateUsage]

    assert output_stream.emit.call_count == 1
    assert output_stream.emit.call_args == call(chunk)


@pytest.mark.asyncio
async def test_download_image_or_set_progress_breaks_on_non_download_chunk(base2_svc: _Base2Impl) -> None:
    existing_stream: Stream[StreamChunk] = Stream()
    finish_chunk: StreamChunk = {"type": "finish", "status": "ok"}  # type: ignore[assignment]
    existing_stream.emit(finish_chunk)
    existing_stream.close()
    image = DockerImage(name="test-image", size="1GB")
    base2_svc.images_download_progress[image.name] = existing_stream  # type: ignore[assignment]
    output_stream = MagicMock()

    await base2_svc._download_image_or_set_progress(output_stream, image)  # pyright: ignore[reportPrivateUsage]

    assert output_stream.emit.call_count == 0


@pytest.mark.asyncio
async def test_docker_pull_emits_progress_when_image_not_present(base2_svc: _Base2Impl, base2_deps: dict[str, Any]) -> None:
    image = DockerImage(name="my-image", size="1GB")
    stream = MagicMock()
    base2_deps["docker_service"].is_docker_image_pulled = AsyncMock(return_value=False)
    base2_deps["docker_service"].get_docker_image_size = AsyncMock(return_value=1024)

    async def mock_docker_pull(*args: Any):  # type: ignore[misc]
        yield 0.5

    base2_deps["docker_service"].docker_pull = mock_docker_pull

    await base2_svc._docker_pull(image, stream)  # pyright: ignore[reportPrivateUsage]

    assert stream.emit.call_count >= 3


@pytest.mark.asyncio
async def test_stop_docker_logs_error_on_exception(base2_svc: _Base2Impl, base2_deps: dict[str, Any]) -> None:
    base2_deps["docker_service"].stop_docker = AsyncMock(side_effect=RuntimeError("boom"))
    docker_options = MagicMock()

    await base2_svc._stop_docker(docker_options)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_verify_docker_image_raises_when_warnings_not_ignored(base2_svc: _Base2Impl, base2_deps: dict[str, Any]) -> None:
    base2_deps["docker_service"].get_image_warnings = AsyncMock(return_value=["w1"])

    with pytest.raises(HTTPException) as exc_info:
        await base2_svc._verify_docker_image("img", False)  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_verify_docker_image_passes_when_warnings_ignored(base2_svc: _Base2Impl, base2_deps: dict[str, Any]) -> None:
    base2_deps["docker_service"].get_image_warnings = AsyncMock(return_value=["w1"])

    await base2_svc._verify_docker_image("img", True)  # pyright: ignore[reportPrivateUsage]


def test_get_specified_hardware_parts_matches_gpu_by_name(base2_deps: dict[str, Any]) -> None:
    gpu = NvidiaGpuInfo(name="RTX 3090", vram="24 GB", id=0)
    hw = MagicMock()
    hw.gpus = [gpu]
    base2_deps["hardware"] = hw
    svc = _Base2Impl(**base2_deps)

    result = svc.get_specified_hardware_parts(f"GPU | {gpu.long_name}")

    assert gpu in result


def test_get_specified_hardware_parts_no_match_returns_empty(base2_deps: dict[str, Any]) -> None:
    gpu = NvidiaGpuInfo(name="RTX 3090", vram="24 GB", id=0)
    hw = MagicMock()
    hw.gpus = [gpu]
    base2_deps["hardware"] = hw
    svc = _Base2Impl(**base2_deps)

    result = svc.get_specified_hardware_parts("GPU | Unknown GPU | 0")

    assert list(result) == []


def test_add_hardware_field_to_spec_no_cpu_without_avx512(base2_deps: dict[str, Any]) -> None:
    hw = MagicMock()
    hw.gpus = []
    hw.cpu.avx512 = False
    base2_deps["hardware"] = hw
    svc = _Base2Impl(**base2_deps)

    fields = svc.add_hardware_field_to_spec(add_cpu_option_only_on_avx512_support=True)

    hw_field = next(f for f in fields if f.name == "hardware")
    assert "CPU" not in [getattr(v, "value", v) for v in hw_field.values]  # pyright: ignore[reportOptionalIterable]


def test_add_hardware_field_to_spec_single_gpu(base2_deps: dict[str, Any]) -> None:
    gpu = NvidiaGpuInfo(name="RTX 3090", vram="24 GB", id=0)
    hw = MagicMock()
    hw.gpus = [gpu]
    hw.cpu.avx512 = True
    base2_deps["hardware"] = hw
    svc = _Base2Impl(**base2_deps)

    fields = svc.add_hardware_field_to_spec()

    hw_field = next(f for f in fields if f.name == "hardware")
    assert hw_field.default == "GPU"


def test_add_hardware_field_to_spec_multiple_gpus(base2_deps: dict[str, Any]) -> None:
    gpu1 = NvidiaGpuInfo(name="RTX 3090", vram="24 GB", id=0)
    gpu2 = NvidiaGpuInfo(name="RTX 4090", vram="24 GB", id=1)
    hw = MagicMock()
    hw.gpus = [gpu1, gpu2]
    hw.cpu.avx512 = True
    base2_deps["hardware"] = hw
    svc = _Base2Impl(**base2_deps)

    fields = svc.add_hardware_field_to_spec()

    hw_field = next(f for f in fields if f.name == "hardware")
    assert hw_field.default == "GPUs"


def test_add_hardware_field_to_model_spec_no_cpu_without_avx512(base2_deps: dict[str, Any]) -> None:
    hw = MagicMock()
    hw.gpus = []
    hw.cpu.avx512 = False
    base2_deps["hardware"] = hw
    svc = _Base2Impl(**base2_deps)

    fields = svc.add_hardware_field_to_model_spec(add_cpu_option_only_on_avx512_support=True)

    hw_field = next(f for f in fields if f.name == "hardware")
    assert "CPU" not in [getattr(v, "value", v) for v in hw_field.values]  # pyright: ignore[reportOptionalIterable]


def test_add_hardware_field_to_model_spec_multiple_gpus(base2_deps: dict[str, Any]) -> None:
    gpu1 = NvidiaGpuInfo(name="RTX 3090", vram="24 GB", id=0)
    gpu2 = NvidiaGpuInfo(name="RTX 4090", vram="24 GB", id=1)
    hw = MagicMock()
    hw.gpus = [gpu1, gpu2]
    hw.cpu.avx512 = True
    base2_deps["hardware"] = hw
    svc = _Base2Impl(**base2_deps)

    fields = svc.add_hardware_field_to_model_spec()

    hw_field = next(f for f in fields if f.name == "hardware")
    assert hw_field.default == "GPUs"


def test_load_download_info_abstract_body_returns_none(base2_svc: _Base2Impl) -> None:
    result = Base2Service._load_download_info(base2_svc, {})  # type: ignore[arg-type] # pyright: ignore[reportPrivateUsage]
    assert result is None


@pytest.mark.asyncio
async def test_install_instance_new_instance_skips_installed_check(base2_svc: _Base2Impl, base2_deps: dict[str, Any]) -> None:
    base2_deps["service_provider"].save_service_config = AsyncMock()
    installed_value: dict[str, Any] = {}
    promise: PromiseWithProgress[Any, StreamChunk] = PromiseWithProgress(value=installed_value)

    with patch.object(base2_svc, "_install_instance", new=AsyncMock(return_value=promise)):
        result = await base2_svc.install_instance("brand-new-instance", InstallServiceIn(spec={}))
        await result.wait()

    assert base2_svc.instances_info["brand-new-instance"].installed is installed_value


@pytest.mark.asyncio
async def test_add_custom_model_with_existing_custom_list(custom_svc: _Base2ImplWithCustom) -> None:
    custom_svc.instances_info["default"].config.custom = []

    model_id = await custom_svc.add_custom_model("default", AddCustomModelIn(spec={"name": "test"}))

    assert model_id in custom_svc._custom_store  # pyright: ignore[reportPrivateUsage]
    assert len(custom_svc.instances_info["default"].config.custom) == 1  # pyright: ignore[reportArgumentType]


@pytest.mark.asyncio
async def test_clear_working_dir_no_op_when_not_exists(base2_svc: _Base2Impl, tmp_path: Path) -> None:
    non_existent = tmp_path / "does-not-exist"

    with patch.object(base2_svc, "_get_working_dir", return_value=non_existent):  # pyright: ignore[reportPrivateUsage]
        await base2_svc._clear_working_dir()  # pyright: ignore[reportPrivateUsage]

    assert not non_existent.exists()


@pytest.mark.asyncio
async def test_base_service_get_loaded_model_info_returns_none(base_svc: _BaseImpl) -> None:
    result = await base_svc.get_loaded_model_info("default")

    assert result is None
