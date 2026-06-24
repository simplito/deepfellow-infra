# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest import mock
from unittest.mock import AsyncMock, MagicMock, call, patch

import aiohttp
import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from server.models.api import McpToolInfo
from server.models.models import InstallModelIn, ListModelsFilters, McpHealthCheckResult, UninstallModelIn
from server.models.services import InstallServiceIn, UninstallServiceIn
from server.services.base2_service import CustomModel, Instance, InstanceConfig
from server.services.mcp_service import (
    DownloadedInfo,
    InstalledInfo,
    McpModelOptions,
    McpService,
    ModelInstalledInfo,
    SrvMcpCustomModel,
    SrvMcpModel,
    _dispatch_sse_event,  # pyright: ignore[reportPrivateUsage]
    _fetch_tools_from_mcp_endpoint,  # pyright: ignore[reportPrivateUsage]
    _fetch_tools_from_sse_endpoint,  # pyright: ignore[reportPrivateUsage]
    _parse_mcp_tools,  # pyright: ignore[reportPrivateUsage]
    _read_first_sse_json,  # pyright: ignore[reportPrivateUsage]
    _run_sse_reader,  # pyright: ignore[reportPrivateUsage]
    _sse_rpc,  # pyright: ignore[reportPrivateUsage]
    _SseState,  # pyright: ignore[reportPrivateUsage]
)


class _AsyncLineIter:
    def __init__(self, lines: list[bytes]) -> None:
        self._lines = lines
        self._index = 0

    def __aiter__(self) -> "_AsyncLineIter":
        return self

    async def __anext__(self) -> bytes:
        if self._index >= len(self._lines):
            raise StopAsyncIteration
        await asyncio.sleep(0)
        val = self._lines[self._index]
        self._index += 1
        return val


def _make_acm(value: Any) -> MagicMock:
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=value)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


def _make_mock_resp(
    status: int = 200,
    content_type: str = "application/json",
    json_data: Any = None,
    session_id: str | None = None,
    sse_lines: list[bytes] | None = None,
) -> MagicMock:
    resp = MagicMock()
    resp.status = status
    _h: dict[str, str] = {"Content-Type": content_type}
    if session_id:
        _h["Mcp-Session-Id"] = session_id
    resp.headers = _h
    resp.json = AsyncMock(return_value=json_data)
    resp.content = _AsyncLineIter(sse_lines or [])
    return resp


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
        "hardware": MagicMock(gpus=[], nvidia_gpus=[]),
    }


@pytest.fixture
def svc(deps: dict[str, Any]) -> McpService:
    return McpService(**deps)


def test_get_type(svc: McpService) -> None:
    assert svc.get_type() == "mcp"


def test_get_description_not_empty(svc: McpService) -> None:
    assert svc.get_description()


def test_service_has_docker(svc: McpService) -> None:
    assert svc.service_has_docker() is False


def test_is_not_cloud_service(svc: McpService) -> None:
    assert svc.is_cloud_service() is False


def test_get_size_empty_string(svc: McpService) -> None:
    assert svc.get_size() == ""


def test_get_spec_has_no_fields(svc: McpService) -> None:
    spec = svc.get_spec()
    assert spec.fields == []


def test_get_default_model_spec_has_prefix_field(svc: McpService) -> None:
    spec = svc.get_default_model_spec("my-prefix")

    field_names = [f.name for f in spec.fields]
    assert "prefix" in field_names


def test_get_default_model_spec_prefix_is_required(svc: McpService) -> None:
    spec = svc.get_default_model_spec("my-prefix")

    prefix_field = next(f for f in spec.fields if f.name == "prefix")
    assert prefix_field.required is True


def test_get_default_model_spec_prefix_default_value(svc: McpService) -> None:
    spec = svc.get_default_model_spec("my-prefix")

    prefix_field = next(f for f in spec.fields if f.name == "prefix")
    assert prefix_field.default == "my-prefix"


def test_get_default_model_spec_has_envs_and_headers(svc: McpService) -> None:
    spec = svc.get_default_model_spec("px")

    field_names = [f.name for f in spec.fields]
    assert "envs" in field_names
    assert "headers" in field_names


def test_get_custom_model_spec_not_none(svc: McpService) -> None:
    assert svc.get_custom_model_spec() is not None


def test_get_custom_model_spec_has_required_fields(svc: McpService) -> None:
    spec = svc.get_custom_model_spec()

    assert spec is not None
    field_names = {f.name for f in spec.fields}
    assert {"id", "default_prefix", "size", "image", "image_port"} <= field_names


def test_srv_mcp_custom_model_valid() -> None:
    m = SrvMcpCustomModel(id="test-id", default_prefix="my-prefix", size="1GB", image="company/image", image_port=8000)

    assert m.id == "test-id"
    assert m.default_prefix == "my-prefix"


@pytest.mark.parametrize("bad_prefix", ["bad prefix", "bad@prefix", "bad.prefix", ""])
def test_srv_mcp_custom_model_invalid_prefix_raises(bad_prefix: str) -> None:
    with pytest.raises(ValidationError):
        SrvMcpCustomModel(id="x", default_prefix=bad_prefix, size="1GB", image="img", image_port=8000)


@pytest.mark.parametrize("good_prefix", ["my-prefix", "MyPrefix", "prefix_123", "abc"])
def test_srv_mcp_custom_model_valid_prefix(good_prefix: str) -> None:
    m = SrvMcpCustomModel(id="x", default_prefix=good_prefix, size="1GB", image="img", image_port=8000)

    assert m.default_prefix == good_prefix


def test_mcp_model_options_valid_prefix() -> None:
    opts = McpModelOptions(prefix="my-prefix")

    assert opts.prefix == "my-prefix"


@pytest.mark.parametrize("bad_prefix", ["bad prefix", "bad@", "dot.dot"])
def test_mcp_model_options_invalid_prefix_raises(bad_prefix: str) -> None:
    with pytest.raises(ValidationError):
        McpModelOptions(prefix=bad_prefix)


def test_mcp_model_options_empty_string_envs_becomes_dict() -> None:
    opts = McpModelOptions(prefix="px", envs="")  # type: ignore[arg-type]

    assert opts.envs == {}


def test_mcp_model_options_empty_string_headers_becomes_dict() -> None:
    opts = McpModelOptions(prefix="px", headers="")  # type: ignore[arg-type]

    assert opts.headers == {}


def _make_custom(custom_id: str = "uuid-1", model_id: str = "my-mcp", prefix: str = "my-prefix") -> CustomModel:
    return CustomModel(
        id=custom_id,
        data={
            "id": model_id,
            "default_prefix": prefix,
            "size": "500MB",
            "image": "company/mcp-server",
            "image_port": 8080,
        },
    )


def test_add_custom_model_stores_model(svc: McpService) -> None:
    svc._add_custom_model("default", _make_custom())  # pyright: ignore[reportPrivateUsage]

    assert "my-mcp" in svc.models["default"]


def test_add_custom_model_stores_custom_id(svc: McpService) -> None:
    svc._add_custom_model("default", _make_custom(custom_id="uuid-42"))  # pyright: ignore[reportPrivateUsage]

    assert svc.models["default"]["my-mcp"].custom == "uuid-42"


def test_add_custom_model_sets_default_prefix(svc: McpService) -> None:
    svc._add_custom_model("default", _make_custom(prefix="cool-mcp"))  # pyright: ignore[reportPrivateUsage]

    assert svc.models["default"]["my-mcp"].default_prefix == "cool-mcp"


def test_add_custom_model_duplicate_raises_http_400(svc: McpService) -> None:
    svc._add_custom_model("default", _make_custom())  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(HTTPException) as exc_info:
        svc._add_custom_model("default", _make_custom())  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400


def test_add_custom_model_invalid_prefix_raises_http_400(svc: McpService) -> None:
    custom = CustomModel(
        id="bad-uuid",
        data={"id": "bad-model", "default_prefix": "bad prefix!", "size": "1GB", "image": "img", "image_port": 8000},
    )
    with pytest.raises(HTTPException) as exc_info:
        svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400


def test_add_custom_model_model_type_is_mcp(svc: McpService) -> None:
    svc._add_custom_model("default", _make_custom())  # pyright: ignore[reportPrivateUsage]

    assert svc.models["default"]["my-mcp"].model_type == "mcp"


def test_mcp_model_options_dict_envs_preserved() -> None:
    opts = McpModelOptions(prefix="px", envs={"KEY": "val"})

    assert opts.envs == {"KEY": "val"}


def test_mcp_model_options_dict_headers_preserved() -> None:
    opts = McpModelOptions(prefix="px", headers={"X-Auth": "token"})

    assert opts.headers == {"X-Auth": "token"}


def test_model_installed_info_get_info_returns_model_info() -> None:
    options = InstallModelIn()
    info = ModelInstalledInfo(
        id="m1",
        options=options,
        docker_options=MagicMock(),
        container_host="172.20.0.2",
        container_port=3000,
        docker_exposed_port=12345,
        registration_id="reg-id",
        prefix="open-websearch",
        base_url="http://172.20.0.2:3000",
        headers={},
        envs={},
    )

    result = info.get_info()

    assert result.registration_id == "reg-id"
    assert result.spec is options.spec


@pytest.mark.asyncio
async def test_stop_instance_no_installed_returns_early(svc: McpService) -> None:
    with patch.object(svc, "_stop_dockers_parallel", new=AsyncMock()) as mock_stop:
        await svc.stop_instance("default")

    assert mock_stop.call_count == 0


@pytest.mark.asyncio
async def test_stop_instance_with_models_calls_stop_parallel(svc: McpService) -> None:
    installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    mock_model = MagicMock()
    installed.models["m1"] = mock_model
    svc.instances_info["default"].installed = installed

    with patch.object(svc, "_stop_dockers_parallel", new=AsyncMock()) as mock_stop:
        await svc.stop_instance("default")

    assert mock_stop.call_count == 1
    assert mock_stop.call_args == call([mock_model.docker_options])


def test_get_installed_info_when_not_installed_returns_false(svc: McpService) -> None:
    result = svc.get_installed_info("default")

    assert result is False


def test_get_installed_info_when_installed_returns_spec(svc: McpService) -> None:
    options = InstallServiceIn(spec={"key": "val"})
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=options)

    result = svc.get_installed_info("default")

    assert result == {"key": "val"}


def test_generate_instance_config_with_none_info(svc: McpService) -> None:
    result = svc._generate_instance_config(None, None)  # pyright: ignore[reportPrivateUsage]

    assert result.options is None
    assert result.models == []
    assert result.custom is None


def test_generate_instance_config_with_info(svc: McpService) -> None:
    options = InstallServiceIn(spec={})
    installed = InstalledInfo(models={}, options=options)

    result = svc._generate_instance_config(installed, None)  # pyright: ignore[reportPrivateUsage]

    assert result.options is options


def test_load_download_info_creates_downloaded_info(svc: McpService) -> None:
    result = svc._load_download_info({"image": "my-image"})  # pyright: ignore[reportPrivateUsage]

    assert isinstance(result, DownloadedInfo)
    assert result.image == "my-image"


@pytest.mark.asyncio
async def test_install_instance_sets_service_downloaded(svc: McpService) -> None:
    promise = await svc._install_instance("default", InstallServiceIn(spec={}))  # pyright: ignore[reportPrivateUsage]

    result = await promise.wait()
    assert svc.service_downloaded is True
    assert isinstance(result, InstalledInfo)


@pytest.mark.asyncio
async def test_install_instance_loads_default_models_for_new_instance(svc: McpService) -> None:
    svc.instances_info["gpu-1"] = Instance(None, None, {}, InstanceConfig())
    assert "gpu-1" not in svc.models

    promise = await svc._install_instance("gpu-1", InstallServiceIn(spec={}))  # pyright: ignore[reportPrivateUsage]

    await promise.wait()
    assert "gpu-1" in svc.models


@pytest.mark.asyncio
async def test_uninstall_instance_clears_installed(svc: McpService) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))

    await svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert svc.instances_info["default"].installed is None


@pytest.mark.asyncio
async def test_uninstall_instance_purge_clears_service_downloaded(svc: McpService) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    svc.service_downloaded = True

    with patch.object(svc, "_clear_working_dir", new=AsyncMock()):
        await svc._uninstall_instance("default", UninstallServiceIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert svc.service_downloaded is False
    assert svc.models_downloaded == {}


@pytest.mark.asyncio
async def test_uninstall_instance_with_models_uninstalls_them(svc: McpService) -> None:
    installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    mock_model = MagicMock()
    mock_model.id = "open-websearch"
    installed.models["open-websearch"] = mock_model
    svc.instances_info["default"].installed = installed

    with (
        patch.object(svc, "_uninstall_model", new=AsyncMock()) as mock_uninstall,
        patch.object(svc, "is_model_installed_in_other_instance", return_value=False),
    ):
        await svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert mock_uninstall.call_count == 1


@pytest.mark.asyncio
async def test_uninstall_instance_not_installed_still_clears(svc: McpService) -> None:
    # installed is None → branch 311->316: skip model loop, go straight to clearing
    assert svc.instances_info["default"].installed is None

    await svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert svc.instances_info["default"].installed is None


@pytest.mark.asyncio
async def test_uninstall_instance_skips_model_installed_in_other_instance(svc: McpService) -> None:
    # branch 313->312: is_model_installed_in_other_instance returns True → _uninstall_model not called
    installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    mock_model = MagicMock()
    mock_model.id = "open-websearch"
    installed.models["open-websearch"] = mock_model
    svc.instances_info["default"].installed = installed

    with (
        patch.object(svc, "_uninstall_model", new=AsyncMock()) as mock_uninstall,
        patch.object(svc, "is_model_installed_in_other_instance", return_value=True),
    ):
        await svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert mock_uninstall.call_count == 0


@pytest.mark.asyncio
async def test_uninstall_non_default_instance_purge_removes_entry(svc: McpService) -> None:
    svc.instances_info["gpu-1"] = Instance(None, None, {}, InstanceConfig())
    svc.instances_info["gpu-1"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    svc.models["gpu-1"] = {}

    with patch.object(svc, "_clear_working_dir", new=AsyncMock()):
        await svc._uninstall_instance("gpu-1", UninstallServiceIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert "gpu-1" not in svc.instances_info


def test_get_docker_compose_file_path_no_model_id_raises_400(svc: McpService) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))

    with pytest.raises(HTTPException) as exc_info:
        svc.get_docker_compose_file_path("default", None)

    assert exc_info.value.status_code == 400


def test_get_docker_compose_file_path_model_not_installed_raises_400(svc: McpService) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))

    with pytest.raises(HTTPException) as exc_info:
        svc.get_docker_compose_file_path("default", "nonexistent")

    assert exc_info.value.status_code == 400


def test_get_docker_compose_file_path_returns_path(svc: McpService, deps: dict[str, Any]) -> None:
    installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    mock_model = MagicMock()
    mock_model.docker_options.name = "my-model"
    installed.models["my-model"] = mock_model
    svc.instances_info["default"].installed = installed
    deps["docker_service"].get_docker_compose_file_path.return_value = Path("/tmp/dc.yml")

    result = svc.get_docker_compose_file_path("default", "my-model")

    assert result == Path("/tmp/dc.yml")


def test_add_custom_model_creates_models_dict_for_new_instance(svc: McpService) -> None:
    svc.instances_info["gpu-1"] = Instance(None, None, {}, InstanceConfig())
    assert "gpu-1" not in svc.models

    svc._add_custom_model("gpu-1", _make_custom())  # pyright: ignore[reportPrivateUsage]

    assert "my-mcp" in svc.models["gpu-1"]


def test_remove_custom_model_happy_path(svc: McpService) -> None:
    svc._add_custom_model("default", _make_custom())  # pyright: ignore[reportPrivateUsage]
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))

    svc._remove_custom_model("default", _make_custom())  # pyright: ignore[reportPrivateUsage]

    assert "my-mcp" not in svc.models["default"]


def test_remove_custom_model_in_use_raises_400(svc: McpService) -> None:
    svc._add_custom_model("default", _make_custom())  # pyright: ignore[reportPrivateUsage]
    installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    installed.models["my-mcp"] = MagicMock()
    svc.instances_info["default"].installed = installed

    with pytest.raises(HTTPException) as exc_info:
        svc._remove_custom_model("default", _make_custom())  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_list_models_invalid_instance_raises_404(svc: McpService) -> None:
    with pytest.raises(HTTPException) as exc_info:
        await svc.list_models("nonexistent", ListModelsFilters())

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_list_models_single_string_instance_returns_models(svc: McpService) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))

    result = await svc.list_models("default", ListModelsFilters())

    assert len(result.list) > 0


@pytest.mark.asyncio
async def test_list_models_skips_instances_not_in_filter(svc: McpService) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    svc.models["other-instance"] = svc.models["default"].copy()

    result = await svc.list_models("default", ListModelsFilters())

    assert all(r.service.startswith("mcp") for r in result.list)


@pytest.mark.asyncio
async def test_list_models_none_instance_returns_all(svc: McpService) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))

    result = await svc.list_models(None, ListModelsFilters())

    assert len(result.list) > 0


@pytest.mark.asyncio
async def test_list_models_list_of_instances(svc: McpService) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))

    result = await svc.list_models(["default"], ListModelsFilters())

    assert isinstance(result.list, list)


@pytest.mark.asyncio
async def test_list_models_filter_installed_true_returns_only_installed(svc: McpService) -> None:
    installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    mock_info = MagicMock()
    mock_info.get_info.return_value = MagicMock()
    installed.models["open-websearch"] = mock_info
    svc.instances_info["default"].installed = installed

    result = await svc.list_models(None, ListModelsFilters(installed=True))

    assert all(m.installed for m in result.list)


@pytest.mark.asyncio
async def test_list_models_filter_installed_false_returns_only_uninstalled(svc: McpService) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))

    result = await svc.list_models(None, ListModelsFilters(installed=False))

    assert all(not m.installed for m in result.list)


@pytest.mark.asyncio
async def test_list_models_installed_model_has_model_info(svc: McpService) -> None:
    installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    mock_info = MagicMock()
    mock_info.get_info.return_value = MagicMock()
    installed.models["open-websearch"] = mock_info
    svc.instances_info["default"].installed = installed

    result = await svc.list_models("default", ListModelsFilters())

    model = next((m for m in result.list if m.id == "open-websearch"), None)
    assert model is not None
    assert mock_info.get_info.call_count == 1


@pytest.mark.asyncio
async def test_get_model_not_found_raises_400(svc: McpService) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))

    with pytest.raises(HTTPException) as exc_info:
        await svc.get_model("default", "nonexistent-model")

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_get_model_reinitializes_empty_models_dict(svc: McpService) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    del svc.models["default"]

    with pytest.raises(HTTPException) as exc_info:
        await svc.get_model("default", "open-websearch")

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_get_model_returns_retrieve_model_out(svc: McpService) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))

    result = await svc.get_model("default", "open-websearch")

    assert result.id == "open-websearch"
    assert result.type == "mcp"


@pytest.mark.asyncio
async def test_get_model_installed_returns_model_info(svc: McpService) -> None:
    installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    mock_info = MagicMock()
    mock_info.get_info.return_value = MagicMock()
    installed.models["open-websearch"] = mock_info
    svc.instances_info["default"].installed = installed

    result = await svc.get_model("default", "open-websearch")

    assert result.id == "open-websearch"
    assert mock_info.get_info.call_count == 1


def test_check_envs_missing_key_raises_422(svc: McpService) -> None:
    with pytest.raises(HTTPException) as exc_info:
        svc.check_envs(["API_KEY"], {"OTHER": "val"})

    assert exc_info.value.status_code == 422
    assert "API_KEY" in exc_info.value.detail


def test_check_envs_empty_value_raises_422(svc: McpService) -> None:
    with pytest.raises(HTTPException) as exc_info:
        svc.check_envs(["API_KEY"], {"API_KEY": ""})

    assert exc_info.value.status_code == 422
    assert "API_KEY" in exc_info.value.detail


def test_check_envs_all_present_and_non_empty_passes(svc: McpService) -> None:
    svc.check_envs(["API_KEY"], {"API_KEY": "secret"})


def test_check_envs_no_required_envs_passes(svc: McpService) -> None:
    svc.check_envs(None, {})


def test_check_headers_missing_key_raises_422(svc: McpService) -> None:
    with pytest.raises(HTTPException) as exc_info:
        svc.check_headers(["Authorization"], {"Other": "val"})

    assert exc_info.value.status_code == 422
    assert "Authorization" in exc_info.value.detail


def test_check_headers_empty_value_raises_422(svc: McpService) -> None:
    with pytest.raises(HTTPException) as exc_info:
        svc.check_headers(["Authorization"], {"Authorization": ""})

    assert exc_info.value.status_code == 422
    assert "Authorization" in exc_info.value.detail


def test_check_headers_all_present_passes(svc: McpService) -> None:
    svc.check_headers(["Authorization"], {"Authorization": "Bearer token"})


def test_check_headers_no_required_headers_passes(svc: McpService) -> None:
    svc.check_headers(None, {})


@pytest.mark.asyncio
async def test_install_model_already_installed_returns_ok(svc: McpService) -> None:
    installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    installed.models["open-websearch"] = MagicMock()
    svc.instances_info["default"].installed = installed

    promise = await svc._install_model("default", "open-websearch", InstallModelIn())  # pyright: ignore[reportPrivateUsage]
    result = await promise.wait()

    assert result.status == "OK"
    assert "Already" in result.details


@pytest.mark.asyncio
async def test_install_model_not_found_raises_400(svc: McpService) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))

    with pytest.raises(HTTPException) as exc_info:
        await svc._install_model("default", "nonexistent", InstallModelIn())  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_install_model_reinitializes_empty_models_dict(svc: McpService) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    del svc.models["default"]

    with pytest.raises(HTTPException) as exc_info:
        await svc._install_model("default", "nonexistent", InstallModelIn())  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_install_model_missing_required_env_raises_422(svc: McpService) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    with pytest.raises(HTTPException) as exc_info:
        await svc._install_model(  # pyright: ignore[reportPrivateUsage]
            "default",
            "brave-search",
            InstallModelIn(spec={"prefix": "brave-search", "envs": {"WRONG_KEY": "val"}}),
        )
    assert exc_info.value.status_code == 422


@pytest.mark.asyncio
async def test_install_model_happy_path(svc: McpService, deps: dict[str, Any], tmp_path: Path) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    model_id = "open-websearch"
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=12345)
    deps["docker_service"].get_container_host.return_value = "172.20.0.2"
    deps["docker_service"].get_container_port.return_value = 3000
    deps["endpoint_registry"].register_mcp_endpoint_as_proxy.return_value = "reg-id"

    with (
        patch.object(svc, "_verify_docker_image", new=AsyncMock()),
        patch.object(svc, "_download_image_or_set_progress", new=AsyncMock()),
        patch.object(svc, "_get_working_dir", return_value=tmp_path),
        patch("server.services.mcp_service.get_base_url", return_value="http://172.20.0.2:3000"),
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn())  # pyright: ignore[reportPrivateUsage]
        result = await promise.wait()

    assert result.status == "OK"
    assert result.details == "Installed"
    info = svc.get_instance_installed_info("default")
    assert model_id in info.models
    assert model_id in svc.models_downloaded


@pytest.mark.asyncio
async def test_uninstall_model_removes_model(svc: McpService, deps: dict[str, Any]) -> None:
    installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    mock_model = MagicMock()
    mock_model.prefix = "open-websearch"
    mock_model.registration_id = "reg-id"
    installed.models["open-websearch"] = mock_model
    svc.instances_info["default"].installed = installed
    deps["docker_service"].uninstall_docker = AsyncMock()

    await svc._uninstall_model("default", "open-websearch", UninstallModelIn())  # pyright: ignore[reportPrivateUsage]

    info = svc.get_instance_installed_info("default")
    assert "open-websearch" not in info.models


@pytest.mark.asyncio
async def test_uninstall_model_purge_removes_downloaded(svc: McpService, deps: dict[str, Any]) -> None:
    installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    mock_model = MagicMock()
    mock_model.prefix = "open-websearch"
    mock_model.registration_id = "reg-id"
    installed.models["open-websearch"] = mock_model
    svc.instances_info["default"].installed = installed
    svc.models_downloaded["open-websearch"] = DownloadedInfo(image="some-image")
    deps["docker_service"].uninstall_docker = AsyncMock()
    deps["docker_service"].remove_image = AsyncMock()

    await svc._uninstall_model("default", "open-websearch", UninstallModelIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert "open-websearch" not in svc.models_downloaded


@pytest.mark.asyncio
async def test_uninstall_model_not_installed_does_nothing(svc: McpService) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    await svc._uninstall_model("default", "nonexistent", UninstallModelIn())  # pyright: ignore[reportPrivateUsage]


def test_get_working_dir_returns_path(svc: McpService, deps: dict[str, Any], tmp_path: Path) -> None:
    deps["config"].get_storage_services_dir.return_value = tmp_path

    result = svc.get_working_dir()

    assert isinstance(result, Path)


@pytest.mark.asyncio
async def test_resolve_custom_model_size_returns_formatted_size(svc: McpService, deps: dict[str, Any]) -> None:
    deps["docker_service"].get_docker_image_size = AsyncMock(return_value=1024**2)
    result = await svc._resolve_custom_model_size({"image": "myimage:latest"})  # pyright: ignore[reportPrivateUsage]

    assert result == "1.0 MB"


@pytest.mark.asyncio
async def test_resolve_custom_model_size_returns_none_when_image_size_is_none(svc: McpService, deps: dict[str, Any]) -> None:
    deps["docker_service"].get_docker_image_size = AsyncMock(return_value=None)
    result = await svc._resolve_custom_model_size({"image": "myimage:latest"})  # pyright: ignore[reportPrivateUsage]

    assert result is None


@pytest.mark.asyncio
async def test_resolve_custom_model_size_returns_none_on_exception(svc: McpService, deps: dict[str, Any]) -> None:
    deps["docker_service"].get_docker_image_size = AsyncMock(side_effect=Exception("fail"))
    result = await svc._resolve_custom_model_size({"image": "myimage:latest"})  # pyright: ignore[reportPrivateUsage]

    assert result is None


@pytest.mark.asyncio
async def test_uninstall_instance_logs_error_when_model_uninstall_fails(svc: McpService) -> None:
    installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    mock_model = MagicMock()
    mock_model.id = "open-websearch"
    installed.models["open-websearch"] = mock_model
    svc.instances_info["default"].installed = installed

    with (
        patch.object(svc, "_uninstall_model", new=AsyncMock(side_effect=RuntimeError("teardown error"))),
        patch.object(svc, "is_model_installed_in_other_instance", return_value=False),
    ):
        await svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert svc.instances_info["default"].installed is None


@pytest.mark.asyncio
async def test_install_model_sse_transport_uses_sse_proxy(svc: McpService, deps: dict[str, Any], tmp_path: Path) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    model_id = "my-sse-mcp"
    svc._add_custom_model(  # pyright: ignore[reportPrivateUsage]
        "default",
        CustomModel(
            id="uuid-sse",
            data={
                "id": model_id,
                "default_prefix": model_id,
                "size": "100MB",
                "image": "company/sse-mcp:latest",
                "image_port": 8080,
                "proxy_transport": "sse",
            },
        ),
    )
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=12345)
    deps["docker_service"].get_container_host.return_value = "172.20.0.2"
    deps["docker_service"].get_container_port.return_value = 8080
    deps["endpoint_registry"].register_mcp_sse_endpoint_as_proxy.return_value = "sse-reg-id"

    with (
        patch.object(svc, "_verify_docker_image", new=AsyncMock()),
        patch.object(svc, "_download_image_or_set_progress", new=AsyncMock()),
        patch.object(svc, "_get_working_dir", return_value=tmp_path),
        patch("server.services.mcp_service.get_base_url", return_value="http://172.20.0.2:8080"),
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn())  # pyright: ignore[reportPrivateUsage]
        result = await promise.wait()

    assert result.status == "OK"
    deps["endpoint_registry"].register_mcp_sse_endpoint_as_proxy.assert_called_once()
    deps["endpoint_registry"].register_mcp_endpoint_as_proxy.assert_not_called()
    info = svc.get_instance_installed_info("default")
    assert info.models[model_id].registration_id == "sse-reg-id"


# ---------------------------------------------------------------------------
# _parse_mcp_tools
# ---------------------------------------------------------------------------


def test_parse_mcp_tools_empty_list_returns_empty() -> None:
    result = _parse_mcp_tools([])

    assert result == []


def test_parse_mcp_tools_creates_tool_info_from_dict() -> None:
    raw: list[Any] = [{"name": "my-tool", "description": "A tool", "inputSchema": {"type": "object"}}]

    result = _parse_mcp_tools(raw)

    assert len(result) == 1
    assert result[0].name == "my-tool"
    assert result[0].description == "A tool"
    assert result[0].input_schema == {"type": "object"}


def test_parse_mcp_tools_skips_non_dict_items() -> None:
    raw: list[Any] = ["string", 42, None, {"name": "valid", "description": "ok"}]

    result = _parse_mcp_tools(raw)

    assert len(result) == 1
    assert result[0].name == "valid"


def test_parse_mcp_tools_uses_input_schema_fallback() -> None:
    raw: list[Any] = [{"name": "tool", "description": "", "input_schema": {"type": "string"}}]

    result = _parse_mcp_tools(raw)

    assert result[0].input_schema == {"type": "string"}


# ---------------------------------------------------------------------------
# _read_first_sse_json
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_read_first_sse_json_returns_parsed_json() -> None:
    mock_resp = MagicMock()
    mock_resp.content = _AsyncLineIter([b'data: {"id": 1}\n', b"\n"])

    result = await _read_first_sse_json(mock_resp)

    assert result == {"id": 1}


@pytest.mark.asyncio
async def test_read_first_sse_json_skips_unicode_error_line() -> None:
    mock_resp = MagicMock()
    mock_resp.content = _AsyncLineIter([b"\xff\xfe", b'data: {"ok": true}\n', b"\n"])

    result = await _read_first_sse_json(mock_resp)

    assert result == {"ok": True}


@pytest.mark.asyncio
async def test_read_first_sse_json_skips_invalid_json() -> None:
    mock_resp = MagicMock()
    mock_resp.content = _AsyncLineIter([b"data: not-json\n", b"\n"])

    result = await _read_first_sse_json(mock_resp)

    assert result is None


@pytest.mark.asyncio
async def test_read_first_sse_json_empty_stream_returns_none() -> None:
    mock_resp = MagicMock()
    mock_resp.content = _AsyncLineIter([])

    result = await _read_first_sse_json(mock_resp)

    assert result is None


# ---------------------------------------------------------------------------
# _dispatch_sse_event
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_sse_event_endpoint_appends_url_and_sets_event() -> None:
    state = _SseState(asyncio.Event(), [], {})

    _dispatch_sse_event("endpoint", "/session/abc", state)

    assert state.session_url_holder == ["/session/abc"]
    assert state.endpoint_ready.is_set()


def test_dispatch_sse_event_endpoint_empty_data_does_nothing() -> None:
    state = _SseState(asyncio.Event(), [], {})

    _dispatch_sse_event("endpoint", "", state)

    assert state.session_url_holder == []
    assert not state.endpoint_ready.is_set()


def test_dispatch_sse_event_empty_data_returns_early() -> None:
    state = _SseState(asyncio.Event(), [], {})

    _dispatch_sse_event("other", "", state)

    assert state.session_url_holder == []


@pytest.mark.asyncio
async def test_dispatch_sse_event_sets_future_result() -> None:
    loop = asyncio.get_event_loop()
    fut: asyncio.Future[dict[str, Any]] = loop.create_future()
    state = _SseState(asyncio.Event(), [], {2: fut})
    data = json.dumps({"id": 2, "result": "ok"})

    _dispatch_sse_event("message", data, state)

    assert fut.done()
    assert await fut == {"id": 2, "result": "ok"}


@pytest.mark.asyncio
async def test_dispatch_sse_event_ignores_non_matching_future_id() -> None:
    loop = asyncio.get_event_loop()
    fut: asyncio.Future[dict[str, Any]] = loop.create_future()
    state = _SseState(asyncio.Event(), [], {99: fut})
    data = json.dumps({"id": 2, "result": "ok"})

    _dispatch_sse_event("message", data, state)

    assert not fut.done()


@pytest.mark.asyncio
async def test_dispatch_sse_event_skips_already_done_future() -> None:
    loop = asyncio.get_event_loop()
    fut: asyncio.Future[dict[str, Any]] = loop.create_future()
    fut.set_result({"id": 2})
    state = _SseState(asyncio.Event(), [], {2: fut})

    _dispatch_sse_event("message", json.dumps({"id": 2, "result": "new"}), state)

    assert await fut == {"id": 2}


def test_dispatch_sse_event_invalid_json_is_ignored() -> None:
    state = _SseState(asyncio.Event(), [], {})

    _dispatch_sse_event("message", "not valid json", state)


# ---------------------------------------------------------------------------
# _run_sse_reader
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_sse_reader_dispatches_endpoint_event() -> None:
    state = _SseState(asyncio.Event(), [], {})
    mock_resp = MagicMock()
    mock_resp.content = _AsyncLineIter(
        [
            b"event: endpoint\r\n",
            b"data: /session/abc\r\n",
            b"\r\n",
        ]
    )

    await _run_sse_reader(mock_resp, state)

    assert state.endpoint_ready.is_set()
    assert state.session_url_holder == ["/session/abc"]


@pytest.mark.asyncio
async def test_run_sse_reader_skips_unicode_error_line() -> None:
    state = _SseState(asyncio.Event(), [], {})
    mock_resp = MagicMock()
    mock_resp.content = _AsyncLineIter(
        [
            b"\xff\xfe",
            b"event: endpoint\r\n",
            b"data: /session/abc\r\n",
            b"\r\n",
        ]
    )

    await _run_sse_reader(mock_resp, state)

    assert state.endpoint_ready.is_set()


# ---------------------------------------------------------------------------
# _sse_rpc
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sse_rpc_no_id_returns_none() -> None:
    _mock_ctx = _make_acm(MagicMock())
    mock_client = MagicMock()
    mock_client.post.return_value = _mock_ctx
    state = _SseState(asyncio.Event(), [], {})
    payload: dict[str, Any] = {"jsonrpc": "2.0", "method": "notifications/initialized"}

    result = await _sse_rpc(mock_client, "http://example.com/session", payload, state)

    assert result is None


@pytest.mark.asyncio
async def test_sse_rpc_with_id_waits_for_future() -> None:
    _mock_ctx = _make_acm(MagicMock())
    mock_client = MagicMock()
    mock_client.post.return_value = _mock_ctx
    state = _SseState(asyncio.Event(), [], {})
    payload: dict[str, Any] = {"jsonrpc": "2.0", "id": 42, "method": "test"}
    expected: dict[str, Any] = {"id": 42, "result": "ok"}

    async def _set_future() -> None:
        await asyncio.sleep(0.01)
        state.response_futures[42].set_result(expected)

    task = asyncio.create_task(_set_future())
    result = await _sse_rpc(mock_client, "http://example.com/session", payload, state)
    await task

    assert result == expected


@pytest.mark.asyncio
async def test_sse_rpc_exception_cancels_future_and_reraises() -> None:
    mock_client = MagicMock()
    mock_client.post.side_effect = RuntimeError("network error")
    state = _SseState(asyncio.Event(), [], {})
    payload: dict[str, Any] = {"jsonrpc": "2.0", "id": 42, "method": "test"}

    with pytest.raises(RuntimeError, match="network error"):
        await _sse_rpc(mock_client, "http://example.com/session", payload, state)

    assert state.response_futures[42].cancelled()


# ---------------------------------------------------------------------------
# _fetch_tools_from_mcp_endpoint
# ---------------------------------------------------------------------------


@mock.patch("server.services.mcp_service.aiohttp.ClientSession")
@pytest.mark.asyncio
async def test_fetch_tools_from_mcp_endpoint_happy_path(mock_cls: MagicMock) -> None:
    init_resp = _make_mock_resp(200, json_data={"result": {}})
    notif_resp = _make_mock_resp(202)
    tools_resp = _make_mock_resp(200, json_data={"result": {"tools": [{"name": "t", "description": "d"}]}})
    mock_client = MagicMock()
    mock_client.post.side_effect = [_make_acm(init_resp), _make_acm(notif_resp), _make_acm(tools_resp)]
    mock_cls.return_value = _make_acm(mock_client)

    result = await _fetch_tools_from_mcp_endpoint("http://example.com/mcp", {})

    assert result.healthy is True
    assert result.transport == "streamable_http"
    assert len(result.tools) == 1
    assert result.tools[0].name == "t"


@mock.patch("server.services.mcp_service.aiohttp.ClientSession")
@pytest.mark.asyncio
async def test_fetch_tools_from_mcp_endpoint_init_error_non_200(mock_cls: MagicMock) -> None:
    init_resp = _make_mock_resp(500)
    mock_client = MagicMock()
    mock_client.post.return_value = _make_acm(init_resp)
    mock_cls.return_value = _make_acm(mock_client)

    result = await _fetch_tools_from_mcp_endpoint("http://example.com/mcp", {})

    assert result.healthy is False
    assert "initialize" in (result.error or "")
    assert "HTTP 500" in (result.error or "")


@mock.patch("server.services.mcp_service.aiohttp.ClientSession")
@pytest.mark.asyncio
async def test_fetch_tools_from_mcp_endpoint_init_202_accepted(mock_cls: MagicMock) -> None:
    init_resp = _make_mock_resp(202)
    notif_resp = _make_mock_resp(202)
    tools_resp = _make_mock_resp(200, json_data={"result": {"tools": []}})
    mock_client = MagicMock()
    mock_client.post.side_effect = [_make_acm(init_resp), _make_acm(notif_resp), _make_acm(tools_resp)]
    mock_cls.return_value = _make_acm(mock_client)

    result = await _fetch_tools_from_mcp_endpoint("http://example.com/mcp", {})

    assert result.healthy is True


@mock.patch("server.services.mcp_service.aiohttp.ClientSession")
@pytest.mark.asyncio
async def test_fetch_tools_from_mcp_endpoint_sse_content_type_empty_returns_error(mock_cls: MagicMock) -> None:
    init_resp = _make_mock_resp(200, "text/event-stream")
    mock_client = MagicMock()
    mock_client.post.return_value = _make_acm(init_resp)
    mock_cls.return_value = _make_acm(mock_client)

    result = await _fetch_tools_from_mcp_endpoint("http://example.com/mcp", {})

    assert result.healthy is False
    assert "Empty response" in (result.error or "")


@mock.patch("server.services.mcp_service.aiohttp.ClientSession")
@pytest.mark.asyncio
async def test_fetch_tools_from_mcp_endpoint_session_id_captured(mock_cls: MagicMock) -> None:
    init_resp = _make_mock_resp(200, json_data={"result": {}}, session_id="sid-abc")
    notif_resp = _make_mock_resp(202)
    tools_resp = _make_mock_resp(200, json_data={"result": {"tools": []}})
    mock_client = MagicMock()
    mock_client.post.side_effect = [_make_acm(init_resp), _make_acm(notif_resp), _make_acm(tools_resp)]
    mock_cls.return_value = _make_acm(mock_client)

    result = await _fetch_tools_from_mcp_endpoint("http://example.com/mcp", {})

    assert result.healthy is True
    _, second_call_kwargs = mock_client.post.call_args_list[1]
    assert second_call_kwargs.get("headers", {}).get("Mcp-Session-Id") == "sid-abc"


@mock.patch("server.services.mcp_service.aiohttp.ClientSession")
@pytest.mark.asyncio
async def test_fetch_tools_from_mcp_endpoint_init_error_in_data(mock_cls: MagicMock) -> None:
    init_resp = _make_mock_resp(200, json_data={"error": {"message": "not initialized"}})
    mock_client = MagicMock()
    mock_client.post.return_value = _make_acm(init_resp)
    mock_cls.return_value = _make_acm(mock_client)

    result = await _fetch_tools_from_mcp_endpoint("http://example.com/mcp", {})

    assert result.healthy is False
    assert "not initialized" in (result.error or "")


@mock.patch("server.services.mcp_service.aiohttp.ClientSession")
@pytest.mark.asyncio
async def test_fetch_tools_from_mcp_endpoint_tools_list_error(mock_cls: MagicMock) -> None:
    init_resp = _make_mock_resp(200, json_data={"result": {}})
    notif_resp = _make_mock_resp(202)
    tools_resp = _make_mock_resp(200, json_data={"error": {"message": "tools unavailable"}})
    mock_client = MagicMock()
    mock_client.post.side_effect = [_make_acm(init_resp), _make_acm(notif_resp), _make_acm(tools_resp)]
    mock_cls.return_value = _make_acm(mock_client)

    result = await _fetch_tools_from_mcp_endpoint("http://example.com/mcp", {})

    assert result.healthy is True
    assert "tools unavailable" in (result.error or "")


@mock.patch("server.services.mcp_service.aiohttp.ClientSession")
@pytest.mark.asyncio
async def test_fetch_tools_from_mcp_endpoint_client_connector_error(mock_cls: MagicMock) -> None:
    mock_cls.return_value.__aenter__ = AsyncMock(side_effect=aiohttp.ClientConnectorError(MagicMock(), OSError("refused")))
    mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

    result = await _fetch_tools_from_mcp_endpoint("http://example.com/mcp", {})

    assert result.healthy is False
    assert "Connection refused" in (result.error or "")


@mock.patch("server.services.mcp_service.aiohttp.ClientSession")
@pytest.mark.asyncio
async def test_fetch_tools_from_mcp_endpoint_generic_exception(mock_cls: MagicMock) -> None:
    mock_cls.side_effect = RuntimeError("unexpected")

    result = await _fetch_tools_from_mcp_endpoint("http://example.com/mcp", {})

    assert result.healthy is False
    assert "RuntimeError" in (result.error or "")


# ---------------------------------------------------------------------------
# _fetch_tools_from_sse_endpoint
# ---------------------------------------------------------------------------


@mock.patch("server.services.mcp_service.aiohttp.ClientSession")
@pytest.mark.asyncio
async def test_fetch_tools_from_sse_endpoint_non_200_get(mock_cls: MagicMock) -> None:
    sse_resp = _make_mock_resp(404)
    mock_client = MagicMock()
    mock_client.get.return_value = _make_acm(sse_resp)
    mock_cls.return_value = _make_acm(mock_client)

    result = await _fetch_tools_from_sse_endpoint("http://example.com/sse", {})

    assert result.healthy is False
    assert "HTTP 404" in (result.error or "")


@mock.patch("server.services.mcp_service.aiohttp.ClientSession")
@pytest.mark.asyncio
async def test_fetch_tools_from_sse_endpoint_non_sse_content_type(mock_cls: MagicMock) -> None:
    sse_resp = _make_mock_resp(200, "application/json")
    mock_client = MagicMock()
    mock_client.get.return_value = _make_acm(sse_resp)
    mock_cls.return_value = _make_acm(mock_client)

    result = await _fetch_tools_from_sse_endpoint("http://example.com/sse", {})

    assert result.healthy is False
    assert "Not an SSE endpoint" in (result.error or "")


@mock.patch("server.services.mcp_service.aiohttp.ClientSession")
@mock.patch("server.services.mcp_service.asyncio.wait_for")
@pytest.mark.asyncio
async def test_fetch_tools_from_sse_endpoint_timeout_waiting_for_endpoint(mock_wait_for: MagicMock, mock_cls: MagicMock) -> None:
    sse_resp = _make_mock_resp(200, "text/event-stream")
    mock_client = MagicMock()
    mock_client.get.return_value = _make_acm(sse_resp)
    mock_cls.return_value = _make_acm(mock_client)
    mock_wait_for.side_effect = TimeoutError()

    result = await _fetch_tools_from_sse_endpoint("http://example.com/sse", {})

    assert result.healthy is True
    assert result.transport == "sse"
    assert "Timeout waiting for SSE endpoint" in (result.error or "")


@mock.patch("server.services.mcp_service._sse_rpc")
@mock.patch("server.services.mcp_service.aiohttp.ClientSession")
@pytest.mark.asyncio
async def test_fetch_tools_from_sse_endpoint_happy_path_relative_url(mock_cls: MagicMock, mock_sse_rpc: AsyncMock) -> None:
    sse_resp = _make_mock_resp(200, "text/event-stream")
    sse_resp.content = _AsyncLineIter(
        [
            b"event: endpoint\r\n",
            b"data: /session/abc\r\n",
            b"\r\n",
        ]
    )
    mock_client = MagicMock()
    mock_client.get.return_value = _make_acm(sse_resp)
    mock_client.post.return_value = _make_acm(MagicMock())
    mock_cls.return_value = _make_acm(mock_client)
    mock_sse_rpc.side_effect = [None, None, {"id": 2, "result": {"tools": [{"name": "t", "description": "d"}]}}]

    result = await _fetch_tools_from_sse_endpoint("http://example.com/sse", {})

    assert result.healthy is True
    assert result.transport == "sse"
    assert len(result.tools) == 1


@mock.patch("server.services.mcp_service._sse_rpc")
@mock.patch("server.services.mcp_service.aiohttp.ClientSession")
@pytest.mark.asyncio
async def test_fetch_tools_from_sse_endpoint_happy_path_absolute_url(mock_cls: MagicMock, mock_sse_rpc: AsyncMock) -> None:
    sse_resp = _make_mock_resp(200, "text/event-stream")
    sse_resp.content = _AsyncLineIter(
        [
            b"event: endpoint\r\n",
            b"data: http://other.host/session/abc\r\n",
            b"\r\n",
        ]
    )
    mock_client = MagicMock()
    mock_client.get.return_value = _make_acm(sse_resp)
    mock_cls.return_value = _make_acm(mock_client)
    mock_sse_rpc.side_effect = [None, None, {"id": 2, "result": {"tools": []}}]

    result = await _fetch_tools_from_sse_endpoint("http://example.com/sse", {})

    assert result.healthy is True
    session_url_used = mock_sse_rpc.call_args_list[0][0][1]
    assert session_url_used == "http://other.host/session/abc"


@mock.patch("server.services.mcp_service._sse_rpc")
@mock.patch("server.services.mcp_service.aiohttp.ClientSession")
@pytest.mark.asyncio
async def test_fetch_tools_from_sse_endpoint_rpc_timeout(mock_cls: MagicMock, mock_sse_rpc: AsyncMock) -> None:
    sse_resp = _make_mock_resp(200, "text/event-stream")
    sse_resp.content = _AsyncLineIter(
        [
            b"event: endpoint\r\n",
            b"data: /session/abc\r\n",
            b"\r\n",
        ]
    )
    mock_client = MagicMock()
    mock_client.get.return_value = _make_acm(sse_resp)
    mock_cls.return_value = _make_acm(mock_client)
    mock_sse_rpc.side_effect = TimeoutError()

    result = await _fetch_tools_from_sse_endpoint("http://example.com/sse", {})

    assert result.healthy is True
    assert result.transport == "sse"
    assert "Timeout waiting for MCP response" in (result.error or "")


@mock.patch("server.services.mcp_service._sse_rpc")
@mock.patch("server.services.mcp_service.aiohttp.ClientSession")
@pytest.mark.asyncio
async def test_fetch_tools_from_sse_endpoint_tools_data_none(mock_cls: MagicMock, mock_sse_rpc: AsyncMock) -> None:
    sse_resp = _make_mock_resp(200, "text/event-stream")
    sse_resp.content = _AsyncLineIter(
        [
            b"event: endpoint\r\n",
            b"data: /session/abc\r\n",
            b"\r\n",
        ]
    )
    mock_client = MagicMock()
    mock_client.get.return_value = _make_acm(sse_resp)
    mock_cls.return_value = _make_acm(mock_client)
    mock_sse_rpc.side_effect = [None, None, None]

    result = await _fetch_tools_from_sse_endpoint("http://example.com/sse", {})

    assert result.healthy is True
    assert result.transport == "sse"
    assert "No response to tools/list" in (result.error or "")


@mock.patch("server.services.mcp_service._sse_rpc")
@mock.patch("server.services.mcp_service.aiohttp.ClientSession")
@pytest.mark.asyncio
async def test_fetch_tools_from_sse_endpoint_tools_error_in_data(mock_cls: MagicMock, mock_sse_rpc: AsyncMock) -> None:
    sse_resp = _make_mock_resp(200, "text/event-stream")
    sse_resp.content = _AsyncLineIter(
        [
            b"event: endpoint\r\n",
            b"data: /session/abc\r\n",
            b"\r\n",
        ]
    )
    mock_client = MagicMock()
    mock_client.get.return_value = _make_acm(sse_resp)
    mock_cls.return_value = _make_acm(mock_client)
    mock_sse_rpc.side_effect = [None, None, {"error": {"message": "tools error"}}]

    result = await _fetch_tools_from_sse_endpoint("http://example.com/sse", {})

    assert result.healthy is True
    assert "tools error" in (result.error or "")


@mock.patch("server.services.mcp_service.aiohttp.ClientSession")
@pytest.mark.asyncio
async def test_fetch_tools_from_sse_endpoint_client_connector_error(mock_cls: MagicMock) -> None:
    mock_cls.return_value.__aenter__ = AsyncMock(side_effect=aiohttp.ClientConnectorError(MagicMock(), OSError("refused")))
    mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

    result = await _fetch_tools_from_sse_endpoint("http://example.com/sse", {})

    assert result.healthy is False
    assert "Connection refused" in (result.error or "")


@mock.patch("server.services.mcp_service.aiohttp.ClientSession")
@pytest.mark.asyncio
async def test_fetch_tools_from_sse_endpoint_generic_exception(mock_cls: MagicMock) -> None:
    mock_cls.side_effect = RuntimeError("unexpected")

    result = await _fetch_tools_from_sse_endpoint("http://example.com/sse", {})

    assert result.healthy is False
    assert "RuntimeError" in (result.error or "")


# ---------------------------------------------------------------------------
# _persist_custom_model_size
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_persist_custom_model_size_no_custom_returns_early(svc: McpService) -> None:
    model = MagicMock(spec=SrvMcpModel)
    model.custom = None

    with patch.object(svc, "_save", new=AsyncMock()) as mock_save:
        await svc._persist_custom_model_size("default", model)  # pyright: ignore[reportPrivateUsage]

    assert mock_save.call_count == 0


@pytest.mark.asyncio
async def test_persist_custom_model_size_matching_model_saves(svc: McpService) -> None:
    custom_entry = CustomModel(id="uuid-99", data={"size": "0 MB"})
    svc.instances_info["default"].config.custom = [custom_entry]
    model = MagicMock(spec=SrvMcpModel)
    model.custom = "uuid-99"
    model.size = "2.5 GB"

    with patch.object(svc, "_save", new=AsyncMock()) as mock_save:
        await svc._persist_custom_model_size("default", model)  # pyright: ignore[reportPrivateUsage]

    assert mock_save.call_count == 1
    assert custom_entry.data["size"] == "2.5 GB"


@pytest.mark.asyncio
async def test_persist_custom_model_size_no_matching_model_does_not_save(svc: McpService) -> None:
    custom_entry = CustomModel(id="uuid-other", data={"size": "0 MB"})
    svc.instances_info["default"].config.custom = [custom_entry]
    model = MagicMock(spec=SrvMcpModel)
    model.custom = "uuid-nonexistent"
    model.size = "1 GB"

    with patch.object(svc, "_save", new=AsyncMock()) as mock_save:
        await svc._persist_custom_model_size("default", model)  # pyright: ignore[reportPrivateUsage]

    assert mock_save.call_count == 0


@pytest.mark.asyncio
async def test_persist_custom_model_size_skips_non_matching_before_match(svc: McpService) -> None:
    entry_a = CustomModel(id="uuid-A", data={"size": "0 MB"})
    entry_b = CustomModel(id="uuid-B", data={"size": "0 MB"})
    svc.instances_info["default"].config.custom = [entry_a, entry_b]
    model = MagicMock(spec=SrvMcpModel)
    model.custom = "uuid-B"
    model.size = "3 GB"

    with patch.object(svc, "_save", new=AsyncMock()) as mock_save:
        await svc._persist_custom_model_size("default", model)  # pyright: ignore[reportPrivateUsage]

    assert mock_save.call_count == 1
    assert entry_b.data["size"] == "3 GB"


# ---------------------------------------------------------------------------
# _fetch_tools_background
# ---------------------------------------------------------------------------


@mock.patch("server.services.mcp_service.asyncio.sleep", new=AsyncMock())
@pytest.mark.asyncio
async def test_fetch_tools_background_returns_early_when_healthy(svc: McpService) -> None:
    healthy_result = McpHealthCheckResult(healthy=True)

    with patch.object(svc, "healthcheck_model", new=AsyncMock(return_value=healthy_result)) as mock_hc:
        await svc._fetch_tools_background("default", "open-websearch")  # pyright: ignore[reportPrivateUsage]

    assert mock_hc.await_count == 1


@mock.patch("server.services.mcp_service.asyncio.sleep", new=AsyncMock())
@pytest.mark.asyncio
async def test_fetch_tools_background_swallows_exceptions(svc: McpService) -> None:
    unhealthy_result = McpHealthCheckResult(healthy=False, error="err")

    with patch.object(
        svc,
        "healthcheck_model",
        new=AsyncMock(side_effect=[RuntimeError("boom"), unhealthy_result, unhealthy_result, unhealthy_result, unhealthy_result]),
    ) as mock_hc:
        await svc._fetch_tools_background("default", "open-websearch")  # pyright: ignore[reportPrivateUsage]

    assert mock_hc.await_count == 5


@mock.patch("server.services.mcp_service.asyncio.sleep", new=AsyncMock())
@pytest.mark.asyncio
async def test_fetch_tools_background_exhausts_all_retries(svc: McpService) -> None:
    unhealthy_result = McpHealthCheckResult(healthy=False, error="err")

    with patch.object(svc, "healthcheck_model", new=AsyncMock(return_value=unhealthy_result)) as mock_hc:
        await svc._fetch_tools_background("default", "open-websearch")  # pyright: ignore[reportPrivateUsage]

    assert mock_hc.await_count == 5


# ---------------------------------------------------------------------------
# _apply_healthcheck_result
# ---------------------------------------------------------------------------


def test_apply_healthcheck_result_updates_model_props_and_registry(svc: McpService) -> None:
    model = MagicMock(spec=SrvMcpModel)
    model.model_props = MagicMock()
    tools = [McpToolInfo(name="t", description="d")]
    reg_model = MagicMock()
    svc.endpoint_registry.mcp_endpoints.models.get.return_value = {"v1": reg_model}  # pyright: ignore[reportAttributeAccessIssue]
    result = McpHealthCheckResult(healthy=True, transport="streamable_http", tools=tools)

    svc._apply_healthcheck_result(model, "my-prefix", result)  # pyright: ignore[reportPrivateUsage]

    assert model.model_props.tools == tools
    assert model.model_props.transport == "streamable_http"
    assert reg_model.healthy is True
    assert reg_model.props.tools == tools
    assert reg_model.props.transport == "streamable_http"


def test_apply_healthcheck_result_no_transport_skips_transport_update(svc: McpService) -> None:
    model = MagicMock(spec=SrvMcpModel)
    model.model_props = MagicMock()
    reg_model = MagicMock()
    svc.endpoint_registry.mcp_endpoints.models.get.return_value = {"v1": reg_model}  # pyright: ignore[reportAttributeAccessIssue]
    result = McpHealthCheckResult(healthy=True, transport=None)

    svc._apply_healthcheck_result(model, "my-prefix", result)  # pyright: ignore[reportPrivateUsage]

    assert model.model_props.tools == []
    assert not hasattr(model.model_props, "transport") or model.model_props.transport != "streamable_http"


# ---------------------------------------------------------------------------
# healthcheck_model
# ---------------------------------------------------------------------------


def _make_installed_model_info(base_url: str = "http://172.20.0.2:3000", prefix: str = "mymcp") -> MagicMock:
    info = MagicMock(spec=ModelInstalledInfo)
    info.base_url = base_url
    info.headers = {}
    info.prefix = prefix
    return info


@pytest.mark.asyncio
async def test_healthcheck_model_model_not_installed_raises_400(svc: McpService) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))

    with pytest.raises(HTTPException) as exc_info:
        await svc.healthcheck_model("default", "nonexistent")

    assert exc_info.value.status_code == 400
    assert "not installed" in exc_info.value.detail


@pytest.mark.asyncio
async def test_healthcheck_model_model_not_in_registry_raises_400(svc: McpService) -> None:
    installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    installed.models["open-websearch"] = _make_installed_model_info()
    svc.instances_info["default"].installed = installed
    svc.models["default"] = {}

    with pytest.raises(HTTPException) as exc_info:
        await svc.healthcheck_model("default", "open-websearch")

    assert exc_info.value.status_code == 400
    assert "not found in registry" in exc_info.value.detail


@mock.patch("server.services.mcp_service._fetch_tools_from_sse_endpoint")
@pytest.mark.asyncio
async def test_healthcheck_model_proxy_sse_transport(mock_fetch: AsyncMock, svc: McpService) -> None:
    installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    installed.models["open-websearch"] = _make_installed_model_info()
    svc.instances_info["default"].installed = installed
    srv_model = svc.models["default"]["open-websearch"]
    srv_model.kind = "proxy"
    srv_model.proxy_transport = "sse"
    mock_fetch.return_value = McpHealthCheckResult(healthy=False, error="refused")

    result = await svc.healthcheck_model("default", "open-websearch")

    assert mock_fetch.call_count == 1
    assert mock_fetch.call_args[0][0] == "http://172.20.0.2:3000"
    assert result.healthy is False


@mock.patch("server.services.mcp_service._fetch_tools_from_mcp_endpoint")
@pytest.mark.asyncio
async def test_healthcheck_model_proxy_streamable_http(mock_fetch: AsyncMock, svc: McpService) -> None:
    installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    installed.models["open-websearch"] = _make_installed_model_info()
    svc.instances_info["default"].installed = installed
    srv_model = svc.models["default"]["open-websearch"]
    srv_model.kind = "proxy"
    srv_model.proxy_transport = "streamable_http"
    mock_fetch.return_value = McpHealthCheckResult(healthy=False, error="refused")

    result = await svc.healthcheck_model("default", "open-websearch")

    assert mock_fetch.call_count == 1
    assert mock_fetch.call_args[0][0] == "http://172.20.0.2:3000"
    assert result.healthy is False


@mock.patch("server.services.mcp_service._fetch_tools_from_mcp_endpoint")
@mock.patch("server.services.mcp_service._fetch_tools_from_sse_endpoint")
@pytest.mark.asyncio
async def test_healthcheck_model_docker_sse_healthy_no_fallback(mock_sse: AsyncMock, mock_mcp: AsyncMock, svc: McpService) -> None:
    installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    installed.models["open-websearch"] = _make_installed_model_info()
    svc.instances_info["default"].installed = installed
    srv_model = svc.models["default"]["open-websearch"]
    srv_model.kind = "custom"
    srv_model.proxy_transport = "sse"
    mock_sse.return_value = McpHealthCheckResult(healthy=True, transport="sse", tools=[])

    with patch.object(svc, "_apply_healthcheck_result") as mock_apply:
        result = await svc.healthcheck_model("default", "open-websearch")

    assert mock_sse.call_count == 1
    assert mock_mcp.call_count == 0
    assert result.healthy is True
    assert mock_apply.call_count == 1


@mock.patch("server.services.mcp_service._fetch_tools_from_mcp_endpoint")
@mock.patch("server.services.mcp_service._fetch_tools_from_sse_endpoint")
@pytest.mark.asyncio
async def test_healthcheck_model_docker_sse_unhealthy_fallback_healthy(mock_sse: AsyncMock, mock_mcp: AsyncMock, svc: McpService) -> None:
    installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    installed.models["open-websearch"] = _make_installed_model_info()
    svc.instances_info["default"].installed = installed
    srv_model = svc.models["default"]["open-websearch"]
    srv_model.kind = "custom"
    srv_model.proxy_transport = "sse"
    mock_sse.return_value = McpHealthCheckResult(healthy=False, error="sse failed")
    mock_mcp.return_value = McpHealthCheckResult(healthy=True, transport="streamable_http", tools=[])

    result = await svc.healthcheck_model("default", "open-websearch")

    assert mock_mcp.call_count == 1
    assert result.transport == "streamable_http"


@mock.patch("server.services.mcp_service._fetch_tools_from_mcp_endpoint")
@mock.patch("server.services.mcp_service._fetch_tools_from_sse_endpoint")
@pytest.mark.asyncio
async def test_healthcheck_model_docker_sse_both_unhealthy_returns_sse_result(
    mock_sse: AsyncMock, mock_mcp: AsyncMock, svc: McpService
) -> None:
    installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    installed.models["open-websearch"] = _make_installed_model_info()
    svc.instances_info["default"].installed = installed
    srv_model = svc.models["default"]["open-websearch"]
    srv_model.kind = "custom"
    srv_model.proxy_transport = "sse"
    sse_result = McpHealthCheckResult(healthy=False, error="sse failed")
    mock_sse.return_value = sse_result
    mock_mcp.return_value = McpHealthCheckResult(healthy=False, error="mcp failed")

    result = await svc.healthcheck_model("default", "open-websearch")

    assert result is sse_result


@mock.patch("server.services.mcp_service._fetch_tools_from_sse_endpoint")
@mock.patch("server.services.mcp_service._fetch_tools_from_mcp_endpoint")
@pytest.mark.asyncio
async def test_healthcheck_model_docker_streamable_http_healthy_no_fallback(
    mock_mcp: AsyncMock, mock_sse: AsyncMock, svc: McpService
) -> None:
    installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    installed.models["open-websearch"] = _make_installed_model_info()
    svc.instances_info["default"].installed = installed
    srv_model = svc.models["default"]["open-websearch"]
    srv_model.kind = "custom"
    srv_model.proxy_transport = "streamable_http"
    mock_mcp.return_value = McpHealthCheckResult(healthy=True, transport="streamable_http", tools=[])

    with patch.object(svc, "_apply_healthcheck_result") as mock_apply:
        result = await svc.healthcheck_model("default", "open-websearch")

    assert mock_mcp.call_count == 1
    assert mock_sse.call_count == 0
    assert result.healthy is True
    assert mock_apply.call_count == 1


@mock.patch("server.services.mcp_service._fetch_tools_from_sse_endpoint")
@mock.patch("server.services.mcp_service._fetch_tools_from_mcp_endpoint")
@pytest.mark.asyncio
async def test_healthcheck_model_docker_streamable_http_unhealthy_fallback_healthy(
    mock_mcp: AsyncMock, mock_sse: AsyncMock, svc: McpService
) -> None:
    installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    installed.models["open-websearch"] = _make_installed_model_info()
    svc.instances_info["default"].installed = installed
    srv_model = svc.models["default"]["open-websearch"]
    srv_model.kind = "custom"
    srv_model.proxy_transport = "streamable_http"
    mock_mcp.return_value = McpHealthCheckResult(healthy=False, error="mcp failed")
    mock_sse.return_value = McpHealthCheckResult(healthy=True, transport="sse", tools=[])

    result = await svc.healthcheck_model("default", "open-websearch")

    assert mock_sse.call_count == 1
    assert result.transport == "sse"


@mock.patch("server.services.mcp_service._fetch_tools_from_sse_endpoint")
@mock.patch("server.services.mcp_service._fetch_tools_from_mcp_endpoint")
@pytest.mark.asyncio
async def test_healthcheck_model_docker_streamable_http_both_unhealthy_returns_mcp_result(
    mock_mcp: AsyncMock, mock_sse: AsyncMock, svc: McpService
) -> None:
    installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    installed.models["open-websearch"] = _make_installed_model_info()
    svc.instances_info["default"].installed = installed
    srv_model = svc.models["default"]["open-websearch"]
    srv_model.kind = "custom"
    srv_model.proxy_transport = "streamable_http"
    mcp_result = McpHealthCheckResult(healthy=False, error="mcp failed")
    mock_mcp.return_value = mcp_result
    mock_sse.return_value = McpHealthCheckResult(healthy=False, error="sse failed")

    result = await svc.healthcheck_model("default", "open-websearch")

    assert result is mcp_result


@pytest.mark.asyncio
async def test_read_first_sse_json_non_data_line_is_ignored() -> None:
    mock_resp = MagicMock()
    mock_resp.content = _AsyncLineIter(
        [
            b"event: message\n",
            b'data: {"id": 1}\n',
            b"\n",
        ]
    )

    result = await _read_first_sse_json(mock_resp)

    assert result == {"id": 1}


@pytest.mark.asyncio
async def test_run_sse_reader_non_event_non_data_line_is_ignored() -> None:
    state = _SseState(asyncio.Event(), [], {})
    mock_resp = MagicMock()
    mock_resp.content = _AsyncLineIter(
        [
            b"id: 123\r\n",
            b"event: endpoint\r\n",
            b"data: /session/abc\r\n",
            b"\r\n",
        ]
    )

    await _run_sse_reader(mock_resp, state)

    assert state.endpoint_ready.is_set()


@pytest.mark.asyncio
async def test_sse_rpc_no_id_exception_reraises_without_cancel() -> None:
    mock_client = MagicMock()
    mock_client.post.side_effect = RuntimeError("fail")
    state = _SseState(asyncio.Event(), [], {})
    payload: dict[str, Any] = {"jsonrpc": "2.0", "method": "notifications/initialized"}

    with pytest.raises(RuntimeError, match="fail"):
        await _sse_rpc(mock_client, "http://example.com/session", payload, state)

    assert not state.response_futures


def test_get_custom_spec_proxy_with_description_includes_it(svc: McpService) -> None:
    model = MagicMock(spec=SrvMcpModel)
    model.custom = "uuid-proxy"
    model.kind = "proxy"
    model.proxy_url = "http://remote-mcp.example.com/mcp"
    model.proxy_transport = "streamable_http"
    model.default_prefix = "remote-mcp"
    model.headers = None
    model.description = "A remote MCP proxy"
    model.repository_url = None

    spec = svc._get_custom_spec("remote-mcp", model)  # pyright: ignore[reportPrivateUsage]

    assert spec is not None
    assert spec["description"] == "A remote MCP proxy"


def test_get_custom_spec_proxy_with_repository_url_includes_it(svc: McpService) -> None:
    model = MagicMock(spec=SrvMcpModel)
    model.custom = "uuid-proxy"
    model.kind = "proxy"
    model.proxy_url = "http://remote-mcp.example.com/mcp"
    model.proxy_transport = "streamable_http"
    model.default_prefix = "remote-mcp"
    model.headers = None
    model.description = ""
    model.repository_url = "https://github.com/example/remote-mcp"

    spec = svc._get_custom_spec("remote-mcp", model)  # pyright: ignore[reportPrivateUsage]

    assert spec is not None
    assert spec["repository_url"] == "https://github.com/example/remote-mcp"


def test_get_custom_spec_user_with_description_includes_it(svc: McpService) -> None:
    model = MagicMock(spec=SrvMcpModel)
    model.custom = "uuid-user"
    model.kind = "user"
    model.command = "python /app/main.py"
    model.variant = "python"
    model.default_prefix = "my-user-mcp"
    model.envs = None
    model.base_image = None
    model.python_version = None
    model.node_version = None
    model.description = "A user-defined MCP server"
    model.repository_url = None

    spec = svc._get_custom_spec("my-user-mcp", model)  # pyright: ignore[reportPrivateUsage]

    assert spec is not None
    assert spec["description"] == "A user-defined MCP server"


def test_get_custom_spec_user_with_repository_url_includes_it(svc: McpService) -> None:
    model = MagicMock(spec=SrvMcpModel)
    model.custom = "uuid-user"
    model.kind = "user"
    model.command = "python /app/main.py"
    model.variant = "python"
    model.default_prefix = "my-user-mcp"
    model.envs = None
    model.base_image = None
    model.python_version = None
    model.node_version = None
    model.description = ""
    model.repository_url = "https://github.com/example/user-mcp"

    spec = svc._get_custom_spec("my-user-mcp", model)  # pyright: ignore[reportPrivateUsage]

    assert spec is not None
    assert spec["repository_url"] == "https://github.com/example/user-mcp"
