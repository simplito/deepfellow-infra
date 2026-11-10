# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any
from unittest import mock
from unittest.mock import AsyncMock, MagicMock, call, patch

import aiohttp
import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from server.docker import DockerOptions
from server.models.api import McpToolInfo, ModelProps
from server.models.models import (
    AddCustomModelIn,
    InstallModelIn,
    ListModelsFilters,
    McpHealthCheckResult,
    ModelSpecification,
    UninstallModelIn,
)
from server.models.services import InstallServiceIn, UninstallServiceIn
from server.services.base2_service import CustomModel, Instance, InstanceConfig
from server.services.mcp_service import (
    DownloadedInfo,
    InstalledInfo,
    McpModelOptions,
    McpOAuthStatusOut,
    McpService,
    ModelInstalledInfo,
    SrvMcpCustomModel,
    SrvMcpModel,
    SrvMcpProxyModel,
    _dispatch_sse_event,  # pyright: ignore[reportPrivateUsage]
    _fetch_tools_from_mcp_endpoint,  # pyright: ignore[reportPrivateUsage]
    _fetch_tools_from_sse_endpoint,  # pyright: ignore[reportPrivateUsage]
    _parse_mcp_tools,  # pyright: ignore[reportPrivateUsage]
    _read_first_sse_json,  # pyright: ignore[reportPrivateUsage]
    _run_sse_reader,  # pyright: ignore[reportPrivateUsage]
    _sse_rpc,  # pyright: ignore[reportPrivateUsage]
    _SseState,  # pyright: ignore[reportPrivateUsage]
    _terminate_mcp_session,  # pyright: ignore[reportPrivateUsage]
)
from server.utils.mcp_oauth import (
    AuthorizationServerMetadata,
    DcrResponse,
    McpOAuthConfig,
    McpOAuthError,
    PendingOAuthFlow,
    TokenResponse,
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
        "services_manager": MagicMock(),
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


def test_load_default_models_web_search_uses_streamable_http_transport(svc: McpService) -> None:
    model = svc.models["default"]["web-search"]

    assert model.proxy_transport == "streamable_http"
    assert model.model_props.transport == "streamable_http"
    assert model.options is not None
    assert model.options.image == "hub.simplito.com/deepfellow/web-search-mcp:v0.3.4"


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


def test_get_model_install_fields_returns_model_spec_fields(svc: McpService) -> None:
    result = svc._get_model_install_fields("default", "web-search")  # pyright: ignore[reportPrivateUsage]

    assert result == svc.models["default"]["web-search"].model_spec.fields


def test_get_model_install_fields_returns_empty_list_for_unknown_model(svc: McpService) -> None:
    result = svc._get_model_install_fields("default", "ghost-model")  # pyright: ignore[reportPrivateUsage]

    assert result == []


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


def test_loaded_open_websearch_model_has_docker_tags_field(svc: McpService) -> None:
    model = svc.models["default"]["open-websearch"]

    field = next(f for f in model.model_spec.fields if f.name == "image_version")
    assert field.type == "docker-tags"
    assert field.docker_image == "hub.simplito.com/deepfellow/open-websearch"
    assert field.depends_on is None


def test_loaded_scrapling_model_has_no_docker_tags_field(svc: McpService) -> None:
    # scrapling's image is digest-pinned (no floating tag), so there's nothing to select.
    model = svc.models["default"]["scrapling"]

    assert not any(f.type == "docker-tags" for f in model.model_spec.fields)


def test_loaded_firecrawl_model_has_no_docker_tags_field(svc: McpService) -> None:
    model = svc.models["default"]["firecrawl"]

    assert not any(f.type == "docker-tags" for f in model.model_spec.fields)


def test_attach_docker_tags_field_skips_proxy_models(svc: McpService) -> None:
    proxy_model = SrvMcpModel(
        model_props=MagicMock(),
        model_spec=ModelSpecification(fields=[]),
        model_type="mcp",
        default_prefix="proxy",
        size="",
        options=None,
        kind="proxy",
    )

    svc._attach_docker_tags_field(proxy_model)  # pyright: ignore[reportPrivateUsage]

    assert not any(f.type == "docker-tags" for f in proxy_model.model_spec.fields)


def test_attach_docker_tags_field_is_idempotent(svc: McpService) -> None:
    model = svc.models["default"]["open-websearch"]
    fields_before = len(model.model_spec.fields)

    svc._attach_docker_tags_field(model)  # pyright: ignore[reportPrivateUsage]

    assert len(model.model_spec.fields) == fields_before


def test_get_docker_image_repo_for_model_open_websearch(svc: McpService) -> None:
    assert svc.get_docker_image_repo_for_model("open-websearch", None) == "hub.simplito.com/deepfellow/open-websearch"


def test_get_docker_image_repo_for_model_unknown_returns_none(svc: McpService) -> None:
    assert svc.get_docker_image_repo_for_model("unknown-model", None) is None
    assert svc.get_docker_image_repo_for_model(None, None) is None


def test_get_docker_image_repo_for_model_proxy_returns_none(svc: McpService) -> None:
    svc.models["default"]["proxy-model"] = SrvMcpModel(
        model_props=MagicMock(),
        model_spec=ModelSpecification(fields=[]),
        model_type="mcp",
        default_prefix="proxy",
        size="",
        options=None,
        kind="proxy",
    )

    assert svc.get_docker_image_repo_for_model("proxy-model", None) is None


def test_get_default_docker_tag_for_model_open_websearch(svc: McpService) -> None:
    assert svc.get_default_docker_tag_for_model("open-websearch", None) == "v2.1.9"


def test_get_default_docker_tag_for_model_unknown_returns_none(svc: McpService) -> None:
    assert svc.get_default_docker_tag_for_model("unknown-model", None) is None


@pytest.mark.asyncio
async def test_get_docker_tags_for_model_fetches_from_registry(svc: McpService) -> None:
    client = MagicMock()
    client.get_tags = AsyncMock(return_value=["v2.1.9", "v2.1.8"])
    with patch("server.services.mcp_service.registry_for", return_value=client) as mock_registry_for:
        tags = await svc.get_docker_tags_for_model("open-websearch", None)

    mock_registry_for.assert_called_once_with("hub.simplito.com/deepfellow/open-websearch")
    client.get_tags.assert_called_once_with("deepfellow/open-websearch")
    assert tags == ["v2.1.9", "v2.1.8"]


@pytest.mark.asyncio
async def test_get_docker_tags_for_model_unknown_returns_empty(svc: McpService) -> None:
    assert await svc.get_docker_tags_for_model("unknown-model", None) == []


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
    result = svc._generate_instance_config("default", None, None)  # pyright: ignore[reportPrivateUsage]

    assert result.options is None
    assert result.models == []
    assert result.custom is None


def test_generate_instance_config_with_info(svc: McpService) -> None:
    options = InstallServiceIn(spec={})
    installed = InstalledInfo(models={}, options=options)

    result = svc._generate_instance_config("default", installed, None)  # pyright: ignore[reportPrivateUsage]

    assert result.options is options


def test_generate_instance_config_persists_resolved_container_name(svc: McpService) -> None:
    docker_model = MagicMock()
    docker_model.id = "open-websearch"
    docker_model.options = InstallModelIn(spec={})
    docker_model.docker_options.container_name = "df-open-websearch_abc123"
    proxy_model = MagicMock()
    proxy_model.id = "remote-proxy"
    proxy_model.options = InstallModelIn(spec={})
    proxy_model.docker_options = None
    installed = InstalledInfo(models={"open-websearch": docker_model, "remote-proxy": proxy_model}, options=InstallServiceIn(spec={}))

    result = svc._generate_instance_config("default", installed, None)  # pyright: ignore[reportPrivateUsage]

    assert result.models is not None
    assert {m.model_id: m.container_name for m in result.models} == {
        "open-websearch": "df-open-websearch_abc123",
        "remote-proxy": None,
    }


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


@pytest.mark.asyncio
async def test_uninstall_instance_purge_with_other_instance_installed_does_not_clear_working_dir(svc: McpService) -> None:
    # branch 693->698: purging one instance while another instance is still installed skips the shared cleanup
    svc.instances_info["gpu-1"] = Instance(None, None, {}, InstanceConfig())
    svc.instances_info["gpu-1"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    svc.service_downloaded = True

    with patch.object(svc, "_clear_working_dir", new=AsyncMock()) as mock_clear:
        await svc._uninstall_instance("default", UninstallServiceIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert mock_clear.call_count == 0
    assert svc.service_downloaded is True
    assert svc.instances_info["default"].installed is None
    assert "gpu-1" in svc.instances_info


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


def test_get_docker_options_returns_docker_options_for_installed_model(svc: McpService) -> None:
    installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    mock_model = MagicMock()
    installed.models["my-model"] = mock_model
    svc.instances_info["default"].installed = installed

    result = svc.get_docker_options("default", "my-model")

    assert result is mock_model.docker_options


def test_get_docker_options_raises_400_when_docker_options_is_none(svc: McpService) -> None:
    installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    mock_model = MagicMock()
    mock_model.docker_options = None
    installed.models["my-model"] = mock_model
    svc.instances_info["default"].installed = installed

    with pytest.raises(HTTPException) as exc_info:
        svc.get_docker_options("default", "my-model")

    assert exc_info.value.status_code == 400


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
    mock_info.prefix = "open-websearch"
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
    mock_info.prefix = "open-websearch"
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
    mock_info.prefix = "open-websearch"
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


def test_check_envs_empty_dict_raises_422(svc: McpService) -> None:
    with pytest.raises(HTTPException) as exc_info:
        svc.check_envs(["API_KEY"], {})

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


def test_check_headers_empty_dict_raises_422(svc: McpService) -> None:
    with pytest.raises(HTTPException) as exc_info:
        svc.check_headers(["Authorization"], {})

    assert exc_info.value.status_code == 422
    assert "Authorization" in exc_info.value.detail


def test_check_headers_all_present_passes(svc: McpService) -> None:
    svc.check_headers(["Authorization"], {"Authorization": "Bearer token"})


def test_check_headers_no_required_headers_passes(svc: McpService) -> None:
    svc.check_headers(None, {})


def test_default_model_spec_exposes_required_env_keys(svc: McpService) -> None:
    spec = svc.get_default_model_spec("brave-search", required_envs=["BRAVE_API_KEY"])
    envs_field = next(f for f in spec.fields if f.name == "envs")
    assert envs_field.required_keys == ["BRAVE_API_KEY"]


def test_default_model_spec_exposes_required_header_keys(svc: McpService) -> None:
    spec = svc.get_default_model_spec("serpapi", required_headers={"Authorization": "Bearer "})
    headers_field = next(f for f in spec.fields if f.name == "headers")
    assert headers_field.required_keys == ["Authorization"]


def test_default_model_spec_no_required_keys_when_none(svc: McpService) -> None:
    spec = svc.get_default_model_spec("open-websearch")
    envs_field = next(f for f in spec.fields if f.name == "envs")
    headers_field = next(f for f in spec.fields if f.name == "headers")
    assert envs_field.required_keys is None
    assert headers_field.required_keys is None


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
async def test_install_model_without_required_headers_raises_422(svc: McpService) -> None:
    """The `Bearer ` template default of a required header must not be filled in as if the client had sent it."""
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))

    with pytest.raises(HTTPException) as exc_info:
        await svc.install_model("default", "serpapi", InstallModelIn(spec={}))

    assert exc_info.value.status_code == 422
    assert "required headers are missing: Authorization" in exc_info.value.detail


@pytest.mark.asyncio
async def test_install_model_happy_path(svc: McpService, deps: dict[str, Any], tmp_path: Path) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    model_id = "open-websearch"
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=(12345, True, False))
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
async def test_install_model_uses_container_name_override_when_given(svc: McpService, deps: dict[str, Any], tmp_path: Path) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    model_id = "open-websearch"
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=(12345, True, False))
    deps["docker_service"].get_container_host.return_value = "172.20.0.2"
    deps["docker_service"].get_container_port.return_value = 3000
    deps["endpoint_registry"].register_mcp_endpoint_as_proxy.return_value = "reg-id"
    original_container_name = svc.models["default"][model_id].options.container_name  # pyright: ignore[reportOptionalMemberAccess]

    with (
        patch.object(svc, "_verify_docker_image", new=AsyncMock()),
        patch.object(svc, "_download_image_or_set_progress", new=AsyncMock()),
        patch.object(svc, "_get_working_dir", return_value=tmp_path),
        patch("server.services.mcp_service.get_base_url", return_value="http://172.20.0.2:3000"),
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn(), "pinned-name")  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    docker_options = deps["docker_service"].install_and_run_docker.call_args.args[0]
    assert docker_options.container_name == "pinned-name"
    installed_docker_options = svc.get_instance_installed_info("default").models[model_id].docker_options
    assert installed_docker_options is not None
    assert installed_docker_options.container_name == "pinned-name"
    assert svc.models["default"][model_id].options.container_name == original_container_name  # pyright: ignore[reportOptionalMemberAccess]


@pytest.mark.asyncio
async def test_install_model_falls_back_to_generated_container_name(svc: McpService, deps: dict[str, Any], tmp_path: Path) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    model_id = "open-websearch"
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=(12345, True, False))
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
        await promise.wait()

    docker_options = deps["docker_service"].install_and_run_docker.call_args.args[0]
    assert docker_options.container_name == "df-open-websearch"


@pytest.mark.asyncio
async def test_install_model_applies_suffix_changed_after_model_list_was_built(
    svc: McpService, deps: dict[str, Any], tmp_path: Path
) -> None:
    """Catalog options are built at load time; a later live suffix change must still reach new installs."""
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    model_id = "open-websearch"
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=(12345, True, False))
    deps["docker_service"].get_container_host.return_value = "172.20.0.2"
    deps["docker_service"].get_container_port.return_value = 3000
    deps["endpoint_registry"].register_mcp_endpoint_as_proxy.return_value = "reg-id"
    deps["docker_service"].get_docker_container_name.side_effect = lambda name: f"{name}_new"  # pyright: ignore[reportUnknownLambdaType]

    with (
        patch.object(svc, "_verify_docker_image", new=AsyncMock()),
        patch.object(svc, "_download_image_or_set_progress", new=AsyncMock()),
        patch.object(svc, "_get_working_dir", return_value=tmp_path),
        patch("server.services.mcp_service.get_base_url", return_value="http://172.20.0.2:3000"),
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn())  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    docker_options = deps["docker_service"].install_and_run_docker.call_args.args[0]
    assert docker_options.container_name == "open-websearch_new"
    assert svc.models["default"][model_id].options.container_name == "df-open-websearch"  # pyright: ignore[reportOptionalMemberAccess]


@pytest.mark.asyncio
async def test_install_model_allows_resolved_container_host_for_duckduckgo(svc: McpService, deps: dict[str, Any], tmp_path: Path) -> None:
    """DDG's DNS-rebinding protection rejects any other Host header with a 421, so it must list the real hostname."""
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=(12345, True, False))
    deps["docker_service"].get_container_host.return_value = "duckduckgo_myinfra_a1b2c3"
    deps["docker_service"].get_container_port.return_value = 8000
    deps["endpoint_registry"].register_mcp_endpoint_as_proxy.return_value = "reg-id"

    with (
        patch.object(svc, "_verify_docker_image", new=AsyncMock()),
        patch.object(svc, "_download_image_or_set_progress", new=AsyncMock()),
        patch.object(svc, "_get_working_dir", return_value=tmp_path),
        patch("server.services.mcp_service.get_base_url", return_value="http://duckduckgo_myinfra_a1b2c3:8000"),
    ):
        promise = await svc._install_model("default", "duckduckgo", InstallModelIn())  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    docker_options = deps["docker_service"].install_and_run_docker.call_args.args[0]
    assert docker_options.env_vars["DDG_ALLOWED_HOSTS"] == "duckduckgo_myinfra_a1b2c3:*"
    assert svc.models["default"]["duckduckgo"].options.env_vars["DDG_ALLOWED_HOSTS"] == "{container_host}:*"  # pyright: ignore[reportOptionalMemberAccess]


@pytest.mark.asyncio
@pytest.mark.parametrize(("override", "expected_host"), [("pinned-name", "pinned-name"), (None, "duckduckgo_new")])
async def test_install_model_derives_container_host_from_final_container_name(
    svc: McpService, deps: dict[str, Any], tmp_path: Path, override: str | None, expected_host: str
) -> None:
    """The host (and DDG's allow-list) must follow the pinned or re-derived name, not the catalog's load-time one."""
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=(12345, True, False))
    deps["docker_service"].get_container_host.side_effect = lambda _subnet, options: options.container_name  # pyright: ignore[reportUnknownLambdaType, reportUnknownMemberType]
    deps["docker_service"].get_container_port.return_value = 8000
    deps["docker_service"].get_docker_container_name.side_effect = lambda name: f"{name}_new"  # pyright: ignore[reportUnknownLambdaType]
    deps["endpoint_registry"].register_mcp_endpoint_as_proxy.return_value = "reg-id"

    with (
        patch.object(svc, "_verify_docker_image", new=AsyncMock()),
        patch.object(svc, "_download_image_or_set_progress", new=AsyncMock()),
        patch.object(svc, "_get_working_dir", return_value=tmp_path),
        patch("server.services.mcp_service.get_base_url", return_value="http://host:8000") as mock_base_url,
    ):
        promise = await svc._install_model("default", "duckduckgo", InstallModelIn(), override)  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    docker_options = deps["docker_service"].install_and_run_docker.call_args.args[0]
    assert docker_options.env_vars["DDG_ALLOWED_HOSTS"] == f"{expected_host}:*"
    assert svc.instances_info["default"].installed.models["duckduckgo"].container_host == expected_host  # pyright: ignore[reportOptionalMemberAccess]
    assert mock_base_url.call_args.args[0] == expected_host


@pytest.mark.asyncio
async def test_install_model_passes_user_envs_over_catalog_envs(svc: McpService, deps: dict[str, Any], tmp_path: Path) -> None:
    """User-supplied envs (e.g. API keys) must reach the container, and win over the catalog's own defaults."""
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=(12345, True, False))
    deps["docker_service"].get_container_host.return_value = "duckduckgo"
    deps["docker_service"].get_container_port.return_value = 8000
    deps["endpoint_registry"].register_mcp_endpoint_as_proxy.return_value = "reg-id"
    spec = {"envs": {"FOO": "bar", "DDG_ALLOWED_HOSTS": "custom:*"}}

    with (
        patch.object(svc, "_verify_docker_image", new=AsyncMock()),
        patch.object(svc, "_download_image_or_set_progress", new=AsyncMock()),
        patch.object(svc, "_get_working_dir", return_value=tmp_path),
        patch("server.services.mcp_service.get_base_url", return_value="http://duckduckgo:8000"),
    ):
        promise = await svc._install_model("default", "duckduckgo", InstallModelIn(spec=spec))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    env_vars = deps["docker_service"].install_and_run_docker.call_args.args[0].env_vars
    assert env_vars["FOO"] == "bar"
    assert env_vars["DDG_ALLOWED_HOSTS"] == "custom:*"


@pytest.mark.asyncio
async def test_install_model_cleans_up_installing_set_when_cancelled_during_setup(svc: McpService) -> None:
    """A cancellation during the pre-promise docker-image verification must still discard the
    `_installing` entry - not just ordinary exceptions - or the model gets stuck looking installed.
    """
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    model_id = "open-websearch"
    reached = asyncio.Event()

    async def hanging_verify(*_args: Any, **_kwargs: Any) -> None:
        reached.set()
        await asyncio.Event().wait()  # never completes on its own - must be cancelled

    with patch.object(svc, "_verify_docker_image", new=hanging_verify):
        task = asyncio.create_task(svc._install_model("default", model_id, InstallModelIn()))  # pyright: ignore[reportPrivateUsage]
        await reached.wait()

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert ("default", model_id) not in svc._installing  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_install_model_rejects_unavailable_image_version(svc: McpService, deps: dict[str, Any]) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    model_id = "open-websearch"
    client = MagicMock()
    client.get_tags = AsyncMock(return_value=["v2.1.9"])
    client.tag_exists = AsyncMock(return_value=False)

    with (
        patch("server.services.mcp_service.registry_for", return_value=client),
        pytest.raises(HTTPException) as exc,
    ):
        await svc._install_model(  # pyright: ignore[reportPrivateUsage]
            "default", model_id, InstallModelIn(spec={"prefix": model_id, "image_version": "9.9.9-bogus"})
        )

    assert exc.value.status_code == 400
    info = svc.get_instance_installed_info("default")
    assert model_id not in info.models


@pytest.mark.asyncio
async def test_install_model_applies_selected_image_version(svc: McpService, deps: dict[str, Any], tmp_path: Path) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    model_id = "open-websearch"
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=(12345, True, False))
    deps["docker_service"].get_container_host.return_value = "172.20.0.2"
    deps["docker_service"].get_container_port.return_value = 3000
    deps["endpoint_registry"].register_mcp_endpoint_as_proxy.return_value = "reg-id"

    with (
        patch.object(svc, "_verify_docker_image", new=AsyncMock()) as mock_verify,
        patch.object(svc, "_download_image_or_set_progress", new=AsyncMock()),
        patch.object(svc, "_get_working_dir", return_value=tmp_path),
        patch.object(svc, "validate_docker_image_version_for_model", new_callable=AsyncMock),
        patch("server.services.mcp_service.get_base_url", return_value="http://172.20.0.2:3000"),
    ):
        promise = await svc._install_model(  # pyright: ignore[reportPrivateUsage]
            "default", model_id, InstallModelIn(spec={"prefix": model_id, "image_version": "v2.1.8"})
        )
        await promise.wait()

    mock_verify.assert_called_once_with("hub.simplito.com/deepfellow/open-websearch:v2.1.8", False)
    info = svc.get_instance_installed_info("default")
    assert info.models[model_id].docker_options.image == "hub.simplito.com/deepfellow/open-websearch:v2.1.8"  # pyright: ignore[reportOptionalMemberAccess]
    assert svc.models_downloaded[model_id].image == "hub.simplito.com/deepfellow/open-websearch:v2.1.8"
    # The model's own default image is untouched for subsequent installs.
    assert svc.models["default"]["open-websearch"].options.image == "hub.simplito.com/deepfellow/open-websearch:v2.1.9"  # pyright: ignore[reportOptionalMemberAccess]


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
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=(12345, True, False))
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


@mock.patch("server.services.mcp_service._fetch_tools_from_mcp_endpoint")
@pytest.mark.asyncio
async def test_install_model_proxy_healthy_probe_skips_background_retry(mock_fetch: AsyncMock, svc: McpService) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    svc._add_custom_model("default", _make_proxy_custom())  # pyright: ignore[reportPrivateUsage]
    mock_fetch.return_value = McpHealthCheckResult(healthy=True, transport="streamable_http", tools=[])

    promise = await svc._install_model("default", "my-remote-mcp", InstallModelIn())  # pyright: ignore[reportPrivateUsage]
    result = await promise.wait()
    await asyncio.sleep(0)

    assert result.status == "OK"
    assert result.details == "Installed"
    assert result.requires_oauth is False
    # Only the periodic health monitor runs; no one-shot retry task.
    assert set(svc._background_tasks) == set(svc._health_monitor_tasks.values())  # pyright: ignore[reportPrivateUsage]
    svc._stop_health_monitor("default", "my-remote-mcp")  # pyright: ignore[reportPrivateUsage]


@mock.patch("server.services.mcp_service._fetch_tools_from_mcp_endpoint")
@pytest.mark.asyncio
async def test_install_model_proxy_401_auto_detects_oauth_before_returning(mock_fetch: AsyncMock, svc: McpService) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    svc._add_custom_model("default", _make_proxy_custom())  # pyright: ignore[reportPrivateUsage]
    mock_fetch.return_value = McpHealthCheckResult(healthy=False, requires_oauth=True, error="401")

    with (
        patch("server.services.mcp_service.discover_authorization_server_for_resource", new=AsyncMock(return_value=None)),
        patch.object(svc, "_save", new=AsyncMock()),
    ):
        promise = await svc._install_model("default", "my-remote-mcp", InstallModelIn())  # pyright: ignore[reportPrivateUsage]
        result = await promise.wait()
        await asyncio.sleep(0)

    assert result.status == "OK"
    assert "authorization required" in result.details.lower()
    assert result.requires_oauth is True
    # Only the periodic health monitor runs; no one-shot retry task.
    assert set(svc._background_tasks) == set(svc._health_monitor_tasks.values())  # pyright: ignore[reportPrivateUsage]
    svc._stop_health_monitor("default", "my-remote-mcp")  # pyright: ignore[reportPrivateUsage]
    updated = svc.models["default"]["my-remote-mcp"]
    assert updated.oauth is not None
    assert updated.oauth.enabled is True


@mock.patch("server.services.mcp_service._fetch_tools_from_mcp_endpoint")
@pytest.mark.asyncio
async def test_install_model_proxy_probe_error_still_starts_background_retry(mock_fetch: AsyncMock, svc: McpService) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    svc._add_custom_model("default", _make_proxy_custom())  # pyright: ignore[reportPrivateUsage]
    mock_fetch.return_value = McpHealthCheckResult(healthy=False, error="connection refused")

    with patch.object(svc, "_fetch_tools_background", new=AsyncMock()) as mock_bg:  # pyright: ignore[reportPrivateUsage]
        promise = await svc._install_model("default", "my-remote-mcp", InstallModelIn())  # pyright: ignore[reportPrivateUsage]
        result = await promise.wait()
        await asyncio.sleep(0)

    assert result.status == "OK"
    assert result.details == "Installed"
    mock_bg.assert_awaited_once_with("default", "my-remote-mcp")


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


@mock.patch("server.services.mcp_service.aiohttp.ClientSession")
@pytest.mark.asyncio
async def test_fetch_tools_from_mcp_endpoint_401_sets_requires_oauth(mock_cls: MagicMock) -> None:
    init_resp = _make_mock_resp(401)
    mock_client = MagicMock()
    mock_client.post.return_value = _make_acm(init_resp)
    mock_cls.return_value = _make_acm(mock_client)

    result = await _fetch_tools_from_mcp_endpoint("http://example.com/mcp", {})

    assert result.healthy is False
    assert result.requires_oauth is True
    assert mock_client.post.call_count == 1  # doesn't try notify/tools-list after a 401


@pytest.mark.parametrize("status", [200, 204, 404, 405])
@pytest.mark.asyncio
async def test_terminate_mcp_session_any_status_returns_none(status: int) -> None:
    mock_client = MagicMock()
    mock_client.delete.return_value = _make_acm(_make_mock_resp(status))

    result = await _terminate_mcp_session(mock_client, "http://example.com/mcp", {"Mcp-Session-Id": "sid-1"})

    assert result is None
    mock_client.delete.assert_called_once()
    args, kwargs = mock_client.delete.call_args
    assert args == ("http://example.com/mcp",)
    assert kwargs["headers"] == {"Mcp-Session-Id": "sid-1"}
    assert kwargs["timeout"].total == 5


@pytest.mark.parametrize(
    "error",
    [
        TimeoutError(),
        aiohttp.ClientConnectorError(MagicMock(), OSError("refused")),
        RuntimeError("boom"),
    ],
)
@pytest.mark.asyncio
async def test_terminate_mcp_session_swallows_errors(error: Exception) -> None:
    mock_client = MagicMock()
    mock_client.delete.side_effect = error

    result = await _terminate_mcp_session(mock_client, "http://example.com/mcp", {})

    assert result is None


def _mcp_probe_client(*post_responses: MagicMock | BaseException, delete: MagicMock | BaseException | None = None) -> MagicMock:
    client = MagicMock()
    client.post.side_effect = [r if isinstance(r, BaseException) else _make_acm(r) for r in post_responses]
    if isinstance(delete, BaseException):
        client.delete.side_effect = delete
    else:
        client.delete.return_value = _make_acm(delete if delete is not None else _make_mock_resp(200))
    return client


@mock.patch("server.services.mcp_service.aiohttp.ClientSession")
@pytest.mark.asyncio
async def test_fetch_tools_from_mcp_endpoint_terminates_session_after_success(mock_cls: MagicMock) -> None:
    client = _mcp_probe_client(
        _make_mock_resp(200, json_data={"result": {}}, session_id="sid-abc"),
        _make_mock_resp(202),
        _make_mock_resp(200, json_data={"result": {"tools": [{"name": "t", "description": "d"}]}}),
    )
    mock_cls.return_value = _make_acm(client)

    result = await _fetch_tools_from_mcp_endpoint("http://example.com/mcp", {"Authorization": "Bearer x"})

    assert result.healthy is True
    assert [t.name for t in result.tools] == ["t"]
    client.delete.assert_called_once()
    args, kwargs = client.delete.call_args
    assert args == ("http://example.com/mcp",)
    assert kwargs["headers"]["Mcp-Session-Id"] == "sid-abc"
    assert kwargs["headers"]["Authorization"] == "Bearer x"


@mock.patch("server.services.mcp_service.aiohttp.ClientSession")
@pytest.mark.asyncio
async def test_fetch_tools_from_mcp_endpoint_terminates_session_after_tools_list_error(mock_cls: MagicMock) -> None:
    client = _mcp_probe_client(
        _make_mock_resp(200, json_data={"result": {}}, session_id="sid-abc"),
        _make_mock_resp(202),
        _make_mock_resp(200, json_data={"error": {"message": "tools unavailable"}}),
    )
    mock_cls.return_value = _make_acm(client)

    result = await _fetch_tools_from_mcp_endpoint("http://example.com/mcp", {})

    assert result.healthy is True
    assert result.error == "tools/list: tools unavailable"
    client.delete.assert_called_once()
    assert client.delete.call_args.kwargs["headers"]["Mcp-Session-Id"] == "sid-abc"


@mock.patch("server.services.mcp_service.aiohttp.ClientSession")
@pytest.mark.asyncio
async def test_fetch_tools_from_mcp_endpoint_terminates_session_after_init_error_in_body(mock_cls: MagicMock) -> None:
    client = _mcp_probe_client(_make_mock_resp(200, json_data={"error": {"message": "not initialized"}}, session_id="sid-abc"))
    mock_cls.return_value = _make_acm(client)

    result = await _fetch_tools_from_mcp_endpoint("http://example.com/mcp", {})

    assert result.healthy is False
    assert result.error == "initialize: not initialized"
    client.delete.assert_called_once()
    assert client.delete.call_args.kwargs["headers"]["Mcp-Session-Id"] == "sid-abc"


@mock.patch("server.services.mcp_service.aiohttp.ClientSession")
@pytest.mark.asyncio
async def test_fetch_tools_from_mcp_endpoint_terminates_session_after_exception(mock_cls: MagicMock) -> None:
    client = _mcp_probe_client(
        _make_mock_resp(200, json_data={"result": {}}, session_id="sid-abc"),
        _make_mock_resp(202),
        TimeoutError(),
    )
    mock_cls.return_value = _make_acm(client)

    result = await _fetch_tools_from_mcp_endpoint("http://example.com/mcp", {})

    assert result.healthy is False
    assert (result.error or "").startswith("TimeoutError")
    client.delete.assert_called_once()
    assert client.delete.call_args.kwargs["headers"]["Mcp-Session-Id"] == "sid-abc"


@mock.patch("server.services.mcp_service.aiohttp.ClientSession")
@pytest.mark.asyncio
async def test_fetch_tools_from_mcp_endpoint_no_session_id_sends_no_delete(mock_cls: MagicMock) -> None:
    client = _mcp_probe_client(
        _make_mock_resp(200, json_data={"result": {}}),
        _make_mock_resp(202),
        _make_mock_resp(200, json_data={"result": {"tools": []}}),
    )
    mock_cls.return_value = _make_acm(client)

    result = await _fetch_tools_from_mcp_endpoint("http://example.com/mcp", {})

    assert result.healthy is True
    client.delete.assert_not_called()


@mock.patch("server.services.mcp_service.aiohttp.ClientSession")
@pytest.mark.asyncio
async def test_fetch_tools_from_mcp_endpoint_connection_error_on_init_sends_no_delete(mock_cls: MagicMock) -> None:
    client = _mcp_probe_client(aiohttp.ClientConnectorError(MagicMock(), OSError("Name or service not known")))
    mock_cls.return_value = _make_acm(client)

    result = await _fetch_tools_from_mcp_endpoint("http://example.com/mcp", {})

    assert result.healthy is False
    assert "Connection refused" in (result.error or "")
    client.delete.assert_not_called()


@pytest.mark.parametrize(
    "delete",
    [_make_mock_resp(405), RuntimeError("boom"), TimeoutError()],
    ids=["405", "raises", "timeout"],
)
@mock.patch("server.services.mcp_service.aiohttp.ClientSession")
@pytest.mark.asyncio
async def test_fetch_tools_from_mcp_endpoint_failing_delete_keeps_result(mock_cls: MagicMock, delete: MagicMock | BaseException) -> None:
    client = _mcp_probe_client(
        _make_mock_resp(200, json_data={"result": {}}, session_id="sid-abc"),
        _make_mock_resp(202),
        _make_mock_resp(200, json_data={"result": {"tools": [{"name": "t", "description": "d"}]}}),
        delete=delete,
    )
    mock_cls.return_value = _make_acm(client)

    result = await _fetch_tools_from_mcp_endpoint("http://example.com/mcp", {})

    assert result.healthy is True
    assert result.error is None
    assert [t.name for t in result.tools] == ["t"]
    client.delete.assert_called_once()


@mock.patch("server.services.mcp_service.aiohttp.ClientSession")
@pytest.mark.asyncio
async def test_fetch_tools_from_sse_endpoint_401_sets_requires_oauth(mock_cls: MagicMock) -> None:
    sse_resp = _make_mock_resp(401)
    mock_client = MagicMock()
    mock_client.get.return_value = _make_acm(sse_resp)
    mock_cls.return_value = _make_acm(mock_client)

    result = await _fetch_tools_from_sse_endpoint("http://example.com/sse", {})

    assert result.healthy is False
    assert result.requires_oauth is True


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


@mock.patch("server.services.mcp_service.asyncio.sleep", new=AsyncMock())
@pytest.mark.asyncio
async def test_fetch_tools_background_returns_early_when_healthy(svc: McpService) -> None:
    healthy_result = McpHealthCheckResult(healthy=True)

    with patch.object(svc, "healthcheck_model", new=AsyncMock(return_value=healthy_result)) as mock_hc:
        await svc._fetch_tools_background("default", "open-websearch")  # pyright: ignore[reportPrivateUsage]

    assert mock_hc.await_count == 1


@mock.patch("server.services.mcp_service.asyncio.sleep", new=AsyncMock())
@pytest.mark.asyncio
async def test_fetch_tools_background_logs_exceptions(svc: McpService) -> None:
    unhealthy_result = McpHealthCheckResult(healthy=False, error="err")

    with (
        patch.object(
            svc,
            "healthcheck_model",
            new=AsyncMock(side_effect=[RuntimeError("boom"), unhealthy_result, unhealthy_result, unhealthy_result, unhealthy_result]),
        ) as mock_hc,
        patch("server.services.mcp_service.logger") as mock_logger,
    ):
        await svc._fetch_tools_background("default", "open-websearch")  # pyright: ignore[reportPrivateUsage]

    assert mock_hc.await_count == 5
    mock_logger.exception.assert_called_once()


@mock.patch("server.services.mcp_service.asyncio.sleep", new=AsyncMock())
@pytest.mark.asyncio
async def test_fetch_tools_background_returns_early_when_requires_oauth(svc: McpService) -> None:
    requires_oauth_result = McpHealthCheckResult(healthy=False, requires_oauth=True, error="401")

    with patch.object(svc, "healthcheck_model", new=AsyncMock(return_value=requires_oauth_result)) as mock_hc:
        await svc._fetch_tools_background("default", "open-websearch")  # pyright: ignore[reportPrivateUsage]

    assert mock_hc.await_count == 1


@mock.patch("server.services.mcp_service.asyncio.sleep", new=AsyncMock())
@pytest.mark.asyncio
async def test_fetch_tools_background_exhausts_all_retries(svc: McpService) -> None:
    unhealthy_result = McpHealthCheckResult(healthy=False, error="err")

    with patch.object(svc, "healthcheck_model", new=AsyncMock(return_value=unhealthy_result)) as mock_hc:
        await svc._fetch_tools_background("default", "open-websearch")  # pyright: ignore[reportPrivateUsage]

    assert mock_hc.await_count == 5


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


def test_apply_healthcheck_result_notifies_parent_when_registration_changes(svc: McpService) -> None:
    props = ModelProps(private=False, type="mcp", endpoints=[])
    model = MagicMock(spec=SrvMcpModel)
    model.model_props = props
    reg_model = MagicMock()
    reg_model.props = props
    reg_model.healthy = False
    svc.endpoint_registry.mcp_endpoints.models.get.return_value = {"v1": reg_model}  # pyright: ignore[reportAttributeAccessIssue]
    result = McpHealthCheckResult(healthy=True, transport="sse", tools=[McpToolInfo(name="t", description="d")])

    svc._apply_healthcheck_result(model, "my-prefix", result)  # pyright: ignore[reportPrivateUsage]

    svc.endpoint_registry.parent_infra.send_models_list.assert_called_once()  # pyright: ignore[reportAttributeAccessIssue]


def test_apply_healthcheck_result_does_not_notify_parent_when_nothing_changed(svc: McpService) -> None:
    props = ModelProps(private=False, type="mcp", endpoints=[], transport="sse")
    model = MagicMock(spec=SrvMcpModel)
    model.model_props = props
    reg_model = MagicMock()
    reg_model.props = props
    reg_model.healthy = True
    svc.endpoint_registry.mcp_endpoints.models.get.return_value = {"v1": reg_model}  # pyright: ignore[reportAttributeAccessIssue]
    result = McpHealthCheckResult(healthy=True, transport="sse", tools=[])

    svc._apply_healthcheck_result(model, "my-prefix", result)  # pyright: ignore[reportPrivateUsage]

    svc.endpoint_registry.parent_infra.send_models_list.assert_not_called()  # pyright: ignore[reportAttributeAccessIssue]


def test_apply_healthcheck_result_no_transport_skips_transport_update(svc: McpService) -> None:
    model = MagicMock(spec=SrvMcpModel)
    model.model_props = MagicMock()
    reg_model = MagicMock()
    svc.endpoint_registry.mcp_endpoints.models.get.return_value = {"v1": reg_model}  # pyright: ignore[reportAttributeAccessIssue]
    result = McpHealthCheckResult(healthy=True, transport=None)

    svc._apply_healthcheck_result(model, "my-prefix", result)  # pyright: ignore[reportPrivateUsage]

    assert model.model_props.tools == []
    assert not hasattr(model.model_props, "transport") or model.model_props.transport != "streamable_http"


def _make_installed_model_info(base_url: str = "http://172.20.0.2:3000", prefix: str = "mymcp") -> MagicMock:
    info = MagicMock(spec=ModelInstalledInfo)
    info.base_url = base_url
    info.headers = {}
    info.prefix = prefix
    info.registration_id = "existing-reg-id"
    return info


@pytest.mark.asyncio
async def test_healthcheck_model_model_not_installed_raises_400(svc: McpService) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))

    with pytest.raises(HTTPException) as exc_info:
        await svc.healthcheck_model("default", "nonexistent")

    assert exc_info.value.status_code == 400
    assert "not installed" in exc_info.value.detail


@pytest.mark.asyncio
async def test_healthcheck_model_records_invalid_request_outcome_metric_when_model_unknown(svc: McpService) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    fake_instruments = MagicMock()

    with mock.patch("server.services.mcp_service.tracer.mcp_instruments", return_value=fake_instruments), pytest.raises(HTTPException):
        await svc.healthcheck_model("default", "nonexistent")

    assert fake_instruments.healthcheck_count.add.call_count == 1
    # A 4xx from bad input is not a genuine operational failure — recording its ~0ms duration would
    # skew p50/p95 and mask real MCP server slowness, so it's excluded from the duration histogram.
    assert fake_instruments.healthcheck_duration_ms.record.call_count == 0
    attrs = fake_instruments.healthcheck_count.add.call_args[0][1]
    assert attrs["mcp.outcome"] == "invalid_request"


@pytest.mark.asyncio
async def test_healthcheck_model_instance_not_installed_raises_400(svc: McpService) -> None:
    svc.instances_info["default"].installed = None

    with pytest.raises(HTTPException) as exc_info:
        await svc.healthcheck_model("default", "open-websearch")

    assert exc_info.value.status_code == 400
    assert "not installed" in exc_info.value.detail


@pytest.mark.asyncio
async def test_healthcheck_model_records_invalid_request_outcome_metric_when_instance_not_installed(svc: McpService) -> None:
    svc.instances_info["default"].installed = None
    fake_instruments = MagicMock()

    with mock.patch("server.services.mcp_service.tracer.mcp_instruments", return_value=fake_instruments), pytest.raises(HTTPException):
        await svc.healthcheck_model("default", "open-websearch")

    assert fake_instruments.healthcheck_count.add.call_count == 1
    assert fake_instruments.healthcheck_duration_ms.record.call_count == 0
    attrs = fake_instruments.healthcheck_count.add.call_args[0][1]
    assert attrs["mcp.outcome"] == "invalid_request"


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
@pytest.mark.asyncio
async def test_healthcheck_model_records_otel_span_and_metrics_when_enabled(mock_fetch: AsyncMock, svc: McpService) -> None:
    installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    installed.models["open-websearch"] = _make_installed_model_info()
    svc.instances_info["default"].installed = installed
    srv_model = svc.models["default"]["open-websearch"]
    srv_model.kind = "proxy"
    srv_model.proxy_transport = "streamable_http"
    mock_fetch.return_value = McpHealthCheckResult(healthy=True, transport="streamable_http", tools=[])

    fake_span = MagicMock()
    fake_instruments = MagicMock()

    with (
        mock.patch("server.services.mcp_service.tracer.span") as mock_span,
        mock.patch("server.services.mcp_service.tracer.mcp_instruments", return_value=fake_instruments),
    ):
        mock_span.return_value.__enter__.return_value = fake_span
        result = await svc.healthcheck_model("default", "open-websearch")

    assert result.healthy is True
    assert fake_instruments.healthcheck_count.add.call_count == 1
    assert fake_instruments.healthcheck_duration_ms.record.call_count == 1
    attrs = fake_instruments.healthcheck_count.add.call_args[0][1]
    assert attrs["mcp.outcome"] == "healthy"
    assert attrs["mcp.instance"] == "default"
    assert attrs["mcp.model_id"] == "open-websearch"
    duration_value, duration_attrs = fake_instruments.healthcheck_duration_ms.record.call_args[0]
    assert duration_value >= 0
    assert duration_attrs == attrs
    span_attrs = {c[0][0]: c[0][1] for c in fake_span.set_attribute.call_args_list}
    assert span_attrs["mcp.outcome"] == "healthy"
    assert span_attrs["mcp.transport"] == "streamable_http"
    assert span_attrs["mcp.tool_count"] == 0


@mock.patch("server.services.mcp_service._fetch_tools_from_mcp_endpoint")
@pytest.mark.asyncio
async def test_healthcheck_model_records_unhealthy_outcome_metric(mock_fetch: AsyncMock, svc: McpService) -> None:
    installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    installed.models["open-websearch"] = _make_installed_model_info()
    svc.instances_info["default"].installed = installed
    srv_model = svc.models["default"]["open-websearch"]
    srv_model.kind = "proxy"
    srv_model.proxy_transport = "streamable_http"
    mock_fetch.return_value = McpHealthCheckResult(healthy=False, error="refused")
    fake_instruments = MagicMock()

    with mock.patch("server.services.mcp_service.tracer.mcp_instruments", return_value=fake_instruments):
        result = await svc.healthcheck_model("default", "open-websearch")

    assert result.healthy is False
    attrs = fake_instruments.healthcheck_count.add.call_args[0][1]
    assert attrs["mcp.outcome"] == "unhealthy"


@mock.patch("server.services.mcp_service._fetch_tools_from_mcp_endpoint")
@pytest.mark.asyncio
async def test_healthcheck_model_proxy_401_triggers_auto_detect_oauth(mock_fetch: AsyncMock, svc: McpService) -> None:
    installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    installed.models["open-websearch"] = _make_installed_model_info()
    svc.instances_info["default"].installed = installed
    srv_model = svc.models["default"]["open-websearch"]
    srv_model.kind = "proxy"
    srv_model.proxy_transport = "streamable_http"
    mock_fetch.return_value = McpHealthCheckResult(healthy=False, requires_oauth=True, error="401")

    fake_instruments = MagicMock()
    with (
        patch.object(svc, "_auto_detect_oauth", new=AsyncMock()) as mock_detect,
        mock.patch("server.services.mcp_service.tracer.mcp_instruments", return_value=fake_instruments),
    ):
        result = await svc.healthcheck_model("default", "open-websearch")

    assert result.requires_oauth is True
    mock_detect.assert_awaited_once_with("default", "open-websearch")
    attrs = fake_instruments.healthcheck_count.add.call_args[0][1]
    assert attrs["mcp.outcome"] == "oauth_required"


@mock.patch("server.services.mcp_service._fetch_tools_from_mcp_endpoint")
@pytest.mark.asyncio
async def test_healthcheck_model_proxy_already_oauth_enabled_skips_auto_detect(mock_fetch: AsyncMock, svc: McpService) -> None:
    installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    installed.models["open-websearch"] = _make_installed_model_info()
    svc.instances_info["default"].installed = installed
    srv_model = svc.models["default"]["open-websearch"]
    srv_model.kind = "proxy"
    srv_model.proxy_transport = "streamable_http"
    srv_model.oauth = McpOAuthConfig(enabled=True)
    mock_fetch.return_value = McpHealthCheckResult(healthy=False, requires_oauth=True, error="401")

    with patch.object(svc, "_auto_detect_oauth", new=AsyncMock()) as mock_detect:
        await svc.healthcheck_model("default", "open-websearch")

    mock_detect.assert_not_awaited()


@mock.patch("server.services.mcp_service._fetch_tools_from_mcp_endpoint")
@pytest.mark.asyncio
async def test_healthcheck_model_proxy_attaches_live_oauth_bearer_token(mock_fetch: AsyncMock, svc: McpService) -> None:
    installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    installed.models["open-websearch"] = _make_installed_model_info()
    svc.instances_info["default"].installed = installed
    srv_model = svc.models["default"]["open-websearch"]
    srv_model.kind = "proxy"
    srv_model.proxy_transport = "streamable_http"
    srv_model.oauth = McpOAuthConfig(enabled=True, access_token="tok-123")
    mock_fetch.return_value = McpHealthCheckResult(healthy=True)

    await svc.healthcheck_model("default", "open-websearch")

    assert mock_fetch.call_args[0][1]["Authorization"] == "Bearer tok-123"


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
async def test_healthcheck_model_docker_transport_mismatch_reregisters_live_proxy(
    mock_mcp: AsyncMock, mock_sse: AsyncMock, svc: McpService
) -> None:
    """If only the non-registered transport actually works, the live proxy must be re-registered at it."""
    model_info = _make_installed_model_info()
    installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    installed.models["open-websearch"] = model_info
    svc.instances_info["default"].installed = installed
    srv_model = svc.models["default"]["open-websearch"]
    srv_model.kind = "custom"
    srv_model.proxy_transport = "streamable_http"
    mock_mcp.return_value = McpHealthCheckResult(healthy=False, error="mcp failed")
    mock_sse.return_value = McpHealthCheckResult(healthy=True, transport="sse", tools=[])
    svc.endpoint_registry.register_mcp_sse_endpoint_as_proxy.return_value = "new-sse-reg-id"  # pyright: ignore[reportAttributeAccessIssue]

    await svc.healthcheck_model("default", "open-websearch")

    svc.endpoint_registry.register_mcp_sse_endpoint_as_proxy.assert_called_once()  # pyright: ignore[reportAttributeAccessIssue]
    svc.endpoint_registry.register_mcp_endpoint_as_proxy.assert_not_called()  # pyright: ignore[reportAttributeAccessIssue]
    svc.endpoint_registry.unregister_mcp_endpoint.assert_called_once_with(  # pyright: ignore[reportAttributeAccessIssue]
        model_info.prefix, "existing-reg-id"
    )
    assert model_info.registration_id == "new-sse-reg-id"
    assert srv_model.proxy_transport == "sse"


@mock.patch("server.services.mcp_service._fetch_tools_from_sse_endpoint")
@mock.patch("server.services.mcp_service._fetch_tools_from_mcp_endpoint")
@pytest.mark.asyncio
async def test_healthcheck_model_docker_transport_match_does_not_reregister(
    mock_mcp: AsyncMock, mock_sse: AsyncMock, svc: McpService
) -> None:
    """If the registered transport already works, the live proxy must not be touched."""
    model_info = _make_installed_model_info()
    installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    installed.models["open-websearch"] = model_info
    svc.instances_info["default"].installed = installed
    srv_model = svc.models["default"]["open-websearch"]
    srv_model.kind = "custom"
    srv_model.proxy_transport = "streamable_http"
    mock_mcp.return_value = McpHealthCheckResult(healthy=True, transport="streamable_http", tools=[])

    await svc.healthcheck_model("default", "open-websearch")

    svc.endpoint_registry.register_mcp_sse_endpoint_as_proxy.assert_not_called()  # pyright: ignore[reportAttributeAccessIssue]
    svc.endpoint_registry.register_mcp_endpoint_as_proxy.assert_not_called()  # pyright: ignore[reportAttributeAccessIssue]
    svc.endpoint_registry.unregister_mcp_endpoint.assert_not_called()  # pyright: ignore[reportAttributeAccessIssue]
    assert model_info.registration_id == "existing-reg-id"
    assert srv_model.proxy_transport == "streamable_http"


@mock.patch("server.services.mcp_service._fetch_tools_from_sse_endpoint")
@mock.patch("server.services.mcp_service._fetch_tools_from_mcp_endpoint")
@pytest.mark.asyncio
async def test_healthcheck_model_docker_transport_mismatch_with_error_does_not_reregister(
    mock_mcp: AsyncMock, mock_sse: AsyncMock, svc: McpService
) -> None:
    """A `healthy=True` fallback result that only got partway through the handshake (`error` set,
    e.g. a timeout waiting for tools/list) must not be treated as proof the other transport works
    — re-registering on it would permanently point real traffic at a transport that never actually
    returned tools either, reproducing the same bug this re-registration logic exists to fix."""
    model_info = _make_installed_model_info()
    installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    installed.models["open-websearch"] = model_info
    svc.instances_info["default"].installed = installed
    srv_model = svc.models["default"]["open-websearch"]
    srv_model.kind = "custom"
    srv_model.proxy_transport = "streamable_http"
    mock_mcp.return_value = McpHealthCheckResult(healthy=False, error="mcp failed")
    mock_sse.return_value = McpHealthCheckResult(healthy=True, transport="sse", error="Timeout waiting for MCP response")

    result = await svc.healthcheck_model("default", "open-websearch")

    assert result.healthy is True
    svc.endpoint_registry.register_mcp_sse_endpoint_as_proxy.assert_not_called()  # pyright: ignore[reportAttributeAccessIssue]
    svc.endpoint_registry.register_mcp_endpoint_as_proxy.assert_not_called()  # pyright: ignore[reportAttributeAccessIssue]
    svc.endpoint_registry.unregister_mcp_endpoint.assert_not_called()  # pyright: ignore[reportAttributeAccessIssue]
    assert model_info.registration_id == "existing-reg-id"
    assert srv_model.proxy_transport == "streamable_http"


@mock.patch("server.services.mcp_service._fetch_tools_from_mcp_endpoint")
@mock.patch("server.services.mcp_service._fetch_tools_from_sse_endpoint")
@pytest.mark.asyncio
async def test_healthcheck_model_docker_transport_mismatch_reregisters_live_proxy_reverse_direction(
    mock_sse: AsyncMock, mock_mcp: AsyncMock, svc: McpService
) -> None:
    """Mirror of the streamable_http->sse case: registered `sse`, only `streamable_http` actually works."""
    model_info = _make_installed_model_info()
    installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    installed.models["open-websearch"] = model_info
    svc.instances_info["default"].installed = installed
    srv_model = svc.models["default"]["open-websearch"]
    srv_model.kind = "custom"
    srv_model.proxy_transport = "sse"
    mock_sse.return_value = McpHealthCheckResult(healthy=False, error="sse failed")
    mock_mcp.return_value = McpHealthCheckResult(healthy=True, transport="streamable_http", tools=[])
    svc.endpoint_registry.register_mcp_endpoint_as_proxy.return_value = "new-mcp-reg-id"  # pyright: ignore[reportAttributeAccessIssue]

    await svc.healthcheck_model("default", "open-websearch")

    svc.endpoint_registry.register_mcp_endpoint_as_proxy.assert_called_once()  # pyright: ignore[reportAttributeAccessIssue]
    svc.endpoint_registry.register_mcp_sse_endpoint_as_proxy.assert_not_called()  # pyright: ignore[reportAttributeAccessIssue]
    svc.endpoint_registry.unregister_mcp_endpoint.assert_called_once_with(  # pyright: ignore[reportAttributeAccessIssue]
        model_info.prefix, "existing-reg-id"
    )
    assert model_info.registration_id == "new-mcp-reg-id"
    assert srv_model.proxy_transport == "streamable_http"


@mock.patch("server.services.mcp_service._fetch_tools_from_sse_endpoint")
@mock.patch("server.services.mcp_service._fetch_tools_from_mcp_endpoint")
@pytest.mark.asyncio
async def test_healthcheck_model_docker_transport_mismatch_reregisters_with_merged_headers(
    mock_mcp: AsyncMock, mock_sse: AsyncMock, svc: McpService
) -> None:
    """The re-registered proxy must carry the model's static headers merged with the install-time ones."""
    model_info = _make_installed_model_info()
    model_info.headers = {"X-Install-Header": "install-value"}
    installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    installed.models["open-websearch"] = model_info
    svc.instances_info["default"].installed = installed
    srv_model = svc.models["default"]["open-websearch"]
    srv_model.kind = "custom"
    srv_model.proxy_transport = "streamable_http"
    srv_model.headers = {"X-Static-Header": "static-value"}
    mock_mcp.return_value = McpHealthCheckResult(healthy=False, error="mcp failed")
    mock_sse.return_value = McpHealthCheckResult(healthy=True, transport="sse", tools=[])
    svc.endpoint_registry.register_mcp_sse_endpoint_as_proxy.return_value = "new-sse-reg-id"  # pyright: ignore[reportAttributeAccessIssue]

    await svc.healthcheck_model("default", "open-websearch")

    _, kwargs = svc.endpoint_registry.register_mcp_sse_endpoint_as_proxy.call_args  # pyright: ignore[reportAttributeAccessIssue]
    assert kwargs["options"].headers == {
        "X-Static-Header": "static-value",
        "X-Install-Header": "install-value",
    }


@mock.patch("server.services.mcp_service._fetch_tools_from_sse_endpoint")
@mock.patch("server.services.mcp_service._fetch_tools_from_mcp_endpoint")
@pytest.mark.asyncio
async def test_healthcheck_model_docker_transport_mismatch_unregister_failure_is_logged_not_raised(
    mock_mcp: AsyncMock, mock_sse: AsyncMock, svc: McpService, caplog: pytest.LogCaptureFixture
) -> None:
    """If unregistering the stale proxy fails, the failure is logged and bookkeeping still moves to the new registration."""
    model_info = _make_installed_model_info()
    installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    installed.models["open-websearch"] = model_info
    svc.instances_info["default"].installed = installed
    srv_model = svc.models["default"]["open-websearch"]
    srv_model.kind = "custom"
    srv_model.proxy_transport = "streamable_http"
    mock_mcp.return_value = McpHealthCheckResult(healthy=False, error="mcp failed")
    mock_sse.return_value = McpHealthCheckResult(healthy=True, transport="sse", tools=[])
    svc.endpoint_registry.register_mcp_sse_endpoint_as_proxy.return_value = "new-sse-reg-id"  # pyright: ignore[reportAttributeAccessIssue]
    svc.endpoint_registry.unregister_mcp_endpoint.side_effect = RuntimeError("registry notification failed")  # pyright: ignore[reportAttributeAccessIssue]

    with caplog.at_level(logging.ERROR, logger="uvicorn"):
        result = await svc.healthcheck_model("default", "open-websearch")

    assert result.healthy is True
    assert model_info.registration_id == "new-sse-reg-id"
    assert srv_model.proxy_transport == "sse"
    assert any("Failed to unregister stale MCP proxy registration" in record.message for record in caplog.records)


@mock.patch("server.services.mcp_service._fetch_tools_from_sse_endpoint")
@mock.patch("server.services.mcp_service._fetch_tools_from_mcp_endpoint")
@pytest.mark.asyncio
async def test_healthcheck_model_docker_transport_mismatch_register_failure_is_logged_and_raised(
    mock_mcp: AsyncMock, mock_sse: AsyncMock, svc: McpService, caplog: pytest.LogCaptureFixture
) -> None:
    """If registering the new proxy fails, the failure is logged, the exception propagates, and the old registration is untouched."""
    model_info = _make_installed_model_info()
    installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    installed.models["open-websearch"] = model_info
    svc.instances_info["default"].installed = installed
    srv_model = svc.models["default"]["open-websearch"]
    srv_model.kind = "custom"
    srv_model.proxy_transport = "streamable_http"
    mock_mcp.return_value = McpHealthCheckResult(healthy=False, error="mcp failed")
    mock_sse.return_value = McpHealthCheckResult(healthy=True, transport="sse", tools=[])
    svc.endpoint_registry.register_mcp_sse_endpoint_as_proxy.side_effect = RuntimeError("registry rejected registration")  # pyright: ignore[reportAttributeAccessIssue]

    with caplog.at_level(logging.ERROR, logger="uvicorn"), pytest.raises(RuntimeError, match="registry rejected registration"):
        await svc.healthcheck_model("default", "open-websearch")

    svc.endpoint_registry.unregister_mcp_endpoint.assert_not_called()  # pyright: ignore[reportAttributeAccessIssue]
    assert model_info.registration_id == "existing-reg-id"
    assert srv_model.proxy_transport == "streamable_http"
    assert any("Failed to register MCP proxy for prefix" in record.message for record in caplog.records)


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


@pytest.mark.asyncio
async def test_install_model_registration_failure_rolls_back_model_without_unregister(
    svc: McpService, deps: dict[str, Any], tmp_path: Path
) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    model_id = "open-websearch"
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=(12345, True, False))
    deps["docker_service"].stop_docker = AsyncMock()
    deps["docker_service"].get_container_host.return_value = "172.20.0.2"
    deps["docker_service"].get_container_port.return_value = 3000
    deps["endpoint_registry"].register_mcp_endpoint_as_proxy.side_effect = RuntimeError("registry down")

    with (
        patch.object(svc, "_verify_docker_image", new=AsyncMock()),
        patch.object(svc, "_download_image_or_set_progress", new=AsyncMock()),
        patch.object(svc, "_get_working_dir", return_value=tmp_path),
        patch("server.services.mcp_service.get_base_url", return_value="http://172.20.0.2:3000"),
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn())  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(RuntimeError):
            await promise.wait()

    info = svc.get_instance_installed_info("default")
    assert model_id not in info.models
    assert deps["endpoint_registry"].unregister_mcp_endpoint.call_count == 0
    deps["docker_service"].stop_docker.assert_called_once()


@pytest.mark.asyncio
async def test_install_model_post_registration_failure_unregisters_and_rolls_back(
    svc: McpService, deps: dict[str, Any], tmp_path: Path
) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    model_id = "open-websearch"
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=(12345, True, False))
    deps["docker_service"].stop_docker = AsyncMock()
    deps["docker_service"].get_container_host.return_value = "172.20.0.2"
    deps["docker_service"].get_container_port.return_value = 3000
    deps["endpoint_registry"].register_mcp_endpoint_as_proxy.return_value = "reg-id"
    bad_tasks: MagicMock = MagicMock()
    bad_tasks.add.side_effect = RuntimeError("add failed")

    with (
        patch.object(svc, "_verify_docker_image", new=AsyncMock()),
        patch.object(svc, "_download_image_or_set_progress", new=AsyncMock()),
        patch.object(svc, "_get_working_dir", return_value=tmp_path),
        patch("server.services.mcp_service.get_base_url", return_value="http://172.20.0.2:3000"),
        patch.object(svc, "_background_tasks", bad_tasks),  # pyright: ignore[reportPrivateUsage]
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn())  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(RuntimeError):
            await promise.wait()

    info = svc.get_instance_installed_info("default")
    assert model_id not in info.models
    assert deps["endpoint_registry"].unregister_mcp_endpoint.call_count == 1
    deps["docker_service"].stop_docker.assert_called_once()


@pytest.mark.asyncio
async def test_install_model_unregister_failure_during_rollback_does_not_skip_docker_rollback(
    svc: McpService, deps: dict[str, Any], tmp_path: Path
) -> None:
    """If the endpoint unregister step itself raises during rollback, the docker container must still be torn down."""
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    model_id = "open-websearch"
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=(12345, True, False))
    deps["docker_service"].stop_docker = AsyncMock()
    deps["docker_service"].get_container_host.return_value = "172.20.0.2"
    deps["docker_service"].get_container_port.return_value = 3000
    deps["endpoint_registry"].register_mcp_endpoint_as_proxy.return_value = "reg-id"
    deps["endpoint_registry"].unregister_mcp_endpoint.side_effect = RuntimeError("unregister exploded")
    bad_tasks: MagicMock = MagicMock()
    bad_tasks.add.side_effect = RuntimeError("add failed")

    with (
        patch.object(svc, "_verify_docker_image", new=AsyncMock()),
        patch.object(svc, "_download_image_or_set_progress", new=AsyncMock()),
        patch.object(svc, "_get_working_dir", return_value=tmp_path),
        patch("server.services.mcp_service.get_base_url", return_value="http://172.20.0.2:3000"),
        patch.object(svc, "_background_tasks", bad_tasks),  # pyright: ignore[reportPrivateUsage]
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn())  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(RuntimeError, match="add failed"):
            await promise.wait()

    info = svc.get_instance_installed_info("default")
    assert model_id not in info.models
    deps["docker_service"].stop_docker.assert_called_once()


@pytest.mark.asyncio
async def test_install_model_registration_failure_skips_rollback_when_already_removed(
    svc: McpService, deps: dict[str, Any], tmp_path: Path
) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    model_id = "open-websearch"
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=(12345, True, False))
    deps["docker_service"].stop_docker = AsyncMock()
    deps["docker_service"].get_container_host.return_value = "172.20.0.2"
    deps["docker_service"].get_container_port.return_value = 3000

    def side_effect(*args: object, **kwargs: object) -> None:
        info = svc.get_instance_installed_info("default")
        info.models.pop(model_id, None)
        raise RuntimeError("registry down")

    deps["endpoint_registry"].register_mcp_endpoint_as_proxy.side_effect = side_effect

    with (
        patch.object(svc, "_verify_docker_image", new=AsyncMock()),
        patch.object(svc, "_download_image_or_set_progress", new=AsyncMock()),
        patch.object(svc, "_get_working_dir", return_value=tmp_path),
        patch("server.services.mcp_service.get_base_url", return_value="http://172.20.0.2:3000"),
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn())  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(RuntimeError):
            await promise.wait()

    info = svc.get_instance_installed_info("default")
    assert model_id not in info.models
    deps["docker_service"].stop_docker.assert_called_once()


@pytest.mark.asyncio
async def test_install_model_cancelled_after_docker_start_rolls_back_container(
    svc: McpService, deps: dict[str, Any], tmp_path: Path
) -> None:
    """Cancelling after `install_and_run_docker` succeeds (e.g. during registration) must tear the container back down."""
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    model_id = "open-websearch"
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=(12345, True, False))
    deps["docker_service"].stop_docker = AsyncMock()
    deps["docker_service"].get_container_host.return_value = "172.20.0.2"
    deps["docker_service"].get_container_port.return_value = 3000
    deps["endpoint_registry"].register_mcp_endpoint_as_proxy.side_effect = asyncio.CancelledError()

    with (
        patch.object(svc, "_verify_docker_image", new=AsyncMock()),
        patch.object(svc, "_download_image_or_set_progress", new=AsyncMock()),
        patch.object(svc, "_get_working_dir", return_value=tmp_path),
        patch("server.services.mcp_service.get_base_url", return_value="http://172.20.0.2:3000"),
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn())  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(asyncio.CancelledError):
            await promise.wait()

    info = svc.get_instance_installed_info("default")
    assert model_id not in info.models
    deps["docker_service"].stop_docker.assert_called_once()
    assert deps["endpoint_registry"].unregister_mcp_endpoint.call_count == 0


@pytest.mark.asyncio
async def test_install_model_promise_cancel_while_docker_starting_rolls_back_container(
    svc: McpService, deps: dict[str, Any], tmp_path: Path
) -> None:
    """Same as above but with a genuine `promise.cancel()` (real task cancellation at a real await point,
    like `cancel_model_install()` performs) instead of a synchronously-raised CancelledError."""
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    model_id = "open-websearch"
    deps["docker_service"].stop_docker = AsyncMock()
    docker_up_started = asyncio.Event()

    async def hang_after_docker_up_starts(*args: object, **kwargs: object) -> tuple[int, bool, bool]:
        docker_up_started.set()
        await asyncio.Event().wait()  # only a real task cancellation ends this
        raise AssertionError("unreachable")

    deps["docker_service"].install_and_run_docker = AsyncMock(side_effect=hang_after_docker_up_starts)

    with (
        patch.object(svc, "_verify_docker_image", new=AsyncMock()),
        patch.object(svc, "_download_image_or_set_progress", new=AsyncMock()),
        patch.object(svc, "_get_working_dir", return_value=tmp_path),
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn())  # pyright: ignore[reportPrivateUsage]
        await asyncio.wait_for(docker_up_started.wait(), timeout=1)

        promise.cancel()
        with pytest.raises(asyncio.CancelledError):
            await promise.task

    info = svc.get_instance_installed_info("default")
    assert model_id not in info.models
    deps["docker_service"].stop_docker.assert_called_once()


@pytest.mark.asyncio
async def test_install_model_failure_before_docker_start_skips_rollback(svc: McpService, deps: dict[str, Any], tmp_path: Path) -> None:
    """A failure before `install_and_run_docker` is even called has nothing to roll back yet."""
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    model_id = "open-websearch"
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=(12345, True, False))
    deps["docker_service"].stop_docker = AsyncMock()

    with (
        patch.object(svc, "_verify_docker_image", new=AsyncMock()),
        patch.object(svc, "_download_image_or_set_progress", new=AsyncMock(side_effect=RuntimeError("download failed"))),
        patch.object(svc, "_get_working_dir", return_value=tmp_path),
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn())  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(RuntimeError):
            await promise.wait()

    info = svc.get_instance_installed_info("default")
    assert model_id not in info.models
    deps["docker_service"].install_and_run_docker.assert_not_called()
    deps["docker_service"].stop_docker.assert_not_called()
    assert deps["endpoint_registry"].unregister_mcp_endpoint.call_count == 0


def _make_proxy_custom(
    custom_id: str = "uuid-proxy-1",
    model_id: str = "my-remote-mcp",
    oauth: dict[str, Any] | None = None,
) -> CustomModel:
    data: dict[str, Any] = {
        "kind": "proxy",
        "id": model_id,
        "name": model_id,
        "server_url": "http://mcp.example.com/mcp",
        "default_prefix": model_id,
    }
    if oauth is not None:
        data["oauth"] = oauth
    return CustomModel(id=custom_id, data=data)


def test_get_custom_spec_proxy_with_oauth_redacts_secrets_and_tokens(svc: McpService) -> None:
    model = MagicMock(spec=SrvMcpModel)
    model.custom = "uuid-proxy"
    model.kind = "proxy"
    model.proxy_url = "http://remote-mcp.example.com/mcp"
    model.proxy_transport = "streamable_http"
    model.default_prefix = "remote-mcp"
    model.headers = None
    model.description = ""
    model.repository_url = None
    model.oauth = McpOAuthConfig(
        enabled=True,
        client_id="client-1",
        client_secret="super-secret",
        scope="tools:read",
        access_token="access-tok",
        refresh_token="refresh-tok",
    )

    spec = svc._get_custom_spec("remote-mcp", model)  # pyright: ignore[reportPrivateUsage]

    assert spec is not None
    assert spec["oauth"] == {"enabled": True, "client_id": "client-1", "scope": "tools:read"}
    assert "client_secret" not in spec["oauth"]
    assert "access_token" not in spec["oauth"]
    assert "refresh_token" not in spec["oauth"]


def test_srv_mcp_proxy_model_without_oauth_key_round_trips_to_none() -> None:
    parsed = SrvMcpProxyModel(id="m1", name="m1", server_url="http://mcp.example.com/mcp")

    assert parsed.oauth is None


@pytest.mark.asyncio
async def test_update_custom_model_proxy_preserves_oauth_secrets_and_tokens(svc: McpService) -> None:
    old = _make_proxy_custom(
        oauth={
            "enabled": True,
            "client_id": "client-1",
            "client_secret": "super-secret",
            "access_token": "access-tok",
            "refresh_token": "refresh-tok",
            "token_expires_at": 12345.0,
        }
    )
    svc._add_custom_model("default", old)  # pyright: ignore[reportPrivateUsage]
    new_data = dict(old.data)
    new_data["oauth"] = {"enabled": True, "client_id": "client-1", "scope": "tools:read"}

    await svc._update_custom_model("default", old, new_data)  # pyright: ignore[reportPrivateUsage]

    updated = svc.models["default"]["my-remote-mcp"]
    assert updated.oauth is not None
    assert updated.oauth.client_secret == "super-secret"
    assert updated.oauth.access_token == "access-tok"
    assert updated.oauth.refresh_token == "refresh-tok"
    assert updated.oauth.token_expires_at == 12345.0
    assert updated.oauth.scope == "tools:read"


@pytest.mark.asyncio
async def test_update_custom_model_proxy_preserves_oauth_when_new_data_omits_it(svc: McpService) -> None:
    old = _make_proxy_custom(
        oauth={
            "enabled": True,
            "client_id": "client-1",
            "client_secret": "super-secret",
            "access_token": "access-tok",
            "refresh_token": "refresh-tok",
            "token_expires_at": 12345.0,
        }
    )
    svc._add_custom_model("default", old)  # pyright: ignore[reportPrivateUsage]
    new_data = dict(old.data)
    del new_data["oauth"]

    await svc._update_custom_model("default", old, new_data)  # pyright: ignore[reportPrivateUsage]

    updated = svc.models["default"]["my-remote-mcp"]
    assert updated.oauth is not None
    assert updated.oauth.client_secret == "super-secret"
    assert updated.oauth.access_token == "access-tok"
    assert updated.oauth.refresh_token == "refresh-tok"


@mock.patch("server.services.mcp_service._fetch_tools_from_mcp_endpoint")
@pytest.mark.asyncio
async def test_edit_model_preserves_oauth_secrets_through_uninstall_reinstall(mock_fetch: AsyncMock, svc: McpService) -> None:
    """DFINFRA-271 regression: OAuth secret preservation still holds when editing an *installed* proxy
    MCP server through `edit_model`'s uninstall->update->reinstall orchestration, not just a direct,
    not-installed `_update_custom_model` call (the scenario the pre-existing tests above cover)."""
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    old = _make_proxy_custom(
        oauth={
            "enabled": True,
            "client_id": "client-1",
            "client_secret": "super-secret",
            "access_token": "access-tok",
            "refresh_token": "refresh-tok",
            "token_expires_at": 12345.0,
        }
    )
    svc._add_custom_model("default", old)  # pyright: ignore[reportPrivateUsage]
    svc.instances_info["default"].config.custom = [old]
    mock_fetch.return_value = McpHealthCheckResult(healthy=True, transport="streamable_http", tools=[])

    with patch.object(svc, "_save", new=AsyncMock()):
        install_promise = await svc.install_model("default", "my-remote-mcp", InstallModelIn())
        await install_promise.wait()
        await asyncio.sleep(0)
        installed = svc.instances_info["default"].installed
        assert installed is not None
        assert "my-remote-mcp" in installed.models

        new_data = dict(old.data)
        new_data["oauth"] = {"enabled": True, "client_id": "client-1", "scope": "tools:read"}

        promise = await svc.edit_model("default", old.id, AddCustomModelIn(spec=new_data))
        assert promise is not None
        await promise.wait()
        await asyncio.sleep(0)

    updated = svc.models["default"]["my-remote-mcp"]
    assert updated.oauth is not None
    assert updated.oauth.client_secret == "super-secret"
    assert updated.oauth.access_token == "access-tok"
    assert updated.oauth.refresh_token == "refresh-tok"
    assert updated.oauth.token_expires_at == 12345.0
    assert updated.oauth.scope == "tools:read"
    installed = svc.instances_info["default"].installed
    assert installed is not None
    assert "my-remote-mcp" in installed.models


@pytest.mark.asyncio
async def test_edit_model_rejects_when_oauth_flow_pending(svc: McpService) -> None:
    """`_validate_edit`'s MCP override rejects the edit upfront, via `Base2Service.edit_model`, before
    its uninstall->update->reinstall orchestration runs - not just deep inside `_update_custom_model`
    (see `test_update_custom_model_proxy_raises_when_oauth_flow_pending`), so an edit that was always
    going to be rejected doesn't uninstall the model for nothing first."""
    old = _make_proxy_custom()
    svc._add_custom_model("default", old)  # pyright: ignore[reportPrivateUsage]
    svc.instances_info["default"].config.custom = [old]
    svc._oauth_state_store.add(  # pyright: ignore[reportPrivateUsage]
        "state-1",
        PendingOAuthFlow(instance="default", model_id="my-remote-mcp", code_verifier="v", redirect_uri="http://x", resource="http://y"),
    )

    with pytest.raises(HTTPException) as exc_info:
        await svc.edit_model("default", old.id, AddCustomModelIn(spec=dict(old.data)))

    assert exc_info.value.status_code == 400
    assert svc.models["default"]["my-remote-mcp"].proxy_url == old.data["server_url"]


@pytest.mark.asyncio
async def test_get_duplicate_spec_catalog_model_synthesizes_from_live_model(svc: McpService) -> None:
    """DFINFRA-271: open-websearch has no stored definition (it's hardcoded in _const.models) - the
    spec must be synthesized from the live registered SrvMcpModel object."""
    spec = await svc.get_duplicate_spec("default", "open-websearch")

    assert spec["id"] == "open-websearch"
    assert spec["default_prefix"] == "open-websearch"
    assert spec["size"] == "427MB"
    assert spec["image"] == "hub.simplito.com/deepfellow/open-websearch:v2.1.9"
    assert spec["image_port"] == 3000
    assert spec["description"] == "Multi-engine customizable web search with no API key required."
    assert spec["repository_url"] == "https://github.com/aas-ee/open-websearch"
    assert "kind" not in spec


@pytest.mark.asyncio
async def test_get_duplicate_spec_catalog_model_uses_installed_envs_and_headers(svc: McpService) -> None:
    """`model.options.env_vars`/`model.headers` are the catalog's static (usually empty) definition -
    the actual values the user supplied at install time (e.g. an API key) live on the installed model
    instead, so duplicating an installed catalog model must carry those over, not the empty defaults."""
    svc.instances_info["default"].installed = InstalledInfo(
        models={
            "brave-search": ModelInstalledInfo(
                id="brave-search",
                options=InstallModelIn(spec={"prefix": "brave-search"}),
                docker_options=None,
                container_host="brave-search",
                container_port=8080,
                docker_exposed_port=8080,
                registration_id="reg-1",
                prefix="brave-search",
                base_url="http://brave-search:8080",
                headers={"X-Custom": "value"},
                envs={"BRAVE_API_KEY": "secret-key"},
            )
        },
        options=InstallServiceIn(spec={}),
    )

    spec = await svc.get_duplicate_spec("default", "brave-search")

    assert spec["envs"] == {"BRAVE_MCP_TRANSPORT": "http", "BRAVE_API_KEY": "secret-key"}
    assert spec["headers"] == {"X-Custom": "value"}


@pytest.mark.asyncio
async def test_get_duplicate_spec_preserves_selected_image_version(svc: McpService) -> None:
    """Duplicating a model pinned to a non-default image_version must keep that version, not
    silently revert to the catalog default tag."""
    svc.instances_info["default"].installed = InstalledInfo(
        models={
            "open-websearch": ModelInstalledInfo(
                id="open-websearch",
                options=InstallModelIn(spec={"prefix": "open-websearch", "image_version": "v2.1.5"}),
                docker_options=None,
                container_host="open-websearch",
                container_port=8080,
                docker_exposed_port=8080,
                registration_id="reg-1",
                prefix="open-websearch",
                base_url="http://open-websearch:8080",
                headers={},
                envs={},
            )
        },
        options=InstallServiceIn(spec={}),
    )

    spec = await svc.get_duplicate_spec("default", "open-websearch")

    assert spec["image"] == "hub.simplito.com/deepfellow/open-websearch:v2.1.5"


@pytest.mark.asyncio
async def test_get_duplicate_spec_catalog_model_merges_default_envs_with_installed(svc: McpService) -> None:
    """DFINFRA-271 regression: `ModelInstalledInfo.envs` holds only what the admin explicitly supplied
    at install time (e.g. an API key), not the catalog's own default env vars (those are merged only
    into the actual container's env_vars, never written back to ModelInstalledInfo). Using the
    installed envs alone would silently drop firecrawl's own HTTP_STREAMABLE_SERVER/HOST/PORT
    defaults from the duplicate spec."""
    svc.instances_info["default"].installed = InstalledInfo(
        models={
            "firecrawl": ModelInstalledInfo(
                id="firecrawl",
                options=InstallModelIn(spec={"prefix": "firecrawl"}),
                docker_options=None,
                container_host="firecrawl",
                container_port=3000,
                docker_exposed_port=3000,
                registration_id="reg-1",
                prefix="firecrawl",
                base_url="http://firecrawl:3000",
                headers={},
                envs={"FIRECRAWL_API_KEY": "secret-key"},
            )
        },
        options=InstallServiceIn(spec={}),
    )

    spec = await svc.get_duplicate_spec("default", "firecrawl")

    assert spec["envs"] == {
        "HTTP_STREAMABLE_SERVER": "true",
        "HOST": "0.0.0.0",
        "PORT": "3000",
        "FIRECRAWL_API_KEY": "secret-key",
    }


@pytest.mark.asyncio
async def test_get_duplicate_spec_catalog_model_with_required_envs(svc: McpService) -> None:
    """brave-search declares required_envs (a list of keys on the live model) - the synthesized spec
    turns them into the dict shape `_add_image_model`/the form expects, as empty placeholders."""
    spec = await svc.get_duplicate_spec("default", "brave-search")

    assert spec["required_envs"] == {"BRAVE_API_KEY": ""}


@pytest.mark.asyncio
async def test_get_duplicate_spec_catalog_model_spec_is_addable(svc: McpService) -> None:
    """End-to-end regression: the synthesized spec must actually pass `_add_image_model`'s validation
    (SrvMcpCustomModel), not just look plausible."""
    spec = await svc.get_duplicate_spec("default", "open-websearch")
    spec["id"] = "open-websearch-copy"
    spec["default_prefix"] = "open-websearch-copy"

    svc._add_custom_model("default", CustomModel(id="new-uuid", data=spec))  # pyright: ignore[reportPrivateUsage]

    assert "open-websearch-copy" in svc.models["default"]


@pytest.mark.asyncio
async def test_get_duplicate_spec_custom_backed_model_returns_stored_definition(svc: McpService) -> None:
    model = _make_proxy_custom()
    svc._add_custom_model("default", model)  # pyright: ignore[reportPrivateUsage]
    svc.instances_info["default"].config.custom = [model]

    spec = await svc.get_duplicate_spec("default", "my-remote-mcp")

    assert spec == model.data


@pytest.mark.asyncio
async def test_get_duplicate_spec_redacts_oauth_secrets(svc: McpService) -> None:
    """DFINFRA-271 regression: the WebUI never round-trips OAuth secrets/tokens (see `_get_custom_spec`)
    - `get_duplicate_spec` must not leak them either by returning the raw stored definition verbatim."""
    model = _make_proxy_custom(
        oauth={
            "enabled": True,
            "client_id": "client-1",
            "client_secret": "super-secret",
            "access_token": "access-tok",
            "refresh_token": "refresh-tok",
            "token_expires_at": 12345.0,
            "scope": "tools:read",
        }
    )
    svc._add_custom_model("default", model)  # pyright: ignore[reportPrivateUsage]
    svc.instances_info["default"].config.custom = [model]

    spec = await svc.get_duplicate_spec("default", "my-remote-mcp")

    assert spec["oauth"] == {"enabled": True, "client_id": "client-1", "scope": "tools:read"}


@pytest.mark.asyncio
async def test_get_duplicate_spec_strips_host_side_off_an_expanded_volume(svc: McpService) -> None:
    """No catalog MCP model declares volumes today, but the code path is the same as CustomService's -
    model.options.volumes would be an already-expanded host:container bind mount, and _add_image_model
    re-prefixes whatever it receives assuming a bare container path. Verified synthetically since it's
    otherwise currently unreachable through any real catalog model."""
    model = svc.models["default"]["open-websearch"]
    assert isinstance(model.options, DockerOptions)
    model.options.volumes = [f"{svc.get_working_dir()}/open-websearch/data:/data"]

    spec = await svc.get_duplicate_spec("default", "open-websearch")

    assert spec["volumes"] == ["/data"]


@pytest.mark.asyncio
async def test_get_duplicate_spec_unknown_model_raises_400(svc: McpService) -> None:
    with pytest.raises(HTTPException) as exc:
        await svc.get_duplicate_spec("default", "ghost")

    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_get_duplicate_spec_custom_backed_missing_definition_raises_404(svc: McpService) -> None:
    """Registry has `custom=<id>` but no matching CustomModel definition (data inconsistency)."""
    model = _make_proxy_custom()
    svc._add_custom_model("default", model)  # pyright: ignore[reportPrivateUsage]
    # Deliberately not setting `svc.instances_info["default"].config.custom` here.

    with pytest.raises(HTTPException) as exc:
        await svc.get_duplicate_spec("default", "my-remote-mcp")

    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_get_duplicate_spec_catalog_model_without_options_raises_400(svc: McpService) -> None:
    """Defensive guard: a catalog (non-custom) model with no docker options at all can't be duplicated.

    Not reachable through any real catalog entry today (they're all image-based), but `model.options`
    is legitimately `None` for other, custom-backed kinds (e.g. proxy) - this guards the catalog branch
    specifically, independent of that.
    """
    model = MagicMock(spec=SrvMcpModel)
    model.custom = None
    model.options = None
    svc.models.setdefault("default", {})["no-options"] = model

    with pytest.raises(HTTPException) as exc:
        await svc.get_duplicate_spec("default", "no-options")

    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_update_custom_model_proxy_raises_when_oauth_flow_pending(svc: McpService) -> None:
    old = _make_proxy_custom()
    svc._add_custom_model("default", old)  # pyright: ignore[reportPrivateUsage]
    svc._oauth_state_store.add(  # pyright: ignore[reportPrivateUsage]
        "state-1",
        PendingOAuthFlow(instance="default", model_id="my-remote-mcp", code_verifier="v", redirect_uri="http://x", resource="http://y"),
    )

    with pytest.raises(HTTPException) as exc_info:
        await svc._update_custom_model("default", old, dict(old.data))  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400


def test_remove_custom_model_proxy_raises_when_oauth_flow_pending(svc: McpService) -> None:
    custom = _make_proxy_custom()
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]
    svc._oauth_state_store.add(  # pyright: ignore[reportPrivateUsage]
        "state-1",
        PendingOAuthFlow(instance="default", model_id="my-remote-mcp", code_verifier="v", redirect_uri="http://x", resource="http://y"),
    )

    with pytest.raises(HTTPException) as exc_info:
        svc._remove_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_start_oauth_flow_raises_when_oauth_disabled(svc: McpService) -> None:
    custom = _make_proxy_custom(oauth={"enabled": False})
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    with pytest.raises(HTTPException) as exc_info:
        await svc.start_oauth_flow("default", "my-remote-mcp")

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_start_oauth_flow_raises_when_no_registration_endpoint(svc: McpService) -> None:
    svc.config.infra_url = "http://infra.example"  # pyright: ignore[reportAttributeAccessIssue]
    custom = _make_proxy_custom(oauth={"enabled": True})
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]
    metadata = AuthorizationServerMetadata(
        issuer="http://as.example",
        authorization_endpoint="http://as.example/authorize",
        token_endpoint="http://as.example/token",
        registration_endpoint=None,
    )

    with (
        patch("server.services.mcp_service.discover_authorization_server_for_resource", new=AsyncMock(return_value=metadata)),
        patch.object(svc, "_save", new=AsyncMock()),
        pytest.raises(HTTPException) as exc_info,
    ):
        await svc.start_oauth_flow("default", "my-remote-mcp")

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_start_oauth_flow_dynamic_client_registration_persists_client_id(svc: McpService) -> None:
    svc.config.infra_url = "http://infra.example"  # pyright: ignore[reportAttributeAccessIssue]
    custom = _make_proxy_custom(oauth={"enabled": True})
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]
    metadata = AuthorizationServerMetadata(
        issuer="http://as.example",
        authorization_endpoint="http://as.example/authorize",
        token_endpoint="http://as.example/token",
        registration_endpoint="http://as.example/register",
    )
    dcr = DcrResponse(client_id="dcr-client-1", client_secret="dcr-secret-1")

    with (
        patch("server.services.mcp_service.discover_authorization_server_for_resource", new=AsyncMock(return_value=metadata)),
        patch("server.services.mcp_service.register_dynamic_client", new=AsyncMock(return_value=dcr)),
        patch.object(svc, "_save", new=AsyncMock()),
    ):
        authorize_url = await svc.start_oauth_flow("default", "my-remote-mcp")

    assert "client_id=dcr-client-1" in authorize_url
    updated = svc.models["default"]["my-remote-mcp"]
    assert updated.oauth is not None
    assert updated.oauth.client_id == "dcr-client-1"
    assert updated.oauth.client_secret == "dcr-secret-1"


@pytest.mark.asyncio
async def test_complete_oauth_callback_unknown_state_returns_invalid_message(svc: McpService) -> None:
    result = await svc.complete_oauth_callback("unknown-state", "code-1", None)

    assert "invalid or has expired" in result.lower()


@pytest.mark.asyncio
async def test_complete_oauth_callback_error_param_persists_last_error(svc: McpService) -> None:
    custom = _make_proxy_custom(
        oauth={
            "enabled": True,
            "client_id": "client-1",
            "token_endpoint": "http://as.example/token",
            "authorization_endpoint": "http://as.example/authorize",
        }
    )
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]
    svc._oauth_state_store.add(  # pyright: ignore[reportPrivateUsage]
        "state-1",
        PendingOAuthFlow(
            instance="default",
            model_id="my-remote-mcp",
            code_verifier="v",
            redirect_uri="http://infra.example/mcp-oauth/callback",
            resource="http://mcp.example.com/mcp",
        ),
    )

    with patch.object(svc, "_save", new=AsyncMock()):
        result = await svc.complete_oauth_callback("state-1", None, "access_denied")

    assert "failed" in result.lower()
    updated = svc.models["default"]["my-remote-mcp"]
    assert updated.oauth is not None
    assert updated.oauth.last_error == "access_denied"


@pytest.mark.asyncio
async def test_complete_oauth_callback_success_persists_tokens(svc: McpService) -> None:
    custom = _make_proxy_custom(
        oauth={
            "enabled": True,
            "client_id": "client-1",
            "token_endpoint": "http://as.example/token",
            "authorization_endpoint": "http://as.example/authorize",
        }
    )
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]
    svc._oauth_state_store.add(  # pyright: ignore[reportPrivateUsage]
        "state-1",
        PendingOAuthFlow(
            instance="default",
            model_id="my-remote-mcp",
            code_verifier="v",
            redirect_uri="http://infra.example/mcp-oauth/callback",
            resource="http://mcp.example.com/mcp",
        ),
    )
    token = TokenResponse(access_token="new-access", refresh_token="new-refresh", expires_in=3600)

    with (
        patch("server.services.mcp_service.exchange_code_for_token", new=AsyncMock(return_value=token)),
        patch.object(svc, "_save", new=AsyncMock()),
        patch.object(svc, "_start_oauth_refresh_background"),  # pyright: ignore[reportPrivateUsage]
    ):
        result = await svc.complete_oauth_callback("state-1", "code-1", None)

    assert "complete" in result.lower()
    updated = svc.models["default"]["my-remote-mcp"]
    assert updated.oauth is not None
    assert updated.oauth.access_token == "new-access"
    assert updated.oauth.refresh_token == "new-refresh"
    assert updated.oauth.token_expires_at is not None


@pytest.mark.asyncio
async def test_complete_oauth_callback_duplicate_with_valid_token_returns_already_authorized(svc: McpService) -> None:
    custom = _make_proxy_custom(
        oauth={
            "enabled": True,
            "client_id": "client-1",
            "token_endpoint": "http://as.example/token",
            "authorization_endpoint": "http://as.example/authorize",
            "access_token": "already-valid",
            "token_expires_at": None,
        }
    )
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]
    svc._oauth_state_store.add(  # pyright: ignore[reportPrivateUsage]
        "state-1",
        PendingOAuthFlow(
            instance="default",
            model_id="my-remote-mcp",
            code_verifier="v",
            redirect_uri="http://infra.example/mcp-oauth/callback",
            resource="http://mcp.example.com/mcp",
        ),
    )

    with (
        patch("server.services.mcp_service.exchange_code_for_token", new=AsyncMock(side_effect=McpOAuthError("invalid_grant"))),
        patch.object(svc, "_save", new=AsyncMock()),
    ):
        result = await svc.complete_oauth_callback("state-1", "code-1", None)

    assert "already authorized" in result.lower()


def test_get_oauth_status_disabled_when_oauth_none(svc: McpService) -> None:
    custom = _make_proxy_custom(oauth=None)
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    status = svc.get_oauth_status("default", "my-remote-mcp")

    assert status.status == "disabled"
    assert status.enabled is False


def test_get_oauth_status_authorized_when_valid_token(svc: McpService) -> None:
    custom = _make_proxy_custom(oauth={"enabled": True, "access_token": "tok", "token_expires_at": None})
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    status = svc.get_oauth_status("default", "my-remote-mcp")

    assert status.status == "authorized"


def test_get_oauth_status_expired_when_token_past_expiry(svc: McpService) -> None:
    custom = _make_proxy_custom(oauth={"enabled": True, "access_token": "tok", "token_expires_at": 1.0})
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    status = svc.get_oauth_status("default", "my-remote-mcp")

    assert status.status == "expired"


def test_get_oauth_status_pending_when_flow_live(svc: McpService) -> None:
    custom = _make_proxy_custom(oauth={"enabled": True})
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]
    svc._oauth_state_store.add(  # pyright: ignore[reportPrivateUsage]
        "state-1",
        PendingOAuthFlow(instance="default", model_id="my-remote-mcp", code_verifier="v", redirect_uri="http://x", resource="http://y"),
    )

    status = svc.get_oauth_status("default", "my-remote-mcp")

    assert status.status == "pending"


def test_get_oauth_status_never_includes_secret_or_token_fields() -> None:
    field_names = set(McpOAuthStatusOut.model_fields.keys())

    assert "client_secret" not in field_names
    assert "access_token" not in field_names
    assert "refresh_token" not in field_names


@pytest.mark.asyncio
async def test_auto_detect_oauth_skips_when_already_enabled(svc: McpService) -> None:
    custom = _make_proxy_custom(oauth={"enabled": True, "client_id": "client-1"})
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    with patch("server.services.mcp_service.discover_authorization_server_for_resource", new=AsyncMock()) as mock_discover:
        await svc._auto_detect_oauth("default", "my-remote-mcp")  # pyright: ignore[reportPrivateUsage]

    mock_discover.assert_not_awaited()


@pytest.mark.asyncio
async def test_auto_detect_oauth_discovery_success_enables_and_persists(svc: McpService) -> None:
    custom = _make_proxy_custom()
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]
    metadata = AuthorizationServerMetadata(
        issuer="http://as.example",
        authorization_endpoint="http://as.example/authorize",
        token_endpoint="http://as.example/token",
        registration_endpoint="http://as.example/register",
    )

    with (
        patch("server.services.mcp_service.discover_authorization_server_for_resource", new=AsyncMock(return_value=metadata)),
        patch.object(svc, "_save", new=AsyncMock()),
        patch.object(svc, "_reregister_installed_proxy", new=AsyncMock()) as mock_reregister,  # pyright: ignore[reportPrivateUsage]
    ):
        await svc._auto_detect_oauth("default", "my-remote-mcp")  # pyright: ignore[reportPrivateUsage]

    updated = svc.models["default"]["my-remote-mcp"]
    assert updated.oauth is not None
    assert updated.oauth.enabled is True
    assert updated.oauth.authorization_endpoint == "http://as.example/authorize"
    assert updated.oauth.token_endpoint == "http://as.example/token"
    assert updated.oauth.registration_endpoint == "http://as.example/register"
    assert updated.oauth.resource == "http://mcp.example.com/mcp"
    assert updated.oauth.last_error is None
    mock_reregister.assert_awaited_once_with("default", "my-remote-mcp")


@pytest.mark.asyncio
async def test_auto_detect_oauth_discovery_failure_still_flags_required(svc: McpService) -> None:
    custom = _make_proxy_custom()
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    with (
        patch("server.services.mcp_service.discover_authorization_server_for_resource", new=AsyncMock(return_value=None)),
        patch.object(svc, "_save", new=AsyncMock()),
        patch.object(svc, "_reregister_installed_proxy", new=AsyncMock()),  # pyright: ignore[reportPrivateUsage]
    ):
        await svc._auto_detect_oauth("default", "my-remote-mcp")  # pyright: ignore[reportPrivateUsage]

    updated = svc.models["default"]["my-remote-mcp"]
    assert updated.oauth is not None
    assert updated.oauth.enabled is True
    assert updated.oauth.authorization_endpoint is None
    assert updated.oauth.last_error is not None


@pytest.mark.asyncio
async def test_auto_detect_oauth_not_a_proxy_model_is_noop(svc: McpService) -> None:
    with patch("server.services.mcp_service.discover_authorization_server_for_resource", new=AsyncMock()) as mock_discover:
        await svc._auto_detect_oauth("default", "open-websearch")  # pyright: ignore[reportPrivateUsage]

    mock_discover.assert_not_awaited()


@pytest.mark.asyncio
async def test_auto_detect_oauth_concurrent_calls_only_discover_once(svc: McpService) -> None:
    """Two healthchecks racing on the same 401 must not double-discover/re-register (the TOCTOU fix)."""
    custom = _make_proxy_custom()
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]
    release = asyncio.Event()

    async def slow_discover(_resource: str) -> None:
        await release.wait()

    with (
        patch(
            "server.services.mcp_service.discover_authorization_server_for_resource", new=AsyncMock(side_effect=slow_discover)
        ) as mock_discover,
        patch.object(svc, "_save", new=AsyncMock()),
        patch.object(svc, "_reregister_installed_proxy", new=AsyncMock()) as mock_reregister,  # pyright: ignore[reportPrivateUsage]
    ):
        first = asyncio.create_task(svc._auto_detect_oauth("default", "my-remote-mcp"))  # pyright: ignore[reportPrivateUsage]
        await asyncio.sleep(0)  # let `first` pass the pre-lock check and start waiting inside the lock
        second = asyncio.create_task(svc._auto_detect_oauth("default", "my-remote-mcp"))  # pyright: ignore[reportPrivateUsage]
        await asyncio.sleep(0)  # let `second` pass the pre-lock check too, then block on the lock
        release.set()
        await asyncio.gather(first, second)

    mock_discover.assert_awaited_once()
    mock_reregister.assert_awaited_once()


@pytest.mark.asyncio
async def test_complete_oauth_callback_success_refreshes_tools_when_installed(svc: McpService) -> None:
    custom = _make_proxy_custom(
        oauth={
            "enabled": True,
            "client_id": "client-1",
            "token_endpoint": "http://as.example/token",
            "authorization_endpoint": "http://as.example/authorize",
        }
    )
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]
    installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    installed.models["my-remote-mcp"] = _make_installed_model_info()
    svc.instances_info["default"].installed = installed
    svc._oauth_state_store.add(  # pyright: ignore[reportPrivateUsage]
        "state-1",
        PendingOAuthFlow(
            instance="default",
            model_id="my-remote-mcp",
            code_verifier="v",
            redirect_uri="http://infra.example/mcp-oauth/callback",
            resource="http://mcp.example.com/mcp",
        ),
    )
    token = TokenResponse(access_token="new-access", expires_in=3600)

    with (
        patch("server.services.mcp_service.exchange_code_for_token", new=AsyncMock(return_value=token)),
        patch.object(svc, "_save", new=AsyncMock()),
        patch.object(svc, "_reregister_installed_proxy", new=AsyncMock()),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_fetch_tools_background", new=AsyncMock()) as mock_fetch_bg,  # pyright: ignore[reportPrivateUsage]
    ):
        await svc.complete_oauth_callback("state-1", "code-1", None)
        await asyncio.sleep(0)  # let the fire-and-forget task run

    mock_fetch_bg.assert_awaited_once_with("default", "my-remote-mcp")


@mock.patch("server.services.mcp_service.asyncio.sleep", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_oauth_refresh_background_gives_up_after_max_consecutive_failures(mock_sleep: AsyncMock, svc: McpService) -> None:
    custom = _make_proxy_custom(
        oauth={"enabled": True, "client_id": "client-1", "refresh_token": "rtok", "token_expires_at": time.time() + 3600}
    )
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    with patch.object(svc, "_refresh_oauth_token", new=AsyncMock(return_value=None)) as mock_refresh:  # pyright: ignore[reportPrivateUsage]
        await svc._oauth_refresh_background("default", "my-remote-mcp")  # pyright: ignore[reportPrivateUsage]

    assert mock_refresh.await_count == McpService._OAUTH_REFRESH_MAX_CONSECUTIVE_FAILURES  # pyright: ignore[reportPrivateUsage]
    sleep_values = [call.args[0] for call in mock_sleep.call_args_list]
    # sleep_values[0] is the initial near-expiry wait; sleep_values[1:] are the failure backoffs,
    # which must grow between consecutive failures rather than retrying at a fixed short interval forever.
    backoffs = sleep_values[1:]
    assert backoffs == sorted(backoffs)
    assert backoffs[0] < backoffs[-1]
    assert all(v <= McpService._OAUTH_REFRESH_MAX_BACKOFF_SECONDS for v in backoffs)  # pyright: ignore[reportPrivateUsage]


@mock.patch("server.services.mcp_service.asyncio.sleep", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_oauth_refresh_background_resets_backoff_after_success(mock_sleep: AsyncMock, svc: McpService) -> None:
    custom = _make_proxy_custom(
        oauth={"enabled": True, "client_id": "client-1", "refresh_token": "rtok", "token_expires_at": time.time() + 3600}
    )
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]
    refreshed = McpOAuthConfig(enabled=True, refresh_token="rtok", access_token="new-tok", token_expires_at=time.time() + 3600)

    # fail, fail, succeed, fail-and-stop: the backoff before the 3rd attempt (after two failures)
    # must be bigger than the backoff before the 2nd (after one failure); the wait scheduled right
    # after the success (4th attempt) must jump back to the near-expiry wait, not continue growing.
    call_results: list[McpOAuthConfig | None] = [None, None, refreshed, None]

    async def fake_refresh(_instance: str, _model_id: str) -> McpOAuthConfig | None:
        result = call_results.pop(0)
        if not call_results:
            # Stop the infinite loop after the final iteration by disabling refresh.
            model = svc.models["default"]["my-remote-mcp"]
            assert model.oauth is not None
            model.oauth.refresh_token = None
        return result

    with patch.object(svc, "_refresh_oauth_token", new=AsyncMock(side_effect=fake_refresh)):  # pyright: ignore[reportPrivateUsage]
        await svc._oauth_refresh_background("default", "my-remote-mcp")  # pyright: ignore[reportPrivateUsage]

    sleep_values = [call.args[0] for call in mock_sleep.call_args_list]
    assert len(sleep_values) == 4
    initial_wait, first_backoff, second_backoff, post_success_wait = sleep_values
    assert second_backoff > first_backoff
    # Reset wait is driven by expiry again, not a continuation of the backoff sequence.
    assert post_success_wait > second_backoff * 2
    assert post_success_wait == pytest.approx(initial_wait, rel=0.05)


def test_remove_custom_model_proxy_drops_oauth_refresh_lock(svc: McpService) -> None:
    custom = _make_proxy_custom()
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]
    svc._oauth_refresh_locks[("default", "my-remote-mcp")] = asyncio.Lock()  # pyright: ignore[reportPrivateUsage]

    svc._remove_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    assert ("default", "my-remote-mcp") not in svc._oauth_refresh_locks  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_persist_proxy_oauth_model_missing_is_noop(svc: McpService) -> None:
    await svc._persist_proxy_oauth("default", "missing-model", McpOAuthConfig())  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_persist_proxy_oauth_writes_to_matching_custom_model(svc: McpService) -> None:
    other = _make_proxy_custom(custom_id="uuid-proxy-other", model_id="other-remote-mcp")
    custom = _make_proxy_custom()
    svc._add_custom_model("default", other)  # pyright: ignore[reportPrivateUsage]
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]
    svc.get_instance_info("default").config.custom = [other, custom]
    oauth = McpOAuthConfig(enabled=True, access_token="tok-123")

    with patch.object(svc, "_save", new=AsyncMock()) as mock_save:
        await svc._persist_proxy_oauth("default", "my-remote-mcp", oauth)  # pyright: ignore[reportPrivateUsage]

    mock_save.assert_awaited_once()
    assert custom.data["oauth"]["access_token"] == "tok-123"
    assert "oauth" not in other.data
    assert svc.models["default"]["my-remote-mcp"].oauth is oauth


def test_get_oauth_lock_creates_and_reuses_lock(svc: McpService) -> None:
    lock1 = svc._get_oauth_lock("default", "my-remote-mcp")  # pyright: ignore[reportPrivateUsage]
    lock2 = svc._get_oauth_lock("default", "my-remote-mcp")  # pyright: ignore[reportPrivateUsage]

    assert lock1 is lock2
    assert isinstance(lock1, asyncio.Lock)


@pytest.mark.asyncio
async def test_refresh_oauth_token_model_missing_returns_none(svc: McpService) -> None:
    result = await svc._refresh_oauth_token("default", "missing-model")  # pyright: ignore[reportPrivateUsage]

    assert result is None


@pytest.mark.asyncio
async def test_refresh_oauth_token_no_refresh_token_returns_oauth_unchanged(svc: McpService) -> None:
    custom = _make_proxy_custom(oauth={"enabled": True, "client_id": "client-1"})
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    result = await svc._refresh_oauth_token("default", "my-remote-mcp")  # pyright: ignore[reportPrivateUsage]

    assert result is not None
    assert result.access_token is None


@pytest.mark.asyncio
async def test_refresh_oauth_token_already_valid_skips_refresh_call(svc: McpService) -> None:
    custom = _make_proxy_custom(
        oauth={
            "enabled": True,
            "client_id": "client-1",
            "refresh_token": "rtok",
            "token_endpoint": "http://as.example/token",
            "access_token": "still-valid",
            "token_expires_at": None,
        }
    )
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    with patch("server.services.mcp_service.refresh_access_token", new=AsyncMock()) as mock_refresh:
        result = await svc._refresh_oauth_token("default", "my-remote-mcp")  # pyright: ignore[reportPrivateUsage]

    mock_refresh.assert_not_awaited()
    assert result is not None
    assert result.access_token == "still-valid"


@pytest.mark.asyncio
async def test_refresh_oauth_token_no_client_id_returns_oauth_unchanged(svc: McpService) -> None:
    custom = _make_proxy_custom(oauth={"enabled": True, "refresh_token": "rtok", "token_endpoint": "http://as.example/token"})
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    result = await svc._refresh_oauth_token("default", "my-remote-mcp")  # pyright: ignore[reportPrivateUsage]

    assert result is not None
    assert result.access_token is None


@pytest.mark.asyncio
async def test_refresh_oauth_token_success_persists_new_tokens(svc: McpService) -> None:
    custom = _make_proxy_custom(
        oauth={"enabled": True, "client_id": "client-1", "refresh_token": "rtok", "token_endpoint": "http://as.example/token"}
    )
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]
    token = TokenResponse(access_token="new-access", refresh_token="new-refresh", expires_in=3600)

    with (
        patch("server.services.mcp_service.refresh_access_token", new=AsyncMock(return_value=token)),
        patch.object(svc, "_save", new=AsyncMock()),
    ):
        result = await svc._refresh_oauth_token("default", "my-remote-mcp")  # pyright: ignore[reportPrivateUsage]

    assert result is not None
    assert result.access_token == "new-access"
    assert result.refresh_token == "new-refresh"
    assert result.token_expires_at is not None
    assert result.last_error is None


@pytest.mark.asyncio
async def test_refresh_oauth_token_success_records_otel_metric_when_enabled(svc: McpService) -> None:
    custom = _make_proxy_custom(
        oauth={"enabled": True, "client_id": "client-1", "refresh_token": "rtok", "token_endpoint": "http://as.example/token"}
    )
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]
    token = TokenResponse(access_token="new-access", refresh_token="new-refresh", expires_in=3600)
    fake_instruments = MagicMock()

    with (
        patch("server.services.mcp_service.refresh_access_token", new=AsyncMock(return_value=token)),
        patch.object(svc, "_save", new=AsyncMock()),
        patch("server.services.mcp_service.tracer.mcp_instruments", return_value=fake_instruments),
    ):
        result = await svc._refresh_oauth_token("default", "my-remote-mcp")  # pyright: ignore[reportPrivateUsage]

    assert result is not None
    assert fake_instruments.oauth_refresh_count.add.call_count == 1
    attrs = fake_instruments.oauth_refresh_count.add.call_args[0][1]
    assert attrs["mcp.outcome"] == "success"


@pytest.mark.asyncio
async def test_refresh_oauth_token_error_records_otel_metric_when_enabled(svc: McpService) -> None:
    custom = _make_proxy_custom(
        oauth={"enabled": True, "client_id": "client-1", "refresh_token": "rtok", "token_endpoint": "http://as.example/token"}
    )
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]
    fake_instruments = MagicMock()

    with (
        patch("server.services.mcp_service.refresh_access_token", new=AsyncMock(side_effect=McpOAuthError("invalid_grant"))),
        patch.object(svc, "_save", new=AsyncMock()),
        patch("server.services.mcp_service.tracer.mcp_instruments", return_value=fake_instruments),
    ):
        result = await svc._refresh_oauth_token("default", "my-remote-mcp")  # pyright: ignore[reportPrivateUsage]

    assert result is None
    assert fake_instruments.oauth_refresh_count.add.call_count == 1
    attrs = fake_instruments.oauth_refresh_count.add.call_args[0][1]
    assert attrs["mcp.outcome"] == "failure"


@pytest.mark.asyncio
async def test_refresh_oauth_token_success_sets_outcome_span_attribute(svc: McpService) -> None:
    custom = _make_proxy_custom(
        oauth={"enabled": True, "client_id": "client-1", "refresh_token": "rtok", "token_endpoint": "http://as.example/token"}
    )
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]
    token = TokenResponse(access_token="new-access", refresh_token="new-refresh", expires_in=3600)
    fake_span = MagicMock()

    with (
        patch("server.services.mcp_service.refresh_access_token", new=AsyncMock(return_value=token)),
        patch.object(svc, "_save", new=AsyncMock()),
        mock.patch("server.services.mcp_service.tracer.span") as mock_span,
    ):
        mock_span.return_value.__enter__.return_value = fake_span
        result = await svc._refresh_oauth_token("default", "my-remote-mcp")  # pyright: ignore[reportPrivateUsage]

    assert result is not None
    assert fake_span.set_attribute.call_args == call("mcp.outcome", "success")


@pytest.mark.asyncio
async def test_refresh_oauth_token_error_sets_outcome_span_attribute(svc: McpService) -> None:
    custom = _make_proxy_custom(
        oauth={"enabled": True, "client_id": "client-1", "refresh_token": "rtok", "token_endpoint": "http://as.example/token"}
    )
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]
    fake_span = MagicMock()

    with (
        patch("server.services.mcp_service.refresh_access_token", new=AsyncMock(side_effect=McpOAuthError("invalid_grant"))),
        patch.object(svc, "_save", new=AsyncMock()),
        mock.patch("server.services.mcp_service.tracer.span") as mock_span,
    ):
        mock_span.return_value.__enter__.return_value = fake_span
        result = await svc._refresh_oauth_token("default", "my-remote-mcp")  # pyright: ignore[reportPrivateUsage]

    assert result is None
    assert fake_span.set_attribute.call_args == call("mcp.outcome", "failure")


@pytest.mark.asyncio
async def test_refresh_oauth_token_success_without_new_refresh_token_keeps_existing(svc: McpService) -> None:
    custom = _make_proxy_custom(
        oauth={
            "enabled": True,
            "client_id": "client-1",
            "refresh_token": "original-rtok",
            "token_endpoint": "http://as.example/token",
        }
    )
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]
    token = TokenResponse(access_token="new-access", refresh_token=None, expires_in=3600)

    with (
        patch("server.services.mcp_service.refresh_access_token", new=AsyncMock(return_value=token)),
        patch.object(svc, "_save", new=AsyncMock()),
    ):
        result = await svc._refresh_oauth_token("default", "my-remote-mcp")  # pyright: ignore[reportPrivateUsage]

    assert result is not None
    assert result.access_token == "new-access"
    assert result.refresh_token == "original-rtok"


@pytest.mark.asyncio
async def test_refresh_oauth_token_error_persists_last_error_and_returns_none(svc: McpService) -> None:
    custom = _make_proxy_custom(
        oauth={"enabled": True, "client_id": "client-1", "refresh_token": "rtok", "token_endpoint": "http://as.example/token"}
    )
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    with (
        patch("server.services.mcp_service.refresh_access_token", new=AsyncMock(side_effect=McpOAuthError("invalid_grant"))),
        patch.object(svc, "_save", new=AsyncMock()),
    ):
        result = await svc._refresh_oauth_token("default", "my-remote-mcp")  # pyright: ignore[reportPrivateUsage]

    assert result is None
    updated = svc.models["default"]["my-remote-mcp"]
    assert updated.oauth is not None
    assert updated.oauth.last_error == "invalid_grant"


@pytest.mark.asyncio
async def test_oauth_header_provider_no_oauth_returns_empty_headers(svc: McpService) -> None:
    svc._add_custom_model("default", _make_proxy_custom(oauth=None))  # pyright: ignore[reportPrivateUsage]
    provider = svc._make_oauth_header_provider("default", "my-remote-mcp")  # pyright: ignore[reportPrivateUsage]

    headers = await provider()

    assert headers == {}


@pytest.mark.asyncio
async def test_oauth_header_provider_valid_token_returns_bearer_header(svc: McpService) -> None:
    custom = _make_proxy_custom(oauth={"enabled": True, "access_token": "tok-123", "token_expires_at": None})
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]
    provider = svc._make_oauth_header_provider("default", "my-remote-mcp")  # pyright: ignore[reportPrivateUsage]

    headers = await provider()

    assert headers == {"Authorization": "Bearer tok-123"}


@pytest.mark.asyncio
async def test_oauth_header_provider_expired_token_refreshes_before_returning(svc: McpService) -> None:
    custom = _make_proxy_custom(oauth={"enabled": True, "access_token": "old-tok", "refresh_token": "rtok", "token_expires_at": 1.0})
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]
    refreshed = McpOAuthConfig(enabled=True, access_token="fresh-tok")

    with patch.object(svc, "_refresh_oauth_token", new=AsyncMock(return_value=refreshed)) as mock_refresh:  # pyright: ignore[reportPrivateUsage]
        provider = svc._make_oauth_header_provider("default", "my-remote-mcp")  # pyright: ignore[reportPrivateUsage]
        headers = await provider()

    mock_refresh.assert_awaited_once_with("default", "my-remote-mcp")
    assert headers == {"Authorization": "Bearer fresh-tok"}


@pytest.mark.asyncio
async def test_oauth_refresh_callback_success_returns_bearer_header(svc: McpService) -> None:
    refreshed = McpOAuthConfig(enabled=True, access_token="fresh-tok")
    svc._add_custom_model("default", _make_proxy_custom())  # pyright: ignore[reportPrivateUsage]

    with patch.object(svc, "_refresh_oauth_token", new=AsyncMock(return_value=refreshed)):  # pyright: ignore[reportPrivateUsage]
        on_reauth = svc._make_oauth_refresh_callback("default", "my-remote-mcp")  # pyright: ignore[reportPrivateUsage]
        headers = await on_reauth()

    assert headers == {"Authorization": "Bearer fresh-tok"}


@pytest.mark.asyncio
async def test_oauth_refresh_callback_failure_returns_none(svc: McpService) -> None:
    svc._add_custom_model("default", _make_proxy_custom())  # pyright: ignore[reportPrivateUsage]

    with patch.object(svc, "_refresh_oauth_token", new=AsyncMock(return_value=None)):  # pyright: ignore[reportPrivateUsage]
        on_reauth = svc._make_oauth_refresh_callback("default", "my-remote-mcp")  # pyright: ignore[reportPrivateUsage]
        headers = await on_reauth()

    assert headers is None


@mock.patch("server.services.mcp_service.asyncio.sleep", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_oauth_refresh_background_stops_if_refresh_token_cleared_during_sleep(mock_sleep: AsyncMock, svc: McpService) -> None:
    custom = _make_proxy_custom(
        oauth={"enabled": True, "client_id": "client-1", "refresh_token": "rtok", "token_expires_at": time.time() + 3600}
    )
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    async def clear_refresh_token(_seconds: float) -> None:
        model = svc.models["default"]["my-remote-mcp"]
        assert model.oauth is not None
        model.oauth.refresh_token = None

    mock_sleep.side_effect = clear_refresh_token

    with patch.object(svc, "_refresh_oauth_token", new=AsyncMock()) as mock_refresh:  # pyright: ignore[reportPrivateUsage]
        await svc._oauth_refresh_background("default", "my-remote-mcp")  # pyright: ignore[reportPrivateUsage]

    mock_refresh.assert_not_awaited()


@mock.patch("server.services.mcp_service.asyncio.sleep", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_oauth_refresh_background_logs_and_continues_on_exception(mock_sleep: AsyncMock, svc: McpService) -> None:
    custom = _make_proxy_custom(
        oauth={"enabled": True, "client_id": "client-1", "refresh_token": "rtok", "token_expires_at": time.time() + 3600}
    )
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    with patch.object(svc, "_refresh_oauth_token", new=AsyncMock(side_effect=RuntimeError("boom"))) as mock_refresh:  # pyright: ignore[reportPrivateUsage]
        await svc._oauth_refresh_background("default", "my-remote-mcp")  # pyright: ignore[reportPrivateUsage]

    assert mock_refresh.await_count == McpService._OAUTH_REFRESH_MAX_CONSECUTIVE_FAILURES  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_start_oauth_refresh_background_creates_tracked_task(svc: McpService) -> None:
    svc._start_oauth_refresh_background("default", "no-such-model")  # pyright: ignore[reportPrivateUsage]

    assert len(svc._background_tasks) == 1  # pyright: ignore[reportPrivateUsage]

    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert len(svc._background_tasks) == 0  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_start_oauth_refresh_background_skips_spawn_when_task_already_running(svc: McpService) -> None:
    svc._start_oauth_refresh_background("default", "no-such-model")  # pyright: ignore[reportPrivateUsage]
    first_task = svc._oauth_refresh_tasks[("default", "no-such-model")]  # pyright: ignore[reportPrivateUsage]

    svc._start_oauth_refresh_background("default", "no-such-model")  # pyright: ignore[reportPrivateUsage]

    assert svc._oauth_refresh_tasks[("default", "no-such-model")] is first_task  # pyright: ignore[reportPrivateUsage]
    assert len(svc._background_tasks) == 1  # pyright: ignore[reportPrivateUsage]

    first_task.cancel()


@pytest.mark.asyncio
async def test_remove_custom_model_proxy_cancels_running_refresh_task(svc: McpService) -> None:
    custom = _make_proxy_custom()
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]
    svc._start_oauth_refresh_background("default", "my-remote-mcp")  # pyright: ignore[reportPrivateUsage]
    task = svc._oauth_refresh_tasks[("default", "my-remote-mcp")]  # pyright: ignore[reportPrivateUsage]

    svc._remove_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    assert ("default", "my-remote-mcp") not in svc._oauth_refresh_tasks  # pyright: ignore[reportPrivateUsage]

    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert task.cancelled()


def test_get_oauth_model_raises_when_not_proxy(svc: McpService) -> None:
    with pytest.raises(HTTPException) as exc_info:
        svc._get_oauth_model("default", "open-websearch")  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_discover_oauth_endpoints_metadata_none_skips_persist(svc: McpService) -> None:
    custom = _make_proxy_custom(oauth={"enabled": True})
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]
    model = svc.models["default"]["my-remote-mcp"]

    with (
        patch("server.services.mcp_service.discover_authorization_server_for_resource", new=AsyncMock(return_value=None)),
        patch.object(svc, "_persist_proxy_oauth", new=AsyncMock()) as mock_persist,  # pyright: ignore[reportPrivateUsage]
    ):
        await svc._discover_oauth_endpoints("default", "my-remote-mcp", model)  # pyright: ignore[reportPrivateUsage]

    mock_persist.assert_not_awaited()
    assert model.oauth is not None
    assert model.oauth.authorization_endpoint is None


@pytest.mark.asyncio
async def test_reregister_installed_proxy_model_not_in_installed_info_is_noop(svc: McpService) -> None:
    svc._add_custom_model("default", _make_proxy_custom())  # pyright: ignore[reportPrivateUsage]
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))

    await svc._reregister_installed_proxy("default", "my-remote-mcp")  # pyright: ignore[reportPrivateUsage]

    svc.endpoint_registry.unregister_mcp_endpoint.assert_not_called()  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.asyncio
async def test_start_oauth_flow_raises_when_discovery_finds_no_endpoints(svc: McpService) -> None:
    svc.config.infra_url = "http://infra.example"  # pyright: ignore[reportAttributeAccessIssue]
    custom = _make_proxy_custom(oauth={"enabled": True})
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    with (
        patch("server.services.mcp_service.discover_authorization_server_for_resource", new=AsyncMock(return_value=None)),
        patch.object(svc, "_save", new=AsyncMock()),
        pytest.raises(HTTPException) as exc_info,
    ):
        await svc.start_oauth_flow("default", "my-remote-mcp")

    assert exc_info.value.status_code == 400
    assert "could not discover" in str(exc_info.value.detail).lower()


@pytest.mark.asyncio
async def test_start_oauth_flow_raises_when_dynamic_registration_fails(svc: McpService) -> None:
    svc.config.infra_url = "http://infra.example"  # pyright: ignore[reportAttributeAccessIssue]
    custom = _make_proxy_custom(oauth={"enabled": True})
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]
    metadata = AuthorizationServerMetadata(
        issuer="http://as.example",
        authorization_endpoint="http://as.example/authorize",
        token_endpoint="http://as.example/token",
        registration_endpoint="http://as.example/register",
    )

    with (
        patch("server.services.mcp_service.discover_authorization_server_for_resource", new=AsyncMock(return_value=metadata)),
        patch("server.services.mcp_service.register_dynamic_client", new=AsyncMock(return_value=None)),
        patch.object(svc, "_save", new=AsyncMock()),
        pytest.raises(HTTPException) as exc_info,
    ):
        await svc.start_oauth_flow("default", "my-remote-mcp")

    assert exc_info.value.status_code == 400
    assert "registration failed" in str(exc_info.value.detail).lower()


@pytest.mark.asyncio
async def test_start_oauth_flow_skips_discovery_and_dcr_when_already_configured(svc: McpService) -> None:
    svc.config.infra_url = "http://infra.example"  # pyright: ignore[reportAttributeAccessIssue]
    custom = _make_proxy_custom(
        oauth={
            "enabled": True,
            "client_id": "existing-client",
            "authorization_endpoint": "http://as.example/authorize",
            "token_endpoint": "http://as.example/token",
        }
    )
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    with (
        patch("server.services.mcp_service.discover_authorization_server_for_resource", new=AsyncMock()) as mock_discover,
        patch("server.services.mcp_service.register_dynamic_client", new=AsyncMock()) as mock_register,
    ):
        authorize_url = await svc.start_oauth_flow("default", "my-remote-mcp")

    mock_discover.assert_not_awaited()
    mock_register.assert_not_awaited()
    assert authorize_url.startswith("http://as.example/authorize?")
    assert "client_id=existing-client" in authorize_url


@pytest.mark.asyncio
async def test_start_oauth_flow_raises_when_infra_url_unconfigured(svc: McpService) -> None:
    svc.config.infra_url = ""  # pyright: ignore[reportAttributeAccessIssue]
    custom = _make_proxy_custom(
        oauth={
            "enabled": True,
            "client_id": "existing-client",
            "authorization_endpoint": "http://as.example/authorize",
            "token_endpoint": "http://as.example/token",
        }
    )
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    with pytest.raises(HTTPException) as exc_info:
        await svc.start_oauth_flow("default", "my-remote-mcp")

    assert exc_info.value.status_code == 400
    assert "infra_url" in str(exc_info.value.detail).lower()


@pytest.mark.asyncio
async def test_start_oauth_flow_redirect_uri_preserves_infra_url_path_prefix(svc: McpService) -> None:
    svc.config.infra_url = "https://host.example/reverse-proxy-prefix/"  # pyright: ignore[reportAttributeAccessIssue]
    custom = _make_proxy_custom(
        oauth={
            "enabled": True,
            "client_id": "existing-client",
            "authorization_endpoint": "http://as.example/authorize",
            "token_endpoint": "http://as.example/token",
        }
    )
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    authorize_url = await svc.start_oauth_flow("default", "my-remote-mcp")

    assert "redirect_uri=https%3A%2F%2Fhost.example%2Freverse-proxy-prefix%2Fmcp-oauth%2Fcallback" in authorize_url


@pytest.mark.asyncio
async def test_complete_oauth_callback_model_gone_returns_message(svc: McpService) -> None:
    svc._oauth_state_store.add(  # pyright: ignore[reportPrivateUsage]
        "state-1",
        PendingOAuthFlow(instance="default", model_id="removed-model", code_verifier="v", redirect_uri="http://x", resource="http://y"),
    )

    result = await svc.complete_oauth_callback("state-1", "code-1", None)

    assert "no longer exists" in result.lower()


@pytest.mark.asyncio
async def test_complete_oauth_callback_missing_code_returns_message(svc: McpService) -> None:
    custom = _make_proxy_custom(oauth={"enabled": True})
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]
    svc._oauth_state_store.add(  # pyright: ignore[reportPrivateUsage]
        "state-1",
        PendingOAuthFlow(instance="default", model_id="my-remote-mcp", code_verifier="v", redirect_uri="http://x", resource="http://y"),
    )

    result = await svc.complete_oauth_callback("state-1", None, None)

    assert "no authorization code" in result.lower()


@pytest.mark.asyncio
async def test_complete_oauth_callback_discovery_fails_returns_message(svc: McpService) -> None:
    custom = _make_proxy_custom(oauth={"enabled": True, "client_id": "client-1"})
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]
    svc._oauth_state_store.add(  # pyright: ignore[reportPrivateUsage]
        "state-1",
        PendingOAuthFlow(
            instance="default",
            model_id="my-remote-mcp",
            code_verifier="v",
            redirect_uri="http://infra.example/mcp-oauth/callback",
            resource="http://mcp.example.com/mcp",
        ),
    )

    with patch("server.services.mcp_service.discover_authorization_server_for_resource", new=AsyncMock(return_value=None)):
        result = await svc.complete_oauth_callback("state-1", "code-1", None)

    assert "could not discover" in result.lower()


@pytest.mark.asyncio
async def test_complete_oauth_callback_discovery_succeeds_mid_flow(svc: McpService) -> None:
    custom = _make_proxy_custom(oauth={"enabled": True, "client_id": "client-1"})
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]
    svc._oauth_state_store.add(  # pyright: ignore[reportPrivateUsage]
        "state-1",
        PendingOAuthFlow(
            instance="default",
            model_id="my-remote-mcp",
            code_verifier="v",
            redirect_uri="http://infra.example/mcp-oauth/callback",
            resource="http://mcp.example.com/mcp",
        ),
    )
    metadata = AuthorizationServerMetadata(
        issuer="http://as.example",
        authorization_endpoint="http://as.example/authorize",
        token_endpoint="http://as.example/token",
    )
    token = TokenResponse(access_token="new-access", expires_in=3600)

    with (
        patch("server.services.mcp_service.discover_authorization_server_for_resource", new=AsyncMock(return_value=metadata)),
        patch("server.services.mcp_service.exchange_code_for_token", new=AsyncMock(return_value=token)),
        patch.object(svc, "_save", new=AsyncMock()),
        patch.object(svc, "_start_oauth_refresh_background"),  # pyright: ignore[reportPrivateUsage]
    ):
        result = await svc.complete_oauth_callback("state-1", "code-1", None)

    assert "complete" in result.lower()
    updated = svc.models["default"]["my-remote-mcp"]
    assert updated.oauth is not None
    assert updated.oauth.token_endpoint == "http://as.example/token"
    assert updated.oauth.access_token == "new-access"


@pytest.mark.asyncio
async def test_complete_oauth_callback_dynamic_registration_fails_returns_message(svc: McpService) -> None:
    custom = _make_proxy_custom(
        oauth={
            "enabled": True,
            "token_endpoint": "http://as.example/token",
            "authorization_endpoint": "http://as.example/authorize",
            "registration_endpoint": "http://as.example/register",
        }
    )
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]
    svc._oauth_state_store.add(  # pyright: ignore[reportPrivateUsage]
        "state-1",
        PendingOAuthFlow(
            instance="default",
            model_id="my-remote-mcp",
            code_verifier="v",
            redirect_uri="http://infra.example/mcp-oauth/callback",
            resource="http://mcp.example.com/mcp",
        ),
    )

    with patch("server.services.mcp_service.register_dynamic_client", new=AsyncMock(return_value=None)):
        result = await svc.complete_oauth_callback("state-1", "code-1", None)

    assert "no longer configured" in result.lower()
    updated = svc.models["default"]["my-remote-mcp"]
    assert updated.oauth is not None
    assert updated.oauth.client_id is None


@pytest.mark.asyncio
async def test_complete_oauth_callback_registers_dynamic_client_when_missing(svc: McpService) -> None:
    custom = _make_proxy_custom(
        oauth={
            "enabled": True,
            "token_endpoint": "http://as.example/token",
            "authorization_endpoint": "http://as.example/authorize",
            "registration_endpoint": "http://as.example/register",
        }
    )
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]
    svc._oauth_state_store.add(  # pyright: ignore[reportPrivateUsage]
        "state-1",
        PendingOAuthFlow(
            instance="default",
            model_id="my-remote-mcp",
            code_verifier="v",
            redirect_uri="http://infra.example/mcp-oauth/callback",
            resource="http://mcp.example.com/mcp",
        ),
    )
    dcr = DcrResponse(client_id="dcr-client-1", client_secret="dcr-secret-1")
    token = TokenResponse(access_token="new-access", expires_in=3600)

    with (
        patch("server.services.mcp_service.register_dynamic_client", new=AsyncMock(return_value=dcr)),
        patch("server.services.mcp_service.exchange_code_for_token", new=AsyncMock(return_value=token)),
        patch.object(svc, "_save", new=AsyncMock()),
        patch.object(svc, "_start_oauth_refresh_background"),  # pyright: ignore[reportPrivateUsage]
    ):
        result = await svc.complete_oauth_callback("state-1", "code-1", None)

    assert "complete" in result.lower()
    updated = svc.models["default"]["my-remote-mcp"]
    assert updated.oauth is not None
    assert updated.oauth.client_id == "dcr-client-1"


@pytest.mark.asyncio
async def test_complete_oauth_callback_no_client_id_and_no_registration_returns_message(svc: McpService) -> None:
    custom = _make_proxy_custom(
        oauth={
            "enabled": True,
            "token_endpoint": "http://as.example/token",
            "authorization_endpoint": "http://as.example/authorize",
        }
    )
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]
    svc._oauth_state_store.add(  # pyright: ignore[reportPrivateUsage]
        "state-1",
        PendingOAuthFlow(
            instance="default",
            model_id="my-remote-mcp",
            code_verifier="v",
            redirect_uri="http://infra.example/mcp-oauth/callback",
            resource="http://mcp.example.com/mcp",
        ),
    )

    result = await svc.complete_oauth_callback("state-1", "code-1", None)

    assert "no longer configured" in result.lower()


@pytest.mark.asyncio
async def test_complete_oauth_callback_token_exchange_failure_persists_error(svc: McpService) -> None:
    custom = _make_proxy_custom(
        oauth={
            "enabled": True,
            "client_id": "client-1",
            "token_endpoint": "http://as.example/token",
            "authorization_endpoint": "http://as.example/authorize",
        }
    )
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]
    svc._oauth_state_store.add(  # pyright: ignore[reportPrivateUsage]
        "state-1",
        PendingOAuthFlow(
            instance="default",
            model_id="my-remote-mcp",
            code_verifier="v",
            redirect_uri="http://infra.example/mcp-oauth/callback",
            resource="http://mcp.example.com/mcp",
        ),
    )

    with (
        patch("server.services.mcp_service.exchange_code_for_token", new=AsyncMock(side_effect=McpOAuthError("invalid_grant"))),
        patch.object(svc, "_save", new=AsyncMock()),
    ):
        result = await svc.complete_oauth_callback("state-1", "code-1", None)

    assert "failed" in result.lower()
    updated = svc.models["default"]["my-remote-mcp"]
    assert updated.oauth is not None
    assert updated.oauth.last_error == "invalid_grant"


def test_get_oauth_status_error_when_last_error_set_without_token(svc: McpService) -> None:
    custom = _make_proxy_custom(oauth={"enabled": True, "last_error": "boom"})
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    status = svc.get_oauth_status("default", "my-remote-mcp")

    assert status.status == "error"
    assert status.last_error == "boom"


def test_get_oauth_status_not_started_when_nothing_set(svc: McpService) -> None:
    custom = _make_proxy_custom(oauth={"enabled": True})
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    status = svc.get_oauth_status("default", "my-remote-mcp")

    assert status.status == "not_started"


@pytest.mark.asyncio
async def test_install_model_proxy_healthcheck_exception_treated_as_unhealthy(svc: McpService) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    svc._add_custom_model("default", _make_proxy_custom())  # pyright: ignore[reportPrivateUsage]

    with (
        patch.object(svc, "healthcheck_model", new=AsyncMock(side_effect=RuntimeError("boom"))),
        patch.object(svc, "_fetch_tools_background", new=AsyncMock()) as mock_bg,  # pyright: ignore[reportPrivateUsage]
    ):
        promise = await svc._install_model("default", "my-remote-mcp", InstallModelIn())  # pyright: ignore[reportPrivateUsage]
        result = await promise.wait()
        await asyncio.sleep(0)

    assert result.status == "OK"
    mock_bg.assert_awaited_once_with("default", "my-remote-mcp")


@mock.patch("server.services.mcp_service._fetch_tools_from_mcp_endpoint")
@pytest.mark.asyncio
async def test_install_model_proxy_with_refresh_token_starts_background_refresh(mock_fetch: AsyncMock, svc: McpService) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    custom = _make_proxy_custom(oauth={"enabled": True, "client_id": "client-1", "refresh_token": "rtok", "access_token": "tok"})
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]
    mock_fetch.return_value = McpHealthCheckResult(healthy=True, transport="streamable_http", tools=[])

    with patch.object(svc, "_start_oauth_refresh_background") as mock_start:  # pyright: ignore[reportPrivateUsage]
        promise = await svc._install_model("default", "my-remote-mcp", InstallModelIn())  # pyright: ignore[reportPrivateUsage]
        await promise.wait()
        await asyncio.sleep(0)

    mock_start.assert_called_once_with("default", "my-remote-mcp")


def _install_model_stub(svc: McpService, prefix: str = "my-prefix") -> None:
    svc.instances_info["default"].installed = InstalledInfo(
        models={"open-websearch": MagicMock(prefix=prefix)},
        options=MagicMock(),
    )


@pytest.mark.asyncio
async def test_health_monitor_marks_unhealthy_after_consecutive_failures(svc: McpService) -> None:
    _install_model_stub(svc)
    reg_model = MagicMock(healthy=True)
    svc.endpoint_registry.mcp_endpoints.models.get.return_value = {"v1": reg_model}  # pyright: ignore[reportAttributeAccessIssue]
    down = McpHealthCheckResult(healthy=False, error="Connection refused")

    with (
        mock.patch("server.services.mcp_service.asyncio.sleep", new=AsyncMock()),
        patch.object(svc, "healthcheck_model", new=AsyncMock(side_effect=[down, down, down, asyncio.CancelledError()])),
        pytest.raises(asyncio.CancelledError),
    ):
        await svc._health_monitor_background("default", "open-websearch")  # pyright: ignore[reportPrivateUsage]

    assert reg_model.healthy is False
    svc.endpoint_registry.parent_infra.send_models_list.assert_called_once()  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.asyncio
async def test_health_monitor_tolerates_transient_failures(svc: McpService) -> None:
    _install_model_stub(svc)
    reg_model = MagicMock(healthy=True)
    svc.endpoint_registry.mcp_endpoints.models.get.return_value = {"v1": reg_model}  # pyright: ignore[reportAttributeAccessIssue]
    down = McpHealthCheckResult(healthy=False, error="timeout")
    up = McpHealthCheckResult(healthy=True)

    with (
        mock.patch("server.services.mcp_service.asyncio.sleep", new=AsyncMock()),
        patch.object(svc, "healthcheck_model", new=AsyncMock(side_effect=[down, down, up, down, down, asyncio.CancelledError()])),
        pytest.raises(asyncio.CancelledError),
    ):
        await svc._health_monitor_background("default", "open-websearch")  # pyright: ignore[reportPrivateUsage]

    assert reg_model.healthy is True
    svc.endpoint_registry.parent_infra.send_models_list.assert_not_called()  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.asyncio
async def test_health_monitor_ignores_probe_crashes_and_oauth(svc: McpService) -> None:
    _install_model_stub(svc)
    reg_model = MagicMock(healthy=True)
    svc.endpoint_registry.mcp_endpoints.models.get.return_value = {"v1": reg_model}  # pyright: ignore[reportAttributeAccessIssue]
    oauth = McpHealthCheckResult(healthy=False, requires_oauth=True)
    outcomes = [RuntimeError("boom")] * 4 + [oauth] * 4 + [asyncio.CancelledError()]

    with (
        mock.patch("server.services.mcp_service.asyncio.sleep", new=AsyncMock()),
        patch.object(svc, "healthcheck_model", new=AsyncMock(side_effect=outcomes)),
        pytest.raises(asyncio.CancelledError),
    ):
        await svc._health_monitor_background("default", "open-websearch")  # pyright: ignore[reportPrivateUsage]

    assert reg_model.healthy is True
    svc.endpoint_registry.parent_infra.send_models_list.assert_not_called()  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.asyncio
async def test_health_monitor_recovers_via_healthcheck_after_unhealthy(svc: McpService) -> None:
    model = MagicMock(spec=SrvMcpModel)
    model.model_props = ModelProps(private=False, type="mcp", endpoints=[])
    reg_model = MagicMock(healthy=False)
    reg_model.props = model.model_props
    svc.endpoint_registry.mcp_endpoints.models.get.return_value = {"v1": reg_model}  # pyright: ignore[reportAttributeAccessIssue]

    svc._apply_healthcheck_result(model, "my-prefix", McpHealthCheckResult(healthy=True))  # pyright: ignore[reportPrivateUsage]

    assert reg_model.healthy is True
    svc.endpoint_registry.parent_infra.send_models_list.assert_called_once()  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.asyncio
async def test_health_monitor_stops_when_model_uninstalled(svc: McpService) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=MagicMock())

    with (
        mock.patch("server.services.mcp_service.asyncio.sleep", new=AsyncMock()),
        patch.object(svc, "healthcheck_model", new=AsyncMock()) as mock_hc,
    ):
        await svc._health_monitor_background("default", "open-websearch")  # pyright: ignore[reportPrivateUsage]

    mock_hc.assert_not_awaited()


def test_mark_unhealthy_skips_already_unhealthy_registrations(svc: McpService) -> None:
    already_down = MagicMock(healthy=False)
    up = MagicMock(healthy=True)
    svc.endpoint_registry.mcp_endpoints.models.get.return_value = {"v1": already_down, "v2": up}  # pyright: ignore[reportAttributeAccessIssue]

    svc._mark_unhealthy("my-prefix")  # pyright: ignore[reportPrivateUsage]

    assert already_down.healthy is False
    assert up.healthy is False
    svc.endpoint_registry.parent_infra.send_models_list.assert_called_once()  # pyright: ignore[reportAttributeAccessIssue]


def test_mark_unhealthy_does_not_notify_when_already_unhealthy(svc: McpService) -> None:
    svc.endpoint_registry.mcp_endpoints.models.get.return_value = {"v1": MagicMock(healthy=False)}  # pyright: ignore[reportAttributeAccessIssue]

    svc._mark_unhealthy("my-prefix")  # pyright: ignore[reportPrivateUsage]

    svc.endpoint_registry.parent_infra.send_models_list.assert_not_called()  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.asyncio
async def test_health_monitor_survives_mark_unhealthy_failure(svc: McpService) -> None:
    _install_model_stub(svc)
    down = McpHealthCheckResult(healthy=False, error="refused")

    with (
        mock.patch("server.services.mcp_service.asyncio.sleep", new=AsyncMock()),
        patch.object(svc, "healthcheck_model", new=AsyncMock(side_effect=[down] * 4 + [asyncio.CancelledError()])),
        patch.object(svc, "_mark_unhealthy", side_effect=RuntimeError("registry broke")) as mock_mark,
        pytest.raises(asyncio.CancelledError),
    ):
        await svc._health_monitor_background("default", "open-websearch")  # pyright: ignore[reportPrivateUsage]

    assert mock_mark.call_count == 2


@pytest.mark.asyncio
async def test_start_health_monitor_is_idempotent_and_stoppable(svc: McpService) -> None:
    async def sleep_forever(*_: object) -> None:
        await asyncio.sleep(3600)

    with patch.object(svc, "_health_monitor_background", new=sleep_forever):
        svc._start_health_monitor("default", "m")  # pyright: ignore[reportPrivateUsage]
        first = svc._health_monitor_tasks[("default", "m")]  # pyright: ignore[reportPrivateUsage]
        svc._start_health_monitor("default", "m")  # pyright: ignore[reportPrivateUsage]
        assert svc._health_monitor_tasks[("default", "m")] is first  # pyright: ignore[reportPrivateUsage]

        await svc.stop_instance("default")
        await asyncio.sleep(0)

    assert first.cancelled()
    assert ("default", "m") not in svc._health_monitor_tasks  # pyright: ignore[reportPrivateUsage]
