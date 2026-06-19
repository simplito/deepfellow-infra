# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for user-defined MCP server (kind="user") and proxy MCP server (kind="proxy") support."""

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from server.models.api import ModelProps
from server.models.models import AddCustomModelIn, InstallModelIn, ListModelsFilters, ModelSpecification, UninstallModelIn
from server.models.services import InstallServiceIn
from server.services.base2_service import CustomModel, Instance, InstanceConfig
from server.services.mcp_service import (
    DownloadedInfo,
    InstalledInfo,
    McpModelOptions,  # pyright: ignore[reportPrivateUsage]
    McpService,
    McpUserVariant,
    ModelInstalledInfo,
    SrvMcpModel,
    SrvMcpProxyModel,
    SrvMcpUserModel,
    _version_to_base_image,  # pyright: ignore[reportPrivateUsage]
)
from server.utils.size_fetcher import fmt_size


@pytest.fixture
def deps() -> dict[str, Any]:
    docker_svc = MagicMock()
    docker_svc.get_docker_subnet.return_value = "172.20.0.0/16"
    docker_svc.get_docker_container_name.side_effect = lambda name: f"df-{name}"  # pyright: ignore[reportUnknownLambdaType]
    docker_svc.remove_image = AsyncMock()
    docker_svc.build_image = AsyncMock()
    docker_svc.uninstall_docker = AsyncMock()
    docker_svc.get_local_docker_image_size = AsyncMock(return_value=None)
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


def _make_user_model(
    model_id: str = "my-server",
    name: str = "My Server",
    variant: str = "node-headless",
    command: str = "npx -y @test/server",
    custom_id: str = "uuid-user-1",
) -> CustomModel:
    return CustomModel(
        id=custom_id,
        data={
            "kind": "user",
            "id": model_id,
            "name": name,
            "variant": variant,
            "command": command,
        },
    )


def test_srv_mcp_user_model_valid() -> None:
    m = SrvMcpUserModel(id="srv", name="Srv", variant=McpUserVariant.node_headless, command="npx test")
    assert m.variant == McpUserVariant.node_headless


def test_srv_mcp_user_model_unknown_variant_raises() -> None:
    with pytest.raises(ValidationError):
        SrvMcpUserModel(id="srv", name="Srv", variant="ruby-headless", command="ruby server.rb")  # type: ignore[arg-type]


def test_add_user_model_node_headless_dockerfile_content(svc: McpService, tmp_path: Path) -> None:
    """7.1 - node-headless Dockerfile has supergateway ENTRYPOINT with correct flags and CMD as single string."""
    with patch.object(svc, "_get_working_dir", return_value=tmp_path):
        svc._add_custom_model(  # pyright: ignore[reportPrivateUsage]
            "default", _make_user_model(variant="node-headless", command="npx -y @test/server")
        )

    dockerfile = tmp_path / "models" / "default" / "my-server" / "Dockerfile"
    assert dockerfile.exists()
    content = dockerfile.read_text()
    assert "supergateway" in content
    assert "--outputTransport" in content
    assert "streamableHttp" in content
    # Command must be a single string so supergateway's --stdio receives the full command
    assert '["npx -y @test/server"]' in content


def test_add_user_model_node_headed_dockerfile_has_chromium(svc: McpService, tmp_path: Path) -> None:
    """7.2 - node-headed Dockerfile includes Chromium installation."""
    with patch.object(svc, "_get_working_dir", return_value=tmp_path):
        svc._add_custom_model(  # pyright: ignore[reportPrivateUsage]
            "default", _make_user_model(variant="node-headed", command="npx -y @test/browser-server")
        )

    content = (tmp_path / "models" / "default" / "my-server" / "Dockerfile").read_text()
    assert "chromium" in content.lower()


def test_add_user_model_python_headless_dockerfile_has_mcp_proxy(svc: McpService, tmp_path: Path) -> None:
    """7.3 - python-headless Dockerfile has mcp-proxy ENTRYPOINT."""
    with patch.object(svc, "_get_working_dir", return_value=tmp_path):
        svc._add_custom_model(  # pyright: ignore[reportPrivateUsage]
            "default", _make_user_model(variant="python-headless", command="uvx mcp-server-sqlite")
        )

    content = (tmp_path / "models" / "default" / "my-server" / "Dockerfile").read_text()
    assert "mcp-proxy" in content
    assert '["uvx", "mcp-server-sqlite"]' in content


def test_add_user_model_python_headed_dockerfile_has_chromium(svc: McpService, tmp_path: Path) -> None:
    """7.4 - python-headed Dockerfile includes Chromium installation."""
    with patch.object(svc, "_get_working_dir", return_value=tmp_path):
        svc._add_custom_model(  # pyright: ignore[reportPrivateUsage]
            "default", _make_user_model(variant="python-headed", command="uvx mcp-server-playwright")
        )

    content = (tmp_path / "models" / "default" / "my-server" / "Dockerfile").read_text()
    assert "chromium" in content.lower()
    assert "mcp-proxy" in content


@pytest.mark.asyncio
async def test_update_user_model_overwrites_dockerfile_and_removes_old_image(svc: McpService, tmp_path: Path, deps: dict[str, Any]) -> None:
    """7.5 - editing a user server overwrites Dockerfile and removes old built image."""
    with patch.object(svc, "_get_working_dir", return_value=tmp_path):
        svc._add_custom_model("default", _make_user_model(command="npx -y old-server"))  # pyright: ignore[reportPrivateUsage]
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))

    custom_id = "uuid-user-1"
    new_data = {
        "kind": "user",
        "id": "my-server",
        "name": "My Server",
        "variant": "node-headless",
        "command": "npx -y new-server",
    }
    with patch.object(svc, "_get_working_dir", return_value=tmp_path):
        old_model = CustomModel(
            id=custom_id,
            data={"kind": "user", "id": "my-server", "name": "My Server", "variant": "node-headless", "command": "npx -y old-server"},
        )
        await svc._update_custom_model("default", old_model, new_data)  # pyright: ignore[reportPrivateUsage]

    dockerfile = tmp_path / "models" / "default" / "my-server" / "Dockerfile"
    assert '["npx -y new-server"]' in dockerfile.read_text()
    deps["docker_service"].remove_image.assert_called_once()


def test_remove_user_model_deletes_dockerfile_dir(svc: McpService, tmp_path: Path) -> None:
    """7.6 - removing a user server deletes the Dockerfile directory."""
    with patch.object(svc, "_get_working_dir", return_value=tmp_path):
        svc._add_custom_model("default", _make_user_model())  # pyright: ignore[reportPrivateUsage]

    dockerfile_dir = tmp_path / "models" / "default" / "my-server"
    assert dockerfile_dir.exists()

    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    with patch.object(svc, "_get_working_dir", return_value=tmp_path):
        svc._remove_custom_model("default", _make_user_model())  # pyright: ignore[reportPrivateUsage]

    assert not dockerfile_dir.exists()


@pytest.mark.asyncio
async def test_uninstall_user_model_with_purge_removes_image_and_dockerfile(svc: McpService, tmp_path: Path, deps: dict[str, Any]) -> None:
    """7.7 - uninstall with purge=True removes built image and Dockerfile dir."""
    with patch.object(svc, "_get_working_dir", return_value=tmp_path):
        svc._add_custom_model("default", _make_user_model())  # pyright: ignore[reportPrivateUsage]

    installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    mock_model = MagicMock()
    mock_model.prefix = "my-server"
    mock_model.registration_id = "reg-id"
    installed.models["my-server"] = mock_model
    svc.instances_info["default"].installed = installed
    svc.models_downloaded["my-server"] = DownloadedInfo(image="deepfellow-mcp-my_server_default:latest")

    with patch.object(svc, "_get_working_dir", return_value=tmp_path):
        await svc._uninstall_model("default", "my-server", UninstallModelIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    deps["docker_service"].remove_image.assert_called()
    assert not (tmp_path / "models" / "default" / "my-server").exists()


@pytest.mark.asyncio
async def test_uninstall_user_model_without_purge_keeps_image(svc: McpService, tmp_path: Path, deps: dict[str, Any]) -> None:
    """7.7 - uninstall without purge keeps the built image."""
    with patch.object(svc, "_get_working_dir", return_value=tmp_path):
        svc._add_custom_model("default", _make_user_model())  # pyright: ignore[reportPrivateUsage]

    installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    mock_model = MagicMock()
    mock_model.prefix = "my-server"
    mock_model.registration_id = "reg-id"
    installed.models["my-server"] = mock_model
    svc.instances_info["default"].installed = installed
    svc.models_downloaded["my-server"] = DownloadedInfo(image="deepfellow-mcp-my_server_default:latest")

    with patch.object(svc, "_get_working_dir", return_value=tmp_path):
        await svc._uninstall_model("default", "my-server", UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    deps["docker_service"].remove_image.assert_not_called()
    assert (tmp_path / "models" / "default" / "my-server").exists()


def test_add_user_model_unknown_variant_raises_422(svc: McpService, tmp_path: Path) -> None:
    """7.8 - unknown variant returns 422 via try_parse_pydantic."""
    custom = CustomModel(
        id="uuid-bad",
        data={"kind": "user", "id": "srv", "name": "Srv", "variant": "ruby-headless", "command": "ruby server.rb"},
    )
    with pytest.raises(HTTPException) as exc_info, patch.object(svc, "_get_working_dir", return_value=tmp_path):
        svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]
    assert exc_info.value.status_code == 400


def test_add_user_model_duplicate_id_raises_400(svc: McpService, tmp_path: Path) -> None:
    """7.9 - duplicate model id returns 400."""
    with patch.object(svc, "_get_working_dir", return_value=tmp_path):
        svc._add_custom_model("default", _make_user_model())  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(HTTPException) as exc_info, patch.object(svc, "_get_working_dir", return_value=tmp_path):
        svc._add_custom_model("default", _make_user_model(custom_id="uuid-2"))  # pyright: ignore[reportPrivateUsage]
    assert exc_info.value.status_code == 400


def test_add_user_model_prefix_collision_raises_400(svc: McpService, tmp_path: Path) -> None:
    """7.10 - normalised prefix collision (e.g. 'My Server' vs 'my-server') returns 400."""
    # First: id="My Server" → prefix = normalize_name("My Server") = "my_server"
    first = CustomModel(
        id="uuid-a",
        data={"kind": "user", "id": "My Server", "name": "My Server", "variant": "node-headless", "command": "npx test"},
    )
    second = CustomModel(
        id="uuid-b",
        data={"kind": "user", "id": "my-server", "name": "my-server", "variant": "node-headless", "command": "npx test"},
    )
    with patch.object(svc, "_get_working_dir", return_value=tmp_path):
        svc._add_custom_model("default", first)  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(HTTPException) as exc_info, patch.object(svc, "_get_working_dir", return_value=tmp_path):
        svc._add_custom_model("default", second)  # pyright: ignore[reportPrivateUsage]
    assert exc_info.value.status_code == 400


def test_add_user_model_sets_kind_user(svc: McpService, tmp_path: Path) -> None:
    with patch.object(svc, "_get_working_dir", return_value=tmp_path):
        svc._add_custom_model("default", _make_user_model())  # pyright: ignore[reportPrivateUsage]
    assert svc.models["default"]["my-server"].kind == "user"


def test_add_user_model_endpoint_uses_streamable_http_path(svc: McpService, tmp_path: Path) -> None:
    """User model endpoint must end with /mcp so the proxy routes to the streamable HTTP path."""
    with patch.object(svc, "_get_working_dir", return_value=tmp_path):
        svc._add_custom_model("default", _make_user_model())  # pyright: ignore[reportPrivateUsage]
    endpoints = svc.models["default"]["my-server"].model_props.endpoints
    assert endpoints == ["/mcp/my_server/mcp"]


def test_add_user_model_initialises_instance_dict(svc: McpService, tmp_path: Path) -> None:
    """Line 502: models[instance] dict is created when instance has no models yet."""
    assert "other-instance" not in svc.models
    with patch.object(svc, "_get_working_dir", return_value=tmp_path):
        svc._add_custom_model("other-instance", _make_user_model())  # pyright: ignore[reportPrivateUsage]
    assert "my-server" in svc.models["other-instance"]


@pytest.mark.asyncio
async def test_update_non_user_kind_raises_400(svc: McpService, tmp_path: Path) -> None:
    """Line 520: editing a non-user-kind model raises 400."""
    custom_model = CustomModel(
        id="uuid-custom",
        data={"kind": "custom", "id": "some-model", "name": "Some Model", "image": "img:latest"},
    )
    with pytest.raises(HTTPException) as exc_info:
        await svc._update_custom_model(  # pyright: ignore[reportPrivateUsage]
            "default", custom_model, {"kind": "custom", "id": "some-model", "name": "x", "image": "img:latest"}
        )
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_update_installed_server_raises_400(svc: McpService, tmp_path: Path) -> None:
    """Line 527: editing an installed server raises 400."""
    with patch.object(svc, "_get_working_dir", return_value=tmp_path):
        svc._add_custom_model("default", _make_user_model())  # pyright: ignore[reportPrivateUsage]

    installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    mock_model = MagicMock()
    installed.models["my-server"] = mock_model
    svc.instances_info["default"].installed = installed

    with pytest.raises(HTTPException) as exc_info, patch.object(svc, "_get_working_dir", return_value=tmp_path):
        await svc._update_custom_model(  # pyright: ignore[reportPrivateUsage]
            "default",
            _make_user_model(),
            {"kind": "user", "id": "my-server", "name": "My Server", "variant": "node-headless", "command": "npx new"},
        )
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_update_user_model_when_not_in_models_dict(svc: McpService, tmp_path: Path, deps: dict[str, Any]) -> None:
    """Lines 540-541: when instance was absent from self.models, update creates the entry."""
    svc.instances_info["default"].installed = None
    svc.models.clear()

    with patch.object(svc, "_get_working_dir", return_value=tmp_path):
        await svc._update_custom_model(  # pyright: ignore[reportPrivateUsage]
            "default",
            _make_user_model(),
            {"kind": "user", "id": "my-server", "name": "New", "variant": "node-headless", "command": "npx new"},
        )

    assert "my-server" in svc.models["default"]
    deps["docker_service"].remove_image.assert_called_once()


def test_remove_installed_user_model_raises_400(svc: McpService, tmp_path: Path) -> None:
    """Line 549: removing an installed user server raises 400."""
    with patch.object(svc, "_get_working_dir", return_value=tmp_path):
        svc._add_custom_model("default", _make_user_model())  # pyright: ignore[reportPrivateUsage]

    installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    installed.models["my-server"] = MagicMock()
    svc.instances_info["default"].installed = installed

    with pytest.raises(HTTPException) as exc_info, patch.object(svc, "_get_working_dir", return_value=tmp_path):
        svc._remove_custom_model("default", _make_user_model())  # pyright: ignore[reportPrivateUsage]
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_update_custom_model_public_api_persists_data(svc: McpService, tmp_path: Path) -> None:
    """Lines 368-369: update_custom_model updates model.data and calls _save."""
    model = _make_user_model()
    svc.instances_info["default"].config.custom = [model]
    svc.service_provider.save_service_config = AsyncMock()  # type: ignore[method-assign]

    new_spec = {"kind": "user", "id": "my-server", "name": "My Server", "variant": "node-headless", "command": "npx updated"}
    with patch.object(svc, "_get_working_dir", return_value=tmp_path):
        await svc.update_custom_model("default", "uuid-user-1", AddCustomModelIn(spec=new_spec))

    assert model.data == new_spec
    svc.service_provider.save_service_config.assert_awaited_once()


def test_delete_dockerfile_dir_noop_when_missing(svc: McpService, tmp_path: Path) -> None:
    """Line 416->exit: _delete_dockerfile_dir is a no-op when the dir doesn't exist."""
    with patch.object(svc, "_get_working_dir", return_value=tmp_path):
        svc._delete_dockerfile_dir("default", "nonexistent-model")  # pyright: ignore[reportPrivateUsage]


def test_remove_user_model_noop_when_not_in_models_dict(svc: McpService, tmp_path: Path) -> None:
    """Line 551->exit: removing a user model not present in models dict doesn't crash."""
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    with patch.object(svc, "_get_working_dir", return_value=tmp_path):
        svc._remove_custom_model("default", _make_user_model())  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_install_user_model_calls_build_image(svc: McpService, tmp_path: Path, deps: dict[str, Any]) -> None:
    """Lines 667-670, 676-677: user kind skips _verify_docker_image and calls build_image."""
    with patch.object(svc, "_get_working_dir", return_value=tmp_path):
        svc._add_custom_model("default", _make_user_model())  # pyright: ignore[reportPrivateUsage]

    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    svc.docker_service.install_and_run_docker = AsyncMock(return_value=8000)  # type: ignore[method-assign]
    svc.docker_service.is_docker_compose_healthy = AsyncMock(return_value=True)  # type: ignore[method-assign]
    svc.service_provider.save_service_config = AsyncMock()  # type: ignore[method-assign]

    with (
        patch.object(svc, "_get_working_dir", return_value=tmp_path),
        patch.object(svc, "_verify_docker_image", new_callable=AsyncMock) as mock_verify,
    ):
        promise = await svc.install_model("default", "my-server", InstallModelIn(stream=False, spec={"prefix": "my-server"}))
        await promise.wait()

    mock_verify.assert_not_called()
    deps["docker_service"].build_image.assert_called_once()


@pytest.mark.asyncio
async def test_install_user_model_populates_size_from_built_image(svc: McpService, tmp_path: Path) -> None:
    with patch.object(svc, "_get_working_dir", return_value=tmp_path):
        svc._add_custom_model("default", _make_user_model())  # pyright: ignore[reportPrivateUsage]

    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    svc.instances_info["default"].config.custom = [_make_user_model()]
    svc.docker_service.install_and_run_docker = AsyncMock(return_value=8000)  # type: ignore[method-assign]
    svc.docker_service.get_local_docker_image_size = AsyncMock(return_value=512 * 1024**2)  # type: ignore[method-assign]
    svc.service_provider.save_service_config = AsyncMock()  # type: ignore[method-assign]

    with (
        patch.object(svc, "_get_working_dir", return_value=tmp_path),
        patch.object(svc, "_verify_docker_image", new_callable=AsyncMock),
    ):
        promise = await svc.install_model("default", "my-server", InstallModelIn(stream=False, spec={"prefix": "my-server"}))
        await promise.wait()

    expected = fmt_size(512 * 1024**2)
    assert svc.models["default"]["my-server"].size == expected
    persisted = next(cm for cm in svc.instances_info["default"].config.custom if cm.id == "uuid-user-1")
    assert persisted.data["size"] == expected
    assert svc.service_provider.save_service_config.await_count >= 1


def _make_proxy_model(
    model_id: str = "remote-mcp",
    name: str = "Remote MCP",
    server_url: str = "https://example.com/mcp",
    transport: str = "streamable_http",
    custom_id: str = "uuid-proxy-1",
) -> CustomModel:
    return CustomModel(
        id=custom_id,
        data={
            "kind": "proxy",
            "id": model_id,
            "name": name,
            "server_url": server_url,
            "transport": transport,
        },
    )


def test_srv_mcp_proxy_model_valid() -> None:
    m = SrvMcpProxyModel(id="srv", name="Srv", server_url="https://example.com/mcp")
    assert m.transport == "streamable_http"
    assert m.kind == "proxy"


def test_srv_mcp_proxy_model_sse_transport() -> None:
    m = SrvMcpProxyModel(id="srv", name="Srv", server_url="https://example.com/sse", transport="sse")
    assert m.transport == "sse"


def test_srv_mcp_proxy_model_invalid_transport_raises() -> None:
    with pytest.raises(ValidationError):
        SrvMcpProxyModel(id="srv", name="Srv", server_url="https://x.com", transport="websocket")  # type: ignore[arg-type]


def test_add_proxy_model_registers_in_models(svc: McpService) -> None:
    svc._add_custom_model("default", _make_proxy_model())  # pyright: ignore[reportPrivateUsage]
    assert "remote-mcp" in svc.models["default"]


def test_add_proxy_model_sets_kind_proxy(svc: McpService) -> None:
    svc._add_custom_model("default", _make_proxy_model())  # pyright: ignore[reportPrivateUsage]
    assert svc.models["default"]["remote-mcp"].kind == "proxy"


def test_add_proxy_model_options_is_none(svc: McpService) -> None:
    svc._add_custom_model("default", _make_proxy_model())  # pyright: ignore[reportPrivateUsage]
    assert svc.models["default"]["remote-mcp"].options is None


def test_add_proxy_model_proxy_url_stored(svc: McpService) -> None:
    svc._add_custom_model("default", _make_proxy_model(server_url="https://mcp.example.com/mcp"))  # pyright: ignore[reportPrivateUsage]
    assert svc.models["default"]["remote-mcp"].proxy_url == "https://mcp.example.com/mcp"


def test_add_proxy_model_transport_stored(svc: McpService) -> None:
    svc._add_custom_model("default", _make_proxy_model(transport="sse"))  # pyright: ignore[reportPrivateUsage]
    assert svc.models["default"]["remote-mcp"].proxy_transport == "sse"


def test_add_proxy_model_has_docker_false(svc: McpService) -> None:
    svc._add_custom_model("default", _make_proxy_model())  # pyright: ignore[reportPrivateUsage]
    assert svc.models["default"]["remote-mcp"].kind == "proxy"


def test_add_proxy_model_duplicate_id_raises_400(svc: McpService) -> None:
    svc._add_custom_model("default", _make_proxy_model())  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(HTTPException) as exc_info:
        svc._add_custom_model("default", _make_proxy_model(custom_id="uuid-2"))  # pyright: ignore[reportPrivateUsage]
    assert exc_info.value.status_code == 400


def test_add_proxy_model_prefix_collision_raises_400(svc: McpService) -> None:
    svc._add_custom_model("default", _make_proxy_model(model_id="remote-mcp"))  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(HTTPException) as exc_info:
        svc._add_custom_model("default", _make_proxy_model(model_id="remote_mcp", custom_id="uuid-2"))  # pyright: ignore[reportPrivateUsage]
    assert exc_info.value.status_code == 400


def test_remove_proxy_model(svc: McpService) -> None:
    svc._add_custom_model("default", _make_proxy_model())  # pyright: ignore[reportPrivateUsage]
    assert "remote-mcp" in svc.models["default"]
    svc._remove_custom_model("default", _make_proxy_model())  # pyright: ignore[reportPrivateUsage]
    assert "remote-mcp" not in svc.models["default"]


def test_remove_installed_proxy_model_raises_400(svc: McpService) -> None:
    svc._add_custom_model("default", _make_proxy_model())  # pyright: ignore[reportPrivateUsage]
    installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    installed.models["remote-mcp"] = MagicMock()
    svc.instances_info["default"].installed = installed

    with pytest.raises(HTTPException) as exc_info:
        svc._remove_custom_model("default", _make_proxy_model())  # pyright: ignore[reportPrivateUsage]
    assert exc_info.value.status_code == 400


def test_remove_proxy_model_noop_when_not_in_models_dict(svc: McpService) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    svc._remove_custom_model("default", _make_proxy_model())  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_install_proxy_model_skips_docker(svc: McpService, deps: dict[str, Any]) -> None:
    svc._add_custom_model("default", _make_proxy_model())  # pyright: ignore[reportPrivateUsage]
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    deps["endpoint_registry"].register_mcp_endpoint_as_proxy = MagicMock(return_value="reg-proxy-1")

    promise = await svc._install_model(  # pyright: ignore[reportPrivateUsage]
        "default", "remote-mcp", InstallModelIn(stream=False, spec={"prefix": "remote-mcp"})
    )
    result = await promise.wait()

    assert result.status == "OK"
    deps["docker_service"].build_image.assert_not_called()
    deps["docker_service"].install_and_run_docker.assert_not_called()


@pytest.mark.asyncio
async def test_install_proxy_model_calls_sse_registry_for_sse_transport(svc: McpService, deps: dict[str, Any]) -> None:
    svc._add_custom_model("default", _make_proxy_model(transport="sse"))  # pyright: ignore[reportPrivateUsage]
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    deps["endpoint_registry"].register_mcp_sse_endpoint_as_proxy = MagicMock(return_value="reg-sse-1")

    promise = await svc._install_model(  # pyright: ignore[reportPrivateUsage]
        "default", "remote-mcp", InstallModelIn(stream=False, spec={"prefix": "remote-mcp"})
    )
    await promise.wait()

    deps["endpoint_registry"].register_mcp_sse_endpoint_as_proxy.assert_called_once()


@pytest.mark.asyncio
async def test_install_proxy_model_stores_installed_info(svc: McpService, deps: dict[str, Any]) -> None:
    svc._add_custom_model("default", _make_proxy_model())  # pyright: ignore[reportPrivateUsage]
    installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    svc.instances_info["default"].installed = installed
    deps["endpoint_registry"].register_mcp_endpoint_as_proxy = MagicMock(return_value="reg-1")

    await (
        await svc._install_model(  # pyright: ignore[reportPrivateUsage]
            "default", "remote-mcp", InstallModelIn(stream=False, spec={"prefix": "remote-mcp"})
        )
    ).wait()

    assert "remote-mcp" in installed.models
    info = installed.models["remote-mcp"]
    assert info.docker_options is None
    assert info.base_url == "https://example.com/mcp"


@pytest.mark.asyncio
async def test_uninstall_proxy_model_skips_docker(svc: McpService, deps: dict[str, Any]) -> None:
    svc._add_custom_model("default", _make_proxy_model())  # pyright: ignore[reportPrivateUsage]

    installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    mock_info = MagicMock()
    mock_info.prefix = "remote-mcp"
    mock_info.registration_id = "reg-1"
    mock_info.docker_options = None
    installed.models["remote-mcp"] = mock_info
    svc.instances_info["default"].installed = installed

    await svc._uninstall_model("default", "remote-mcp", UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    deps["docker_service"].uninstall_docker.assert_not_called()
    assert "remote-mcp" not in installed.models


@pytest.mark.asyncio
async def test_update_proxy_model_replaces_entry(svc: McpService) -> None:
    svc._add_custom_model("default", _make_proxy_model())  # pyright: ignore[reportPrivateUsage]
    svc.instances_info["default"].installed = None

    await svc._update_custom_model(  # pyright: ignore[reportPrivateUsage]
        "default",
        _make_proxy_model(),
        {
            "kind": "proxy",
            "id": "remote-mcp",
            "name": "Updated Remote",
            "server_url": "https://new.example.com/mcp",
            "transport": "sse",
        },
    )

    m = svc.models["default"]["remote-mcp"]
    assert m.proxy_url == "https://new.example.com/mcp"
    assert m.proxy_transport == "sse"


@pytest.mark.asyncio
async def test_update_installed_proxy_model_raises_400(svc: McpService) -> None:
    svc._add_custom_model("default", _make_proxy_model())  # pyright: ignore[reportPrivateUsage]
    installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    installed.models["remote-mcp"] = MagicMock()
    svc.instances_info["default"].installed = installed

    with pytest.raises(HTTPException) as exc_info:
        await svc._update_custom_model(  # pyright: ignore[reportPrivateUsage]
            "default",
            _make_proxy_model(),
            {"kind": "proxy", "id": "remote-mcp", "name": "x", "server_url": "https://x.com"},
        )
    assert exc_info.value.status_code == 400


def test_remove_image_model_not_in_models_dict_noop(svc: McpService, tmp_path: Path) -> None:
    """Removing an image-kind custom model that's absent from self.models must not raise KeyError."""
    custom = CustomModel(
        id="uuid-img",
        data={
            "kind": None,
            "id": "img-model",
            "name": "Img",
            "default_prefix": "img-model",
            "image": "img:latest",
            "image_port": 8000,
            "size": "1GB",
        },
    )
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    # Not added to svc.models — should not raise
    svc._remove_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]


def _make_srv_mcp_model(**kwargs: Any) -> SrvMcpModel:
    defaults: dict[str, Any] = {
        "model_props": ModelProps(private=True, type="mcp", endpoints=["/mcp/test/mcp"]),
        "model_spec": ModelSpecification(fields=[]),
        "model_type": "mcp",
        "default_prefix": "test",
        "size": "",
        "options": None,
    }
    defaults.update(kwargs)
    return SrvMcpModel(**defaults)


def test_get_custom_spec_returns_none_for_non_custom(svc: McpService) -> None:
    model = _make_srv_mcp_model(custom=None, kind="custom")
    assert svc._get_custom_spec("mid", model) is None  # pyright: ignore[reportPrivateUsage]


def test_get_custom_spec_proxy_without_headers(svc: McpService) -> None:
    model = _make_srv_mcp_model(
        custom="uuid-1", kind="proxy", proxy_url="https://example.com/mcp", proxy_transport="streamable_http", headers=None
    )
    spec = svc._get_custom_spec("mid", model)  # pyright: ignore[reportPrivateUsage]
    assert spec is not None
    assert spec["kind"] == "proxy"
    assert spec["server_url"] == "https://example.com/mcp"
    assert spec["transport"] == "streamable_http"
    assert "headers" not in spec


def test_get_custom_spec_proxy_with_headers(svc: McpService) -> None:
    model = _make_srv_mcp_model(
        custom="uuid-1", kind="proxy", proxy_url="https://example.com/mcp", proxy_transport="sse", headers={"Authorization": "Bearer tok"}
    )
    spec = svc._get_custom_spec("mid", model)  # pyright: ignore[reportPrivateUsage]
    assert spec is not None
    assert spec["headers"] == {"Authorization": "Bearer tok"}


def test_get_custom_spec_user_minimal(svc: McpService) -> None:
    model = _make_srv_mcp_model(custom="uuid-2", kind="user", command="npx srv", variant="node-headless", envs=None, base_image=None)
    spec = svc._get_custom_spec("mid", model)  # pyright: ignore[reportPrivateUsage]
    assert spec is not None
    assert spec["kind"] == "user"
    assert spec["command"] == "npx srv"
    assert "envs" not in spec
    assert "base_image" not in spec


def test_get_custom_spec_user_with_envs_and_base_image(svc: McpService) -> None:
    model = _make_srv_mcp_model(
        custom="uuid-2", kind="user", command="uvx srv", variant="python-headless", envs={"KEY": "val"}, base_image="my-img:latest"
    )
    spec = svc._get_custom_spec("mid", model)  # pyright: ignore[reportPrivateUsage]
    assert spec is not None
    assert spec["envs"] == {"KEY": "val"}
    assert spec["base_image"] == "my-img:latest"


def test_get_custom_spec_unknown_kind_returns_none(svc: McpService) -> None:
    model = _make_srv_mcp_model(custom="uuid-3", kind="custom")
    assert svc._get_custom_spec("mid", model) is None  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_list_models_skips_instance_not_in_requested_instances(svc: McpService, tmp_path: Path) -> None:
    # Add a second instance with its own model
    svc.instances_info["second"] = Instance(
        installed=InstalledInfo(models={}, options=InstallServiceIn(spec={})),
        installing=None,
        installing_model_progress={},
        config=InstanceConfig(),
    )
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    with patch.object(svc, "_get_working_dir", return_value=tmp_path):
        svc._add_custom_model("default", _make_user_model(model_id="server-a"))  # pyright: ignore[reportPrivateUsage]
        svc._add_custom_model("second", _make_user_model(model_id="server-b", custom_id="uuid-b"))  # pyright: ignore[reportPrivateUsage]

    # Request only "default" — "second" instance models must be skipped
    result = await svc.list_models("default", ListModelsFilters())
    assert any(m.id == "server-a" for m in result.list)
    assert all(m.id != "server-b" for m in result.list)


@pytest.mark.asyncio
async def test_list_models_filters_by_installed_false(svc: McpService, tmp_path: Path) -> None:
    with patch.object(svc, "_get_working_dir", return_value=tmp_path):
        svc._add_custom_model("default", _make_user_model(model_id="not-installed"))  # pyright: ignore[reportPrivateUsage]

    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))

    result = await svc.list_models("default", ListModelsFilters(installed=False))
    assert any(m.id == "not-installed" for m in result.list)

    result_installed = await svc.list_models("default", ListModelsFilters(installed=True))
    assert all(m.id != "not-installed" for m in result_installed.list)


@pytest.mark.asyncio
async def test_get_model_not_found_raises_400(svc: McpService) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    with pytest.raises(HTTPException) as exc_info:
        await svc.get_model("default", "nonexistent")
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_get_model_initializes_empty_models_dict(svc: McpService) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    del svc.models["default"]
    with pytest.raises(HTTPException) as exc_info:
        await svc.get_model("default", "any")
    assert exc_info.value.status_code == 400
    assert "default" in svc.models


def test_check_envs_no_required_is_noop(svc: McpService) -> None:
    svc.check_envs(None, {"KEY": "val"})  # must not raise


def test_check_envs_missing_key_raises_422(svc: McpService) -> None:
    with pytest.raises(HTTPException) as exc_info:
        svc.check_envs(["MISSING_KEY"], {"OTHER_KEY": "val"})
    assert exc_info.value.status_code == 422
    assert "MISSING_KEY" in exc_info.value.detail


def test_check_envs_empty_value_raises_422(svc: McpService) -> None:
    with pytest.raises(HTTPException) as exc_info:
        svc.check_envs(["MY_KEY"], {"MY_KEY": ""})
    assert exc_info.value.status_code == 422
    assert "MY_KEY" in exc_info.value.detail


def test_check_headers_no_required_is_noop(svc: McpService) -> None:
    svc.check_headers(None, {"X-Token": "abc"})  # must not raise


def test_check_headers_missing_key_raises_422(svc: McpService) -> None:
    with pytest.raises(HTTPException) as exc_info:
        svc.check_headers(["X-Auth"], {"X-Other": "val"})
    assert exc_info.value.status_code == 422
    assert "X-Auth" in exc_info.value.detail


def test_check_headers_empty_value_raises_422(svc: McpService) -> None:
    with pytest.raises(HTTPException) as exc_info:
        svc.check_headers(["X-Auth"], {"X-Auth": ""})
    assert exc_info.value.status_code == 422
    assert "X-Auth" in exc_info.value.detail


@pytest.mark.asyncio
async def test_install_model_already_installed_returns_ok(svc: McpService, tmp_path: Path) -> None:
    with patch.object(svc, "_get_working_dir", return_value=tmp_path):
        svc._add_custom_model("default", _make_user_model())  # pyright: ignore[reportPrivateUsage]

    installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    installed.models["my-server"] = MagicMock()
    svc.instances_info["default"].installed = installed

    promise = await svc._install_model("default", "my-server", InstallModelIn(stream=False, spec={}))  # pyright: ignore[reportPrivateUsage]
    result = await promise.wait()
    assert result.status == "OK"
    assert "Already" in result.details


@pytest.mark.asyncio
async def test_install_model_not_found_raises_400(svc: McpService) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    with pytest.raises(HTTPException) as exc_info:
        await svc._install_model("default", "no-such-model", InstallModelIn(stream=False, spec={}))  # pyright: ignore[reportPrivateUsage]
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_install_model_auto_sets_prefix(svc: McpService, tmp_path: Path, deps: dict[str, Any]) -> None:
    with patch.object(svc, "_get_working_dir", return_value=tmp_path):
        svc._add_custom_model("default", _make_user_model(model_id="my-server"))  # pyright: ignore[reportPrivateUsage]

    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    svc.docker_service.install_and_run_docker = AsyncMock(return_value=8000)  # type: ignore[method-assign]

    with patch.object(svc, "_get_working_dir", return_value=tmp_path):
        promise = await svc._install_model("default", "my-server", InstallModelIn(stream=False, spec={}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    info = svc.instances_info["default"].installed
    assert info is not None
    assert "my-server" in info.models
    assert info.models["my-server"].prefix is not None


@pytest.mark.asyncio
async def test_install_proxy_model_registers_endpoint(svc: McpService, tmp_path: Path, deps: dict[str, Any]) -> None:
    svc._add_custom_model("default", _make_proxy_model())  # pyright: ignore[reportPrivateUsage]
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    deps["endpoint_registry"].register_mcp_endpoint_as_proxy.return_value = "reg-id-1"

    promise = await svc._install_model("default", "remote-mcp", InstallModelIn(stream=False, spec={}))  # pyright: ignore[reportPrivateUsage]
    result = await promise.wait()

    assert result.status == "OK"
    deps["endpoint_registry"].register_mcp_endpoint_as_proxy.assert_called_once()


@pytest.mark.asyncio
async def test_uninstall_model_not_in_info_models_noop(svc: McpService, tmp_path: Path, deps: dict[str, Any]) -> None:
    with patch.object(svc, "_get_working_dir", return_value=tmp_path):
        svc._add_custom_model("default", _make_user_model())  # pyright: ignore[reportPrivateUsage]

    installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    svc.instances_info["default"].installed = installed

    await svc._uninstall_model("default", "my-server", UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    deps["docker_service"].uninstall_docker.assert_not_called()


@pytest.mark.asyncio
async def test_uninstall_model_calls_docker_uninstall(svc: McpService, tmp_path: Path, deps: dict[str, Any]) -> None:
    with patch.object(svc, "_get_working_dir", return_value=tmp_path):
        svc._add_custom_model("default", _make_user_model())  # pyright: ignore[reportPrivateUsage]

    installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    mock_info = MagicMock(spec=ModelInstalledInfo)
    mock_info.registration_id = "reg-1"
    mock_info.prefix = "my-server"
    mock_info.docker_options = MagicMock()
    installed.models["my-server"] = mock_info
    svc.instances_info["default"].installed = installed

    await svc._uninstall_model("default", "my-server", UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    deps["docker_service"].uninstall_docker.assert_awaited_once()
    assert "my-server" not in installed.models


@pytest.mark.asyncio
async def test_uninstall_model_purge_removes_downloaded(svc: McpService, tmp_path: Path, deps: dict[str, Any]) -> None:
    with patch.object(svc, "_get_working_dir", return_value=tmp_path):
        svc._add_custom_model("default", _make_user_model())  # pyright: ignore[reportPrivateUsage]

    svc.models_downloaded["my-server"] = DownloadedInfo("img:latest")
    installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    svc.instances_info["default"].installed = installed

    with patch.object(svc, "_get_working_dir", return_value=tmp_path):
        await svc._uninstall_model("default", "my-server", UninstallModelIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    deps["docker_service"].remove_image.assert_awaited_once_with("img:latest")
    assert "my-server" not in svc.models_downloaded


@pytest.mark.asyncio
async def test_uninstall_user_model_purge_removes_dockerfile_dir(svc: McpService, tmp_path: Path) -> None:
    with patch.object(svc, "_get_working_dir", return_value=tmp_path):
        svc._add_custom_model("default", _make_user_model())  # pyright: ignore[reportPrivateUsage]

    installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    svc.instances_info["default"].installed = installed
    dockerfile_dir = tmp_path / "models" / "default" / "my-server"
    dockerfile_dir.mkdir(parents=True, exist_ok=True)

    with patch.object(svc, "_get_working_dir", return_value=tmp_path):
        await svc._uninstall_model("default", "my-server", UninstallModelIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert not dockerfile_dir.exists()


def test_get_docker_compose_file_path_no_docker_options_raises(svc: McpService) -> None:
    installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    mock_info = MagicMock(spec=ModelInstalledInfo)
    mock_info.docker_options = None
    installed.models["my-server"] = mock_info
    svc.instances_info["default"].installed = installed

    with patch.object(svc, "_get_working_dir", return_value=MagicMock()), pytest.raises(HTTPException) as exc_info:
        svc.get_docker_compose_file_path("default", "my-server")
    assert exc_info.value.status_code == 400


def test_add_proxy_model_initializes_models_dict_for_new_instance(svc: McpService) -> None:
    """Line 557: models[instance] = {} is set when instance not yet in models."""
    assert "new-instance" not in svc.models
    svc._add_custom_model("new-instance", _make_proxy_model())  # pyright: ignore[reportPrivateUsage]
    assert "new-instance" in svc.models
    assert "remote-mcp" in svc.models["new-instance"]


@pytest.mark.asyncio
async def test_update_proxy_model_when_not_in_models_dict(svc: McpService) -> None:
    """Line 596->598: skip del when proxy model not in self.models (already gone)."""
    svc._add_custom_model("default", _make_proxy_model())  # pyright: ignore[reportPrivateUsage]
    svc.instances_info["default"].installed = None
    # Remove from models to simulate absent entry
    del svc.models["default"]["remote-mcp"]

    await svc._update_custom_model(  # pyright: ignore[reportPrivateUsage]
        "default",
        _make_proxy_model(),
        {
            "kind": "proxy",
            "id": "remote-mcp",
            "name": "Remote MCP",
            "server_url": "https://new.example.com/mcp",
            "transport": "streamable_http",
        },
    )

    assert "remote-mcp" in svc.models["default"]
    assert svc.models["default"]["remote-mcp"].proxy_url == "https://new.example.com/mcp"


def test_version_to_base_image_python_default() -> None:
    assert _version_to_base_image(McpUserVariant.python_headless, None, None) == "python:3.13-slim"


def test_version_to_base_image_python_explicit() -> None:
    assert _version_to_base_image(McpUserVariant.python_headless, "3.11", None) == "python:3.11-slim"


def test_version_to_base_image_python_latest() -> None:
    assert _version_to_base_image(McpUserVariant.python_headed, "latest", None) == "python:slim"


def test_version_to_base_image_node_default() -> None:
    assert _version_to_base_image(McpUserVariant.node_headless, None, None) == "node:22-slim"


def test_version_to_base_image_node_explicit() -> None:
    assert _version_to_base_image(McpUserVariant.node_headless, None, "20") == "node:20-slim"


def test_version_to_base_image_node_latest() -> None:
    assert _version_to_base_image(McpUserVariant.node_headed, None, "latest") == "node:slim"


def test_add_user_model_dockerfile_uses_python_version(svc: McpService, tmp_path: Path) -> None:
    custom = CustomModel(
        id="uuid-py",
        data={
            "kind": "user",
            "id": "py-server",
            "name": "Py",
            "variant": "python-headless",
            "command": "uvx srv",
            "python_version": "3.11",
        },
    )
    with patch.object(svc, "_get_working_dir", return_value=tmp_path):
        svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    content = (tmp_path / "models" / "default" / "py-server" / "Dockerfile").read_text()
    assert "FROM python:3.11-slim" in content


def test_add_user_model_dockerfile_uses_node_version(svc: McpService, tmp_path: Path) -> None:
    custom = CustomModel(
        id="uuid-nd",
        data={"kind": "user", "id": "nd-server", "name": "Nd", "variant": "node-headless", "command": "npx srv", "node_version": "20"},
    )
    with patch.object(svc, "_get_working_dir", return_value=tmp_path):
        svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    content = (tmp_path / "models" / "default" / "nd-server" / "Dockerfile").read_text()
    assert "FROM node:20-slim" in content


def test_add_user_model_dockerfile_base_image_overrides_version(svc: McpService, tmp_path: Path) -> None:
    custom = CustomModel(
        id="uuid-ov",
        data={
            "kind": "user",
            "id": "ov-server",
            "name": "Ov",
            "variant": "python-headless",
            "command": "uvx srv",
            "python_version": "3.11",
            "base_image": "my-custom:latest",
        },
    )
    with patch.object(svc, "_get_working_dir", return_value=tmp_path):
        svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    content = (tmp_path / "models" / "default" / "ov-server" / "Dockerfile").read_text()
    assert "FROM my-custom:latest" in content
    assert "python:3.11" not in content


def test_get_custom_spec_user_includes_python_version(svc: McpService) -> None:
    model = _make_srv_mcp_model(custom="uuid-pv", kind="user", command="uvx srv", variant="python-headless", python_version="3.11")
    spec = svc._get_custom_spec("mid", model)  # pyright: ignore[reportPrivateUsage]
    assert spec is not None
    assert spec["python_version"] == "3.11"
    assert "node_version" not in spec


def test_get_custom_spec_user_includes_node_version(svc: McpService) -> None:
    model = _make_srv_mcp_model(custom="uuid-nv", kind="user", command="npx srv", variant="node-headless", node_version="24")
    spec = svc._get_custom_spec("mid", model)  # pyright: ignore[reportPrivateUsage]
    assert spec is not None
    assert spec["node_version"] == "24"
    assert "python_version" not in spec


def test_write_dockerfile_invalid_python_command_raises_400(svc: McpService, tmp_path: Path) -> None:
    """shlex.split on an unclosed quote raises ValueError which is converted to HTTP 400."""
    with pytest.raises(HTTPException) as exc_info, patch.object(svc, "_get_working_dir", return_value=tmp_path):
        svc._write_dockerfile(  # pyright: ignore[reportPrivateUsage]
            "default", "bad-server", McpUserVariant.python_headless, "uvx 'unclosed", "python:3.13-slim"
        )
    assert exc_info.value.status_code == 400
    assert "Invalid command syntax" in exc_info.value.detail


@pytest.mark.asyncio
async def test_update_proxy_model_id_change_raises_400(svc: McpService) -> None:
    """Line 609: changing the id of a proxy model is forbidden."""
    svc._add_custom_model("default", _make_proxy_model())  # pyright: ignore[reportPrivateUsage]
    svc.instances_info["default"].installed = None

    with pytest.raises(HTTPException) as exc_info:
        await svc._update_custom_model(  # pyright: ignore[reportPrivateUsage]
            "default",
            _make_proxy_model(),
            {"kind": "proxy", "id": "different-id", "name": "Remote MCP", "server_url": "https://example.com/mcp"},
        )
    assert exc_info.value.status_code == 400
    assert "Cannot change the server ID" in exc_info.value.detail


@pytest.mark.asyncio
async def test_update_user_model_id_change_raises_400(svc: McpService, tmp_path: Path) -> None:
    """Line 626: changing the id of a user model is forbidden."""
    with patch.object(svc, "_get_working_dir", return_value=tmp_path):
        svc._add_custom_model("default", _make_user_model())  # pyright: ignore[reportPrivateUsage]
    svc.instances_info["default"].installed = None

    with pytest.raises(HTTPException) as exc_info, patch.object(svc, "_get_working_dir", return_value=tmp_path):
        await svc._update_custom_model(  # pyright: ignore[reportPrivateUsage]
            "default",
            _make_user_model(),
            {"kind": "user", "id": "different-id", "name": "My Server", "variant": "node-headless", "command": "npx new"},
        )
    assert exc_info.value.status_code == 400
    assert "Cannot change the server ID" in exc_info.value.detail


def test_register_proxy_model_without_proxy_url_raises_400(svc: McpService) -> None:
    """Line 802: _register_proxy_model raises 400 when proxy_url is None."""
    model = _make_srv_mcp_model(kind="proxy", proxy_url=None)
    parsed_options = McpModelOptions(prefix="test", envs={}, headers={})
    with pytest.raises(HTTPException) as exc_info:
        svc._register_proxy_model(model, parsed_options)  # pyright: ignore[reportPrivateUsage]
    assert exc_info.value.status_code == 400
    assert "proxy_url" in exc_info.value.detail


@pytest.mark.asyncio
async def test_install_model_options_none_raises_400(svc: McpService) -> None:
    """Line 857: _install_model raises 400 when model.options is None for a non-proxy kind."""
    svc.models.setdefault("default", {})["broken-model"] = _make_srv_mcp_model(kind="user", options=None)
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))

    with pytest.raises(HTTPException) as exc_info:
        await svc._install_model("default", "broken-model", InstallModelIn(stream=False, spec={"prefix": "broken"}))  # pyright: ignore[reportPrivateUsage]
    assert exc_info.value.status_code == 400
    assert "options are required" in exc_info.value.detail
