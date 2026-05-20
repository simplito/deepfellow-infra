# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import subprocess
from collections.abc import Generator
from contextlib import ExitStack
from unittest import mock
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

import pytest
from fastapi import FastAPI

from server.config import ConfigError
from server.lifecycle import check_subnet, lifespan
from server.utils.exceptions import AppStartError


@pytest.fixture(name="subnet")
def subnet_fixture() -> str:
    return "subnet"


def test_check_subnet_invalid_name() -> None:
    subnet = ";@#$%^"

    with pytest.raises(AppStartError, match="Invalid subnet name"):
        check_subnet(subnet)


@patch("server.lifecycle.subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 0))
def test_check_subnet_subprocess_timeout(mock_subprocess_run: Mock, subnet: str) -> None:
    with pytest.raises(
        subprocess.CalledProcessError,
        match=re.escape("Command '['docker', 'network', 'inspect', 'subnet']' returned non-zero exit status 124."),
    ):
        check_subnet(subnet)


@patch("server.lifecycle.subprocess.run")
def test_check_subnet_subprocess_other_error(mock_subprocess_run: Mock, subnet: str) -> None:
    mock_subprocess_run.side_effect = subprocess.CalledProcessError(999, ["docker", "network", "inspect", subnet], "output", "error msg")

    with pytest.raises(
        subprocess.CalledProcessError,
        match=re.escape("Command '['docker', 'network', 'inspect', 'subnet']' returned non-zero exit status 999."),
    ):
        check_subnet(subnet)


@patch("server.lifecycle.subprocess.run")
def test_check_subnet_subprocess_success(mock_subprocess_run: Mock, subnet: str) -> None:
    result = check_subnet(subnet)

    assert result is None
    assert mock_subprocess_run.call_count == 1
    assert mock_subprocess_run.call_args == mock.call(["docker", "network", "inspect", subnet], capture_output=True, check=True, timeout=10)


@patch("server.lifecycle.subprocess.run")
def test_check_subnet_network_not_found(mock_subprocess_run: Mock, subnet: str) -> None:
    mock_subprocess_run.side_effect = subprocess.CalledProcessError(1, ["docker", "network", "inspect", subnet])

    with pytest.raises(AppStartError, match="does not exist"):
        check_subnet(subnet)


SERVICE_CLASSES = [
    "ClaudeService",
    "CoquiService",
    "CustomService",
    "GoogleAIService",
    "LLamacppService",
    "McpService",
    "OllamaExternalService",
    "OllamaService",
    "OpenAIService",
    "RerankService",
    "SindriService",
    "SpeachesAIService",
    "StableDiffusionService",
    "VllmService",
]

_BASE_PATCHES = [
    "server.lifecycle.load_config",
    "server.lifecycle.Hardware",
    "server.lifecycle.MetricsRegistry",
    "server.lifecycle.TaskManager",
    "server.lifecycle.ServiceProvider",
    "server.lifecycle.ParentInfra",
    "server.lifecycle.ServicesManager",
    "server.lifecycle.ModelTester",
    "server.lifecycle.EndpointRegistry",
    "server.lifecycle.InfraWebsocketServer",
    "server.lifecycle.MetricsService",
    "server.lifecycle.ApplicationContext",
    "server.lifecycle.PortService",
    "server.lifecycle.create_docker_service",
    "server.lifecycle.ModelDownloader",
    *[f"server.lifecycle.{cls}" for cls in SERVICE_CLASSES],
]


def _make_config(*, docker_subnet: str = "", stop_on_shutdown: bool = False) -> MagicMock:
    cfg = MagicMock()
    cfg.docker_subnet = docker_subnet
    cfg.is_stop_containers_on_shutdown_enabled.return_value = stop_on_shutdown
    return cfg


def _apply_base_patches(active_patches: dict[str, Mock], *, config: MagicMock | None = None) -> MagicMock:
    cfg = config or _make_config()
    active_patches["server.lifecycle.load_config"].return_value = cfg

    hw = active_patches["server.lifecycle.Hardware"].return_value
    hw.init_async = AsyncMock()

    docker = active_patches["server.lifecycle.create_docker_service"]
    docker.return_value = AsyncMock()
    docker.side_effect = None

    ctx = active_patches["server.lifecycle.ApplicationContext"].return_value
    ctx.load_services = AsyncMock()

    task_mgr = active_patches["server.lifecycle.TaskManager"].return_value
    task_mgr.add_task = MagicMock()

    parent = active_patches["server.lifecycle.ParentInfra"].return_value
    parent.run = MagicMock(return_value=MagicMock())

    svc_mgr = active_patches["server.lifecycle.ServicesManager"].return_value
    svc_mgr.stop_all_services = AsyncMock()

    return cfg


@pytest.fixture
def app() -> FastAPI:
    return FastAPI()


@pytest.fixture
def base_mocks() -> Generator[dict[str, Mock]]:
    with ExitStack() as stack:
        yield {t: stack.enter_context(patch(t)) for t in _BASE_PATCHES}


@pytest.mark.asyncio
async def test_lifespan_calls_check_subnet_when_set(app: FastAPI, base_mocks: dict[str, Mock]) -> None:
    cfg = _make_config(docker_subnet="my-net")
    _apply_base_patches(base_mocks, config=cfg)

    with patch("server.lifecycle.check_subnet") as mock_check_subnet:
        async with lifespan(app):
            pass

        assert mock_check_subnet.call_count == 1
        assert mock_check_subnet.call_args == call("my-net")


@pytest.mark.asyncio
async def test_lifespan_skips_check_subnet_when_empty(app: FastAPI, base_mocks: dict[str, Mock]) -> None:
    _apply_base_patches(base_mocks)

    with patch("server.lifecycle.check_subnet") as mock_check_subnet:
        async with lifespan(app):
            pass

        assert mock_check_subnet.call_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raised_exception",
    [
        ConfigError("bad config"),
        RuntimeError("unexpected"),
    ],
    ids=["config_error", "generic_exception"],
)
async def test_lifespan_exit_on_exceptions(app: FastAPI, base_mocks: dict[str, Mock], raised_exception: Exception) -> None:
    base_mocks["server.lifecycle.load_config"].side_effect = raised_exception

    with patch("server.lifecycle.os._exit", side_effect=SystemExit(1)) as mock_exit:
        with pytest.raises(SystemExit):
            async with lifespan(app):
                pass

        assert mock_exit.call_count == 1
        assert mock_exit.call_args[0][0] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stop_on_shutdown", "expected_call_count"),
    [
        (True, 1),
        (False, 0),
    ],
)
async def test_lifespan_shutdown_behavior(
    app: FastAPI, base_mocks: dict[str, Mock], stop_on_shutdown: bool, expected_call_count: int
) -> None:
    cfg = _make_config(stop_on_shutdown=stop_on_shutdown)
    _apply_base_patches(base_mocks, config=cfg)
    svc_mgr = base_mocks["server.lifecycle.ServicesManager"].return_value

    async with lifespan(app):
        pass

    assert svc_mgr.stop_all_services.call_count == expected_call_count
