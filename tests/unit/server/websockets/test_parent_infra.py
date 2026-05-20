# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import AsyncMock, MagicMock, call

import pytest

from server.models.mesh import CheckMeshConnection
from server.utils.exceptions import ApiError
from server.websockets.models import InitRequest, UsageChangeRequest
from server.websockets.parent_infra import ParentInfra


def make_parent_infra(url: str = "ws://mesh") -> ParentInfra:
    config = MagicMock()
    config.connect_to_mesh_url = url
    config.connect_to_mesh_key.get_secret_value.return_value = "mesh-secret"
    config.name = "test-infra"
    config.infra_url = "http://infra"
    config.infra_api_key.get_secret_value.return_value = "api-key"

    task_manager = MagicMock()
    infra = ParentInfra(config=config, task_manager=task_manager)

    # Replace collaborators with mocks so tests don't need real JsonRpc/InfraClient
    infra.infra_client = MagicMock()
    infra.infra_client.init = AsyncMock(return_value=None)
    infra.infra_client.usage_change = MagicMock()
    infra.infra_client.update_models = MagicMock()

    infra.endpoint_registry = MagicMock()
    infra.endpoint_registry.list_models.return_value = []  # pyright: ignore[reportAttributeAccessIssue]

    return infra


def test_init_enabled_when_url_is_set():
    infra = make_parent_infra(url="ws://mesh")
    assert infra.enabled is True
    assert infra.uri == "ws://mesh/ws"


def test_init_disabled_when_url_is_empty():
    infra = make_parent_infra(url="")
    assert infra.enabled is False


@pytest.mark.asyncio
async def test_send_raises_when_disabled():
    infra = make_parent_infra(url="")
    with pytest.raises(RuntimeError):
        await infra._send("x")  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_send_raises_when_enabled_but_ws_is_none():
    infra = make_parent_infra()
    assert infra.ws is None
    with pytest.raises(RuntimeError):
        await infra._send("x")  # pyright: ignore[reportPrivateUsage]


def test_on_message_calls_client_resolve():
    infra = make_parent_infra()
    infra.client = MagicMock()

    infra.on_message("payload")

    assert infra.client.resolve.call_count == 1
    assert infra.client.resolve.call_args == call("payload")


def test_on_disconnect_calls_client_clear():
    infra = make_parent_infra()
    infra.client = MagicMock()

    infra.on_disconnect()

    assert infra.client.clear.call_count == 1


@pytest.mark.asyncio
async def test_before_loop_returns_true_when_enabled():
    infra = make_parent_infra(url="ws://mesh")

    assert await infra.before_loop() is True


@pytest.mark.asyncio
async def test_before_loop_returns_false_when_disabled():
    infra = make_parent_infra(url="")

    assert await infra.before_loop() is False


@pytest.mark.asyncio
async def test_on_start_calls_infra_client_init():
    infra = make_parent_infra()
    infra.endpoint_registry.list_models.return_value = []  # pyright: ignore[reportAttributeAccessIssue]

    await infra.on_start()

    infra.infra_client.init.assert_awaited_once()  # pyright: ignore[reportAttributeAccessIssue]
    call_args = infra.infra_client.init.call_args[0][0]  # pyright: ignore[reportAttributeAccessIssue]
    assert isinstance(call_args, InitRequest)
    assert call_args.auth == "mesh-secret"
    assert call_args.name == "test-infra"
    assert call_args.url == "http://infra"
    assert call_args.api_key == "api-key"
    assert call_args.models == []


@pytest.mark.asyncio
async def test_on_start_stops_process_loop_on_invalid_api_key():
    infra = make_parent_infra()
    infra.infra_client.init = AsyncMock(side_effect=ApiError("Invalid api key", 2))

    with pytest.raises(ApiError):
        await infra.on_start()

    assert infra.process_loop is False


@pytest.mark.asyncio
async def test_on_start_reraises_other_api_errors_without_stopping_loop():
    infra = make_parent_infra()
    infra.infra_client.init = AsyncMock(side_effect=ApiError("Other error", 5))

    with pytest.raises(ApiError):
        await infra.on_start()

    assert infra.process_loop is True


def test_send_usage_noop_when_disabled():
    infra = make_parent_infra(url="")
    usage = UsageChangeRequest(id="r1", usage=1)

    infra.send_usage(usage)

    assert infra.task_manager.add_task_safe.call_count == 0  # pyright: ignore[reportAttributeAccessIssue]


def test_send_usage_noop_when_ws_is_none():
    infra = make_parent_infra()
    assert infra.ws is None
    usage = UsageChangeRequest(id="r1", usage=1)

    infra.send_usage(usage)

    assert infra.task_manager.add_task_safe.call_count == 0  # pyright: ignore[reportAttributeAccessIssue]


def test_send_usage_schedules_task_when_connected():
    infra = make_parent_infra()
    infra.ws = (MagicMock(), MagicMock())
    usage = UsageChangeRequest(id="r1", usage=3)

    infra.send_usage(usage)

    assert infra.task_manager.add_task_safe.call_count == 1  # pyright: ignore[reportAttributeAccessIssue]


def test_send_models_list_noop_when_disabled():
    infra = make_parent_infra(url="")

    infra.send_models_list()

    assert infra.task_manager.add_task_safe.call_count == 0  # pyright: ignore[reportAttributeAccessIssue]


def test_send_models_list_noop_when_ws_is_none():
    infra = make_parent_infra()
    assert infra.ws is None

    infra.send_models_list()

    assert infra.task_manager.add_task_safe.call_count == 0  # pyright: ignore[reportAttributeAccessIssue]


def test_send_models_list_schedules_task_when_connected():
    infra = make_parent_infra()
    infra.ws = (MagicMock(), MagicMock())

    infra.send_models_list()

    assert infra.task_manager.add_task_safe.call_count == 1  # pyright: ignore[reportAttributeAccessIssue]


def test_check_subinfra_connection_returns_true_when_both_match():
    infra = make_parent_infra()
    infra.one_time_key = MagicMock()
    infra.one_time_key.check.return_value = True

    model = CheckMeshConnection(infra_api_key="api-key", connection_verifier="ck")

    assert infra.check_subinfra_connection(model) is True


def test_check_subinfra_connection_returns_false_when_key_invalid():
    infra = make_parent_infra()
    infra.one_time_key = MagicMock()
    infra.one_time_key.check.return_value = False

    model = CheckMeshConnection(infra_api_key="api-key", connection_verifier="bad")

    assert infra.check_subinfra_connection(model) is False


def test_check_subinfra_connection_returns_false_when_api_key_mismatch():
    infra = make_parent_infra()
    infra.one_time_key = MagicMock()
    infra.one_time_key.check.return_value = True

    model = CheckMeshConnection(infra_api_key="wrong-key", connection_verifier="ck")

    assert infra.check_subinfra_connection(model) is False


@pytest.mark.asyncio
async def test_send_delegates_when_enabled_and_ws_set():
    infra = make_parent_infra()
    mock_queue = MagicMock()
    mock_queue.put_nowait = MagicMock()
    infra.ws = (MagicMock(), mock_queue)

    await infra._send("hello")  # pyright: ignore[reportPrivateUsage]

    assert mock_queue.put_nowait.call_count == 1
    assert mock_queue.put_nowait.call_args == call("hello")
