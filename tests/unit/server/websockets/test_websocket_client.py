# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

import server.websockets.websocket_client as _wc_module
from server.websockets.websocket_client import WebSocketClient


def make_client(uri: str = "ws://test") -> WebSocketClient:
    return WebSocketClient(uri)


def make_mock_ws() -> MagicMock:
    """Websocket mock that acts as an empty async iterator."""
    ws = MagicMock()
    ws.__aiter__ = MagicMock(return_value=ws)
    ws.__anext__ = AsyncMock(side_effect=StopAsyncIteration())
    ws.close = AsyncMock()
    ws.send = AsyncMock()
    return ws


def make_connect_patch(mock_ws: MagicMock) -> MagicMock:
    """Return a mock for websockets.connect that yields mock_ws as async CM."""
    mock_connect = MagicMock()
    mock_connect.return_value.__aenter__ = AsyncMock(return_value=mock_ws)
    mock_connect.return_value.__aexit__ = AsyncMock(return_value=False)
    return mock_connect


def test_init_stores_uri_and_defaults():
    client = WebSocketClient("ws://host")

    assert client.uri == "ws://host"
    assert client.process_loop is True
    assert client.ws is None


def test_send_raises_when_not_connected():
    client = make_client()

    with pytest.raises(RuntimeError):
        client.send("msg")


def test_send_enqueues_message_when_connected():
    client = make_client()

    queue: asyncio.Queue[str | None] = asyncio.Queue()
    client.ws = (MagicMock(), queue)
    client.send("hello")

    assert queue.get_nowait() == "hello"


@pytest.mark.asyncio
async def test_run_skips_loop_when_before_loop_false():
    client = make_client()
    client.before_loop = AsyncMock(return_value=False)
    client.loop = AsyncMock()
    client.after_loop = AsyncMock()

    await client.run()

    assert client.loop.call_count == 0
    assert client.after_loop.call_count == 0


@pytest.mark.asyncio
async def test_run_calls_loop_and_after_loop():
    client = make_client()
    client.before_loop = AsyncMock(return_value=True)
    client.loop = AsyncMock()
    client.after_loop = AsyncMock()

    await client.run()

    assert client.loop.call_count == 1
    assert client.after_loop.call_count == 1


@pytest.mark.asyncio
async def test_loop_exits_without_connecting_when_process_loop_false():
    client = make_client()
    client.process_loop = False
    mock_ws_module = MagicMock()

    with patch.object(_wc_module, "websockets", mock_ws_module):
        await client.loop()

    assert mock_ws_module.connect.call_count == 0


@pytest.mark.asyncio
async def test_loop_single_connect_disconnect_cycle():
    client = make_client()
    on_start_calls: list[bool] = []
    on_disconnect_calls: list[bool] = []

    async def fake_on_start() -> None:
        on_start_calls.append(True)
        client.process_loop = False

    client.on_start = fake_on_start  # type: ignore[method-assign]
    client.on_disconnect = lambda: on_disconnect_calls.append(True)  # type: ignore[method-assign]
    mock_ws = make_mock_ws()
    mock_connect = make_connect_patch(mock_ws)
    mock_ws_module = MagicMock()
    mock_ws_module.connect = mock_connect

    with patch.object(_wc_module, "websockets", mock_ws_module), patch("asyncio.sleep", new=AsyncMock()):
        await client.loop()

    assert on_start_calls, "on_start should have been called"
    assert on_disconnect_calls, "on_disconnect should have been called"


@pytest.mark.asyncio
async def test_sender_sends_messages_and_stops_on_none():
    client = make_client()
    mock_ws = MagicMock()
    mock_ws.send = AsyncMock()
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    await queue.put("a")
    await queue.put("b")
    await queue.put(None)

    await client._sender(mock_ws, queue)  # pyright: ignore[reportPrivateUsage]

    assert mock_ws.send.await_count == 2
    mock_ws.send.assert_any_await("a")
    mock_ws.send.assert_any_await("b")


@pytest.mark.asyncio
async def test_receive_task_calls_on_message_for_each_incoming_message():
    client = make_client()
    received: list[str] = []
    client.on_message = lambda msg: received.append(msg)  # type: ignore[method-assign]

    async def async_messages():
        for m in ["msg1", "msg2"]:
            yield m

    await client._receive_task(async_messages())  # type: ignore[arg-type]

    assert received == ["msg1", "msg2"]


@pytest.mark.asyncio
async def test_default_before_loop_returns_true():
    client = make_client()

    result = await client.before_loop()

    assert result is True


@pytest.mark.asyncio
async def test_default_after_loop_returns_none():
    client = make_client()

    result = await client.after_loop()

    assert result is None


@pytest.mark.asyncio
async def test_loop_swallows_connect_exception():
    client = make_client()
    calls: list[str] = []
    client.on_disconnect = lambda: calls.append("disconnect")  # type: ignore[method-assign]
    client.process_loop = True
    mock_ws_module = MagicMock()
    mock_ws_module.connect.side_effect = RuntimeError("cannot connect")

    with patch.object(_wc_module, "websockets", mock_ws_module), patch("asyncio.sleep", new=AsyncMock()):
        # After one failed connect + sleep, stop the loop
        original_sleep = AsyncMock(side_effect=lambda _: setattr(client, "process_loop", False))  # pyright: ignore[reportUnknownLambdaType]

        with patch("asyncio.sleep", new=original_sleep):
            await client.loop()

    assert "disconnect" in calls


@pytest.mark.asyncio
async def test_sender_swallows_send_exception():
    client = make_client()
    mock_ws = MagicMock()
    mock_ws.send = AsyncMock(side_effect=RuntimeError("send error"))
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    await queue.put("msg")
    await queue.put(None)

    await client._sender(mock_ws, queue)  # pyright: ignore[reportPrivateUsage]

    mock_ws.send.assert_awaited()


@pytest.mark.asyncio
async def test_receive_task_swallows_on_message_exception():
    client = make_client()
    client.on_message = MagicMock(side_effect=RuntimeError("processing error"))  # type: ignore[method-assign]

    async def async_messages():
        yield "msg1"

    # Should not raise
    await client._receive_task(async_messages())  # type: ignore[arg-type]

    assert client.on_message.call_count == 1
    assert client.on_message.call_args == call("msg1")
