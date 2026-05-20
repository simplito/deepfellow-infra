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
from fastapi import WebSocketDisconnect

from server.websockets.websocket_server import (
    WebSockerContextImpl,
    WebSocketContext,
    WebSocketServer,
)


class ConcreteServer(WebSocketServer[dict[str, Any]]):
    def __init__(self) -> None:
        super().__init__()
        self.processed: list[tuple[str, object]] = []
        self.disconnected: list[object] = []

    def create_bag(self) -> dict[str, Any]:
        return {}

    async def process_message(self, msg: str, context: WebSocketContext[dict[str, Any]]) -> None:
        self.processed.append((msg, context))

    def handle_disconnect(self, context: WebSocketContext[dict[str, Any]]) -> None:
        self.disconnected.append(context)


def make_mock_websocket(messages: list[str] | None = None) -> MagicMock:
    """FastAPI WebSocket mock that accepts, yields messages, then disconnects."""
    ws = MagicMock()
    ws.accept = AsyncMock()
    side_effects: list[Any] = [*list(messages or []), WebSocketDisconnect()]
    ws.receive_text = AsyncMock(side_effect=side_effects)
    ws.send_text = AsyncMock()
    return ws


def test_websocket_context_stores_data():
    ctx: WebSocketContext[int] = WebSocketContext(42)  # type: ignore[abstract]

    assert ctx.data == 42


@pytest.mark.asyncio
async def test_context_impl_send_enqueues_message():
    mock_ws = MagicMock()
    mock_ws.send_text = AsyncMock()
    ctx = WebSockerContextImpl(mock_ws, None)

    ctx.send("hello")
    await ctx.close()  # drains queue, lets sender task finish

    assert mock_ws.send_text.await_count == 1
    assert mock_ws.send_text.await_args == call("hello")


@pytest.mark.asyncio
async def test_context_impl_close_terminates_send_task():
    mock_ws = MagicMock()
    mock_ws.send_text = AsyncMock()
    ctx = WebSockerContextImpl(mock_ws, None)

    await ctx.close()

    assert ctx.send_task.done()


@pytest.mark.asyncio
async def test_context_impl_sender_calls_send_text():
    mock_ws = MagicMock()
    mock_ws.send_text = AsyncMock()
    ctx = WebSockerContextImpl(mock_ws, None)

    ctx.send("data1")
    ctx.send("data2")
    await ctx.close()

    assert mock_ws.send_text.await_count == 2
    mock_ws.send_text.assert_any_await("data1")
    mock_ws.send_text.assert_any_await("data2")


def test_websocket_server_init_has_empty_connections():
    server = ConcreteServer()

    assert isinstance(server.connections, set)
    assert len(server.connections) == 0


@pytest.mark.asyncio
async def test_connect_accepts_and_cleans_up_on_disconnect():
    server = ConcreteServer()
    mock_ws = make_mock_websocket(messages=[])

    await server.connect(mock_ws)

    mock_ws.accept.assert_awaited_once()
    assert len(server.connections) == 0


@pytest.mark.asyncio
async def test_connect_dispatches_messages_to_process_message():
    server = ConcreteServer()
    mock_ws = make_mock_websocket(messages=["hello"])

    await server.connect(mock_ws)

    assert len(server.processed) == 1
    assert server.processed[0][0] == "hello"


@pytest.mark.asyncio
async def test_connect_calls_handle_disconnect_on_websocket_disconnect():
    server = ConcreteServer()
    mock_ws = make_mock_websocket(messages=[])

    await server.connect(mock_ws)

    assert len(server.disconnected) == 1


@pytest.mark.asyncio
async def test_process_message_safe_swallows_exceptions():
    server = ConcreteServer()
    ctx = MagicMock()
    ctx.data = {}

    async def bad_process(msg: str, context: object) -> None:
        raise RuntimeError("boom")

    server.process_message = bad_process  # type: ignore[method-assign]

    # Should not raise
    await server._process_message_safe("msg", ctx)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_connect_calls_handle_disconnect_on_generic_exception():
    server = ConcreteServer()
    ws = MagicMock()
    ws.accept = AsyncMock()
    ws.receive_text = AsyncMock(side_effect=RuntimeError("network error"))

    await server.connect(ws)

    assert len(server.disconnected) == 1


@pytest.mark.asyncio
async def test_sender_swallows_send_text_exception():
    mock_ws = MagicMock()
    mock_ws.send_text = AsyncMock(side_effect=RuntimeError("send failed"))
    ctx = WebSockerContextImpl(mock_ws, None)

    ctx.send("hello")
    # close() puts None which stops the sender loop after the error is swallowed
    await ctx.close()

    mock_ws.send_text.assert_awaited()
