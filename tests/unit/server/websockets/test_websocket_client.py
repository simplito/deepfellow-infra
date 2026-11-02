# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

import asyncio
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

import server.websockets.websocket_client as _wc_module
from server.utils.sentry import FINGERPRINT_ATTR, REPORTED_ATTR
from server.websockets.websocket_client import WebSocketClient


@pytest.mark.asyncio
async def test_connect_passes_ping_parameters() -> None:
    """websockets.connect is called with ping_interval=5 and ping_timeout=10."""
    client = WebSocketClient("ws://localhost:9999")

    connect_cm = MagicMock()
    connect_cm.__aenter__ = AsyncMock(side_effect=RuntimeError("stop"))
    connect_cm.__aexit__ = AsyncMock(return_value=False)
    connect_mock = MagicMock(return_value=connect_cm)
    mock_websockets = MagicMock()
    mock_websockets.connect = connect_mock

    def stop_on_disconnect() -> None:
        client.process_loop = False

    client.on_disconnect = stop_on_disconnect  # type: ignore[method-assign]

    with (
        patch.object(_wc_module, "websockets", mock_websockets),
        patch("server.websockets.websocket_client.asyncio.sleep", AsyncMock()),
    ):
        await client.loop()

    connect_mock.assert_called_once_with("ws://localhost:9999", ping_interval=5, ping_timeout=10)


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
async def test_stop_without_connection_just_flags_loop():
    client = make_client()

    await client.stop()

    assert client.process_loop is False


@pytest.mark.asyncio
async def test_stop_closes_active_connection():
    client = make_client()
    connection = AsyncMock()
    client.ws = (connection, asyncio.Queue())

    await client.stop()

    assert client.process_loop is False
    connection.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_stop_suppresses_close_exception():
    client = make_client()
    connection = AsyncMock()
    connection.close.side_effect = Exception("boom")
    client.ws = (connection, asyncio.Queue())

    await client.stop()  # should not raise

    assert client.process_loop is False


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
async def test_loop_closes_and_exits_promptly_when_stopped_during_connect():
    """A stop() racing an in-flight connect() must not leave the client connected indefinitely.

    While `websockets.connect()` is still resolving, `self.ws` is still `None`, so a concurrent
    `stop()` call has nothing to close and only flips `process_loop`. Once the connect succeeds,
    `loop()` must notice that flag before calling `on_start()`/awaiting `receive_task` — otherwise
    the "stopped" client would stay connected until it happens to disconnect on its own.
    """
    client = make_client()
    on_start_calls: list[bool] = []
    client.on_start = AsyncMock(side_effect=lambda: on_start_calls.append(True))  # type: ignore[method-assign]

    mock_ws = make_mock_ws()
    mock_connect = make_connect_patch(mock_ws)

    async def stop_mid_connect() -> MagicMock:
        # Simulate stop() running while this connect() was in flight and self.ws was still None.
        client.process_loop = False
        return mock_ws

    mock_connect.return_value.__aenter__ = AsyncMock(side_effect=stop_mid_connect)
    mock_ws_module = MagicMock()
    mock_ws_module.connect = mock_connect

    with patch.object(_wc_module, "websockets", mock_ws_module):
        await client.loop()

    assert on_start_calls == [], "on_start must not run once stop() has been observed"
    mock_ws.close.assert_awaited_once()
    assert client.ws is None


@pytest.mark.asyncio
async def test_loop_calls_on_disconnect_when_stopped_during_connect():
    """The stop-during-connect path is still a teardown — subclasses must get on_disconnect() too.

    Every other exit from loop() (normal disconnect, connect exception) reaches on_disconnect();
    this path shouldn't be an undocumented exception to that contract.
    """
    client = make_client()
    on_disconnect_calls: list[bool] = []
    client.on_disconnect = lambda: on_disconnect_calls.append(True)  # type: ignore[method-assign]

    mock_ws = make_mock_ws()
    mock_connect = make_connect_patch(mock_ws)

    async def stop_mid_connect() -> MagicMock:
        client.process_loop = False
        return mock_ws

    mock_connect.return_value.__aenter__ = AsyncMock(side_effect=stop_mid_connect)
    mock_ws_module = MagicMock()
    mock_ws_module.connect = mock_connect

    with patch.object(_wc_module, "websockets", mock_ws_module):
        await client.loop()

    assert on_disconnect_calls, "on_disconnect should have been called"


@pytest.mark.asyncio
async def test_loop_cancels_hanging_receive_task_when_stopped_during_connect():
    """The receive_task cancellation must be real, not just a no-op await.

    `_receive_task` here hangs forever (waiting on an `Event` nobody sets), so if `loop()` merely
    awaited it without cancelling first, `loop()` itself would never return. `asyncio.wait(...,
    timeout=...)` (unlike `wait_for`) never touches the task's cancellation state itself, so it
    can't be fooled by `loop()`'s own `suppress(asyncio.CancelledError)` the way `wait_for` can —
    it only reports whether the task actually finished in time.
    """
    client = make_client()
    never_set = asyncio.Event()

    async def hanging_receive_task(_ws: object) -> None:
        await never_set.wait()

    client._receive_task = hanging_receive_task  # type: ignore[method-assign]

    mock_ws = make_mock_ws()
    mock_connect = make_connect_patch(mock_ws)

    async def stop_mid_connect() -> MagicMock:
        client.process_loop = False
        return mock_ws

    mock_connect.return_value.__aenter__ = AsyncMock(side_effect=stop_mid_connect)
    mock_ws_module = MagicMock()
    mock_ws_module.connect = mock_connect

    with patch.object(_wc_module, "websockets", mock_ws_module):
        loop_task = asyncio.create_task(client.loop())
        done, pending = await asyncio.wait([loop_task], timeout=1.0)

    if pending:
        loop_task.cancel()
        pytest.fail("loop() did not return — the hanging receive_task was never cancelled")

    assert loop_task in done
    mock_ws.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_loop_logs_and_continues_when_close_fails_during_disconnect(caplog: pytest.LogCaptureFixture):
    client = make_client()

    async def fake_on_start() -> None:
        client.process_loop = False

    client.on_start = fake_on_start  # type: ignore[method-assign]
    mock_ws = make_mock_ws()
    mock_ws.close = AsyncMock(side_effect=RuntimeError("close failed"))
    mock_connect = make_connect_patch(mock_ws)
    mock_ws_module = MagicMock()
    mock_ws_module.connect = mock_connect

    with (
        patch.object(_wc_module, "websockets", mock_ws_module),
        patch("asyncio.sleep", new=AsyncMock()),
        caplog.at_level("ERROR", logger="uvicorn"),
    ):
        await client.loop()  # should not raise

    assert "Error closing websocket connection during disconnect" in caplog.text


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
async def test_loop_marks_only_the_first_failure_of_an_outage_for_sentry(caplog: pytest.LogCaptureFixture):
    client = make_client()
    client.process_loop = True
    mock_ws_module = MagicMock()
    mock_ws_module.connect.side_effect = RuntimeError("cannot connect")
    sleeps: list[float] = []

    async def stop_after_three(delay: float) -> None:
        sleeps.append(delay)
        if len(sleeps) == 3:
            client.process_loop = False

    with patch.object(_wc_module, "websockets", mock_ws_module), patch("asyncio.sleep", new=stop_after_three):
        await client.loop()

    records = [record for record in caplog.records if record.getMessage() == "WS client disconnected"]
    assert [hasattr(record, FINGERPRINT_ATTR) for record in records] == [True, False, False]
    assert [getattr(record, REPORTED_ATTR, False) for record in records] == [False, True, True]


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
