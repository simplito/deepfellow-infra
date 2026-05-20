# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for server/utils/json_rpc_client.py."""

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

from server.utils.exceptions import ApiError
from server.utils.json_rpc_client import JsonRpcClient  # pyright: ignore[reportPrivateUsage]


def make_response(id: int, result: object = None, error: object = None) -> str:
    obj: dict[str, object] = {"jsonrpc": "2.0", "id": id}
    if error is not None:
        obj["error"] = error
    else:
        obj["result"] = result
    return json.dumps(obj)


def make_client(send: AsyncMock | None = None, timeout: int = 30) -> JsonRpcClient:
    if send is None:
        send = AsyncMock()
    return JsonRpcClient(send=send, timeout=timeout)


@pytest.mark.asyncio
async def test_request_increments_id_on_each_request() -> None:
    send = AsyncMock()
    client = make_client(send)
    loop = asyncio.get_running_loop()

    fut1 = loop.create_future()
    fut2 = loop.create_future()

    async def do_request(result_fut: asyncio.Future[object]) -> None:
        task = asyncio.create_task(client.request("method", {}))
        await asyncio.sleep(0)
        result_fut.set_result(client.id)
        for p in client.pending.values():
            if not p.future.done():
                p.future.set_result("ok")
        await task

    await do_request(fut1)

    await do_request(fut2)

    assert await fut1 == 1
    assert await fut2 == 2


@pytest.mark.asyncio
async def test_request_sends_json_rpc_request() -> None:
    send = AsyncMock()
    client = make_client(send)

    async def resolve_immediately() -> None:
        await asyncio.sleep(0)
        client.resolve(make_response(1, result="ok"))

    asyncio.create_task(resolve_immediately())  # noqa: RUF006
    result = await client.request("test_method", {"key": "val"})
    assert result == "ok"

    sent_payload = json.loads(send.call_args[0][0])

    assert sent_payload["jsonrpc"] == "2.0"
    assert sent_payload["method"] == "test_method"
    assert sent_payload["params"] == {"key": "val"}


@pytest.mark.asyncio
async def test_request_returns_result_from_response() -> None:
    send = AsyncMock()
    client = make_client(send)

    async def resolve() -> None:
        await asyncio.sleep(0)
        client.resolve(make_response(1, result={"data": 42}))

    asyncio.create_task(resolve())  # noqa: RUF006

    result = await client.request("m", {})

    assert result == {"data": 42}


@pytest.mark.asyncio
async def test_request_raises_api_error_on_rpc_error() -> None:
    send = AsyncMock()
    client = make_client(send)

    async def resolve() -> None:
        await asyncio.sleep(0)
        client.resolve(make_response(1, error={"code": 404, "message": "Not found"}))

    asyncio.create_task(resolve())  # noqa: RUF006

    with pytest.raises(ApiError) as exc_info:
        await client.request("m", {})

    assert exc_info.value.code == 404
    assert exc_info.value.message == "Not found"


@pytest.mark.asyncio
async def test_request_raises_runtime_error_on_unprocessable_error() -> None:
    send = AsyncMock()
    client = make_client(send)

    async def resolve() -> None:
        await asyncio.sleep(0)
        client.resolve(make_response(1, error="not-a-dict"))

    asyncio.create_task(resolve())  # noqa: RUF006

    with pytest.raises(RuntimeError, match="Unprocessable error"):
        await client.request("m", {})


@pytest.mark.asyncio
async def test_request_raises_runtime_error_when_missing_result_and_error() -> None:
    send = AsyncMock()
    client = make_client(send)

    async def resolve() -> None:
        await asyncio.sleep(0)
        obj = {"jsonrpc": "2.0", "id": 1}
        client.resolve(json.dumps(obj))

    asyncio.create_task(resolve())  # noqa: RUF006

    with pytest.raises(RuntimeError, match="missing both"):
        await client.request("m", {})


def test_resolve_ignores_invalid_json() -> None:
    client = make_client()

    with patch("server.utils.json_rpc_client.logger"):
        client.resolve("not json {{{")


def test_resolve_ignores_non_rpc_structure() -> None:
    client = make_client()

    client.resolve(json.dumps({"foo": "bar"}))


def test_resolve_ignores_unknown_id() -> None:
    client = make_client()

    client.resolve(make_response(999, result="ok"))


def test_resolve_accepts_bytes() -> None:
    loop = asyncio.new_event_loop()
    try:
        client = make_client()
        fut = loop.create_future()
        client.pending[1] = type("P", (), {"future": fut, "timeout": loop.create_future()})()  # type: ignore[attr-defined]

        client.resolve(make_response(1, result="bytes-ok").encode())

        assert fut.result() == "bytes-ok"
    finally:
        loop.close()


@pytest.mark.asyncio
async def test_resolve_error_with_data_field() -> None:
    send = AsyncMock()
    client = make_client(send)

    async def resolve() -> None:
        await asyncio.sleep(0)

        client.resolve(make_response(1, error={"code": 500, "message": "err", "data": {"detail": "info"}}))

    asyncio.create_task(resolve())  # noqa: RUF006
    with pytest.raises(ApiError) as exc_info:
        await client.request("m", {})
    assert exc_info.value.data == {"detail": "info"}


@pytest.mark.asyncio
async def test_resolve_error_without_data_field() -> None:
    send = AsyncMock()
    client = make_client(send)

    async def resolve() -> None:
        await asyncio.sleep(0)

        client.resolve(make_response(1, error={"code": 500, "message": "err"}))

    asyncio.create_task(resolve())  # noqa: RUF006
    with pytest.raises(ApiError) as exc_info:
        await client.request("m", {})
    assert exc_info.value.data is None


@pytest.mark.asyncio
async def test_clear_cancels_timeouts_and_raises_runtime_error() -> None:
    send = AsyncMock()
    client = make_client(send)
    loop = asyncio.get_running_loop()
    fut = loop.create_future()
    timeout_task = asyncio.create_task(asyncio.sleep(9999))

    class FakePending:
        future = fut
        timeout = timeout_task

    client.pending[1] = FakePending()  # type: ignore[assignment]

    client.clear()

    assert client.pending == {}
    assert fut.done()
    with pytest.raises(RuntimeError, match="Connection closed"):
        fut.result()


@pytest.mark.asyncio
async def test_clear_empties_pending() -> None:
    send = AsyncMock()
    client = make_client(send)
    loop = asyncio.get_running_loop()

    for i in (1, 2):
        fut = loop.create_future()
        timeout_task = asyncio.create_task(asyncio.sleep(9999))

        class FakePending:
            pass

        p = FakePending()
        p.future = fut  # type: ignore[attr-defined]
        p.timeout = timeout_task  # type: ignore[attr-defined]
        client.pending[i] = p  # type: ignore[assignment]

    client.clear()

    assert client.pending == {}


@pytest.mark.asyncio
async def test_timeout_times_out_and_raises() -> None:
    send = AsyncMock()

    client = make_client(send, timeout=0)

    with pytest.raises(TimeoutError):
        await client.request("m", {})


@pytest.mark.asyncio
async def test_timeout_cleans_pending() -> None:
    send = AsyncMock()

    client = make_client(send, timeout=0)

    with pytest.raises(TimeoutError):
        await client.request("m", {})

    assert client.pending == {}


@pytest.mark.asyncio
async def test_timeout_skips_when_future_already_done() -> None:
    client = make_client()

    loop = asyncio.get_running_loop()
    fut: asyncio.Future[object] = loop.create_future()
    fut.set_result("already done")

    await client._timeout_future(fut, 0, 99)  # pyright: ignore[reportPrivateUsage]
    assert fut.result() == "already done"


@pytest.mark.asyncio
async def test_timeout_when_id_already_removed_from_pending() -> None:
    client = make_client()

    loop = asyncio.get_running_loop()
    fut: asyncio.Future[object] = loop.create_future()
    await client._timeout_future(fut, 0, 42)  # pyright: ignore[reportPrivateUsage]
    assert 42 not in client.pending

    with pytest.raises(TimeoutError):
        await fut
