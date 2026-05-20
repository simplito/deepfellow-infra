# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from typing import Any

import pytest

from server.utils.exceptions import ApiError
from server.utils.json_rpc_server import JsonRpcServer
from server.utils.loading import Progress  # pyright: ignore[reportPrivateUsage]


def make_request(method: str = "test", params: object = None, id: int = 1) -> str:
    return json.dumps({"jsonrpc": "2.0", "id": id, "method": method, "params": params})


@pytest.fixture
def server() -> JsonRpcServer[Any]:
    async def registry(method: Any, params: Any, ctx: Any) -> dict[str, Any]:
        return {"echo": params}

    return JsonRpcServer(registry)


@pytest.mark.asyncio
async def test_process_returns_none_on_invalid_json(server: JsonRpcServer[Any]) -> None:
    result = await server.process("not json {{{", context=None)
    assert result is None


@pytest.mark.asyncio
async def test_process_returns_none_on_invalid_rpc_structure(server: JsonRpcServer[Any]) -> None:
    result = await server.process(json.dumps({"foo": "bar"}), context=None)
    assert result is None


@pytest.mark.asyncio
async def test_process_successful_response(server: JsonRpcServer[Any]) -> None:
    result = await server.process(make_request(params={"x": 1}), context=None)

    assert result is not None
    data = json.loads(result)
    assert data["jsonrpc"] == "2.0"
    assert data["id"] == 1
    assert data["result"] == {"echo": {"x": 1}}
    assert "error" not in data


@pytest.mark.asyncio
async def test_process_api_error_returns_rpc_error() -> None:
    async def registry(method: Any, params: Any, ctx: Any) -> None:
        raise ApiError("Not found", code=404, data={"detail": "missing"})

    srv = JsonRpcServer(registry)

    result = await srv.process(make_request(), context=None)

    assert result is not None
    data = json.loads(result)
    assert data["error"]["code"] == 404
    assert data["error"]["message"] == "Not found"
    assert data["error"]["data"] == {"detail": "missing"}


@pytest.mark.asyncio
async def test_process_generic_exception_returns_internal_error() -> None:
    async def registry(method: Any, params: Any, ctx: Any) -> None:
        raise RuntimeError("boom")

    srv = JsonRpcServer(registry)

    result = await srv.process(make_request(), context=None)

    assert result is not None
    data = json.loads(result)
    assert data["error"]["code"] == -32603
    assert data["error"]["message"] == "Internal error"


@pytest.mark.asyncio
async def test_process_accepts_bytes_input(server: JsonRpcServer[Any]) -> None:
    payload = make_request(params=42).encode()

    result = await server.process(payload, context=None)

    assert result is not None
    data = json.loads(result)
    assert data["result"] == {"echo": 42}


def test_set_max_value():
    p = Progress(max_value=100.0, actual_value=50.0)
    p.calculate_percentage()

    p.set_max_value(200.0)

    assert p.max == 200.0
    assert p.percentage == 0.25


def test_get_percentage_returns_stored_percentage():
    p = Progress(max_value=100.0, actual_value=50.0)
    p.calculate_percentage()
    assert p.get_percentage() == 0.5


@pytest.mark.asyncio
async def test_process_context_forwarded_to_registry() -> None:
    received = {}

    async def registry(method: Any, params: Any, ctx: Any) -> None:
        received["ctx"] = ctx
        return

    srv = JsonRpcServer(registry)

    await srv.process(make_request(), context="my-ctx")

    assert received["ctx"] == "my-ctx"


@pytest.mark.asyncio
async def test_process_error_response_excludes_none_data() -> None:
    async def registry(method: Any, params: Any, ctx: Any) -> None:
        raise ApiError("Oops", code=500)

    srv = JsonRpcServer(registry)

    result = await srv.process(make_request(), context=None)

    assert result is not None
    data = json.loads(result)
    assert "data" not in data["error"]
