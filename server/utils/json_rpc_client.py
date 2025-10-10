"""JSON RPC Client."""

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any, Literal, NotRequired, TypedDict, TypeGuard

from pydantic import BaseModel

from server.utils.exceptions import ApiError

logger = logging.getLogger("uvicorn.error")


class Pending:
    def __init__(self, future: asyncio.Future[Any], timeout: asyncio.Future[None]):
        self.future = future
        self.timeout = timeout


class JsonRpcClient:
    def __init__(self, send: Callable[[str], Awaitable[None]], timeout: int):
        self.id = 0
        self._send = send
        self.pending = dict[int, Pending]()
        self.timeout = timeout

    async def request(self, method: str, params: Any) -> Any:  # noqa: ANN401
        """Make request."""
        id = self.id = self.id + 1
        fut = asyncio.get_running_loop().create_future()
        timeout = asyncio.create_task(self._timeout_future(fut, self.timeout, id))
        self.pending[id] = Pending(future=fut, timeout=timeout)
        request = JsonRpcRequest(jsonrpc="2.0", id=id, method=method, params=params)
        await self._send(request.model_dump_json(indent=None, exclude_none=True))
        return await fut

    def resolve(self, data: str | bytes) -> None:
        """Resolve json response."""
        try:
            obj = json.loads(data)
        except Exception:
            logger.exception("Error during parsing json-rpc response", data)
            return
        if not _is_json_rpc_response(obj):
            logger.info("Invalid json-rpc response", obj)
            return
        id = obj["id"]
        if id not in self.pending:
            logger.info("There is no correspoding future for a json-rpc response")
            return
        future = self.pending.pop(id)
        future.timeout.cancel()
        if "result" in obj:
            future.future.set_result(obj["result"])
        elif "error" in obj:
            error = obj["error"]
            if _is_json_rpc_error(error):
                future.future.set_exception(ApiError(code=error["code"], message=error["message"], data=(error.get("data", None))))
            else:
                future.future.set_exception(RuntimeError("Unprocessable error object in json-rpc response"))
        else:
            future.future.set_exception(RuntimeError("Invalid JSON-RPC response: missing both 'result' and 'error' fields."))

    def clear(self) -> None:
        """Resolve all pending futures with error and clear pending list."""
        for future in self.pending.values():
            future.future.set_exception(RuntimeError("Connection closed"))
            future.timeout.cancel()
        self.pending = {}

    async def _timeout_future(self, fut: asyncio.Future[Any], timeout: int, id: int) -> None:
        await asyncio.sleep(timeout)
        if not fut.done():
            fut.set_exception(TimeoutError("Timeout"))
            if id in self.pending:
                del self.pending[id]


class JsonRpcRequest(BaseModel):
    jsonrpc: Literal["2.0"]
    id: str | int
    method: str
    params: Any


class JsonRpcResponse(TypedDict):
    jsonrpc: Literal["2.0"]
    id: int
    result: NotRequired[Any]
    error: NotRequired[Any]


class JsonRpcError(TypedDict):
    code: int
    message: str
    data: NotRequired[Any]


def _is_json_rpc_response(x: Any) -> TypeGuard[JsonRpcResponse]:  # noqa: ANN401
    return (
        isinstance(x, dict)
        and "jsonrpc" in x
        and isinstance(x["jsonrpc"], str)
        and x["jsonrpc"] == "2.0"
        and "id" in x
        and isinstance(x["id"], int)
    )


def _is_json_rpc_error(x: Any) -> TypeGuard[JsonRpcError]:  # noqa: ANN401
    return isinstance(x, dict) and "code" in x and isinstance(x["code"], int) and "message" in x and isinstance(x["message"], str)
