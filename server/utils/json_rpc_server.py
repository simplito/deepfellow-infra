# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""JSON RPC Server."""

import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any, Literal, TypedDict, TypeGuard

from pydantic import BaseModel

from server.utils.exceptions import ApiError

logger = logging.getLogger("uvicorn.error")


class JsonRpcServer[T]:
    def __init__(self, registry: Callable[[str, Any, T], Awaitable[Any]]):
        self.registry = registry

    async def process(self, data: str | bytes, context: T) -> str | None:
        """Process json rpc request."""
        try:
            obj = json.loads(data)
        except Exception:
            logger.exception("Error during parsing json-rpc request", data)
            return None
        if not _is_json_rpc_request(obj):
            logger.info("Invalid json-rpc request", obj)
            return None
        try:
            result = await self.registry(obj["method"], obj["params"], context)
            success_result = JsonRpcResponseSuccess(jsonrpc="2.0", id=obj["id"], result=result)
            return success_result.model_dump_json(indent=False, exclude_none=True)
        except Exception as e:
            logger.exception("Error during processing json-rpc request", obj)
            error = (
                JsonRpcError(code=e.code, message=e.message, data=e.data)
                if isinstance(e, ApiError)
                else JsonRpcError(code=-32603, message="Internal error")
            )
            error_result = JsonRpcResponseError(jsonrpc="2.0", id=obj["id"], error=error)
            return error_result.model_dump_json(indent=False, exclude_none=True)


class JsonRpcRequest(TypedDict):
    jsonrpc: Literal["2.0"]
    id: int
    method: str
    params: Any


def _is_json_rpc_request(x: Any) -> TypeGuard[JsonRpcRequest]:  # noqa: ANN401
    return (
        isinstance(x, dict)
        and "jsonrpc" in x
        and isinstance(x["jsonrpc"], str)
        and x["jsonrpc"] == "2.0"
        and "id" in x
        and isinstance(x["id"], int)
        and "method" in x
        and isinstance(x["method"], str)
        and "params" in x
    )


class JsonRpcResponseSuccess(BaseModel):
    jsonrpc: Literal["2.0"]
    id: int
    result: Any


class JsonRpcError(BaseModel):
    code: int
    message: str
    data: Any = None


class JsonRpcResponseError(BaseModel):
    jsonrpc: Literal["2.0"]
    id: int
    error: JsonRpcError
