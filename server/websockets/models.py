"""Websocket models."""

from typing import Any, Literal, TypeVar
from uuid import uuid4

from pydantic import BaseModel, Field, SecretStr, field_serializer, field_validator

PydanticModel = TypeVar("PydanticModel", bound=BaseModel)


class JsonRpcBase(BaseModel):
    jsonrpc: Literal["2.0"] = "2.0"


class JsonRpcIdCheck(BaseModel):
    id: str


class JsonRpcIdOptional(BaseModel):
    id: str | None = None


class JsonRpcId(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))


class JsonRpcNotificationRequest(JsonRpcBase):
    method: str
    params: Any  # ToDo


class JsonRpcNotificationResponse(JsonRpcBase):
    result: Any  # ToDo


class JsonRpcCallRequest(JsonRpcNotificationRequest, JsonRpcId):
    pass


class JsonRpcCallResponse(JsonRpcNotificationResponse, JsonRpcIdCheck):
    pass


class JsonRpcError(BaseModel):
    code: int
    message: str


class JsonRpcErrorResponse(JsonRpcBase, JsonRpcId):
    error: JsonRpcError


class InfraInfo(BaseModel):
    name: str
    url: str
    api_key: SecretStr

    @field_serializer("api_key", when_used="json")
    def dump_secret(self, v: SecretStr) -> str:
        """Return secret visible."""
        return v.get_secret_value()


class InfraConnect(JsonRpcNotificationRequest):
    method: Literal["infra-connect"] = "infra-connect"  # type: ignore
    params: list[InfraInfo]


type Models = dict[str, dict[str, list[str]]]  # url | type | models
type Usages = dict[str, dict[str, int]]  # model | url | usage


class ModelsList(JsonRpcNotificationRequest):
    method: Literal["models-list"] = "models-list"  # type: ignore
    params: Models


class ModelsUsage(JsonRpcNotificationRequest):
    method: Literal["models-usage"] = "models-usage"  # type: ignore
    params: Usages


class ModelsClear(JsonRpcNotificationRequest):
    method: Literal["models-clear"] = "models-clear"  # type: ignore
    params: list[str]  # url


JsonRpc = ModelsUsage | ModelsList | ModelsClear | InfraConnect


class WebsocketMsgs(BaseModel):
    msgs: list[BaseModel]

    @field_validator("msgs", mode="before")
    @classmethod
    def ensure_list(cls, v: Any) -> list[Any]:  # noqa: ANN401
        """Convert object to list of objects. If is list do nothing."""
        return ensure_list(v)  # type: ignore


class SubInfraMsgs(WebsocketMsgs):
    msgs: list[JsonRpc]  # type: ignore


def ensure_list(v: Any) -> list[Any]:  # noqa: ANN401
    """Convert object to list of objects. If is list do nothing."""
    return v if isinstance(v, list) else [v]  # type: ignore
