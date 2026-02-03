# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Infra Websocket Server."""

import logging
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ValidationError

from server.config import AppSettings
from server.endpointregistry import EndpointRegistry
from server.models.api import Model
from server.models.mesh import CheckMeshConnection, MeshInfo, MeshInfoInfra, MeshInfoModel
from server.utils.core import make_http_request
from server.utils.exceptions import ApiError
from server.utils.json_rpc_server import JsonRpcServer
from server.websockets.models import InitRequest, UpdateModelsRequest, UsageChangeRequest
from server.websockets.parent_infra import ParentInfra
from server.websockets.websocket_server import WebSocketContext, WebSocketServer

logger = logging.getLogger("uvicorn.error")
T = TypeVar("T")


class Authorized(BaseModel):
    name: str
    url: str
    api_key: str
    models: list[Model]


class InfraWsData(BaseModel):
    authorized: Authorized | None


InfraWsContext = WebSocketContext[InfraWsData]


class InfraWebsocketServer(WebSocketServer[InfraWsData]):
    def __init__(self, config: AppSettings, parent_infra: ParentInfra, endpoint_registry: EndpointRegistry):
        super().__init__()
        self.config = config
        self.parent_infra = parent_infra
        self.endpoint_registry = endpoint_registry
        self.server = JsonRpcServer[InfraWsData](lambda method, params, context: self._handle_json_rpc_request(method, params, context))

    def create_bag(self) -> InfraWsData:
        """Create websocket bag."""
        return InfraWsData(authorized=None)

    def handle_disconnect(self, context: InfraWsContext) -> None:
        """Clear data from infra."""
        if context.data.authorized:
            self.endpoint_registry.update_models(
                context.data.authorized.models,
                [],
                context.data.authorized.url,
                context.data.authorized.api_key,
            )

    async def process_message(self, msg: str, context: InfraWsContext) -> None:
        """Handle a websocket message."""
        result = await self.server.process(msg, context.data)
        if result is not None:
            context.send(result)

    async def _handle_json_rpc_request(self, method: str, params: Any, context: InfraWsData) -> Any:  # noqa: ANN401
        if method == "init":
            return await self._on_init(self._try_parse(params, InitRequest), context)
        if method == "usage_change":
            return self._on_usage_change(self._try_parse(params, UsageChangeRequest), context)
        if method == "update_models":
            return self._on_update_models(self._try_parse(params, UpdateModelsRequest), context)
        raise ApiError(code=-32601, message="Method not found", data=method)

    def _try_parse(self, data: Any, cls: type[T]) -> T:  # noqa: ANN401
        try:
            return cls(**data)
        except Exception as e:
            raise ApiError(code=-32602, message="Invalid params", data=e.errors if isinstance(e, ValidationError) else None) from e

    async def _on_init(self, params: InitRequest, context: InfraWsData) -> Literal["OK"]:
        if context.authorized:
            raise ApiError(code=1, message="Already authorized")
        if params.auth != self.config.mesh_key.get_secret_value():
            raise ApiError(code=2, message="Invalid api key")

        if params.check_key:  # TODO: params.check_key should be switched to requried parameter after old infra migration
            url = f"{params.url}/admin/mesh/check"
            connection_info = await make_http_request(
                method="POST",
                url=url,
                data=CheckMeshConnection(connection_verifier=params.check_key, infra_api_key=params.api_key)
                .model_dump_json()
                .encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            if connection_info.response.status != 200:
                raise ApiError(code=4, message="Subinfra verification error", data={"url": url, "status": connection_info.response.status})

        context.authorized = Authorized(name=params.name, url=params.url, api_key=params.api_key, models=params.models)
        self.endpoint_registry.update_models([], params.models, params.url, params.api_key)
        msg = f"Connected with subinfra {params.name} on {params.url}."
        logger.info(msg)
        return "OK"

    def _on_usage_change(self, params: UsageChangeRequest, context: InfraWsData) -> Literal["OK"]:
        if not context.authorized:
            raise ApiError(code=3, message="Not authorized")
        self.endpoint_registry.update_usage(params)
        return "OK"

    def _on_update_models(self, params: UpdateModelsRequest, context: InfraWsData) -> Literal["OK"]:
        if not context.authorized:
            raise ApiError(code=3, message="Not authorized")
        self.endpoint_registry.update_models(context.authorized.models, params.models, context.authorized.url, context.authorized.api_key)
        context.authorized.models = params.models
        return "OK"

    def get_mesh_info(self) -> MeshInfo:
        """Get mesh info."""
        connections = [
            MeshInfoInfra(
                name=connection.data.authorized.name,
                url=connection.data.authorized.url,
                models=[MeshInfoModel(name=model.name, type=model.type) for model in connection.data.authorized.models],
            )
            for connection in self.connections
            if connection.data.authorized
        ]
        return MeshInfo(connections=connections)
