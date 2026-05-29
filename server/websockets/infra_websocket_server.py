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
from server.models.mesh import CheckMeshConnection, MeshInfo, MeshInfoInfra, MeshInfoModel, MeshTopologyNode
from server.utils.core import make_http_request
from server.utils.exceptions import ApiError
from server.utils.json_rpc_server import JsonRpcServer
from server.websockets.models import AncestorInfo, InitRequest, InitResponse, TopologyUpdateRequest, UpdateModelsRequest, UsageChangeRequest
from server.websockets.parent_infra_group import ParentInfraGroup
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
    def __init__(self, config: AppSettings, parent_infra: ParentInfraGroup, endpoint_registry: EndpointRegistry):
        super().__init__()
        self.config = config
        self.parent_infra = parent_infra
        self.endpoint_registry = endpoint_registry
        self._sub_connections: dict[str, str] = {}
        self._nested_topology: dict[str, TopologyUpdateRequest] = {}
        self.server = JsonRpcServer[InfraWsData](lambda method, params, context: self._handle_json_rpc_request(method, params, context))
        for parent in parent_infra.parents:
            parent.get_children = lambda: dict(self._nested_topology)

    @property
    def _own_ancestors(self) -> list[AncestorInfo]:
        """Full ordered list of ancestors for this node (parent first)."""
        return self.parent_infra.ancestors

    def create_bag(self) -> InfraWsData:
        """Create websocket bag."""
        return InfraWsData(authorized=None)

    def handle_disconnect(self, context: InfraWsContext) -> None:
        """Clear data from infra."""
        if context.data.authorized:
            url = context.data.authorized.url
            self.endpoint_registry.update_models(
                context.data.authorized.models,
                [],
                url,
                context.data.authorized.api_key,
            )
            self._sub_connections.pop(url, None)
            self._nested_topology.pop(url, None)
            self._send_topology_update("leave", url, "", [])

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
        if method == "topology_update":
            return self._on_topology_update(self._try_parse(params, TopologyUpdateRequest), context)
        raise ApiError(code=-32601, message="Method not found", data=method)

    def _try_parse(self, data: Any, cls: type[T]) -> T:  # noqa: ANN401
        try:
            return cls(**data)
        except Exception as e:
            raise ApiError(code=-32602, message="Invalid params", data=e.errors if isinstance(e, ValidationError) else None) from e

    async def _on_init(self, params: InitRequest, context: InfraWsData) -> InitResponse:
        if context.authorized:
            raise ApiError(code=1, message="Already authorized")
        if params.auth != self.config.mesh_key.get_secret_value():
            raise ApiError(code=2, message="Invalid api key")

        url = f"{params.url}/admin/mesh/check"
        connection_info = await make_http_request(
            method="POST",
            url=url,
            data=CheckMeshConnection(connection_verifier=params.check_key, infra_api_key=params.api_key).model_dump_json().encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        if connection_info.response.status != 200:
            raise ApiError(code=4, message="Subinfra verification error", data={"url": url, "status": connection_info.response.status})

        loop_urls = {self.config.infra_url} | {a.url for a in self._own_ancestors}
        clean_children = {url: self._strip_loop_urls(child, loop_urls) for url, child in params.children.items() if url not in loop_urls}
        context.authorized = Authorized(name=params.name, url=params.url, api_key=params.api_key, models=params.models)
        self.endpoint_registry.update_models([], params.models, params.url, params.api_key)
        self._sub_connections[params.url] = params.name
        self._nested_topology[params.url] = TopologyUpdateRequest(
            action="join", url=params.url, name=params.name, models=params.models, children=clean_children
        )
        self._send_topology_update("join", params.url, params.name, params.models, clean_children)
        msg = f"Connected with subinfra {params.name} on {params.url}."
        logger.info(msg)
        own = AncestorInfo(
            url=self.config.infra_url,
            name=self.config.name,
            models=self.endpoint_registry.list_models(),
        )
        return InitResponse(ancestors=[own, *self._own_ancestors])

    def _on_topology_update(self, params: TopologyUpdateRequest, context: InfraWsData) -> Literal["OK"]:
        if not context.authorized:
            raise ApiError(code=3, message="Not authorized")
        reporter_url = context.authorized.url
        reporter_name = context.authorized.name
        reporter_models = context.authorized.models
        loop_urls = {self.config.infra_url} | {a.url for a in self._own_ancestors}
        if params.url in loop_urls:
            logger.warning("Circular mesh connection detected from %s reporting %s — skipping", reporter_url, params.url)
            return "OK"
        if reporter_url not in self._nested_topology:
            self._nested_topology[reporter_url] = TopologyUpdateRequest(
                action="join", url=reporter_url, name=reporter_name, models=reporter_models
            )
        reporter_subtree = self._nested_topology[reporter_url]
        if params.action == "join":
            reporter_subtree.children[params.url] = self._strip_loop_urls(params, loop_urls)
        else:
            reporter_subtree.children.pop(params.url, None)
        self._send_topology_update("join", reporter_url, reporter_name, reporter_models, reporter_subtree.children)
        return "OK"

    def _strip_loop_urls(self, params: TopologyUpdateRequest, loop_urls: set[str]) -> TopologyUpdateRequest:
        """Recursively remove children whose URL would create a topology loop."""
        filtered = {url: self._strip_loop_urls(child, loop_urls) for url, child in params.children.items() if url not in loop_urls}
        if len(filtered) == len(params.children):
            return params
        return TopologyUpdateRequest(action=params.action, url=params.url, name=params.name, models=params.models, children=filtered)

    def _send_topology_update(
        self,
        action: Literal["join", "leave"],
        url: str,
        name: str,
        models: list[Model],
        children: dict[str, TopologyUpdateRequest] | None = None,
    ) -> None:
        self.parent_infra.send_topology_update(action, url, name, models, children or {})

    def _on_usage_change(self, params: UsageChangeRequest, context: InfraWsData) -> Literal["OK"]:
        if not context.authorized:
            raise ApiError(code=3, message="Not authorized")
        self.endpoint_registry.update_usage(params)
        return "OK"

    def _on_update_models(self, params: UpdateModelsRequest, context: InfraWsData) -> Literal["OK"]:
        if not context.authorized:
            raise ApiError(code=3, message="Not authorized")
        url = context.authorized.url
        self.endpoint_registry.update_models(context.authorized.models, params.models, url, context.authorized.api_key)
        context.authorized.models = params.models
        if url in self._nested_topology:
            self._nested_topology[url].models = params.models
        subtree = self._nested_topology.get(url)
        self._send_topology_update("join", url, context.authorized.name, params.models, subtree.children if subtree else {})
        return "OK"

    def _subtree_to_node(self, req: TopologyUpdateRequest) -> MeshTopologyNode:
        return MeshTopologyNode(
            url=req.url,
            name=req.name,
            models=[MeshInfoModel(name=m.name, type=m.type) for m in req.models],
            children=[self._subtree_to_node(child) for child in req.children.values()],
        )

    def get_topology(self) -> list[MeshTopologyNode]:
        """Build full mesh tree. Ancestors are stub nodes wrapping the current infra."""
        self_children: list[MeshTopologyNode] = []
        for conn in self.connections:
            if conn.data.authorized:
                child_url = conn.data.authorized.url
                child_name = conn.data.authorized.name
                child_models = [MeshInfoModel(name=m.name, type=m.type) for m in conn.data.authorized.models]
                subtree = self._nested_topology.get(child_url)
                children = [self._subtree_to_node(child) for child in subtree.children.values()] if subtree else []
                self_children.append(MeshTopologyNode(url=child_url, name=child_name, models=child_models, children=children))

        current: MeshTopologyNode = MeshTopologyNode(
            url=self.config.infra_url,
            name=self.config.name,
            models=[MeshInfoModel(name=m.name, type=m.type) for m in self.endpoint_registry.list_models()],
            children=self_children,
            you_are_here=True,
        )
        for ancestor in self.parent_infra.ancestors:
            current = MeshTopologyNode(
                url=ancestor.url,
                name=ancestor.name,
                models=[MeshInfoModel(name=m.name, type=m.type) for m in ancestor.models],
                children=[current],
            )
        return [current]

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
