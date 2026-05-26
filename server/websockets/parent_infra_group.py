# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Group of ParentInfra connections for multi-parent mesh topology."""

import asyncio
from typing import TYPE_CHECKING, Literal

from server.models.api import Model
from server.models.mesh import CheckMeshConnection
from server.websockets.models import AncestorInfo, TopologyUpdateRequest, UsageChangeRequest
from server.websockets.parent_infra import ParentInfra

if TYPE_CHECKING:
    from server.endpointregistry import EndpointRegistry


class ParentInfraGroup:
    def __init__(self, parents: list[ParentInfra]):
        self.parents = parents

    @property
    def enabled(self) -> bool:
        """Return True if any parent connection is enabled."""
        return any(parent.enabled for parent in self.parents)

    @property
    def ancestors(self) -> list[AncestorInfo]:
        """Deduplicated union of all ancestor chains (parent first)."""
        seen: set[str] = set()
        result: list[AncestorInfo] = []
        for parent in self.parents:
            if not parent.enabled:
                continue
            for ancestor in parent.ancestors:
                if ancestor.url not in seen:
                    seen.add(ancestor.url)
                    result.append(ancestor)
        return result

    @property
    def parent_urls(self) -> list[str]:
        """Direct parent URLs (no ancestor chains)."""
        return [parent.parent_url for parent in self.parents if parent.enabled]

    @property
    def endpoint_registry(self) -> "EndpointRegistry | None":
        """Return the endpoint registry from the first parent."""
        return self.parents[0].endpoint_registry if self.parents else None

    @endpoint_registry.setter
    def endpoint_registry(self, value: "EndpointRegistry") -> None:
        for parent in self.parents:
            parent.endpoint_registry = value

    def send_models_list(self) -> None:
        """Broadcast the models list to all parents."""
        for parent in self.parents:
            parent.send_models_list()

    def send_usage(self, usage: UsageChangeRequest) -> None:
        """Broadcast a usage change to all parents."""
        for parent in self.parents:
            parent.send_usage(usage)

    def send_topology_update(
        self,
        action: Literal["join", "leave"],
        url: str,
        name: str,
        models: list[Model],
        children: "dict[str, TopologyUpdateRequest] | None" = None,
    ) -> None:
        """Send a topology update to all enabled parents."""
        for parent in self.parents:
            if not parent.enabled or not parent.ws:
                # Dropped: on reconnect, on_start sends the full sub-tree atomically via InitRequest.children.
                continue
            parent.task_manager.add_task_safe(
                parent.infra_client.topology_update(
                    TopologyUpdateRequest(action=action, url=url, name=name, models=models, children=children or {})
                ),
                "infra_websocket_server.topology_update",
            )

    def check_subinfra_connection(self, model: CheckMeshConnection) -> bool:
        """Return True if any parent can reach the given subinfra model."""
        return any(parent.check_subinfra_connection(model) for parent in self.parents)

    async def run(self) -> None:
        """Run all parent connections concurrently."""
        if not self.parents:
            return
        await asyncio.gather(*[parent.run() for parent in self.parents])
