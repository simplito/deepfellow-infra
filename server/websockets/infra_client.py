# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Infra client."""

from typing import Literal

from server.utils.json_rpc_client import JsonRpcClient
from server.websockets.models import InitRequest, InitResponse, TopologyUpdateRequest, UpdateModelsRequest, UsageChangeRequest


class InfraClient:
    def __init__(self, client: JsonRpcClient):
        self.client = client

    async def init(self, params: InitRequest) -> InitResponse:
        """Initialize connection."""
        result = await self.client.request("init", params)
        if isinstance(result, str):
            return InitResponse(ancestors=[])
        return InitResponse.model_validate(result)

    async def topology_update(self, params: TopologyUpdateRequest) -> Literal["OK"]:
        """Notify parent about a sub-connection joining or leaving."""
        return await self.client.request("topology_update", params)

    async def usage_change(self, params: UsageChangeRequest) -> Literal["OK"]:
        """Inform about usage change."""
        return await self.client.request("usage_change", params)

    async def update_models(self, params: UpdateModelsRequest) -> Literal["OK"]:
        """Inform about new list of models."""
        return await self.client.request("update_models", params)
