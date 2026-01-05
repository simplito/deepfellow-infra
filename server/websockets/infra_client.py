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
from server.websockets.models import InitRequest, UpdateModelsRequest, UsageChangeRequest


class InfraClient:
    def __init__(self, client: JsonRpcClient):
        self.client = client

    async def init(self, params: InitRequest) -> Literal["OK"]:
        """Initialize connection."""
        return await self.client.request("init", params)

    async def usage_change(self, params: UsageChangeRequest) -> Literal["OK"]:
        """Inform about usage change."""
        return await self.client.request("usage_change", params)

    async def update_models(self, params: UpdateModelsRequest) -> Literal["OK"]:
        """Inform about new list of models."""
        return await self.client.request("update_models", params)
