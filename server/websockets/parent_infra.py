# DeepFellow Software Framework.
# Copyright © 2025 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Websocket Manager for subinfras."""

import logging
from typing import TYPE_CHECKING

from server.config import AppSettings
from server.task_manager import TaskManager
from server.utils.exceptions import ApiError
from server.utils.json_rpc_client import JsonRpcClient
from server.websockets.infra_client import InfraClient
from server.websockets.models import InitRequest, UpdateModelsRequest, UsageChangeRequest
from server.websockets.websocket_client import WebSocketClient

if TYPE_CHECKING:
    from server.endpointregistry import EndpointRegistry

logger = logging.getLogger("uvicorn.error")


class ParentInfra(WebSocketClient):
    endpoint_registry: "EndpointRegistry"

    def __init__(
        self,
        config: AppSettings,
        task_manager: TaskManager,
    ):
        self.config = config
        self.task_manager = task_manager
        self.client = JsonRpcClient(send=lambda x: self._send(x), timeout=30)
        self.infra_client = InfraClient(self.client)
        self.enabled = self.config.connect_to_mesh_url != ""
        uri = f"{self.config.connect_to_mesh_url}/ws" if self.enabled else ""
        super().__init__(uri)

    async def _send(self, data: str) -> None:
        if not self.enabled or not self.ws:
            raise RuntimeError("Not connected")
        self.send(data)

    def on_message(self, msg: str | bytes) -> None:
        """Perform action on new message."""
        self.client.resolve(msg)

    def on_disconnect(self) -> None:
        """Perform action on disconnect."""
        self.client.clear()

    async def before_loop(self) -> bool:
        """Load models."""
        return self.enabled

    async def on_start(self) -> None:
        """On start functions."""
        try:
            await self.infra_client.init(
                InitRequest(
                    auth=self.config.connect_to_mesh_key.get_secret_value(),
                    name=self.config.name,
                    url=self.config.infra_url,
                    api_key=self.config.infra_api_key.get_secret_value(),
                    models=self.endpoint_registry.list_models(),
                )
            )
        except ApiError as e:
            if e.code == 2 and e.message == "Invalid api key":
                self.process_loop = False
            raise

    def send_usage(self, usage: UsageChangeRequest) -> None:
        """Send usage."""
        if not self.enabled or not self.ws:
            return

        self.task_manager.add_task_safe(self.infra_client.usage_change(usage), "parent_infra.usage_change")

    def send_models_list(self) -> None:
        """Send usage."""
        if not self.enabled or not self.ws:
            return

        self.task_manager.add_task_safe(
            self.infra_client.update_models(UpdateModelsRequest(models=self.endpoint_registry.list_models())),
            "parent_infra.update_models",
        )
