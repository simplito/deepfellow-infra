"""Websocket Manager for subinfras."""

import asyncio
import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel

from server.config import AppSettings, ParentInfraConfig
from server.websockets.websocket_client import WebSocketClient

from .models import (
    InfraConnect,
    InfraInfo,
    JsonRpc,
    ModelsClear,
    ModelsList,
    ModelsUsage,
    SubInfraMsgs,
    Usages,
    WebsocketMsgs,
)

if TYPE_CHECKING:
    from server.websockets.usage_manager import UsageManager

logger = logging.getLogger("uvicorn.error")


def create_infra_uri(parent_infra_config: ParentInfraConfig) -> str:
    """Create infra uri."""
    uri = ""

    if parent_infra_config.ws_url:
        uri = f"{parent_infra_config.ws_url}/ws"

    if parent_infra_config.api_key:
        uri += f"?key={parent_infra_config.api_key}"

    return uri


class ParentInfra(WebSocketClient):
    msgs_type: type[WebsocketMsgs] = SubInfraMsgs
    usage_manager: "UsageManager"

    def __init__(
        self,
        uri: str,
        config: AppSettings,
    ):
        super().__init__(uri)
        self.config = config

    async def before_send_msg(self, msg: BaseModel) -> None:
        """Add logger msg."""
        debug_msg = f"Send msg {msg.model_dump()} to parent infra: {self.uri}"
        logger.debug(debug_msg)

    async def before_loop(self) -> bool:
        """Load models."""
        self.usage_manager.load_models()
        return bool(self.uri)

    async def on_start(self) -> None:
        """On strat functions."""
        await self.send_info()
        self.send_models()

    async def send_info(self) -> None:
        """Send info through websocket."""
        param = InfraInfo(name=self.config.name, url=self.config.url, api_key=self.config.api_key)
        msg = InfraConnect(params=[param])
        self.send_message_in_a_while(msg)

    def send_models(self) -> None:
        """Send models through websocket."""
        self.usage_manager.load_models()

        if not self.uri:
            return

        msg = ModelsList(params=self.usage_manager.get_models())
        self.send_message_in_a_while(msg)

    def send_clear(self, urls: list[str]) -> None:
        """Send clear models and usage from websocket."""
        if not self.uri:
            return

        self.send_message_in_a_while(ModelsClear(params=urls))

    def send_usage(self, usages: Usages) -> None:
        """Send usage."""
        if not self.uri:
            return

        msg = ModelsUsage(params=usages)
        self.send_message_in_a_while(msg)

    def send_message_in_a_while(self, msg: JsonRpc) -> None:
        """Send message in task."""
        if not self.uri:
            return

        self.tasks.add(asyncio.create_task(self.send_message(msg)))
