"""Infra Websocket Server."""

import logging

from fastapi import WebSocket

from server.websockets.models import InfraConnect, InfraInfo, JsonRpc, ModelsClear, ModelsList, ModelsUsage, SubInfraMsgs, WebsocketMsgs
from server.websockets.parent_infra import ParentInfra
from server.websockets.usage_manager import UsageManager
from server.websockets.websocket_server import WebSocketServer

logger = logging.getLogger("uvicorn.error")


class WebSocketContext:
    urls: list[str]


class InfraWebsocketServer(WebSocketServer[WebSocketContext]):
    msgs_type: type[WebsocketMsgs] = SubInfraMsgs

    def __init__(
        self,
        infra_infos: list[InfraInfo],
        parent_infra: ParentInfra,
        usage_manager: UsageManager,
    ):
        self.infra_infos = infra_infos
        self.parent_infra = parent_infra
        self.usage_manager = usage_manager

    async def handle_disconnect(self, ws: WebSocket, context: WebSocketContext) -> None:  # noqa: ARG002
        """Clear data from infra."""
        self.parent_infra.send_clear(context.urls)
        self.usage_manager.clear_urls(context.urls)
        debug_msg = f"Infras disconnected: {context.urls}"
        logger.debug(debug_msg)

    def handle_models_list_msg(self, msg: JsonRpc, context: WebSocketContext) -> None:
        """Handle models list msg."""
        if isinstance(msg, ModelsList):
            debug_msg = f"Start handling models list: {msg}"
            logger.debug(debug_msg)
            self.parent_infra.send_message_in_a_while(msg)

            self.usage_manager.update_models(msg.params, context.urls)

    def handle_models_usage_msg(self, msg: JsonRpc) -> None:
        """Handle models usage msg."""
        if isinstance(msg, ModelsUsage):
            debug_msg = f"Start handling models usage: {msg}"
            logger.debug(debug_msg)
            self.parent_infra.send_message_in_a_while(msg)

            self.usage_manager.update_usage(msg.params)

    def handle_infra_clear(self, msg: JsonRpc) -> None:
        """Handle models clear msg."""
        if isinstance(msg, ModelsClear):
            debug_msg = f"Start handling models clear: {msg}"
            logger.debug(debug_msg)
            self.parent_infra.send_message_in_a_while(msg)

            self.usage_manager.clear_urls(msg.params)

    def handle_infra_info(self, msg: JsonRpc) -> None:
        """Handle infra info msg."""
        if isinstance(msg, InfraConnect):
            debug_msg = f"Add infra info: {msg}"
            logger.debug(debug_msg)
            self.parent_infra.send_message_in_a_while(msg)

            self.usage_manager.update_infos(msg.params)

    async def handle_msg(self, msg: JsonRpc, context: WebSocketContext) -> None:  # type: ignore
        """Place to handle all msgs from jsonrpc."""
        self.handle_models_usage_msg(msg)
        self.handle_models_list_msg(msg, context)
        self.handle_infra_clear(msg)
        self.handle_infra_info(msg)
