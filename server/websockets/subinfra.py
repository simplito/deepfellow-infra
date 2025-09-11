"""Websocket Manager for subinfras."""

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from fastapi import WebSocket
from pydantic import BaseModel

from server.models.services import ListAllModelsFilters, ListAllModelsOut
from server.websockets.manager import ExternalWebsocketManager, InternalWebsocketManager

from .models import (
    InfraConnect,
    InfraInfo,
    JsonRpc,
    ModelsClear,
    ModelsList,
    ModelsUsage,
    SubInfraMsgs,
    WebsocketMsgs,
)

if TYPE_CHECKING:
    from server.config import AppSettings
    from server.services_manager import ServicesManager

logger = logging.getLogger("uvicorn.error")


class ExternalInfraWsManager(ExternalWebsocketManager):
    msgs_type: type[WebsocketMsgs] = SubInfraMsgs

    async def before_send_msg(self, msg: BaseModel) -> None:
        """Add logger msg."""
        debug_msg = f"Send msg {msg.model_dump()} to parent infra: {self.uri}"
        logger.debug(debug_msg)

    async def before_loop(self) -> bool:
        """Load models."""
        await self.load_models()
        return bool(self.uri)

    async def on_start(self) -> None:
        """On strat functions."""
        await self.send_info()
        await self.send_models()

    async def send_info(self) -> None:
        """Send info through websocket."""
        config: AppSettings = self.state.config
        param = InfraInfo(name=config.name, url=config.url, api_key=config.api_key)
        msg = InfraConnect(params=[param])
        self.send_message_in_a_while(msg)

    async def send_models(self) -> None:
        """Send models through websocket."""
        await self.load_models()

        if not self.uri:
            return

        params: dict[str, dict[str, list[str]]] = self.state.models_list
        msg = ModelsList(params=params)
        self.send_message_in_a_while(msg)

    async def load_models(self) -> None:
        """Load models."""
        config: AppSettings = self.state.config
        services_manager: ServicesManager = self.state.services_manager
        all_models: ListAllModelsOut = await services_manager.list_models_from_all_services(ListAllModelsFilters(installed=True))
        models: dict[str, list[str]] = {}
        model_names: list[str] = []
        for model in all_models.list:
            if not models.get(model.type):
                models[model.type] = []

            models[model.type].append(model.id)
            model_names.append(model.id)

        self.state.models_list = {config.url: models}
        self.state.models_usage = {model: {config.url: 0} for model in model_names}

    def clear_from_infra(self, urls: list[str]) -> None:
        """Send clear models and usage from websocket."""
        if not self.uri:
            return

        self.send_message_in_a_while(ModelsClear(params=urls))

    def add_usage(self, model: str) -> None:
        """Add usage to specified model."""
        config: AppSettings = self.state.config
        url: str = config.url
        models_usage: dict[str, dict[str, int]] = self.state.models_usage
        new_usage: int
        new_usage = 1
        if (url_and_usage := models_usage.get(model)) and (usage := url_and_usage.get(url)):
            new_usage = usage + 1

        if models_usage.get(model):
            models_usage[model].update({url: new_usage})
        else:
            models_usage[model] = {url: new_usage}

        debug_msg = f"Add usage for local infra in model: {model} to level {new_usage}"
        logger.debug(debug_msg)

        if not self.uri:
            return

        self.send_usage()

    def remove_usage(self, model: str) -> None:
        """Remove usage to specified model."""
        config: AppSettings = self.state.config
        url: str = config.url
        models_usage: dict[str, dict[str, int]] = self.state.models_usage
        new_usage: int
        new_usage = 0

        if (url_and_usage := models_usage.get(model)) and (usage := url_and_usage.get(url)):
            new_usage = usage - 1

        if models_usage.get(model):
            models_usage[model].update({url: new_usage})

        debug_msg = f"Remove usage for local infra in model: {model} to level {new_usage}"
        logger.debug(debug_msg)

        if not self.uri:
            return

        self.send_usage()

    def send_usage(self) -> None:
        """Remove usage to specified model."""
        if not self.uri:
            return

        usages = self.state.models_usage
        msg = ModelsUsage(params=usages)
        self.send_message_in_a_while(msg)

    def send_message_in_a_while(self, msg: JsonRpc) -> None:
        """Send message in task."""
        if not self.uri:
            return

        tasks: set[asyncio.Task[None]] = self.tasks
        tasks.add(asyncio.create_task(self.send_message(msg)))


class InternalInfraWsManager(InternalWebsocketManager):
    msgs_type: type[WebsocketMsgs] = SubInfraMsgs

    async def handle_disconnect(self, ws: WebSocket, context: dict[str, Any]) -> None:
        """Clear data from infra."""
        urls = context.get("urls", [])
        self.remove_infras_data(ws, urls)
        debug_msg = f"Infras disconnected: {urls}"
        logger.debug(debug_msg)

    def clear_models(self, ws: WebSocket, urls: list[str]) -> None:
        """Clear models for specified infra urls."""
        models_list: dict[str, dict[str, list[str]]] = ws.app.state.models_list
        urls_to_delete: list[str] = []
        for url, _ in models_list.items():
            if url in urls:
                urls_to_delete.append(url)

        for url in urls_to_delete:
            del models_list[url]

        debug_msg = f"Models list after clear: {models_list}"
        logger.debug(debug_msg)

    def clear_usage(self, ws: WebSocket, urls: list[str]) -> None:
        """Clear usage for specified infra urls."""
        models_usage: dict[str, dict[str, int]] = ws.app.state.models_usage
        to_delete: list[tuple[str, str]] = []
        for model, url_and_usage in models_usage.items():
            for url, _ in url_and_usage.items():
                if url in urls:
                    to_delete.append((model, url))

        for model, url in to_delete:
            del models_usage[model][url]

            if models_usage.get(model, {}) == {}:
                del models_usage[model]

        debug_msg = f"Models usage after clear: {models_usage}"
        logger.debug(debug_msg)

    def clear_infos(self, ws: WebSocket, urls: list[str]) -> None:
        """Clear infos for specified infra urls."""
        infra_infos: list[InfraInfo] = ws.app.state.infra_infos
        new_infos: list[InfraInfo] = []

        # Create new list instead of removing element from old
        for info in infra_infos:
            if info.url not in urls:
                new_infos.append(info)

        infra_infos = new_infos

        debug_msg = f"Infra infos list after clear: {new_infos}"
        logger.debug(debug_msg)

    def remove_infras_data(self, ws: WebSocket, urls: list[str]) -> None:
        """Remove infra models."""
        external_ws: ExternalInfraWsManager = ws.app.state.external_ws_manager
        external_ws.clear_from_infra(list(urls))

        self.clear_infos(ws, urls)
        self.clear_models(ws, urls)
        self.clear_usage(ws, urls)

    def handle_models_list_msg(self, msg: JsonRpc, ws: WebSocket, context: dict[str, Any]) -> None:
        """Handle models list msg."""
        if isinstance(msg, ModelsList):
            debug_msg = f"Start handling models list: {msg}"
            logger.debug(debug_msg)
            external_ws: ExternalInfraWsManager = ws.app.state.external_ws_manager
            external_ws.send_message_in_a_while(msg)
            models_list: dict[str, dict[str, list[str]]] = ws.app.state.models_list
            models_usage: dict[str, dict[str, int]] = ws.app.state.models_usage

            for url, type_and_models in msg.params.items():
                if urls := context.get("urls"):
                    urls.append(url)
                else:
                    context["urls"] = [url]

                models_list[url] = type_and_models
                for _, models in type_and_models.items():
                    for model in models:
                        if models_usage.get(model, {}).get(url):
                            continue

                        models_usage[model].update({url: 0})

            debug_msg = f"Models list after handle message: {models_list}"
            logger.debug(debug_msg)

            debug_msg = f"Models usage after handle message: {models_usage}"
            logger.debug(debug_msg)

    def handle_models_usage_msg(self, msg: JsonRpc, ws: WebSocket) -> None:
        """Handle models usage msg."""
        if isinstance(msg, ModelsUsage):
            debug_msg = f"Start handling models usage: {msg}"
            logger.debug(debug_msg)
            external_ws: ExternalInfraWsManager = ws.app.state.external_ws_manager
            external_ws.send_message_in_a_while(msg)
            models_usage: dict[str, dict[str, int]] = ws.app.state.models_usage

            debug_msg = f"Models usage before handle message: {models_usage}"
            logger.debug(debug_msg)

            for model, url_and_usage in msg.params.items():
                url_and_usage: dict[str, int]
                if not models_usage.get(model, {}):
                    models_usage[model] = url_and_usage
                else:
                    models_usage[model].update(url_and_usage)

            debug_msg = f"Models usage after handle message: {models_usage}"
            logger.debug(debug_msg)

    def handle_infra_clear(self, msg: JsonRpc, ws: WebSocket) -> None:
        """Handle models clear msg."""
        if isinstance(msg, ModelsClear):
            debug_msg = f"Start handling models clear: {msg}"
            logger.debug(debug_msg)
            external_ws: ExternalInfraWsManager = ws.app.state.external_ws_manager
            external_ws.send_message_in_a_while(msg)
            self.remove_infras_data(ws, msg.params)

    def handle_infra_info(self, msg: JsonRpc, ws: WebSocket) -> None:
        """Handle infra info msg."""
        if isinstance(msg, InfraConnect):
            infra_infos: list[InfraInfo] = ws.app.state.infra_infos
            infra_infos.extend(msg.params)
            external_ws: ExternalInfraWsManager = ws.app.state.external_ws_manager
            external_ws.send_message_in_a_while(msg)

            debug_msg = f"Add infra info: {msg}"
            logger.debug(debug_msg)

    async def handle_msg(self, msg: JsonRpc, ws: WebSocket, context: dict[str, Any]) -> None:  # type: ignore
        """Place to handle all msgs from jsonrpc."""
        self.handle_models_usage_msg(msg, ws)
        self.handle_models_list_msg(msg, ws, context)
        self.handle_infra_clear(msg, ws)
        self.handle_infra_info(msg, ws)
