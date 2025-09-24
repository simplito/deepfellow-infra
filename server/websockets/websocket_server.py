"""Websocket Manager."""

import asyncio
import logging
from abc import abstractmethod
from contextlib import suppress

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ValidationError

from .models import JsonRpc, WebsocketMsgs

logger = logging.getLogger("uvicorn.error")


class WebSocketServer[T]:
    active_websockets: list[WebSocket]
    subscriptions: dict[str, list[WebSocket]]
    subscription_events: list[str]
    tasks: set[asyncio.Task[None]]
    msgs_type: type[WebsocketMsgs] = WebsocketMsgs

    def __init__(self):
        self.active_websockets = []
        self.subscriptions = {}
        self.subscription_events = []
        self.tasks = set()

    async def connect(self, ws: WebSocket) -> None:
        """Run on connect to datalink websocket."""
        await ws.accept()
        self.active_websockets.append(ws)

    async def disconnect(self, ws: WebSocket) -> None:
        """Run on disconnect to datalink websocket."""
        with suppress(ValueError):
            self.active_websockets.remove(ws)
        await self.unsubscribe_from_all_events(ws)

    async def subscribe(self, ws: WebSocket, event: str) -> None:
        """Subscribe websocket broadcast event."""
        if event not in self.subscription_events:
            self.subscription_events.append(event)

        if not self.subscriptions.get(event):
            self.subscriptions[event] = [ws]
        elif ws not in self.subscriptions.get(event, []):
            self.subscriptions[event].append(ws)

    async def unsubscribe(self, ws: WebSocket, event: str) -> None:
        """Uncubscribe websocket broadcast from event."""
        if self.subscriptions.get(event):
            with suppress(ValueError):
                self.subscriptions[event].remove(ws)

    async def unsubscribe_from_all_events(self, ws: WebSocket) -> None:
        """Uncubscribe websocket broadcast from all event."""
        for event in self.subscriptions:
            await self.unsubscribe(ws, event)

    async def broadcast_to_all(self, msg: JsonRpc) -> None:
        """Broadcast event for every websocket."""
        for ws in self.active_websockets:
            await ws.send_json(msg.model_dump())

    async def broadcast_event(self, event: str, msg: JsonRpc) -> None:
        """Broadcast event for every subscriber of event."""
        if subscriptions := self.subscriptions.get(event):
            for ws in subscriptions:
                try:
                    await ws.send_json(msg.model_dump())
                except Exception as e:
                    logger.debug("WS.SEND() exception! %s", e)

            await asyncio.sleep(0.01)

    @abstractmethod
    async def handle_msg(self, msg: BaseModel, ws: WebSocket, context: T) -> None:
        """Place to handle all msgs from jsonrpc."""

    async def read_msgs(self, ws: WebSocket) -> list[JsonRpc]:
        """Read websocket messages."""
        msg_raw = await ws.receive_json()
        msgs: list[JsonRpc] = []
        try:
            msgs = self.msgs_type(msgs=msg_raw).msgs  # type: ignore
            debug_msg = f"Receive msgs: {msgs}"
            logger.debug(debug_msg)
        except ValidationError:
            debug_msg = f"Receive raw json: {msg_raw}"
            logger.debug(debug_msg)

        return msgs

    @abstractmethod
    async def handle_disconnect(self, ws: WebSocket, context: T) -> None:
        """Handle the websocket disconnect."""

    def task_done_callback(self, task: asyncio.Task[None]) -> None:
        """Discard task after finish."""
        self.tasks.discard(task)

    @abstractmethod
    def create_bag(self) -> T:
        """Create websocket bag."""

    async def loop(self, ws: WebSocket) -> None:
        """Websocket message handler."""
        await self.connect(ws)
        debug_msg = "Websocket new connection."
        logger.debug(debug_msg)
        context = self.create_bag()
        while True:
            try:
                for msg_raw in await self.read_msgs(ws):
                    task: asyncio.Task[None] = asyncio.create_task(self.handle_msg(msg_raw, ws, context))
                    task.add_done_callback(self.task_done_callback)
                    self.tasks.add(task)
            except WebSocketDisconnect:
                await self.handle_disconnect(ws, context)
                await self.disconnect(ws)
                break
            except RuntimeError:
                await self.handle_disconnect(ws, context)
                await self.disconnect(ws)
                break
            except KeyboardInterrupt:
                await self.disconnect(ws)
                break
            except Exception:
                logger.exception("Error in handling websocket msg.")
                continue
