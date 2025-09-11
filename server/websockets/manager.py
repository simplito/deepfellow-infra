"""Websocket Manager."""

import ast
import asyncio
import logging
from contextlib import suppress
from typing import Any, NoReturn

import websockets
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ValidationError
from starlette.datastructures import State

from .models import (
    JsonRpc,
    WebsocketMsgs,
)

logger = logging.getLogger("uvicorn.error")


class ExternalWebsocketManager:
    queue: asyncio.Queue[Any] = asyncio.Queue()
    uri: str
    ws: websockets.ClientConnection
    tasks: set[asyncio.Task[None]]
    state: State
    msgs_types: type[WebsocketMsgs] = WebsocketMsgs

    def __init__(self, uri: str, state: State):
        self.uri = uri
        self.state = state
        self.tasks = set()

    async def run(self) -> None:
        """Manage the websocket connection and send messages from the queue."""
        want_continue = await self.before_loop()
        if not want_continue:
            return
        await self.loop()
        await self.after_loop()

    async def before_loop(self) -> bool:
        """Function before loop.\

        If true on output it continue
        """  # noqa: D401
        return True

    async def loop(self) -> None:
        """Loop for external websocket."""
        while True:
            with suppress(Exception):
                async with websockets.connect(self.uri) as self.ws:
                    await self.on_start()
                    await asyncio.gather(*[self.send_loop(), self.get_loop()])
            await asyncio.sleep(10)

    async def on_start(self) -> None:
        """On start functions."""

    async def send_loop(self) -> NoReturn:
        """Loop for sending message."""
        while True:
            message = await self.queue.get()
            try:
                await self.ws.send(message)
            except websockets.exceptions.ConnectionClosed:
                await self.queue.put(message)
                raise

    async def get_loop(self) -> None:
        """Loop for getting message."""
        async for msg_raw in self.ws:
            try:
                msgs = self.convert_msg(msg_raw)
                for msg in msgs:
                    task: asyncio.Task[None] = asyncio.create_task(self.handle_msg(msg))
                    task.add_done_callback(self.task_done_callback)
                    self.tasks.add(task)
            except websockets.exceptions.ConnectionClosed:
                raise
            except Exception:
                continue

    def convert_msg(self, msg: str | bytes) -> list[BaseModel]:
        """Convert raw messages to pydantics."""
        msg_str: str = msg.decode() if isinstance(msg, bytes) else msg
        msgs = ast.literal_eval(msg_str)
        return self.msgs_types(msgs=msgs).msgs

    async def handle_msg(self, msg: BaseModel) -> None:
        """Place to handle all msgs from jsonrpc."""

    def task_done_callback(self, task: asyncio.Task[None]) -> None:
        """Discard task after finish."""
        self.tasks.discard(task)

    async def after_loop(self) -> None:
        """Function before loop."""  # noqa: D401
        return

    async def send_message(self, msg: BaseModel) -> None:
        """Add the message to the queue."""
        await self.before_send_msg(msg)
        await self.queue.put(msg.model_dump_json())

    async def before_send_msg(self, msg: BaseModel) -> None:  # noqa: ARG002
        """Function before send msg."""  # noqa: D401
        return


class InternalWebsocketManager:
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

    async def handle_msg(self, msg: BaseModel, ws: WebSocket, context: dict[str, Any]) -> None:  # noqa: ARG002
        """Place to handle all msgs from jsonrpc."""
        return

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

    async def handle_disconnect(self, ws: WebSocket, context: dict[str, Any]) -> None:  # noqa: ARG002
        """Function to hadle disconnect."""  # noqa: D401
        return

    def task_done_callback(self, task: asyncio.Task[None]) -> None:
        """Discard task after finish."""
        self.tasks.discard(task)

    async def loop(self, ws: WebSocket) -> None:
        """Websocket message handler."""
        await self.connect(ws)
        debug_msg = "Websocket new connection."
        logger.debug(debug_msg)
        context: dict[str, Any] = {}
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
