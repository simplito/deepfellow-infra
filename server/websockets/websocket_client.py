"""WebSocket Client."""

import ast
import asyncio
from contextlib import suppress
from typing import Any, NoReturn

import websockets
from pydantic import BaseModel

from .models import WebsocketMsgs


class WebSocketClient:
    queue: asyncio.Queue[Any] = asyncio.Queue()
    uri: str
    ws: websockets.ClientConnection
    tasks: set[asyncio.Task[None]]
    msgs_types: type[WebsocketMsgs] = WebsocketMsgs

    def __init__(self, uri: str):
        self.uri = uri
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
