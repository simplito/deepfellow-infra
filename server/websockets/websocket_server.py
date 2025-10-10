"""Websocket Manager."""

import asyncio
import logging
from abc import abstractmethod

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger("uvicorn.error")


class WebSocketContext[T]:
    def __init__(self, data: T):
        self.data = data

    @abstractmethod
    def send(self, data: str) -> None:
        """Send data to websocket."""


class WebSockerContextImpl[T](WebSocketContext[T]):
    def __init__(self, websocket: WebSocket, data: T):
        super().__init__(data)
        self.websocket = websocket
        self.send_queue = asyncio.Queue[str | None]()

        async def sender() -> None:
            """Send message from queue one by one, it is required because websocket lib does not support concurrency write."""
            while True:
                msg = await self.send_queue.get()
                if msg is None:
                    break
                try:
                    await websocket.send_text(msg)
                except Exception:
                    logger.exception("Error during sending message to websocket")

        self.send_task = asyncio.create_task(sender())

    def send(self, data: str) -> None:
        """Send data to websocket."""
        self.send_queue.put_nowait(data)

    async def close(self) -> None:
        """Stop sending queue."""
        await self.send_queue.put(None)
        await self.send_task


class WebSocketServer[T]:
    async def connect(self, websocket: WebSocket) -> None:
        """Run on connect to datalink websocket."""
        await websocket.accept()
        context = WebSockerContextImpl(websocket, self.create_bag())
        active_tasks = set[asyncio.Task[None]]()

        try:
            while True:
                msg = await websocket.receive_text()
                task = asyncio.create_task(self._process_message_safe(msg, context))
                active_tasks.add(task)
                task.add_done_callback(active_tasks.discard)
        except WebSocketDisconnect:
            self.handle_disconnect(context)
            logger.info("WebSocket disconnected")
        except Exception:
            self.handle_disconnect(context)
            logger.exception("Error during reading websocket")
        finally:
            await context.close()
            if active_tasks:
                await asyncio.gather(*active_tasks, return_exceptions=True)

    @abstractmethod
    def create_bag(self) -> T:
        """Create websocket bag."""

    async def _process_message_safe(self, msg: str, context: WebSocketContext[T]) -> None:
        try:
            await self.process_message(msg, context)
        except Exception:
            logger.exception("Error during processing message")

    @abstractmethod
    async def process_message(self, msg: str, context: WebSocketContext[T]) -> None:
        """Process message."""

    @abstractmethod
    def handle_disconnect(self, context: WebSocketContext[T]) -> None:
        """Handle disconnect."""
