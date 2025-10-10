"""WebSocket Client."""

import asyncio
import logging
from contextlib import suppress

import websockets

logger = logging.getLogger("uvicorn.error")


class WebSocketClient:
    def __init__(self, uri: str):
        self.uri = uri
        self.process_loop = True
        self.ws: tuple[websockets.ClientConnection, asyncio.Queue[str | None]] | None = None

    def send(self, msg: str) -> None:
        """Send message to websocket."""
        if not self.ws:
            raise RuntimeError("Cannot send websocket is disconnected")
        self.ws[1].put_nowait(msg)

    async def run(self) -> None:
        """Manage the websocket connection and send messages from the queue."""
        want_continue = await self.before_loop()
        if not want_continue:
            return
        await self.loop()
        await self.after_loop()

    async def before_loop(self) -> bool:
        """Return a value that indicates whether the loop should start or not."""
        return True

    async def loop(self) -> None:
        """Loop for external websocket."""
        while True:
            if not self.process_loop:
                logger.info("WS client loop exit")
                break
            try:
                async with websockets.connect(self.uri) as ws:
                    logger.info("WS client connected")
                    queue = asyncio.Queue[str | None]()
                    send_task = asyncio.create_task(self._sender(ws, queue))
                    receive_task = asyncio.create_task(self._receive_task(ws))
                    self.ws = (ws, queue)
                    try:
                        await self.on_start()
                        logger.info("WS client setup finished")
                        await receive_task
                    finally:
                        with suppress(Exception):
                            await ws.close()
                        self.ws = None
                        await queue.put(None)
                        await send_task
            except Exception:
                logger.exception("WS client disconnected")
            self.on_disconnect()
            await asyncio.sleep(10)

    async def _sender(self, ws: websockets.ClientConnection, queue: asyncio.Queue[str | None]) -> None:
        """Send message from queue one by one, it is required because websocket lib does not support concurrency write."""
        while True:
            message = await queue.get()
            if message is None:
                break
            try:
                await ws.send(message)
            except Exception:
                logger.exception("Error during sending message to websocket")

    async def _receive_task(self, ws: websockets.ClientConnection) -> None:
        async for msg_raw in ws:
            try:
                self.on_message(msg_raw)
            except Exception:
                logger.exception("Error during processing web socket message in client")

    async def on_start(self) -> None:
        """On start functions."""

    def on_message(self, msg: str | bytes) -> None:
        """Perform action on new message."""

    def on_disconnect(self) -> None:
        """Perform action on disconnect."""

    async def after_loop(self) -> None:
        """Function before loop."""  # noqa: D401
        return
