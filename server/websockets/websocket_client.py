# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""WebSocket Client."""

from __future__ import annotations

import asyncio
from contextlib import suppress

import websockets

from server.utils.logger import logger
from server.utils.sentry import FINGERPRINT_ATTR, REPORTED_ATTR, TAGS_ATTR

_RECONNECT_FINGERPRINT = ["mesh-websocket-disconnected"]


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

    async def stop(self) -> None:
        """Stop the reconnect loop and close the connection, so `run()` returns promptly."""
        self.process_loop = False
        if self.ws:
            try:
                await self.ws[0].close()
            except Exception:
                logger.exception("Error closing websocket connection during stop")

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

    def get_next_uri(self, failure_count: int) -> str:  # noqa: ARG002
        """Return the URI to connect to for the given failure count."""
        return self.uri

    async def loop(self) -> None:
        """Loop for external websocket."""
        failure_count = 0
        while True:
            if not self.process_loop:
                logger.info("WS client loop exit")
                break
            uri = self.get_next_uri(failure_count)
            try:
                async with websockets.connect(uri, ping_interval=5, ping_timeout=10) as ws:
                    logger.info("WS client connected")
                    queue = asyncio.Queue[str | None]()
                    send_task = asyncio.create_task(self._sender(ws, queue))
                    receive_task = asyncio.create_task(self._receive_task(ws))
                    self.ws = (ws, queue)
                    try:
                        if self.process_loop:
                            await self.on_start()
                            failure_count = 0
                            logger.info("WS client setup finished")
                            await receive_task
                        else:
                            # stop() ran while this connection attempt was in flight — self.ws was
                            # still None then, so it had nothing to close. Honor it now instead of
                            # proceeding to on_start()/receive_task, which could otherwise keep this
                            # "stopped" client connected indefinitely. Falling through here (instead
                            # of returning) still reaches on_disconnect() and the prompt-exit check
                            # below, like every other teardown path does.
                            logger.info("WS client loop exit (stop requested during connect)")
                    finally:
                        if not receive_task.done():
                            # Only reached via the stop-during-connect branch above, since the
                            # normal path already awaited receive_task to completion. Cancel it so it
                            # doesn't keep running (and log an unretrieved-exception warning) after we
                            # close `ws` below out from under it.
                            receive_task.cancel()
                            with suppress(asyncio.CancelledError):
                                await receive_task
                        try:
                            await ws.close()
                        except Exception:
                            logger.exception("Error closing websocket connection during disconnect")
                        self.ws = None
                        await queue.put(None)
                        await send_task
            except Exception:
                if failure_count == 0:
                    extra = {FINGERPRINT_ATTR: _RECONNECT_FINGERPRINT, TAGS_ATTR: {"operation": "mesh_connection"}}
                else:
                    extra = {REPORTED_ATTR: True}
                logger.exception("WS client disconnected", extra=extra)
                failure_count += 1
            self.on_disconnect()
            if not self.process_loop:
                break
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
