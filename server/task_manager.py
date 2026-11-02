# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Task Manager."""

import asyncio
import uuid
from collections.abc import Coroutine
from typing import Any

from server.utils.logger import logger
from server.utils.sentry import REPORTED_ATTR, capture_exception, is_expected_error, mark_reported


class TaskManager:
    def __init__(self):
        self.tasks: set[asyncio.Task[Any]] = set()

    def add_task(self, coroutine: Coroutine[Any, Any, Any]) -> asyncio.Task[Any]:
        """Create task in background and hold reference to it."""
        task = asyncio.create_task(coroutine)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return task

    def add_task_safe(self, coroutine: Coroutine[Any, Any, Any], err_msg: str = "") -> asyncio.Task[Any]:
        """Create task in background hold reference to it and on error log this error and report it to Sentry."""

        async def func() -> None:
            try:
                await coroutine
            except Exception as exc:
                msg = "Error during calling task " + err_msg
                logger.exception(msg, extra={REPORTED_ATTR: True})
                if not is_expected_error(exc):
                    capture_exception(exc, error_id=str(uuid.uuid4()), tags={"operation": "background_task", "task": err_msg})
                mark_reported(exc)

        return self.add_task(func())
