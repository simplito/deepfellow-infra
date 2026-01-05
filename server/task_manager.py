# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Task Manager."""

import asyncio
import logging
from collections.abc import Coroutine
from typing import Any

logger = logging.getLogger("uvicorn.error")


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
        """Create task in background hold reference to it and on error log this error."""

        async def func() -> None:
            try:
                await coroutine
            except Exception:
                msg = "Error during calling task " + err_msg
                logger.exception(msg)

        return self.add_task(func())
