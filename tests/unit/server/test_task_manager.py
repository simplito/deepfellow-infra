# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import logging

import pytest

from server.task_manager import TaskManager


@pytest.fixture
def task_manager():
    return TaskManager()


@pytest.mark.asyncio
async def test_init_creates_empty_task_set(task_manager: TaskManager):
    assert task_manager.tasks == set()


@pytest.mark.asyncio
async def test_add_task_returns_task(task_manager: TaskManager):
    async def noop() -> None:
        pass

    task = task_manager.add_task(noop())

    assert isinstance(task, asyncio.Task)
    await task


@pytest.mark.asyncio
async def test_add_task_holds_reference_during_execution(task_manager: TaskManager):
    event = asyncio.Event()

    async def waiter() -> None:
        await event.wait()

    task = task_manager.add_task(waiter())

    assert task in task_manager.tasks
    event.set()
    await task


@pytest.mark.asyncio
async def test_add_task_removes_reference_after_completion(task_manager: TaskManager):
    async def noop() -> None:
        pass

    task = task_manager.add_task(noop())
    await task

    assert task not in task_manager.tasks


@pytest.mark.asyncio
async def test_add_task_safe_runs_coroutine(task_manager: TaskManager):
    result = []

    async def work() -> None:
        result.append(1)

    task = task_manager.add_task_safe(work())
    await task

    assert result == [1]


@pytest.mark.asyncio
async def test_add_task_safe_logs_exception_and_does_not_propagate(task_manager: TaskManager, caplog: pytest.LogCaptureFixture):
    async def failing() -> None:
        raise ValueError("boom")

    with caplog.at_level(logging.ERROR, logger="uvicorn.error"):
        task = task_manager.add_task_safe(failing(), err_msg="test error")
        await task

    assert any("test error" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_add_task_safe_includes_err_msg_in_log(task_manager: TaskManager, caplog: pytest.LogCaptureFixture):
    async def failing() -> None:
        raise RuntimeError("oops")

    with caplog.at_level(logging.ERROR, logger="uvicorn.error"):
        task = task_manager.add_task_safe(failing(), err_msg="custom context")
        await task

    assert any("custom context" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_add_task_safe_without_err_msg(task_manager: TaskManager):
    async def failing() -> None:
        raise Exception("silent")  # noqa: TRY002

    task = task_manager.add_task_safe(failing())
    await task  # should not raise
