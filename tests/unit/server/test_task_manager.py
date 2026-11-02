# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

import asyncio
import logging
from unittest import mock

import pytest

from server.task_manager import TaskManager
from server.utils.exceptions import ApiError
from server.utils.sentry import is_reported


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

    with caplog.at_level(logging.ERROR, logger="uvicorn"):
        task = task_manager.add_task_safe(failing(), err_msg="test error")
        await task

    assert any("test error" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_add_task_safe_includes_err_msg_in_log(task_manager: TaskManager, caplog: pytest.LogCaptureFixture):
    async def failing() -> None:
        raise RuntimeError("oops")

    with caplog.at_level(logging.ERROR, logger="uvicorn"):
        task = task_manager.add_task_safe(failing(), err_msg="custom context")
        await task

    assert any("custom context" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_add_task_safe_without_err_msg(task_manager: TaskManager):
    async def failing() -> None:
        raise Exception("silent")  # noqa: TRY002

    task = task_manager.add_task_safe(failing())
    await task  # should not raise


@pytest.mark.asyncio
async def test_add_task_safe_reports_exception_to_sentry(task_manager: TaskManager):
    async def failing() -> None:
        raise RuntimeError("boom")

    with mock.patch("server.task_manager.capture_exception") as mock_capture:
        await task_manager.add_task_safe(failing(), "my.task")

    assert mock_capture.call_count == 1
    assert isinstance(mock_capture.call_args.args[0], RuntimeError)
    assert mock_capture.call_args.kwargs["tags"] == {"operation": "background_task", "task": "my.task"}
    assert is_reported(mock_capture.call_args.args[0])


@pytest.mark.asyncio
async def test_add_task_safe_does_not_report_expected_exception(task_manager: TaskManager):
    async def failing() -> None:
        raise ApiError("Invalid api key", 2)

    with mock.patch("server.task_manager.capture_exception") as mock_capture:
        await task_manager.add_task_safe(failing())

    assert mock_capture.call_count == 0
