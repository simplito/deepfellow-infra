# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from server.utils.core import (
    DownloadedPacket,
    HttpClientError,
    HttpResponse,
    OneTimeKey,
    PreDownloadPacket,
    PromiseWithProgress,
    Stream,
    StreamChunk,
    SuccessDownloadPacket,
    Utils,
    convert_promise_with_progress_to_fastapi_response,
    convert_size_to_bytes,
    download_file,
    fetch_from,
    get_cpu_architecture,
    get_os,
    make_http_request,
    stream_fetch_from,
)


@pytest.mark.asyncio
async def test_run_command_success():
    result = await Utils.run_command(["echo", "hello"])

    assert result.exit_code == 0
    assert result.stdout == "hello"


@pytest.mark.asyncio
async def test_run_command_nonzero_exit():
    result = await Utils.run_command(["false"])

    assert result.exit_code == 1


@pytest.mark.asyncio
async def test_run_command_for_success_raises_on_nonzero():
    with pytest.raises(RuntimeError):
        await Utils.run_command_for_success(["sh", "-c", "exit 2"])


@pytest.mark.asyncio
async def test_run_command_for_success_returns_on_zero():
    result = await Utils.run_command_for_success(["echo", "ok"])

    assert result.stdout == "ok"


@pytest.mark.asyncio
async def test_read_file(tmp_path: Path):
    f = tmp_path / "hello.txt"
    f.write_text("content123")

    result = await Utils.read_file(f)

    assert result == "content123"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("hello world", "hello-world"),
        ("--leading", "leading"),
        ("123start", "service-123start"),
        ("_start", "_start"),
        ("UPPER", "upper"),
        ("a!!b", "a-b"),
        ("a--b", "a-b"),
    ],
)
def test_sanitize_service_name(name: str, expected: str):
    assert Utils.sanitize_service_name(name) == expected


@pytest.mark.asyncio
async def test_ensure_model_downloaded_local_path(tmp_path: Path):
    local_file = tmp_path / "model.bin"
    local_file.write_bytes(b"data")

    packets = []
    async for packet in Utils.ensure_model_downloaded(str(local_file), model_dir=tmp_path, temp_dir=tmp_path):
        packets.append(packet)

    assert len(packets) == 1
    assert packets[0].local_path == local_file  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_ensure_model_downloaded_url_already_exists(tmp_path: Path):
    filename = "model.bin"
    (tmp_path / filename).write_bytes(b"existing")

    packets = []
    async for packet in Utils.ensure_model_downloaded("https://example.com/model.bin", model_dir=tmp_path, temp_dir=tmp_path):
        packets.append(packet)

    assert len(packets) == 1
    assert packets[0].local_path == tmp_path / filename  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_ensure_model_downloaded_url_downloads(tmp_path: Path):
    async def fake_download(url: str, path: Path, headers: dict[str, str]):
        path.write_bytes(b"downloaded")
        yield PreDownloadPacket(10)
        yield DownloadedPacket(10)

    with patch("server.utils.core.download_file", side_effect=fake_download):
        packets = []

        async for packet in Utils.ensure_model_downloaded(
            "https://example.com/new_model.bin",
            model_dir=tmp_path,
            temp_dir=tmp_path / "tmp",
        ):
            packets.append(packet)

    success = [p for p in packets if isinstance(p, SuccessDownloadPacket)]
    assert len(success) == 1
    assert success[0].local_path.name == "new_model.bin"


@pytest.mark.asyncio
async def test_ensure_model_downloaded_cleans_up_on_error(tmp_path: Path):
    async def fake_download(url: str, path: Path, headers: dict[str, str]):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"partial")
        yield MagicMock()
        raise RuntimeError("network error")

    with patch("server.utils.core.download_file", side_effect=fake_download), pytest.raises(RuntimeError, match="network error"):
        async for _ in Utils.ensure_model_downloaded(
            "https://example.com/fail_model.bin",
            model_dir=tmp_path,
            temp_dir=tmp_path / "tmp",
        ):
            pass


@pytest.mark.asyncio
async def test_ensure_model_downloaded_error_without_temp_file(tmp_path: Path):
    async def fake_download(url: str, path: Path, headers: dict[str, str]):
        raise RuntimeError("immediate error")
        yield  # required to make this an async generator

    with patch("server.utils.core.download_file", side_effect=fake_download), pytest.raises(RuntimeError, match="immediate error"):
        async for _ in Utils.ensure_model_downloaded(
            "https://example.com/fail.bin",
            model_dir=tmp_path,
            temp_dir=tmp_path / "tmp",
        ):
            pass


@pytest.mark.asyncio
async def test_ensure_model_downloaded_url_with_explicit_headers(tmp_path: Path):
    filename = "model.bin"
    (tmp_path / filename).write_bytes(b"existing")

    packets = []
    async for packet in Utils.ensure_model_downloaded(
        "https://example.com/model.bin",
        model_dir=tmp_path,
        temp_dir=tmp_path,
        headers={"Authorization": "Bearer token"},
    ):
        packets.append(packet)

    assert len(packets) == 1
    assert packets[0].local_path == tmp_path / filename  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_ensure_model_downloaded_raises_when_temp_file_missing(tmp_path: Path):
    async def fake_download(url: str, path: Path, headers: dict[str, str]):
        # Do NOT create the file - simulates a download that produces no output
        yield MagicMock()

    with (
        patch("server.utils.core.download_file", side_effect=fake_download),
        pytest.raises(RuntimeError, match="Downloaded file is missing"),
    ):
        async for _ in Utils.ensure_model_downloaded(
            "https://example.com/missing.bin",
            model_dir=tmp_path,
            temp_dir=tmp_path / "tmp",
        ):
            pass


@pytest.mark.asyncio
async def test_ensure_model_downloaded_raises_when_temp_file_empty(tmp_path: Path):
    async def fake_download(url: str, path: Path, headers: dict[str, str]):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")  # Create empty file
        yield MagicMock()

    with patch("server.utils.core.download_file", side_effect=fake_download), pytest.raises(RuntimeError, match="Downloaded file is empty"):
        async for _ in Utils.ensure_model_downloaded(
            "https://example.com/empty.bin",
            model_dir=tmp_path,
            temp_dir=tmp_path / "tmp",
        ):
            pass


@pytest.mark.parametrize(
    ("url", "param", "value", "expected_contains"),
    [
        ("http://x.com/path", "key", "val", "key=val"),
        ("http://x.com/path?key=existing", "key", "val", "key=existing"),
    ],
)
def test_add_url_parameter_if_missing(url: str, param: str, value: str, expected_contains: str):
    assert expected_contains in Utils.add_url_parameter_if_missing(url, param, value)


@pytest.mark.parametrize(
    ("url", "param", "value"),
    [
        ("http://x.com/path", "key", ""),
    ],
)
def test_add_url_parameter_if_missing_unchanged(url: str, param: str, value: str):
    assert Utils.add_url_parameter_if_missing(url, param, value) == url


@pytest.mark.asyncio
async def test_download_file_raises_on_non_200(tmp_path: Path):
    mock_resp = AsyncMock()
    mock_resp.status = 404
    mock_resp.text = AsyncMock(return_value="not found")
    mock_resp.headers = {}
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)
    mock_session = AsyncMock()
    mock_session.get = MagicMock(return_value=mock_resp)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with patch("server.utils.core.aiohttp.ClientSession", return_value=mock_session), pytest.raises(HttpClientError) as exc_info:
        async for _ in download_file("http://example.com/file", tmp_path / "f", {}):
            pass

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_download_file_yields_packets(tmp_path: Path):
    chunk = b"hello world"

    async def fake_iter_any():
        yield chunk

    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.headers = {"Content-Length": str(len(chunk))}
    mock_resp.content.iter_any = fake_iter_any
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)
    mock_session = AsyncMock()
    mock_session.get = MagicMock(return_value=mock_resp)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    packets = []
    with patch("server.utils.core.aiohttp.ClientSession", return_value=mock_session):
        async for packet in download_file("http://example.com/file", tmp_path / "f", {}):
            packets.append(packet)

    assert isinstance(packets[0], PreDownloadPacket)
    assert packets[0].file_bytes_size == len(chunk)
    assert isinstance(packets[1], DownloadedPacket)
    assert packets[1].downloaded_bytes_size == len(chunk)


@pytest.mark.asyncio
async def test_fetch_from():
    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.text = AsyncMock(return_value='{"ok": true}')
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)
    mock_session = AsyncMock()
    mock_session.request = MagicMock(return_value=mock_resp)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with patch("server.utils.core.aiohttp.ClientSession", return_value=mock_session):
        result = await fetch_from("http://example.com/api")

    assert result.status_code == 200
    assert result.data == '{"ok": true}'


@pytest.mark.asyncio
async def test_stream_fetch_from():
    async def fake_iter_chunks():
        yield (b"chunk1", False)
        yield (b"chunk2", False)

    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.content.iter_chunks = fake_iter_chunks
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock()
    mock_session.request = MagicMock(return_value=mock_resp)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    results = []
    with patch("server.utils.core.aiohttp.ClientSession", return_value=mock_session):
        async for item in stream_fetch_from("http://example.com/stream"):
            results.append(item)

    assert len(results) == 2
    assert results[0].data == "chunk1"
    assert results[1].data == "chunk2"


@pytest.mark.parametrize(
    ("size_str", "expected"),
    [
        ("1MB", 1_000_000),
        ("1.5GB", 1_500_000_000),
        ("512KB", 512_000),
        ("1TB", 1_000_000_000_000),
        ("10B", 10),
        ("1K", 1000),
        ("badXB", None),
        ("MB", None),
        ("notanumber MB", None),
    ],
)
def test_convert_size_to_bytes(size_str: str, expected: int | None):
    assert convert_size_to_bytes(size_str) == expected


@pytest.mark.asyncio
async def test_stream_emit_after_close_is_noop():
    stream: Stream[int] = Stream()
    stream.close()

    stream.emit(42)  # should not raise

    assert stream._history == []  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_stream_emit_notifies_subscribers():
    stream: Stream[int] = Stream()

    collected: list[int] = []

    async def consume():
        async for item in stream.as_generator():
            collected.append(item)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0)
    stream.emit(1)

    stream.emit(2)

    stream.close()
    await task
    assert collected == [1, 2]


@pytest.mark.asyncio
async def test_stream_as_generator_replays_history():
    stream: Stream[str] = Stream()
    stream.emit("a")
    stream.emit("b")
    stream.close()

    collected = []
    async for item in stream.as_generator():
        collected.append(item)

    assert collected == ["a", "b"]


@pytest.mark.asyncio
async def test_convert_promise_no_stream():
    class MyModel(BaseModel):
        value: int

    promise: PromiseWithProgress[MyModel, StreamChunk] = PromiseWithProgress(value=MyModel(value=42))

    response = await convert_promise_with_progress_to_fastapi_response(promise)

    assert isinstance(response, JSONResponse)


@pytest.mark.asyncio
async def test_convert_promise_with_stream_ok():
    class MyModel(BaseModel):
        value: int

    async def work(stream: Stream[StreamChunk]) -> MyModel:
        chunk: StreamChunk = {"type": "progress", "stage": "install", "value": 0.5, "data": {}}
        stream.emit(chunk)
        return MyModel(value=7)

    promise: PromiseWithProgress[MyModel, StreamChunk] = PromiseWithProgress(func=work)

    response = await convert_promise_with_progress_to_fastapi_response(promise)

    assert isinstance(response, StreamingResponse)

    chunks = []
    async for chunk in response.body_iterator:  # type: ignore[attr-defined]
        chunks.append(chunk)

    assert any("finish" in c for c in chunks)
    assert any("progress" in c for c in chunks)


@pytest.mark.asyncio
async def test_convert_promise_with_stream_error():
    class MyModel(BaseModel):
        value: int

    async def failing_work(stream: Stream[StreamChunk]) -> MyModel:
        raise HTTPException(500, "internal error")

    promise: PromiseWithProgress[MyModel, StreamChunk] = PromiseWithProgress(func=failing_work)

    response = await convert_promise_with_progress_to_fastapi_response(promise)

    assert isinstance(response, StreamingResponse)

    chunks = []
    async for chunk in response.body_iterator:  # type: ignore[attr-defined]
        chunks.append(chunk)

    assert any('"error"' in c for c in chunks)


@pytest.mark.asyncio
async def test_convert_promise_stream_completes_cleanly_when_cancelled():
    class MyModel(BaseModel):
        value: int

    started = asyncio.Event()
    release = asyncio.Event()

    async def work(_stream: Stream[StreamChunk]) -> MyModel:
        started.set()
        await release.wait()
        return MyModel(value=1)

    promise: PromiseWithProgress[MyModel, StreamChunk] = PromiseWithProgress(func=work)
    response = await convert_promise_with_progress_to_fastapi_response(promise)
    assert isinstance(response, StreamingResponse)

    await started.wait()
    promise.cancel()

    chunks = []
    async for chunk in response.body_iterator:  # type: ignore[attr-defined]
        chunks.append(chunk)

    assert not any('"error"' in c for c in chunks)


@pytest.mark.asyncio
async def test_cancel_propagates_through_chain():
    class MyModel(BaseModel):
        value: int

    started = asyncio.Event()
    release = asyncio.Event()

    async def work(_stream: Stream[StreamChunk]) -> MyModel:
        started.set()
        await release.wait()
        return MyModel(value=1)

    async def step(model: MyModel) -> MyModel:
        return MyModel(value=model.value + 1)

    parent: PromiseWithProgress[MyModel, StreamChunk] = PromiseWithProgress(func=work)
    child = parent.next(step, lambda _e: None)

    await started.wait()

    parent.cancel()
    await asyncio.gather(*parent.tasks(), return_exceptions=True)

    assert parent.task.done()
    assert child.task.done()
    assert parent._future.cancelled()  # pyright: ignore[reportPrivateUsage]
    assert child._future.cancelled()  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(asyncio.CancelledError):
        await child.wait()


@pytest.mark.parametrize(
    ("machine", "expected"),
    [
        ("x86_64", "amd64"),
        ("AMD64", "amd64"),
        ("aarch64", "arm64"),
        ("arm64", "arm64"),
        ("armv7l", "arm"),
        ("ppc64le", "ppc64le"),
        ("s390x", "s390x"),
        ("riscv64", "unknown"),
    ],
)
def test_get_cpu_architecture(machine: str, expected: str):
    with patch("server.utils.core.platform.machine", return_value=machine):
        assert get_cpu_architecture() == expected


def test_get_os():
    with patch("server.utils.core.platform.system", return_value="Linux"):
        assert get_os() == "linux"

    with patch("server.utils.core.platform.system", return_value="Darwin"):
        assert get_os() == "darwin"


@pytest.mark.asyncio
async def test_http_response_as_streaming_response():
    mock_client_response = MagicMock()
    mock_client_response.content_type = "application/json"
    mock_client_response.status = 200
    mock_client_response.headers = {"Content-Type": "application/json", "X-Custom": "value"}

    async def gen():
        yield b"body"

    http_response = HttpResponse(response=mock_client_response, content=gen())
    streaming = http_response.as_streaming_response(allowed_response_headers=["Content-Type"])

    assert isinstance(streaming, StreamingResponse)
    assert streaming.status_code == 200
    assert streaming.headers.get("content-type", "").startswith("application/json")
    assert "x-custom" not in streaming.headers


@pytest.mark.asyncio
async def test_make_http_request_success():
    async def fake_iter_chunks():
        yield (b"response body", False)

    mock_resp = MagicMock()
    mock_resp.content_type = "text/plain"
    mock_resp.status = 200
    mock_resp.headers = {}
    mock_resp.content.iter_chunks = fake_iter_chunks
    mock_resp.release = AsyncMock()

    mock_session = MagicMock()
    mock_session.request = AsyncMock(return_value=mock_resp)
    mock_session.close = AsyncMock()

    with patch("server.utils.core.ClientSession", return_value=mock_session):
        result = await make_http_request("http://example.com/")

    assert isinstance(result, HttpResponse)

    chunks = []
    async for chunk in result.content:
        chunks.append(chunk)

    assert b"response body" in chunks


@pytest.mark.asyncio
async def test_make_http_request_skips_falsy_chunks():
    """iter_chunks yields a falsy (empty) chunk — covers the False branch of 'if chunk' (559->558)."""

    async def fake_iter_chunks():
        yield ()  # falsy — should be skipped
        yield (b"real data", False)

    mock_resp = MagicMock()
    mock_resp.content_type = "text/plain"
    mock_resp.status = 200
    mock_resp.headers = {}
    mock_resp.content.iter_chunks = fake_iter_chunks
    mock_resp.release = AsyncMock()

    mock_session = MagicMock()
    mock_session.request = AsyncMock(return_value=mock_resp)
    mock_session.close = AsyncMock()

    with patch("server.utils.core.ClientSession", return_value=mock_session):
        result = await make_http_request("http://example.com/")

    chunks = []
    async for chunk in result.content:
        chunks.append(chunk)

    assert b"real data" in chunks
    assert () not in chunks


@pytest.mark.asyncio
async def test_make_http_request_raises_and_cleans_up():
    mock_session = MagicMock()
    mock_session.request = AsyncMock(side_effect=ConnectionError("refused"))
    mock_session.close = AsyncMock()

    with patch("server.utils.core.ClientSession", return_value=mock_session), pytest.raises(ConnectionError):
        await make_http_request("http://example.com/")

    mock_session.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_make_http_request_releases_response_when_exception_after_response_set():
    mock_resp = MagicMock()
    mock_resp.release = AsyncMock()

    mock_session = MagicMock()
    mock_session.request = AsyncMock(return_value=mock_resp)
    mock_session.close = AsyncMock()

    with (
        patch("server.utils.core.ClientSession", return_value=mock_session),
        patch("server.utils.core.HttpResponse", side_effect=RuntimeError("constructor error")),
        pytest.raises(RuntimeError, match="constructor error"),
    ):
        await make_http_request("http://example.com/")

    mock_resp.release.assert_awaited_once()
    mock_session.close.assert_awaited_once()


def test_one_time_key_valid():
    otk = OneTimeKey()
    key = otk.key
    assert otk.check(key) is True
    assert otk._key is None  # type: ignore[attr-defined]


def test_one_time_key_invalid():
    otk = OneTimeKey()
    _ = otk.key
    assert otk.check("wrong-key") is False


def test_one_time_key_consumed_after_correct_check():
    otk = OneTimeKey()
    key = otk.key
    otk.check(key)
    assert otk.check(key) is False


def test_one_time_key_generates_new_key_each_time():
    otk = OneTimeKey()
    key1 = otk.key
    key2 = otk.key
    assert key1 != key2
