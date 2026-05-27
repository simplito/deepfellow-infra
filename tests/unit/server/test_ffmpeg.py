# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

import subprocess
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from server.ffmpeg import (
    _stream_reader,  # pyright: ignore[reportPrivateUsage]
    _stream_writer,  # pyright: ignore[reportPrivateUsage]
    ffmpeg_audio_convert_async_gen,
    ffmpeg_command,
)


def test_ffmpeg_command_returns_decoded_output_on_success():
    mock_result = MagicMock()
    mock_result.stdout = b"output"
    mock_result.stderr = b""

    with patch("subprocess.run", return_value=mock_result) as mock_run:
        output, err = ffmpeg_command(["-version"])

    assert err is None
    assert "output" in output  # pyright: ignore[reportOperatorIssue]
    assert mock_run.call_args[0][0][0] == "ffmpeg"


def test_ffmpeg_command_returns_bytes_when_stream_output():
    mock_result = MagicMock()
    mock_result.stdout = b"\x00\x01\x02"
    mock_result.stderr = b""

    with patch("subprocess.run", return_value=mock_result):
        output, err = ffmpeg_command(["-i", "pipe:0"], stream_output=True)

    assert err is None
    assert output == b"\x00\x01\x02"


def test_ffmpeg_command_passes_input_data():
    mock_result = MagicMock()
    mock_result.stdout = b""
    mock_result.stderr = b""

    with patch("subprocess.run", return_value=mock_result) as mock_run:
        ffmpeg_command(["-i", "pipe:0", "pipe:1"], input_data=b"audio-data", stream_output=True)

    _, kwargs = mock_run.call_args
    assert kwargs["input"] == b"audio-data"


def test_ffmpeg_command_returns_error_text_on_failure():
    error = subprocess.CalledProcessError(1, "ffmpeg")
    error.stdout = b"stdout error"
    error.stderr = b" stderr error"

    with patch("subprocess.run", side_effect=error):
        output, err = ffmpeg_command(["-bad-arg"])

    assert err is error
    assert "stdout error" in output  # pyright: ignore[reportOperatorIssue]
    assert "stderr error" in output  # pyright: ignore[reportOperatorIssue]


def test_ffmpeg_command_sets_env_to_os_environ():
    mock_result = MagicMock()
    mock_result.stdout = b""
    mock_result.stderr = b""

    with patch("subprocess.run", return_value=mock_result) as mock_run:
        ffmpeg_command(["-version"])

    _, kwargs = mock_run.call_args
    assert kwargs["env"] is not None


def test_ffmpeg_command_combines_stdout_and_stderr_without_stream_output():
    mock_result = MagicMock()
    mock_result.stdout = b"out"
    mock_result.stderr = b"err"

    with patch("subprocess.run", return_value=mock_result):
        output, _err = ffmpeg_command(["-version"])

    assert "out" in output  # pyright: ignore[reportOperatorIssue]
    assert "err" in output  # pyright: ignore[reportOperatorIssue]


@pytest.mark.asyncio
async def test_ffmpeg_audio_convert_async_gen_passthrough_when_same_format():
    async def source():
        yield b"chunk1"
        yield b"chunk2"

    chunks = []

    async for chunk in ffmpeg_audio_convert_async_gen(source(), "mp3", "mp3"):
        chunks.append(chunk)

    assert chunks == [b"chunk1", b"chunk2"]


@pytest.mark.asyncio
async def test_ffmpeg_audio_convert_async_gen_converts_different_format():
    async def source():
        yield b"input-chunk"

    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_process.stdin = MagicMock()
    mock_process.stdin.write = MagicMock()
    mock_process.stdin.close = MagicMock()
    mock_process.stdout = MagicMock()
    mock_process.wait = AsyncMock()

    output_chunks = [b"out1", b"out2"]

    async def fake_stream_reader(stdout: MagicMock):
        for chunk in output_chunks:
            yield chunk

    with (
        patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=mock_process),
        patch("server.ffmpeg._stream_reader", side_effect=fake_stream_reader),
        patch("server.ffmpeg._stream_writer", new_callable=AsyncMock),
    ):
        chunks = []
        async for chunk in ffmpeg_audio_convert_async_gen(source(), "wav", "mp3"):
            chunks.append(chunk)

    assert chunks == output_chunks


@pytest.mark.asyncio
async def test_ffmpeg_audio_convert_async_gen_raises_on_nonzero_returncode():
    async def source():
        yield b"data"

    mock_process = MagicMock()
    mock_process.returncode = 1
    mock_process.stdin = MagicMock()
    mock_process.stdin.write = MagicMock()
    mock_process.stdin.close = MagicMock()
    mock_process.stdout = MagicMock()
    mock_process.wait = AsyncMock()

    async def fake_stream_reader(stdout: MagicMock):
        return
        yield  # make it an async generator

    with (
        patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=mock_process),
        patch("server.ffmpeg._stream_reader", side_effect=fake_stream_reader),
        patch("server.ffmpeg._stream_writer", new_callable=AsyncMock),
        pytest.raises(RuntimeError, match="Conversion failed"),
    ):
        async for _ in ffmpeg_audio_convert_async_gen(source(), "wav", "mp3"):
            pass


@pytest.mark.asyncio
async def test_ffmpeg_audio_convert_async_gen_raises_when_no_stdin():
    async def source():
        yield b"data"

    mock_process = MagicMock()
    mock_process.stdin = None
    mock_process.stdout = MagicMock()

    with (
        patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=mock_process),
        pytest.raises(RuntimeError, match="No stdin"),
    ):
        async for _ in ffmpeg_audio_convert_async_gen(source(), "wav", "mp3"):
            pass


@pytest.mark.asyncio
async def test_ffmpeg_audio_convert_async_gen_raises_when_no_stdout():
    async def source():
        yield b"data"

    mock_process = MagicMock()
    mock_process.stdin = MagicMock()
    mock_process.stdout = None

    with (
        patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=mock_process),
        pytest.raises(RuntimeError, match="No stdout"),
    ):
        async for _ in ffmpeg_audio_convert_async_gen(source(), "wav", "mp3"):
            pass


@pytest.mark.asyncio
async def test_stream_writer_writes_chunks_and_closes():
    async def source():
        yield b"chunk1"
        yield b"chunk2"

    stdin = MagicMock()
    stdin.write = MagicMock()
    stdin.close = MagicMock()

    await _stream_writer(source(), stdin)

    assert stdin.write.call_count == 2
    assert stdin.close.call_count == 1


@pytest.mark.asyncio
async def test_stream_writer_closes_even_with_empty_source():
    async def source():
        return
        yield

    stdin = MagicMock()
    stdin.write = MagicMock()
    stdin.close = MagicMock()

    await _stream_writer(source(), stdin)

    assert stdin.write.call_count == 0
    assert stdin.close.call_count == 1


@pytest.mark.asyncio
async def test_stream_reader_yields_chunks_until_empty():
    stdout = MagicMock()
    stdout.read = AsyncMock(side_effect=[b"abc", b"def", b""])

    chunks = []
    async for chunk in _stream_reader(stdout):
        chunks.append(chunk)

    assert chunks == [b"abc", b"def"]


@pytest.mark.asyncio
async def test_stream_reader_yields_nothing_when_immediate_eof():
    stdout = MagicMock()
    stdout.read = AsyncMock(return_value=b"")

    chunks = []
    async for chunk in _stream_reader(stdout):
        chunks.append(chunk)

    assert chunks == []
