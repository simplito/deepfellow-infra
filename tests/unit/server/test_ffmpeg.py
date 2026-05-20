# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

import subprocess
import wave
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from server.ffmpeg import (
    _stream_reader,  # pyright: ignore[reportPrivateUsage]
    _stream_writer,  # pyright: ignore[reportPrivateUsage]
    audio_convert,
    audio_convert_stream,
    audio_to_wav,
    audio_to_wav_stream,
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


def test_audio_to_wav_stream_writes_converted_output():
    input_stream = BytesIO(b"fake-audio-data")
    output_stream = BytesIO()

    with patch("server.ffmpeg.ffmpeg_command", return_value=(b"converted-wav", None)):
        audio_to_wav_stream(input_stream, output_stream)

    assert output_stream.getvalue() == b"converted-wav"


def test_audio_to_wav_stream_raises_on_error():
    input_stream = BytesIO(b"bad-audio")
    output_stream = BytesIO()
    err = MagicMock(spec=subprocess.CalledProcessError)

    with (
        patch("server.ffmpeg.ffmpeg_command", return_value=("error text", err)),
        pytest.raises(RuntimeError, match="Conversion failed"),
    ):
        audio_to_wav_stream(input_stream, output_stream)


def test_audio_to_wav_stream_adds_input_format_hint():
    input_stream = BytesIO(b"ogg-data")
    output_stream = BytesIO()

    with patch("server.ffmpeg.ffmpeg_command", return_value=(b"wav-data", None)) as mock_cmd:
        audio_to_wav_stream(input_stream, output_stream, input_format="ogg")

    args = mock_cmd.call_args[0][0]
    assert "-f" in args
    assert "ogg" in args


def test_audio_to_wav_stream_no_format_hint_by_default():
    input_stream = BytesIO(b"data")
    output_stream = BytesIO()

    with patch("server.ffmpeg.ffmpeg_command", return_value=(b"wav-data", None)) as mock_cmd:
        audio_to_wav_stream(input_stream, output_stream)

    args = mock_cmd.call_args[0][0]
    # Without format hint, -f should not be the first argument
    assert args[0] == "-i"


def test_audio_to_wav_stream_flushes_output():
    input_stream = BytesIO(b"data")
    output_stream = MagicMock()

    with patch("server.ffmpeg.ffmpeg_command", return_value=(b"wav-data", None)):
        audio_to_wav_stream(input_stream, output_stream)

    assert output_stream.flush.call_count == 1


@pytest.mark.asyncio
async def test_audio_convert_stream_returns_stdout_on_success():
    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_process.communicate = AsyncMock(return_value=(b"converted-audio", b""))

    with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=mock_process):
        result = await audio_convert_stream(b"input-wav", "mp3")

    assert result == b"converted-audio"


@pytest.mark.asyncio
async def test_audio_convert_stream_raises_on_nonzero_returncode():
    mock_process = MagicMock()
    mock_process.returncode = 1
    mock_process.communicate = AsyncMock(return_value=(b"", b"some error"))

    with (
        patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=mock_process),
        pytest.raises(RuntimeError, match="Conversion failed"),
    ):
        await audio_convert_stream(b"bad-input", "mp3")


@pytest.mark.asyncio
async def test_audio_convert_stream_uses_known_format_options():
    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_process.communicate = AsyncMock(return_value=(b"audio", b""))

    with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=mock_process) as mock_exec:
        await audio_convert_stream(b"data", "opus")

    cmd = mock_exec.call_args[0]
    cmd_str = " ".join(cmd)
    assert "libopus" in cmd_str


@pytest.mark.asyncio
async def test_audio_convert_stream_fallback_format_for_unknown():
    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_process.communicate = AsyncMock(return_value=(b"audio", b""))

    with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=mock_process) as mock_exec:
        await audio_convert_stream(b"data", "custom-format")

    cmd = mock_exec.call_args[0]
    cmd_str = " ".join(cmd)
    assert "custom-format" in cmd_str


@pytest.mark.asyncio
async def test_audio_convert_stream_uses_input_format():
    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_process.communicate = AsyncMock(return_value=(b"audio", b""))

    with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=mock_process) as mock_exec:
        await audio_convert_stream(b"data", "mp3", input_format="ogg")

    cmd = mock_exec.call_args[0]
    cmd_str = " ".join(cmd)
    assert "ogg" in cmd_str


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


def test_audio_to_wav_renames_already_correct_wav(tmp_path: Path) -> None:
    src = tmp_path / "audio.wav"
    dst = tmp_path / "output.wav"

    # Create a proper mono 16kHz 16-bit WAV
    with wave.open(str(src), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 100)

    audio_to_wav(str(src), str(dst))

    assert dst.exists()
    assert not src.exists()


def test_audio_to_wav_converts_wav_with_wrong_specs(tmp_path: Path) -> None:
    src = tmp_path / "audio.wav"
    dst = tmp_path / "output.wav"

    # Create a WAV with wrong specs (stereo, 44100 Hz)
    with wave.open(str(src), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(b"\x00\x00" * 100)

    with patch("server.ffmpeg.ffmpeg_command", return_value=("", None)) as mock_cmd:
        audio_to_wav(str(src), str(dst))

    assert mock_cmd.call_count == 1


def test_audio_to_wav_converts_non_wav_file(tmp_path: Path) -> None:
    src = tmp_path / "audio.mp3"
    dst = tmp_path / "output.wav"
    src.write_bytes(b"fake-mp3-data")

    with patch("server.ffmpeg.ffmpeg_command", return_value=("", None)) as mock_cmd:
        audio_to_wav(str(src), str(dst))

    assert mock_cmd.call_count == 1
    args = mock_cmd.call_args[0][0]
    assert str(src) in args
    assert str(dst) in args


def test_audio_to_wav_raises_on_conversion_error(tmp_path: Path) -> None:
    src = tmp_path / "audio.mp3"
    dst = tmp_path / "output.wav"
    src.write_bytes(b"bad-data")

    err = MagicMock(spec=subprocess.CalledProcessError)
    with patch("server.ffmpeg.ffmpeg_command", return_value=("error", err)), pytest.raises(RuntimeError):
        audio_to_wav(str(src), str(dst))


def test_audio_to_wav_falls_through_when_wav_unreadable(tmp_path: Path) -> None:
    src = tmp_path / "corrupt.wav"
    dst = tmp_path / "output.wav"
    src.write_bytes(b"not-a-real-wav")

    with patch("server.ffmpeg.ffmpeg_command", return_value=("", None)) as mock_cmd:
        audio_to_wav(str(src), str(dst))

    assert mock_cmd.call_count == 1


def test_audio_convert_returns_src_for_wav_format(tmp_path: Path) -> None:
    src = tmp_path / "audio.wav"
    src.write_bytes(b"wav-data")

    result = audio_convert(str(src), "wav")

    assert result == str(src)


def test_audio_convert_converts_to_mp3(tmp_path: Path) -> None:
    src = tmp_path / "audio.wav"
    src.write_bytes(b"wav-data")
    expected_dst = str(src).replace(".wav", ".mp3")

    with patch("server.ffmpeg.ffmpeg_command", return_value=("", None)) as mock_cmd:
        result = audio_convert(str(src), "mp3")

    assert result == expected_dst
    assert mock_cmd.call_count == 1


def test_audio_convert_converts_to_opus(tmp_path: Path) -> None:
    src = tmp_path / "audio.wav"
    src.write_bytes(b"wav-data")
    expected_dst = str(src).replace(".wav", ".ogg")

    with patch("server.ffmpeg.ffmpeg_command", return_value=("", None)):
        result = audio_convert(str(src), "opus")

    assert result == expected_dst


def test_audio_convert_converts_to_aac(tmp_path: Path) -> None:
    src = tmp_path / "audio.wav"
    src.write_bytes(b"wav-data")
    expected_dst = str(src).replace(".wav", ".aac")

    with patch("server.ffmpeg.ffmpeg_command", return_value=("", None)):
        result = audio_convert(str(src), "aac")

    assert result == expected_dst


def test_audio_convert_converts_to_flac(tmp_path: Path) -> None:
    src = tmp_path / "audio.wav"
    src.write_bytes(b"wav-data")
    expected_dst = str(src).replace(".wav", ".flac")

    with patch("server.ffmpeg.ffmpeg_command", return_value=("", None)):
        result = audio_convert(str(src), "flac")

    assert result == expected_dst


def test_audio_convert_raises_on_error(tmp_path: Path) -> None:
    src = tmp_path / "audio.wav"
    src.write_bytes(b"wav-data")

    err = MagicMock(spec=subprocess.CalledProcessError)
    with patch("server.ffmpeg.ffmpeg_command", return_value=("error text", err)), pytest.raises(RuntimeError):
        audio_convert(str(src), "mp3")


def test_audio_convert_passes_minus_y_flag(tmp_path: Path) -> None:
    src = tmp_path / "audio.wav"
    src.write_bytes(b"wav-data")

    with patch("server.ffmpeg.ffmpeg_command", return_value=("", None)) as mock_cmd:
        audio_convert(str(src), "mp3")

    args = mock_cmd.call_args[0][0]
    assert "-y" in args
