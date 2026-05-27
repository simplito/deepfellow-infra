# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""FFMPEG module."""

import asyncio
import os
import subprocess
from collections.abc import AsyncGenerator


def ffmpeg_command(
    args: list[str], input_data: bytes | None = None, stream_output: bool = False
) -> tuple[bytes | str, subprocess.CalledProcessError | None]:
    """Execute ffmpeg command with given arguments, optionally with stdin input and stdout output.

    Args:
        args: List of command arguments for ffmpeg
        input_data: Optional bytes to pipe to ffmpeg's stdin
        stream_output: If True, returns raw bytes from stdout; if False, returns decoded text

    Returns:
        Tuple of (output bytes/string, error if any)
    """
    try:
        # Constrain this to ffmpeg to permit security scanner to see that the command is safe
        cmd = ["ffmpeg", *args]
        result = subprocess.run(cmd, input=input_data, capture_output=True, env=os.environ.copy(), check=True)

        if stream_output:
            # Return raw bytes for audio data
            return result.stdout, None
        # Return decoded text for logging/errors
        return (result.stdout + result.stderr).decode("utf-8", errors="replace"), None

    except subprocess.CalledProcessError as e:
        error_text = (e.stdout + e.stderr).decode("utf-8", errors="replace")
        return error_text, e


async def ffmpeg_audio_convert_async_gen(source: AsyncGenerator[bytes], input_format: str, output_format: str) -> AsyncGenerator[bytes]:
    """Convert given source to output format."""
    if input_format == output_format:
        async for chunk in source:
            yield chunk
    else:
        format_options = {
            "opus": ["-f", "ogg", "-acodec", "libopus"],
            "mp3": ["-f", "mp3", "-acodec", "libmp3lame"],
            "aac": ["-f", "adts", "-acodec", "aac"],
            "flac": ["-f", "flac", "-acodec", "flac"],
            "wav": ["-f", "wav", "-acodec", "pcm_s16le"],
        }

        # Get format-specific options
        format_args = format_options.get(output_format, ["-f", output_format])

        # Build ffmpeg command
        command_args = [
            "-f",
            input_format,
            "-i",
            "pipe:0",  # Read from stdin
            "-vn",  # No video
            *format_args,
            "pipe:1",  # Write to stdout
        ]

        cmd = ["ffmpeg", *command_args]
        process = await asyncio.create_subprocess_exec(
            *cmd, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=os.environ.copy()
        )

        if not process.stdin:
            raise RuntimeError("No stdin in ffmpeg process")
        if not process.stdout:
            raise RuntimeError("No stdout in ffmpeg process")

        writer_task = asyncio.create_task(_stream_writer(source, process.stdin))

        async for chunk in _stream_reader(process.stdout):
            yield chunk

        await writer_task
        await process.wait()

        if process.returncode != 0:
            raise RuntimeError("Conversion failed")


async def _stream_writer(source: AsyncGenerator[bytes], stdin: asyncio.StreamWriter) -> None:
    async for chunk in source:
        stdin.write(chunk)
    stdin.close()


async def _stream_reader(stdout: asyncio.StreamReader) -> AsyncGenerator[bytes]:
    while True:
        data = await stdout.read(512)
        if not data:
            break
        yield data
