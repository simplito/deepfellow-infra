"""FFMPEG module."""

import asyncio
import os
import subprocess
import wave
from collections.abc import AsyncGenerator
from typing import BinaryIO, cast


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


def audio_to_wav_stream(input_stream: BinaryIO, output_stream: BinaryIO, input_format: str | None = None) -> None:
    """Convert audio from stdin to WAV format on stdout.

    Args:
        input_stream: Binary input stream (e.g., sys.stdin.buffer)
        output_stream: Binary output stream (e.g., sys.stdout.buffer)
        input_format: Optional input format hint (e.g., 'mp3', 'ogg', etc.)

    Raises:
        Exception: If conversion fails
    """
    # Read all input data
    input_data = input_stream.read()

    # Build ffmpeg command for stdin/stdout streaming
    command_args = [
        "-i",
        "pipe:0",  # Read from stdin
        "-f",
        "wav",  # Output format
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        "pipe:1",  # Write to stdout
    ]

    # Add input format hint if provided
    if input_format:
        command_args = ["-f", input_format, *command_args]

    output_data, err = ffmpeg_command(command_args, input_data=input_data, stream_output=True)
    if err:
        raise RuntimeError("Conversion failed", output_data)

    # Write output to stream
    output_stream.write(cast("bytes", output_data))
    output_stream.flush()


async def audio_convert_stream(input_data: bytes, output_format: str, input_format: str = "wav") -> bytes:
    """Async version of audio_convert_stream that works with bytes directly."""
    # Map output formats to ffmpeg options
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

    stdout, stderr = await process.communicate(input=input_data)

    if process.returncode != 0:
        error_text = stderr.decode("utf-8", errors="replace")
        raise RuntimeError("Conversion failed", error_text)

    return stdout


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


# Keep the original file-based functions for compatibility
def audio_to_wav(src: str, dst: str) -> None:
    """Convert audio file to wav format for transcription.

    If the source is already a WAV file with the correct specifications
    (16-bit, mono, 16kHz), it will be moved directly to the destination.
    Otherwise, ffmpeg is used for conversion.

    Args:
        src: Source audio file path
        dst: Destination wav file path

    Raises:
        Exception: If conversion fails
    """
    # Check if source is already a WAV file with correct specs
    if src.endswith(".wav"):
        try:
            with wave.open(src, "rb") as wav_file:
                params = wav_file.getparams()
                # Check if WAV has correct specifications
                if (
                    params.sampwidth == 2  # 16-bit (2 bytes)
                    and params.nchannels == 1  # mono
                    and params.framerate == 16000
                ):  # 16kHz
                    os.rename(src, dst)  # noqa: PTH104
                    return
        except Exception:
            # If we can't read the WAV file, proceed with conversion
            pass

    # Convert using ffmpeg
    command_args = ["-i", src, "-f", "s16le", "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le", dst]

    out, err = ffmpeg_command(command_args)
    if err:
        raise RuntimeError("error", (err, out))


def audio_convert(src: str, format: str) -> str:
    """Convert generated wav file from TTS to other output formats.

    Args:
        src: Source WAV file path
        format: Target format (opus, mp3, aac, flac, or wav)

    Returns:
        Path to the converted file

    Raises:
        Exception: If conversion fails
    """
    # Compute file extension from format
    extension_map = {"opus": ".ogg", "mp3": ".mp3", "aac": ".aac", "flac": ".flac"}

    extension = extension_map.get(format, ".wav")

    # If target format is WAV, do nothing
    if extension == ".wav":
        return src

    # Create destination filename
    dst = src.replace(".wav", extension)

    command_args = ["-y", "-i", src, "-vn", dst]

    out, err = ffmpeg_command(command_args)
    if err:
        raise RuntimeError("error", (err, out))

    return dst
