import os
import subprocess
import wave
import sys
import asyncio
from typing import Tuple, Optional, BinaryIO
from collections.abc import AsyncGenerator


def ffmpeg_command(args: list[str], input_data: bytes = None, stream_output: bool = False) -> Tuple[bytes | str, subprocess.CalledProcessError | None]:
    """
    Execute ffmpeg command with given arguments, optionally with stdin input and stdout output.
    
    Args:
        args: List of command arguments for ffmpeg
        input_data: Optional bytes to pipe to ffmpeg's stdin
        stream_output: If True, returns raw bytes from stdout; if False, returns decoded text
        
    Returns:
        Tuple of (output bytes/string, error if any)
    """
    try:
        # Constrain this to ffmpeg to permit security scanner to see that the command is safe
        cmd = ["ffmpeg"] + args
        result = subprocess.run(
            cmd,
            input=input_data,
            capture_output=True,
            env=os.environ.copy(),
            check=True
        )
        
        if stream_output:
            # Return raw bytes for audio data
            return result.stdout, None
        else:
            # Return decoded text for logging/errors
            return (result.stdout + result.stderr).decode('utf-8', errors='replace'), None
            
    except subprocess.CalledProcessError as e:
        error_text = (e.stdout + e.stderr).decode('utf-8', errors='replace')
        return error_text, e


def audio_to_wav_stream(input_stream: BinaryIO, output_stream: BinaryIO, input_format: str = None) -> None:
    """
    Convert audio from stdin to WAV format on stdout.
    
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
        "-i", "pipe:0",  # Read from stdin
        "-f", "wav",     # Output format
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "pipe:1"  # Write to stdout
    ]
    
    # Add input format hint if provided
    if input_format:
        command_args = ["-f", input_format] + command_args
    
    output_data, err = ffmpeg_command(command_args, input_data=input_data, stream_output=True)
    if err:
        raise Exception(f"Conversion failed: {output_data}")
    
    # Write output to stream
    output_stream.write(output_data)
    output_stream.flush()

async def audio_convert_stream(input_data: bytes, output_format: str, input_format: str = "wav") -> bytes:
    """
    Async version of audio_convert_stream that works with bytes directly.
    """
    # Map output formats to ffmpeg options
    format_options = {
        "opus": ["-f", "ogg", "-acodec", "libopus"],
        "mp3": ["-f", "mp3", "-acodec", "libmp3lame"],
        "aac": ["-f", "adts", "-acodec", "aac"],
        "flac": ["-f", "flac", "-acodec", "flac"],
        "wav": ["-f", "wav", "-acodec", "pcm_s16le"]
    }
    
    # Get format-specific options
    format_args = format_options.get(output_format, ["-f", output_format])
    
    # Build ffmpeg command
    command_args = [
        "-f", input_format,
        "-i", "pipe:0",  # Read from stdin
        "-vn"  # No video
    ] + format_args + ["pipe:1"]  # Write to stdout
    
    cmd = ["ffmpeg"] + command_args
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=os.environ.copy()
    )
    
    stdout, stderr = await process.communicate(input=input_data)
    
    if process.returncode != 0:
        error_text = stderr.decode('utf-8', errors='replace')
        raise Exception(f"Conversion failed: {error_text}")
    
    return stdout


async def ffmpeg_audio_convert_async_gen(source, input_format: str, output_format: str) -> AsyncGenerator[bytes]:
    if input_format == output_format:
        async for chunk in source:
            yield chunk
    else:
        format_options = {
            "opus": ["-f", "ogg", "-acodec", "libopus"],
            "mp3": ["-f", "mp3", "-acodec", "libmp3lame"],
            "aac": ["-f", "adts", "-acodec", "aac"],
            "flac": ["-f", "flac", "-acodec", "flac"],
            "wav": ["-f", "wav", "-acodec", "pcm_s16le"]
        }
        
        # Get format-specific options
        format_args = format_options.get(output_format, ["-f", output_format])
        
        # Build ffmpeg command
        command_args = [
            "-f", input_format,
            "-i", "pipe:0",  # Read from stdin
            "-vn"  # No video
        ] + format_args + ["pipe:1"]  # Write to stdout
        
        cmd = ["ffmpeg"] + command_args
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=os.environ.copy()
        )
        
        writer_task = asyncio.create_task(_stream_writer(source, process))
        
        async for chunk in _stream_reader(process):
            yield chunk
        
        await writer_task
        await process.wait()
        
        if process.returncode != 0:
            raise Exception(f"Conversion failed")

async def _stream_writer(source, proc):
    async for chunk in source:
        proc.stdin.write(chunk)
    proc.stdin.close()

async def _stream_reader(proc):
    while True:
        data = await proc.stdout.read(512)
        if not data:
            break
        yield data

# Keep the original file-based functions for compatibility
def audio_to_wav(src: str, dst: str) -> None:
    """
    Convert audio file to wav format for transcription.
    
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
            with wave.open(src, 'rb') as wav_file:
                params = wav_file.getparams()
                # Check if WAV has correct specifications
                if (params.sampwidth == 2 and  # 16-bit (2 bytes)
                    params.nchannels == 1 and   # mono
                    params.framerate == 16000): # 16kHz
                    os.rename(src, dst)
                    return
        except Exception:
            # If we can't read the WAV file, proceed with conversion
            pass
    
    # Convert using ffmpeg
    command_args = [
        "-i", src,
        "-f", "s16le",
        "-ar", "16000",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        dst
    ]
    
    out, err = ffmpeg_command(command_args)
    if err:
        raise Exception(f"error: {err} out: {out}")


def audio_convert(src: str, format: str) -> str:
    """
    Convert generated wav file from TTS to other output formats.
    
    Args:
        src: Source WAV file path
        format: Target format (opus, mp3, aac, flac, or wav)
        
    Returns:
        Path to the converted file
        
    Raises:
        Exception: If conversion fails
    """
    # Compute file extension from format
    extension_map = {
        "opus": ".ogg",
        "mp3": ".mp3",
        "aac": ".aac",
        "flac": ".flac"
    }
    
    extension = extension_map.get(format, ".wav")
    
    # If target format is WAV, do nothing
    if extension == ".wav":
        return src
    
    # Create destination filename
    dst = src.replace(".wav", extension)
    
    command_args = ["-y", "-i", src, "-vn", dst]
    
    out, err = ffmpeg_command(command_args)
    if err:
        raise Exception(f"error: {err} out: {out}")
    
    return dst


# Example usage functions
def convert_stdin_to_wav():
    """Example: Convert audio from stdin to WAV on stdout"""
    audio_to_wav_stream(sys.stdin.buffer, sys.stdout.buffer)


def convert_stdin_to_format(output_format: str, input_format: str = "wav"):
    """Example: Convert audio from stdin to specified format on stdout"""
    audio_convert_stream(sys.stdin.buffer, sys.stdout.buffer, output_format, input_format)


if __name__ == "__main__":
    # Example command-line usage
    if len(sys.argv) > 1:
        if sys.argv[1] == "to-wav":
            # Usage: python audio_utils.py to-wav < input.mp3 > output.wav
            convert_stdin_to_wav()
        elif sys.argv[1] == "convert" and len(sys.argv) > 2:
            # Usage: python audio_utils.py convert mp3 < input.wav > output.mp3
            output_format = sys.argv[2]
            input_format = sys.argv[3] if len(sys.argv) > 3 else "wav"
            convert_stdin_to_format(output_format, input_format)
        else:
            print("Usage:", file=sys.stderr)
            print("  python audio_utils.py to-wav < input.mp3 > output.wav", file=sys.stderr)
            print("  python audio_utils.py convert <format> [input_format] < input > output", file=sys.stderr)
            print("  Formats: opus, mp3, aac, flac, wav", file=sys.stderr)