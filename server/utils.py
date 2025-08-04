import asyncio
import base64
import io
import json
import re
import shlex
from typing import NamedTuple
from pathlib import Path
from aiohttp import ClientSession
from urllib.parse import quote
from PIL import Image
from fastapi import HTTPException


class NotImplementedException(Exception):
    pass

class CommandResult(NamedTuple):
    exitCode: int
    stdout: str
    stderr: str

class CommandResult2(NamedTuple):
    stdout: str
    stderr: str

class Utils:

    @staticmethod
    async def run_command(cmd: str) -> CommandResult:
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await proc.communicate()

        return CommandResult(
            exitCode= proc.returncode or 0,
            stdout = stdout.decode().strip(),
            stderr = stderr.decode().strip(),
        )

    @staticmethod
    async def run_command_for_success(cmd: str) -> CommandResult2:
        result = await Utils.run_command(cmd)
        if result.exitCode != 0:
            raise Exception(f"Invalid exit code {result.exitCode} for command {cmd}, stdout: {result.stdout}, stderr: {result.stderr}")
        return CommandResult2(stdout=result.stdout, stderr=result.stderr)

    @staticmethod
    async def wait_for_service(url: str, max_attempts: int = 30, delay: float = 1.0):
        """Wait for a service to become available"""
        for attempt in range(max_attempts):
            try:
                async with ClientSession() as session:
                    async with session.get(f"{url}") as resp:
                        if resp.status == 200:
                            return True
            except Exception:
                pass

            if attempt < max_attempts - 1:
                await asyncio.sleep(delay)

        return False

    @staticmethod
    def join_url(a: str, b: str) -> str:
        aa = a[:-1] if a.endswith("/") else a
        bb = b[1:]  if b.startswith("/") else b
        return f"{aa}/{bb}"

    @staticmethod
    def save_file(filename, file_content, location):
        file_path = Path(location) / filename
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(file_content)

    @staticmethod
    def sanitize_service_name(name: str) -> str:
        """Convert model name to valid Docker service name"""
        name = re.sub(r'[^a-zA-Z0-9_-]', '-', name)
        name = re.sub(r'-+', '-', name)
        name = name.strip('-')
        if name and not name[0].isalpha() and name[0] != '_':
            name = f"service-{name}"
        return name.lower()

    @staticmethod
    def shell_escape(arg: str) -> str:
        return shlex.quote(arg)

    @staticmethod
    def str_encode(text: str, safe: str = '') -> str:
        return quote(text, safe=safe)

    @staticmethod
    async def ensure_model_downloaded(context, model_url: str) -> tuple[Path, str]:
        """Download model if it's a URL, return (local_path, filename)"""

        if not model_url.startswith('https://'):
            # Already a local path
            local_path = Path(model_url)
            return local_path, local_path.name

        # Get filename from URL (last part after /)
        filename = model_url.split('/')[-1]
        local_path = context.get_model_dir() / filename

        if not local_path.exists():
            print(f"Downloading {filename}...")

            # Ensure the models directory exists
            local_path.parent.mkdir(parents=True, exist_ok=True)

            # Download with and error handling
            try:
                # Use curl instead of wget for better macOS compatibility
                result = await Utils.run_command(f"curl -L '{model_url}' -o '{local_path}'")

                if result.exitCode != 0:
                    # Clean up partial download
                    if local_path.exists():
                        local_path.unlink()
                    raise Exception(f"Download failed with exit code {result.exitCode}: {result.stderr}")

                # Validate the downloaded file
                if not local_path.exists() or local_path.stat().st_size == 0:
                    if local_path.exists():
                        local_path.unlink()
                    raise Exception(f"Downloaded file is empty or missing: {local_path}")

                print(f"Successfully downloaded {filename} ({local_path.stat().st_size} bytes)")

            except Exception as e:
                # Clean up any partial download
                if local_path.exists():
                    local_path.unlink()
                raise Exception(f"Failed to download {filename}: {str(e)}")

        return local_path, filename
