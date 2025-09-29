"""Utils module."""

import asyncio
import re
import shlex
from pathlib import Path
from typing import NamedTuple
from urllib.parse import quote

import aiofiles
import aiohttp
from aiohttp import ClientSession, ClientTimeout
from pydantic import BaseModel

from server.models.common import JsonSerializable


class CommandResult(NamedTuple):
    exit_code: int
    stdout: str
    stderr: str


class CommandResult2(NamedTuple):
    stdout: str
    stderr: str


class Utils:
    @staticmethod
    async def run_command(cmd: str) -> CommandResult:
        """Run given command."""
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await proc.communicate()

        return CommandResult(
            exit_code=proc.returncode or 0,
            stdout=stdout.decode().strip(),
            stderr=stderr.decode().strip(),
        )

    @staticmethod
    async def run_command_for_success(cmd: str) -> CommandResult2:
        """Run given command and if the exit code is not 0 raise and exception."""
        result = await Utils.run_command(cmd)
        if result.exit_code != 0:
            raise RuntimeError("Invalid exit code for command", (result.exit_code, cmd, result.stdout, result.stderr))
        return CommandResult2(stdout=result.stdout, stderr=result.stderr)

    @staticmethod
    async def wait_for_service(url: str, max_attempts: int = 30, delay: float = 1.0) -> bool:
        """Wait for a service to become available."""
        for attempt in range(max_attempts):
            try:
                async with ClientSession() as session, session.get(f"{url}") as resp:
                    if resp.status == 200:
                        return True
            except Exception:
                pass
            if attempt < max_attempts - 1:
                await asyncio.sleep(delay)
        return False

    @staticmethod
    def join_url(a: str, b: str) -> str:
        """Join two strings into one url."""
        aa = a[:-1] if a.endswith("/") else a
        bb = b[1:] if b.startswith("/") else b
        return f"{aa}/{bb}"

    @staticmethod
    def save_file(file_path: Path, file_content: str) -> None:
        """Save given content under given path."""
        with file_path.open("w", encoding="utf-8") as f:
            f.write(file_content)

    @staticmethod
    def sanitize_service_name(name: str) -> str:
        """Convert model name to valid Docker service name."""
        name = re.sub(r"[^a-zA-Z0-9_-]", "-", name)
        name = re.sub(r"-+", "-", name)
        name = name.strip("-")
        if name and not name[0].isalpha() and name[0] != "_":
            name = f"service-{name}"
        return name.lower()

    @staticmethod
    def shell_escape(arg: str) -> str:
        """Return shell-escaped version of a string."""
        return shlex.quote(arg)

    @staticmethod
    def str_encode(text: str, safe: str = "") -> str:
        """Encode string for url."""
        return quote(text, safe=safe)

    @staticmethod
    async def ensure_model_downloaded(model_url: str, model_dir: Path, filename: str | None = None) -> tuple[Path, str]:
        """Download model if it's a URL, return (local_path, filename)."""
        if not model_url.startswith("https://"):
            # Already a local path
            local_path = Path(model_url)
            return local_path, local_path.name

        # Get filename from URL (last part after /)
        filename2 = filename if filename is not None else model_url.split("/")[-1]
        dir = model_dir
        local_path = dir / filename2

        if not local_path.exists():
            print(f"Downloading {filename2}...")

            # Ensure the models directory exists
            local_path.parent.mkdir(parents=True, exist_ok=True)

            # Download with and error handling
            try:
                # Use curl instead of wget for better macOS compatibility
                await download_file(model_url, local_path)

                # Validate the downloaded file
                if not local_path.exists() or local_path.stat().st_size == 0:
                    if local_path.exists():
                        local_path.unlink()
                    raise RuntimeError("Downloaded file is empty or missing", local_path)  # noqa: TRY301

                print(f"Successfully downloaded {filename2} ({local_path.stat().st_size} bytes)")

            except Exception as e:
                # Clean up any partial download
                if local_path.exists():
                    local_path.unlink()
                raise RuntimeError("Failed to download", filename2) from e

        return local_path, filename2


class FetchResult(BaseModel):
    status_code: int
    data: str


async def download_file(url: str, file_path: Path) -> None:
    """Download file from given url and save it under given path."""
    async with aiohttp.ClientSession() as session, session.get(url, timeout=ClientTimeout(3600)) as response:
        if response.status != 200:
            body = await response.text()
            msg = f"Cannot download file from {url} get status code {response.status}, body: {body}"
            raise RuntimeError(msg)

        async with aiofiles.open(file_path, "wb") as f:
            async for chunk in response.content.iter_chunked(1024):
                await f.write(chunk)


async def fetch_from(url: str, method: str = "GET", data: JsonSerializable | None = None) -> FetchResult:
    """Make HTTP request to host on given port."""
    async with aiohttp.ClientSession() as session, session.request(method, url, json=data) as response:
        return FetchResult(status_code=response.status, data=await response.text())


def add_token_to_civitai(url: str, token: str) -> str:
    """Add token to civit."""
    return f"{url}&token={token}"
