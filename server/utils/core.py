"""Utils module."""

import asyncio
import re
import shlex
from pathlib import Path
from typing import Any, NamedTuple
from urllib.parse import parse_qs, quote, urlencode, urlparse, urlunparse
from uuid import uuid4

import aiofiles
import aiohttp
from aiohttp import ClientSession, ClientTimeout
from fastapi import HTTPException
from multidict import CIMultiDictProxy
from pydantic import BaseModel, ValidationError

from server.models.common import JsonSerializable


class HttpClientError(Exception):
    def __init__(self, message: str, status_code: int, headers: CIMultiDictProxy[str], body: str):
        super().__init__(message)
        self.status_code = status_code
        self.headers = headers
        self.body = body


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
    def create_bearer_header(key: str) -> dict[str, str]:
        """Create bearer header."""
        return {"Authorization": f"Bearer {key}"} if key else {}

    @staticmethod
    async def ensure_model_downloaded(
        model_url: str,
        model_dir: Path,
        temp_dir: Path,
        filename: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[Path, str]:
        """Download model if it's a URL, return (local_path, filename)."""
        if headers is None:
            headers = {}

        if not model_url.startswith("https://"):
            # Already a local path
            local_path = Path(model_url)
            return local_path, local_path.name

        # Get filename from URL (last part after /)
        filename2 = filename if filename is not None else model_url.split("/")[-1].split("?")[0]
        dir = model_dir
        local_path = dir / filename2

        if not local_path.exists():
            print(f"Downloading {filename2}...")

            # Ensure the models directory exists

            temp_path = temp_dir / str(uuid4())

            temp_path.parent.mkdir(parents=True, exist_ok=True)

            # Download with and error handling
            try:
                # Use curl instead of wget for better macOS compatibility
                await download_file(model_url, temp_path, headers)

                # Validate the downloaded file
                if not temp_path.exists() or temp_path.stat().st_size == 0:
                    if temp_path.exists():
                        temp_path.unlink()
                    raise RuntimeError("Downloaded file is empty or missing", temp_path)  # noqa: TRY301

            except Exception:
                # Clean up any partial download
                if temp_path.exists():
                    temp_path.unlink()
                raise

            local_path.parent.mkdir(parents=True, exist_ok=True)

            temp_path.rename(local_path)

            print(f"Successfully downloaded {filename2} ({local_path.stat().st_size} bytes)")

        return local_path, filename2

    @staticmethod
    def add_url_parameter_if_missing(url: str, param_name: str, param_value: str) -> str:
        """Check parameter existance in a URL's query string and adds it if it doesn't."""
        if param_value:
            parsed_url = urlparse(url)
            query_params = parse_qs(parsed_url.query, keep_blank_values=True)
            if param_name not in query_params:
                query_params[param_name] = [param_value]
                new_query = urlencode(query_params, doseq=True)
                return urlunparse(parsed_url._replace(query=new_query))

        return url


class FetchResult(BaseModel):
    status_code: int
    data: str


async def download_file(url: str, file_path: Path, headers: dict[str, str]) -> None:
    """Download file from given url and save it under given path."""
    async with aiohttp.ClientSession() as session, session.get(url, headers=headers, timeout=ClientTimeout(3600)) as response:
        if response.status != 200:
            body = await response.text()
            msg = f"Cannot download file from {url} get status code {response.status}, {body}"
            raise HttpClientError(message=msg, status_code=response.status, headers=response.headers, body=body)

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


def normalize_name(s: str) -> str:
    """Normalize name."""
    result: list[str] = []
    for ch in s.lower():
        if ch.isalnum():
            result.append(ch)
        else:
            result.append("_")
    return "".join(result)


def try_parse_pydantic[T](cls: type[T], data: Any) -> T:  # noqa: ANN401
    """Try parse pydantic, if it fails raise HTTPException with details."""
    try:
        return cls(**data)
    except Exception as e:
        raise HTTPException(400, e.errors if isinstance(e, ValidationError) else "Unknown") from e
