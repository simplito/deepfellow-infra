# DeepFellow Software Framework.
# Copyright © 2025 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utils module."""

import asyncio
import json
import logging
import re
import shlex
from collections.abc import AsyncGenerator, Awaitable, Callable
from pathlib import Path
from typing import Any, Literal, NamedTuple, TypedDict
from urllib.parse import parse_qs, quote, urlencode, urlparse, urlunparse
from uuid import uuid4

import aiofiles
import aiohttp
from aiohttp import ClientSession, ClientTimeout
from fastapi import HTTPException
from fastapi.responses import JSONResponse, Response, StreamingResponse
from multidict import CIMultiDictProxy
from pydantic import BaseModel, ValidationError

from server.models.common import JsonSerializable

logger = logging.getLogger("uvicorn.error")


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


class DownloadPacket(NamedTuple):
    downloaded_bytes_size: int = 0
    success: bool = False
    local_path: Path | None = None
    filename: str = ""


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
    async def read_file(file_path: Path) -> str:
        """Save given content under given path."""
        async with aiofiles.open(file_path) as f:
            return await f.read()

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
    ) -> AsyncGenerator[DownloadPacket]:
        """Download model if it's a URL, return (local_path, filename)."""
        if headers is None:
            headers = {}

        if not model_url.startswith("https://"):
            # Already a local path
            local_path = Path(model_url)
            yield DownloadPacket(success=True, local_path=local_path, filename=filename or "")

        else:
            # Get filename from URL (last part after /)
            filename_out = filename if filename is not None else model_url.split("/")[-1].split("?")[0]
            dir = model_dir
            local_path = dir / filename_out

            if not local_path.exists():
                print(f"Downloading {filename_out}...")
                temp_path = temp_dir / str(uuid4())
                temp_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    async for downloaded_bytes_size in download_file(model_url, temp_path, headers):
                        yield DownloadPacket(downloaded_bytes_size=downloaded_bytes_size)

                except Exception:
                    # Clean up any partial download
                    if temp_path.exists():
                        temp_path.unlink()
                    raise

                # Validate the downloaded file
                if not temp_path.exists():
                    raise RuntimeError("Downloaded file is missing", temp_path)

                if temp_path.stat().st_size == 0:
                    temp_path.unlink()
                    raise RuntimeError("Downloaded file is empty", temp_path)

                local_path.parent.mkdir(parents=True, exist_ok=True)
                temp_path.rename(local_path)

                print(f"Successfully downloaded {filename_out} ({local_path.stat().st_size} bytes)")

            yield DownloadPacket(success=True, local_path=local_path, filename=filename_out)

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


async def download_file(url: str, file_path: Path, headers: dict[str, str]) -> AsyncGenerator[int]:
    """Download file from given url and save it under given path."""
    async with aiohttp.ClientSession() as session, session.get(url, headers=headers, timeout=ClientTimeout(3600)) as response:
        if response.status != 200:
            body = await response.text()
            msg = f"Cannot download file from {url} get status code {response.status}, {body}"
            raise HttpClientError(message=msg, status_code=response.status, headers=response.headers, body=body)

        async with aiofiles.open(file_path, "wb") as f:
            async for chunk in response.content.iter_any():
                downloaded_bytes_size = len(chunk)
                await f.write(chunk)
                yield downloaded_bytes_size


async def fetch_from(url: str, method: str = "GET", data: JsonSerializable | None = None) -> FetchResult:
    """Make HTTP request to host on given port."""
    async with aiohttp.ClientSession() as session, session.request(method, url, json=data) as response:
        return FetchResult(status_code=response.status, data=await response.text())


async def stream_fetch_from(url: str, method: str = "GET", data: JsonSerializable | None = None) -> AsyncGenerator[FetchResult]:
    """Make stream HTTP request to host on given port."""
    async with aiohttp.ClientSession() as session, session.request(method, url, json=data) as response:
        async for chunk in response.content.iter_any():
            yield FetchResult(status_code=response.status, data=chunk.decode())


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


def format_pydantic_errors(exc: ValidationError) -> list[dict[str, str]]:
    """Convert a Pydantic ValidationError's detailed errors into a user-friendly list of dictionaries."""
    formatted_errors: list[dict[str, str]] = []
    for error in exc.errors():
        formatted_errors.append({"field": str(error["loc"][0]), "message": error["msg"]})
    return formatted_errors


def try_parse_pydantic[T](cls: type[T], data: Any) -> T:  # noqa: ANN401
    """Try parse pydantic, if it fails raise HTTPException with details."""
    try:
        return cls(**data)
    except Exception as e:
        raise HTTPException(400, format_pydantic_errors(e) if isinstance(e, ValidationError) else "Unknown") from e


def convert_size_to_bytes(size_str: str) -> None | int:
    """Convert a string representation of a size (e.g., "46MB", "1.2GB", "512K") into an integer number of bytes.

    Args:
        size_str (str): The size string to convert.

    Returns:
        int: The size in bytes, or None if the format is invalid.
    """
    # 1. Clean and standardize the string
    size_cleaned: str = size_str.strip().upper().replace(" ", "")

    # 2. Define the multipliers for Kilo, Mega, Giga, and Tera bytes
    multipliers = {
        "B": 1,
        "K": 1000,
        "KB": 1000,
        "M": 1000**2,
        "MB": 1000**2,
        "G": 1000**3,
        "GB": 1000**3,
        "T": 1000**4,
        "TB": 1000**4,
    }

    # 3. Find the unit in the string
    unit = None
    number_str = ""

    # Iterate backwards to find the unit (e.g., 'MB') and separate the number
    for u in sorted(multipliers.keys(), key=len, reverse=True):
        if size_cleaned.endswith(u):
            unit = u
            # The number part is everything before the unit
            number_str = size_cleaned[: -len(u)]
            break

    if not unit or not number_str:
        # If no recognizable unit is found, or the number part is empty
        return None

    # 4. Convert the number part to a float and multiply
    try:
        number = float(number_str)
        multiplier = multipliers.get(unit, 1)  # Default to 1 if something went wrong

        # Calculate the final byte count. Use int() to get a whole number.
        return int(number * multiplier)

    except ValueError:
        # Handle cases where the number part (number_str) isn't a valid float
        return None


class StreamChunkFinish(TypedDict):
    type: Literal["finish"]
    status: Literal["ok", "error"]
    details: Any


class StreamChunkProgress(TypedDict):
    value: float
    type: Literal["progress"]


type StreamChunk = StreamChunkFinish | StreamChunkProgress


class Stream[T]:
    def __init__(self):
        self.queue = asyncio.Queue[T | None]()
        self.closed = False
        self.consumed = False

    def emit(self, data: T) -> None:
        """Emit event."""
        if not self.closed:
            self.queue.put_nowait(data)

    def close(self) -> None:
        """Close stream."""
        if not self.closed:
            self.closed = True
            self.queue.put_nowait(None)

    async def as_generator(self) -> AsyncGenerator[T]:
        """Convert to generator."""
        if self.consumed:
            raise RuntimeError("Stream already consumed")
        self.consumed = True
        if self.closed:
            return
        while True:
            item = await self.queue.get()
            if item is None:
                break
            yield item

    def pipe(self, stream: "Stream[T]") -> None:
        """Send all events to given stream."""

        async def proxy() -> None:
            async for data in self.as_generator():
                stream.emit(data)
            stream.close()

        self.pipe_task = asyncio.create_task(proxy())


class PromiseWithProgress[T, U]:
    def __init__(self, value: T | None = None, func: Callable[[Stream[U]], Awaitable[T]] | None = None):
        self._future = asyncio.get_running_loop().create_future()
        self.progress = Stream[U]()

        if value is not None:
            self.has_stream = False
            self._future.set_result(value)

        if func is not None:
            self.has_stream = True

            async def the_func() -> None:
                try:
                    result = await func(self.progress)
                    self._future.set_result(result)
                except Exception as e:
                    self._future.set_exception(e)
                finally:
                    self.progress.close()

            self.task = asyncio.create_task(the_func())

    async def wait(self) -> T:
        """Wait for the result."""
        return await self._future

    def next[Z](self, func: Callable[[T], Awaitable[Z]]) -> "PromiseWithProgress[Z, U]":
        """Add next step."""

        async def the_func(_stream: Stream[U]) -> Z:
            res = await self.wait()
            return await func(res)

        promise = PromiseWithProgress[Z, U](func=the_func)
        promise.progress = self.progress
        return promise


async def convert_promise_with_progress_to_fastapi_response(promise: PromiseWithProgress[BaseModel, StreamChunk]) -> Response:
    """Convert to StreamingResult."""
    if not promise.has_stream:
        model = await promise.wait()
        return JSONResponse(model.model_dump())

    async def generator() -> AsyncGenerator[str]:
        try:
            async for data in promise.progress.as_generator():
                yield "data: " + json.dumps(data) + "\n\n"
            result = await promise.wait()
            chunk: StreamChunk = {
                "type": "finish",
                "status": "ok",
                "details": result.model_dump(),
            }
            yield "data: " + json.dumps(chunk) + "\n\n"
        except Exception as e:
            logger.exception("Error during generator")
            chunk: StreamChunk = {
                "type": "finish",
                "status": "error",
                "details": f"{e.status_code} {e.detail}" if isinstance(e, HTTPException) else "<unknown>",
            }
            yield "data: " + json.dumps(chunk) + "\n\n"

    return StreamingResponse(content=generator(), media_type="text/event-stream")
