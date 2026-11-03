# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Model downloaders."""

import json
import re
from abc import abstractmethod
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any, Never, NotRequired, TypedDict
from urllib.parse import quote, urlparse, urlunparse
from uuid import uuid4

from aiohttp import ClientSession
from fastapi import HTTPException

from scripts.get_huggingface_models import HfRepoAccessError
from scripts.utils.hf_common import HfErrorKind
from server.config import AppSettings
from server.utils.core import (
    DownloadedPacket,
    DownloadPacket,
    HttpClientError,
    PreDownloadPacket,
    SuccessDownloadPacket,
    Utils,
    download_file,
)
from server.utils.loading import Progress
from server.utils.logger import logger

_PROGRESS_LOG_STEP = 0.1  # log every ~10% of progress so the console isn't flooded with updates
# One file of a repository: https://huggingface.co/<owner>/<repo>/resolve/<revision>/<path>
_HF_FILE_URL = re.compile(r"^https://huggingface\.co/[^/]+/[^/]+/resolve/")


class BDownloader:
    error_msg_modifiers: list[tuple[str, str]]

    @abstractmethod
    def check_url(self, url: str) -> bool:
        """Check is url handled by this downloader."""

    def create_error_msg(self, msg: str, model_page_url: str | None = None) -> str | dict[str, Any]:
        """Create new string with error msg modifiers."""
        try:
            data = json.loads(msg)
            if isinstance(data, dict) and "message" in data and isinstance(data["message"], str):
                data["message"] = self._replace_str(data["message"], model_page_url)
                return data  # pyright: ignore[reportUnknownVariableType]
        except Exception:
            pass
        return self._replace_str(msg, model_page_url)

    def _raise_http_error(self, e: HttpClientError, model_page_url: str | None = None) -> Never:
        raise HTTPException(404 if e.status_code == 404 else 500, self.create_error_msg(e.body, model_page_url)) from e

    def _replace_str(self, msg: str, model_page_url: str | None = None) -> str:
        new_msg = msg
        for msg_to_replace, replacement in self.error_msg_modifiers:
            new_msg = new_msg.replace(msg_to_replace, replacement)
        if model_page_url:
            new_msg = new_msg.replace("{MODEL_URL}", f" Visit [the model page]({model_page_url}) to review and accept the model license.")
        else:
            new_msg = new_msg.replace("{MODEL_URL}", "")

        return new_msg


class BaseDownloader(BDownloader):
    @abstractmethod
    def check_url(self, url: str) -> bool:
        """Check is url handled by this downloader."""

    @abstractmethod
    def download(
        self, url: str, model_dir: Path, temp_dir: Path, filename: str | None = None, revision: str | None = None
    ) -> AsyncGenerator[DownloadPacket]:
        """Download model."""


class StandardModelDownloader(BaseDownloader):
    def check_url(self, url: str) -> bool:  # noqa: ARG002
        """Check is url handled by this downloader."""
        return True

    async def download(
        self,
        url: str,
        model_dir: Path,
        temp_dir: Path,
        filename: str | None = None,
        revision: str | None = None,  # noqa: ARG002
    ) -> AsyncGenerator[DownloadPacket]:
        """Download model."""
        async for packet in Utils.ensure_model_downloaded(url, model_dir, temp_dir, filename):
            yield packet


class HGRepoInfo(TypedDict):
    sha: str


class HGLFSFileInfo(TypedDict):
    oid: str
    size: int
    pointerSize: int


class HGFileInfo(TypedDict):
    type: str
    oid: str
    size: int
    path: str
    lfs: NotRequired[HGLFSFileInfo]
    xetHash: NotRequired[str]


class MyFileInfo(TypedDict):
    oid: str
    size: int
    path: str


class HuggingFaceRepoWithBlobsDownloader(BDownloader):
    header: dict[str, str]

    def __init__(self, key: str, temp_dir: Path):
        self.headers = Utils.create_bearer_header(key)
        self.temp_dir = temp_dir
        self.error_msg_modifiers = [
            (
                "Invalid credentials in Authorization header",
                "This model need HuggingFace Token to download. "
                "Setup DF_HUGGING_FACE_TOKEN env in enviromental variables to download this model. "
                "if token is set up probably is wrong or expired.",
            ),
            (
                "is restricted. You must have access to it and be authenticated to access it. Please log in.",
                "is restricted. "
                "This model need HuggingFace Token to download. "
                "Account also need access to this project. Check project site for rules approve.{MODEL_URL} "
                "After that setup DF_HUGGING_FACE_TOKEN env in enviromental variables to download this model. "
                "if token is set up probably is wrong or expired.",
            ),
            (
                "is restricted and you are not in the authorized list.",
                "is restricted and you are not in the authorized list. "
                "Account with your DF_HUGGING_FACE_TOKEN env doesn't have access to this model.{MODEL_URL}",
            ),
        ]

    def check_url(self, url: str) -> bool:  # noqa: ARG002
        """Check is url handled by this downloader."""
        return False

    @staticmethod
    async def _get_commit_id(model_id: str) -> str:
        """Get current commit id."""
        url = f"https://huggingface.co/api/models/{model_id}"
        async with ClientSession() as session, session.get(url) as response:
            data: HGRepoInfo = await response.json()
            return data["sha"]

    @staticmethod
    async def _get_files(model_id: str) -> list[MyFileInfo]:
        """Get filenames for repository."""
        files: list[MyFileInfo] = []
        url = f"https://huggingface.co/api/models/{model_id}/tree/main"
        data: list[HGFileInfo] = []
        async with ClientSession() as session, session.get(url) as response:
            data = await response.json()
        for item in data:
            if item["type"] == "file":
                files.append(
                    {
                        "oid": item["lfs"]["oid"] if "lfs" in item else item["oid"],
                        "size": item["lfs"]["size"] if "lfs" in item else item["size"],
                        "path": item["path"],
                    }
                )
        return files

    async def download(self, url: str, model_dir: Path, filter_out_other_modelfiles: bool = False) -> AsyncGenerator[DownloadPacket]:
        """Download model."""
        url_parsed = urlparse(url)
        model_id = urlunparse(url_parsed._replace(query=""))
        commit_id = await self._get_commit_id(model_id)
        files = await self._get_files(model_id)
        blobs_dir = model_dir / "blobs"
        blobs_dir.mkdir(parents=True, exist_ok=True)
        refs_dir = model_dir / "refs"
        refs_dir.mkdir(parents=True, exist_ok=True)
        snapshot_dir = model_dir / "snapshots" / commit_id
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        ref_file = refs_dir / "main"
        if not ref_file.exists():
            ref_file.write_text(commit_id)

        if filter_out_other_modelfiles and any(x for x in files if x["path"] == "model.safetensors"):
            files = [x for x in files if (x["path"] != "pytorch_model.bin") and (x["path"] != "flax_model.msgpack")]

        for file in files:
            whole_url = f"https://huggingface.co/{model_id}/resolve/main/{file['path']}"
            try:
                async for packet in Utils.ensure_model_downloaded(whole_url, blobs_dir, self.temp_dir, file["oid"], self.headers):
                    yield packet
                symlink_path = snapshot_dir / file["path"]
                if symlink_path.exists():
                    continue
                symlink_path.symlink_to(f"../../blobs/{file['oid']}", target_is_directory=False)
            except HttpClientError as e:
                self._raise_http_error(e, f"https://huggingface.co/{model_id}")

        yield SuccessDownloadPacket(model_dir)


class HuggingFaceRepoDownloader(BaseDownloader):
    header: dict[str, str]
    huggingface_url_pattern = r"https://huggingface\.co/(.*?)(?:/tree/main)?$"

    def __init__(self, key: str):
        self.headers = Utils.create_bearer_header(key)
        self.error_msg_modifiers = [
            (
                "Invalid credentials in Authorization header",
                "This model need HuggingFace Token to download. "
                "Setup DF_HUGGING_FACE_TOKEN env in enviromental variables to download this model. "
                "if token is set up probably is wrong or expired.",
            ),
            (
                "is restricted. You must have access to it and be authenticated to access it. Please log in.",
                "is restricted. "
                "This model need HuggingFace Token to download. "
                "Account also need access to this project. Check project site for rules approve.{MODEL_URL} "
                "After that setup DF_HUGGING_FACE_TOKEN env in enviromental variables to download this model. "
                "if token is set up probably is wrong or expired.",
            ),
            (
                "is restricted and you are not in the authorized list.",
                "is restricted and you are not in the authorized list. "
                "Account with your DF_HUGGING_FACE_TOKEN env doesn't have access to this model.{MODEL_URL}",
            ),
        ]

    def check_url(self, url: str) -> bool:
        """Check is url handled by this downloader.

        It should work with:

        username/repo-name
        https://huggingface.co/username/repo-name
        https://huggingface.co/username/repo-name/tree/main
        https://huggingface.co/api/models/username/repo-name/tree/main

        A single file (`.../resolve/<revision>/<path>`) is not a repository: HuggingFaceModelDownloader handles it.
        """
        if _HF_FILE_URL.match(url):
            return False
        return not url.startswith("http") or (bool(re.search(self.huggingface_url_pattern, url)))

    @staticmethod
    async def get_filenames(model_id: str, revision: str | None = None, headers: dict[str, str] | None = None) -> tuple[list[str], int]:
        """Get filenames from repository.

        `headers` carries the HuggingFace token: HF answers an anonymous request for a private repo with 401.
        """
        filenames: list[str] = []
        url = f"https://huggingface.co/api/models/{model_id}/tree/{quote(revision, safe='') if revision else 'main'}"
        if headers is None:
            headers = {}

        data: list[dict[str, Any]] = []
        size = 0
        filenames = []
        async with ClientSession() as session, session.get(url, headers=headers) as response:
            if response.status != 200:
                body = await response.text()
                msg = f"Cannot download file from {url}, get status code {response.status}, {body}"
                raise HttpClientError(message=msg, status_code=response.status, headers=response.headers, body=body)
            data = await response.json()
        for item in data:
            # If we find repo with folder we should add support for it.
            if item.get("type") == "file" and (filename := item.get("path")):
                filenames.append(filename)
                size += int(item.get("size", 0))

        return filenames, size

    async def download(
        self, url: str, model_dir: Path, temp_dir: Path, filename: str | None = None, revision: str | None = None
    ) -> AsyncGenerator[DownloadPacket]:
        """Download model."""
        url_parsed = urlparse(url)
        model_id = urlunparse(url_parsed._replace(query=""))
        if model_id.startswith("http"):
            match = re.search(self.huggingface_url_pattern, url)
            if match:
                model_id = match.group(1)

        try:
            filenames, size = await self.get_filenames(model_id, revision, headers=self.headers)
        except HttpClientError as e:
            if e.status_code == 401 or (e.status_code == 404 and e.headers.get("X-Error-Code") == "RepoNotFound"):
                # HF hides whether a repo exists from a caller without access: 401 (no or invalid token) and
                # 404 RepoNotFound (valid token without access, or no such repo) can't be told apart.
                error = HfRepoAccessError(model_id, revision, HfErrorKind.NO_ACCESS, authenticated=bool(self.headers))
                raise HTTPException(404, str(error)) from e
            self._raise_http_error(e, f"https://huggingface.co/{model_id}")
        yield PreDownloadPacket(size)
        for filename in filenames:
            whole_url = f"https://huggingface.co/{model_id}/resolve/{quote(revision, safe='') if revision else 'main'}/{filename}"
            try:
                async for packet in Utils.ensure_model_downloaded(whole_url, model_dir, temp_dir, filename, self.headers):
                    if not isinstance(packet, PreDownloadPacket):
                        yield packet
            except HttpClientError as e:
                self._raise_http_error(e, f"https://huggingface.co/{model_id}")

        yield SuccessDownloadPacket(model_dir)


class HuggingFaceModelDownloader(BaseDownloader):
    header: dict[str, str]

    def __init__(self, key: str):
        self.headers = Utils.create_bearer_header(key)
        self.error_msg_modifiers = [
            (
                "Invalid credentials in Authorization header",
                "This model need HuggingFace Token to download. "
                "Setup DF_HUGGING_FACE_TOKEN env in enviromental variables to download this model. "
                "if token is set up probably is wrong or expired.",
            ),
            (
                "is restricted. You must have access to it and be authenticated to access it. Please log in.",
                "is restricted. "
                "This model need HuggingFace Token to download. "
                "Account also need access to this project. Check project site for rules approve.{MODEL_URL} "
                "After that setup DF_HUGGING_FACE_TOKEN env in enviromental variables to download this model. "
                "if token is set up probably is wrong or expired.",
            ),
            (
                "is restricted and you are not in the authorized list.",
                "is restricted and you are not in the authorized list. "
                "Account with your DF_HUGGING_FACE_TOKEN env doesn't have access to this model.{MODEL_URL}",
            ),
        ]

    def check_url(self, url: str) -> bool:
        """Check is url handled by this downloader: a single Hugging Face file (`.../resolve/...`) or a `.gguf` URL."""
        return bool(_HF_FILE_URL.match(url) or (url.startswith("https://huggingface.co/") and ".gguf" in url))

    async def download(
        self,
        url: str,
        model_dir: Path,
        temp_dir: Path,
        filename: str | None = None,
        revision: str | None = None,  # noqa: ARG002
    ) -> AsyncGenerator[DownloadPacket]:
        """Download model."""
        url_parsed = urlparse(url)
        url_without_query = urlunparse(url_parsed._replace(query=""))
        try:
            async for packet in Utils.ensure_model_downloaded(url_without_query, model_dir, temp_dir, filename, self.headers):
                yield packet
        except HttpClientError as e:
            match = re.match(r"(https://huggingface\.co/[^/]+/[^/]+)", url_without_query)
            self._raise_http_error(e, match.group(1) if match else None)


class CivitaiModelDownloader(BaseDownloader):
    token: str

    def __init__(self, token: str):
        self.token = token
        self.error_msg_modifiers = [
            (
                "The creator of this asset requires you to be logged in to download it",
                "Setup DF_CIVITAI_TOKEN env in enviromental variables to download this model. "
                "if token is set up probably is wrong or expired.",
            )
        ]

    def check_url(self, url: str) -> bool:
        """Check is url handled by this downloader."""
        return url.startswith("https://civitai.com/")

    def add_token_to_url(self, url: str) -> str:
        """Add token to url."""
        return Utils.add_url_parameter_if_missing(url, "token", self.token)

    async def download(
        self,
        url: str,
        model_dir: Path,
        temp_dir: Path,
        filename: str | None = None,
        revision: str | None = None,  # noqa: ARG002
    ) -> AsyncGenerator[DownloadPacket]:
        """Download model."""
        try:
            async for packet in Utils.ensure_model_downloaded(self.add_token_to_url(url), model_dir, temp_dir, filename):
                yield packet
        except HttpClientError as e:
            self._raise_http_error(e)


class AdapterRegistryDownloader(BaseDownloader):
    """Downloader for adapter registry with Bearer token authentication.

    Not registered in `ModelDownloader.custom_downloaders` - built ad hoc and `.download()` called directly.
    """

    def __init__(self, secret: str):
        self.headers = Utils.create_bearer_header(secret)
        self.error_msg_modifiers = [
            (
                "401",
                "Adapter registry authentication failed. Check the adapter_registry_secret setting "
                "(Settings page or PUT /admin/config). If set, it may be wrong.",
            ),
        ]

    def check_url(self, url: str) -> bool:  # noqa: ARG002
        """Check is url handled by this downloader."""
        return False

    async def download(
        self,
        url: str,
        model_dir: Path,
        temp_dir: Path,
        filename: str | None = None,
        revision: str | None = None,  # noqa: ARG002
    ) -> AsyncGenerator[DownloadPacket]:
        """Download adapter directly from the given URL using Bearer token authentication."""
        filename_out = filename if filename is not None else url.split("/")[-1].split("?")[0]
        local_path = model_dir / filename_out

        if local_path.exists():
            yield SuccessDownloadPacket(local_path, filename_out)
            return

        temp_path = temp_dir / str(uuid4())
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            async for packet in download_file(url, temp_path, self.headers):
                yield packet
        except HttpClientError as e:
            if temp_path.exists():
                temp_path.unlink()
            self._raise_http_error(e)
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise

        if not temp_path.exists() or temp_path.stat().st_size == 0:
            if temp_path.exists():
                temp_path.unlink()
            msg = f"Downloaded adapter file is missing or empty: {url}"
            raise RuntimeError(msg)

        local_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path.rename(local_path)
        yield SuccessDownloadPacket(local_path, filename_out)


class ModelDownloader:
    temp_dir: Path
    standard_downloader: StandardModelDownloader
    custom_downloaders: list[BaseDownloader]

    def __init__(self, config: AppSettings):
        self.create_downloaders(config)

    def create_downloaders(self, config: AppSettings) -> None:
        """Create downloaders."""
        self.temp_dir: Path = config.get_storage_dir() / "temp"
        self.standard_downloader = StandardModelDownloader()
        self.custom_downloaders: list[BaseDownloader] = [
            HuggingFaceRepoDownloader(self.get_hugging_face_token(config)),
            HuggingFaceModelDownloader(self.get_hugging_face_token(config)),
            CivitaiModelDownloader(self.get_civitai_token(config)),
        ]
        self.hugging_face_repo_with_blobs_downloader = HuggingFaceRepoWithBlobsDownloader(
            self.get_hugging_face_token(config),
            self.temp_dir,
        )

    async def download(
        self, url: str, model_dir: Path, filename: str | None = None, revision: str | None = None
    ) -> AsyncGenerator[DownloadPacket]:
        """Download model."""
        specified_downloader = self.standard_downloader
        for downloader in self.custom_downloaders:
            if downloader.check_url(url):
                specified_downloader = downloader

        logger.debug("ModelDownloader.download: url=%s, selected=%s", url, type(specified_downloader).__name__)
        progress = Progress(0)
        last_logged_percentage = 0.0
        async for packet in specified_downloader.download(url, model_dir, self.temp_dir, filename, revision):
            if isinstance(packet, PreDownloadPacket) and packet.file_bytes_size:
                progress.set_max_value(packet.file_bytes_size)
            elif isinstance(packet, DownloadedPacket) and progress.max:
                progress.add_to_actual_value(packet.downloaded_bytes_size)
                percentage = progress.get_percentage()
                if percentage - last_logged_percentage >= _PROGRESS_LOG_STEP:
                    logger.info("Downloading %s: %.0f%%", url, percentage * 100)
                    last_logged_percentage = percentage
            elif isinstance(packet, SuccessDownloadPacket):
                logger.info("Downloaded %s", url)
            yield packet

    def get_hugging_face_token(self, config: AppSettings) -> str:
        """Return Hugging Face Key."""
        return config.hugging_face_token.get_secret_value()

    def get_civitai_token(self, config: AppSettings) -> str:
        """Return Civitai Face Key."""
        return config.civitai_token.get_secret_value()
