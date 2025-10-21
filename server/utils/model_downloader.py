"""Model downloaders."""

import json
from abc import abstractmethod
from contextlib import suppress
from pathlib import Path
from typing import Any

from aiohttp import ClientSession
from fastapi import HTTPException

from server.config import AppSettings
from server.utils.core import HttpClientError, Utils


class BaseDownloader:
    error_msg_modifiers: list[tuple[str, str]]

    @staticmethod
    @abstractmethod
    def check_url(url: str) -> bool:
        """Check is url handled by this downloader."""

    def create_error_msg(self, msg: str) -> str | dict[str, Any]:
        """Create new string with error msg modifiers."""
        try:
            data = json.loads(msg)
            if isinstance(data, dict) and "message" in data and isinstance(data["message"], str):
                data["message"] = self._replace_str(data["message"])
                return data  # pyright: ignore[reportUnknownVariableType]
        except Exception:
            pass
        return self._replace_str(msg)

    def _replace_str(self, msg: str) -> str:
        new_msg = msg
        for msg_to_replace, replacement in self.error_msg_modifiers:
            new_msg = new_msg.replace(msg_to_replace, replacement)

        return new_msg

    @abstractmethod
    async def download(self, url: str, model_dir: Path, temp_dir: Path, filename: str | None = None) -> tuple[Path, str]:
        """Download model."""


class StandardModelDownloader(BaseDownloader):
    @staticmethod
    def check_url(url: str) -> bool:  # noqa: ARG004
        """Check is url handled by this downloader."""
        return True

    async def download(self, url: str, model_dir: Path, temp_dir: Path, filename: str | None = None) -> tuple[Path, str]:
        """Download model."""
        return await Utils.ensure_model_downloaded(url, model_dir, temp_dir, filename)


class HuggingFaceRepoDownloader(BaseDownloader):
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
                "Account also need access to this project. Check project site for rules approve. "
                "After that setup DF_HUGGING_FACE_TOKEN env in enviromental variables to download this model. "
                "if token is set up probably is wrong or expired.",
            ),
            (
                "is restricted and you are not in the authorized list.",
                "is restricted and you are not in the authorized list. "
                "Account with your DF_HUGGING_FACE_TOKEN env doesn't have access to this model.",
            ),
        ]

    @staticmethod
    def check_url(url: str) -> bool:
        """Check is url handled by this downloader."""
        return not url.startswith("http")

    @staticmethod
    async def get_filenames(model_id: str) -> list[str]:
        """Get filenames fro repository."""
        filenames: list[str] = []
        url = f"https://huggingface.co/api/models/{model_id}/tree/main"
        data: list[dict[str, Any]] = []
        with suppress(Exception):
            # Adding authentication header in here make error in public repos
            async with ClientSession() as session, session.get(url) as response:
                data = await response.json()
        try:
            for item in data:
                # If we find repo with folder we should add support for it.
                if item.get("type") == "file" and (filename := item.get("path")):
                    filenames.append(filename)
        except Exception:
            filenames = []

        return filenames

    async def download(self, url: str, model_dir: Path, temp_dir: Path, filename: str | None = None) -> tuple[Path, str]:
        """Download model."""
        model_id = url
        filenames = await self.get_filenames(model_id)
        for filename in filenames:
            whole_url = f"https://huggingface.co/{model_id}/resolve/main/{filename}"
            try:
                await Utils.ensure_model_downloaded(whole_url, model_dir, temp_dir, filename, self.headers)
            except HttpClientError as e:
                raise HTTPException(500, self.create_error_msg(e.body)) from e

        return model_dir, ""


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
                "Account also need access to this project. Check project site for rules approve. "
                "After that setup DF_HUGGING_FACE_TOKEN env in enviromental variables to download this model. "
                "if token is set up probably is wrong or expired.",
            ),
            (
                "is restricted and you are not in the authorized list.",
                "is restricted and you are not in the authorized list. "
                "Account with your DF_HUGGING_FACE_TOKEN env doesn't have access to this model.",
            ),
        ]

    @staticmethod
    def check_url(url: str) -> bool:
        """Check is url handled by this downloader."""
        return url.startswith("https://huggingface.co/")

    async def download(self, url: str, model_dir: Path, temp_dir: Path, filename: str | None = None) -> tuple[Path, str]:
        """Download model."""
        try:
            return await Utils.ensure_model_downloaded(url, model_dir, temp_dir, filename, self.headers)
        except HttpClientError as e:
            raise HTTPException(500, self.create_error_msg(e.body)) from e


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

    @staticmethod
    def check_url(url: str) -> bool:
        """Check is url handled by this downloader."""
        return url.startswith("https://civitai.com/")

    def add_token_to_url(self, url: str) -> str:
        """Add token to url."""
        return Utils.add_url_parameter_if_missing(url, "token", self.token)

    async def download(self, url: str, model_dir: Path, temp_dir: Path, filename: str | None = None) -> tuple[Path, str]:
        """Download model."""
        try:
            return await Utils.ensure_model_downloaded(self.add_token_to_url(url), model_dir, temp_dir, filename)
        except HttpClientError as e:
            raise HTTPException(500, self.create_error_msg(e.body)) from e


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
        self.custom_downloaders = [
            HuggingFaceRepoDownloader(self.get_hugging_face_token(config)),
            HuggingFaceModelDownloader(self.get_hugging_face_token(config)),
            CivitaiModelDownloader(self.get_civitai_token(config)),
        ]

    async def download(self, url: str, model_dir: Path, filename: str | None = None) -> tuple[Path, str]:
        """Download model."""
        specified_downloader = self.standard_downloader
        for downloader in self.custom_downloaders:
            if downloader.check_url(url):
                specified_downloader = downloader

        return await specified_downloader.download(url, model_dir, self.temp_dir, filename)

    def get_hugging_face_token(self, config: AppSettings) -> str:
        """Return Hugging Face Key."""
        return config.hugging_face_token

    def get_civitai_token(self, config: AppSettings) -> str:
        """Return Civitai Face Key."""
        return config.civitai_token
