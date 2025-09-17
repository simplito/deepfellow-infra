"""Module which load and save settings."""

import asyncio
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Literal, TypedDict, cast

from server.config import AppSettings
from server.models.common import JsonSerializable

ServiceRawConfig = JsonSerializable


class FileContent(TypedDict):
    services: dict[str, ServiceRawConfig]


class ServiceProvider:
    def __init__(self, config: AppSettings):
        self.config = config

    def _get_file_path(self) -> Path:
        return (self.config.get_storage_dir() / "./services.json").resolve()

    def load(self) -> FileContent:
        """Load settings file content."""
        fpath = self._get_file_path()
        try:
            with fpath.open(encoding="utf-8") as f:
                content = f.read()
                data: FileContent = json.loads(content)
                if "services" not in data:
                    data["services"] = {}
                return data
        except FileNotFoundError:
            return {"services": {}}

    async def save(self, content: FileContent) -> None:
        """Save settings file content."""
        fpath = self._get_file_path()
        json_data = json.dumps(content, indent=2)
        await asyncio.to_thread(self._write_file, str(fpath), json_data)

    def _write_file(self, path: str, data: str) -> None:
        fpath = Path(path)
        fpath.parent.mkdir(parents=True, exist_ok=True)
        with fpath.open("w", encoding="utf-8") as f:
            f.write(data)

    async def _modify(self, func: Callable[[FileContent], Literal[False] | FileContent | Awaitable[Literal[False] | FileContent]]) -> None:
        content = self.load()
        maybe_new_content = func(content)
        if asyncio.iscoroutine(maybe_new_content):
            new_content = cast("Literal[False] | FileContent", await maybe_new_content)
        else:
            new_content = cast("Literal[False] | FileContent", maybe_new_content)
        if new_content is not False:
            await self.save(new_content)

    async def save_service_config(self, service_id: str, data: ServiceRawConfig) -> None:
        """Save service config."""

        async def handler(content: FileContent) -> Literal[False] | FileContent:
            content["services"][service_id] = data
            return content

        await self._modify(handler)

    async def clear_service_config(self, service_id: str) -> None:
        """Clear service config."""

        async def handler(content: FileContent) -> Literal[False] | FileContent:
            if service_id in content["services"]:
                del content["services"][service_id]
            return content

        await self._modify(handler)
