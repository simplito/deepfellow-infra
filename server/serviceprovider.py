# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module which load and save settings."""

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

import aiofiles

from server.config import AppSettings
from server.models.common import JsonSerializable

logger = logging.getLogger("uvicorn.error")


FIRST_CONFIG_VERSION = "v1"
ACTUAL_CONFIG_VERSION = "v2"


type ConfigVersions = Literal["v1", "v2"]


ServiceRawConfig = JsonSerializable


class FileContent(TypedDict):
    version: ConfigVersions
    services: dict[str, ServiceRawConfig]


class ServiceProvider:
    def __init__(self, config: AppSettings):
        self.config = config
        self.file_lock = asyncio.Lock()

    def _get_file_path(self) -> Path:
        return (self.config.get_storage_dir() / "./services.json").resolve()

    def convert_v1_to_v2_config(self, data: dict[str, Any]) -> dict[str, Any]:
        """Convert v1 to v2 config."""
        new_data = data.copy()
        for service_name, service_data in data.get("services", {}).items():
            service = new_data["services"][service_name]
            service["instances"] = {}
            service["instances"]["default"] = {"options": None, "models": [], "custom": []}
            default_instance = service["instances"]["default"]
            if isinstance(service_data, dict):
                options = service_data.get("options")
                if options is not None:
                    del service["options"]
                    default_instance["options"] = options

                models = service_data.get("models")
                if models is not None:
                    del service["models"]
                    default_instance["models"] = models

                custom = service_data.get("custom")
                if custom is not None:
                    del service["custom"]
                    default_instance["custom"] = custom

        return new_data

    async def load(self) -> FileContent:
        """Load settings file content."""
        update_config: bool = False
        fpath = self._get_file_path()
        async with self.file_lock:
            logger.debug("Enter to read/write service.json file")
            try:
                async with aiofiles.open(fpath, encoding="utf-8") as f:
                    logger.debug("Starting reading service.json file")
                    content = await f.read()
                    logger.debug("Ending reading service.json file")
                    data: dict[str, Any] = json.loads(content)

                    if "services" not in data:
                        data["services"] = {}

                    version: ConfigVersions = data.get("version", FIRST_CONFIG_VERSION)
                    if version == "v1":
                        logger.debug("Migrate service.json from v1 to v2 version")
                        data = self.convert_v1_to_v2_config(data)
                        update_config = True

                    data_out = cast("FileContent", data)
            except FileNotFoundError:
                return {"version": ACTUAL_CONFIG_VERSION, "services": {}}

            if "version" not in data or data["version"] != ACTUAL_CONFIG_VERSION:
                update_config = True
                data["version"] = ACTUAL_CONFIG_VERSION

            if update_config:
                logger.debug("Config update")
                await self.save(data_out)

        logger.debug("Leaving read/write service.json file")

        return data_out

    async def save(self, content: FileContent) -> None:
        """Save settings file content."""
        fpath = self._get_file_path()
        json_data = json.dumps(content, indent=2)
        logger.debug("Staring writing service.json file")
        await self._write_file(str(fpath), json_data)
        logger.debug("Ended writing service.json file")

    async def _write_file(self, path: str, data: str) -> None:
        fpath = Path(path)
        fpath.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(fpath, mode="w", encoding="utf-8") as f:
            await f.write(data)

    async def _modify(self, func: Callable[[FileContent], Literal[False] | FileContent | Awaitable[Literal[False] | FileContent]]) -> None:
        content = await self.load()
        maybe_new_content = func(content)
        if asyncio.iscoroutine(maybe_new_content):
            new_content = cast("Literal[False] | FileContent", await maybe_new_content)
        else:
            new_content = cast("Literal[False] | FileContent", maybe_new_content)
        if new_content is not False:
            async with self.file_lock:
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
