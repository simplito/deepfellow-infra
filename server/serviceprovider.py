# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Module which load and save settings."""

import asyncio
import json
import os
import uuid
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

import aiofiles

from server.config import AppSettings, get_services_json_path
from server.models.common import JsonSerializable
from server.utils.logger import logger

FIRST_CONFIG_VERSION = "v1"
ACTUAL_CONFIG_VERSION = "v4"
_MAX_WARNING_MESSAGE_LEN = 2000


type ConfigVersions = Literal["v1", "v2", "v3", "v4"]


ServiceRawConfig = JsonSerializable


class WarningEntry(TypedDict):
    id: str
    created_at: str
    service_id: str
    instance: str | None
    model_id: str | None
    message: str


class FileContent(TypedDict):
    version: ConfigVersions
    services: dict[str, ServiceRawConfig]
    cloud_enabled: bool
    warnings: list[WarningEntry]


class ServicesConfigError(ValueError):
    """Raised when services.json parses as JSON but does not have the expected structure."""


class ServicesMigrationError(Exception):
    """Raised when services.json was read but its content migrated to the current version could not be saved."""

    def __init__(self, cause: OSError):
        super().__init__(str(cause))
        self.cause = cause


class ServiceProvider:
    def __init__(self, config: AppSettings):
        self.config = config
        self.file_lock = asyncio.Lock()
        self.load_error: str | None = None

    def _get_file_path(self) -> Path:
        return get_services_json_path(self.config)

    def convert_v3_to_v4_config(self, data: dict[str, Any]) -> dict[str, Any]:
        """Convert v3 to v4 config - adds the warnings list."""
        new_data = data.copy()
        new_data.setdefault("warnings", [])
        return new_data

    def convert_v2_to_v3_config(self, data: dict[str, Any]) -> dict[str, Any]:
        """Convert v2 to v3 config - adds cloud_enabled flag."""
        cloud_service_ids = {"claude", "googleai", "openai"}
        new_data = data.copy()
        services = new_data.get("services", {})
        has_cloud_installed = any(service_id in cloud_service_ids and bool(service_data) for service_id, service_data in services.items())
        new_data["cloud_enabled"] = has_cloud_installed
        return new_data

    def convert_v1_to_v2_config(self, data: dict[str, Any]) -> dict[str, Any]:
        """Convert v1 to v2 config."""
        new_data = data.copy()
        for service_name, service_data in data.get("services", {}).items():
            service = new_data["services"][service_name]
            service["instances"] = {}
            service["instances"]["default"] = {"options": {}, "models": [], "custom": []}
            default_instance = service["instances"]["default"]
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
        async with self.file_lock:
            data_out = await self._load_and_migrate_locked()
        logger.debug("Leaving read/write service.json file")
        return data_out

    async def _load_and_migrate_locked(self) -> FileContent:
        """Read the file and apply pending migrations. Caller must hold ``file_lock``."""
        update_config: bool = False
        fpath = self._get_file_path()
        logger.debug("Enter to read/write service.json file")
        try:
            async with aiofiles.open(fpath, encoding="utf-8") as f:
                logger.debug("Starting reading service.json file")
                content = await f.read()
                logger.debug("Ending reading service.json file")
                data: dict[str, Any] = json.loads(content)
                self._validate_structure(data)

                if "services" not in data:
                    data["services"] = {}
                if "warnings" not in data:
                    data["warnings"] = []

                version: ConfigVersions = data.get("version", FIRST_CONFIG_VERSION)
                if version == "v1":
                    logger.debug("Migrate service.json from v1 to v2 version")
                    data = self.convert_v1_to_v2_config(data)
                    update_config = True
                if version in ("v1", "v2"):
                    logger.debug("Migrate service.json from v2 to v3 version")
                    data = self.convert_v2_to_v3_config(data)
                    update_config = True
                if version in ("v1", "v2", "v3"):
                    logger.debug("Migrate service.json from v3 to v4 version")
                    data = self.convert_v3_to_v4_config(data)
                    update_config = True

                data_out = cast("FileContent", data)
        except FileNotFoundError:
            return {"version": ACTUAL_CONFIG_VERSION, "services": {}, "cloud_enabled": False, "warnings": []}

        if "version" not in data or data["version"] != ACTUAL_CONFIG_VERSION:
            update_config = True
            data["version"] = ACTUAL_CONFIG_VERSION

        if update_config:
            logger.debug("Config update")
            try:
                await self._save_locked(data_out)
            except OSError as exc:
                raise ServicesMigrationError(exc) from exc

        return data_out

    @staticmethod
    def _validate_structure(data: object) -> None:
        """Reject valid JSON of the wrong shape, which would otherwise fail later with an unclear TypeError."""
        if not isinstance(data, dict):
            msg = f"expected a JSON object at the top level, got {type(data).__name__}"
            raise ServicesConfigError(msg)
        content = cast("dict[str, Any]", data)
        if not isinstance(content.get("services", {}), dict):
            msg = "expected 'services' to be a JSON object"
            raise ServicesConfigError(msg)
        if not isinstance(content.get("warnings", []), list):
            msg = "expected 'warnings' to be a JSON array"
            raise ServicesConfigError(msg)

    async def save(self, content: FileContent) -> None:
        """Save settings file content."""
        async with self.file_lock:
            await self._save_locked(content)

    async def _save_locked(self, content: FileContent) -> None:
        """Write the file. Caller must hold ``file_lock``."""
        fpath = self._get_file_path()
        json_data = json.dumps(content, indent=2)
        logger.debug("Staring writing service.json file")
        await self._write_file(str(fpath), json_data)
        logger.debug("Ended writing service.json file")

    async def _write_file(self, path: str, data: str) -> None:
        fpath = Path(path)
        fpath.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = fpath.with_name(f"{fpath.name}.tmp")
        try:
            async with aiofiles.open(tmp_path, mode="w", encoding="utf-8") as f:
                await f.write(data)
            await asyncio.to_thread(os.replace, tmp_path, fpath)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

    async def _modify(self, func: Callable[[FileContent], Literal[False] | FileContent | Awaitable[Literal[False] | FileContent]]) -> None:
        """Read, mutate and write back the config as a single atomic unit under ``file_lock``.

        Holding the lock across the whole cycle (not just the save) prevents two concurrent
        callers from each reading the same snapshot and one overwriting the other's change.
        """
        async with self.file_lock:
            content = await self._load_and_migrate_locked()
            maybe_new_content = func(content)
            if asyncio.iscoroutine(maybe_new_content):
                new_content = cast("Literal[False] | FileContent", await maybe_new_content)
            else:
                new_content = cast("Literal[False] | FileContent", maybe_new_content)
            if new_content is not False:
                await self._save_locked(new_content)

    async def save_service_config(self, service_id: str, data: ServiceRawConfig) -> None:
        """Save service config.

        Skipped while services.json is not loaded: the in-memory config then holds only what was started since,
        so writing it would erase the saved instances once the admin repairs the file before restarting.
        """
        if self.load_error is not None:
            logger.warning(f"{service_id} config not saved: the services configuration was not loaded")
            return

        async def handler(content: FileContent) -> Literal[False] | FileContent:
            content["services"][service_id] = data
            return content

        await self._modify(handler)

    async def get_cloud_enabled(self) -> bool:
        """Get cloud_enabled flag from config."""
        content = await self.load()
        return content["cloud_enabled"]

    async def set_cloud_enabled(self, value: bool) -> None:
        """Set cloud_enabled flag in config."""

        async def handler(content: FileContent) -> FileContent:
            content["cloud_enabled"] = value
            return content

        await self._modify(handler)

    async def clear_service_config(self, service_id: str) -> None:
        """Clear service config, dropping any warnings recorded for it (it no longer exists to recover)."""

        async def handler(content: FileContent) -> Literal[False] | FileContent:
            changed = False
            if service_id in content["services"]:
                del content["services"][service_id]
                changed = True
            before = len(content["warnings"])
            content["warnings"] = [w for w in content["warnings"] if w["service_id"] != service_id]
            changed = changed or len(content["warnings"]) != before
            return content if changed else False

        await self._modify(handler)

    async def add_warning(self, service_id: str, message: str, *, instance: str | None = None, model_id: str | None = None) -> None:
        """Record a warning so a failure isn't visible only in the server logs.

        Replaces any existing warning for the same (service_id, instance, model_id) instead of
        appending a duplicate, so a failure that recurs across restarts doesn't pile up rows.
        """
        entry: WarningEntry = {
            "id": str(uuid.uuid4()),
            "created_at": datetime.now(UTC).isoformat(),
            "service_id": service_id,
            "instance": instance,
            "model_id": model_id,
            "message": message[:_MAX_WARNING_MESSAGE_LEN],
        }

        async def handler(content: FileContent) -> FileContent:
            content["warnings"] = [
                w
                for w in content["warnings"]
                if not (w["service_id"] == service_id and w["instance"] == instance and w["model_id"] == model_id)
            ]
            content["warnings"].append(entry)
            return content

        await self._modify(handler)

    async def list_warnings(self) -> list[WarningEntry]:
        """List all recorded warnings."""
        content = await self.load()
        return content["warnings"]

    async def dismiss_warning(self, warning_id: str) -> bool:
        """Dismiss a warning by id. Returns False if no warning with that id exists."""
        removed = False

        async def handler(content: FileContent) -> Literal[False] | FileContent:
            nonlocal removed
            before = len(content["warnings"])
            content["warnings"] = [w for w in content["warnings"] if w["id"] != warning_id]
            removed = len(content["warnings"]) != before
            if not removed:
                return False
            return content

        await self._modify(handler)
        return removed

    async def dismiss_all_warnings(self) -> None:
        """Dismiss every recorded warning."""

        async def handler(content: FileContent) -> Literal[False] | FileContent:
            if not content["warnings"]:
                return False
            content["warnings"] = []
            return content

        await self._modify(handler)

    async def dismiss_warnings_matching(self, service_id: str, *, instance: str | None = None, model_id: str | None = None) -> None:
        """Auto-dismiss warnings that exactly match (service_id, instance, model_id), e.g. once the thing recovers."""
        await self.dismiss_warnings_matching_any(service_id, [(instance, model_id)])

    async def dismiss_warnings_matching_any(self, service_id: str, keys: Sequence[tuple[str | None, str | None]]) -> None:
        """Auto-dismiss warnings matching any of the given (instance, model_id) pairs, in a single write.

        Equivalent to calling ``dismiss_warnings_matching`` once per pair, but does one read-modify-write
        cycle of services.json instead of one per pair.
        """

        async def handler(content: FileContent) -> Literal[False] | FileContent:
            before = len(content["warnings"])
            content["warnings"] = [
                w for w in content["warnings"] if not (w["service_id"] == service_id and (w["instance"], w["model_id"]) in keys)
            ]
            if len(content["warnings"]) == before:
                return False
            return content

        await self._modify(handler)

    async def dismiss_warnings_for_instance(self, service_id: str, instance: str) -> None:
        """Dismiss every warning for a service's instance regardless of which model it names.

        Used when the instance itself is uninstalled: at that point no single (instance, model_id)
        warning can ever auto-clear again since the model it names is gone too.
        """

        async def handler(content: FileContent) -> Literal[False] | FileContent:
            before = len(content["warnings"])
            content["warnings"] = [w for w in content["warnings"] if not (w["service_id"] == service_id and w["instance"] == instance)]
            if len(content["warnings"]) == before:
                return False
            return content

        await self._modify(handler)
