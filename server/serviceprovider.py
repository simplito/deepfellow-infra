"""Module which load and save settings."""

import asyncio
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Literal, TypedDict, cast


class BootstrapCommand(TypedDict):
    args: list[str]


class FileContent(TypedDict):
    bootstrapCommands: list[BootstrapCommand]


class ServiceProvider:
    def _get_file_path(self) -> Path:
        return (Path(__file__).parent / "../storage/services.json").resolve()

    def load(self) -> FileContent:
        """Load settings file content."""
        fpath = self._get_file_path()
        try:
            with fpath.open(encoding="utf-8") as f:
                content = f.read()
                return json.loads(content)
        except FileNotFoundError:
            return {"bootstrapCommands": []}

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

    async def save_command(self, args: list[str]) -> None:
        """Add command to bootstrap."""

        async def handler(content: FileContent) -> Literal[False] | FileContent:
            if any(cmd["args"] == args for cmd in content["bootstrapCommands"]):
                return False
            content["bootstrapCommands"].append({"args": args})
            return content

        await self._modify(handler)

    async def remove_command(self, args: list[str]) -> None:
        """Remove command from bootstrap."""

        async def handler(content: FileContent) -> Literal[False] | FileContent:
            index = next((i for i, cmd in enumerate(content["bootstrapCommands"]) if cmd["args"] == args), -1)
            if index == -1:
                return False
            content["bootstrapCommands"].pop(index)
            return content

        await self._modify(handler)
