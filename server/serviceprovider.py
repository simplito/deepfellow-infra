import json
from typing import TypedDict, List, Callable, Awaitable, Union
import asyncio
from pathlib import Path

def array_equals(a: list[str], b: list[str]) -> bool:
    return a == b

class BootstrapCommand(TypedDict):
    args: List[str]

class FileContent(TypedDict):
    bootstrapCommands: List[BootstrapCommand]


class ServiceProvider:
    def get_file_path(self) -> str:
        return (Path(__file__).parent / "../../storage/services.json").resolve()

    def load(self) -> FileContent:
        fpath = self.get_file_path()
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
                return json.loads(content)
        except FileNotFoundError:
            return {"bootstrapCommands": []}
        except Exception as e:
            raise e

    async def save(self, content: FileContent):
        fpath = self.get_file_path()
        json_data = json.dumps(content, indent=2)
        await asyncio.to_thread(self._write_file, fpath, json_data)

    def _write_file(self, path: str, data: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(data)

    async def modify(
        self,
        func: Callable[[FileContent], Union[bool, FileContent, Awaitable[Union[bool, FileContent]]]]
    ):
        content = self.load()
        maybe_new_content = func(content)
        if asyncio.iscoroutine(maybe_new_content):
            maybe_new_content = await maybe_new_content
        if maybe_new_content is not False:
            await self.save(maybe_new_content)

    async def save_command(self, args: List[str]):
        async def handler(content: FileContent):
            if any(array_equals(cmd["args"], args) for cmd in content["bootstrapCommands"]):
                return False
            content["bootstrapCommands"].append({"args": args})
            return content
        await self.modify(handler)

    async def remove_command(self, args: List[str]):
        async def handler(content: FileContent):
            index = next((i for i, cmd in enumerate(content["bootstrapCommands"])
                          if array_equals(cmd["args"], args)), -1)
            if index == -1:
                return False
            content["bootstrapCommands"].pop(index)
            return content
        await self.modify(handler)