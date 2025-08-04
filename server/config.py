from typing import NamedTuple

class OllamaConfig(NamedTuple):
    cmd: str
    url: str

class ConfigValues(NamedTuple):
    port: int
    hostname: str
    ollama: OllamaConfig

class Config:
    def __init__(self):
        self.values = ConfigValues(
            port=3456,
            hostname="localhost",
            ollama=OllamaConfig(
                cmd="ollama",
                url="http://localhost:11434"
            )
        )