# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Serve the tokenizer of an Ollama embedding model, rebuilt from its GGUF file (`/api/show` verbose)."""

import asyncio
import json

from server.utils.core import fetch_from
from server.utils.gguf_tokenizer import build_gguf_tokenizer, tokenizer_files
from server.utils.logger import logger
from server.utils.tokenizer_archive import pack_tokenizer_files

_SHOW_TIMEOUT_SECONDS = 60


class OllamaTokenizerError(Exception):
    """Ollama could not be asked for the model's tokenizer; it may still have one."""


def _show_object(text: str, model_name: str) -> dict[str, object]:
    try:
        data = json.loads(text)
    except ValueError as e:
        msg = f"Ollama /api/show for {model_name!r} returned invalid JSON"
        raise OllamaTokenizerError(msg) from e
    if not isinstance(data, dict):
        msg = f"Ollama /api/show for {model_name!r} returned no object"
        raise OllamaTokenizerError(msg)
    return data  # pyright: ignore[reportUnknownVariableType]


def _archive_from_show(text: str, model_name: str) -> bytes | None:
    """Build the zip from a verbose `/api/show` answer, or None when its tokenizer is unsupported or broken; out of memory raises."""
    model_info = _show_object(text, model_name).get("model_info")
    if not isinstance(model_info, dict):
        msg = f"Ollama /api/show for {model_name!r} returned no model_info"
        raise OllamaTokenizerError(msg)
    try:
        tokenizer = build_gguf_tokenizer(model_info)  # pyright: ignore[reportUnknownArgumentType]
        if tokenizer is None:
            logger.info("No tokenizer for Ollama model %r: unsupported GGUF tokenizer", model_name)
            return None
        return pack_tokenizer_files(tokenizer_files(model_info, tokenizer))  # pyright: ignore[reportUnknownArgumentType]
    except MemoryError:
        raise
    except Exception:
        logger.exception("Rebuilding the tokenizer of Ollama model %r failed", model_name)
        return None


class OllamaTokenizers:
    """Build and remember the tokenizer zip of each Ollama model, until the model changes."""

    def __init__(self) -> None:
        self._archives: dict[tuple[str, str], tuple[str, bytes | None]] = {}
        self._locks: dict[tuple[str, str], asyncio.Lock] = {}

    async def archive(self, base_url: str, model_name: str) -> bytes | None:
        """Return the zip for `model_name`, or None if Ollama has no tokenizer for it; raise `OllamaTokenizerError` if it can't tell."""
        key = (base_url, model_name)
        async with self._locks.setdefault(key, asyncio.Lock()):
            plain = await self._fetch_show(base_url, model_name, verbose=False)
            if plain is None:
                return None
            modified_at = str(_show_object(plain, model_name).get("modified_at", ""))
            cached = self._archives.get(key)
            if cached is not None and cached[0] == modified_at:
                return cached[1]
            verbose = await self._fetch_show(base_url, model_name, verbose=True)
            if verbose is None:
                return None
            archive = await asyncio.to_thread(_archive_from_show, verbose, model_name)
            self._archives[key] = (modified_at, archive)
            return archive

    @staticmethod
    async def _fetch_show(base_url: str, model_name: str, *, verbose: bool) -> str | None:
        """Return `/api/show`'s body, or None when Ollama doesn't know the model (removed outside Infra)."""
        try:
            result = await fetch_from(
                f"{base_url}/api/show", "POST", {"model": model_name, "verbose": verbose}, timeout=_SHOW_TIMEOUT_SECONDS
            )
        except Exception as e:
            msg = f"Ollama /api/show for {model_name!r} failed: {e}"
            raise OllamaTokenizerError(msg) from e
        if result.status_code == 404:
            return None
        if result.status_code != 200:
            msg = f"Ollama /api/show for {model_name!r} returned status {result.status_code}"
            raise OllamaTokenizerError(msg)
        return result.data
