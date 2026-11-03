# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Measure and cache how many tokens an Ollama embedding model embeds before it truncates."""

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from server.utils.core import fetch_from
from server.utils.logger import logger

_SHOW_TIMEOUT_SECONDS = 10
_VERSION_TIMEOUT_SECONDS = 5
_PS_TIMEOUT_SECONDS = 5
_MEASURE_TIMEOUT_SECONDS = 60
_MAX_MEASURED_TOKENS = 32768
_PROBE_WORD = " a"


class _CachedLimit(BaseModel):
    modified_at: str
    ollama_version: str = ""
    scope: str
    limit: int
    measured_at: float = 0.0


@dataclass
class _ShowInfo:
    modified_at: str
    context_length: int | None
    num_ctx: int | None


def _parse_num_ctx(parameters: object) -> int | None:
    """Read `num_ctx` from `/api/show`'s `parameters`, a text block of `name   value` lines."""
    if not isinstance(parameters, str):
        return None
    for line in parameters.splitlines():
        name, _, value = line.strip().partition(" ")
        if name == "num_ctx" and value.strip().isdigit():
            return int(value.strip())
    return None


def _positive_int(value: object) -> int | None:
    return value if isinstance(value, int) and value > 0 else None


class OllamaEmbeddingTokenLimits:
    """Resolve and remember the input token limit Ollama applies to each embedding model."""

    def __init__(self, cache_file: Path, max_age_seconds: float | None = None) -> None:
        """Remember results in `cache_file`, for at most `max_age_seconds` when given."""
        self._cache_file = cache_file
        self._max_age_seconds = max_age_seconds
        self._cache: dict[str, _CachedLimit] | None = None
        self._measure_locks: dict[str, asyncio.Lock] = {}

    async def resolve(
        self, instance: str, base_url: str, model_name: str, scope: str = "", context_ceiling: int | None = None
    ) -> int | None:
        """Return the tokens `model_name` embeds before Ollama truncates (measured, else the smallest known bound), or None."""
        show = await self._fetch_show(base_url, model_name)
        if show is None:
            return None

        ollama_version = await self._fetch_version(base_url)
        key = f"{instance}/{model_name}"
        cached = self._load().get(key)
        if cached and self._still_valid(cached, show.modified_at, ollama_version, scope):
            return cached.limit

        measured = await self._measure(base_url, model_name, show.context_length)
        if measured is not None:
            self._store(
                key,
                _CachedLimit(
                    modified_at=show.modified_at, ollama_version=ollama_version, scope=scope, limit=measured, measured_at=time.time()
                ),
            )
            return measured

        known = [v for v in (show.context_length, show.num_ctx, context_ceiling) if v is not None]
        return min(known) if known else None

    def has_fresh_measurement(self, instance: str, model_name: str) -> bool:
        """Tell, without asking Ollama, whether `model_name` was measured successfully and the result hasn't expired."""
        cached = self._load().get(f"{instance}/{model_name}")
        return cached is not None and self._not_expired(cached)

    def _still_valid(self, cached: _CachedLimit, modified_at: str, ollama_version: str, scope: str) -> bool:
        if (cached.modified_at, cached.ollama_version, cached.scope) != (modified_at, ollama_version, scope):
            return False
        return self._not_expired(cached)

    def _not_expired(self, cached: _CachedLimit) -> bool:
        return self._max_age_seconds is None or time.time() - cached.measured_at < self._max_age_seconds

    async def _fetch_show(self, base_url: str, model_name: str) -> _ShowInfo | None:
        try:
            result = await fetch_from(f"{base_url}/api/show", "POST", {"model": model_name}, timeout=_SHOW_TIMEOUT_SECONDS)
            if result.status_code != 200:
                logger.warning("Ollama /api/show for %r returned status %d", model_name, result.status_code)
                return None
            data = json.loads(result.data)
        except Exception:
            logger.warning("Ollama /api/show for %r failed", model_name, exc_info=True)
            return None
        if not isinstance(data, dict):
            logger.warning("Ollama /api/show for %r returned no object", model_name)
            return None
        info = data.get("model_info")
        if not isinstance(info, dict):
            info = {}
        arch = info.get("general.architecture", "")
        return _ShowInfo(
            modified_at=str(data.get("modified_at", "")),
            context_length=_positive_int(info.get(f"{arch}.context_length")),
            num_ctx=_parse_num_ctx(data.get("parameters")),
        )

    async def _fetch_version(self, base_url: str) -> str:
        """Return Ollama's version, or "" if it can't be read (an unknown version never matches a known one)."""
        try:
            result = await fetch_from(f"{base_url}/api/version", timeout=_VERSION_TIMEOUT_SECONDS)
            if result.status_code == 200:
                version = json.loads(result.data).get("version")
                if isinstance(version, str):
                    return version
        except Exception:
            logger.warning("Ollama /api/version failed", exc_info=True)
        return ""

    async def _is_loaded(self, base_url: str, model_name: str) -> bool | None:
        """Return whether `/api/ps` lists `model_name` as loaded, or None if it can't be read."""
        try:
            result = await fetch_from(f"{base_url}/api/ps", timeout=_PS_TIMEOUT_SECONDS)
            if result.status_code != 200:
                return None
            loaded = json.loads(result.data).get("models") or []
            names = {m.get("name") for m in loaded if isinstance(m, dict)}
        except Exception:
            logger.warning("Ollama /api/ps failed", exc_info=True)
            return None
        return model_name in names or f"{model_name}:latest" in names

    async def _measure(self, base_url: str, model_name: str, context_length: int | None) -> int | None:
        words = min(context_length or _MAX_MEASURED_TOKENS, _MAX_MEASURED_TOKENS)
        payload: dict[str, object] = {"model": model_name, "input": _PROBE_WORD * words, "truncate": True}
        try:
            async with self._measure_locks.setdefault(base_url, asyncio.Lock()):
                if await self._is_loaded(base_url, model_name) is False:
                    payload["keep_alive"] = 0
                result = await fetch_from(f"{base_url}/api/embed", "POST", payload, timeout=_MEASURE_TIMEOUT_SECONDS)
            if result.status_code != 200:
                logger.warning("Measuring the token limit of %r: /api/embed returned status %d", model_name, result.status_code)
                return None
            return _positive_int(json.loads(result.data).get("prompt_eval_count"))
        except Exception:
            logger.warning("Measuring the token limit of %r failed", model_name, exc_info=True)
            return None

    def _load(self) -> dict[str, _CachedLimit]:
        if self._cache is None:
            self._cache = {}
            try:
                raw = json.loads(self._cache_file.read_text())
                self._cache = {key: _CachedLimit.model_validate(value) for key, value in raw.items()}
            except FileNotFoundError:
                pass
            except Exception:
                logger.warning("Ignoring unreadable embedding token limit cache %s", self._cache_file, exc_info=True)
        return self._cache

    def _store(self, key: str, value: _CachedLimit) -> None:
        cache = self._load()
        cache[key] = value
        try:
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
            self._cache_file.write_text(json.dumps({k: v.model_dump() for k, v in cache.items()}, indent=2))
        except Exception:
            logger.warning("Could not save embedding token limit cache %s", self._cache_file, exc_info=True)
