# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Tests for server/utils/tokenizer_archive.py."""

import io
import zipfile
from pathlib import Path

import pytest

from server.utils.tokenizer_archive import pack_tokenizer_dir, pack_tokenizer_files, tokenizer_from_dir


def _write(model_dir: Path, *names: str) -> None:
    model_dir.mkdir(exist_ok=True)
    for name in names:
        (model_dir / name).write_text(f"content of {name}")


def _names(archive: bytes) -> set[str]:
    return set(zipfile.ZipFile(io.BytesIO(archive)).namelist())


def test_pack_tokenizer_dir_keeps_only_tokenizer_files_at_root(tmp_path: Path) -> None:
    _write(tmp_path, "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "config.json", "model.safetensors")

    archive = pack_tokenizer_dir(tmp_path)

    assert archive is not None
    assert _names(archive) == {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"}
    assert zipfile.ZipFile(io.BytesIO(archive)).read("tokenizer.json") == b"content of tokenizer.json"


def test_pack_tokenizer_dir_skips_missing_optional_files(tmp_path: Path) -> None:
    _write(tmp_path, "tokenizer.json", "tokenizer_config.json")

    archive = pack_tokenizer_dir(tmp_path)

    assert archive is not None
    assert _names(archive) == {"tokenizer.json", "tokenizer_config.json"}


def test_pack_tokenizer_dir_without_tokenizer_config_returns_none(tmp_path: Path) -> None:
    _write(tmp_path, "tokenizer.json", "special_tokens_map.json")

    assert pack_tokenizer_dir(tmp_path) is None


def test_pack_tokenizer_dir_without_tokenizer_json_returns_none(tmp_path: Path) -> None:
    _write(tmp_path, "tokenizer.model", "tokenizer_config.json")

    assert pack_tokenizer_dir(tmp_path) is None


@pytest.mark.asyncio
async def test_tokenizer_from_dir_packs_off_the_event_loop(tmp_path: Path) -> None:
    _write(tmp_path, "tokenizer.json", "tokenizer_config.json")

    archive = await tokenizer_from_dir(tmp_path)

    assert archive is not None
    assert _names(archive) == {"tokenizer.json", "tokenizer_config.json"}


def test_pack_tokenizer_files_writes_given_files_at_root() -> None:
    archive = pack_tokenizer_files({"tokenizer.json": b"{}", "tokenizer_config.json": b"{}"})

    assert _names(archive) == {"tokenizer.json", "tokenizer_config.json"}
    assert zipfile.ZipFile(io.BytesIO(archive)).read("tokenizer.json") == b"{}"
