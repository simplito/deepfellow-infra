# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import struct
from pathlib import Path

import pytest

from server.utils.files import (
    detect_context_window_from_path,
    get_gguf_context_window,
    get_model_dir_context_window,
)


def _make_gguf_chunk(key: bytes, val_type: int, value: int) -> bytes:
    """Build a minimal GGUF-like byte sequence with a context_length key."""
    if val_type == 4:
        value_bytes = struct.pack("<I", value)
    elif val_type == 5:
        value_bytes = struct.pack("<i", value)
    else:
        value_bytes = struct.pack("<Q", value)
    return key + struct.pack("<I", val_type) + value_bytes


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("value_type", "expected"),
    [
        (4, 2048),
        (4, 4096),
        (5, 8192),
        (6, 32768),
    ],
)
async def test_get_gguf_context_window_returns_context(tmp_path: Path, value_type: int, expected: int) -> None:
    key = b"llama.context_length"
    chunk = _make_gguf_chunk(key, value_type, expected)
    file = tmp_path / "model.gguf"
    file.write_bytes(chunk)

    result = await get_gguf_context_window(file)

    assert result == expected


@pytest.mark.asyncio
async def test_get_gguf_context_window_returns_none_when_pattern_not_found(tmp_path: Path) -> None:
    file = tmp_path / "model.gguf"
    file.write_bytes(b"\x00" * 64)

    result = await get_gguf_context_window(file)

    assert result is None


@pytest.mark.asyncio
async def test_get_gguf_context_window_returns_none_for_unknown_val_type(tmp_path: Path) -> None:
    key = b"llama.context_length"
    chunk = key + struct.pack("<I", 99) + struct.pack("<I", 1234)
    file = tmp_path / "model.gguf"
    file.write_bytes(chunk)

    result = await get_gguf_context_window(file)

    assert result is None


@pytest.mark.asyncio
async def test_get_gguf_context_window_returns_none_when_chunk_too_short_after_key(tmp_path: Path) -> None:
    file = tmp_path / "model.gguf"
    file.write_bytes(b"llama.context_length" + b"\x00" * 4)

    result = await get_gguf_context_window(file)

    assert result is None


@pytest.mark.asyncio
async def test_get_gguf_context_window_returns_none_on_file_not_found() -> None:
    result = await get_gguf_context_window(Path("/nonexistent/model.gguf"))

    assert result is None


@pytest.mark.asyncio
async def test_get_model_dir_context_window_reads_max_position_embeddings(tmp_path: Path) -> None:
    config = {"max_position_embeddings": 2048}
    (tmp_path / "config.json").write_text(json.dumps(config))

    result = await get_model_dir_context_window(tmp_path)

    assert result == 2048


@pytest.mark.asyncio
async def test_get_model_dir_context_window_returns_none_when_key_missing(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"hidden_size": 768}))

    result = await get_model_dir_context_window(tmp_path)

    assert result is None


@pytest.mark.asyncio
async def test_get_model_dir_context_window_returns_none_when_file_not_found(tmp_path: Path) -> None:
    result = await get_model_dir_context_window(tmp_path)

    assert result is None


@pytest.mark.asyncio
async def test_get_model_dir_context_window_returns_none_on_malformed_json(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("not json{{{")

    result = await get_model_dir_context_window(tmp_path)

    assert result is None


@pytest.mark.asyncio
async def test_get_model_dir_context_window_returns_none_on_non_int_value(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"max_position_embeddings": "big"}))

    result = await get_model_dir_context_window(tmp_path)

    assert result is None


@pytest.mark.asyncio
async def test_get_model_dir_context_window_accepts_string_path(tmp_path: Path) -> None:
    config = {"max_position_embeddings": 512}
    (tmp_path / "config.json").write_text(json.dumps(config))

    result = await get_model_dir_context_window(str(tmp_path))

    assert result == 512


@pytest.mark.asyncio
async def test_detect_context_window_dispatches_to_gguf_for_gguf_extension(tmp_path: Path) -> None:
    key = b"llama.context_length"
    chunk = _make_gguf_chunk(key, 4, 16384)
    file = tmp_path / "model.gguf"
    file.write_bytes(chunk)

    result = await detect_context_window_from_path(file)

    assert result == 16384


@pytest.mark.asyncio
async def test_detect_context_window_dispatches_to_model_dir_for_non_gguf(tmp_path: Path) -> None:
    config = {"max_position_embeddings": 1024}
    (tmp_path / "config.json").write_text(json.dumps(config))

    result = await detect_context_window_from_path(tmp_path)

    assert result == 1024


@pytest.mark.asyncio
async def test_detect_context_window_dispatches_to_model_dir_for_string_non_gguf(tmp_path: Path) -> None:
    config = {"max_position_embeddings": 4096}
    (tmp_path / "config.json").write_text(json.dumps(config))

    result = await detect_context_window_from_path(str(tmp_path))

    assert result == 4096


@pytest.mark.asyncio
async def test_detect_context_window_returns_none_for_gguf_without_context(tmp_path: Path) -> None:
    file = tmp_path / "model.gguf"
    file.write_bytes(b"\x00" * 64)

    result = await detect_context_window_from_path(file)

    assert result is None
