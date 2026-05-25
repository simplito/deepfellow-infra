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
    _parse_gguf_metadata,  # pyright: ignore[reportPrivateUsage]
    _parse_gguf_value,  # pyright: ignore[reportPrivateUsage]
    detect_context_window_from_path,
    get_gguf_arch_params,
    get_gguf_context_window,
    get_model_dir_context_window,
)
from server.utils.vram_calculator import ArchParams


def _kv_string(key: str, value: str) -> bytes:
    key_bytes = key.encode()
    val_bytes = value.encode()
    return struct.pack("<Q", len(key_bytes)) + key_bytes + struct.pack("<I", 8) + struct.pack("<Q", len(val_bytes)) + val_bytes


def _kv_uint32(key: str, value: int) -> bytes:
    key_bytes = key.encode()
    return struct.pack("<Q", len(key_bytes)) + key_bytes + struct.pack("<I", 4) + struct.pack("<I", value)


def _build_gguf(kv_pairs: list[bytes], version: int = 3) -> bytes:
    header = b"GGUF" + struct.pack("<I", version) + struct.pack("<Q", 0) + struct.pack("<Q", len(kv_pairs))
    return header + b"".join(kv_pairs)


def _write_gguf(tmp_path: Path, kv_pairs: list[bytes]) -> Path:
    path = tmp_path / "model.gguf"
    path.write_bytes(_build_gguf(kv_pairs))
    return path


def _kv_int32(key: str, value: int) -> bytes:
    key_bytes = key.encode()
    return struct.pack("<Q", len(key_bytes)) + key_bytes + struct.pack("<I", 5) + struct.pack("<i", value)


@pytest.mark.parametrize(
    ("val_type", "raw", "expected"),
    [
        (0, struct.pack("<B", 1), 1),  # UINT8 / BOOL
        (1, struct.pack("<b", -5), -5),  # INT8
        (4, struct.pack("<I", 4096), 4096),  # UINT32
        (5, struct.pack("<i", -1), -1),  # INT32
        (10, struct.pack("<Q", 131072), 131072),  # UINT64
    ],
)
def test_parse_gguf_value_scalars(val_type: int, raw: bytes, expected: int):
    value, offset = _parse_gguf_value(raw, 0, val_type)

    assert value == expected
    assert offset == len(raw)


def test_parse_gguf_value_string():
    text = "llama"
    raw = struct.pack("<Q", len(text)) + text.encode()

    value, offset = _parse_gguf_value(raw, 0, 8)

    assert value == text
    assert offset == len(raw)


def test_parse_gguf_value_unknown_type_raises():
    with pytest.raises(ValueError, match="Unknown GGUF value type"):
        _parse_gguf_value(b"\x00" * 8, 0, 99)


def test_parse_gguf_metadata_valid():
    data = _build_gguf(
        [
            _kv_string("general.architecture", "llama"),
            _kv_uint32("llama.embedding_length", 4096),
            _kv_uint32("llama.block_count", 32),
            _kv_uint32("llama.attention.head_count", 32),
            _kv_uint32("llama.attention.head_count_kv", 8),
            _kv_uint32("llama.context_length", 131072),
        ]
    )

    result = _parse_gguf_metadata(data)

    assert result["general.architecture"] == "llama"
    assert result["llama.embedding_length"] == 4096
    assert result["llama.block_count"] == 32
    assert result["llama.attention.head_count"] == 32
    assert result["llama.attention.head_count_kv"] == 8
    assert result["llama.context_length"] == 131072


def test_parse_gguf_metadata_wrong_magic():
    result = _parse_gguf_metadata(b"NOTG" + b"\x00" * 20)

    assert result == {}


@pytest.mark.parametrize("version", [0, 4, 99])
def test_parse_gguf_metadata_unsupported_version(version: int):
    data = b"GGUF" + struct.pack("<I", version) + b"\x00" * 16

    result = _parse_gguf_metadata(data)

    assert result == {}


def test_parse_gguf_metadata_empty_kv():
    result = _parse_gguf_metadata(_build_gguf([]))

    assert result == {}


def test_parse_gguf_metadata_truncated_returns_partial():
    full = _build_gguf(
        [
            _kv_string("general.architecture", "llama"),
            _kv_uint32("llama.embedding_length", 4096),
        ]
    )

    result = _parse_gguf_metadata(full[: len(full) - 4])

    assert "general.architecture" in result


@pytest.mark.asyncio
async def test_get_gguf_arch_params_valid(tmp_path: Path):
    path = _write_gguf(
        tmp_path,
        [
            _kv_string("general.architecture", "llama"),
            _kv_uint32("llama.embedding_length", 4096),
            _kv_uint32("llama.attention.head_count", 32),
            _kv_uint32("llama.attention.head_count_kv", 8),
            _kv_uint32("llama.block_count", 32),
        ],
    )

    result = await get_gguf_arch_params(path)

    assert result == ArchParams(
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        num_hidden_layers=32,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "missing_key",
    [
        "llama.embedding_length",
        "llama.attention.head_count",
        "llama.attention.head_count_kv",
        "llama.block_count",
    ],
)
async def test_get_gguf_arch_params_missing_field_returns_none(tmp_path: Path, missing_key: str):
    all_kv = {
        "llama.embedding_length": _kv_uint32("llama.embedding_length", 4096),
        "llama.attention.head_count": _kv_uint32("llama.attention.head_count", 32),
        "llama.attention.head_count_kv": _kv_uint32("llama.attention.head_count_kv", 8),
        "llama.block_count": _kv_uint32("llama.block_count", 32),
    }
    kv_pairs = [v for k, v in all_kv.items() if k != missing_key]
    path = _write_gguf(tmp_path, kv_pairs)

    result = await get_gguf_arch_params(path)

    assert result is None


@pytest.mark.asyncio
async def test_get_gguf_arch_params_nonexistent_file_returns_none(tmp_path: Path):
    result = await get_gguf_arch_params(tmp_path / "nonexistent.gguf")

    assert result is None


@pytest.mark.asyncio
async def test_get_gguf_arch_params_invalid_magic_returns_none(tmp_path: Path):
    path = tmp_path / "bad.gguf"
    path.write_bytes(b"NOTG" + b"\x00" * 50)

    result = await get_gguf_arch_params(path)

    assert result is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kv_bytes", "expected"),
    [
        (_kv_uint32("llama.context_length", 2048), 2048),
        (_kv_uint32("llama.context_length", 4096), 4096),
        (_kv_int32("llama.context_length", 8192), 8192),
        (_kv_uint32("llama.context_length", 32768), 32768),
    ],
)
async def test_get_gguf_context_window_returns_context(tmp_path: Path, kv_bytes: bytes, expected: int) -> None:
    file = tmp_path / "model.gguf"
    file.write_bytes(_build_gguf([kv_bytes]))

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
    file = _write_gguf(tmp_path, [_kv_uint32("llama.context_length", 16384)])

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


def test_parse_gguf_value_array() -> None:
    # item_type=4 (UINT32), count=2, two uint32 items
    raw = struct.pack("<I", 4) + struct.pack("<Q", 2) + struct.pack("<I", 100) + struct.pack("<I", 200)

    value, offset = _parse_gguf_value(raw, 0, 9)

    assert value is None
    assert offset == 4 + 8 + 4 + 4


def test_parse_gguf_metadata_breaks_when_key_len_exceeds_data() -> None:
    # Header claims n_kv=1, then key_len points beyond remaining data
    header = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", 1)
    header += struct.pack("<Q", 9999)  # key_len too large

    result = _parse_gguf_metadata(header)

    assert result == {}


def test_parse_gguf_metadata_breaks_when_no_room_for_val_type() -> None:
    # Header with one valid key but data ends before val_type (4 bytes) can be read
    key = b"test.key"
    header = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", 1)
    header += struct.pack("<Q", len(key)) + key  # key_len + key, but no val_type follows

    result = _parse_gguf_metadata(header)

    assert result == {}
