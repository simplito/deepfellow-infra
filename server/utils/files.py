# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""GGuf utils."""

import json
import struct
from contextlib import suppress
from pathlib import Path
from typing import Any

import aiofiles

from server.utils.vram_calculator import ArchParams

_GGUF_SCALAR_FMTS: dict[int, str] = {
    0: "<B",
    7: "<B",  # UINT8, BOOL
    1: "<b",  # INT8
    2: "<H",
    3: "<h",  # UINT16, INT16
    4: "<I",
    5: "<i",
    6: "<f",  # UINT32, INT32, FLOAT32
    10: "<Q",
    11: "<q",
    12: "<d",  # UINT64, INT64, FLOAT64
}


def _parse_gguf_value(data: bytes, offset: int, val_type: int) -> tuple[Any, int]:
    match val_type:
        case t if fmt := _GGUF_SCALAR_FMTS.get(t):
            return struct.unpack_from(fmt, data, offset)[0], offset + struct.calcsize(fmt)
        case 8:  # STRING
            str_len = struct.unpack_from("<Q", data, offset)[0]
            offset += 8
            val = data[offset : offset + str_len].decode("utf-8", errors="replace")
            return val, offset + str_len
        case 9:  # ARRAY
            item_type = struct.unpack_from("<I", data, offset)[0]
            count = struct.unpack_from("<Q", data, offset + 4)[0]
            offset += 12
            for _ in range(count):
                _, offset = _parse_gguf_value(data, offset, item_type)
            return None, offset
        case _:
            msg = f"Unknown GGUF value type: {val_type}"
            raise ValueError(msg)


def _parse_gguf_metadata(data: bytes) -> dict[str, Any]:
    """Parse key-value metadata from a GGUF binary blob.

    GGUF is the binary format used by llama.cpp for quantized model weights.
    Spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

    The header layout (little-endian):a
      bytes 0-3   magic "GGUF"
      bytes 4-7   version (uint32, must be 1/2/3)
      bytes 8-11  tensor count (uint64, skipped here)
      bytes 12-19 n_kv — number of metadata key-value pairs (uint64)
      bytes 20+   key-value pairs: each is (str_key, uint32_type, value)

    Example output for a Llama 3 8B Q4_K_M file:
      {
        "general.architecture":          "llama",
        "llama.context_length":          131072,
        "llama.embedding_length":        4096,
        "llama.block_count":             32,
        "llama.attention.head_count":    32,
        "llama.attention.head_count_kv": 8,
      }
    """
    if data[:4] != b"GGUF":
        return {}
    version = struct.unpack_from("<I", data, 4)[0]
    if version not in (1, 2, 3):
        return {}
    n_kv = struct.unpack_from("<Q", data, 12)[0]
    offset = 24
    metadata: dict[str, Any] = {}
    for _ in range(n_kv):
        if offset + 8 > len(data):
            break
        key_len = struct.unpack_from("<Q", data, offset)[0]
        offset += 8
        if offset + key_len > len(data):
            break
        key = data[offset : offset + key_len].decode("utf-8", errors="replace")
        offset += key_len
        if offset + 4 > len(data):
            break
        val_type = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        try:
            value, offset = _parse_gguf_value(data, offset, val_type)
            metadata[key] = value
        except Exception:
            break
    return metadata


async def get_gguf_arch_params(file_path: Path | str) -> ArchParams | None:
    """Parse hidden_size, attention heads and layer count from a GGUF file."""
    with suppress(Exception):
        async with aiofiles.open(file_path, mode="rb") as f:
            chunk = await f.read(1_000_000)

        metadata = _parse_gguf_metadata(chunk)

        def find_int(suffix: str) -> int | None:
            return next((v for k, v in metadata.items() if k.endswith(suffix) and isinstance(v, int)), None)

        hidden_size = find_int(".embedding_length")
        num_heads = find_int(".attention.head_count")
        num_kv_heads = find_int(".attention.head_count_kv")
        num_layers = find_int(".block_count")

        if hidden_size is not None and num_heads is not None and num_kv_heads is not None and num_layers is not None:
            return ArchParams(
                hidden_size=hidden_size,
                num_attention_heads=num_heads,
                num_key_value_heads=num_kv_heads,
                num_hidden_layers=num_layers,
            )

    return None


async def get_gguf_context_window(file_path: Path | str) -> int | None:
    """Find context length in a .gguf file."""
    try:
        async with aiofiles.open(file_path, mode="rb") as f:
            chunk = await f.read(1_000_000)

        metadata = _parse_gguf_metadata(chunk)
        return next((v for k, v in metadata.items() if k.endswith(".context_length") and isinstance(v, int)), None)

    except Exception:
        return None


async def get_model_dir_context_window(model_dir: Path | str) -> int | None:
    """Read max_position_embeddings from config.json."""
    model_dir_path = Path(model_dir) if isinstance(model_dir, str) else model_dir
    config_path = model_dir_path / "config.json"

    try:
        async with aiofiles.open(config_path) as f:
            content = await f.read()
            config_data = json.loads(content)

            # Extract the specific parameter
            value = config_data.get("max_position_embeddings")

            return int(value) if value is not None else None

    except (FileNotFoundError, json.JSONDecodeError, ValueError, KeyError):
        # Gracefully handle missing files, malformed JSON, or non-int values
        return None


async def detect_context_window_from_path(model_path: Path | str) -> int | None:
    """Detect context window under given path."""
    if str(model_path).endswith(".gguf"):
        return await get_gguf_context_window(model_path)
    return await get_model_dir_context_window(model_path)
