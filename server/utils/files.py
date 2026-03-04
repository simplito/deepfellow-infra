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
import re
import struct
from pathlib import Path

import aiofiles


async def get_gguf_context_window(file_path: Path | str) -> int | None:
    """Find context length in .gguf file."""
    # Read the first 1MB to find metadata
    bytes_to_read = 1_000_000

    # Regex for bytes: looks for any sequence ending in 'context_length'
    pattern = re.compile(rb"[a-zA-Z0-9_]*\.context_length")

    try:
        async with aiofiles.open(file_path, mode="rb") as f:
            # Read the first 1MB of data.
            chunk = await f.read(bytes_to_read)

            match = pattern.search(chunk)
            if not match:
                return None

            # Get the position where the key ends
            key_end_pos = match.end()

            # After the key name string, there is a 4-byte integer (Type ID)
            type_offset = key_end_pos

            # Ensure we have enough bytes left in the chunk to read the type and value
            if len(chunk) < type_offset + 8:
                return None

            val_type = struct.unpack("<I", chunk[type_offset : type_offset + 4])[0]

            # Read the actual value based on the type
            value_offset = type_offset + 4

            if val_type == 4:  # UINT32
                context = struct.unpack("<I", chunk[value_offset : value_offset + 4])[0]
            elif val_type == 5:  # INT32
                context = struct.unpack("<i", chunk[value_offset : value_offset + 4])[0]
            elif val_type == 6:  # UINT64
                context = struct.unpack("<Q", chunk[value_offset : value_offset + 8])[0]
            else:
                return None

            return int(context)

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
