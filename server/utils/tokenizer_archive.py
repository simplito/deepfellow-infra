# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Pack a Hugging Face tokenizer into the zip doc-chunker registers (`POST /tokenizers`)."""

import asyncio
import io
import zipfile
from collections.abc import Mapping
from pathlib import Path

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
REQUIRED_FILES = ("tokenizer.json", "tokenizer_config.json")


def pack_tokenizer_files(files: Mapping[str, bytes]) -> bytes:
    """Zip tokenizer files, given by name, at the archive root."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def pack_tokenizer_dir(model_dir: Path) -> bytes | None:
    """Zip the tokenizer files of a model directory at the archive root, or return None when a required file is missing."""
    if not all((model_dir / name).is_file() for name in REQUIRED_FILES):
        return None
    return pack_tokenizer_files({name: (model_dir / name).read_bytes() for name in TOKENIZER_FILES if (model_dir / name).is_file()})


async def tokenizer_from_dir(model_dir: Path) -> bytes | None:
    """Pack `model_dir`'s tokenizer off the event loop; a `tokenizer.json` runs to tens of megabytes."""
    return await asyncio.to_thread(pack_tokenizer_dir, model_dir)
