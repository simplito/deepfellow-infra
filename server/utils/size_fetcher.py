# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for auto-fetching model sizes from remote sources."""

import logging
import re

import aiohttp

from server.utils.core import convert_size_to_bytes

logger = logging.getLogger("uvicorn.error")

HF_API = "https://huggingface.co/api"
OLLAMA_BASE = "https://ollama.com"
_HEADERS = {"User-Agent": "Mozilla/5.0"}


def fmt_size(n: int) -> str:
    """Get string size with unit."""
    size: float = n
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


async def _fetch_file_size_bytes(url: str) -> int | None:
    timeout = aiohttp.ClientTimeout(total=15)
    async with aiohttp.ClientSession() as session, session.head(url, headers=_HEADERS, allow_redirects=True, timeout=timeout) as resp:
        if resp.status != 200:
            return None
        content_length = resp.headers.get("Content-Length")
        return int(content_length) if content_length else None


async def _fetch_hf_size_bytes(hf_id: str) -> int | None:
    async with (
        aiohttp.ClientSession() as session,
        session.get(
            f"{HF_API}/models/{hf_id}",
            params={"blobs": "true"},
            headers=_HEADERS,
            timeout=aiohttp.ClientTimeout(total=15),
        ) as resp,
    ):
        resp.raise_for_status()
        data = await resp.json()
    total = sum(f.get("size", 0) for f in data.get("siblings", []) if f.get("size"))
    return total if total else None


def _parse_ollama_tag_size(html: str, tag: str) -> str | None:
    """Extract size string for a specific tag from ollama.com/library/{name}/tags HTML."""
    parts = html.split("group px-4 py-3")[1:]
    for part in parts:
        tag_m = re.search(r'href="/(?:library/)?([^"]+)"', part)
        if not tag_m:
            continue
        name = tag_m.group(1).removesuffix(":latest")
        col_spans = re.findall(r"<p[^>]*col-span-2[^>]*>(.*?)</p>", part, re.DOTALL)
        entry_tag = name.split(":")[-1] if ":" in name else "latest"
        if entry_tag == tag and col_spans:
            return col_spans[0].strip()
    return None


async def _fetch_ollama_size_bytes(model_id: str) -> int | None:
    if ":" in model_id:
        name, tag = model_id.rsplit(":", 1)
    else:
        name, tag = model_id, "latest"
    url = f"{OLLAMA_BASE}/library/{name}/tags"
    async with aiohttp.ClientSession() as session, session.get(url, headers=_HEADERS, timeout=aiohttp.ClientTimeout(total=15)) as resp:
        resp.raise_for_status()
        html = await resp.text()
    size_str = _parse_ollama_tag_size(html, tag)
    return convert_size_to_bytes(size_str) if size_str else None


def _is_hf_ref(ref: str) -> bool:
    return "huggingface.co" in ref or "hf.co" in ref or re.match(r"^[^/\s:]+/[^/\s]+$", ref) is not None


def _hf_id_from_ref(ref: str) -> str:
    return ref.split("huggingface.co/")[-1].split("hf.co/")[-1]


async def fetch_ollama_ref_bytes(ref: str) -> int | None:
    """Return byte size for a remote Ollama model reference (HuggingFace or Ollama library).

    Local file paths are not handled here — the caller must resolve them via Docker exec.
    """
    if ref.startswith(("/", "./")):
        return None
    if _is_hf_ref(ref):
        return await _fetch_hf_size_bytes(_hf_id_from_ref(ref))
    return await _fetch_ollama_size_bytes(ref)


async def fetch_file_size_from_url(url: str) -> str | None:
    """Return human-readable file size via HTTP HEAD Content-Length, or None on failure."""
    try:
        n = await _fetch_file_size_bytes(url)
        return fmt_size(n) if n is not None else None
    except Exception:
        logger.warning("Failed to fetch file size from URL: %s", url)
        return None


async def fetch_huggingface_model_size(hf_id: str) -> str | None:
    """Return human-readable total size from HuggingFace Hub API, or None on failure."""
    try:
        n = await _fetch_hf_size_bytes(hf_id)
        return fmt_size(n) if n is not None else None
    except Exception:
        logger.warning("Failed to fetch HuggingFace model size for: %s", hf_id)
        return None


async def fetch_ollama_model_size(model_id: str) -> str | None:
    """Return size string for an Ollama library model (e.g. 'gemma3:1b'), or None on failure."""
    try:
        n = await _fetch_ollama_size_bytes(model_id)
        return fmt_size(n) if n is not None else None
    except Exception:
        logger.warning("Failed to fetch Ollama model size for: %s", model_id)
        return None


async def fetch_ollama_modelfile_size(modelfile_text: str) -> str | None:
    """Return total human-readable size for all FROM and ADAPTER references in a Modelfile.

    Sums the sizes of every FROM and ADAPTER directive. Returns None if nothing could be resolved.
    """
    refs: list[str] = []
    for line in modelfile_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("FROM "):
            refs.append(stripped[5:].strip())
        elif stripped.startswith("ADAPTER "):
            refs.append(stripped[8:].strip())

    if not refs:
        return None

    total = 0
    any_resolved = False
    for ref in refs:
        try:
            n = await fetch_ollama_ref_bytes(ref)
            if n is not None:
                total += n
                any_resolved = True
        except Exception:
            logger.warning("Failed to resolve Ollama modelfile reference size: %s", ref)

    return fmt_size(total) if any_resolved else None
