# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Get HuggingFace Models info.

Uses the HuggingFace Hub API (no API key required for public data).
Outputs a vllm-min.json compatible JSON to stdout.
"""

import argparse
import asyncio
import json
import sys
from typing import Any

import aiohttp

# ruff: noqa: T201

HF_API = "https://huggingface.co/api"
LLM_TAGS = ["text-generation", "text2text-generation"]
CONCURRENCY = 10  # parallel size-fetch requests
PAGE_SIZE = 100  # max models per API page (HF limit)


def fmt_size(n: int) -> str:
    """Format byte count as a human-readable string."""
    size: float = n
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


def fmt_size_compact(size_str: str) -> str:
    """Convert '2.0 GB' to '2GB'."""
    if size_str == "N/A":
        return size_str
    parts = size_str.split()
    if len(parts) == 2:
        return f"{int(float(parts[0]))}{parts[1]}"
    return size_str


def is_llm(model: dict[str, Any]) -> bool:
    """Filter out non-LLM models using a basic heuristic."""
    name = model.get("id", "").lower()
    exclude = ["translation", "summarization", "classification", "ner", "qa-"]
    return not any(kw in name for kw in exclude)


def is_reranker(model: dict[str, Any]) -> bool:
    """Filter reranker models by name heuristic."""
    name = model.get("id", "").lower()
    return "rerank" in name


async def fetch_popular_models(session: aiohttp.ClientSession, tag: str, sort: str, limit: int) -> list[dict[str, Any]]:
    """Fetch popular LLMs from HuggingFace API sorted by the given criterion."""
    params = {
        "pipeline_tag": tag,
        "sort": sort,
        "direction": "-1",
        "limit": str(limit),
        "cardData": "true",
    }
    async with session.get(f"{HF_API}/models", params=params) as resp:
        resp.raise_for_status()
        return await resp.json()


async def fetch_popular_reranker_models(session: aiohttp.ClientSession, sort: str, limit: int) -> list[dict[str, Any]]:
    """Fetch popular reranker models from HuggingFace API sorted by the given criterion."""
    params = {
        "other": "reranker",
        "sort": sort,
        "direction": "-1",
        "limit": str(limit),
        "cardData": "true",
    }
    async with session.get(f"{HF_API}/models", params=params) as resp:
        resp.raise_for_status()
        return await resp.json()


async def collect_llm_models(session: aiohttp.ClientSession, active: dict[str, int]) -> list[dict[str, Any]]:
    """Fetch and deduplicate LLM models across all sort/tag combinations."""
    fetch_tasks = [fetch_popular_models(session, tag, sort, limit) for sort, limit in active.items() for tag in LLM_TAGS]
    results = await asyncio.gather(*fetch_tasks)
    seen: set[str] = set()
    models: list[dict[str, Any]] = []
    for batch in results:
        for m in batch:
            mid = m["id"]
            if mid not in seen and is_llm(m):
                seen.add(mid)
                models.append(m)
    return models


async def collect_reranker_models(session: aiohttp.ClientSession, active: dict[str, int]) -> list[dict[str, Any]]:
    """Fetch and deduplicate reranker models across all sort combinations."""
    fetch_tasks = [fetch_popular_reranker_models(session, sort, limit) for sort, limit in active.items()]
    results = await asyncio.gather(*fetch_tasks)
    seen: set[str] = set()
    models: list[dict[str, Any]] = []
    for batch in results:
        for m in batch:
            mid = m["id"]
            if mid not in seen and is_reranker(m):
                seen.add(mid)
                models.append(m)
    return models


async def fetch_model_size(session: aiohttp.ClientSession, model_id: str, sem: asyncio.Semaphore) -> tuple[str, str]:
    """Return (model_id, human_readable_size)."""
    async with sem:
        try:
            async with session.get(f"{HF_API}/models/{model_id}", params={"blobs": "true"}) as resp:
                resp.raise_for_status()
                data = await resp.json()
            total = sum(f.get("size", 0) for f in data.get("siblings", []) if f.get("size"))
            return model_id, fmt_size(total) if total else "N/A"
        except Exception:
            return model_id, "N/A"


async def main(top_by_downloads: int, top_by_likes: int, top_by_trending: int, raw: bool, model_type: str) -> None:
    """Fetch models from HuggingFace and print a vllm-min.json compatible registry."""

    def log(msg: str) -> None:
        if not raw:
            print(msg, file=sys.stderr)

    active = {
        k: v
        for k, v in {
            "downloads": top_by_downloads,
            "likes": top_by_likes,
            "trendingScore": top_by_trending,
        }.items()
        if v > 0
    }

    if not active:
        print("Specify at least one of: --top-by-downloads, --top-by-likes, --top-by-trending", file=sys.stderr)
        return

    label = {"downloads": "downloads", "likes": "likes", "trendingScore": "trending"}
    summary = ", ".join(f"top {n} by {label[s]}" for s, n in active.items())
    log(f"Fetching {model_type} models from HuggingFace ({summary})...\n")

    async with aiohttp.ClientSession() as session:
        if model_type == "reranker":
            models = await collect_reranker_models(session, active)
            registry_key = "rerankers"
        else:
            models = await collect_llm_models(session, active)
            registry_key = "llms"

        log(f"Fetching disk sizes for {len(models)} unique models concurrently (max {CONCURRENCY} at a time)...")
        sem = asyncio.Semaphore(CONCURRENCY)
        sizes = dict(await asyncio.gather(*[fetch_model_size(session, m["id"], sem) for m in models]))

    registry: dict[str, Any] = {registry_key: []}
    for m in models:
        mid = m["id"]
        size = fmt_size_compact(sizes.get(mid, "N/A"))
        registry[registry_key].append({"name": mid, "size": size})
    print(json.dumps(registry, indent=4))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Scrape popular models from HuggingFace")
    p.add_argument("--top-by-downloads", type=int, default=0, metavar="N", help="Fetch top N models sorted by downloads")
    p.add_argument("--top-by-likes", type=int, default=0, metavar="N", help="Fetch top N models sorted by likes")
    p.add_argument("--top-by-trending", type=int, default=0, metavar="N", help="Fetch top N models sorted by trending score")
    p.add_argument("--raw", action="store_true", help="Output JSON only, suppress progress messages")
    p.add_argument("--type", choices=["llm", "reranker"], default="llm", dest="model_type", help="Model type to fetch (default: llm)")
    args = p.parse_args()
    asyncio.run(main(args.top_by_downloads, args.top_by_likes, args.top_by_trending, args.raw, args.model_type))
