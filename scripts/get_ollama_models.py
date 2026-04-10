#!/usr/bin/env python3

# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Scrapes Ollama library pages and produces static/ollama-min.json.

Usage:
    python3 scripts/get-ollama-models.py [--cache] [--clear-cache] [--output PATH] [--concurrency N]

    --cache         Re-use previously downloaded HTML files from ./ollama/
    --clear-cache   Delete ./ollama/ cache directory and exit
    --output        Output path (default: static/ollama-min.json)
    --concurrency   Max parallel requests (default: 20)
"""

import argparse
import asyncio
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Literal, NotRequired, TypedDict

import aiofiles
import aiohttp

# ruff: noqa: T201


CACHE_DIR = Path("./ollama")
BASE_URL = "https://ollama.com"
HEADERS = {"User-Agent": "Mozilla/5.0"}

Category = Literal["embeddings", "llms", "txt2img"]


class ModelInfo(TypedDict):
    name: str
    capabilities: list[str]


class ModelEntry(TypedDict):
    name: str
    size: str
    hash: str
    context: int | None
    modelfile: NotRequired[str]


class MinJson(TypedDict):
    embeddings: list[ModelEntry]
    llms: list[ModelEntry]
    txt2img: list[ModelEntry]


class ExtraSpecExtra(TypedDict):
    llms: list[str]
    embeddings: list[str]
    txt2img: list[str]


class ExtraSpec(TypedDict):
    extra: ExtraSpecExtra
    pickTags: dict[str, list[str]]


async def fetch(session: aiohttp.ClientSession, url: str) -> str:
    """Fetch URL and return response text."""
    async with session.get(url, headers=HEADERS) as resp:
        resp.raise_for_status()
        return await resp.text()


async def fetch_cached(
    session: aiohttp.ClientSession,
    url: str,
    cache_path: Path,
    use_cache: bool,
) -> str:
    """Fetch URL, reading from cache_path if use_cache is set and the file exists."""
    if use_cache and cache_path.exists():
        async with aiofiles.open(cache_path, encoding="utf-8") as f:
            return await f.read()
    html = await fetch(session, url)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    async with aiofiles.open(cache_path, "w", encoding="utf-8") as f:
        await f.write(html)
    return html


async def fetch_with_fallback(
    session: aiohttp.ClientSession,
    url_candidates: list[str],
    cache_path: Path,
    use_cache: bool,
) -> str:
    """Try URL candidates in order; on 404 advance to the next candidate. Caches the first success."""
    if use_cache and cache_path.exists():
        async with aiofiles.open(cache_path, encoding="utf-8") as f:
            return await f.read()
    last_exc: Exception | None = None
    for i, url in enumerate(url_candidates):
        try:
            return await fetch_cached(session, url, cache_path, False)
        except aiohttp.ClientResponseError as exc:
            last_exc = exc
            if exc.status == 404 and i < len(url_candidates) - 1:
                continue
            raise
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def find_all_between(text: str, attr: str, open_tag: str, close_tag: str) -> list[str]:
    """Find every VALUE in <... attr ...>VALUE</close_tag>."""
    results = []
    pos = 0
    while True:
        idx = text.find(attr, pos)
        if idx == -1:
            break
        open_idx = text.find(open_tag, idx)
        if open_idx == -1:
            break
        open_idx += len(open_tag)
        close_idx = text.find(close_tag, open_idx)
        if close_idx == -1:
            break
        results.append(text[open_idx:close_idx].strip())
        pos = close_idx + len(close_tag)
    return results


# ---------------------------------------------------------------------------
# Library page: returns list of {name, capabilities}
# ---------------------------------------------------------------------------


def parse_library(html: str) -> list[ModelInfo]:
    """Parse the Ollama library page and return a list of models with capabilities."""
    models: list[ModelInfo] = []
    for li_match in re.finditer(r"<li[^>]*x-test-model[^>]*>(.*?)</li>", html, re.DOTALL):
        li = li_match.group(1)
        name_m = re.search(r'href="/(?:library/)?([^"]+)"', li)
        if not name_m:
            continue
        capabilities = find_all_between(li, "x-test-capability", ">", "<")
        models.append({"name": name_m.group(1), "capabilities": capabilities})
    return models


# ---------------------------------------------------------------------------
# Tags page: returns list of ModelEntry
# ---------------------------------------------------------------------------


def parse_context(raw: str | None) -> int | None:
    """Convert context strings like '128K', '2048' to int."""
    if not raw:
        return None
    raw = raw.strip()
    if not raw:
        return None
    multiplier = 1
    if raw.endswith("K"):
        multiplier = 1024
        raw = raw[:-1]
    elif raw.endswith("M"):
        multiplier = 1024 * 1024
        raw = raw[:-1]
    try:
        return int(float(raw) * multiplier)
    except ValueError:
        return None


def skip_suffixes(name: str) -> bool:
    """Skip models with specifed suffixes."""
    skip_parts = (
        ":cloud",
        "-cloud",
        # "-alpha",
        # ":alpha",
        # ":beta",
        # "-beta",
        # ":e2b",
        # "e2b-",
        # ":e4b",
        # "e4b-",
        # ":a4b",
        # "a4b-",
        # "q2_",
        # "q3_",
        # "q4_",
        # "q5_",
        # "q6_",
        # "q8_",
        # "bf16",
        # "fp16",
        # "-text",
        # "-base",
        # "-uncensored",
        # "-it",
        # "-v",
        # ":v",
        # "-instruct"
    )
    skip_regex = re.compile("|".join(re.escape(p) for p in skip_parts))
    return bool(skip_regex.search(name))


def deduplicate_by_hash(entries: list[ModelEntry]) -> list[ModelEntry]:
    """For entries sharing the same non-empty hash, keep only the shortest-named one.

    Entries with and without ':' are deduplicated independently and always coexist.
    """
    no_hash: list[ModelEntry] = []
    by_hash_with_colon: dict[str, ModelEntry] = {}
    by_hash_no_colon: dict[str, ModelEntry] = {}
    for entry in entries:
        h = entry["hash"]
        name = entry["name"]
        if not h:
            no_hash.append(entry)
        elif ":" not in name:
            if h not in by_hash_no_colon or len(name) < len(by_hash_no_colon[h]["name"]):
                by_hash_no_colon[h] = entry
        elif h not in by_hash_with_colon or len(name) < len(by_hash_with_colon[h]["name"]):
            by_hash_with_colon[h] = entry
    return no_hash + list(by_hash_no_colon.values()) + list(by_hash_with_colon.values())


def parse_model_page(html: str, only_tags: list[str] | None) -> list[ModelEntry]:
    """Parse the model's main page and return entries (hashes not yet populated)."""
    prefix_re = re.search(r"<(.*?)x-test-model-namespace(.*?)>(.*?)<", html)
    prefix = prefix_re.group(3) if prefix_re else None
    parts = html.split("group px-4 py-3")[1:]
    entries: list[ModelEntry] = []
    for part in parts:
        tag_m = re.search(r"<(.*?)block(.*?)>(.*?)<", part)
        if not tag_m:
            continue
        col_spans = re.findall(r"<p[^>]*col-span-2[^>]*>(.*?)</p>", part, re.DOTALL)
        name = tag_m.group(3).removesuffix(":latest")
        if skip_suffixes(name):
            continue
        full_name = f"{prefix + '/' if prefix else ''}{name}"
        if only_tags and full_name not in only_tags:
            continue
        entries.append(
            {
                "name": full_name,
                "size": col_spans[0].strip() if col_spans else "",
                "hash": "",
                "context": parse_context(col_spans[1].strip() if len(col_spans) > 1 else None),
            }
        )
    return entries


def parse_tags_hashes(html: str) -> dict[str, str]:
    """Parse a model tags page and return a name → hash mapping."""
    hash_map: dict[str, str] = {}
    parts = html.split("group px-4 py-3")[1:]
    for part in parts:
        tag_m = re.search(r'href="/(?:library/)?([^"]+)"', part)
        hash_m = re.search(r"<span[^>]*font-mono[^>]*>(.*?)</span>", part, re.DOTALL)
        if not tag_m or not hash_m:
            continue
        name = tag_m.group(1).removesuffix(":latest")
        hash_map[name] = hash_m.group(1).strip()
    return hash_map


# ---------------------------------------------------------------------------
# Capability -> category mapping
# ---------------------------------------------------------------------------


def capability_to_category(capabilities: list[str]) -> Category:
    """Map a model's capability list to an output category key."""
    caps_lower = [c.lower() for c in capabilities]
    if any("embed" in c for c in caps_lower):
        return "embeddings"
    if any(c in ("image generation", "text-to-image", "txt2img", "image") for c in caps_lower):
        return "txt2img"
    return "llms"


def guess_category(name: str) -> Category:
    """Guess category from model name when library page capabilities are unavailable."""
    lower = name.lower()
    if any(k in lower for k in ("embed", "minilm", "e5-")):
        return "embeddings"
    if any(k in lower for k in ("flux", "diffus", "txt2img", "text-to-image")):
        return "txt2img"
    return "llms"


def parse_tags_full(html: str) -> list[ModelEntry]:
    """Parse a model tags page and return all entries with full info (no filtering)."""
    parts = html.split("group px-4 py-3")[1:]
    entries: list[ModelEntry] = []
    for part in parts:
        tag_m = re.search(r'href="/(?:library/)?([^"]+)"', part)
        if not tag_m:
            continue
        col_spans = re.findall(r"<p[^>]*col-span-2[^>]*>(.*?)</p>", part, re.DOTALL)
        hash_m = re.search(r"<span[^>]*font-mono[^>]*>(.*?)</span>", part, re.DOTALL)
        name = tag_m.group(1).removesuffix(":latest")
        entries.append(
            {
                "name": name,
                "size": col_spans[0].strip() if col_spans else "",
                "hash": hash_m.group(1).strip() if hash_m else "",
                "context": parse_context(col_spans[1].strip() if len(col_spans) > 1 else None),
            }
        )
    return entries


# ---------------------------------------------------------------------------
# Async fetch helpers
# ---------------------------------------------------------------------------


async def fetch_model_tags(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    model: ModelInfo,
    use_cache: bool,
    pick_tags: dict[str, list[str]],
) -> tuple[Category, str, list[ModelEntry]]:
    """Fetch model page and tags page, merge hashes, return entries."""
    name = model["name"]
    category = capability_to_category(model["capabilities"])
    model_cache = CACHE_DIR / "models" / f"{name.replace('/', '_')}.html"
    tags_cache = CACHE_DIR / "models" / f"{name.replace('/', '_')}-tags.html"
    async with semaphore:
        try:
            model_html = await fetch_cached(session, f"{BASE_URL}/library/{name}", model_cache, use_cache)
            tags_html = await fetch_cached(session, f"{BASE_URL}/library/{name}/tags", tags_cache, use_cache)
        except Exception as exc:
            print(f"  {name} - failed: {exc}", file=sys.stderr)
            return category, name, []
    only_tags = pick_tags.get(name)
    entries = parse_model_page(model_html, only_tags)
    hash_map = parse_tags_hashes(tags_html)
    for entry in entries:
        entry["hash"] = hash_map.get(entry["name"], "")
    return category, name, deduplicate_by_hash(entries)


async def fetch_extra_model(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    spec: str,
    category: Category,
    use_cache: bool,
    pick_tags: dict[str, list[str]],
) -> tuple[Category, str, list[ModelEntry]]:
    """Fetch an extra model by spec ('name' or 'namespace/name' or 'name:tag').

    Tries BASE_URL/library/{model_path} first; falls back to BASE_URL/{model_path} on 404.
    """
    if ":" in spec:
        model_path, specific_tag = spec.rsplit(":", 1)
    else:
        model_path, specific_tag = spec, None

    tags_candidates = [f"{BASE_URL}/library/{model_path}/tags", f"{BASE_URL}/{model_path}/tags"]
    tags_cache = CACHE_DIR / "models" / f"{model_path.replace('/', '_')}-tags.html"

    async with semaphore:
        try:
            tags_html = await fetch_with_fallback(session, tags_candidates, tags_cache, use_cache)
        except Exception as exc:
            print(f"  {spec} (extra) - failed: {exc}", file=sys.stderr)
            return category, model_path, []

        if specific_tag:
            full_name = f"{model_path}:{specific_tag}"
            matched = [e for e in parse_tags_full(tags_html) if e["name"] == full_name]
            return category, model_path, matched

        model_candidates = [f"{BASE_URL}/library/{model_path}", f"{BASE_URL}/{model_path}"]
        model_cache = CACHE_DIR / "models" / f"{model_path.replace('/', '_')}.html"
        try:
            model_html = await fetch_with_fallback(session, model_candidates, model_cache, use_cache)
        except Exception as exc:
            print(f"  {spec} (extra) model page - failed: {exc}", file=sys.stderr)
            return category, model_path, []

    only_tags = pick_tags.get(model_path)
    entries = parse_model_page(model_html, only_tags)
    hash_map = parse_tags_hashes(tags_html)
    for entry in entries:
        entry["hash"] = hash_map.get(entry["name"], "")
    return category, model_path, deduplicate_by_hash(entries)


async def build_min_json(use_cache: bool, concurrency: int, extra_specs: ExtraSpec) -> MinJson:
    """Scrape Ollama library and return categorized model data."""
    async with aiohttp.ClientSession() as session:
        library_html = await fetch_cached(session, f"{BASE_URL}/library", CACHE_DIR / "library.html", use_cache)
        models = parse_library(library_html)
        print(f"Found {len(models)} models in library", file=sys.stderr)

        semaphore = asyncio.Semaphore(concurrency)
        tasks = (
            [asyncio.ensure_future(fetch_model_tags(session, semaphore, model, use_cache, extra_specs["pickTags"])) for model in models]
            + [
                asyncio.ensure_future(fetch_extra_model(session, semaphore, spec, "embeddings", use_cache, extra_specs["pickTags"]))
                for spec in extra_specs["extra"]["embeddings"]
            ]
            + [
                asyncio.ensure_future(fetch_extra_model(session, semaphore, spec, "llms", use_cache, extra_specs["pickTags"]))
                for spec in extra_specs["extra"]["llms"]
            ]
            + [
                asyncio.ensure_future(fetch_extra_model(session, semaphore, spec, "txt2img", use_cache, extra_specs["pickTags"]))
                for spec in extra_specs["extra"]["txt2img"]
            ]
        )

        output: MinJson = {"embeddings": [], "llms": [], "txt2img": []}
        for i, done in enumerate(asyncio.as_completed(tasks), 1):
            category, name, entries = await done
            print(f"[{i}/{len(tasks)}] {name} ({category}) - {len(entries)} tags", file=sys.stderr)
            output[category].extend(entries)

    return output


async def main() -> None:
    """Parse arguments, run the scraper, and write output JSON."""
    parser = argparse.ArgumentParser(description="Generate ollama-min.json from Ollama library pages.")
    parser.add_argument("--cache", action="store_true", help="Use cached HTML files")
    parser.add_argument("--clear-cache", action="store_true", help="Delete cache directory and exit")
    parser.add_argument("--output", default="static/ollama-min.json", help="Output file path")
    parser.add_argument("--concurrency", type=int, default=20, help="Max parallel requests")
    parser.add_argument(
        "--extra",
        default=Path(__file__).resolve().parent / "extra_models.json",
        metavar="FILE",
        help="File with extra model specs to include (one per line, '#' for comments)",
    )
    parser.add_argument(
        "--merge",
        default=Path(__file__).resolve().parent / "custom_models.json",
        metavar="FILE",
        help="JSON file with additional model entries to merge into the output",
    )
    args = parser.parse_args()

    if args.clear_cache:
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)
            print(f"Removed {CACHE_DIR}", file=sys.stderr)
        else:
            print(f"Cache directory {CACHE_DIR} does not exist", file=sys.stderr)
        return

    extra_specs: ExtraSpec = {"extra": {"embeddings": [], "llms": [], "txt2img": []}, "pickTags": {}}
    if args.extra and Path(args.extra).exists():
        async with aiofiles.open(args.extra, encoding="utf-8") as f:
            content = await f.read()
        extra_specs = json.loads(content)
        print(
            f"Extra models: {len(extra_specs['extra']['embeddings'])}, {len(extra_specs['extra']['llms'])}, "
            f"{len(extra_specs['extra']['txt2img'])} specs from {args.extra}",
            file=sys.stderr,
        )

    data: MinJson = await build_min_json(use_cache=args.cache, concurrency=args.concurrency, extra_specs=extra_specs)

    if args.merge and Path(args.merge).exists():
        merge_path = Path(args.merge)
        merge_data: dict[str, list[ModelEntry]] = json.loads(merge_path.read_text(encoding="utf-8"))
        for category in ("embeddings", "llms", "txt2img"):
            data[category].extend(merge_data.get(category, []))  # type: ignore[literal-required]
        total_merged = sum(len(v) for v in merge_data.values())
        print(f"Merged {total_merged} entries from {args.merge}", file=sys.stderr)

    for entries in (data["embeddings"], data["llms"], data["txt2img"]):
        entries.sort(key=lambda e: e["name"].casefold())

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    async with aiofiles.open(output_path, "w", encoding="utf-8") as f:
        await f.write(json.dumps(data, indent=4))

    total = len(data["embeddings"]) + len(data["llms"]) + len(data["txt2img"])
    print(f"Written {total} tags to {output_path}", file=sys.stderr)
    print(f"  embeddings: {len(data['embeddings'])}", file=sys.stderr)
    print(f"  llms:       {len(data['llms'])}", file=sys.stderr)
    print(f"  txt2img:    {len(data['txt2img'])}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
