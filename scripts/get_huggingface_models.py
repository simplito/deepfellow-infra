# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Get HuggingFace Models info.

Uses the HuggingFace Hub API (no API key required for public data).
Outputs a vllm-min.json compatible JSON to stdout.
"""

import argparse
import asyncio
import json
import logging
import re
import sys
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any
from urllib.parse import quote

import aiohttp

from scripts.utils.hf_common import (
    DEFAULT_MODEL_DETAIL_CACHE_DIR,
    HF_API,
    HfErrorKind,
    classify_hf_error,
    get_json_with_retry,
    get_model_detail_cached,
    is_embedding,
    is_gguf,
)

# ruff: noqa: T201

# This module runs both as an offline CLI script and, since `validate_model_compatibility` was
# added, imported into the running server - use the server's logger name so warnings raised from
# that path are picked up by its configured handlers instead of falling through to
# `logging.lastResort` unformatted. Standalone CLI runs are unaffected: neither name has a handler
# configured outside the server, so both fall back to `lastResort` there too.
logger = logging.getLogger("uvicorn.error")

LLM_TAGS = ["text-generation", "text2text-generation", "image-text-to-text"]
CONCURRENCY = 10  # parallel size-fetch requests
PAGE_SIZE = 100  # max models per API page (HF limit)
NON_LLM_TASK_WORDS = re.compile(r"\b(?:translation|summarization|classification|ner|qa)\b")


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
    """Filter out non-LLM models using a basic heuristic.

    Task words are matched on word boundaries so that a substring buried in an unrelated name does
    not disqualify a real LLM — "ner" occurs inside "ForConditionalGeneration", which every
    multimodal repo carries. "ocr" is matched as a plain substring instead, because OCR models spell
    it glued to the rest of the name ("PaddleOCR-VL", "olmOCR", "GOT-OCR2_0"); those are
    single-purpose text extractors rather than chat models, so they don't belong in an LLM registry.
    """
    name = model.get("id", "").lower()
    return not NON_LLM_TASK_WORDS.search(name) and "ocr" not in name


CI_FIXTURE_ORG_PREFIXES = ("trl-internal-testing/", "peft-internal-testing/")
MLX_RE = re.compile(r"\bmlx\b", re.IGNORECASE)


def is_test_fixture(model: dict[str, Any]) -> bool:
    """Detect CI fixtures and test repos (tiny random-weight checkpoints).

    These dominate the downloads ranking because HuggingFace's counter measures CI traffic, but their
    weights are random so they are useless to serve. Matched by org prefix and the "tiny-random" name.
    """
    name = model.get("id", "").lower()
    return name.startswith(CI_FIXTURE_ORG_PREFIXES) or "tiny-random" in name


def is_mlx(model: dict[str, Any]) -> bool:
    """Detect MLX builds (Apple-silicon only), which neither vLLM nor SGLang can load.

    "mlx" is matched as a token so it is found in "Qwen3.5-9B-MLX-4bit" and "Bonsai-27B-mlx-1bit"
    but not buried inside an unrelated word.
    """
    name = model.get("id", "").lower()
    return name.startswith("mlx-community/") or bool(MLX_RE.search(name.replace("_", "-")))


def is_reranker(model: dict[str, Any]) -> bool:
    """Filter reranker models by name heuristic."""
    name = model.get("id", "").lower()
    return "rerank" in name


NON_RERANKER_CROSS_ENCODER_TASK_WORDS = re.compile(r"\b(?:sts|stsb|nli|qnli|mnli|quora|qqp|mrpc|paraphrase)\b", re.IGNORECASE)


def is_reranking_cross_encoder(model: dict[str, Any]) -> bool:
    """Exclude cross-encoders trained for a different scoring task than reranking.

    `library_name == "sentence-transformers"` plus a text-ranking/text-classification pipeline_tag
    only means "some CrossEncoder regression/classification head" — the same architecture is reused
    for semantic-textual-similarity (STS/STSB), natural-language-inference (NLI/QNLI/MNLI), and
    duplicate-question (Quora/QQP/MRPC) benchmarks, which score something other than relevance and
    would return meaningless rankings if served as a reranker. These tasks are conventionally named
    in the repo id (e.g. cross-encoder/stsb-*, cross-encoder/qnli-*, cross-encoder/quora-*), so
    they're excluded by name.
    """
    name = model.get("id", "").lower()
    return not NON_RERANKER_CROSS_ENCODER_TASK_WORDS.search(name)


# Each pipeline_tag maps to a validator that decides whether a candidate carrying that tag is kept.
# "sentence-similarity" is HF's dedicated tag for that task and is trusted as-is. "text-ranking" and
# "text-classification" both get HF-tagged onto any CrossEncoder-style model regardless of what it
# was actually trained to score — reranking relevance, but also semantic-textual-similarity (STS),
# natural-language-inference (NLI), or duplicate-question detection, which return meaningless
# rankings if served as a reranker — so every candidate from either tag is filtered through
# `is_reranking_cross_encoder` to exclude those by name. "text-classification" is additionally far
# broader still (sentiment classifiers, prompt-injection detectors, ...), so it also requires
# corroborating evidence: rerankers published under text-classification are consistently shipped
# with the sentence-transformers library (a survey of the top 100 by downloads found this held for
# all of them and none of the non-reranker classifiers). "feature-extraction" is embeddings' analog
# of the broad, noisy tag and is validated against the existing embedding name heuristic instead.
RERANKER_TAGS: dict[str, Callable[[dict[str, Any]], bool]] = {
    "text-ranking": is_reranking_cross_encoder,
    "text-classification": lambda m: m.get("library_name") == "sentence-transformers" and is_reranking_cross_encoder(m),
}
EMBEDDING_TAGS: dict[str, Callable[[dict[str, Any]], bool]] = {
    "sentence-similarity": lambda _m: True,
    "feature-extraction": lambda m: is_embedding(m),
}


# vLLM's /v1/audio/transcriptions only serves encoder-decoder / audio-LLM speech models it ships a
# transcription implementation for. HF's "automatic-speech-recognition" tag is far broader (CTC
# models like wav2vec2/hubert, ctranslate2/onnx re-exports, ...), so STT candidates are kept only
# when their `config.architectures` is on this best-effort allowlist.
STT_TAGS = ["automatic-speech-recognition"]
# HF model-family tags queried on top of STT_TAGS so the candidate limit isn't spent on wav2vec2 & co.
STT_FAMILY_FILTERS = ["whisper", "voxtral", "voxtral_realtime", "qwen3_asr", "granite_speech"]
SUPPORTED_STT_ARCHITECTURES = {
    "WhisperForConditionalGeneration",
    "VoxtralForConditionalGeneration",
    "VoxtralRealtimeForConditionalGeneration",
    "GraniteSpeechForConditionalGeneration",
    "Qwen3ASRForConditionalGeneration",
}


def is_non_servable_stt_repo(model: dict[str, Any]) -> bool:
    """Exclude repos that carry a supported architecture in `config.json` but that vLLM can't load.

    ONNX-only exports live in the onnx-community/* and Xenova/* namespaces (the `onnx` tag alone isn't
    enough: official repos often ship safetensors next to ONNX files), CoreML repos carry no servable
    weights, and forced aligners reuse the ASR architecture without producing transcriptions.
    """
    tags = {str(t).lower() for t in model.get("tags") or []}
    name = model.get("id", "").lower()
    return "coreml" in tags or name.startswith(("onnx-community/", "xenova/")) or "forcedaligner" in name


def is_supported_stt(architectures: list[str]) -> bool:
    """Return True if any architecture is one vLLM can serve for audio transcription."""
    return any(a in SUPPORTED_STT_ARCHITECTURES for a in architectures)


def is_supported_cross_encoder(architectures: list[str]) -> bool:
    """Return True if any architecture is a standard HF sequence-classification head.

    vLLM's rerank/score task only serves this kind of cross-encoder. Models named "reranker"
    that use a different architecture (e.g. a causal LM doing generative ranking, or a custom
    multi-vector ranking head) fail to load in vLLM even though they match the name heuristic.
    """
    return any(a.endswith("ForSequenceClassification") for a in architectures)


# Best-effort allowlist of known vision-language / audio-language model architecture class names
# (and name substrings) reported in a HF repo's `config.architectures`. Not exhaustive - new VLM/ALM
# architectures show up in HF constantly - just covers the common families well enough to flag
# vision/audio support in the catalog without any extra network calls.
_VISION_ARCHITECTURE_SUBSTRINGS = ("VL",)
_VISION_ARCHITECTURES = {
    "LlavaForConditionalGeneration",
    "LlavaNextForConditionalGeneration",
    "Qwen2VLForConditionalGeneration",
    "Qwen2_5_VLForConditionalGeneration",
    "MllamaForConditionalGeneration",
    "Idefics2ForConditionalGeneration",
    "PaliGemmaForConditionalGeneration",
    "InternVLChatModel",
    "MiniCPMV",
}
_AUDIO_ARCHITECTURES = {
    "Qwen2AudioForConditionalGeneration",
    "UltravoxModel",
    "Qwen2_5OmniModel",
    "Qwen2AudioModel",
}


def classify_modality_support(architectures: list[str]) -> tuple[bool | None, bool | None]:
    """Best-effort (supports_vision, supports_audio) from a HF repo's `config.architectures`.

    Matches against a short, known-incomplete allowlist of vision-language/audio-language model
    architecture class names. A match is a confident `True`; anything else - no match, or an
    empty/missing `architectures` list - is `None` (unknown), never `False`: this allowlist is
    known-incomplete, so a miss doesn't mean the model actually lacks the capability, only that
    it couldn't be confirmed from data already fetched for the catalog (no extra HTTP requests).
    """
    supports_vision = (
        True if any(a in _VISION_ARCHITECTURES or any(sub in a for sub in _VISION_ARCHITECTURE_SUBSTRINGS) for a in architectures) else None
    )
    supports_audio = True if any(a in _AUDIO_ARCHITECTURES for a in architectures) else None
    return supports_vision, supports_audio


async def has_chat_template(
    session: aiohttp.ClientSession,
    model_id: str,
    siblings: list[dict[str, Any]],
    revision: str | None = None,
    max_retries: int = 8,
    strict: bool = False,
) -> bool:
    """Return True if the model ships a chat template vLLM can use for chat completions.

    As of transformers v4.44, vLLM no longer falls back to a generic template, so serving a
    model whose tokenizer defines none raises a 400 on every chat request. Newer repos carry a
    standalone `chat_template.jinja` file; older ones embed a `chat_template` key in
    `tokenizer_config.json`. Base/pretrain models typically have neither.

    By default, a failed fetch of `tokenizer_config.json` (network issue, HF outage, exhausted
    rate-limit retries) is treated the same as "no chat template" - acceptable for the catalog
    crawler, which just drops one candidate among hundreds. Pass `strict=True` to instead re-raise
    `aiohttp.ClientError`/`TimeoutError` so a caller validating a single user-initiated add (where
    that same fetch failure must NOT read as "confirmed incompatible") can tell the two apart.
    """
    filenames = {f.get("rfilename", "") for f in siblings}
    if "chat_template.jinja" in filenames:
        return True
    if "tokenizer_config.json" not in filenames:
        return False
    ref = quote(revision, safe="") if revision else "main"
    try:
        tokenizer_config = await get_json_with_retry(
            session, f"https://huggingface.co/{model_id}/raw/{ref}/tokenizer_config.json", {}, max_retries=max_retries
        )
        return bool(tokenizer_config.get("chat_template"))
    except (aiohttp.ClientError, TimeoutError):
        if strict:
            raise
        return False


class ModelIncompatibleError(Exception):
    """Raised when a HuggingFace repo is structurally incompatible with the engine's serving mode."""


class HfRepoAccessError(ModelIncompatibleError):
    """Raised when the repo/revision doesn't exist on HuggingFace or can't be accessed with the configured token."""

    def __init__(self, hf_id: str, revision: str | None, kind: HfErrorKind, *, authenticated: bool) -> None:
        self.kind = kind
        super().__init__(_access_error_message(hf_id, revision, kind, authenticated=authenticated))


def _access_error_message(hf_id: str, revision: str | None, kind: HfErrorKind, *, authenticated: bool) -> str:
    match kind:
        case HfErrorKind.REVISION_NOT_FOUND:
            return f"Revision {revision!r} doesn't exist in {hf_id}; check the branch, tag or commit SHA."
        case HfErrorKind.NOT_FOUND:
            return f"{hf_id} doesn't exist on HuggingFace; check the repo id."
        case HfErrorKind.GATED if authenticated:
            return (
                f"{hf_id} is gated, and the configured HuggingFace token is invalid or its account has no access to it; "
                "check DF_HUGGING_FACE_TOKEN in Configuration and request access on its HuggingFace page."
            )
        case HfErrorKind.GATED:
            return f"{hf_id} is gated; request access on its HuggingFace page and set DF_HUGGING_FACE_TOKEN in Configuration."
        case _ if authenticated:
            return f"{hf_id} doesn't exist, or the configured HuggingFace token is invalid or has no access to it."
        case _:
            return f"{hf_id} doesn't exist or is private; check the repo id, or set DF_HUGGING_FACE_TOKEN in Configuration if it's private."


async def fetch_model_metadata(
    session: aiohttp.ClientSession, hf_id: str, revision: str | None, *, authenticated: bool, max_retries: int = 3
) -> dict[str, Any] | None:
    """Fetch `hf_id`'s metadata, raising `HfRepoAccessError` if it can't be downloaded with the session's credentials.

    The model downloader pulls from the same host with the same token, so a definite "not found" /
    "no access" here would fail the download too - unlike the compatibility heuristics, this check
    can't misfire and isn't skipped by `skip_validation`. A gated repo's metadata is public, so gating is detected from
    the `gated` field instead: without a token its files can't be downloaded, and with one a single
    file request checks that the token's account was granted access. Returns `None` on a
    transient failure (network issue, HF outage, exhausted rate-limit retries), so a temporary HF
    hiccup or an offline add with a manual size isn't blocked.
    """
    url = f"{HF_API}/models/{hf_id}" + (f"/revision/{quote(revision, safe='')}" if revision else "")
    try:
        data = await get_json_with_retry(session, url, {"blobs": "true"}, max_retries=max_retries)
    except (aiohttp.ClientError, TimeoutError) as exc:
        kind = classify_hf_error(exc)
        if kind is HfErrorKind.UNAVAILABLE:
            logger.warning("Couldn't reach HuggingFace to validate %s: %s", hf_id, exc)
            return None
        raise HfRepoAccessError(hf_id, revision, kind, authenticated=authenticated) from exc
    if data.get("gated"):
        if not authenticated:
            raise HfRepoAccessError(hf_id, revision, HfErrorKind.GATED, authenticated=False)
        await _check_gated_file_access(session, hf_id, revision, data.get("siblings", []))
    return data


def _gated_probe_file(siblings: list[dict[str, Any]]) -> str | None:
    """Pick a file of a gated repo that's served only to accounts with access.

    Not every file of a gated repo is gated - README and LICENSE files stay public - so the probe
    sticks to files the download needs anyway: `config.json`, then weights, then any JSON.
    """
    filenames = [f.get("rfilename", "") for f in siblings]
    if "config.json" in filenames:
        return "config.json"
    for suffix in (".safetensors", ".json"):
        found = next((name for name in filenames if name.endswith(suffix)), None)
        if found:
            return found
    return None


async def _check_gated_file_access(
    session: aiohttp.ClientSession, hf_id: str, revision: str | None, siblings: list[dict[str, Any]]
) -> None:
    """Raise `HfRepoAccessError` if the session's token can't download files of gated `hf_id`.

    A gated repo's metadata is public, so `fetch_model_metadata` can't tell from it whether the
    token's account was granted access - only a file request can. A HEAD without following
    redirects is enough: HF answers 401/403 `GatedRepo` before redirecting a file to its CDN. Any
    other failure (a missing file, an outage) stays lenient, like the rest of the validation.
    """
    probe_file = _gated_probe_file(siblings)
    if probe_file is None:
        return
    ref = quote(revision, safe="") if revision else "main"
    url = f"https://huggingface.co/{hf_id}/resolve/{ref}/{quote(probe_file)}"
    try:
        async with session.head(url, allow_redirects=False) as resp:
            resp.raise_for_status()
    except (aiohttp.ClientError, TimeoutError) as exc:
        kind = classify_hf_error(exc)
        if kind in (HfErrorKind.GATED, HfErrorKind.NO_ACCESS):
            raise HfRepoAccessError(hf_id, revision, kind, authenticated=True) from exc
        logger.warning("Couldn't verify access to gated %s: %s", hf_id, exc)


def _check_architecture(hf_id: str, data: dict[str, Any], model_type: str) -> None:
    """Raise `ModelIncompatibleError` if the repo's `config.architectures` isn't one vLLM serves for `model_type`."""
    architectures = data.get("config", {}).get("architectures") or []
    found = ", ".join(architectures) or "unknown"
    if model_type == "reranker" and not is_supported_cross_encoder(architectures):
        msg = (
            f"{hf_id}'s architecture ({found}) isn't a supported cross-encoder; "
            "the rerank/score task only serves ForSequenceClassification models."
        )
        raise ModelIncompatibleError(msg)
    if model_type == "stt" and not is_supported_stt(architectures):
        msg = f"{hf_id}'s architecture ({found}) isn't one vLLM serves for /v1/audio/transcriptions."
        raise ModelIncompatibleError(msg)


async def validate_model_compatibility(
    session: aiohttp.ClientSession,
    hf_id: str,
    revision: str | None,
    model_type: str,
    max_retries: int = 3,
    *,
    authenticated: bool = False,
) -> dict[str, Any] | None:
    """Raise `ModelIncompatibleError` with a clear reason if `hf_id` can't be served, given `model_type`.

    Reuses the same heuristics the catalog refresh already applies when building the curated model
    list (`is_gguf`, `has_chat_template`, `is_supported_cross_encoder`, `is_supported_stt`), so adding a custom model by
    hf_id fails fast with a clear message instead of only surfacing as an opaque container-start
    error. `max_retries` defaults low (unlike the catalog crawler's default of 8) because this runs
    on an interactive add-model request rather than an offline batch job.

    Returns the fetched HuggingFace metadata (siblings/tags/config) on success, so a caller that also
    needs it (e.g. to compute the model's size) can reuse it instead of issuing a second identical
    request. Returns `None` when either HTTP fetch failed transiently (network issue, HF outage,
    exhausted rate-limit retries) - that's not treated as incompatibility, so a temporary HF hiccup
    doesn't block a valid add; the chat-template check (`has_chat_template(..., strict=True)`) is
    included in that leniency for the same reason. A repo or revision that doesn't exist or can't be
    accessed is a permanent, user-fixable problem and raises `HfRepoAccessError` instead - see
    `fetch_model_metadata`; `authenticated` (whether the session sends a HuggingFace token) only
    picks the wording of that error and whether a gated repo is reachable at all.
    """
    data = await fetch_model_metadata(session, hf_id, revision, authenticated=authenticated, max_retries=max_retries)
    if data is None:
        return None

    if is_gguf(data):
        msg = f"{hf_id} is a GGUF (quantized) repo, which this engine can't serve directly."
        raise ModelIncompatibleError(msg)

    if model_type == "llm":
        siblings = data.get("siblings", [])
        try:
            template_ok = await has_chat_template(session, hf_id, siblings, revision=revision, max_retries=max_retries, strict=True)
        except (aiohttp.ClientError, TimeoutError) as exc:
            # A gated repo's metadata is public but its files aren't: a 401/403 here means the download
            # would fail too. A 404 only means this one file is missing, so it stays lenient.
            kind = classify_hf_error(exc)
            if kind in (HfErrorKind.GATED, HfErrorKind.NO_ACCESS):
                raise HfRepoAccessError(hf_id, revision, kind, authenticated=authenticated) from exc
            logger.warning("Couldn't verify chat template for %s: %s", hf_id, exc)
            return None
        if not template_ok:
            msg = f"{hf_id} doesn't ship a chat template, so it can't be served for chat completions."
            raise ModelIncompatibleError(msg)
    elif model_type in ("reranker", "stt"):
        _check_architecture(hf_id, data, model_type)

    return data


async def fetch_popular_models(
    session: aiohttp.ClientSession, tag: str, sort: str, limit: int, extra_filter: str | None = None
) -> list[dict[str, Any]]:
    """Fetch popular models with the given pipeline_tag (and optional extra tag filter) sorted by the given criterion."""
    params = {
        "pipeline_tag": tag,
        "sort": sort,
        "direction": "-1",
        "limit": str(limit),
        "cardData": "true",
    }
    if extra_filter:
        params["filter"] = extra_filter
    return await get_json_with_retry(session, f"{HF_API}/models", params)


async def fetch_popular_reranker_models(session: aiohttp.ClientSession, sort: str, limit: int) -> list[dict[str, Any]]:
    """Fetch popular reranker models from HuggingFace API sorted by the given criterion."""
    params = {
        "search": "rerank",
        "sort": sort,
        "direction": "-1",
        "limit": str(limit),
        "cardData": "true",
    }
    return await get_json_with_retry(session, f"{HF_API}/models", params)


async def fetch_popular_embedding_models(session: aiohttp.ClientSession, sort: str, limit: int) -> list[dict[str, Any]]:
    """Fetch popular embedding models from HuggingFace API sorted by the given criterion."""
    params = {
        "search": "embed",
        "sort": sort,
        "direction": "-1",
        "limit": str(limit),
        "cardData": "true",
    }
    return await get_json_with_retry(session, f"{HF_API}/models", params)


async def _is_gguf_resolved(session: aiohttp.ClientSession, model: dict[str, Any], sem: asyncio.Semaphore) -> bool:
    """Return True if the search-result `model` is a GGUF repo, checking the repo's file list when only the tag flags it.

    Search results carry no file list, so `is_gguf` alone would reject multi-format repos that ship native weights next
    to a GGUF variant (e.g. mixedbread-ai/mxbai-embed-large-v1). Repos with "gguf" in their name are almost always
    GGUF-only, so they are dropped without a request; only tag-only candidates are fetched, and the detail fetch is
    disk-cached. If it fails, the model stays classified as GGUF (the safe, previous behaviour).
    """
    if not is_gguf(model):
        return False
    if "gguf" in model.get("id", "").lower():
        return True
    async with sem:
        try:
            data = await get_model_detail_cached(session, model["id"], cache_dir=DEFAULT_MODEL_DETAIL_CACHE_DIR)
        except Exception:
            logger.warning("Failed to fetch details for %s to verify GGUF status; treating it as GGUF", model["id"], exc_info=True)
            return True
    return is_gguf(data)


async def _drop_gguf_models(session: aiohttp.ClientSession, models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop GGUF-only repos from the deduplicated candidates, keeping their order."""
    sem = asyncio.Semaphore(CONCURRENCY)
    flags = await asyncio.gather(*[_is_gguf_resolved(session, m, sem) for m in models])
    return [m for m, gguf in zip(models, flags, strict=True) if not gguf]


async def collect_llm_models(session: aiohttp.ClientSession, active: dict[str, int]) -> list[dict[str, Any]]:
    """Fetch and deduplicate LLM models across all sort/tag combinations."""
    fetch_tasks = [fetch_popular_models(session, tag, sort, limit) for sort, limit in active.items() for tag in LLM_TAGS]
    results = await asyncio.gather(*fetch_tasks)
    seen: set[str] = set()
    models: list[dict[str, Any]] = []
    for batch in results:
        for m in batch:
            mid = m["id"]
            if mid not in seen and is_llm(m) and not is_test_fixture(m) and not is_mlx(m):
                seen.add(mid)
                models.append(m)
    return await _drop_gguf_models(session, models)


async def collect_stt_models(
    session: aiohttp.ClientSession, active: dict[str, int], log: Callable[[str], None] = lambda _msg: None
) -> list[dict[str, Any]]:
    """Fetch speech-to-text models: per sort key, the global top N across all vLLM-servable families.

    Querying only the broad `automatic-speech-recognition` tag fills the top N with wav2vec2/CTC
    models (and repos without a `config.json`) vLLM can't serve, so each family in `STT_FAMILY_FILTERS`
    is queried on its own and the merged results are ranked by the sort key before cutting to N.
    """
    fetch_tasks = [
        fetch_popular_models(session, tag, sort, limit, family)
        for sort, limit in active.items()
        for tag in STT_TAGS
        for family in STT_FAMILY_FILTERS
    ]
    results = iter(await asyncio.gather(*fetch_tasks))
    fetched_by_sort = {sort: {m["id"]: m for _ in STT_TAGS for _ in STT_FAMILY_FILTERS for m in next(results)} for sort in active}
    servable = {
        mid: m
        for fetched in fetched_by_sort.values()
        for mid, m in fetched.items()
        if not is_test_fixture(m) and not is_mlx(m) and not is_non_servable_stt_repo(m)
    }
    non_gguf = {m["id"] for m in await _drop_gguf_models(session, list(servable.values()))}

    seen: set[str] = set()
    prefiltered: set[str] = set()
    models: list[dict[str, Any]] = []
    for sort, limit in active.items():
        fetched = fetched_by_sort[sort].values()
        candidates = [m for m in fetched if m["id"] in non_gguf]
        prefiltered.update(m["id"] for m in fetched if m["id"] not in non_gguf)
        candidates.sort(key=lambda m: m.get(sort) or 0, reverse=True)
        for m in candidates[:limit]:
            if m["id"] not in seen:
                seen.add(m["id"])
                models.append(m)
    if prefiltered:
        log(f"Dropped {len(prefiltered)} STT candidate(s) as gguf/mlx/ONNX/CoreML/aligner/test repos before ranking.")
    return models


async def _collect_by_tag_and_name(
    session: aiohttp.ClientSession,
    active: dict[str, int],
    tags: dict[str, Callable[[dict[str, Any]], bool]],
    fetch_by_name: Callable[[aiohttp.ClientSession, str, int], Coroutine[Any, Any, list[dict[str, Any]]]],
    is_named_match: Callable[[dict[str, Any]], bool],
) -> list[dict[str, Any]]:
    """Merge tag-driven and name-driven candidate discovery, deduplicated across all sources.

    A candidate qualifies if it satisfies *any* source: its pipeline_tag's own validator (see
    `RERANKER_TAGS`/`EMBEDDING_TAGS`), or the legacy name-search heuristic. This is a strict OR —
    the tag query surfaces models the name search can never find (its repo name doesn't contain
    the word being searched for), while the name search still catches candidates that carry
    neither pipeline_tag (e.g. generative rerankers tagged as plain text-generation models).
    """
    query_tags = [tag for _ in active for tag in tags]
    tag_fetch_tasks = [fetch_popular_models(session, tag, sort, limit) for sort, limit in active.items() for tag in tags]
    name_fetch_tasks = [fetch_by_name(session, sort, limit) for sort, limit in active.items()]
    tag_results = await asyncio.gather(*tag_fetch_tasks)
    name_results = await asyncio.gather(*name_fetch_tasks)

    seen: set[str] = set()
    models: list[dict[str, Any]] = []

    def add(m: dict[str, Any]) -> None:
        mid = m["id"]
        if mid not in seen:
            seen.add(mid)
            models.append(m)

    for tag, batch in zip(query_tags, tag_results, strict=True):
        for m in batch:
            if tags[tag](m):
                add(m)
    for batch in name_results:
        for m in batch:
            if is_named_match(m):
                add(m)
    return await _drop_gguf_models(session, models)


async def collect_reranker_models(session: aiohttp.ClientSession, active: dict[str, int]) -> list[dict[str, Any]]:
    """Fetch and deduplicate reranker models across all sort combinations, by tag and by name."""
    return await _collect_by_tag_and_name(session, active, RERANKER_TAGS, fetch_popular_reranker_models, is_reranker)


async def collect_embedding_models(session: aiohttp.ClientSession, active: dict[str, int]) -> list[dict[str, Any]]:
    """Fetch and deduplicate embedding models across all sort combinations, by tag and by name."""
    return await _collect_by_tag_and_name(session, active, EMBEDDING_TAGS, fetch_popular_embedding_models, is_embedding)


async def fetch_model_details(
    session: aiohttp.ClientSession,
    model_id: str,
    sem: asyncio.Semaphore,
    check_chat_template: bool = False,
    cache_dir: Path | None = None,
) -> tuple[str, str, list[str] | None, bool]:
    """Return (model_id, human_readable_size, architectures, has_chat_template).

    `has_chat_template` is only checked when `check_chat_template` is set (LLM candidates); it is
    True by default for reranker/embedding candidates, for which the check is meaningless.

    `architectures` is `None` when the detail fetch itself failed (e.g. exhausted retries under
    rate limiting), as opposed to `[]` when it succeeded but the model genuinely reports none —
    `_drop_unsupported_models` needs to tell these apart so a fetch failure doesn't get treated as
    "confirmed not a cross-encoder" and silently kept with fabricated data.

    `cache_dir` is forwarded to `get_model_detail_cached` - see its docstring for why caching is
    opt-in rather than on by default.
    """
    async with sem:
        try:
            data = await get_model_detail_cached(session, model_id, cache_dir=cache_dir)
            siblings = data.get("siblings", [])
            total = sum(f.get("size", 0) for f in siblings if f.get("size"))
            architectures = data.get("config", {}).get("architectures") or []
            chat_ok = await has_chat_template(session, model_id, siblings) if check_chat_template else True
            return model_id, fmt_size(total) if total else "N/A", architectures, chat_ok
        except Exception:
            logger.warning("Failed to fetch details for %s; it will be dropped from the catalog", model_id, exc_info=True)
            return model_id, "N/A", None, False


def _drop_unsupported_models(
    registry_key: str,
    models: list[dict[str, Any]],
    architectures: dict[str, list[str] | None],
    chat_capable: dict[str, bool],
    log: Callable[[str], None],
    keep_generative_rerankers: bool = False,
) -> list[dict[str, Any]]:
    """Drop candidates the target engine can't actually serve, logging what was dropped.

    `keep_generative_rerankers` opts a registry out of the cross-encoder-only filter: engines
    that can serve generative rerankers (e.g. SGLang, given the right startup flags) pass this so
    those candidates stay in and get tagged via `is_generative` instead of being dropped outright.
    """
    if registry_key == "rerankers":
        before = len(models)
        models = [m for m in models if architectures.get(m["id"]) is not None]
        if dropped := before - len(models):
            log(f"Dropped {dropped} reranker candidate(s) whose details couldn't be fetched from HuggingFace.")
        if not keep_generative_rerankers:
            before = len(models)
            models = [m for m in models if is_supported_cross_encoder(architectures.get(m["id"]) or [])]
            if dropped := before - len(models):
                log(f"Dropped {dropped} reranker candidate(s) whose architecture vLLM can't serve as a cross-encoder.")
    elif registry_key == "stts":
        before = len(models)
        models = [m for m in models if is_supported_stt(architectures.get(m["id"]) or [])]
        if dropped := before - len(models):
            log(f"Dropped {dropped} STT candidate(s) whose architecture vLLM can't serve for transcription.")
    elif registry_key == "llms":
        before = len(models)
        models = [m for m in models if chat_capable.get(m["id"], False)]
        if dropped := before - len(models):
            log(f"Dropped {dropped} LLM candidate(s) without a usable chat template (base/pretrain models vLLM can't serve for chat).")
    return models


async def fetch_registry_entries(
    session: aiohttp.ClientSession,
    top_by_downloads: int,
    top_by_likes: int,
    top_by_trending: int,
    model_type: str,
    log: Callable[[str], None],
    allow_generative_rerankers: bool = False,
) -> tuple[str, list[dict[str, Any]]]:
    """Fetch one registry_key's worth of entries (llms/rerankers/embeddings/stts) from HuggingFace.

    Shared by the CLI's `main()` and the server's in-process runtime refresh, so both go through
    the exact same fetch/filter/format pipeline. Returns (registry_key, entries); raises on a hard
    fetch failure (network/parse error propagates from `collect_*_models`/`fetch_model_details`).
    """
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
        raise ValueError("Specify at least one of: top_by_downloads, top_by_likes, top_by_trending")

    label = {"downloads": "downloads", "likes": "likes", "trendingScore": "trending"}
    summary = ", ".join(f"top {n} by {label[s]}" for s, n in active.items())
    log(f"Fetching {model_type} models from HuggingFace ({summary})...\n")

    if model_type == "reranker":
        models = await collect_reranker_models(session, active)
        registry_key = "rerankers"
    elif model_type == "embedding":
        models = await collect_embedding_models(session, active)
        registry_key = "embeddings"
    elif model_type == "stt":
        models = await collect_stt_models(session, active, log)
        registry_key = "stts"
    else:
        models = await collect_llm_models(session, active)
        registry_key = "llms"

    log(f"Fetching disk sizes for {len(models)} unique models concurrently (max {CONCURRENCY} at a time)...")
    sem = asyncio.Semaphore(CONCURRENCY)
    check_chat_template = registry_key == "llms"
    details = await asyncio.gather(
        *[fetch_model_details(session, m["id"], sem, check_chat_template, cache_dir=DEFAULT_MODEL_DETAIL_CACHE_DIR) for m in models]
    )
    sizes = {mid: size for mid, size, _, _ in details}
    architectures = {mid: arch for mid, _, arch, _ in details}
    chat_capable = {mid: chat_ok for mid, _, _, chat_ok in details}

    models = _drop_unsupported_models(registry_key, models, architectures, chat_capable, log, allow_generative_rerankers)

    entries: list[dict[str, Any]] = []
    for m in models:
        mid = m["id"]
        size = fmt_size_compact(sizes.get(mid, "N/A"))
        entry: dict[str, Any] = {"name": mid, "size": size}
        if registry_key == "rerankers":
            entry["is_generative"] = not is_supported_cross_encoder(architectures.get(mid) or [])
        if registry_key == "llms":
            supports_vision, supports_audio = classify_modality_support(architectures.get(mid) or [])
            entry["supports_vision"] = supports_vision
            entry["supports_audio"] = supports_audio
        entries.append(entry)
    entries.sort(key=lambda e: e["name"].casefold())
    return registry_key, entries


async def main(
    top_by_downloads: int,
    top_by_likes: int,
    top_by_trending: int,
    raw: bool,
    model_type: str,
    output: str | None = None,
    allow_generative_rerankers: bool = False,
) -> None:
    """Fetch models from HuggingFace and print (or write) a vllm-min.json compatible registry."""

    def log(msg: str) -> None:
        if not raw:
            print(msg, file=sys.stderr)

    if top_by_downloads <= 0 and top_by_likes <= 0 and top_by_trending <= 0:
        print("Specify at least one of: --top-by-downloads, --top-by-likes, --top-by-trending", file=sys.stderr)
        return

    async with aiohttp.ClientSession() as session:
        registry_key, entries = await fetch_registry_entries(
            session, top_by_downloads, top_by_likes, top_by_trending, model_type, log, allow_generative_rerankers
        )

    if output is None:
        print(json.dumps({registry_key: entries}, indent=4))
        return

    output_path = Path(output)
    registry: dict[str, Any] = json.loads(output_path.read_text(encoding="utf-8")) if output_path.exists() else {}
    registry[registry_key] = entries
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(registry, indent=4), encoding="utf-8")
    log(f"Written {len(entries)} {registry_key} to {output_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Scrape popular models from HuggingFace")
    p.add_argument("--top-by-downloads", type=int, default=0, metavar="N", help="Fetch top N models sorted by downloads")
    p.add_argument("--top-by-likes", type=int, default=0, metavar="N", help="Fetch top N models sorted by likes")
    p.add_argument("--top-by-trending", type=int, default=0, metavar="N", help="Fetch top N models sorted by trending score")
    p.add_argument("--raw", action="store_true", help="Output JSON only, suppress progress messages")
    p.add_argument(
        "--type",
        choices=["llm", "reranker", "embedding", "stt"],
        default="llm",
        dest="model_type",
        help="Model type to fetch (default: llm)",
    )
    p.add_argument("--output", default=None, metavar="PATH", help="Write into PATH, merging into its existing keys, instead of stdout")
    p.add_argument(
        "--allow-generative-rerankers",
        action="store_true",
        help="Keep generative (non-cross-encoder) reranker candidates instead of dropping them; tags each entry with 'is_generative'",
    )
    args = p.parse_args()
    asyncio.run(
        main(
            args.top_by_downloads,
            args.top_by_likes,
            args.top_by_trending,
            args.raw,
            args.model_type,
            args.output,
            args.allow_generative_rerankers,
        )
    )
