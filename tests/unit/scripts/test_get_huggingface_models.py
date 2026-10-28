# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Tests for get_huggingface_models.py script."""

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

import aiohttp
import pytest
from multidict import CIMultiDict

from scripts.get_huggingface_models import (
    EMBEDDING_TAGS,
    RERANKER_TAGS,
    HfRepoAccessError,
    ModelIncompatibleError,
    _drop_unsupported_models,  # pyright: ignore[reportPrivateUsage]
    _gated_probe_file,  # pyright: ignore[reportPrivateUsage]
    classify_modality_support,
    collect_embedding_models,
    collect_llm_models,
    collect_reranker_models,
    collect_stt_models,
    fetch_model_details,
    fetch_model_metadata,
    fetch_popular_embedding_models,
    fetch_popular_models,
    fetch_popular_reranker_models,
    fetch_registry_entries,
    fmt_size,
    fmt_size_compact,
    get_json_with_retry,
    has_chat_template,
    is_embedding,
    is_llm,
    is_mlx,
    is_non_servable_stt_repo,
    is_reranker,
    is_reranking_cross_encoder,
    is_supported_cross_encoder,
    is_supported_stt,
    is_test_fixture,
    main,
    validate_model_compatibility,
)
from scripts.utils.hf_common import DEFAULT_MODEL_DETAIL_CACHE_DIR, HfErrorKind


@pytest.mark.parametrize(
    ("size_bytes", "expected_output"),
    [
        (512, "512.0 B"),
        (1024, "1.0 KB"),
        (1024**2, "1.0 MB"),
        (1024**3, "1.0 GB"),
        (1024**4, "1.0 TB"),
        (1024**5, "1.0 PB"),
        (0, "0.0 B"),
        (int(2.5 * 1024**3), "2.5 GB"),
    ],
)
def test_fmt_size_variants(size_bytes: int, expected_output: str) -> None:
    assert fmt_size(size_bytes) == expected_output


@pytest.mark.parametrize(
    ("input_val", "expected"),
    [
        ("N/A", "N/A"),
        ("2.0 GB", "2GB"),
        ("4.9 GB", "4GB"),
        ("512.0 MB", "512MB"),
        ("broken", "broken"),
    ],
)
def test_fmt_size_compact(input_val: str, expected: str):
    assert fmt_size_compact(input_val) == expected


@pytest.mark.parametrize(
    ("model_data", "expected"),
    [
        ({"id": "meta-llama/Llama-3-8b"}, True),
        ({"id": "Helsinki-NLP/opus-mt-en-translation"}, False),
        ({"id": "sshleifer/distilbart-cnn-summarization"}, False),
        ({"id": "bert-base-uncased-classification"}, False),
        ({"id": "dslim/bert-base-ner"}, False),
        ({"id": "deepset/roberta-base-qa-squad2"}, False),
        ({"id": ""}, True),
        ({}, True),
        ({"id": "Model-TRANSLATION-v2"}, False),
        ({"id": "deepseek-ai/DeepSeek-OCR"}, False),
        ({"id": "PaddlePaddle/PaddleOCR-VL"}, False),
        ({"id": "stepfun-ai/GOT-OCR2_0"}, False),
        ({"id": "trl-internal-testing/tiny-Gemma3ForConditionalGeneration"}, True),
        ({"id": "gaunernst/gemma-3-27b-it-int4-awq"}, True),
    ],
    ids=[
        "plain_llm",
        "translation_excluded",
        "summarization_excluded",
        "classification_excluded",
        "ner_excluded",
        "qa_dash_excluded",
        "empty_id",
        "missing_id",
        "case_insensitive",
        "ocr_excluded",
        "ocr_glued_to_name_excluded",
        "ocr_followed_by_digit_excluded",
        "ner_inside_generation_kept",
        "ner_inside_author_name_kept",
    ],
)
def test_is_llm(model_data: dict[str, str], expected: bool) -> None:
    assert is_llm(model_data) is expected


@pytest.mark.parametrize(
    ("input_dict", "expected"),
    [
        ({"id": "cross-encoder/ms-marco-reranker"}, True),
        ({"id": "bert-base-uncased"}, False),
        ({"id": ""}, False),
        ({}, False),
        ({"id": "BAAI/bge-RERANK-v2"}, True),
    ],
    ids=["reranker_in_name", "no_reranker", "empty_id", "missing_id", "case_insensitive"],
)
def test_is_reranker(input_dict: dict[str, str], expected: bool):
    assert is_reranker(input_dict) is expected


@pytest.mark.parametrize(
    ("architectures", "expected"),
    [
        (["XLMRobertaForSequenceClassification"], True),
        (["ModernBertForSequenceClassification"], True),
        (["Qwen3ForCausalLM"], False),
        (["JinaForRanking"], False),
        ([], False),
    ],
    ids=[
        "sequence_classification_head",
        "another_sequence_classification_head",
        "causal_lm_excluded",
        "custom_ranking_head_excluded",
        "no_architecture_info",
    ],
)
def test_is_supported_cross_encoder(architectures: list[str], expected: bool) -> None:
    assert is_supported_cross_encoder(architectures) is expected


@pytest.mark.parametrize(
    ("architectures", "expected"),
    [
        (["Qwen2_5_VLForConditionalGeneration"], (True, None)),
        (["SomeNewQwen3VLForConditionalGeneration"], (True, None)),
        (["Qwen2AudioForConditionalGeneration"], (None, True)),
        (["Qwen3ForCausalLM"], (None, None)),
        ([], (None, None)),
    ],
    ids=[
        "exact_allowlist_match",
        "vl_substring_match",
        "audio_allowlist_match",
        "unlisted_architecture_is_unknown_not_false",
        "empty_architectures_is_unknown_not_false",
    ],
)
def test_classify_modality_support(architectures: list[str], expected: tuple[bool | None, bool | None]) -> None:
    assert classify_modality_support(architectures) == expected


@pytest.mark.asyncio
async def test_has_chat_template_true_for_standalone_jinja_file() -> None:
    session = MagicMock()
    siblings = [{"rfilename": "chat_template.jinja"}, {"rfilename": "config.json"}]

    result = await has_chat_template(session, "org/model", siblings)

    assert result is True
    session.get.assert_not_called()


@pytest.mark.asyncio
async def test_has_chat_template_false_when_no_tokenizer_config() -> None:
    session = MagicMock()
    siblings = [{"rfilename": "config.json"}]

    result = await has_chat_template(session, "org/model", siblings)

    assert result is False
    session.get.assert_not_called()


@pytest.mark.asyncio
async def test_has_chat_template_true_when_embedded_in_tokenizer_config() -> None:
    session = _make_mock_session({"chat_template": "{% for message in messages %}...{% endfor %}"})
    siblings = [{"rfilename": "tokenizer_config.json"}]

    result = await has_chat_template(session, "org/model", siblings)

    assert result is True


@pytest.mark.asyncio
async def test_has_chat_template_false_when_tokenizer_config_lacks_key() -> None:
    session = _make_mock_session({"tokenizer_class": "GPT2Tokenizer"})
    siblings = [{"rfilename": "tokenizer_config.json"}]

    result = await has_chat_template(session, "org/model", siblings)

    assert result is False


@pytest.mark.asyncio
async def test_has_chat_template_fetches_pinned_revision_when_given() -> None:
    session = _make_mock_session({"chat_template": "..."})
    siblings = [{"rfilename": "tokenizer_config.json"}]

    await has_chat_template(session, "org/model", siblings, revision="v2")

    url = session.get.call_args.args[0]
    assert url == "https://huggingface.co/org/model/raw/v2/tokenizer_config.json"


@pytest.mark.asyncio
async def test_has_chat_template_fetches_main_when_no_revision_given() -> None:
    session = _make_mock_session({"chat_template": "..."})
    siblings = [{"rfilename": "tokenizer_config.json"}]

    await has_chat_template(session, "org/model", siblings)

    url = session.get.call_args.args[0]
    assert url == "https://huggingface.co/org/model/raw/main/tokenizer_config.json"


@pytest.mark.asyncio
async def test_has_chat_template_strict_reraises_on_fetch_error() -> None:
    mock_resp = AsyncMock()
    mock_resp.raise_for_status = MagicMock(side_effect=aiohttp.ClientResponseError(MagicMock(), (), status=503))
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)
    session = MagicMock()
    session.get = MagicMock(return_value=mock_resp)
    siblings = [{"rfilename": "tokenizer_config.json"}]

    with pytest.raises(aiohttp.ClientResponseError):
        await has_chat_template(session, "org/model", siblings, max_retries=1, strict=True)


@pytest.mark.asyncio
async def test_has_chat_template_false_on_fetch_error() -> None:
    mock_resp = AsyncMock()
    mock_resp.raise_for_status = MagicMock(side_effect=aiohttp.ClientResponseError(MagicMock(), (), status=404))
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)
    session = MagicMock()
    session.get = MagicMock(return_value=mock_resp)
    siblings = [{"rfilename": "tokenizer_config.json"}]

    result = await has_chat_template(session, "org/missing", siblings)

    assert result is False


@pytest.mark.parametrize(
    ("input_dict", "expected"),
    [
        ({"id": "org/text-embedding-3-large"}, True),
        ({"id": "bert-base-uncased"}, False),
        ({"id": ""}, False),
        ({}, False),
        ({"id": "BAAI/bge-EMBED-v2"}, True),
    ],
    ids=["embed_in_name", "no_embed", "empty_id", "missing_id", "case_insensitive"],
)
def test_is_embedding(input_dict: dict[str, str], expected: bool):
    assert is_embedding(input_dict) is expected


def _make_mock_session(json_data: object) -> MagicMock:
    mock_resp = AsyncMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json = AsyncMock(return_value=json_data)
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)
    mock_session = MagicMock()
    mock_session.get = MagicMock(return_value=mock_resp)
    return mock_session


@pytest.mark.asyncio
async def test_validate_model_compatibility_raises_for_gguf_repo() -> None:
    session = _make_mock_session({"id": "org/model-GGUF", "tags": ["gguf"], "siblings": []})

    with pytest.raises(ModelIncompatibleError, match="GGUF"):
        await validate_model_compatibility(session, "org/model-GGUF", None, "llm")


@pytest.mark.asyncio
async def test_validate_model_compatibility_raises_for_llm_without_chat_template() -> None:
    session = _make_mock_session({"id": "org/base-model", "tags": [], "siblings": [{"rfilename": "config.json"}]})

    with pytest.raises(ModelIncompatibleError, match="chat template"):
        await validate_model_compatibility(session, "org/base-model", None, "llm")


@pytest.mark.asyncio
async def test_validate_model_compatibility_passes_for_llm_with_chat_template() -> None:
    data = {"id": "org/chat-model", "tags": [], "siblings": [{"rfilename": "chat_template.jinja"}]}
    session = _make_mock_session(data)

    result = await validate_model_compatibility(session, "org/chat-model", None, "llm")

    assert result == data


@pytest.mark.asyncio
async def test_validate_model_compatibility_raises_for_unsupported_reranker_architecture() -> None:
    session = _make_mock_session(
        {"id": "org/generative-reranker", "tags": [], "siblings": [], "config": {"architectures": ["Qwen3ForCausalLM"]}}
    )

    with pytest.raises(ModelIncompatibleError, match="cross-encoder"):
        await validate_model_compatibility(session, "org/generative-reranker", None, "reranker")


@pytest.mark.asyncio
async def test_validate_model_compatibility_raises_for_unsupported_stt_architecture() -> None:
    session = _make_mock_session(
        {"id": "facebook/wav2vec2-base", "tags": [], "siblings": [], "config": {"architectures": ["Wav2Vec2ForCTC"]}}
    )

    with pytest.raises(ModelIncompatibleError, match="audio/transcriptions"):
        await validate_model_compatibility(session, "facebook/wav2vec2-base", None, "stt")


@pytest.mark.asyncio
async def test_validate_model_compatibility_passes_for_supported_stt() -> None:
    data = {"id": "openai/whisper-small", "tags": [], "siblings": [], "config": {"architectures": ["WhisperForConditionalGeneration"]}}
    session = _make_mock_session(data)

    result = await validate_model_compatibility(session, "openai/whisper-small", None, "stt")

    assert result == data


@pytest.mark.asyncio
async def test_validate_model_compatibility_passes_for_supported_reranker() -> None:
    session = _make_mock_session(
        {
            "id": "org/cross-encoder-reranker",
            "tags": [],
            "siblings": [],
            "config": {"architectures": ["XLMRobertaForSequenceClassification"]},
        }
    )

    await validate_model_compatibility(session, "org/cross-encoder-reranker", None, "reranker")


@pytest.mark.asyncio
async def test_validate_model_compatibility_skips_check_for_embedding_model_type() -> None:
    session = _make_mock_session({"id": "org/embedding-model", "tags": [], "siblings": []})

    await validate_model_compatibility(session, "org/embedding-model", None, "embedding")


@pytest.mark.asyncio
async def test_validate_model_compatibility_returns_none_on_transient_fetch_error() -> None:
    mock_resp = AsyncMock()
    mock_resp.raise_for_status = MagicMock(side_effect=aiohttp.ClientResponseError(MagicMock(), (), status=503))
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)
    session = MagicMock()
    session.get = MagicMock(return_value=mock_resp)

    result = await validate_model_compatibility(session, "org/missing-model", None, "llm", max_retries=1)

    assert result is None


@pytest.mark.asyncio
async def test_validate_model_compatibility_returns_none_on_transient_chat_template_fetch_error() -> None:
    metadata_resp = AsyncMock()
    metadata_resp.raise_for_status = MagicMock()
    metadata_resp.json = AsyncMock(return_value={"id": "org/model", "tags": [], "siblings": [{"rfilename": "tokenizer_config.json"}]})
    metadata_resp.__aenter__ = AsyncMock(return_value=metadata_resp)
    metadata_resp.__aexit__ = AsyncMock(return_value=False)

    template_resp = AsyncMock()
    template_resp.raise_for_status = MagicMock(side_effect=aiohttp.ClientResponseError(MagicMock(), (), status=503))
    template_resp.__aenter__ = AsyncMock(return_value=template_resp)
    template_resp.__aexit__ = AsyncMock(return_value=False)

    session = MagicMock()
    session.get = MagicMock(side_effect=[metadata_resp, template_resp])

    result = await validate_model_compatibility(session, "org/model", None, "llm", max_retries=1)

    assert result is None


@pytest.mark.asyncio
async def test_validate_model_compatibility_builds_revision_pinned_url() -> None:
    session = _make_mock_session({"id": "org/model", "tags": [], "siblings": [{"rfilename": "chat_template.jinja"}]})

    await validate_model_compatibility(session, "org/model", "v2", "llm")

    url = session.get.call_args.args[0]
    assert url == "https://huggingface.co/api/models/org/model/revision/v2"


@pytest.mark.asyncio
async def test_validate_model_compatibility_returns_none_on_connection_error() -> None:
    session = MagicMock()
    session.get = MagicMock(side_effect=aiohttp.ClientConnectionError("boom"))

    result = await validate_model_compatibility(session, "org/missing-model", None, "llm", max_retries=1)

    assert result is None


def _make_mock_response(json_data: object = None, status: int | None = None, error_code: str | None = None) -> AsyncMock:
    mock_resp = AsyncMock()
    if status is None:
        mock_resp.raise_for_status = MagicMock()
    else:
        headers = CIMultiDict({"X-Error-Code": error_code}) if error_code else None
        mock_resp.raise_for_status = MagicMock(side_effect=aiohttp.ClientResponseError(MagicMock(), (), status=status, headers=headers))
    mock_resp.json = AsyncMock(return_value=json_data)
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)
    return mock_resp


@pytest.mark.parametrize(
    ("status", "error_code", "authenticated", "expected_kind", "expected_message"),
    [
        (404, None, False, HfErrorKind.NOT_FOUND, "org/missing-model doesn't exist on HuggingFace; check the repo id."),
        (404, "RevisionNotFound", False, HfErrorKind.REVISION_NOT_FOUND, "Revision 'v2' doesn't exist in org/missing-model"),
        (
            401,
            None,
            False,
            HfErrorKind.NO_ACCESS,
            "doesn't exist or is private; check the repo id, or set DF_HUGGING_FACE_TOKEN in Configuration",
        ),
        (401, None, True, HfErrorKind.NO_ACCESS, "the configured HuggingFace token is invalid or has no access to it"),
        (403, None, True, HfErrorKind.NO_ACCESS, "the configured HuggingFace token is invalid or has no access to it"),
        (
            401,
            "GatedRepo",
            False,
            HfErrorKind.GATED,
            "is gated; request access on its HuggingFace page and set DF_HUGGING_FACE_TOKEN in Configuration",
        ),
        (
            403,
            "GatedRepo",
            True,
            HfErrorKind.GATED,
            "is gated, and the configured HuggingFace token is invalid or its account has no access",
        ),
    ],
    ids=(
        "repo_not_found",
        "revision_not_found",
        "missing_or_private_anonymous",
        "missing_or_no_access_with_token",
        "forbidden_with_token",
        "gated_anonymous",
        "gated_with_token",
    ),
)
@pytest.mark.asyncio
async def test_validate_model_compatibility_raises_access_error_for_permanent_fetch_error(
    status: int, error_code: str | None, authenticated: bool, expected_kind: HfErrorKind, expected_message: str
) -> None:
    session = MagicMock()
    session.get = MagicMock(return_value=_make_mock_response(status=status, error_code=error_code))

    with pytest.raises(HfRepoAccessError) as exc_info:
        await validate_model_compatibility(session, "org/missing-model", "v2", "llm", max_retries=1, authenticated=authenticated)

    assert isinstance(exc_info.value, ModelIncompatibleError)
    assert exc_info.value.kind is expected_kind
    assert expected_message in str(exc_info.value)


@pytest.mark.parametrize(
    "exc",
    [aiohttp.ClientConnectionError("boom"), TimeoutError()],
    ids=("connection_error", "timeout"),
)
@pytest.mark.asyncio
async def test_fetch_model_metadata_returns_none_on_transient_error(exc: Exception) -> None:
    session = MagicMock()
    session.get = MagicMock(side_effect=exc)

    result = await fetch_model_metadata(session, "org/model", None, authenticated=False, max_retries=1)

    assert result is None


@pytest.mark.parametrize("gated", ["manual", "auto"])
@pytest.mark.asyncio
async def test_fetch_model_metadata_raises_gated_error_without_token(gated: str) -> None:
    session = _make_mock_session({"id": "org/gated-model", "gated": gated, "siblings": []})

    with pytest.raises(HfRepoAccessError) as exc_info:
        await fetch_model_metadata(session, "org/gated-model", None, authenticated=False)

    assert exc_info.value.kind is HfErrorKind.GATED
    assert "set DF_HUGGING_FACE_TOKEN in Configuration" in str(exc_info.value)


@pytest.mark.asyncio
async def test_fetch_model_metadata_returns_gated_metadata_when_token_has_access() -> None:
    data = {"id": "org/gated-model", "gated": "manual", "siblings": [{"rfilename": "model.safetensors", "size": 1024}]}
    session = _make_mock_session(data)
    session.head = MagicMock(return_value=_make_mock_response())

    result = await fetch_model_metadata(session, "org/gated-model", None, authenticated=True)

    assert result == data
    assert session.head.call_count == 1
    assert session.head.call_args == call("https://huggingface.co/org/gated-model/resolve/main/model.safetensors", allow_redirects=False)


@pytest.mark.parametrize(
    ("status", "error_code", "expected_kind", "expected_message"),
    [
        (401, "GatedRepo", HfErrorKind.GATED, "is gated, and the configured HuggingFace token is invalid or its account has no access"),
        (403, None, HfErrorKind.NO_ACCESS, "the configured HuggingFace token is invalid or has no access to it"),
    ],
    ids=("gated", "no_access"),
)
@pytest.mark.asyncio
async def test_fetch_model_metadata_raises_when_token_has_no_access_to_gated_files(
    status: int, error_code: str | None, expected_kind: HfErrorKind, expected_message: str
) -> None:
    session = _make_mock_session({"id": "org/gated-model", "gated": "manual", "siblings": [{"rfilename": "config.json"}]})
    session.head = MagicMock(return_value=_make_mock_response(status=status, error_code=error_code))

    with pytest.raises(HfRepoAccessError) as exc_info:
        await fetch_model_metadata(session, "org/gated-model", None, authenticated=True)

    assert exc_info.value.kind is expected_kind
    assert expected_message in str(exc_info.value)


@pytest.mark.parametrize(
    "head_kwargs",
    [
        {"return_value": _make_mock_response(status=404, error_code="EntryNotFound")},
        {"return_value": _make_mock_response(status=503)},
        {"side_effect": TimeoutError()},
        {"side_effect": aiohttp.ClientConnectionError("boom")},
    ],
    ids=("file_missing", "outage", "timeout", "connection_error"),
)
@pytest.mark.asyncio
async def test_fetch_model_metadata_stays_lenient_when_gated_access_check_inconclusive(head_kwargs: dict[str, Any]) -> None:
    data = {"id": "org/gated-model", "gated": "manual", "siblings": [{"rfilename": "config.json"}]}
    session = _make_mock_session(data)
    session.head = MagicMock(**head_kwargs)

    result = await fetch_model_metadata(session, "org/gated-model", None, authenticated=True)

    assert result == data


@pytest.mark.asyncio
async def test_fetch_model_metadata_probes_gated_file_at_encoded_revision() -> None:
    session = _make_mock_session({"id": "org/gated-model", "gated": "auto", "siblings": [{"rfilename": "config.json"}]})
    session.head = MagicMock(return_value=_make_mock_response())

    await fetch_model_metadata(session, "org/gated-model", "refs/pr/1", authenticated=True)

    assert session.head.call_args == call("https://huggingface.co/org/gated-model/resolve/refs%2Fpr%2F1/config.json", allow_redirects=False)


@pytest.mark.parametrize(
    "data",
    [
        {"id": "org/public-model", "gated": False, "siblings": [{"rfilename": "config.json"}]},
        {"id": "org/gated-model", "gated": "manual", "siblings": [{"rfilename": "README.md"}, {"rfilename": "LICENSE.txt"}]},
    ],
    ids=("ungated", "gated_without_probe_file"),
)
@pytest.mark.asyncio
async def test_fetch_model_metadata_skips_gated_access_check(data: dict[str, Any]) -> None:
    session = _make_mock_session(data)
    session.head = MagicMock()

    result = await fetch_model_metadata(session, data["id"], None, authenticated=True)

    assert result == data
    assert session.head.call_count == 0


@pytest.mark.asyncio
async def test_fetch_model_metadata_returns_ungated_metadata_without_token() -> None:
    data = {"id": "org/public-model", "gated": False, "siblings": []}
    session = _make_mock_session(data)

    result = await fetch_model_metadata(session, "org/public-model", None, authenticated=False)

    assert result == data


@pytest.mark.parametrize(
    ("siblings", "expected"),
    [
        ([{"rfilename": "README.md"}, {"rfilename": "model.safetensors"}, {"rfilename": "config.json"}], "config.json"),
        (
            [{"rfilename": "LICENSE.txt"}, {"rfilename": "model-00001-of-00002.safetensors"}, {"rfilename": "tokenizer.json"}],
            "model-00001-of-00002.safetensors",
        ),
        ([{"rfilename": "README.md"}, {"rfilename": "tokenizer.json"}], "tokenizer.json"),
        ([{"rfilename": "README.md"}, {"rfilename": "LICENSE"}, {"rfilename": ".gitattributes"}], None),
        ([], None),
    ],
    ids=("config_json_first", "weights_next", "any_json", "only_public_files", "no_files"),
)
def test_gated_probe_file_prefers_files_needed_for_download(siblings: list[dict[str, Any]], expected: str | None) -> None:
    result = _gated_probe_file(siblings)

    assert result == expected


@pytest.mark.asyncio
async def test_validate_model_compatibility_raises_for_gated_embedding_model_without_access() -> None:
    session = _make_mock_session({"id": "org/gated-embedder", "gated": "manual", "tags": [], "siblings": [{"rfilename": "config.json"}]})
    session.head = MagicMock(return_value=_make_mock_response(status=403, error_code="GatedRepo"))

    with pytest.raises(HfRepoAccessError) as exc_info:
        await validate_model_compatibility(session, "org/gated-embedder", None, "embedding", authenticated=True)

    assert exc_info.value.kind is HfErrorKind.GATED


@pytest.mark.parametrize(
    ("status", "error_code", "expected_kind"),
    [
        (403, "GatedRepo", HfErrorKind.GATED),
        (401, None, HfErrorKind.NO_ACCESS),
    ],
    ids=("gated", "no_access"),
)
@pytest.mark.asyncio
async def test_validate_model_compatibility_raises_access_error_when_chat_template_fetch_denied(
    status: int, error_code: str | None, expected_kind: HfErrorKind
) -> None:
    metadata = {"id": "org/model", "tags": [], "siblings": [{"rfilename": "tokenizer_config.json"}]}
    session = MagicMock()
    session.get = MagicMock(side_effect=[_make_mock_response(metadata), _make_mock_response(status=status, error_code=error_code)])

    with pytest.raises(HfRepoAccessError) as exc_info:
        await validate_model_compatibility(session, "org/model", None, "llm", max_retries=1, authenticated=True)

    assert exc_info.value.kind is expected_kind


@pytest.mark.asyncio
async def test_validate_model_compatibility_returns_none_when_tokenizer_config_missing() -> None:
    metadata = {"id": "org/model", "tags": [], "siblings": [{"rfilename": "tokenizer_config.json"}]}
    session = MagicMock()
    session.get = MagicMock(side_effect=[_make_mock_response(metadata), _make_mock_response(status=404, error_code="EntryNotFound")])

    result = await validate_model_compatibility(session, "org/model", None, "llm", max_retries=1)

    assert result is None


@pytest.mark.asyncio
async def test_validate_model_compatibility_propagates_unexpected_errors() -> None:
    session = MagicMock()
    session.get = MagicMock(side_effect=ValueError("not a network failure"))

    with pytest.raises(ValueError, match="not a network failure"):
        await validate_model_compatibility(session, "org/model", None, "llm")


@pytest.mark.asyncio
async def test_fetch_popular_models_returns_list() -> None:
    data = [{"id": "org/model-a"}, {"id": "org/model-b"}]
    session = _make_mock_session(data)

    result = await fetch_popular_models(session, "text-generation", "downloads", 10)

    assert result == data


@pytest.mark.asyncio
async def test_fetch_popular_models_passes_correct_params() -> None:
    session = _make_mock_session([])

    await fetch_popular_models(session, "text-generation", "likes", 50)

    call_kwargs = session.get.call_args
    params = call_kwargs[1]["params"]
    assert params["pipeline_tag"] == "text-generation"
    assert params["sort"] == "likes"
    assert params["limit"] == "50"


@pytest.mark.asyncio
async def test_fetch_popular_models_passes_extra_filter() -> None:
    session = _make_mock_session([])

    await fetch_popular_models(session, "automatic-speech-recognition", "downloads", 10, extra_filter="whisper")

    params = session.get.call_args[1]["params"]
    assert params["filter"] == "whisper"


@pytest.mark.asyncio
async def test_fetch_popular_models_omits_filter_without_extra_filter() -> None:
    session = _make_mock_session([])

    await fetch_popular_models(session, "text-generation", "downloads", 10)

    params = session.get.call_args[1]["params"]
    assert "filter" not in params


def test_drop_unsupported_models_keeps_all_stts_when_all_supported() -> None:
    models = [{"id": "openai/whisper-small"}, {"id": "openai/whisper-base"}]
    architectures: dict[str, list[str] | None] = {
        "openai/whisper-small": ["WhisperForConditionalGeneration"],
        "openai/whisper-base": ["WhisperForConditionalGeneration"],
    }
    log = MagicMock()

    result = _drop_unsupported_models("stts", models, architectures, {}, log)

    assert result == models
    assert log.call_count == 0


@pytest.mark.asyncio
async def test_fetch_popular_models_raises_on_http_error() -> None:
    mock_resp = AsyncMock()
    mock_resp.raise_for_status = MagicMock(side_effect=aiohttp.ClientResponseError(MagicMock(), (), status=429))
    mock_resp.headers = {}
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)
    session = MagicMock()
    session.get = MagicMock(return_value=mock_resp)

    with patch("asyncio.sleep", AsyncMock()), pytest.raises(aiohttp.ClientResponseError):
        await fetch_popular_models(session, "text-generation", "downloads", 10)


@pytest.mark.asyncio
async def test_get_json_with_retry_retries_on_429_then_succeeds() -> None:
    failing_resp = AsyncMock()
    failing_resp.raise_for_status = MagicMock(side_effect=aiohttp.ClientResponseError(MagicMock(), (), status=429))
    failing_resp.headers = {}
    failing_resp.__aenter__ = AsyncMock(return_value=failing_resp)
    failing_resp.__aexit__ = AsyncMock(return_value=False)

    ok_resp = AsyncMock()
    ok_resp.raise_for_status = MagicMock()
    ok_resp.json = AsyncMock(return_value=[{"id": "org/model"}])
    ok_resp.__aenter__ = AsyncMock(return_value=ok_resp)
    ok_resp.__aexit__ = AsyncMock(return_value=False)

    session = MagicMock()
    session.get = MagicMock(side_effect=[failing_resp, failing_resp, ok_resp])

    with patch("asyncio.sleep", AsyncMock()) as mock_sleep:
        result = await get_json_with_retry(session, "https://huggingface.co/api/models", {})

    assert result == [{"id": "org/model"}]
    assert mock_sleep.await_count == 2


@pytest.mark.asyncio
async def test_get_json_with_retry_respects_retry_after_header() -> None:
    failing_resp = AsyncMock()
    failing_resp.raise_for_status = MagicMock(side_effect=aiohttp.ClientResponseError(MagicMock(), (), status=429))
    failing_resp.headers = {"Retry-After": "7"}
    failing_resp.__aenter__ = AsyncMock(return_value=failing_resp)
    failing_resp.__aexit__ = AsyncMock(return_value=False)

    ok_resp = AsyncMock()
    ok_resp.raise_for_status = MagicMock()
    ok_resp.json = AsyncMock(return_value=[])
    ok_resp.__aenter__ = AsyncMock(return_value=ok_resp)
    ok_resp.__aexit__ = AsyncMock(return_value=False)

    session = MagicMock()
    session.get = MagicMock(side_effect=[failing_resp, ok_resp])

    with patch("asyncio.sleep", AsyncMock()) as mock_sleep:
        await get_json_with_retry(session, "https://huggingface.co/api/models", {})

    mock_sleep.assert_awaited_once_with(7.0)


@pytest.mark.asyncio
async def test_get_json_with_retry_does_not_retry_on_client_error() -> None:
    mock_resp = AsyncMock()
    mock_resp.raise_for_status = MagicMock(side_effect=aiohttp.ClientResponseError(MagicMock(), (), status=404))
    mock_resp.headers = {}
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    session = MagicMock()
    session.get = MagicMock(return_value=mock_resp)

    with patch("asyncio.sleep", AsyncMock()) as mock_sleep, pytest.raises(aiohttp.ClientResponseError):
        await get_json_with_retry(session, "https://huggingface.co/api/models", {})

    mock_sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_fetch_popular_reranker_models_returns_list() -> None:
    data = [{"id": "org/reranker-a"}]
    session = _make_mock_session(data)

    result = await fetch_popular_reranker_models(session, "downloads", 10)

    assert result == data


@pytest.mark.asyncio
async def test_fetch_popular_reranker_models_passes_search_param() -> None:
    session = _make_mock_session([])

    await fetch_popular_reranker_models(session, "trending", 20)

    call_kwargs = session.get.call_args
    params = call_kwargs[1]["params"]
    assert params["search"] == "rerank"
    assert params["sort"] == "trending"
    assert params["limit"] == "20"


@pytest.mark.asyncio
async def test_fetch_popular_embedding_models_returns_list() -> None:
    data = [{"id": "org/embed-a"}]
    session = _make_mock_session(data)

    result = await fetch_popular_embedding_models(session, "downloads", 10)

    assert result == data


@pytest.mark.asyncio
async def test_fetch_popular_embedding_models_passes_search_param() -> None:
    session = _make_mock_session([])

    await fetch_popular_embedding_models(session, "trending", 20)

    call_kwargs = session.get.call_args
    params = call_kwargs[1]["params"]
    assert params["search"] == "embed"
    assert params["sort"] == "trending"
    assert params["limit"] == "20"


@pytest.mark.asyncio
async def test_collect_llm_models_deduplicates() -> None:
    llm = {"id": "org/llama"}
    excluded = {"id": "org/translation-model"}

    async def fake_fetch(session: Mock, tag: str, sort: str, limit: int):
        return [llm, excluded]

    with patch("scripts.get_huggingface_models.fetch_popular_models", side_effect=fake_fetch):
        result = await collect_llm_models(MagicMock(), {"downloads": 10})

    ids = [m["id"] for m in result]
    assert ids.count("org/llama") == 1


@pytest.mark.parametrize(
    "model_id",
    [
        "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
        "trl-internal-testing/small-Qwen3ForCausalLM",
        "peft-internal-testing/foo",
        "hmellor/tiny-random-LlamaForCausalLM",
        "llamafactory/tiny-random-Llama-3",
    ],
)
def test_is_test_fixture_true(model_id: str) -> None:
    assert is_test_fixture({"id": model_id})


@pytest.mark.parametrize("model_id", ["Qwen/Qwen3-8B", "meta-llama/Llama-3.1-8B-Instruct", "org/tiny-model"])
def test_is_test_fixture_false(model_id: str) -> None:
    assert not is_test_fixture({"id": model_id})


@pytest.mark.parametrize(
    "model_id",
    [
        "lmstudio-community/Qwen3.5-9B-MLX-4bit",
        "mlx-community/Llama-3.1-8B-Instruct-4bit",
        "prism-ml/Bonsai-27B-mlx-1bit",
        "orcarouter/Qwen3.8-27B-Uncensored-MLX",
    ],
)
def test_is_mlx_true(model_id: str) -> None:
    assert is_mlx({"id": model_id})


@pytest.mark.parametrize("model_id", ["Qwen/Qwen3-8B", "org/mlxtral-7b", "org/Gemlx-8B"])
def test_is_mlx_false(model_id: str) -> None:
    assert not is_mlx({"id": model_id})


@pytest.mark.asyncio
async def test_collect_llm_models_filters_fixtures_and_mlx() -> None:
    async def fake_fetch(session: Mock, tag: str, sort: str, limit: int):
        return [
            {"id": "org/llama"},
            {"id": "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"},
            {"id": "lmstudio-community/Qwen3.5-9B-MLX-4bit"},
        ]

    with patch("scripts.get_huggingface_models.fetch_popular_models", side_effect=fake_fetch):
        result = await collect_llm_models(MagicMock(), {"downloads": 10})

    assert [m["id"] for m in result] == ["org/llama"]


@pytest.mark.asyncio
async def test_collect_llm_models_filters_non_llm() -> None:
    async def fake_fetch(session: Mock, tag: str, sort: str, limit: int):
        return [{"id": "org/translation-model"}]

    with patch("scripts.get_huggingface_models.fetch_popular_models", side_effect=fake_fetch):
        result = await collect_llm_models(MagicMock(), {"downloads": 10})

    assert result == []


@pytest.mark.asyncio
async def test_collect_llm_models_merges_multiple_sorts() -> None:
    calls: list[str] = []

    async def fake_fetch(session: Mock, tag: str, sort: str, limit: int):
        calls.append(sort)
        return [{"id": f"org/model-{sort}"}]

    with patch("scripts.get_huggingface_models.fetch_popular_models", side_effect=fake_fetch):
        result = await collect_llm_models(MagicMock(), {"downloads": 5, "likes": 5})

    assert len(result) == 2
    assert len(calls) == 6  # 2 sorts x 3 LLM_TAGS


@pytest.mark.asyncio
async def test_collect_llm_models_queries_multimodal_tag() -> None:
    tags: list[str] = []

    async def fake_fetch(session: Mock, tag: str, sort: str, limit: int):
        tags.append(tag)
        return [{"id": "google/gemma-4-31B-it-qat-w4a16-ct"}] if tag == "image-text-to-text" else []

    with patch("scripts.get_huggingface_models.fetch_popular_models", side_effect=fake_fetch):
        result = await collect_llm_models(MagicMock(), {"downloads": 10})

    assert "image-text-to-text" in tags
    assert [m["id"] for m in result] == ["google/gemma-4-31B-it-qat-w4a16-ct"]


@pytest.mark.asyncio
async def test_collect_llm_models_filters_ocr_models() -> None:
    async def fake_fetch(session: Mock, tag: str, sort: str, limit: int):
        return [{"id": "deepseek-ai/DeepSeek-OCR"}, {"id": "google/gemma-4-31B-it"}]

    with patch("scripts.get_huggingface_models.fetch_popular_models", side_effect=fake_fetch):
        result = await collect_llm_models(MagicMock(), {"downloads": 10})

    assert [m["id"] for m in result] == ["google/gemma-4-31B-it"]


async def _no_tag_candidates(session: Mock, tag: str, sort: str, limit: int) -> list[dict[str, str]]:
    return []


@pytest.mark.asyncio
async def test_collect_reranker_models_deduplicates() -> None:
    reranker = {"id": "org/bge-reranker-v2"}

    async def fake_fetch(session: Mock, sort: str, limit: int):
        return [reranker]

    with (
        patch("scripts.get_huggingface_models.fetch_popular_models", side_effect=_no_tag_candidates),
        patch("scripts.get_huggingface_models.fetch_popular_reranker_models", side_effect=fake_fetch),
    ):
        result = await collect_reranker_models(MagicMock(), {"downloads": 10, "likes": 10})

    assert len(result) == 1


@pytest.mark.asyncio
async def test_collect_reranker_models_filters_non_rerankers() -> None:
    async def fake_fetch(session: Mock, sort: str, limit: int):
        return [{"id": "org/bert-base"}]

    with (
        patch("scripts.get_huggingface_models.fetch_popular_models", side_effect=_no_tag_candidates),
        patch("scripts.get_huggingface_models.fetch_popular_reranker_models", side_effect=fake_fetch),
    ):
        result = await collect_reranker_models(MagicMock(), {"downloads": 10})

    assert result == []


@pytest.mark.asyncio
async def test_collect_reranker_models_finds_by_trusted_tag_without_name_match() -> None:
    """A model with no "rerank" in its name (invisible to the name search) is still found via the
    trusted text-ranking pipeline_tag."""

    async def fake_tag_fetch(session: Mock, tag: str, sort: str, limit: int):
        return [{"id": "cross-encoder/ms-marco-MiniLM-L6-v2"}] if tag == "text-ranking" else []

    with (
        patch("scripts.get_huggingface_models.fetch_popular_models", side_effect=fake_tag_fetch),
        patch("scripts.get_huggingface_models.fetch_popular_reranker_models", return_value=[]),
    ):
        result = await collect_reranker_models(MagicMock(), {"downloads": 10})

    assert [m["id"] for m in result] == ["cross-encoder/ms-marco-MiniLM-L6-v2"]


@pytest.mark.asyncio
async def test_collect_reranker_models_text_ranking_excludes_non_reranking_cross_encoders() -> None:
    """text-ranking is tagged onto any CrossEncoder model regardless of what it scores, so STS/NLI/
    duplicate-question cross-encoders sharing that tag with real rerankers must still be excluded."""

    async def fake_tag_fetch(session: Mock, tag: str, sort: str, limit: int) -> list[dict[str, str]]:
        if tag != "text-ranking":
            return []
        return [{"id": "cross-encoder/ms-marco-MiniLM-L6-v2"}, {"id": "cross-encoder/stsb-roberta-large"}]

    with (
        patch("scripts.get_huggingface_models.fetch_popular_models", side_effect=fake_tag_fetch),
        patch("scripts.get_huggingface_models.fetch_popular_reranker_models", return_value=[]),
    ):
        result = await collect_reranker_models(MagicMock(), {"downloads": 10})

    assert [m["id"] for m in result] == ["cross-encoder/ms-marco-MiniLM-L6-v2"]


@pytest.mark.asyncio
async def test_collect_reranker_models_text_classification_requires_sentence_transformers_library() -> None:
    """text-classification is a noisy tag (dominated by sentiment/other classifiers sharing the
    same architecture), so only sentence-transformers-library candidates from it are kept."""

    async def fake_tag_fetch(session: Mock, tag: str, sort: str, limit: int) -> list[dict[str, str]]:
        if tag != "text-classification":
            return []
        return [
            {"id": "BAAI/bge-reranker-v2-m3", "library_name": "sentence-transformers"},
            {"id": "ProsusAI/finbert", "library_name": "transformers"},
        ]

    with (
        patch("scripts.get_huggingface_models.fetch_popular_models", side_effect=fake_tag_fetch),
        patch("scripts.get_huggingface_models.fetch_popular_reranker_models", return_value=[]),
    ):
        result = await collect_reranker_models(MagicMock(), {"downloads": 10})

    assert [m["id"] for m in result] == ["BAAI/bge-reranker-v2-m3"]


@pytest.mark.asyncio
async def test_collect_embedding_models_deduplicates() -> None:
    embedding = {"id": "org/bge-embedding-v2"}

    async def fake_fetch(session: Mock, sort: str, limit: int):
        return [embedding]

    with (
        patch("scripts.get_huggingface_models.fetch_popular_models", side_effect=_no_tag_candidates),
        patch("scripts.get_huggingface_models.fetch_popular_embedding_models", side_effect=fake_fetch),
    ):
        result = await collect_embedding_models(MagicMock(), {"downloads": 10, "trendingScore": 10})

    assert len(result) == 1


@pytest.mark.asyncio
async def test_collect_embedding_models_filters_non_embeddings() -> None:
    async def fake_fetch(session: Mock, sort: str, limit: int):
        return [{"id": "org/bert-base"}]

    with (
        patch("scripts.get_huggingface_models.fetch_popular_models", side_effect=_no_tag_candidates),
        patch("scripts.get_huggingface_models.fetch_popular_embedding_models", side_effect=fake_fetch),
    ):
        result = await collect_embedding_models(MagicMock(), {"downloads": 10})

    assert result == []


@pytest.mark.asyncio
async def test_collect_embedding_models_finds_by_trusted_tag_without_name_match() -> None:
    """A model with no "embed" in its name (invisible to the name search) is still found via the
    trusted sentence-similarity pipeline_tag."""

    async def fake_tag_fetch(session: Mock, tag: str, sort: str, limit: int):
        return [{"id": "BAAI/bge-m3"}] if tag == "sentence-similarity" else []

    with (
        patch("scripts.get_huggingface_models.fetch_popular_models", side_effect=fake_tag_fetch),
        patch("scripts.get_huggingface_models.fetch_popular_embedding_models", return_value=[]),
    ):
        result = await collect_embedding_models(MagicMock(), {"downloads": 10})

    assert [m["id"] for m in result] == ["BAAI/bge-m3"]


@pytest.mark.asyncio
async def test_collect_embedding_models_feature_extraction_requires_name_match() -> None:
    """feature-extraction is a noisy tag (audio/code encoders share it), so candidates from it
    still need to pass the embedding name heuristic."""

    async def fake_tag_fetch(session: Mock, tag: str, sort: str, limit: int) -> list[dict[str, str]]:
        if tag != "feature-extraction":
            return []
        return [{"id": "intfloat/multilingual-e5-large"}, {"id": "facebook/w2v-bert-2.0"}]

    with (
        patch("scripts.get_huggingface_models.fetch_popular_models", side_effect=fake_tag_fetch),
        patch("scripts.get_huggingface_models.fetch_popular_embedding_models", return_value=[]),
    ):
        result = await collect_embedding_models(MagicMock(), {"downloads": 10})

    assert [m["id"] for m in result] == ["intfloat/multilingual-e5-large"]


@pytest.mark.parametrize(
    ("input_dict", "expected"),
    [
        ({"id": "cross-encoder/ms-marco-MiniLM-L6-v2"}, True),
        ({"id": "BAAI/bge-reranker-v2-m3"}, True),
        ({"id": "cross-encoder/stsb-roberta-large"}, False),
        ({"id": "cross-encoder/stsb-distilroberta-base"}, False),
        ({"id": "cross-encoder/qnli-electra-base"}, False),
        ({"id": "cross-encoder/quora-roberta-base"}, False),
        ({"id": ""}, True),
        ({}, True),
    ],
    ids=[
        "reranker_kept",
        "reranker_kept_2",
        "stsb_excluded",
        "sts_excluded",
        "qnli_excluded",
        "quora_excluded",
        "empty_id",
        "missing_id",
    ],
)
def test_is_reranking_cross_encoder(input_dict: dict[str, str], expected: bool) -> None:
    assert is_reranking_cross_encoder(input_dict) is expected


def test_reranker_tags_trust_text_ranking_only_for_reranking_cross_encoders() -> None:
    assert RERANKER_TAGS["text-ranking"]({"id": "cross-encoder/ms-marco-MiniLM-L6-v2", "library_name": "transformers"}) is True
    assert RERANKER_TAGS["text-ranking"]({"id": "cross-encoder/stsb-roberta-large", "library_name": "sentence-transformers"}) is False


def test_reranker_tags_gate_text_classification_on_sentence_transformers_library() -> None:
    assert RERANKER_TAGS["text-classification"]({"id": "BAAI/bge-reranker-v2-m3", "library_name": "sentence-transformers"}) is True
    assert RERANKER_TAGS["text-classification"]({"id": "BAAI/bge-reranker-v2-m3", "library_name": "transformers"}) is False
    assert RERANKER_TAGS["text-classification"]({"id": "BAAI/bge-reranker-v2-m3"}) is False
    assert RERANKER_TAGS["text-classification"]({"id": "cross-encoder/qnli-electra-base", "library_name": "sentence-transformers"}) is False


def test_embedding_tags_trust_sentence_similarity_unconditionally() -> None:
    assert EMBEDDING_TAGS["sentence-similarity"]({"id": "anything"}) is True


def test_embedding_tags_gate_feature_extraction_on_name_heuristic() -> None:
    assert EMBEDDING_TAGS["feature-extraction"]({"id": "BAAI/bge-large-en-v1.5"}) is True
    assert EMBEDDING_TAGS["feature-extraction"]({"id": "facebook/w2v-bert-2.0"}) is False


@pytest.mark.asyncio
async def test_fetch_model_details_returns_size() -> None:
    data = {"siblings": [{"size": 1024**3}, {"size": 1024**3}]}
    session = _make_mock_session(data)
    sem = asyncio.Semaphore(1)

    model_id, size, architectures, chat_ok = await fetch_model_details(session, "org/model", sem)

    assert model_id == "org/model"
    assert size == "2.0 GB"
    assert architectures == []
    assert chat_ok is True


@pytest.mark.asyncio
async def test_fetch_model_details_returns_na_when_no_siblings() -> None:
    session = _make_mock_session({"siblings": []})
    sem = asyncio.Semaphore(1)

    _, size, _, _ = await fetch_model_details(session, "org/model", sem)

    assert size == "N/A"


@pytest.mark.asyncio
async def test_fetch_model_details_returns_na_on_exception() -> None:
    mock_resp = AsyncMock()
    mock_resp.raise_for_status = MagicMock(side_effect=aiohttp.ClientResponseError(MagicMock(), (), status=404))
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)
    session = MagicMock()
    session.get = MagicMock(return_value=mock_resp)
    sem = asyncio.Semaphore(1)

    model_id, size, architectures, chat_ok = await fetch_model_details(session, "org/missing", sem)

    assert model_id == "org/missing"
    assert size == "N/A"
    assert architectures is None
    assert chat_ok is False


@pytest.mark.asyncio
async def test_fetch_model_details_skips_siblings_without_size() -> None:
    data = {"siblings": [{"name": "config.json"}, {"size": 1024}]}
    session = _make_mock_session(data)
    sem = asyncio.Semaphore(1)

    _, size, _, _ = await fetch_model_details(session, "org/model", sem)

    assert size == "1.0 KB"


@pytest.mark.asyncio
async def test_fetch_model_details_returns_architectures() -> None:
    data = {"siblings": [], "config": {"architectures": ["XLMRobertaForSequenceClassification"]}}
    session = _make_mock_session(data)
    sem = asyncio.Semaphore(1)

    _, _, architectures, _ = await fetch_model_details(session, "org/model", sem)

    assert architectures == ["XLMRobertaForSequenceClassification"]


@pytest.mark.asyncio
async def test_fetch_model_details_skips_chat_template_check_by_default() -> None:
    data = {"siblings": []}
    session = _make_mock_session(data)
    sem = asyncio.Semaphore(1)

    _, _, _, chat_ok = await fetch_model_details(session, "org/model", sem)

    assert chat_ok is True


@pytest.mark.asyncio
async def test_fetch_model_details_checks_chat_template_when_requested() -> None:
    data = {"siblings": [{"rfilename": "config.json"}]}
    session = _make_mock_session(data)
    sem = asyncio.Semaphore(1)

    _, _, _, chat_ok = await fetch_model_details(session, "org/base-model", sem, check_chat_template=True)

    assert chat_ok is False


@pytest.mark.asyncio
async def test_fetch_model_details_chat_template_present_when_requested() -> None:
    data = {"siblings": [{"rfilename": "chat_template.jinja"}]}
    session = _make_mock_session(data)
    sem = asyncio.Semaphore(1)

    _, _, _, chat_ok = await fetch_model_details(session, "org/instruct-model", sem, check_chat_template=True)

    assert chat_ok is True


@pytest.mark.asyncio
async def test_main_no_active_sort_prints_error(capsys: pytest.CaptureFixture[str]) -> None:
    await main(0, 0, 0, raw=False, model_type="llm")

    captured = capsys.readouterr()
    assert "Specify at least one" in captured.err


@pytest.mark.asyncio
async def test_main_llm_output_json(capsys: pytest.CaptureFixture[str]) -> None:
    models = [{"id": "org/llama"}]
    sizes = {"org/llama": "4.0 GB"}

    async def fake_collect(session: Mock, _active: dict[str, int]) -> list[dict[str, str]]:
        return models

    async def fake_details(
        session: Mock, model_id: str, _sem: asyncio.Semaphore, check_chat_template: bool = False, cache_dir: Path | None = None
    ) -> tuple[str, str, list[str], bool]:
        return model_id, sizes.get(model_id, "N/A"), [], True

    with (
        patch("scripts.get_huggingface_models.collect_llm_models", side_effect=fake_collect),
        patch("scripts.get_huggingface_models.fetch_model_details", side_effect=fake_details),
        patch(
            "aiohttp.ClientSession",
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=MagicMock()), __aexit__=AsyncMock(return_value=False)),
        ),
    ):
        await main(10, 0, 0, raw=True, model_type="llm")

    out = capsys.readouterr().out
    data = json.loads(out)
    assert "llms" in data
    assert data["llms"][0]["name"] == "org/llama"
    assert data["llms"][0]["size"] == "4GB"


@pytest.mark.asyncio
async def test_main_reranker_output_json(capsys: pytest.CaptureFixture[str]) -> None:
    models = [{"id": "org/bge-reranker"}]
    sizes = {"org/bge-reranker": "1.0 GB"}
    architectures = {"org/bge-reranker": ["XLMRobertaForSequenceClassification"]}

    async def fake_collect(session: Mock, active: dict[str, str]):
        return models

    async def fake_details(
        session: Mock, model_id: str, sem: asyncio.Semaphore, check_chat_template: bool = False, cache_dir: Path | None = None
    ) -> tuple[str, str, list[str], bool]:
        return model_id, sizes.get(model_id, "N/A"), architectures.get(model_id, []), True

    with (
        patch("scripts.get_huggingface_models.collect_reranker_models", side_effect=fake_collect),
        patch("scripts.get_huggingface_models.fetch_model_details", side_effect=fake_details),
        patch(
            "aiohttp.ClientSession",
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=MagicMock()), __aexit__=AsyncMock(return_value=False)),
        ),
    ):
        await main(10, 0, 0, raw=True, model_type="reranker")

    out = capsys.readouterr().out
    data = json.loads(out)
    assert "rerankers" in data
    assert data["rerankers"][0]["name"] == "org/bge-reranker"


@pytest.mark.asyncio
async def test_main_reranker_drops_unsupported_architecture(capsys: pytest.CaptureFixture[str]) -> None:
    models = [{"id": "org/bge-reranker"}, {"id": "org/generative-reranker"}]
    sizes = {"org/bge-reranker": "1.0 GB", "org/generative-reranker": "4.0 GB"}
    architectures = {
        "org/bge-reranker": ["XLMRobertaForSequenceClassification"],
        "org/generative-reranker": ["Qwen3ForCausalLM"],
    }

    async def fake_collect(session: Mock, active: dict[str, str]):
        return models

    async def fake_details(
        session: Mock, model_id: str, sem: asyncio.Semaphore, check_chat_template: bool = False, cache_dir: Path | None = None
    ) -> tuple[str, str, list[str], bool]:
        return model_id, sizes.get(model_id, "N/A"), architectures.get(model_id, []), True

    with (
        patch("scripts.get_huggingface_models.collect_reranker_models", side_effect=fake_collect),
        patch("scripts.get_huggingface_models.fetch_model_details", side_effect=fake_details),
        patch(
            "aiohttp.ClientSession",
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=MagicMock()), __aexit__=AsyncMock(return_value=False)),
        ),
    ):
        await main(10, 0, 0, raw=False, model_type="reranker")

    out = capsys.readouterr()
    data = json.loads(out.out)
    names = [e["name"] for e in data["rerankers"]]
    assert names == ["org/bge-reranker"]
    assert "Dropped 1 reranker candidate" in out.err


@pytest.mark.asyncio
async def test_main_reranker_keeps_generative_when_allowed(capsys: pytest.CaptureFixture[str]) -> None:
    models = [{"id": "org/bge-reranker"}, {"id": "org/generative-reranker"}]
    sizes = {"org/bge-reranker": "1.0 GB", "org/generative-reranker": "4.0 GB"}
    architectures = {
        "org/bge-reranker": ["XLMRobertaForSequenceClassification"],
        "org/generative-reranker": ["Qwen3ForCausalLM"],
    }

    async def fake_collect(session: Mock, active: dict[str, str]):
        return models

    async def fake_details(
        session: Mock, model_id: str, sem: asyncio.Semaphore, check_chat_template: bool = False, cache_dir: Path | None = None
    ) -> tuple[str, str, list[str], bool]:
        return model_id, sizes.get(model_id, "N/A"), architectures.get(model_id, []), True

    with (
        patch("scripts.get_huggingface_models.collect_reranker_models", side_effect=fake_collect),
        patch("scripts.get_huggingface_models.fetch_model_details", side_effect=fake_details),
        patch(
            "aiohttp.ClientSession",
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=MagicMock()), __aexit__=AsyncMock(return_value=False)),
        ),
    ):
        await main(10, 0, 0, raw=False, model_type="reranker", allow_generative_rerankers=True)

    out = capsys.readouterr()
    data = json.loads(out.out)
    entries = {e["name"]: e["is_generative"] for e in data["rerankers"]}
    assert entries == {"org/bge-reranker": False, "org/generative-reranker": True}
    assert "Dropped" not in out.err


@pytest.mark.asyncio
async def test_main_reranker_drops_failed_detail_fetch_even_when_generative_allowed(capsys: pytest.CaptureFixture[str]) -> None:
    """A candidate whose detail fetch failed outright (e.g. exhausted retries under rate limiting)
    must be dropped, not kept with fabricated `size: "N/A"` / `is_generative: true` — even under
    `allow_generative_rerankers`, which otherwise skips the architecture-based drop entirely."""
    models = [{"id": "org/bge-reranker"}, {"id": "org/unreachable-reranker"}]
    sizes = {"org/bge-reranker": "1.0 GB"}
    architectures: dict[str, list[str] | None] = {
        "org/bge-reranker": ["XLMRobertaForSequenceClassification"],
        "org/unreachable-reranker": None,
    }

    async def fake_collect(session: Mock, active: dict[str, str]) -> list[dict[str, str]]:
        return models

    async def fake_details(
        session: Mock, model_id: str, sem: asyncio.Semaphore, check_chat_template: bool = False, cache_dir: Path | None = None
    ) -> tuple[str, str, list[str] | None, bool]:
        return model_id, sizes.get(model_id, "N/A"), architectures.get(model_id), model_id in sizes

    with (
        patch("scripts.get_huggingface_models.collect_reranker_models", side_effect=fake_collect),
        patch("scripts.get_huggingface_models.fetch_model_details", side_effect=fake_details),
        patch(
            "aiohttp.ClientSession",
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=MagicMock()), __aexit__=AsyncMock(return_value=False)),
        ),
    ):
        await main(10, 0, 0, raw=False, model_type="reranker", allow_generative_rerankers=True)

    out = capsys.readouterr()
    data = json.loads(out.out)
    names = [e["name"] for e in data["rerankers"]]
    assert names == ["org/bge-reranker"]
    assert "Dropped 1 reranker candidate(s) whose details couldn't be fetched" in out.err


@pytest.mark.asyncio
async def test_main_llm_drops_models_without_chat_template(capsys: pytest.CaptureFixture[str]) -> None:
    models = [{"id": "org/llama-instruct"}, {"id": "org/llama-base"}]
    sizes = {"org/llama-instruct": "4.0 GB", "org/llama-base": "4.0 GB"}
    chat_capable = {"org/llama-instruct": True, "org/llama-base": False}

    async def fake_collect(session: Mock, active: dict[str, int]) -> list[dict[str, str]]:
        return models

    async def fake_details(
        session: Mock, model_id: str, sem: asyncio.Semaphore, check_chat_template: bool = False, cache_dir: Path | None = None
    ) -> tuple[str, str, list[str], bool]:
        assert check_chat_template is True
        return model_id, sizes.get(model_id, "N/A"), [], chat_capable.get(model_id, False)

    with (
        patch("scripts.get_huggingface_models.collect_llm_models", side_effect=fake_collect),
        patch("scripts.get_huggingface_models.fetch_model_details", side_effect=fake_details),
        patch(
            "aiohttp.ClientSession",
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=MagicMock()), __aexit__=AsyncMock(return_value=False)),
        ),
    ):
        await main(10, 0, 0, raw=False, model_type="llm")

    out = capsys.readouterr()
    data = json.loads(out.out)
    names = [e["name"] for e in data["llms"]]
    assert names == ["org/llama-instruct"]
    assert "Dropped 1 LLM candidate" in out.err


@pytest.mark.asyncio
async def test_main_embedding_output_json(capsys: pytest.CaptureFixture[str]) -> None:
    models = [{"id": "org/bge-embedding"}]
    sizes = {"org/bge-embedding": "1.0 GB"}

    async def fake_collect(session: Mock, active: dict[str, str]):
        return models

    async def fake_details(
        session: Mock, model_id: str, sem: asyncio.Semaphore, check_chat_template: bool = False, cache_dir: Path | None = None
    ) -> tuple[str, str, list[str], bool]:
        return model_id, sizes.get(model_id, "N/A"), [], True

    with (
        patch("scripts.get_huggingface_models.collect_embedding_models", side_effect=fake_collect),
        patch("scripts.get_huggingface_models.fetch_model_details", side_effect=fake_details),
        patch(
            "aiohttp.ClientSession",
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=MagicMock()), __aexit__=AsyncMock(return_value=False)),
        ),
    ):
        await main(10, 0, 0, raw=True, model_type="embedding")

    out = capsys.readouterr().out
    data = json.loads(out)
    assert "embeddings" in data
    assert data["embeddings"][0]["name"] == "org/bge-embedding"


@pytest.mark.asyncio
async def test_main_raw_false_logs_to_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    async def fake_collect(session: Mock, active: dict[str, str]) -> list[dict[str, str]]:
        return []

    async def fake_details(
        session: Mock, model_id: str, sem: asyncio.Semaphore, check_chat_template: bool = False, cache_dir: Path | None = None
    ) -> tuple[str, str, list[str], bool]:
        return model_id, "N/A", [], True

    with (
        patch("scripts.get_huggingface_models.collect_llm_models", side_effect=fake_collect),
        patch("scripts.get_huggingface_models.fetch_model_details", side_effect=fake_details),
        patch(
            "aiohttp.ClientSession",
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=MagicMock()), __aexit__=AsyncMock(return_value=False)),
        ),
    ):
        await main(10, 0, 0, raw=False, model_type="llm")

    err = capsys.readouterr().err
    assert "Fetching" in err


@pytest.mark.asyncio
async def test_main_output_writes_file_and_preserves_other_key(tmp_path: Path) -> None:
    output_path = tmp_path / "vllm-min.json"
    output_path.write_text(json.dumps({"rerankers": [{"name": "org/existing-reranker", "size": "1GB"}]}))

    models = [{"id": "org/llama"}]
    sizes = {"org/llama": "4.0 GB"}

    async def fake_collect(session: Mock, _active: dict[str, int]) -> list[dict[str, str]]:
        return models

    async def fake_details(
        session: Mock, model_id: str, _sem: asyncio.Semaphore, check_chat_template: bool = False, cache_dir: Path | None = None
    ) -> tuple[str, str, list[str], bool]:
        return model_id, sizes.get(model_id, "N/A"), [], True

    with (
        patch("scripts.get_huggingface_models.collect_llm_models", side_effect=fake_collect),
        patch("scripts.get_huggingface_models.fetch_model_details", side_effect=fake_details),
        patch(
            "aiohttp.ClientSession",
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=MagicMock()), __aexit__=AsyncMock(return_value=False)),
        ),
    ):
        await main(10, 0, 0, raw=True, model_type="llm", output=str(output_path))

    data = json.loads(output_path.read_text())
    assert data["llms"] == [{"name": "org/llama", "size": "4GB", "supports_vision": None, "supports_audio": None}]
    assert data["rerankers"] == [{"name": "org/existing-reranker", "size": "1GB"}]


@pytest.mark.asyncio
async def test_main_output_overwrites_same_key(tmp_path: Path) -> None:
    output_path = tmp_path / "vllm-min.json"
    output_path.write_text(json.dumps({"llms": [{"name": "org/old-model", "size": "1GB"}]}))

    async def fake_collect(session: Mock, _active: dict[str, int]) -> list[dict[str, str]]:
        return [{"id": "org/new-model"}]

    async def fake_details(
        session: Mock, model_id: str, _sem: asyncio.Semaphore, check_chat_template: bool = False, cache_dir: Path | None = None
    ) -> tuple[str, str, list[str], bool]:
        return model_id, "2.0 GB", [], True

    with (
        patch("scripts.get_huggingface_models.collect_llm_models", side_effect=fake_collect),
        patch("scripts.get_huggingface_models.fetch_model_details", side_effect=fake_details),
        patch(
            "aiohttp.ClientSession",
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=MagicMock()), __aexit__=AsyncMock(return_value=False)),
        ),
    ):
        await main(10, 0, 0, raw=True, model_type="llm", output=str(output_path))

    data = json.loads(output_path.read_text())
    assert data["llms"] == [{"name": "org/new-model", "size": "2GB", "supports_vision": None, "supports_audio": None}]


@pytest.mark.asyncio
async def test_main_active_dict_built_correctly() -> None:
    captured_active: dict[str, str] = {}

    async def fake_collect(session: Mock, active: dict[str, str]) -> list[dict[str, str]]:
        captured_active.update(active)
        return []

    async def fake_details(
        session: Mock, model_id: str, sem: asyncio.Semaphore, check_chat_template: bool = False, cache_dir: Path | None = None
    ) -> tuple[str, str, list[str], bool]:
        return model_id, "N/A", [], True

    with (
        patch("scripts.get_huggingface_models.collect_llm_models", side_effect=fake_collect),
        patch("scripts.get_huggingface_models.fetch_model_details", side_effect=fake_details),
        patch(
            "aiohttp.ClientSession",
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=MagicMock()), __aexit__=AsyncMock(return_value=False)),
        ),
    ):
        await main(5, 0, 3, raw=True, model_type="llm")

    assert "downloads" in captured_active
    assert captured_active["downloads"] == 5
    assert "likes" not in captured_active
    assert "trendingScore" in captured_active
    assert captured_active["trendingScore"] == 3


@pytest.mark.asyncio
async def test_main_passes_default_cache_dir_to_fetch_model_details() -> None:
    seen: list[Path | None] = []

    async def fake_collect(session: Mock, _active: dict[str, int]) -> list[dict[str, str]]:
        return [{"id": "org/llama"}]

    async def fake_details(
        session: Mock, model_id: str, _sem: asyncio.Semaphore, check_chat_template: bool = False, cache_dir: Path | None = None
    ) -> tuple[str, str, list[str], bool]:
        seen.append(cache_dir)
        return model_id, "N/A", [], True

    with (
        patch("scripts.get_huggingface_models.collect_llm_models", side_effect=fake_collect),
        patch("scripts.get_huggingface_models.fetch_model_details", side_effect=fake_details),
        patch(
            "aiohttp.ClientSession",
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=MagicMock()), __aexit__=AsyncMock(return_value=False)),
        ),
    ):
        await main(10, 0, 0, raw=True, model_type="llm")

    assert seen == [DEFAULT_MODEL_DETAIL_CACHE_DIR]


@pytest.mark.asyncio
async def test_fetch_registry_entries_raises_without_any_active_criterion() -> None:
    session = MagicMock()
    with pytest.raises(ValueError, match="Specify at least one"):
        await fetch_registry_entries(session, 0, 0, 0, "llm", log=lambda _msg: None)


@pytest.mark.asyncio
async def test_collect_embedding_models_keeps_gguf_tagged_repo_with_native_weights() -> None:
    multi_format = {"id": "mixedbread-ai/mxbai-embed-large-v1", "tags": ["gguf"]}
    gguf_only = {"id": "org/bge-small-GGUF", "tags": ["gguf"]}
    details = {
        "mixedbread-ai/mxbai-embed-large-v1": {**multi_format, "siblings": [{"rfilename": "model.safetensors"}]},
        "org/bge-small-GGUF": {**gguf_only, "siblings": [{"rfilename": "bge-small.Q4_K_M.gguf"}]},
    }

    async def fake_name_fetch(session: Mock, sort: str, limit: int):
        return [multi_format, gguf_only]

    async def fake_detail(session: Mock, model_id: str, **kwargs: object):
        return details[model_id]

    with (
        patch("scripts.get_huggingface_models.fetch_popular_models", side_effect=_no_tag_candidates),
        patch("scripts.get_huggingface_models.fetch_popular_embedding_models", side_effect=fake_name_fetch),
        patch("scripts.get_huggingface_models.get_model_detail_cached", side_effect=fake_detail),
    ):
        result = await collect_embedding_models(MagicMock(), {"downloads": 10})

    assert [m["id"] for m in result] == ["mixedbread-ai/mxbai-embed-large-v1"]


@pytest.mark.asyncio
async def test_collect_llm_models_keeps_gguf_tagged_repo_with_native_weights() -> None:
    multi_format = {"id": "org/llama", "tags": ["gguf"]}
    detail = {**multi_format, "siblings": [{"rfilename": "model-00001-of-00002.safetensors"}]}

    async def fake_fetch(session: Mock, tag: str, sort: str, limit: int):
        return [multi_format]

    with (
        patch("scripts.get_huggingface_models.fetch_popular_models", side_effect=fake_fetch),
        patch("scripts.get_huggingface_models.get_model_detail_cached", return_value=detail) as detail_mock,
    ):
        result = await collect_llm_models(MagicMock(), {"downloads": 10, "likes": 10})

    assert [m["id"] for m in result] == ["org/llama"]
    assert detail_mock.await_count == 1  # fetched once per unique candidate, not per sort/tag combination


@pytest.mark.asyncio
async def test_collect_llm_models_drops_gguf_named_repo_without_fetching_details() -> None:
    async def fake_fetch(session: Mock, tag: str, sort: str, limit: int):
        return [{"id": "org/llama-GGUF", "tags": ["gguf"]}, {"id": "org/plain"}]

    with (
        patch("scripts.get_huggingface_models.fetch_popular_models", side_effect=fake_fetch),
        patch("scripts.get_huggingface_models.get_model_detail_cached") as detail_mock,
    ):
        result = await collect_llm_models(MagicMock(), {"downloads": 10})

    assert [m["id"] for m in result] == ["org/plain"]
    detail_mock.assert_not_called()


@pytest.mark.asyncio
async def test_collect_llm_models_drops_gguf_candidate_when_detail_fetch_fails() -> None:
    async def fake_fetch(session: Mock, tag: str, sort: str, limit: int):
        return [{"id": "org/llama", "tags": ["gguf"]}, {"id": "org/plain"}]

    with (
        patch("scripts.get_huggingface_models.fetch_popular_models", side_effect=fake_fetch),
        patch("scripts.get_huggingface_models.get_model_detail_cached", side_effect=aiohttp.ClientError("boom")),
    ):
        result = await collect_llm_models(MagicMock(), {"downloads": 10})

    assert [m["id"] for m in result] == ["org/plain"]


def test_is_supported_stt_accepts_whisper_and_rejects_ctc() -> None:
    assert is_supported_stt(["WhisperForConditionalGeneration"])
    assert not is_supported_stt(["Wav2Vec2ForCTC"])
    assert not is_supported_stt([])


@pytest.mark.asyncio
async def test_collect_stt_models_deduplicates_and_skips_gguf_and_mlx() -> None:
    async def fake_fetch(session: Mock, tag: str, sort: str, limit: int, extra_filter: str | None = None) -> list[dict[str, Any]]:
        assert tag == "automatic-speech-recognition"
        return [{"id": "openai/whisper-small"}, {"id": "mlx-community/whisper-tiny"}, {"id": "org/whisper-gguf", "tags": ["gguf"]}]

    with patch("scripts.get_huggingface_models.fetch_popular_models", side_effect=fake_fetch):
        result = await collect_stt_models(MagicMock(), {"downloads": 10, "trendingScore": 10})

    assert [m["id"] for m in result] == ["openai/whisper-small"]


@pytest.mark.asyncio
async def test_collect_stt_models_logs_prefiltered_count() -> None:
    async def fake_fetch(session: Mock, tag: str, sort: str, limit: int, extra_filter: str | None = None) -> list[dict[str, Any]]:
        return [{"id": "openai/whisper-small"}, {"id": "mlx-community/whisper-tiny"}, {"id": "onnx-community/whisper-tiny"}]

    log = MagicMock()
    with patch("scripts.get_huggingface_models.fetch_popular_models", side_effect=fake_fetch):
        await collect_stt_models(MagicMock(), {"downloads": 10, "trendingScore": 10}, log)

    assert log.call_args_list == [call("Dropped 2 STT candidate(s) as gguf/mlx/ONNX/CoreML/aligner/test repos before ranking.")]


def test_is_non_servable_stt_repo_flags_onnx_and_aligners() -> None:
    assert is_non_servable_stt_repo({"id": "onnx-community/whisper-tiny"})
    assert is_non_servable_stt_repo({"id": "Xenova/whisper-tiny"})
    assert is_non_servable_stt_repo({"id": "argmaxinc/whisperkit-coreml", "tags": ["coreml"]})
    assert not is_non_servable_stt_repo({"id": "distil-whisper/distil-large-v3", "tags": ["onnx", "transformers.js", "safetensors"]})
    assert is_non_servable_stt_repo({"id": "Qwen/Qwen3-ForcedAligner-0.6B"})
    assert not is_non_servable_stt_repo({"id": "openai/whisper-small", "tags": ["transformers"]})


@pytest.mark.asyncio
async def test_collect_stt_models_ranks_families_globally_and_cuts_to_limit() -> None:
    async def fake_fetch(session: Mock, tag: str, sort: str, limit: int, extra_filter: str | None = None) -> list[dict[str, Any]]:
        if extra_filter == "whisper":
            return [{"id": "a/whisper-1", "downloads": 100}, {"id": "a/whisper-2", "downloads": 10}]
        if extra_filter == "qwen3_asr":
            return [{"id": "a/qwen-asr", "downloads": 50}, {"id": "a/whisper-1", "downloads": 100}]
        return []

    with patch("scripts.get_huggingface_models.fetch_popular_models", side_effect=fake_fetch):
        result = await collect_stt_models(MagicMock(), {"downloads": 2})

    assert [m["id"] for m in result] == ["a/whisper-1", "a/qwen-asr"]


@pytest.mark.asyncio
async def test_collect_stt_models_keeps_gguf_tagged_repo_with_native_weights() -> None:
    multi_format = {"id": "audarai/Audar-ASR-V1-Turbo", "tags": ["gguf"], "downloads": 100, "trendingScore": 100}
    detail = {**multi_format, "siblings": [{"rfilename": "model.safetensors"}, {"rfilename": "gguf/model-Q4_K_M.gguf"}]}

    async def fake_fetch(session: Mock, tag: str, sort: str, limit: int, extra_filter: str | None = None) -> list[dict[str, Any]]:
        return [multi_format, {"id": "openai/whisper-small", "downloads": 10, "trendingScore": 10}]

    with (
        patch("scripts.get_huggingface_models.fetch_popular_models", side_effect=fake_fetch),
        patch("scripts.get_huggingface_models.get_model_detail_cached", return_value=detail) as detail_mock,
    ):
        result = await collect_stt_models(MagicMock(), {"downloads": 1, "trendingScore": 1})

    assert [m["id"] for m in result] == ["audarai/Audar-ASR-V1-Turbo"]
    assert detail_mock.await_count == 1  # fetched once per unique candidate, not per sort key/family


@pytest.mark.asyncio
async def test_collect_stt_models_drops_gguf_only_tagged_repo() -> None:
    gguf_only = {"id": "org/whisper-large", "tags": ["gguf"]}
    detail = {**gguf_only, "siblings": [{"rfilename": "whisper-large.Q4_K_M.gguf"}]}

    async def fake_fetch(session: Mock, tag: str, sort: str, limit: int, extra_filter: str | None = None) -> list[dict[str, Any]]:
        return [gguf_only, {"id": "openai/whisper-small"}]

    log = MagicMock()
    with (
        patch("scripts.get_huggingface_models.fetch_popular_models", side_effect=fake_fetch),
        patch("scripts.get_huggingface_models.get_model_detail_cached", return_value=detail),
    ):
        result = await collect_stt_models(MagicMock(), {"downloads": 10}, log)

    assert [m["id"] for m in result] == ["openai/whisper-small"]
    assert log.call_args_list == [call("Dropped 1 STT candidate(s) as gguf/mlx/ONNX/CoreML/aligner/test repos before ranking.")]


@pytest.mark.asyncio
async def test_collect_stt_models_drops_gguf_only_repo_before_cutting_to_limit() -> None:
    gguf_only = {"id": "org/whisper-large", "tags": ["gguf"], "downloads": 100}
    detail = {**gguf_only, "siblings": [{"rfilename": "whisper-large.Q4_K_M.gguf"}]}

    async def fake_fetch(session: Mock, tag: str, sort: str, limit: int, extra_filter: str | None = None) -> list[dict[str, Any]]:
        return [gguf_only, {"id": "openai/whisper-small", "downloads": 10}]

    with (
        patch("scripts.get_huggingface_models.fetch_popular_models", side_effect=fake_fetch),
        patch("scripts.get_huggingface_models.get_model_detail_cached", return_value=detail),
    ):
        result = await collect_stt_models(MagicMock(), {"downloads": 1})

    assert [m["id"] for m in result] == ["openai/whisper-small"]


@pytest.mark.asyncio
async def test_collect_stt_models_drops_gguf_tagged_repo_when_detail_fetch_fails() -> None:
    async def fake_fetch(session: Mock, tag: str, sort: str, limit: int, extra_filter: str | None = None) -> list[dict[str, Any]]:
        return [{"id": "org/whisper-large", "tags": ["gguf"]}, {"id": "openai/whisper-small"}]

    with (
        patch("scripts.get_huggingface_models.fetch_popular_models", side_effect=fake_fetch),
        patch("scripts.get_huggingface_models.get_model_detail_cached", side_effect=aiohttp.ClientError("boom")),
    ):
        result = await collect_stt_models(MagicMock(), {"downloads": 10})

    assert [m["id"] for m in result] == ["openai/whisper-small"]


@pytest.mark.asyncio
async def test_collect_stt_models_skips_detail_fetch_for_prefiltered_repos() -> None:
    async def fake_fetch(session: Mock, tag: str, sort: str, limit: int, extra_filter: str | None = None) -> list[dict[str, Any]]:
        return [{"id": "mlx-community/whisper-tiny", "tags": ["gguf"]}, {"id": "onnx-community/whisper-tiny", "tags": ["gguf"]}]

    with (
        patch("scripts.get_huggingface_models.fetch_popular_models", side_effect=fake_fetch),
        patch("scripts.get_huggingface_models.get_model_detail_cached") as detail_mock,
    ):
        result = await collect_stt_models(MagicMock(), {"downloads": 10})

    assert result == []
    detail_mock.assert_not_called()


@pytest.mark.asyncio
async def test_main_stt_drops_unsupported_architectures(capsys: pytest.CaptureFixture[str]) -> None:
    models = [{"id": "openai/whisper-small"}, {"id": "facebook/wav2vec2-base"}, {"id": "org/unfetched"}]
    architectures: dict[str, list[str] | None] = {
        "openai/whisper-small": ["WhisperForConditionalGeneration"],
        "facebook/wav2vec2-base": ["Wav2Vec2ForCTC"],
        "org/unfetched": None,
    }

    async def fake_collect(session: Mock, active: dict[str, int], _log: Callable[[str], None]) -> list[dict[str, str]]:
        return models

    async def fake_details(
        session: Mock, model_id: str, sem: asyncio.Semaphore, check_chat_template: bool = False, cache_dir: Path | None = None
    ) -> tuple[str, str, list[str] | None, bool]:
        return model_id, "1.0 GB", architectures[model_id], True

    with (
        patch("scripts.get_huggingface_models.collect_stt_models", side_effect=fake_collect),
        patch("scripts.get_huggingface_models.fetch_model_details", side_effect=fake_details),
        patch(
            "aiohttp.ClientSession",
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=MagicMock()), __aexit__=AsyncMock(return_value=False)),
        ),
    ):
        await main(10, 0, 0, raw=False, model_type="stt")

    out = capsys.readouterr()
    data = json.loads(out.out)
    assert [e["name"] for e in data["stts"]] == ["openai/whisper-small"]
    assert "Dropped 2 STT candidate" in out.err
