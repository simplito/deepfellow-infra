# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for get_huggingface_models.py script."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import aiohttp
import pytest

from scripts.get_huggingface_models import (
    collect_llm_models,
    collect_reranker_models,
    fetch_model_size,
    fetch_popular_models,
    fetch_popular_reranker_models,
    fmt_size,
    fmt_size_compact,
    is_llm,
    is_reranker,
    main,
)


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
async def test_fetch_popular_models_raises_on_http_error() -> None:
    mock_resp = AsyncMock()
    mock_resp.raise_for_status = MagicMock(side_effect=aiohttp.ClientResponseError(MagicMock(), (), status=429))
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)
    session = MagicMock()
    session.get = MagicMock(return_value=mock_resp)

    with pytest.raises(aiohttp.ClientResponseError):
        await fetch_popular_models(session, "text-generation", "downloads", 10)


@pytest.mark.asyncio
async def test_fetch_popular_reranker_models_returns_list() -> None:
    data = [{"id": "org/reranker-a"}]
    session = _make_mock_session(data)

    result = await fetch_popular_reranker_models(session, "downloads", 10)

    assert result == data


@pytest.mark.asyncio
async def test_fetch_popular_reranker_models_passes_other_param() -> None:
    session = _make_mock_session([])

    await fetch_popular_reranker_models(session, "trending", 20)

    call_kwargs = session.get.call_args
    params = call_kwargs[1]["params"]
    assert params["other"] == "reranker"
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
    assert len(calls) == 4  # 2 sorts x 2 LLM_TAGS


@pytest.mark.asyncio
async def test_collect_reranker_models_deduplicates() -> None:
    reranker = {"id": "org/bge-reranker-v2"}

    async def fake_fetch(session: Mock, sort: str, limit: int):
        return [reranker]

    with patch(
        "scripts.get_huggingface_models.fetch_popular_reranker_models",
        side_effect=fake_fetch,
    ):
        result = await collect_reranker_models(MagicMock(), {"downloads": 10, "likes": 10})

    assert len(result) == 1


@pytest.mark.asyncio
async def test_collect_reranker_models_filters_non_rerankers() -> None:
    async def fake_fetch(session: Mock, sort: str, limit: int):
        return [{"id": "org/bert-base"}]

    with patch(
        "scripts.get_huggingface_models.fetch_popular_reranker_models",
        side_effect=fake_fetch,
    ):
        result = await collect_reranker_models(MagicMock(), {"downloads": 10})

    assert result == []


@pytest.mark.asyncio
async def test_fetch_model_size_returns_size() -> None:
    data = {"siblings": [{"size": 1024**3}, {"size": 1024**3}]}
    session = _make_mock_session(data)
    sem = asyncio.Semaphore(1)

    model_id, size = await fetch_model_size(session, "org/model", sem)

    assert model_id == "org/model"
    assert size == "2.0 GB"


@pytest.mark.asyncio
async def test_fetch_model_size_returns_na_when_no_siblings() -> None:
    session = _make_mock_session({"siblings": []})
    sem = asyncio.Semaphore(1)

    _, size = await fetch_model_size(session, "org/model", sem)

    assert size == "N/A"


@pytest.mark.asyncio
async def test_fetch_model_size_returns_na_on_exception() -> None:
    mock_resp = AsyncMock()
    mock_resp.raise_for_status = MagicMock(side_effect=aiohttp.ClientResponseError(MagicMock(), (), status=404))
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)
    session = MagicMock()
    session.get = MagicMock(return_value=mock_resp)
    sem = asyncio.Semaphore(1)

    model_id, size = await fetch_model_size(session, "org/missing", sem)

    assert model_id == "org/missing"
    assert size == "N/A"


@pytest.mark.asyncio
async def test_fetch_model_size_skips_siblings_without_size() -> None:
    data = {"siblings": [{"name": "config.json"}, {"size": 1024}]}
    session = _make_mock_session(data)
    sem = asyncio.Semaphore(1)

    _, size = await fetch_model_size(session, "org/model", sem)

    assert size == "1.0 KB"


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

    async def fake_size(session: Mock, model_id: str, _sem: asyncio.Semaphore) -> tuple[str, str]:
        return model_id, sizes.get(model_id, "N/A")

    with (
        patch("scripts.get_huggingface_models.collect_llm_models", side_effect=fake_collect),
        patch("scripts.get_huggingface_models.fetch_model_size", side_effect=fake_size),
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

    async def fake_collect(session: Mock, active: dict[str, str]):
        return models

    async def fake_size(session: Mock, model_id: str, sem: asyncio.Semaphore):
        return model_id, sizes.get(model_id, "N/A")

    with (
        patch("scripts.get_huggingface_models.collect_reranker_models", side_effect=fake_collect),
        patch("scripts.get_huggingface_models.fetch_model_size", side_effect=fake_size),
        patch(
            "aiohttp.ClientSession",
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=MagicMock()), __aexit__=AsyncMock(return_value=False)),
        ),
    ):
        await main(10, 0, 0, raw=True, model_type="reranker")

    out = capsys.readouterr().out
    data = json.loads(out)
    assert "rerankers" in data


@pytest.mark.asyncio
async def test_main_raw_false_logs_to_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    async def fake_collect(session: Mock, active: dict[str, str]) -> list[dict[str, str]]:
        return []

    async def fake_size(session: Mock, model_id: str, sem: asyncio.Semaphore):
        return model_id, "N/A"

    with (
        patch("scripts.get_huggingface_models.collect_llm_models", side_effect=fake_collect),
        patch("scripts.get_huggingface_models.fetch_model_size", side_effect=fake_size),
        patch(
            "aiohttp.ClientSession",
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=MagicMock()), __aexit__=AsyncMock(return_value=False)),
        ),
    ):
        await main(10, 0, 0, raw=False, model_type="llm")

    err = capsys.readouterr().err
    assert "Fetching" in err


@pytest.mark.asyncio
async def test_main_active_dict_built_correctly() -> None:
    captured_active: dict[str, str] = {}

    async def fake_collect(session: Mock, active: dict[str, str]) -> list[dict[str, str]]:
        captured_active.update(active)
        return []

    async def fake_size(session: Mock, model_id: str, sem: asyncio.Semaphore):
        return model_id, "N/A"

    with (
        patch("scripts.get_huggingface_models.collect_llm_models", side_effect=fake_collect),
        patch("scripts.get_huggingface_models.fetch_model_size", side_effect=fake_size),
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
