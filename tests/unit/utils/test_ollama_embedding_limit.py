# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Tests for server/utils/ollama_embedding_limit.py."""

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from server.utils.core import FetchResult
from server.utils.ollama_embedding_limit import OllamaEmbeddingTokenLimits

BASE_URL = "http://ollama:11434"


def _show(context_length: int | None = 8192, parameters: str = "", modified_at: str = "2026-10-21T12:00:00Z") -> FetchResult:
    model_info: dict[str, Any] = {"general.architecture": "bert"}
    if context_length is not None:
        model_info["bert.context_length"] = context_length
    return FetchResult(status_code=200, data=json.dumps({"model_info": model_info, "parameters": parameters, "modified_at": modified_at}))


def _embed(prompt_eval_count: int) -> FetchResult:
    return FetchResult(status_code=200, data=json.dumps({"embeddings": [[0.1]], "prompt_eval_count": prompt_eval_count}))


def _version(version: str = "0.40.5") -> FetchResult:
    return FetchResult(status_code=200, data=json.dumps({"version": version}))


def _ps(*names: str) -> FetchResult:
    return FetchResult(status_code=200, data=json.dumps({"models": [{"name": name} for name in names]}))


def _fake_ollama(show: FetchResult, embed: FetchResult, version: FetchResult | None = None, ps: FetchResult | None = None) -> AsyncMock:
    async def fetch(url: str, *_args: Any, **_kwargs: Any) -> FetchResult:
        if url.endswith("/api/version"):
            return version or _version()
        if url.endswith("/api/ps"):
            return ps or _ps()
        return show if url.endswith("/api/show") else embed

    return AsyncMock(side_effect=fetch)


def _embed_calls(mock_fetch: AsyncMock) -> list[Any]:
    return [c for c in mock_fetch.call_args_list if c.args[0].endswith("/api/embed")]


@pytest.mark.asyncio
async def test_resolve_returns_measured_prompt_eval_count(tmp_path: Path) -> None:
    limits = OllamaEmbeddingTokenLimits(tmp_path / "limits.json")

    with patch("server.utils.ollama_embedding_limit.fetch_from", _fake_ollama(_show(8192), _embed(2048))):
        result = await limits.resolve("default", BASE_URL, "bge-m3")

    assert result == 2048


@pytest.mark.asyncio
async def test_resolve_probes_with_one_word_per_window_token(tmp_path: Path) -> None:
    limits = OllamaEmbeddingTokenLimits(tmp_path / "limits.json")
    mock_fetch = _fake_ollama(_show(512), _embed(512))

    with patch("server.utils.ollama_embedding_limit.fetch_from", mock_fetch):
        await limits.resolve("default", BASE_URL, "mxbai-embed-large")

    payload = _embed_calls(mock_fetch)[0].args[2]
    assert payload["model"] == "mxbai-embed-large"
    assert payload["input"] == " a" * 512


@pytest.mark.asyncio
async def test_resolve_caps_probe_length_for_huge_windows(tmp_path: Path) -> None:
    limits = OllamaEmbeddingTokenLimits(tmp_path / "limits.json")
    mock_fetch = _fake_ollama(_show(262144), _embed(4096))

    with patch("server.utils.ollama_embedding_limit.fetch_from", mock_fetch):
        await limits.resolve("default", BASE_URL, "embeddinggemma-2")

    assert _embed_calls(mock_fetch)[0].args[2]["input"] == " a" * 32768


@pytest.mark.asyncio
async def test_resolve_reuses_cached_limit_without_measuring_again(tmp_path: Path) -> None:
    cache_file = tmp_path / "limits.json"
    mock_fetch = _fake_ollama(_show(8192), _embed(2048))

    with patch("server.utils.ollama_embedding_limit.fetch_from", mock_fetch):
        await OllamaEmbeddingTokenLimits(cache_file).resolve("default", BASE_URL, "bge-m3")
        result = await OllamaEmbeddingTokenLimits(cache_file).resolve("default", BASE_URL, "bge-m3")

    assert result == 2048
    assert len(_embed_calls(mock_fetch)) == 1


@pytest.mark.asyncio
async def test_resolve_measures_again_when_model_changes(tmp_path: Path) -> None:
    limits = OllamaEmbeddingTokenLimits(tmp_path / "limits.json")

    with patch("server.utils.ollama_embedding_limit.fetch_from", _fake_ollama(_show(8192, modified_at="v1"), _embed(2048))):
        await limits.resolve("default", BASE_URL, "bge-m3")
    with patch("server.utils.ollama_embedding_limit.fetch_from", _fake_ollama(_show(8192, modified_at="v2"), _embed(8192))):
        result = await limits.resolve("default", BASE_URL, "bge-m3")

    assert result == 8192


@pytest.mark.asyncio
async def test_resolve_measures_again_when_ollama_is_upgraded(tmp_path: Path) -> None:
    limits = OllamaEmbeddingTokenLimits(tmp_path / "limits.json")

    with patch("server.utils.ollama_embedding_limit.fetch_from", _fake_ollama(_show(8192), _embed(2048), _version("0.40.5"))):
        await limits.resolve("default", BASE_URL, "qwen3-embedding")
    with patch("server.utils.ollama_embedding_limit.fetch_from", _fake_ollama(_show(8192), _embed(8192), _version("0.41.0"))):
        result = await limits.resolve("default", BASE_URL, "qwen3-embedding")

    assert result == 8192


@pytest.mark.asyncio
async def test_resolve_measures_again_when_version_is_unreadable(tmp_path: Path) -> None:
    limits = OllamaEmbeddingTokenLimits(tmp_path / "limits.json")

    with patch("server.utils.ollama_embedding_limit.fetch_from", _fake_ollama(_show(8192), _embed(2048))):
        await limits.resolve("default", BASE_URL, "bge-m3")
    with patch(
        "server.utils.ollama_embedding_limit.fetch_from",
        _fake_ollama(_show(8192), _embed(4096), FetchResult(status_code=404, data="")),
    ):
        result = await limits.resolve("default", BASE_URL, "bge-m3")

    assert result == 4096


@pytest.mark.asyncio
async def test_resolve_measures_again_when_scope_changes(tmp_path: Path) -> None:
    limits = OllamaEmbeddingTokenLimits(tmp_path / "limits.json")

    with patch("server.utils.ollama_embedding_limit.fetch_from", _fake_ollama(_show(8192), _embed(2048))):
        await limits.resolve("default", BASE_URL, "bge-m3", scope="2048")
    with patch("server.utils.ollama_embedding_limit.fetch_from", _fake_ollama(_show(8192), _embed(4096))):
        result = await limits.resolve("default", BASE_URL, "bge-m3", scope="4096")

    assert result == 4096


@pytest.mark.asyncio
async def test_resolve_keeps_instances_apart(tmp_path: Path) -> None:
    limits = OllamaEmbeddingTokenLimits(tmp_path / "limits.json")

    with patch("server.utils.ollama_embedding_limit.fetch_from", _fake_ollama(_show(8192), _embed(2048))):
        await limits.resolve("default", BASE_URL, "bge-m3")
    with patch("server.utils.ollama_embedding_limit.fetch_from", _fake_ollama(_show(8192), _embed(4096))):
        result = await limits.resolve("gpu-1", BASE_URL, "bge-m3")

    assert result == 4096


@pytest.mark.asyncio
async def test_resolve_falls_back_to_modelfile_num_ctx_when_measuring_fails(tmp_path: Path) -> None:
    limits = OllamaEmbeddingTokenLimits(tmp_path / "limits.json")
    show = _show(512, parameters='stop                           "</s>"\nnum_ctx                        128')

    with patch("server.utils.ollama_embedding_limit.fetch_from", _fake_ollama(show, FetchResult(status_code=500, data=""))):
        result = await limits.resolve("default", BASE_URL, "paraphrase-multilingual")

    assert result == 128


@pytest.mark.asyncio
async def test_resolve_falls_back_to_window_without_num_ctx(tmp_path: Path) -> None:
    limits = OllamaEmbeddingTokenLimits(tmp_path / "limits.json")

    with patch("server.utils.ollama_embedding_limit.fetch_from", _fake_ollama(_show(8192), FetchResult(status_code=500, data=""))):
        result = await limits.resolve("default", BASE_URL, "bge-m3")

    assert result == 8192


@pytest.mark.asyncio
async def test_resolve_does_not_cache_a_fallback(tmp_path: Path) -> None:
    limits = OllamaEmbeddingTokenLimits(tmp_path / "limits.json")

    with patch("server.utils.ollama_embedding_limit.fetch_from", _fake_ollama(_show(8192), FetchResult(status_code=500, data=""))):
        await limits.resolve("default", BASE_URL, "bge-m3")
    with patch("server.utils.ollama_embedding_limit.fetch_from", _fake_ollama(_show(8192), _embed(2048))):
        result = await limits.resolve("default", BASE_URL, "bge-m3")

    assert result == 2048


@pytest.mark.asyncio
async def test_resolve_returns_none_when_show_fails(tmp_path: Path) -> None:
    limits = OllamaEmbeddingTokenLimits(tmp_path / "limits.json")

    with patch("server.utils.ollama_embedding_limit.fetch_from", _fake_ollama(FetchResult(status_code=404, data=""), _embed(2048))):
        result = await limits.resolve("default", BASE_URL, "missing")

    assert result is None


@pytest.mark.asyncio
async def test_resolve_ignores_unreadable_cache_file(tmp_path: Path) -> None:
    cache_file = tmp_path / "limits.json"
    cache_file.write_text("not json")
    limits = OllamaEmbeddingTokenLimits(cache_file)

    with patch("server.utils.ollama_embedding_limit.fetch_from", _fake_ollama(_show(8192), _embed(2048))):
        result = await limits.resolve("default", BASE_URL, "bge-m3")

    assert result == 2048
    assert json.loads(cache_file.read_text())["default/bge-m3"]["limit"] == 2048


@pytest.mark.asyncio
async def test_resolve_ignores_parameters_that_are_not_text(tmp_path: Path) -> None:
    limits = OllamaEmbeddingTokenLimits(tmp_path / "limits.json")
    show = FetchResult(
        status_code=200,
        data=json.dumps(
            {"model_info": {"general.architecture": "bert", "bert.context_length": 512}, "parameters": None, "modified_at": "v1"}
        ),
    )

    with patch("server.utils.ollama_embedding_limit.fetch_from", _fake_ollama(show, FetchResult(status_code=500, data=""))):
        result = await limits.resolve("default", BASE_URL, "granite-embedding")

    assert result == 512


@pytest.mark.asyncio
async def test_resolve_returns_none_when_ollama_is_unreachable(tmp_path: Path) -> None:
    limits = OllamaEmbeddingTokenLimits(tmp_path / "limits.json")

    with patch("server.utils.ollama_embedding_limit.fetch_from", AsyncMock(side_effect=OSError("connection refused"))):
        result = await limits.resolve("default", BASE_URL, "bge-m3")

    assert result is None


@pytest.mark.asyncio
async def test_resolve_falls_back_when_measuring_raises(tmp_path: Path) -> None:
    limits = OllamaEmbeddingTokenLimits(tmp_path / "limits.json")
    show = _show(512, parameters="num_ctx                        256")

    async def fetch(url: str, *_args: Any, **_kwargs: Any) -> FetchResult:
        if url.endswith("/api/embed"):
            raise TimeoutError
        return _version() if url.endswith("/api/version") else show

    with patch("server.utils.ollama_embedding_limit.fetch_from", AsyncMock(side_effect=fetch)):
        result = await limits.resolve("default", BASE_URL, "all-minilm")

    assert result == 256


@pytest.mark.asyncio
async def test_resolve_returns_measured_limit_when_cache_cannot_be_saved(tmp_path: Path) -> None:
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("")
    limits = OllamaEmbeddingTokenLimits(blocker / "limits.json")

    with patch("server.utils.ollama_embedding_limit.fetch_from", _fake_ollama(_show(8192), _embed(2048))):
        result = await limits.resolve("default", BASE_URL, "bge-m3")

    assert result == 2048


@pytest.mark.asyncio
@pytest.mark.parametrize("version_error", [OSError("connection reset"), FetchResult(status_code=200, data=json.dumps({"version": None}))])
async def test_resolve_measures_when_version_cannot_be_read(tmp_path: Path, version_error: OSError | FetchResult) -> None:
    limits = OllamaEmbeddingTokenLimits(tmp_path / "limits.json")

    async def fetch(url: str, *_args: Any, **_kwargs: Any) -> FetchResult:
        if url.endswith("/api/version"):
            if isinstance(version_error, OSError):
                raise version_error
            return version_error
        return _show(8192) if url.endswith("/api/show") else _embed(2048)

    with patch("server.utils.ollama_embedding_limit.fetch_from", AsyncMock(side_effect=fetch)):
        result = await limits.resolve("default", BASE_URL, "bge-m3")

    assert result == 2048


@pytest.mark.asyncio
async def test_measure_unloads_a_model_that_was_not_loaded(tmp_path: Path) -> None:
    limits = OllamaEmbeddingTokenLimits(tmp_path / "limits.json")
    mock_fetch = _fake_ollama(_show(512), _embed(512), ps=_ps("other:latest"))

    with patch("server.utils.ollama_embedding_limit.fetch_from", mock_fetch):
        await limits.resolve("default", BASE_URL, "mxbai-embed-large")

    assert _embed_calls(mock_fetch)[0].args[2]["keep_alive"] == 0


@pytest.mark.asyncio
async def test_measure_keeps_a_model_that_was_already_loaded(tmp_path: Path) -> None:
    limits = OllamaEmbeddingTokenLimits(tmp_path / "limits.json")
    mock_fetch = _fake_ollama(_show(512), _embed(512), ps=_ps("mxbai-embed-large:latest"))

    with patch("server.utils.ollama_embedding_limit.fetch_from", mock_fetch):
        await limits.resolve("default", BASE_URL, "mxbai-embed-large")

    assert "keep_alive" not in _embed_calls(mock_fetch)[0].args[2]


@pytest.mark.asyncio
@pytest.mark.parametrize("ps", [FetchResult(status_code=500, data=""), FetchResult(status_code=200, data="not json")])
async def test_measure_leaves_keep_alive_alone_when_ps_is_unreadable(tmp_path: Path, ps: FetchResult) -> None:
    limits = OllamaEmbeddingTokenLimits(tmp_path / "limits.json")
    mock_fetch = _fake_ollama(_show(512), _embed(512), ps=ps)

    with patch("server.utils.ollama_embedding_limit.fetch_from", mock_fetch):
        result = await limits.resolve("default", BASE_URL, "mxbai-embed-large")

    assert result == 512
    assert "keep_alive" not in _embed_calls(mock_fetch)[0].args[2]


@pytest.mark.asyncio
async def test_resolve_measures_again_once_the_limit_expires(tmp_path: Path) -> None:
    cache_file = tmp_path / "limits.json"
    mock_fetch = _fake_ollama(_show(8192), _embed(2048))

    with patch("server.utils.ollama_embedding_limit.fetch_from", mock_fetch):
        with patch("server.utils.ollama_embedding_limit.time.time", return_value=1000.0):
            await OllamaEmbeddingTokenLimits(cache_file, max_age_seconds=60).resolve("remote", BASE_URL, "bge-m3")
        with patch("server.utils.ollama_embedding_limit.time.time", return_value=1059.0):
            await OllamaEmbeddingTokenLimits(cache_file, max_age_seconds=60).resolve("remote", BASE_URL, "bge-m3")
        with patch("server.utils.ollama_embedding_limit.time.time", return_value=1061.0):
            await OllamaEmbeddingTokenLimits(cache_file, max_age_seconds=60).resolve("remote", BASE_URL, "bge-m3")

    assert len(_embed_calls(mock_fetch)) == 2


@pytest.mark.asyncio
async def test_has_fresh_measurement_until_the_limit_expires(tmp_path: Path) -> None:
    limits = OllamaEmbeddingTokenLimits(tmp_path / "limits.json", max_age_seconds=60)

    with (
        patch("server.utils.ollama_embedding_limit.fetch_from", _fake_ollama(_show(8192), _embed(2048))),
        patch("server.utils.ollama_embedding_limit.time.time", return_value=1000.0),
    ):
        assert limits.has_fresh_measurement("remote", "bge-m3") is False
        await limits.resolve("remote", BASE_URL, "bge-m3")
        assert limits.has_fresh_measurement("remote", "bge-m3") is True
    with patch("server.utils.ollama_embedding_limit.time.time", return_value=1061.0):
        assert limits.has_fresh_measurement("remote", "bge-m3") is False


@pytest.mark.asyncio
async def test_has_no_fresh_measurement_after_a_fallback(tmp_path: Path) -> None:
    limits = OllamaEmbeddingTokenLimits(tmp_path / "limits.json")

    with patch("server.utils.ollama_embedding_limit.fetch_from", _fake_ollama(_show(8192), FetchResult(status_code=500, data=""))):
        assert await limits.resolve("remote", BASE_URL, "bge-m3") == 8192

    assert limits.has_fresh_measurement("remote", "bge-m3") is False


@pytest.mark.asyncio
async def test_measurement_without_max_age_stays_fresh(tmp_path: Path) -> None:
    limits = OllamaEmbeddingTokenLimits(tmp_path / "limits.json")

    with patch("server.utils.ollama_embedding_limit.fetch_from", _fake_ollama(_show(8192), _embed(2048))):
        await limits.resolve("default", BASE_URL, "bge-m3")

    assert limits.has_fresh_measurement("default", "bge-m3") is True


@pytest.mark.asyncio
async def test_fallback_is_capped_by_context_ceiling(tmp_path: Path) -> None:
    limits = OllamaEmbeddingTokenLimits(tmp_path / "limits.json")

    with patch("server.utils.ollama_embedding_limit.fetch_from", _fake_ollama(_show(8192), FetchResult(status_code=500, data=""))):
        result = await limits.resolve("default", BASE_URL, "bge-m3", context_ceiling=4096)

    assert result == 4096


@pytest.mark.asyncio
@pytest.mark.parametrize("body", ["null", "[]", '"text"'])
async def test_resolve_returns_none_when_show_is_not_an_object(tmp_path: Path, body: str) -> None:
    limits = OllamaEmbeddingTokenLimits(tmp_path / "limits.json")

    with patch("server.utils.ollama_embedding_limit.fetch_from", _fake_ollama(FetchResult(status_code=200, data=body), _embed(2048))):
        result = await limits.resolve("default", BASE_URL, "bge-m3")

    assert result is None


@pytest.mark.asyncio
async def test_resolve_measures_when_model_info_is_not_an_object(tmp_path: Path) -> None:
    limits = OllamaEmbeddingTokenLimits(tmp_path / "limits.json")
    show = FetchResult(status_code=200, data=json.dumps({"model_info": None, "modified_at": "v1"}))
    mock_fetch = _fake_ollama(show, _embed(512))

    with patch("server.utils.ollama_embedding_limit.fetch_from", mock_fetch):
        result = await limits.resolve("default", BASE_URL, "all-minilm")

    assert result == 512
    assert _embed_calls(mock_fetch)[0].args[2]["input"] == " a" * 32768


@pytest.mark.asyncio
async def test_probes_to_one_ollama_run_one_at_a_time(tmp_path: Path) -> None:
    limits = OllamaEmbeddingTokenLimits(tmp_path / "limits.json")
    running = 0
    most_running = 0

    async def fetch(url: str, *_args: Any, **_kwargs: Any) -> FetchResult:
        nonlocal running, most_running
        if url.endswith("/api/show"):
            return _show(512)
        if url.endswith("/api/version"):
            return _version()
        if url.endswith("/api/ps"):
            return _ps()
        running += 1
        most_running = max(most_running, running)
        await asyncio.sleep(0.01)
        running -= 1
        return _embed(512)

    with patch("server.utils.ollama_embedding_limit.fetch_from", AsyncMock(side_effect=fetch)):
        results = await asyncio.gather(*(limits.resolve("default", BASE_URL, f"model-{i}") for i in range(4)))

    assert results == [512] * 4
    assert most_running == 1
