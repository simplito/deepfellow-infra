# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Tests for server/utils/ollama_tokenizer.py."""

import asyncio
import io
import json
import threading
import zipfile
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from tokenizers import Tokenizer

from server.utils.core import FetchResult
from server.utils.gguf_tokenizer import build_gguf_tokenizer
from server.utils.ollama_tokenizer import OllamaTokenizerError, OllamaTokenizers

BASE_URL = "http://ollama:11434"

_MODEL_INFO = {
    "tokenizer.ggml.model": "bert",
    "tokenizer.ggml.tokens": ["[UNK]", "[CLS]", "[SEP]", "▁hello"],
    "tokenizer.ggml.token_type": [3, 3, 3, 1],
    "tokenizer.ggml.unknown_token_id": 0,
    "tokenizer.ggml.cls_token_id": 1,
    "tokenizer.ggml.seperator_token_id": 2,
}


def _show(model_info: dict[str, Any] | None = None, modified_at: str = "v1") -> FetchResult:
    return FetchResult(status_code=200, data=json.dumps({"model_info": model_info or _MODEL_INFO, "modified_at": modified_at}))


def _fake_ollama(*answers: FetchResult) -> AsyncMock:
    return AsyncMock(side_effect=list(answers))


def _verbose_calls(mock_fetch: AsyncMock) -> int:
    return sum(1 for c in mock_fetch.call_args_list if c.args[2]["verbose"])


@pytest.mark.asyncio
async def test_archive_holds_tokenizer_files_at_root() -> None:
    with patch("server.utils.ollama_tokenizer.fetch_from", _fake_ollama(_show(), _show())):
        archive = await OllamaTokenizers().archive(BASE_URL, "all-minilm")

    assert archive is not None
    names = zipfile.ZipFile(io.BytesIO(archive)).namelist()
    assert names == ["tokenizer.json", "tokenizer_config.json"]


@pytest.mark.asyncio
async def test_archive_is_built_off_the_event_loop_thread() -> None:
    built_on: list[int] = []

    def build(_text: str, _model_name: str) -> bytes:
        built_on.append(threading.get_ident())
        return b"zip"

    with (
        patch("server.utils.ollama_tokenizer.fetch_from", _fake_ollama(_show(), _show())),
        patch("server.utils.ollama_tokenizer._archive_from_show", side_effect=build),
    ):
        assert await OllamaTokenizers().archive(BASE_URL, "all-minilm") == b"zip"

    assert built_on
    assert built_on != [threading.get_ident()]


@pytest.mark.asyncio
async def test_archive_is_built_once_per_model_version() -> None:
    tokenizers = OllamaTokenizers()
    mock_fetch = _fake_ollama(_show(), _show(), _show())

    with patch("server.utils.ollama_tokenizer.fetch_from", mock_fetch):
        first = await tokenizers.archive(BASE_URL, "all-minilm")
        second = await tokenizers.archive(BASE_URL, "all-minilm")

    assert first == second
    assert _verbose_calls(mock_fetch) == 1


@pytest.mark.asyncio
async def test_archive_is_rebuilt_when_the_model_changes() -> None:
    tokenizers = OllamaTokenizers()
    mock_fetch = _fake_ollama(_show(modified_at="v1"), _show(modified_at="v1"), _show(modified_at="v2"), _show(modified_at="v2"))

    with patch("server.utils.ollama_tokenizer.fetch_from", mock_fetch):
        await tokenizers.archive(BASE_URL, "all-minilm")
        await tokenizers.archive(BASE_URL, "all-minilm")

    assert _verbose_calls(mock_fetch) == 2


@pytest.mark.asyncio
async def test_unsupported_tokenizer_returns_none_and_is_remembered() -> None:
    tokenizers = OllamaTokenizers()
    unsupported = _show({"tokenizer.ggml.model": "rwkv", "tokenizer.ggml.tokens": ["a"]})
    mock_fetch = _fake_ollama(unsupported, unsupported, unsupported)

    with patch("server.utils.ollama_tokenizer.fetch_from", mock_fetch):
        first = await tokenizers.archive(BASE_URL, "odd-model")
        second = await tokenizers.archive(BASE_URL, "odd-model")

    assert first is None
    assert second is None
    assert _verbose_calls(mock_fetch) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer",
    [
        FetchResult(status_code=500, data=""),
        FetchResult(status_code=200, data="not json"),
        FetchResult(status_code=200, data="[]"),
    ],
)
async def test_archive_raises_when_ollama_cannot_answer(answer: FetchResult) -> None:
    with patch("server.utils.ollama_tokenizer.fetch_from", _fake_ollama(answer)), pytest.raises(OllamaTokenizerError):
        await OllamaTokenizers().archive(BASE_URL, "all-minilm")


@pytest.mark.asyncio
async def test_archive_raises_when_ollama_is_unreachable() -> None:
    with (
        patch("server.utils.ollama_tokenizer.fetch_from", AsyncMock(side_effect=OSError("connection refused"))),
        pytest.raises(OllamaTokenizerError),
    ):
        await OllamaTokenizers().archive(BASE_URL, "all-minilm")


@pytest.mark.asyncio
async def test_archive_without_model_info_raises_and_is_not_remembered() -> None:
    tokenizers = OllamaTokenizers()
    no_info = FetchResult(status_code=200, data=json.dumps({"modified_at": "v1"}))
    mock_fetch = _fake_ollama(no_info, no_info, _show(), _show())

    with patch("server.utils.ollama_tokenizer.fetch_from", mock_fetch):
        with pytest.raises(OllamaTokenizerError):
            await tokenizers.archive(BASE_URL, "all-minilm")
        archive = await tokenizers.archive(BASE_URL, "all-minilm")

    assert archive is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("verbose", [False, True])
async def test_model_ollama_no_longer_knows_has_no_tokenizer_and_is_not_remembered(verbose: bool) -> None:
    tokenizers = OllamaTokenizers()
    missing = FetchResult(status_code=404, data="")
    first = [_show(), missing] if verbose else [missing]
    mock_fetch = _fake_ollama(*first, _show(), _show())

    with patch("server.utils.ollama_tokenizer.fetch_from", mock_fetch):
        gone = await tokenizers.archive(BASE_URL, "all-minilm")
        back = await tokenizers.archive(BASE_URL, "all-minilm")

    assert gone is None
    assert back is not None


@pytest.mark.asyncio
async def test_tokenizer_that_fails_to_build_returns_none_and_is_remembered() -> None:
    tokenizers = OllamaTokenizers()
    broken = _show(_MODEL_INFO | {"tokenizer.ggml.token_type": ["x", 3, 3, 1]})
    mock_fetch = _fake_ollama(broken, broken, broken)

    with patch("server.utils.ollama_tokenizer.fetch_from", mock_fetch):
        first = await tokenizers.archive(BASE_URL, "all-minilm")
        second = await tokenizers.archive(BASE_URL, "all-minilm")

    assert first is None
    assert second is None
    assert _verbose_calls(mock_fetch) == 1


@pytest.mark.asyncio
async def test_tokenizer_that_fails_to_build_is_logged_as_an_error_not_as_unsupported() -> None:
    broken = _show(_MODEL_INFO | {"tokenizer.ggml.token_type": ["x", 3, 3, 1]})

    with (
        patch("server.utils.ollama_tokenizer.fetch_from", _fake_ollama(broken, broken)),
        patch("server.utils.ollama_tokenizer.logger") as mock_logger,
    ):
        assert await OllamaTokenizers().archive(BASE_URL, "all-minilm") is None

    assert mock_logger.exception.call_count == 1
    assert mock_logger.info.call_count == 0


@pytest.mark.asyncio
async def test_running_out_of_memory_raises_and_is_not_remembered() -> None:
    tokenizers = OllamaTokenizers()
    real_build = build_gguf_tokenizer
    calls = 0

    def build(model_info: dict[str, Any]) -> Tokenizer | None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise MemoryError
        return real_build(model_info)

    with (
        patch("server.utils.ollama_tokenizer.fetch_from", _fake_ollama(_show(), _show(), _show(), _show())),
        patch("server.utils.ollama_tokenizer.build_gguf_tokenizer", side_effect=build),
    ):
        with pytest.raises(MemoryError):
            await tokenizers.archive(BASE_URL, "all-minilm")
        archive = await tokenizers.archive(BASE_URL, "all-minilm")

    assert archive is not None


@pytest.mark.asyncio
async def test_concurrent_requests_build_the_archive_once() -> None:
    tokenizers = OllamaTokenizers()

    async def fetch(*_args: Any, **_kwargs: Any) -> FetchResult:
        await asyncio.sleep(0.01)
        return _show()

    mock_fetch = AsyncMock(side_effect=fetch)
    with patch("server.utils.ollama_tokenizer.fetch_from", mock_fetch):
        archives = await asyncio.gather(*(tokenizers.archive(BASE_URL, "all-minilm") for _ in range(3)))

    assert len(set(archives)) == 1
    assert _verbose_calls(mock_fetch) == 1


@pytest.mark.asyncio
async def test_archive_that_fails_to_pack_returns_none() -> None:
    with (
        patch("server.utils.ollama_tokenizer.fetch_from", _fake_ollama(_show(), _show())),
        patch("server.utils.ollama_tokenizer.tokenizer_files", side_effect=ValueError("unencodable")),
    ):
        archive = await OllamaTokenizers().archive(BASE_URL, "all-minilm")

    assert archive is None
