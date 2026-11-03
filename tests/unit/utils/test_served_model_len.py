# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Tests for server/utils/served_model_len.py."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from server.utils.core import FetchResult
from server.utils.served_model_len import fetch_served_max_model_len

BASE_URL = "http://vllm:8000"


def _models(*cards: dict[str, object]) -> FetchResult:
    return FetchResult(status_code=200, data=json.dumps({"object": "list", "data": list(cards)}))


@pytest.mark.asyncio
async def test_returns_max_model_len_of_served_model() -> None:
    response = _models({"id": "other", "max_model_len": 4096}, {"id": "all-minilm", "max_model_len": 256})

    with patch("server.utils.served_model_len.fetch_from", new_callable=AsyncMock, return_value=response) as mock_fetch:
        result = await fetch_served_max_model_len(BASE_URL, "all-minilm")

    assert result == 256
    assert mock_fetch.call_args.args[0] == "http://vllm:8000/v1/models"


@pytest.mark.asyncio
async def test_returns_none_when_model_is_not_served() -> None:
    with patch(
        "server.utils.served_model_len.fetch_from", new_callable=AsyncMock, return_value=_models({"id": "other", "max_model_len": 1})
    ):
        result = await fetch_served_max_model_len(BASE_URL, "all-minilm")

    assert result is None


@pytest.mark.asyncio
async def test_returns_none_when_card_has_no_length() -> None:
    with patch("server.utils.served_model_len.fetch_from", new_callable=AsyncMock, return_value=_models({"id": "all-minilm"})):
        result = await fetch_served_max_model_len(BASE_URL, "all-minilm")

    assert result is None


@pytest.mark.asyncio
async def test_returns_none_on_error_status() -> None:
    with patch("server.utils.served_model_len.fetch_from", new_callable=AsyncMock, return_value=FetchResult(status_code=503, data="")):
        result = await fetch_served_max_model_len(BASE_URL, "all-minilm")

    assert result is None


@pytest.mark.asyncio
async def test_returns_none_when_server_is_unreachable() -> None:
    with patch("server.utils.served_model_len.fetch_from", new_callable=AsyncMock, side_effect=OSError("connection refused")):
        result = await fetch_served_max_model_len(BASE_URL, "all-minilm")

    assert result is None


@pytest.mark.asyncio
async def test_returns_none_when_data_is_not_a_list() -> None:
    response = FetchResult(status_code=200, data=json.dumps({"data": {"id": "all-minilm", "max_model_len": 256}}))

    with patch("server.utils.served_model_len.fetch_from", new_callable=AsyncMock, return_value=response):
        result = await fetch_served_max_model_len(BASE_URL, "all-minilm")

    assert result is None


@pytest.mark.asyncio
async def test_skips_cards_that_are_not_objects() -> None:
    response = _models(None, "x", {"id": "all-minilm", "max_model_len": 256})  # pyright: ignore[reportArgumentType]

    with patch("server.utils.served_model_len.fetch_from", new_callable=AsyncMock, return_value=response):
        result = await fetch_served_max_model_len(BASE_URL, "all-minilm")

    assert result == 256
