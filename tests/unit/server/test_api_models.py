# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError
from starlette.datastructures import UploadFile

from server.models.api import (
    ChatCompletionRequest,
    ComparisonFilter,
    CompletionLegacyRequest,
    CompoundFilter,
    CreateSpeechRequest,
    CreateTranscriptionRequest,
    EmbeddingRequest,
    ImagesRequest,
)


def _make_filter(type_: str, key: str = "score", value: int = 10) -> ComparisonFilter:
    return ComparisonFilter(key=key, type=type_, value=value)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize(
    ("operator", "data", "expected"),
    [
        ("eq", {}, False),
        ("eq", {"score": 10}, True),
        ("eq", {"score": 5}, False),
        ("ne", {"score": 5}, True),
        ("ne", {"score": 10}, False),
        ("lt", {"score": 5}, True),
        ("lt", {"score": 15}, False),
        ("lte", {"score": 10}, True),
        ("lte", {"score": 9}, True),
        ("lte", {"score": 11}, False),
        ("gt", {"score": 15}, True),
        ("gt", {"score": 5}, False),
        ("gte", {"score": 10}, True),
        ("gte", {"score": 11}, True),
        ("gte", {"score": 9}, False),
    ],
    ids=[
        "missing_key",
        "eq_true",
        "eq_false",
        "ne_true",
        "ne_false",
        "lt_true",
        "lt_false",
        "lte_equal",
        "lte_less",
        "lte_false",
        "gt_true",
        "gt_false",
        "gte_equal",
        "gte_greater",
        "gte_false",
    ],
)
def test_comparison_filters(operator: str, data: dict[str, int], expected: bool) -> None:
    f = _make_filter(operator)
    assert f.matches(data) is expected


@pytest.mark.parametrize(
    ("filter_type", "filters", "data", "expected"),
    [
        ("and", [ComparisonFilter(key="a", type="eq", value=1), ComparisonFilter(key="b", type="eq", value=2)], {"a": 1, "b": 2}, True),
        ("and", [ComparisonFilter(key="a", type="eq", value=1), ComparisonFilter(key="b", type="eq", value=99)], {"a": 1, "b": 2}, False),
        ("or", [ComparisonFilter(key="a", type="eq", value=99), ComparisonFilter(key="b", type="eq", value=2)], {"a": 1, "b": 2}, True),
        ("or", [ComparisonFilter(key="a", type="eq", value=99), ComparisonFilter(key="b", type="eq", value=99)], {"a": 1, "b": 2}, False),
    ],
    ids=["and_all_match", "and_one_fails", "or_one_matches", "or_none_match"],
)
def test_compound_filter_logic(filter_type: str, filters: list[ComparisonFilter], data: dict[str, int], expected: bool):
    f = CompoundFilter(type=filter_type, filters=filters)  # pyright: ignore[reportArgumentType]
    assert f.matches(data) is expected


def _make_upload_file(content: bytes = b"audio") -> UploadFile:
    mock = MagicMock(spec=UploadFile)
    mock.read = AsyncMock(return_value=content)
    mock.filename = "audio.wav"
    mock.content_type = "audio/wav"
    return mock


@pytest.mark.asyncio
async def test_to_form_serializes_list_field() -> None:
    req = CreateTranscriptionRequest.model_construct(
        file=_make_upload_file(),
        model="whisper-1",
        include=["logprobs"],
        stream=None,
        temperature=None,
        timestamp_granularities=None,
        known_speaker_names=None,
        known_speaker_references=None,
        language=None,
        prompt=None,
        response_format=None,
        chunking_strategy=None,
    )

    form = await req.to_form(remove_model=False, rewrite_model_to=None)

    fields = {f[0]["name"]: f[2] for f in form._fields}  # type: ignore[attr-defined]
    assert "include[]" in fields


@pytest.mark.asyncio
async def test_to_form_serializes_bool_field() -> None:
    req = CreateTranscriptionRequest.model_construct(
        file=_make_upload_file(),
        model="whisper-1",
        include=None,
        stream=True,
        temperature=None,
        timestamp_granularities=None,
        known_speaker_names=None,
        known_speaker_references=None,
        language=None,
        prompt=None,
        response_format=None,
        chunking_strategy=None,
    )

    form = await req.to_form(remove_model=False, rewrite_model_to=None)

    fields = {f[0]["name"]: f[2] for f in form._fields}  # type: ignore[attr-defined]
    assert fields.get("stream") == "true"


@pytest.mark.asyncio
async def test_to_form_serializes_string_field() -> None:
    req = CreateTranscriptionRequest.model_construct(
        file=_make_upload_file(),
        model="whisper-1",
        include=None,
        stream=None,
        temperature=None,
        timestamp_granularities=None,
        known_speaker_names=None,
        known_speaker_references=None,
        language=None,
        prompt="transcribe this",
        response_format=None,
        chunking_strategy=None,
    )

    form = await req.to_form(remove_model=False, rewrite_model_to=None)

    fields = {f[0]["name"]: f[2] for f in form._fields}  # type: ignore[attr-defined]
    assert fields.get("prompt") == "transcribe this"


@pytest.mark.asyncio
async def test_to_form_serializes_float_field() -> None:
    req = CreateTranscriptionRequest.model_construct(
        file=_make_upload_file(),
        model="whisper-1",
        include=None,
        stream=None,
        temperature=0.5,
        timestamp_granularities=None,
        known_speaker_names=None,
        known_speaker_references=None,
        language=None,
        prompt=None,
        response_format=None,
        chunking_strategy=None,
    )

    form = await req.to_form(remove_model=False, rewrite_model_to=None)

    fields = {f[0]["name"]: f[2] for f in form._fields}  # type: ignore[attr-defined]
    assert fields.get("temperature") == "0.5"


@pytest.mark.asyncio
async def test_to_form_remove_model() -> None:
    model = "whisper-1"
    req = CreateTranscriptionRequest.model_construct(
        file=_make_upload_file(),
        model=model,
        include=None,
        stream=None,
        temperature=0.5,
        timestamp_granularities=None,
        known_speaker_names=None,
        known_speaker_references=None,
        language=None,
        prompt=None,
        response_format=None,
        chunking_strategy=None,
    )

    form = await req.to_form(remove_model=True, rewrite_model_to=None)

    fields = {f[0]["name"]: f[2] for f in form._fields}  # type: ignore[attr-defined]
    assert fields.get("model") is None


def _base_chat_req(**kwargs):  # type: ignore[no-untyped-def]
    return {"messages": [{"role": "user", "content": "hi"}], "model": "gpt-4", **kwargs}


@pytest.mark.parametrize("temperature", [-0.1, 2.1])
def test_chat_completion_temperature_out_of_range(temperature: float) -> None:
    with pytest.raises(ValidationError):
        ChatCompletionRequest(**_base_chat_req(temperature=temperature))  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize("temperature", [0.0, 1.0, 2.0])
def test_chat_completion_temperature_valid(temperature: float) -> None:
    req = ChatCompletionRequest(**_base_chat_req(temperature=temperature))  # pyright: ignore[reportArgumentType]

    assert req.temperature == temperature


@pytest.mark.parametrize("top_p", [-0.1, 1.1])
def test_chat_completion_top_p_out_of_range(top_p: float) -> None:
    with pytest.raises(ValidationError):
        ChatCompletionRequest(**_base_chat_req(top_p=top_p))  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize("n", [-1, 129])
def test_chat_completion_n_out_of_range(n: int) -> None:
    with pytest.raises(ValidationError):
        ChatCompletionRequest(**_base_chat_req(n=n))  # pyright: ignore[reportArgumentType]


def test_images_request_n_exceeds_limit() -> None:
    with pytest.raises(ValidationError):
        ImagesRequest(model="dall-e-3", prompt="cat", n=11)


def test_images_request_n_at_limit() -> None:
    req = ImagesRequest(model="dall-e-3", prompt="cat", n=10)

    assert req.n == 10


def test_embedding_request_dimensions_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        EmbeddingRequest(input="text", model="ada", dimensions=0)


def test_create_speech_instructions_max_length() -> None:
    with pytest.raises(ValidationError):
        CreateSpeechRequest(input="hi", model="tts-1", instructions="x" * 4097)


def test_completion_legacy_best_of_constraint() -> None:
    with pytest.raises(ValidationError):
        CompletionLegacyRequest(prompt="hi", model="gpt-3.5", best_of=21)


def test_completion_legacy_max_tokens_zero_allowed() -> None:
    # ge=0 means 0 is valid now
    req = CompletionLegacyRequest(prompt="hi", model="gpt-3.5", max_tokens=0)

    assert req.max_tokens == 0
