# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from server.endpointregistry import ChatCompletionEndpoint, CustomEndpoint, McpEndpoint, RegisteredModel, RegistryEntry, SimpleEndpoint
from server.model_tester import ModelTester, ModelTestError, Response, UnpackedResponse


def _make_json_streaming_response(data: Any, status_code: int = 200) -> StreamingResponse:
    body = json.dumps(data).encode("utf-8")

    async def gen():
        yield body

    return StreamingResponse(gen(), media_type="application/json", status_code=status_code)


def _make_streaming_response(content: bytes, media_type: str = "audio/wav", status_code: int = 200) -> StreamingResponse:
    async def gen():
        yield content

    return StreamingResponse(gen(), media_type=media_type, status_code=status_code)


class _JsonModel(BaseModel):  # pyright: ignore[reportUnusedClass]
    data: dict[str, Any]

    def model_dump_json(self, **kwargs: Any) -> str:  # type: ignore[override]
        return json.dumps(self.data)


def _make_registered_model(model_type: str) -> RegistryEntry:
    endpoint = ChatCompletionEndpoint()
    registered = MagicMock(spec=RegisteredModel)
    registered.name = "test-model"
    registered.type = model_type
    registered.endpoint = endpoint
    entry = MagicMock(spec=RegistryEntry)
    entry.registered_model = registered
    return entry


def test_response_namedtuple_fields() -> None:
    r = Response(status_code=200, media_type="application/json", content=b"hello")

    assert r.status_code == 200
    assert r.media_type == "application/json"
    assert r.content == b"hello"


def test_unpacked_response_fields() -> None:
    ur = UnpackedResponse(status_code=200, media_type="application/json", content=b"{}", json={}, json_error=None)

    assert ur.json == {}
    assert ur.json_error is None


def test_test_error_stores_message_and_response() -> None:
    r = Response(status_code=500, media_type="text/plain", content=b"err")

    e = ModelTestError("something failed", r)
    assert e.error == "something failed"
    assert e.response is r


def test_test_error_response_can_be_none() -> None:
    e = ModelTestError("oops")

    assert e.response is None


@pytest.mark.asyncio
async def test_read_response_streaming_bytes() -> None:
    tester = ModelTester()
    resp = _make_json_streaming_response({"hello": "world"})

    result = await tester._read_response(resp)  # pyright: ignore[reportPrivateUsage]

    assert result.status_code == 200
    assert result.media_type == "application/json"
    assert json.loads(result.content) == {"hello": "world"}


@pytest.mark.asyncio
async def test_read_response_streaming_raises_on_non_bytes() -> None:
    tester = ModelTester()

    async def gen():
        yield "not bytes"

    resp = StreamingResponse(gen(), media_type="application/json")
    with pytest.raises(TypeError):
        await tester._read_response(resp)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_read_response_pydantic_base_model() -> None:
    tester = ModelTester()

    class MyModel(BaseModel):
        value: int

    model = MyModel(value=42)

    result = await tester._read_response(model)  # pyright: ignore[reportPrivateUsage]

    assert result.status_code == 200
    assert result.media_type == "application/json"
    assert json.loads(result.content)["value"] == 42


@pytest.mark.asyncio
async def test_read_response_unsupported_type_raises_runtime_error() -> None:
    tester = ModelTester()
    with pytest.raises(RuntimeError, match="Unsupported StarletteResponse"):
        await tester._read_response("not a valid response")  # pyright: ignore[reportPrivateUsage, reportArgumentType]


@pytest.mark.asyncio
async def test_read_successfully_returns_response_on_200() -> None:
    tester = ModelTester()
    resp = _make_streaming_response(b"audio-bytes", media_type="audio/wav")

    result = await tester._read_successfully(resp)  # pyright: ignore[reportPrivateUsage]

    assert result.status_code == 200
    assert result.content == b"audio-bytes"


@pytest.mark.asyncio
async def test_read_successfully_raises_test_error_on_non_200() -> None:
    tester = ModelTester()
    resp = _make_streaming_response(b"bad", status_code=500)

    with pytest.raises(ModelTestError) as exc_info:
        await tester._read_successfully(resp)  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.error == "Invalid status code"


@pytest.mark.asyncio
async def test_read_json_returns_parsed_json() -> None:
    tester = ModelTester()
    resp = _make_json_streaming_response({"key": "value"})

    (my_resp, data) = await tester._read_json(resp)  # pyright: ignore[reportPrivateUsage]

    assert data == {"key": "value"}
    assert my_resp.media_type == "application/json"


@pytest.mark.asyncio
async def test_read_json_raises_on_non_200() -> None:
    tester = ModelTester()
    resp = _make_json_streaming_response({"err": "bad"}, status_code=422)
    with pytest.raises(ModelTestError) as exc_info:
        await tester._read_json(resp)  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.error == "Invalid status code"


@pytest.mark.asyncio
async def test_read_json_raises_on_wrong_content_type() -> None:
    tester = ModelTester()

    async def gen():
        yield b"hello"

    resp = StreamingResponse(gen(), media_type="text/plain", status_code=200)
    with pytest.raises(ModelTestError) as exc_info:
        await tester._read_json(resp)  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.error == "Invalid content type"


@pytest.mark.asyncio
async def test_read_json_raises_on_invalid_json() -> None:
    tester = ModelTester()

    async def gen():
        yield b"not-json"

    resp = StreamingResponse(gen(), media_type="application/json", status_code=200)

    with pytest.raises(ModelTestError) as exc_info:
        await tester._read_json(resp)  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.error == "Cannot read content as text"


@pytest.mark.asyncio
async def test_test_messages_returns_ok_with_text() -> None:
    tester = ModelTester()
    endpoint = ChatCompletionEndpoint()
    endpoint.on_messages = AsyncMock(return_value=_make_json_streaming_response({"content": [{"text": "hello!"}]}))

    result = await tester._test_messages("test-model", endpoint)  # pyright: ignore[reportPrivateUsage]

    assert result["result"] == "ok"
    assert result["output"] == "hello!"


@pytest.mark.asyncio
async def test_test_messages_returns_ok_with_thinking() -> None:
    tester = ModelTester()
    endpoint = ChatCompletionEndpoint()
    endpoint.on_messages = AsyncMock(return_value=_make_json_streaming_response({"content": [{"thinking": "i think"}]}))

    result = await tester._test_messages("test-model", endpoint)  # pyright: ignore[reportPrivateUsage]

    assert result["output"] == "i think"


@pytest.mark.asyncio
async def test_test_messages_raises_test_error_on_bad_shape() -> None:
    tester = ModelTester()
    endpoint = ChatCompletionEndpoint()
    endpoint.on_messages = AsyncMock(return_value=_make_json_streaming_response({"no_content_key": []}))

    with pytest.raises(ModelTestError) as exc_info:
        await tester._test_messages("test-model", endpoint)  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.error == "Cannot read message content"


@pytest.mark.asyncio
async def test_test_messages_raises_runtime_error_when_no_callback() -> None:
    tester = ModelTester()
    endpoint = ChatCompletionEndpoint()
    with pytest.raises(RuntimeError, match="on_messages"):
        await tester._test_messages("test-model", endpoint)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_test_responses_returns_ok_with_content() -> None:
    tester = ModelTester()
    endpoint = ChatCompletionEndpoint()
    endpoint.on_responses = AsyncMock(return_value=_make_json_streaming_response({"output": [{"content": [{"text": "hi there"}]}]}))

    result = await tester._test_responses("test-model", endpoint)  # pyright: ignore[reportPrivateUsage]

    assert result["result"] == "ok"
    assert result["output"] == "hi there"


@pytest.mark.asyncio
async def test_test_responses_returns_ok_with_summary() -> None:
    tester = ModelTester()
    endpoint = ChatCompletionEndpoint()
    endpoint.on_responses = AsyncMock(return_value=_make_json_streaming_response({"output": [{"summary": [{"text": "summary text"}]}]}))

    result = await tester._test_responses("test-model", endpoint)  # pyright: ignore[reportPrivateUsage]

    assert result["output"] == "summary text"


@pytest.mark.asyncio
async def test_test_responses_returns_empty_string_for_empty_summary() -> None:
    tester = ModelTester()
    endpoint = ChatCompletionEndpoint()
    endpoint.on_responses = AsyncMock(return_value=_make_json_streaming_response({"output": [{"summary": []}]}))

    result = await tester._test_responses("test-model", endpoint)  # pyright: ignore[reportPrivateUsage]

    assert result["output"] == ""


@pytest.mark.asyncio
async def test_test_responses_raises_test_error_on_bad_shape() -> None:
    tester = ModelTester()
    endpoint = ChatCompletionEndpoint()
    endpoint.on_responses = AsyncMock(return_value=_make_json_streaming_response({"output": []}))

    with pytest.raises(ModelTestError) as exc_info:
        await tester._test_responses("test-model", endpoint)  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.error == "Cannot read message content"


@pytest.mark.asyncio
async def test_test_responses_skips_part_without_summary_or_content() -> None:
    tester = ModelTester()
    endpoint = ChatCompletionEndpoint()
    # Second part (first in reverse) has neither key; first part has content
    endpoint.on_responses = AsyncMock(
        return_value=_make_json_streaming_response({"output": [{"content": [{"text": "found"}]}, {"other": "stuff"}]})
    )

    result = await tester._test_responses("test-model", endpoint)  # pyright: ignore[reportPrivateUsage]

    assert result["output"] == "found"


@pytest.mark.asyncio
async def test_test_responses_raises_runtime_error_when_no_callback() -> None:
    tester = ModelTester()
    endpoint = ChatCompletionEndpoint()
    with pytest.raises(RuntimeError, match="on_responses"):
        await tester._test_responses("test-model", endpoint)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_test_chat_completions_returns_ok() -> None:
    tester = ModelTester()
    endpoint = ChatCompletionEndpoint()
    endpoint.on_chat_completion = AsyncMock(return_value=_make_json_streaming_response({"choices": [{"message": {"content": "hello!"}}]}))

    result = await tester._test_chat_completions("test-model", endpoint)  # pyright: ignore[reportPrivateUsage]

    assert result["result"] == "ok"
    assert result["output"] == "hello!"


@pytest.mark.asyncio
async def test_test_chat_completions_raises_test_error_on_bad_shape() -> None:
    tester = ModelTester()
    endpoint = ChatCompletionEndpoint()
    endpoint.on_chat_completion = AsyncMock(return_value=_make_json_streaming_response({"choices": []}))

    with pytest.raises(ModelTestError) as exc_info:
        await tester._test_chat_completions("test-model", endpoint)  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.error == "Cannot read message content"


@pytest.mark.asyncio
async def test_test_chat_completions_raises_runtime_error_when_no_callback() -> None:
    tester = ModelTester()
    endpoint = ChatCompletionEndpoint()

    with pytest.raises(RuntimeError, match="on_chat_completion"):
        await tester._test_chat_completions("test-model", endpoint)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_test_completions_returns_ok() -> None:
    tester = ModelTester()
    endpoint = ChatCompletionEndpoint()
    endpoint.on_completion = AsyncMock(return_value=_make_json_streaming_response({"choices": [{"text": "hello!"}]}))

    result = await tester._test_completions("test-model", endpoint)  # pyright: ignore[reportPrivateUsage]

    assert result["result"] == "ok"
    assert result["output"] == "hello!"


@pytest.mark.asyncio
async def test_test_completions_raises_test_error_on_bad_shape() -> None:
    tester = ModelTester()
    endpoint = ChatCompletionEndpoint()
    endpoint.on_completion = AsyncMock(return_value=_make_json_streaming_response({"choices": []}))

    with pytest.raises(ModelTestError) as exc_info:
        await tester._test_completions("test-model", endpoint)  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.error == "Cannot read message content"


@pytest.mark.asyncio
async def test_test_completions_raises_runtime_error_when_no_callback() -> None:
    tester = ModelTester()
    endpoint = ChatCompletionEndpoint()

    with pytest.raises(RuntimeError, match="on_completion"):
        await tester._test_completions("test-model", endpoint)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_test_embedding_returns_ok() -> None:
    tester = ModelTester()
    endpoint: SimpleEndpoint[Any] = SimpleEndpoint(
        on_request=AsyncMock(return_value=_make_json_streaming_response({"object": "list", "data": []}))
    )

    result = await tester._test_embedding("test-model", endpoint)  # pyright: ignore[reportPrivateUsage]

    assert result["result"] == "ok"


@pytest.mark.asyncio
async def test_test_tts_returns_base64_audio() -> None:
    tester = ModelTester()
    endpoint: SimpleEndpoint[Any] = SimpleEndpoint(
        on_request=AsyncMock(return_value=_make_streaming_response(b"audio-data", media_type="audio/mp3"))
    )

    result = await tester._test_tts("test-model", endpoint)  # pyright: ignore[reportPrivateUsage]

    assert result["result"] == "ok"
    assert "data" in result["output"]
    assert result["output"]["content_type"] == "audio/mp3"


@pytest.mark.asyncio
async def test_test_stt_returns_ok_with_text() -> None:
    tester = ModelTester()
    endpoint: SimpleEndpoint[Any] = SimpleEndpoint(
        on_request=AsyncMock(return_value=_make_json_streaming_response({"text": "transcribed text"}))
    )

    result = await tester._test_stt("test-model", endpoint)  # pyright: ignore[reportPrivateUsage]

    assert result["result"] == "ok"
    assert result["output"] == "transcribed text"


@pytest.mark.asyncio
async def test_test_stt_raises_test_error_on_missing_text() -> None:
    tester = ModelTester()
    endpoint: SimpleEndpoint[Any] = SimpleEndpoint(on_request=AsyncMock(return_value=_make_json_streaming_response({"no_text": True})))

    with pytest.raises(ModelTestError) as exc_info:
        await tester._test_stt("test-model", endpoint)  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.error == "Cannot read response text"


@pytest.mark.asyncio
async def test_test_txt2img_returns_ok_with_b64() -> None:
    tester = ModelTester()
    endpoint: SimpleEndpoint[Any] = SimpleEndpoint(
        on_request=AsyncMock(return_value=_make_json_streaming_response({"data": [{"b64_json": "abc123"}]}))
    )

    result = await tester._test_txt2img("test-model", endpoint)  # pyright: ignore[reportPrivateUsage]

    assert result["result"] == "ok"
    assert result["output"]["data"] == "abc123"
    assert result["output"]["content_type"] == "image/png"


@pytest.mark.asyncio
async def test_test_txt2img_raises_test_error_on_bad_shape() -> None:
    tester = ModelTester()
    endpoint: SimpleEndpoint[Any] = SimpleEndpoint(on_request=AsyncMock(return_value=_make_json_streaming_response({"data": []})))

    with pytest.raises(ModelTestError) as exc_info:
        await tester._test_txt2img("test-model", endpoint)  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.error == "Cannot read image data"


@pytest.mark.asyncio
async def test_test_rerank_returns_ok() -> None:
    tester = ModelTester()
    endpoint: SimpleEndpoint[Any] = SimpleEndpoint(
        on_request=AsyncMock(return_value=_make_json_streaming_response({"results": [{"document": {"text": "A cat is an animal."}}]}))
    )

    result = await tester._test_rerank("test-model", endpoint)  # pyright: ignore[reportPrivateUsage]

    assert result["result"] == "ok"
    assert result["output"] == "A cat is an animal."


@pytest.mark.asyncio
async def test_test_rerank_raises_test_error_on_bad_shape() -> None:
    tester = ModelTester()
    endpoint: SimpleEndpoint[Any] = SimpleEndpoint(on_request=AsyncMock(return_value=_make_json_streaming_response({"results": []})))

    with pytest.raises(ModelTestError) as exc_info:
        await tester._test_rerank("test-model", endpoint)  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.error == "Cannot read image data"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_type",
    ["llm", "llm-v1-v2-v3", "llm-v3", "llm-v3-ant", "llm-v2-v3", "llm-v2-v3-ant", "llm-v1-v3", "llm-v1-v3-ant"],
)
async def test_perform_test_dispatches_responses_for_llm_types(model_type: str) -> None:
    tester = ModelTester()
    entry = _make_registered_model(model_type)
    entry.registered_model.endpoint.on_responses = AsyncMock(
        return_value=_make_json_streaming_response({"output": [{"content": [{"text": "hi"}]}]})
    )

    result = await tester._perform_test(entry)  # pyright: ignore[reportPrivateUsage]

    assert result["result"] == "ok"


@pytest.mark.asyncio
@pytest.mark.parametrize("model_type", ["llm-v1-v2-ant", "llm-v1-ant", "llm-v2-ant", "llm-ant"])
async def test_perform_test_dispatches_messages_for_ant_types(model_type: str) -> None:
    tester = ModelTester()
    entry = _make_registered_model(model_type)
    entry.registered_model.endpoint.on_messages = AsyncMock(return_value=_make_json_streaming_response({"content": [{"text": "hello!"}]}))

    result = await tester._perform_test(entry)  # pyright: ignore[reportPrivateUsage]

    assert result["result"] == "ok"


@pytest.mark.asyncio
@pytest.mark.parametrize("model_type", ["llm-v2", "llm-v1-v2"])
async def test_perform_test_dispatches_chat_completions(model_type: str) -> None:
    tester = ModelTester()
    entry = _make_registered_model(model_type)
    entry.registered_model.endpoint.on_chat_completion = AsyncMock(
        return_value=_make_json_streaming_response({"choices": [{"message": {"content": "hi"}}]})
    )

    result = await tester._perform_test(entry)  # pyright: ignore[reportPrivateUsage]

    assert result["result"] == "ok"


@pytest.mark.asyncio
async def test_perform_test_dispatches_completions_for_llm_v1() -> None:
    tester = ModelTester()
    entry = _make_registered_model("llm-v1")
    entry.registered_model.endpoint.on_completion = AsyncMock(return_value=_make_json_streaming_response({"choices": [{"text": "hi"}]}))

    result = await tester._perform_test(entry)  # pyright: ignore[reportPrivateUsage]

    assert result["result"] == "ok"


@pytest.mark.asyncio
async def test_perform_test_dispatches_embedding() -> None:
    tester = ModelTester()
    entry = _make_registered_model("embedding")
    entry.registered_model.endpoint = SimpleEndpoint(on_request=AsyncMock(return_value=_make_json_streaming_response({"object": "list"})))

    result = await tester._perform_test(entry)  # pyright: ignore[reportPrivateUsage]

    assert result["result"] == "ok"


@pytest.mark.asyncio
async def test_perform_test_dispatches_tts() -> None:
    tester = ModelTester()
    entry = _make_registered_model("tts")
    entry.registered_model.endpoint = SimpleEndpoint(
        on_request=AsyncMock(return_value=_make_streaming_response(b"audio", media_type="audio/mp3"))
    )

    result = await tester._perform_test(entry)  # pyright: ignore[reportPrivateUsage]

    assert result["result"] == "ok"


@pytest.mark.asyncio
async def test_perform_test_dispatches_stt() -> None:
    tester = ModelTester()
    entry = _make_registered_model("stt")
    entry.registered_model.endpoint = SimpleEndpoint(on_request=AsyncMock(return_value=_make_json_streaming_response({"text": "hello"})))

    result = await tester._perform_test(entry)  # pyright: ignore[reportPrivateUsage]

    assert result["result"] == "ok"


@pytest.mark.asyncio
async def test_perform_test_dispatches_txt2img() -> None:
    tester = ModelTester()
    entry = _make_registered_model("txt2img")
    entry.registered_model.endpoint = SimpleEndpoint(
        on_request=AsyncMock(return_value=_make_json_streaming_response({"data": [{"b64_json": "abc"}]}))
    )

    result = await tester._perform_test(entry)  # pyright: ignore[reportPrivateUsage]

    assert result["result"] == "ok"


@pytest.mark.asyncio
async def test_perform_test_dispatches_rerank() -> None:
    tester = ModelTester()
    entry = _make_registered_model("rerank")
    entry.registered_model.endpoint = SimpleEndpoint(
        on_request=AsyncMock(return_value=_make_json_streaming_response({"results": [{"document": {"text": "cat"}}]}))
    )

    result = await tester._perform_test(entry)  # pyright: ignore[reportPrivateUsage]

    assert result["result"] == "ok"


def _make_custom_entry(*responses: StreamingResponse) -> tuple[RegistryEntry, AsyncMock]:
    on_request = AsyncMock(side_effect=responses)
    entry = _make_registered_model("custom")
    entry.registered_model.name = "deepfellow-bge-m3"
    entry.registered_model.endpoint = CustomEndpoint(on_request=on_request)
    return entry, on_request


def _custom_response(status_code: int, content: bytes = b"") -> StreamingResponse:
    return _make_streaming_response(content, media_type="text/plain", status_code=status_code)


@pytest.mark.asyncio
async def test_test_systemone_returns_chosen_option() -> None:
    tester = ModelTester()
    _mock_on_request = AsyncMock(
        return_value=_make_json_streaming_response(
            {
                "model": "tev1:0.8b",
                "answers": {"label": {"type": "choice", "choice": "bug", "probabilities": {"bug": 0.92}, "confidence": 0.88}},
                "usage": {"input_tokens": 174, "output_tokens": 0},
            }
        )
    )
    endpoint: SimpleEndpoint[Any] = SimpleEndpoint(on_request=_mock_on_request)

    result = await tester._test_systemone("tev1:0.8b", endpoint)  # pyright: ignore[reportPrivateUsage]

    assert result["result"] == "ok"
    assert result["output"] == "bug"
    assert _mock_on_request.call_count == 1
    sent = _mock_on_request.call_args.args[0]
    assert sent.model == "tev1:0.8b"
    assert sent.questions["label"].type == "choice"


@pytest.mark.asyncio
async def test_test_systemone_raises_test_error_without_answers() -> None:
    tester = ModelTester()
    endpoint: SimpleEndpoint[Any] = SimpleEndpoint(on_request=AsyncMock(return_value=_make_json_streaming_response({"model": "x"})))

    with pytest.raises(ModelTestError) as exc_info:
        await tester._test_systemone("tev1:0.8b", endpoint)  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.error == "Cannot read decision answer"


@pytest.mark.asyncio
async def test_test_systemone_raises_test_error_on_backend_error() -> None:
    tester = ModelTester()
    endpoint: SimpleEndpoint[Any] = SimpleEndpoint(
        on_request=AsyncMock(return_value=_make_json_streaming_response({"error": "does not support chat"}, status_code=400))
    )

    with pytest.raises(ModelTestError) as exc_info:
        await tester._test_systemone("tev1:0.8b", endpoint)  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.error == "Invalid status code"


@pytest.mark.asyncio
async def test_perform_test_dispatches_decision() -> None:
    tester = ModelTester()
    entry = _make_registered_model("decision")
    entry.registered_model.endpoint = SimpleEndpoint(
        on_request=AsyncMock(return_value=_make_json_streaming_response({"answers": {"label": {"type": "choice", "choice": "bug"}}}))
    )

    result = await tester._perform_test(entry)  # pyright: ignore[reportPrivateUsage]

    assert result["result"] == "ok"
    assert result["output"] == "bug"


@pytest.mark.asyncio
async def test_perform_test_custom_sends_get_to_health_then_test_model() -> None:
    tester = ModelTester()
    entry, on_request = _make_custom_entry(_custom_response(200, b"up"), _custom_response(200, b"passed"))

    result = await tester._perform_test(entry)  # pyright: ignore[reportPrivateUsage]

    requests = [c.args[0] for c in on_request.call_args_list]
    assert [r.method for r in requests] == ["GET", "GET"]
    assert [r.path_params["full_path"] for r in requests] == ["deepfellow-bge-m3/health", "deepfellow-bge-m3/test_model"]
    assert result == {"result": "ok", "output": "passed", "details": {"health": "up", "test_model": "passed"}}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("media_type", "body", "expected_health"),
    [
        ("application/json", b'{"status":"ok"}', {"status": "ok"}),
        ("text/plain", b'{"status":"ok"}', '{"status":"ok"}'),
        ("application/json", b"not json", "not json"),
        ("application/json", b'{"a":1,"a":2}', '{"a":1,"a":2}'),
        ("application/json", b'{"a":NaN}', '{"a":NaN}'),
    ],
)
async def test_perform_test_custom_details_parse_json_only_when_lossless(media_type: str, body: bytes, expected_health: Any) -> None:
    tester = ModelTester()
    entry, _ = _make_custom_entry(_make_streaming_response(body, media_type=media_type), _custom_response(404))

    result = await tester._perform_test(entry)  # pyright: ignore[reportPrivateUsage]

    assert result == {"result": "ok", "output": body.decode(), "details": {"health": expected_health, "test_model": None}}


@pytest.mark.asyncio
async def test_perform_test_custom_empty_test_model_body_does_not_show_health_body() -> None:
    tester = ModelTester()
    entry, _ = _make_custom_entry(_custom_response(200, b"up"), _custom_response(200))

    result = await tester._perform_test(entry)  # pyright: ignore[reportPrivateUsage]

    assert result == {"result": "ok", "output": "passed", "details": {"health": "up", "test_model": ""}}


@pytest.mark.asyncio
async def test_perform_test_custom_reports_unreachable_model() -> None:
    tester = ModelTester()
    entry = _make_registered_model("custom")
    entry.registered_model.endpoint = CustomEndpoint(on_request=AsyncMock(side_effect=aiohttp.ClientConnectionError("Connection refused")))

    with pytest.raises(ModelTestError) as exc_info:
        await tester._perform_test(entry)  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.error == "Model unreachable: Connection refused"


@pytest.mark.asyncio
async def test_perform_test_custom_falls_back_to_healthz() -> None:
    tester = ModelTester()
    entry, on_request = _make_custom_entry(_custom_response(404), _custom_response(200), _custom_response(404))

    result = await tester._perform_test(entry)  # pyright: ignore[reportPrivateUsage]

    assert on_request.call_args_list[1].args[0].path_params["full_path"] == "deepfellow-bge-m3/healthz"
    assert result == {"result": "ok", "output": "healthy", "details": {"health": "", "test_model": None}}


@pytest.mark.asyncio
async def test_perform_test_custom_passes_with_test_model_only() -> None:
    tester = ModelTester()
    entry, _ = _make_custom_entry(_custom_response(404), _custom_response(404), _custom_response(200, b"passed"))

    result = await tester._perform_test(entry)  # pyright: ignore[reportPrivateUsage]

    assert result == {"result": "ok", "output": "passed", "details": {"health": None, "test_model": "passed"}}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("responses", "error"),
    [
        ([503], "Health check failed"),
        ([404, 500], "Health check failed"),
        ([200, 500], "Model test failed"),
        ([404, 404, 500], "Model test failed"),
        ([404, 404, 404], "No health or test endpoint found (tried: /health, /healthz, /test_model)"),
    ],
)
async def test_perform_test_custom_raises_test_error(responses: list[int], error: str) -> None:
    tester = ModelTester()
    entry, on_request = _make_custom_entry(*(_custom_response(status) for status in responses))

    with pytest.raises(ModelTestError) as exc_info:
        await tester._perform_test(entry)  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.error == error
    assert on_request.call_count == len(responses)


@pytest.mark.asyncio
async def test_perform_test_custom_rejects_oversized_body_and_closes_stream() -> None:
    tester = ModelTester()
    closed: list[bool] = []

    async def endless() -> AsyncIterator[bytes]:
        try:
            while True:
                yield b"x" * 1024
        finally:
            closed.append(True)

    entry, _ = _make_custom_entry(StreamingResponse(endless(), media_type="text/plain"))

    with pytest.raises(ModelTestError) as exc_info:
        await tester._perform_test(entry)  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.error == "Response too large (limit 64 KB)"
    assert closed == [True]


class _EndlessBody:
    """Async iterable without `aclose`, unlike a generator."""

    def __aiter__(self) -> "_EndlessBody":
        return self

    async def __anext__(self) -> bytes:
        return b"x" * 1024


@pytest.mark.asyncio
async def test_perform_test_custom_rejects_oversized_body_without_aclose() -> None:
    tester = ModelTester()
    entry, _ = _make_custom_entry(StreamingResponse(_EndlessBody(), media_type="text/plain"))

    with pytest.raises(ModelTestError) as exc_info:
        await tester._perform_test(entry)  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.error == "Response too large (limit 64 KB)"


@pytest.mark.asyncio
@patch("server.model_tester.CUSTOM_TEST_TIMEOUT_SECONDS", 0.01)
async def test_perform_test_custom_times_out() -> None:
    tester = ModelTester()

    async def hang(_: Any) -> StreamingResponse:
        await asyncio.sleep(1)
        raise AssertionError("unreachable")

    entry = _make_registered_model("custom")
    entry.registered_model.name = "deepfellow-bge-m3"
    entry.registered_model.endpoint = CustomEndpoint(on_request=hang)

    with pytest.raises(ModelTestError) as exc_info:
        await tester._perform_test(entry)  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.error == "Model test timed out (limit 0.01 s)"


@pytest.mark.asyncio
async def test_perform_test_raises_runtime_error_for_unknown_type() -> None:
    tester = ModelTester()
    entry = _make_registered_model("unknown-type")

    with pytest.raises(RuntimeError, match="unsupported type"):
        await tester._perform_test(entry)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_perform_test_raises_runtime_error_for_llm_type_without_testable_endpoint() -> None:
    tester = ModelTester()
    entry = _make_registered_model("llm-ollama")

    with pytest.raises(RuntimeError, match="unsupported type"):
        await tester._perform_test(entry)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_test_model_returns_error_on_test_error() -> None:
    tester = ModelTester()
    tester._perform_test = AsyncMock(side_effect=ModelTestError("boom"))  # pyright: ignore[reportPrivateUsage]

    result = await tester.test_model(_make_registered_model("custom"))

    assert result == {"error": "boom"}


@pytest.mark.asyncio
async def test_test_model_returns_error_with_details_when_response_present() -> None:
    tester = ModelTester()
    entry = _make_registered_model("llm-v2")
    entry.registered_model.endpoint.on_chat_completion = AsyncMock(
        return_value=_make_json_streaming_response({"choices": []}, status_code=422)
    )

    result = await tester.test_model(entry)

    assert "error" in result
    assert "details" in result


@pytest.mark.asyncio
async def test_test_model_unhealthy_custom_with_nan_body_stays_serializable() -> None:
    tester = ModelTester()
    body = b'{"status":"down","score":NaN}'
    entry, _ = _make_custom_entry(_make_streaming_response(body, media_type="application/json", status_code=503))

    result = await tester.test_model(entry)

    assert result == {
        "error": "Health check failed",
        "details": {"status_code": 503, "content_type": "application/json", "data": body.decode()},
    }
    JSONResponse(result)


@pytest.mark.asyncio
async def test_test_model_returns_internal_error_on_unexpected_exception() -> None:
    tester = ModelTester()
    entry = _make_registered_model("llm-v2")
    entry.registered_model.endpoint.on_chat_completion = AsyncMock(side_effect=RuntimeError("boom"))

    result = await tester.test_model(entry)

    assert result["error"] == "Internal Server Error"


@pytest.mark.asyncio
async def test_test_model_binary_response_details_shown_as_binary() -> None:
    tester = ModelTester()
    entry = _make_registered_model("llm-v2")

    async def gen():
        yield b"\x80\x81\x82"

    entry.registered_model.endpoint.on_chat_completion = AsyncMock(
        return_value=StreamingResponse(gen(), media_type="application/json", status_code=422)
    )

    result = await tester.test_model(entry)

    assert result["details"]["data"] == "binary"


# ---------------------------------------------------------------------------
# MCP helpers
# ---------------------------------------------------------------------------


def _make_mcp_json_response(data: Any, status_code: int = 200, extra_headers: dict[str, str] | None = None) -> StreamingResponse:
    body = json.dumps(data).encode("utf-8")

    async def gen():
        yield body

    headers = extra_headers or {}
    return StreamingResponse(gen(), media_type="application/json", status_code=status_code, headers=headers)


def _make_mcp_sse_response(events: list[Any], status_code: int = 200) -> StreamingResponse:
    lines = "".join(f"data: {json.dumps(e)}\n\n" for e in events).encode("utf-8")

    async def gen():
        yield lines

    return StreamingResponse(gen(), media_type="text/event-stream", status_code=status_code)


def _make_mcp_registered_model() -> RegistryEntry:
    endpoint = McpEndpoint(on_request=AsyncMock())
    registered = MagicMock(spec=RegisteredModel)
    registered.name = "my-mcp-server"
    registered.type = "mcp"
    registered.endpoint = endpoint
    entry = MagicMock(spec=RegistryEntry)
    entry.registered_model = registered
    return entry


_INIT_RESULT = {
    "jsonrpc": "2.0",
    "id": 1,
    "result": {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "serverInfo": {"name": "test-server", "version": "0.1"},
    },
}

_TOOLS_RESULT = {
    "jsonrpc": "2.0",
    "id": 2,
    "result": {
        "tools": [{"name": "ping", "description": "Ping the server"}],
    },
}


# ---------------------------------------------------------------------------
# Tests for _read_mcp_response
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_read_mcp_response_parses_json() -> None:
    tester = ModelTester()
    resp = _make_mcp_json_response(_INIT_RESULT)

    (_, data) = await tester._read_mcp_response(resp, 1)  # pyright: ignore[reportPrivateUsage]

    assert data["result"]["serverInfo"]["name"] == "test-server"


@pytest.mark.asyncio
async def test_read_mcp_response_parses_sse_single_event() -> None:
    tester = ModelTester()
    resp = _make_mcp_sse_response([_INIT_RESULT])

    (_, data) = await tester._read_mcp_response(resp, 1)  # pyright: ignore[reportPrivateUsage]

    assert data["id"] == 1
    assert data["result"]["protocolVersion"] == "2024-11-05"


@pytest.mark.asyncio
async def test_read_mcp_response_parses_sse_multiple_events_returns_matching_id() -> None:
    tester = ModelTester()
    other_event = {"jsonrpc": "2.0", "id": 99, "result": {}}
    resp = _make_mcp_sse_response([other_event, _INIT_RESULT])

    (_, data) = await tester._read_mcp_response(resp, 1)  # pyright: ignore[reportPrivateUsage]

    assert data["id"] == 1


@pytest.mark.asyncio
async def test_read_mcp_response_raises_when_no_matching_sse_event() -> None:
    tester = ModelTester()
    resp = _make_mcp_sse_response([{"jsonrpc": "2.0", "id": 99, "result": {}}])

    with pytest.raises(ModelTestError) as exc_info:
        await tester._read_mcp_response(resp, 1)  # pyright: ignore[reportPrivateUsage]

    assert "SSE" in exc_info.value.error


@pytest.mark.asyncio
async def test_read_mcp_response_raises_on_unsupported_content_type() -> None:
    tester = ModelTester()

    async def gen():
        yield b"hello"

    resp = StreamingResponse(gen(), media_type="text/plain", status_code=200)

    with pytest.raises(ModelTestError) as exc_info:
        await tester._read_mcp_response(resp, 1)  # pyright: ignore[reportPrivateUsage]

    assert "content type" in exc_info.value.error.lower()


# ---------------------------------------------------------------------------
# Tests for _test_mcp
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_test_mcp_sends_initialize_with_correct_payload() -> None:
    tester = ModelTester()
    on_request = AsyncMock(side_effect=[_make_mcp_json_response(_INIT_RESULT), _make_mcp_json_response(_TOOLS_RESULT)])
    endpoint = McpEndpoint(on_request=on_request)

    await tester._test_mcp("model", endpoint)  # pyright: ignore[reportPrivateUsage]

    first_call_request = on_request.call_args_list[0][0][0]
    body = await first_call_request.body()
    payload = json.loads(body)
    assert payload["method"] == "initialize"
    assert payload["id"] == 1
    assert "protocolVersion" in payload["params"]
    assert first_call_request.headers.get("content-type") == "application/json"
    assert "text/event-stream" in first_call_request.headers.get("accept", "")


@pytest.mark.asyncio
async def test_test_mcp_passes_session_id_to_tools_list() -> None:
    tester = ModelTester()
    init_resp = _make_mcp_json_response(_INIT_RESULT, extra_headers={"mcp-session-id": "sess-abc"})
    on_request = AsyncMock(side_effect=[init_resp, _make_mcp_json_response(_TOOLS_RESULT)])
    endpoint = McpEndpoint(on_request=on_request)

    await tester._test_mcp("model", endpoint)  # pyright: ignore[reportPrivateUsage]

    second_call_request = on_request.call_args_list[1][0][0]
    assert second_call_request.headers.get("mcp-session-id") == "sess-abc"


@pytest.mark.asyncio
async def test_test_mcp_no_session_id_when_server_omits_it() -> None:
    tester = ModelTester()
    on_request = AsyncMock(side_effect=[_make_mcp_json_response(_INIT_RESULT), _make_mcp_json_response(_TOOLS_RESULT)])
    endpoint = McpEndpoint(on_request=on_request)

    await tester._test_mcp("model", endpoint)  # pyright: ignore[reportPrivateUsage]

    second_call_request = on_request.call_args_list[1][0][0]
    assert second_call_request.headers.get("mcp-session-id") is None


@pytest.mark.asyncio
async def test_test_mcp_returns_ok_with_tools_and_server_info() -> None:
    tester = ModelTester()
    on_request = AsyncMock(side_effect=[_make_mcp_json_response(_INIT_RESULT), _make_mcp_json_response(_TOOLS_RESULT)])
    endpoint = McpEndpoint(on_request=on_request)

    result = await tester._test_mcp("model", endpoint)  # pyright: ignore[reportPrivateUsage]

    assert result["result"] == "ok"
    assert result["output"]["tools"] == [{"name": "ping", "description": "Ping the server"}]  # type: ignore[index]
    assert result["output"]["server_info"]["name"] == "test-server"  # type: ignore[index]


@pytest.mark.asyncio
async def test_test_mcp_returns_tools_error_when_tools_list_fails() -> None:
    tester = ModelTester()

    async def gen():
        yield b"bad"

    tools_fail_resp = StreamingResponse(gen(), media_type="application/json", status_code=500)
    on_request = AsyncMock(side_effect=[_make_mcp_json_response(_INIT_RESULT), tools_fail_resp])
    endpoint = McpEndpoint(on_request=on_request)

    result = await tester._test_mcp("model", endpoint)  # pyright: ignore[reportPrivateUsage]

    assert result["result"] == "ok"
    assert result["output"]["tools"] is None  # type: ignore[index]
    assert "tools_error" in result["output"]  # type: ignore[index]
    assert result["output"]["server_info"]["name"] == "test-server"  # type: ignore[index]


@pytest.mark.asyncio
async def test_test_model_dispatches_to_test_mcp_for_mcp_type() -> None:
    tester = ModelTester()
    on_request = AsyncMock(side_effect=[_make_mcp_json_response(_INIT_RESULT), _make_mcp_json_response(_TOOLS_RESULT)])
    entry = _make_mcp_registered_model()
    entry.registered_model.endpoint = McpEndpoint(on_request=on_request)

    result = await tester.test_model(entry)

    assert result["result"] == "ok"
    assert "tools" in result["output"]  # type: ignore[index]


@pytest.mark.asyncio
async def test_read_mcp_response_raises_json_parse_error_on_invalid_json() -> None:
    tester = ModelTester()

    async def gen():
        yield b"not-valid-json"

    resp = StreamingResponse(gen(), media_type="application/json", status_code=200)

    with pytest.raises(ModelTestError) as exc_info:
        await tester._read_mcp_response(resp, 1)  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.error == "JSON parse error"


@pytest.mark.asyncio
async def test_read_mcp_response_raises_on_non_utf8_sse_body() -> None:
    tester = ModelTester()

    async def gen():
        yield b"\x80\x81\x82"

    resp = StreamingResponse(gen(), media_type="text/event-stream", status_code=200)

    with pytest.raises(ModelTestError) as exc_info:
        await tester._read_mcp_response(resp, 1)  # pyright: ignore[reportPrivateUsage]

    assert "SSE" in exc_info.value.error


@pytest.mark.asyncio
async def test_read_mcp_response_skips_malformed_sse_data_line() -> None:
    tester = ModelTester()
    body = b"data: not-valid-json\n\ndata: " + json.dumps(_INIT_RESULT).encode() + b"\n\n"

    async def gen():
        yield body

    resp = StreamingResponse(gen(), media_type="text/event-stream", status_code=200)

    (_, data) = await tester._read_mcp_response(resp, 1)  # pyright: ignore[reportPrivateUsage]

    assert data["id"] == 1


@pytest.mark.asyncio
async def test_test_mcp_non_streaming_init_response_session_id_is_none() -> None:
    tester = ModelTester()

    class _McpModel(BaseModel):
        payload: dict[str, Any]

        def model_dump_json(self, **kwargs: Any) -> str:  # type: ignore[override]
            return json.dumps(self.payload)

    on_request = AsyncMock(side_effect=[_McpModel(payload=_INIT_RESULT), _make_mcp_json_response(_TOOLS_RESULT)])
    endpoint = McpEndpoint(on_request=on_request)

    result = await tester._test_mcp("model", endpoint)  # pyright: ignore[reportPrivateUsage]

    assert result["result"] == "ok"
    second_request = on_request.call_args_list[1][0][0]
    assert second_request.headers.get("mcp-session-id") is None


@pytest.mark.asyncio
async def test_test_mcp_raises_when_initialize_returns_jsonrpc_error() -> None:
    tester = ModelTester()
    error_resp = _make_mcp_json_response({"jsonrpc": "2.0", "id": 1, "error": {"code": -32601, "message": "Method not found"}})
    on_request = AsyncMock(return_value=error_resp)
    endpoint = McpEndpoint(on_request=on_request)

    result = await tester.test_model(MagicMock(registered_model=MagicMock(type="mcp", name="x", endpoint=endpoint)))

    assert "error" in result


@pytest.mark.asyncio
async def test_test_mcp_raises_when_initialize_returns_no_result() -> None:
    tester = ModelTester()
    no_result_resp = _make_mcp_json_response({"jsonrpc": "2.0", "id": 1})
    on_request = AsyncMock(return_value=no_result_resp)
    endpoint = McpEndpoint(on_request=on_request)

    result = await tester.test_model(MagicMock(registered_model=MagicMock(type="mcp", name="x", endpoint=endpoint)))

    assert "error" in result


def _sid_init_response() -> StreamingResponse:
    return _make_mcp_json_response(_INIT_RESULT, extra_headers={"mcp-session-id": "sess-abc"})


@pytest.mark.asyncio
async def test_test_mcp_terminates_session_after_success() -> None:
    tester = ModelTester()
    on_request = AsyncMock(side_effect=[_sid_init_response(), _make_mcp_json_response(_TOOLS_RESULT), _make_mcp_json_response({})])
    endpoint = McpEndpoint(on_request=on_request)

    result = await tester._test_mcp("model", endpoint)  # pyright: ignore[reportPrivateUsage]

    assert result["output"]["tools"] == _TOOLS_RESULT["result"]["tools"]  # type: ignore[index]
    assert on_request.call_count == 3
    delete_request = on_request.call_args_list[2][0][0]
    assert delete_request.method == "DELETE"
    assert delete_request.headers.get("mcp-session-id") == "sess-abc"


@pytest.mark.asyncio
async def test_test_mcp_terminates_session_after_tools_list_error() -> None:
    tester = ModelTester()

    async def gen():
        yield b"bad"

    tools_fail_resp = StreamingResponse(gen(), media_type="application/json", status_code=500)
    on_request = AsyncMock(side_effect=[_sid_init_response(), tools_fail_resp, _make_mcp_json_response({})])
    endpoint = McpEndpoint(on_request=on_request)

    result = await tester._test_mcp("model", endpoint)  # pyright: ignore[reportPrivateUsage]

    assert result["output"]["tools"] is None  # type: ignore[index]
    assert "tools_error" in result["output"]  # type: ignore[index]
    delete_request = on_request.call_args_list[2][0][0]
    assert delete_request.method == "DELETE"
    assert delete_request.headers.get("mcp-session-id") == "sess-abc"


@pytest.mark.asyncio
async def test_test_mcp_terminates_session_when_initialize_fails() -> None:
    tester = ModelTester()
    init_resp = _make_mcp_json_response({"jsonrpc": "2.0", "id": 1}, extra_headers={"mcp-session-id": "sess-abc"})
    on_request = AsyncMock(side_effect=[init_resp, _make_mcp_json_response({})])
    endpoint = McpEndpoint(on_request=on_request)

    with pytest.raises(ModelTestError):
        await tester._test_mcp("model", endpoint)  # pyright: ignore[reportPrivateUsage]

    delete_request = on_request.call_args_list[1][0][0]
    assert delete_request.method == "DELETE"
    assert delete_request.headers.get("mcp-session-id") == "sess-abc"


@pytest.mark.asyncio
async def test_test_mcp_sends_no_delete_without_session_id() -> None:
    tester = ModelTester()
    on_request = AsyncMock(side_effect=[_make_mcp_json_response(_INIT_RESULT), _make_mcp_json_response(_TOOLS_RESULT)])
    endpoint = McpEndpoint(on_request=on_request)

    await tester._test_mcp("model", endpoint)  # pyright: ignore[reportPrivateUsage]

    assert on_request.call_count == 2
    assert all(c[0][0].method == "POST" for c in on_request.call_args_list)


@pytest.mark.asyncio
async def test_test_mcp_failing_delete_keeps_result() -> None:
    tester = ModelTester()
    on_request = AsyncMock(side_effect=[_sid_init_response(), _make_mcp_json_response(_TOOLS_RESULT), RuntimeError("boom")])
    endpoint = McpEndpoint(on_request=on_request)

    result = await tester._test_mcp("model", endpoint)  # pyright: ignore[reportPrivateUsage]

    assert result["result"] == "ok"
    assert result["output"]["tools"] == _TOOLS_RESULT["result"]["tools"]  # type: ignore[index]
    assert on_request.call_count == 3


@pytest.mark.asyncio
async def test_test_mcp_terminates_session_when_tools_list_raises() -> None:
    tester = ModelTester()
    on_request = AsyncMock(side_effect=[_sid_init_response(), RuntimeError("boom"), _make_mcp_json_response({})])
    endpoint = McpEndpoint(on_request=on_request)

    with pytest.raises(RuntimeError, match="boom"):
        await tester._test_mcp("model", endpoint)  # pyright: ignore[reportPrivateUsage]

    assert on_request.call_count == 3
    delete_request = on_request.call_args_list[2][0][0]
    assert delete_request.method == "DELETE"
    assert delete_request.headers.get("mcp-session-id") == "sess-abc"
