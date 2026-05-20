# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from server.endpointregistry import ChatCompletionEndpoint, RegisteredModel, RegistryEntry, SimpleEndpoint
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


@pytest.mark.asyncio
async def test_perform_test_raises_test_error_for_custom() -> None:
    tester = ModelTester()
    entry = _make_registered_model("custom")

    with pytest.raises(ModelTestError) as exc_info:
        await tester._perform_test(entry)  # pyright: ignore[reportPrivateUsage]

    assert "Custom" in exc_info.value.error


@pytest.mark.asyncio
async def test_perform_test_raises_runtime_error_for_unknown_type() -> None:
    tester = ModelTester()
    entry = _make_registered_model("unknown-type")

    with pytest.raises(RuntimeError, match="unsupported type"):
        await tester._perform_test(entry)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_test_model_returns_error_on_test_error() -> None:
    tester = ModelTester()
    entry = _make_registered_model("custom")

    result = await tester.test_model(entry)

    assert "error" in result


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
