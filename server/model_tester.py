# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Model tester."""

import base64
import io
import json
import logging
import wave
from typing import Any, NamedTuple

from fastapi import UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from starlette.datastructures import Headers

from server.endpointregistry import ChatCompletionEndpoint, RegistryEntry, SimpleEndpoint
from server.models.api import (
    ChatCompletionRequest,
    CompletionLegacyRequest,
    CreateSpeechRequest,
    CreateTranscriptionRequest,
    EmbeddingRequest,
    ImagesRequest,
    ResponsesRequest,
    UserMessage,
)
from server.models.common import JsonSerializable, StarletteResponse

logger = logging.getLogger("uvicorn.error")


class Response(NamedTuple):
    status_code: int
    media_type: str
    content: bytes


class UnpackedResponse(NamedTuple):
    status_code: int
    media_type: str
    content: bytes
    json: Any
    json_error: str | None


class TestError(Exception):
    def __init__(self, error: str, response: Response | None = None):
        super().__init__()
        self.error = error
        self.response = response


class ModelTester:
    async def test_model(self, entry: RegistryEntry) -> JsonSerializable:
        """Test model."""
        try:
            return await self._perform_test(entry)
        except TestError as e:
            result: dict[str, Any] = {"error": e.error}
            if e.response:
                details = {
                    "status_code": e.response.status_code,
                    "content_type": e.response.media_type,
                }
                details["data"] = "binary"
                try:
                    details["data"] = e.response.content.decode("utf-8")
                    details["data"] = json.loads(details["data"])
                except Exception:
                    pass
                result["details"] = details
            return result
        except Exception:
            logger.exception("Error during executing model test")
            return {"error": "Internal Server Error"}

    async def _perform_test(self, entry: RegistryEntry) -> JsonSerializable:
        if entry.registered_model.type in ["llm", "llm-v3", "llm-v2-v3", "llm-v1-v3"]:
            return await self._test_responses(entry.registered_model.name, entry.registered_model.endpoint)
        if entry.registered_model.type in ["llm-v2", "llm-v1-v2"]:
            return await self._test_chat_completions(entry.registered_model.name, entry.registered_model.endpoint)
        if entry.registered_model.type == "llm-v1":
            return await self._test_completions(entry.registered_model.name, entry.registered_model.endpoint)
        if entry.registered_model.type == "embedding":
            return await self._test_embedding(entry.registered_model.name, entry.registered_model.endpoint)
        if entry.registered_model.type == "tts":
            return await self._test_tts(entry.registered_model.name, entry.registered_model.endpoint)
        if entry.registered_model.type == "stt":
            return await self._test_stt(entry.registered_model.name, entry.registered_model.endpoint)
        if entry.registered_model.type == "txt2img":
            return await self._test_txt2img(entry.registered_model.name, entry.registered_model.endpoint)
        if entry.registered_model.type == "custom":
            raise TestError("Custom model cannot be tested")
        raise RuntimeError(f"Given model cannot be tested, unsupported type {entry.registered_model.type}")  # noqa: EM102

    async def _test_responses(self, model: str, endpoint: ChatCompletionEndpoint) -> JsonSerializable:
        if not endpoint.on_responses:
            raise RuntimeError("LLM has no on_responses callback")
        (my_resp, json) = await self._read_json(
            await endpoint.on_responses(
                ResponsesRequest(model=model, max_output_tokens=50, input="Say hello!"),
                None,
            )
        )
        try:
            for part in json["output"][::-1]:
                if "summary" in part:
                    if len(part["summary"]):
                        return {"result": "ok", "output": part["summary"][0]["text"], "details": json}
                    return {"result": "ok", "output": "", "details": json}
                if "content" in part:
                    return {"result": "ok", "output": part["content"][0]["text"], "details": json}
            raise RuntimeError("No content output")  # noqa: TRY301
        except Exception:
            raise TestError("Cannot read message content", my_resp)  # noqa: B904

    async def _test_chat_completions(self, model: str, endpoint: ChatCompletionEndpoint) -> JsonSerializable:
        if not endpoint.on_chat_completion:
            raise RuntimeError("LLM has no on_chat_completion callback")
        (my_resp, json) = await self._read_json(
            await endpoint.on_chat_completion(
                ChatCompletionRequest(model=model, max_completion_tokens=50, messages=[UserMessage(role="user", content="Say hello!")]),
                None,
            )
        )
        try:
            return {"result": "ok", "output": json["choices"][0]["message"]["content"], "details": json}
        except Exception:
            raise TestError("Cannot read message content", my_resp)  # noqa: B904

    async def _test_completions(self, model: str, endpoint: ChatCompletionEndpoint) -> JsonSerializable:
        if not endpoint.on_completion:
            raise RuntimeError("LLM has no on_completion callback")
        (my_resp, json) = await self._read_json(
            await endpoint.on_completion(
                CompletionLegacyRequest(model=model, max_tokens=50, prompt="Say hello!"),
                None,
            )
        )
        try:
            return {"result": "ok", "output": json["choices"][0]["text"], "details": json}
        except Exception:
            raise TestError("Cannot read message content", my_resp)  # noqa: B904

    async def _test_embedding(self, model: str, endpoint: SimpleEndpoint[EmbeddingRequest]) -> JsonSerializable:
        (_, json) = await self._read_json(
            await endpoint.on_request(
                EmbeddingRequest(model=model, input="Say hello!"),
                None,
            )
        )
        return {"result": "ok", "details": json}

    async def _test_tts(self, model: str, endpoint: SimpleEndpoint[CreateSpeechRequest]) -> JsonSerializable:
        my_resp = await self._read_successfully(
            await endpoint.on_request(
                CreateSpeechRequest(model=model, input="Say hello", voice="Adam"),
                None,
            )
        )
        return {"result": "ok", "output": {"content_type": my_resp.media_type, "data": base64.b64encode(my_resp.content).decode()}}

    async def _test_stt(self, model: str, endpoint: SimpleEndpoint[CreateTranscriptionRequest]) -> JsonSerializable:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(16000)  # 16kHz
            wav_file.writeframes(b"\x00" * 3200)  # 0.1 second of silence
        audio_data = buffer.getvalue()
        headers = Headers({"content-type": "audio/wav"})

        (my_resp, json) = await self._read_json(
            await endpoint.on_request(
                CreateTranscriptionRequest(model=model, file=UploadFile(filename="test.wav", file=io.BytesIO(audio_data), headers=headers)),
                None,
            )
        )
        try:
            return {"result": "ok", "output": json["text"], "details": json}
        except Exception:
            raise TestError("Cannot read response text", my_resp)  # noqa: B904

    async def _test_txt2img(self, model: str, endpoint: SimpleEndpoint[ImagesRequest]) -> JsonSerializable:
        (my_resp, json) = await self._read_json(
            await endpoint.on_request(
                ImagesRequest(model=model, prompt="Little cat", size="256x256", quality="low", output_format="png"),
                None,
            )
        )
        try:
            return {
                "result": "ok",
                "output": {"content_type": "image/png", "data": json["data"][0]["b64_json"]},
                "details": json,
            }
        except Exception:
            raise TestError("Cannot read image data", my_resp)  # noqa: B904

    # ====================
    #       HELPERS
    # ====================

    async def _read_json(self, response: StarletteResponse) -> tuple[Response, Any]:
        my_resp = await self._read_response(response)
        if my_resp.status_code != 200:
            raise TestError(error="Invalid status code", response=my_resp)
        if my_resp.media_type != "application/json":
            raise TestError(error="Invalid content type", response=my_resp)
        try:
            text = my_resp.content.decode("utf-8")
            try:
                data = json.loads(text)
                return (my_resp, data)  # noqa: TRY300
            except Exception:
                raise TestError(error="JSON parse error", response=my_resp)  # noqa: B904
        except Exception:
            raise TestError(error="Cannot read content as text", response=my_resp)  # noqa: B904

    async def _read_successfully(self, response: StarletteResponse) -> Response:
        my_resp = await self._read_response(response)
        if my_resp.status_code != 200:
            raise TestError(error="Invalid status code", response=my_resp)
        return my_resp

    async def _read_response(self, response: StarletteResponse) -> Response:
        """Read response and return content and content type."""
        if isinstance(response, StreamingResponse):
            content = bytearray()
            async for chunk in response.body_iterator:
                if isinstance(chunk, bytes):
                    content += chunk
                else:
                    raise TypeError("There is no bytes in StreamingResponse")
            return Response(
                status_code=response.status_code,
                content=bytes(content),
                media_type=response.media_type or "application/octet-stream",
            )
        if isinstance(response, BaseModel):
            return Response(
                status_code=200,
                content=response.model_dump_json().encode("utf-8"),
                media_type="application/json",
            )
        raise RuntimeError(f"Unsupported StarletteResponse type {type(response)}")  # noqa: EM102
