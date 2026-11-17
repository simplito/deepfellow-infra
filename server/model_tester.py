# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Model tester."""

import asyncio
import base64
import io
import json
import wave
from typing import Any, NamedTuple, NoReturn, cast

import aiohttp
from fastapi import HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from starlette.datastructures import Headers
from starlette.requests import Request

from server.endpointregistry import ChatCompletionEndpoint, CustomEndpoint, McpEndpoint, RegistryEntry, SimpleEndpoint
from server.models.api import (
    ALL_LLM_SUFFIXES,
    ChatCompletionRequest,
    CompletionLegacyRequest,
    CreateSpeechRequest,
    CreateTranscriptionRequest,
    EmbeddingRequest,
    ImagesRequest,
    MessageParam,
    MessagesRequest,
    RerankRequest,
    ResponsesRequest,
    SystemOneRequest,
    ThinkingConfigDisabled,
    UserMessage,
)
from server.models.common import JsonSerializable, StarletteResponse
from server.utils.logger import logger

LLM_TEST_HANDLERS = {
    "v3": "_test_responses",
    "ant": "_test_messages",
    "v2": "_test_chat_completions",
    "v1": "_test_completions",
}
# Health routes probed in order for custom models: ours serve /health, BentoML serves /healthz.
CUSTOM_HEALTH_PATHS = ("health", "healthz")
# Optional self-test route a custom model image may expose. Contract: GET without body, side-effect free and cheap
# (the /custom proxy exposes every container route to API-key holders, so it can be called repeatedly), returns 200 when
# the model works and a short text/JSON verdict as the body (no media, no secrets).
CUSTOM_TEST_PATH = "test_model"
# Caps on the custom checks: their bodies are short verdicts, and a buggy image must not stream into Infra's memory or hang the test.
CUSTOM_TEST_MAX_BYTES = 64 * 1024
CUSTOM_TEST_TIMEOUT_SECONDS = 30


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("Duplicate JSON keys")
    return result


def _reject_constant(name: str) -> NoReturn:
    raise ValueError(f"Non-standard JSON constant {name}")  # noqa: EM102


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


class ModelTestError(Exception):
    def __init__(self, error: str, response: Response | None = None):
        super().__init__()
        self.error = error
        self.response = response


def _as_json_if_lossless(response: Response | None, text: str | None) -> object:
    """Return a declared-JSON body parsed so the UI shows it as structure, else the verbatim text.

    Falls back to text whenever parsing would alter the body: duplicate keys (all but the last would be dropped)
    or NaN/Infinity (not valid JSON, and the JSON response could not serialize them).
    """
    if response is None or response.media_type != "application/json":
        return text
    try:
        return json.loads(cast("str", text), object_pairs_hook=_reject_duplicate_keys, parse_constant=_reject_constant)
    except ValueError:
        return text


class ModelTester:
    async def test_model(self, entry: RegistryEntry) -> JsonSerializable:
        """Test model."""
        try:
            return await self._perform_test(entry)
        except ModelTestError as e:
            result: dict[str, Any] = {"error": e.error}
            if e.response:
                details: dict[str, object] = {
                    "status_code": e.response.status_code,
                    "content_type": e.response.media_type,
                }
                try:
                    details["data"] = _as_json_if_lossless(e.response, e.response.content.decode("utf-8"))
                except UnicodeDecodeError:
                    details["data"] = "binary"
                result["details"] = details
            return result
        except Exception:
            logger.exception("Error during executing model test")
            return {"error": "Internal Server Error"}

    async def _perform_test(self, entry: RegistryEntry) -> JsonSerializable:  # noqa: C901
        if entry.registered_model.type == "llm" or entry.registered_model.type.startswith("llm-"):
            parts = set(ALL_LLM_SUFFIXES) if entry.registered_model.type == "llm" else set(entry.registered_model.type.split("-")[1:])
            for suffix, method_name in LLM_TEST_HANDLERS.items():
                if suffix in parts:
                    method = getattr(self, method_name)
                    return await method(entry.registered_model.name, entry.registered_model.endpoint)
        if entry.registered_model.type == "embedding":
            return await self._test_embedding(entry.registered_model.name, entry.registered_model.endpoint)
        if entry.registered_model.type == "tts":
            return await self._test_tts(entry.registered_model.name, entry.registered_model.endpoint)
        if entry.registered_model.type == "stt":
            return await self._test_stt(entry.registered_model.name, entry.registered_model.endpoint)
        if entry.registered_model.type == "txt2img":
            return await self._test_txt2img(entry.registered_model.name, entry.registered_model.endpoint)
        if entry.registered_model.type == "rerank":
            return await self._test_rerank(entry.registered_model.name, entry.registered_model.endpoint)
        if entry.registered_model.type == "decision":
            return await self._test_systemone(entry.registered_model.name, entry.registered_model.endpoint)
        if entry.registered_model.type == "mcp":
            return await self._test_mcp(entry.registered_model.name, entry.registered_model.endpoint)
        if entry.registered_model.type == "custom":
            return await self._test_custom(entry.registered_model.name, entry.registered_model.endpoint)
        raise RuntimeError(f"Given model cannot be tested, unsupported type {entry.registered_model.type}")  # noqa: EM102

    async def _test_messages(self, model: str, endpoint: ChatCompletionEndpoint) -> JsonSerializable:
        if not endpoint.on_messages:
            raise RuntimeError("LLM has no on_messages callback")
        (my_resp, json) = await self._read_json(
            await endpoint.on_messages(
                MessagesRequest(
                    model=model,
                    max_tokens=40,
                    messages=[MessageParam(role="user", content="Only say `hello!`")],
                    thinking=ThinkingConfigDisabled(type="disabled"),
                ),
                None,
            )
        )
        try:
            return {"result": "ok", "output": json["content"][0].get("text") or json["content"][0].get("thinking"), "details": json}
        except Exception:
            raise ModelTestError("Cannot read message content", my_resp)  # noqa: B904

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
            raise ModelTestError("Cannot read message content", my_resp)  # noqa: B904

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
            raise ModelTestError("Cannot read message content", my_resp)  # noqa: B904

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
            raise ModelTestError("Cannot read message content", my_resp)  # noqa: B904

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
            raise ModelTestError("Cannot read response text", my_resp)  # noqa: B904

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
            raise ModelTestError("Cannot read image data", my_resp)  # noqa: B904

    async def _test_rerank(self, model: str, endpoint: SimpleEndpoint[RerankRequest]) -> JsonSerializable:
        (my_resp, json) = await self._read_json(
            await endpoint.on_request(
                RerankRequest(
                    model=model,
                    query="What is a cat?",
                    return_documents=True,
                    documents=[
                        "Water is wet.",
                        "The Yeti is a legendary creature.",
                        "A cat is an animal. It is a mammal with a tail.",
                    ],
                ),
                None,
            )
        )
        try:
            return {
                "result": "ok",
                "output": json["results"][0]["document"]["text"],
                "details": json,
            }
        except Exception:
            raise ModelTestError("Cannot read image data", my_resp)  # noqa: B904

    async def _test_custom(self, prefix: str, endpoint: CustomEndpoint) -> JsonSerializable:
        """Run the health probe and the optional self-test; at least one must exist and every existing one must pass."""
        try:
            async with asyncio.timeout(CUSTOM_TEST_TIMEOUT_SECONDS):
                health = await self._probe_custom_health(prefix, endpoint)
                test = await self._get_custom(prefix, endpoint, CUSTOM_TEST_PATH)
        except TimeoutError:
            raise ModelTestError(f"Model test timed out (limit {CUSTOM_TEST_TIMEOUT_SECONDS} s)") from None  # noqa: EM102
        if test.status_code == 404:
            test = None
        elif test.status_code != 200:
            raise ModelTestError("Model test failed", test)
        if not health and not test:
            paths = ", ".join(f"/{path}" for path in (*CUSTOM_HEALTH_PATHS, CUSTOM_TEST_PATH))
            raise ModelTestError(f"No health or test endpoint found (tried: {paths})")  # noqa: EM102
        health_output = health.content.decode("utf-8", errors="replace") if health else None
        test_output = test.content.decode("utf-8", errors="replace") if test else None
        return {
            "result": "ok",
            "output": (test_output or "passed") if test else (health_output or "healthy"),
            "details": {
                "health": _as_json_if_lossless(health, health_output),
                "test_model": _as_json_if_lossless(test, test_output),
            },
        }

    async def _probe_custom_health(self, prefix: str, endpoint: CustomEndpoint) -> Response | None:
        """Return the first health response that isn't 404, or None when the model defines no health route."""
        for path in CUSTOM_HEALTH_PATHS:
            health = await self._get_custom(prefix, endpoint, path)
            # Only a missing route falls through; any other non-200 is the service reporting itself unhealthy.
            if health.status_code == 404:
                continue
            if health.status_code != 200:
                raise ModelTestError("Health check failed", health)
            return health
        return None

    async def _get_custom(self, prefix: str, endpoint: CustomEndpoint, path: str) -> Response:
        try:
            response = await endpoint.on_request(self._make_request(b"", method="GET", full_path=f"{prefix}/{path}"))
        except HTTPException as e:
            # The proxy (`make_upstream_request`) answers 502/504 for an unreachable or slow upstream; any other status isn't ours to hide.
            if e.status_code == 504:
                raise ModelTestError("Model timed out") from e
            if e.status_code == 502:
                raise ModelTestError(f"Model unreachable: {e.detail}") from e  # noqa: EM102
            raise
        except aiohttp.ClientError as e:
            # Typically a container that is still starting: show it instead of a generic internal error.
            raise ModelTestError(f"Model unreachable: {e}") from e  # noqa: EM102
        return await self._read_response(response, max_bytes=CUSTOM_TEST_MAX_BYTES)

    def _make_request(
        self, body: bytes, extra_headers: list[tuple[bytes, bytes]] | None = None, method: str = "POST", full_path: str = "/"
    ) -> Request:
        headers: list[tuple[bytes, bytes]] = [
            (b"content-type", b"application/json"),
            (b"accept", b"application/json, text/event-stream"),
            *(extra_headers or []),
        ]
        scope: dict[str, Any] = {
            "type": "http",
            "method": method,
            "path": "/",
            "query_string": b"",
            "scheme": "http",
            "server": ("localhost", 80),
            "headers": headers,
            "path_params": {"full_path": full_path},
        }

        async def receive() -> dict[str, Any]:
            return {"type": "http.request", "body": body, "more_body": False}

        return Request(scope, receive)

    async def _read_mcp_response(self, response: StarletteResponse, request_id: int) -> tuple[Response, Any]:
        my_resp = await self._read_response(response)
        if my_resp.status_code != 200:
            raise ModelTestError(error="Invalid status code", response=my_resp)
        if my_resp.media_type == "application/json":
            try:
                data = json.loads(my_resp.content.decode("utf-8"))
                return (my_resp, data)  # noqa: TRY300
            except Exception:
                raise ModelTestError(error="JSON parse error", response=my_resp)  # noqa: B904
        if "text/event-stream" in my_resp.media_type:
            try:
                text = my_resp.content.decode("utf-8")
            except Exception:
                raise ModelTestError(error="Cannot decode SSE response", response=my_resp)  # noqa: B904
            for line in text.splitlines():
                if line.startswith("data: "):
                    try:
                        event_data = json.loads(line[6:])
                        if isinstance(event_data, dict) and event_data.get("id") == request_id:
                            return (my_resp, event_data)
                    except (json.JSONDecodeError, AttributeError):
                        continue
            raise ModelTestError("No matching MCP response in SSE stream", my_resp)
        raise ModelTestError(f"Unexpected content type: {my_resp.media_type}", my_resp)  # noqa: EM102

    async def _test_mcp(self, model: str, endpoint: McpEndpoint) -> JsonSerializable:  # noqa: ARG002
        init_payload = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "deepfellow-tester", "version": "1.0"},
                },
            }
        ).encode()
        init_response = await endpoint.on_request(self._make_request(init_payload))
        session_id: str | None = None
        if isinstance(init_response, StreamingResponse):
            session_id = init_response.headers.get("mcp-session-id")
        try:
            (init_raw, init_data) = await self._read_mcp_response(init_response, 1)
            if "error" in init_data:
                raise ModelTestError(f"MCP initialize error: {init_data['error']}", init_raw)  # noqa: EM102
            if "result" not in init_data:
                raise ModelTestError("MCP initialize returned no result", init_raw)
            server_info: dict[str, Any] = init_data["result"].get("serverInfo", {})

            extra_headers: list[tuple[bytes, bytes]] = [(b"mcp-session-id", session_id.encode())] if session_id else []
            tools_payload = json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/list",
                    "params": {},
                }
            ).encode()
            try:
                tools_response = await endpoint.on_request(self._make_request(tools_payload, extra_headers))
                (_, tools_data) = await self._read_mcp_response(tools_response, 2)
                tools: list[Any] | None = tools_data.get("result", {}).get("tools")
            except ModelTestError as e:
                return {
                    "result": "ok",
                    "output": {"tools": None, "tools_error": e.error, "server_info": server_info},
                    "details": {"initialize": init_data},
                }
            else:
                return {
                    "result": "ok",
                    "output": {"tools": tools, "server_info": server_info},
                    "details": {"initialize": init_data, "tools_list": tools_data},
                }
        finally:
            if session_id:
                await self._terminate_mcp_session(endpoint, session_id)

    async def _terminate_mcp_session(self, endpoint: McpEndpoint, session_id: str) -> None:
        """Best-effort DELETE of the MCP session opened by the test; failures are logged and ignored."""
        try:
            response = await endpoint.on_request(self._make_request(b"", [(b"mcp-session-id", session_id.encode())], method="DELETE"))
            await self._read_response(response)
        except Exception as exc:
            logger.debug("MCP session termination in model test failed: %s: %s", type(exc).__name__, exc)

    async def _test_systemone(self, model: str, endpoint: SimpleEndpoint[SystemOneRequest]) -> JsonSerializable:
        (my_resp, json) = await self._read_json(
            await endpoint.on_request(
                SystemOneRequest.model_validate(
                    {
                        "model": model,
                        "state": "Our checkout has returned 500 errors since 9am.",
                        "questions": {
                            "label": {
                                "type": "choice",
                                "instructions": "Which label fits this ticket?",
                                "criteria": {
                                    "billing": "Payments and refunds",
                                    "bug": "Software errors",
                                    "account": "Login and account access",
                                },
                            }
                        },
                    }
                ),
                None,
            )
        )
        try:
            return {
                "result": "ok",
                "output": json["answers"]["label"]["choice"],
                "details": json,
            }
        except Exception:
            raise ModelTestError("Cannot read decision answer", my_resp)  # noqa: B904

    async def _read_json(self, response: StarletteResponse) -> tuple[Response, Any]:
        my_resp = await self._read_response(response)
        if my_resp.status_code != 200:
            raise ModelTestError(error="Invalid status code", response=my_resp)
        if my_resp.media_type != "application/json":
            raise ModelTestError(error="Invalid content type", response=my_resp)
        try:
            text = my_resp.content.decode("utf-8")
            try:
                data = json.loads(text)
                return (my_resp, data)  # noqa: TRY300
            except Exception:
                raise ModelTestError(error="JSON parse error", response=my_resp)  # noqa: B904
        except Exception:
            raise ModelTestError(error="Cannot read content as text", response=my_resp)  # noqa: B904

    async def _read_successfully(self, response: StarletteResponse) -> Response:
        my_resp = await self._read_response(response)
        if my_resp.status_code != 200:
            raise ModelTestError(error="Invalid status code", response=my_resp)
        return my_resp

    async def _read_response(self, response: StarletteResponse, max_bytes: int | None = None) -> Response:
        """Read response and return content and content type, failing once the body exceeds `max_bytes`."""
        if isinstance(response, StreamingResponse):
            content = bytearray()
            async for chunk in response.body_iterator:
                if isinstance(chunk, bytes):
                    content += chunk
                else:
                    raise TypeError("There is no bytes in StreamingResponse")
                if max_bytes is not None and len(content) > max_bytes:
                    # Close the stream so the proxied upstream connection is released instead of left half-read.
                    aclose = getattr(response.body_iterator, "aclose", None)
                    if aclose:
                        await aclose()
                    raise ModelTestError(f"Response too large (limit {max_bytes // 1024} KB)")  # noqa: EM102
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
