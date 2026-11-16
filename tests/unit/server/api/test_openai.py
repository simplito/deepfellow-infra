# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Tests for server/api/openai.py endpoints."""

from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from starlette.testclient import TestClient

from server.api.openai import router
from server.config import AppSettings
from server.core.dependencies import auth_server, get_config, get_endpoint_registry
from server.error_handlers import register_exception_handlers
from server.models.api import ApiModel, ApiModelProps, ApiModels
from server.utils.exceptions import OpenAIHTTPException


def _make_model_props(alias_of: list[str] | None = None) -> ApiModelProps:
    return ApiModelProps(private=False, type="llm", endpoints=["chat"], alias_of=alias_of or [])


def _make_api_model(model_id: str = "gpt-4", alias_of: list[str] | None = None) -> ApiModel:
    return ApiModel(id=model_id, object="model", created=0, owned_by="local", props=_make_model_props(alias_of))


@pytest.fixture
def registry() -> MagicMock:
    mock = MagicMock()
    mock.execute_messages = AsyncMock(return_value=JSONResponse({"ok": True}))
    mock.execute_responses = AsyncMock(return_value=JSONResponse({"ok": True}))
    mock.execute_chat_completion = AsyncMock(return_value=JSONResponse({"ok": True}))
    mock.execute_completion = AsyncMock(return_value=JSONResponse({"ok": True}))
    mock.execute_ollama_chat = AsyncMock(return_value=JSONResponse({"ok": True}))
    mock.execute_embeddings = AsyncMock(return_value=JSONResponse({"ok": True}))
    mock.execute_audio_speech = AsyncMock(return_value=JSONResponse({"ok": True}))
    mock.execute_audio_transcriptions = AsyncMock(return_value=JSONResponse({"ok": True}))
    mock.execute_images_generations = AsyncMock(return_value=JSONResponse({"ok": True}))
    mock.execute_rerank = AsyncMock(return_value=JSONResponse({"ok": True}))
    mock.execute_systemone = AsyncMock(return_value=JSONResponse({"ok": True}))
    mock.execute_custom_endpoints = AsyncMock(return_value=JSONResponse({"ok": True}))
    mock.execute_mcp_endpoints = AsyncMock(return_value=JSONResponse({"ok": True}))
    return mock


def _make_app(registry: MagicMock) -> FastAPI:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(router)
    app.dependency_overrides[auth_server] = lambda: "test-key"
    app.dependency_overrides[get_endpoint_registry] = lambda: registry
    app.dependency_overrides[get_config] = lambda: AppSettings()  # pyright: ignore[reportCallIssue]
    return app


@pytest.fixture
def client(registry: MagicMock) -> Generator[TestClient]:
    with TestClient(_make_app(registry)) as c:
        yield c


def test_auth_required_returns_403_without_token() -> None:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(router)
    # No dependency overrides — auth_server will reject missing token
    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.get("/v1/models")

    assert resp.status_code in (401, 403, 422)


def test_get_models_returns_compatible_by_default(registry: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    model = _make_api_model("gpt-4")
    registry.get_models = MagicMock(return_value=ApiModels(data=[model]))

    resp = client.get("/v1/models", headers=auth_header)

    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    assert any(m["id"] == "gpt-4" for m in body["data"])
    # compatible response should NOT include props
    assert "props" not in body["data"][0]


def test_get_models_with_additional_data_returns_full_model(registry: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    model = _make_api_model("gpt-4")
    registry.get_models = MagicMock(return_value=ApiModels(data=[model]))

    resp = client.get("/v1/models?additional_data=true", headers=auth_header)

    assert resp.status_code == 200
    body = resp.json()
    assert "props" in body["data"][0]


def test_get_models_empty_list(registry: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    registry.get_models = MagicMock(return_value=ApiModels(data=[]))

    resp = client.get("/v1/models", headers=auth_header)

    assert resp.status_code == 200
    assert resp.json()["data"] == []


def test_get_model_returns_compatible_by_default(registry: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    model = _make_api_model("gpt-4")
    registry.get_model = MagicMock(return_value=model)

    resp = client.get("/v1/models/gpt-4", headers=auth_header)

    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == "gpt-4"
    assert "props" not in body


def test_get_model_with_additional_data_returns_full_model(registry: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    model = _make_api_model("gpt-4")
    registry.get_model = MagicMock(return_value=model)

    resp = client.get("/v1/models/gpt-4?additional_data=true", headers=auth_header)

    assert resp.status_code == 200
    assert "props" in resp.json()


def test_get_models_with_additional_data_includes_alias_of(registry: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    registry.get_models = MagicMock(return_value=ApiModels(data=[_make_api_model("gpt-4"), _make_api_model("smart", ["gpt-4"])]))

    resp = client.get("/v1/models?additional_data=true", headers=auth_header)

    assert resp.status_code == 200
    assert {m["id"]: m["props"]["alias_of"] for m in resp.json()["data"]} == {"gpt-4": [], "smart": ["gpt-4"]}


def test_get_model_with_additional_data_includes_alias_of(registry: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    registry.get_model = MagicMock(return_value=_make_api_model("smart", ["gpt-4"]))

    resp = client.get("/v1/models/smart?additional_data=true", headers=auth_header)

    assert resp.status_code == 200
    assert resp.json()["props"]["alias_of"] == ["gpt-4"]


def test_get_model_passes_model_id_to_registry(registry: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    model = _make_api_model("some-model")
    registry.get_model = MagicMock(return_value=model)

    client.get("/v1/models/some-model", headers=auth_header)

    assert registry.get_model.call_count == 1
    assert registry.get_model.call_args == call("some-model")


def test_get_model_with_path_model_id(registry: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    model = _make_api_model("org/some-model")
    registry.get_model = MagicMock(return_value=model)

    resp = client.get("/v1/models/org/some-model", headers=auth_header)

    assert resp.status_code == 200
    assert registry.get_model.call_count == 1
    assert registry.get_model.call_args == call("org/some-model")


MESSAGES_BODY = {
    "model": "claude-model",
    "messages": [{"role": "user", "content": "Hello"}],
}


def test_post_messages_delegates_to_registry(registry: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    resp = client.post("/v1/messages", json=MESSAGES_BODY, headers=auth_header)

    assert resp.status_code == 200
    assert registry.execute_messages.call_count == 1


def test_post_messages_passes_request_body(registry: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    client.post("/v1/messages", json=MESSAGES_BODY, headers=auth_header)

    call_args = registry.execute_messages.call_args
    body_arg = call_args.args[0]
    assert body_arg.model == "claude-model"


RESPONSES_BODY = {"model": "resp-model", "input": "Hello"}


def test_post_responses_delegates_to_registry(registry: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    resp = client.post("/v1/responses", json=RESPONSES_BODY, headers=auth_header)

    assert resp.status_code == 200
    assert registry.execute_responses.call_count == 1


def test_post_responses_passes_model_to_registry(registry: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    client.post("/v1/responses", json=RESPONSES_BODY, headers=auth_header)

    call_args = registry.execute_responses.call_args
    assert call_args.args[0].model == "resp-model"


def test_post_responses_with_reasoning_omits_temperature_and_top_p(
    registry: MagicMock, client: TestClient, auth_header: dict[str, str]
) -> None:
    client.post(
        "/v1/responses",
        json={**RESPONSES_BODY, "reasoning": {"effort": "high"}},
        headers=auth_header,
    )

    body = registry.execute_responses.call_args.args[0]
    raw = body.model_dump(exclude_none=True)
    assert raw["reasoning"] == {"effort": "high"}
    assert "temperature" not in raw
    assert "top_p" not in raw


def _make_function_tool(strict: bool | None = None) -> dict[str, Any]:
    tool: dict[str, Any] = {
        "type": "function",
        "name": "f",
        "description": "d",
        "parameters": {
            "type": "object",
            "properties": {"q": {"type": "string"}, "n": {"type": "integer"}},
            "required": ["q"],
            "additionalProperties": False,
        },
    }
    if strict is not None:
        tool["strict"] = strict
    return tool


def test_post_responses_with_optional_tool_param_defaults_strict_to_false(
    registry: MagicMock, client: TestClient, auth_header: dict[str, str]
) -> None:
    client.post(
        "/v1/responses",
        json={**RESPONSES_BODY, "tools": [_make_function_tool()]},
        headers=auth_header,
    )

    body = registry.execute_responses.call_args.args[0]
    raw = body.model_dump(exclude_none=True)
    assert raw["tools"][0]["strict"] is False


def test_post_responses_with_explicit_strict_tool_is_preserved(
    registry: MagicMock, client: TestClient, auth_header: dict[str, str]
) -> None:
    client.post(
        "/v1/responses",
        json={**RESPONSES_BODY, "tools": [_make_function_tool(strict=True)]},
        headers=auth_header,
    )

    body = registry.execute_responses.call_args.args[0]
    raw = body.model_dump(exclude_none=True)
    assert raw["tools"][0]["strict"] is True


FUNCTION_CALL_HISTORY_INPUT = [
    {"role": "user", "content": "hi"},
    {"type": "function_call", "call_id": "call_1", "name": "f", "arguments": "{}"},
    {"type": "function_call_output", "call_id": "call_1", "output": "{}"},
]


def test_post_responses_with_function_call_history_accepted(registry: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    resp = client.post(
        "/v1/responses",
        json={**RESPONSES_BODY, "input": FUNCTION_CALL_HISTORY_INPUT},
        headers=auth_header,
    )

    assert resp.status_code == 200
    body = registry.execute_responses.call_args.args[0]
    raw = body.model_dump(exclude_none=True)
    function_call, function_call_output = raw["input"][1], raw["input"][2]
    assert function_call == {"type": "function_call", "status": "completed", "call_id": "call_1", "name": "f", "arguments": "{}"}
    assert function_call_output == {"type": "function_call_output", "status": "completed", "call_id": "call_1", "output": "{}"}


def test_post_responses_with_function_call_history_preserves_explicit_id_and_status(
    registry: MagicMock, client: TestClient, auth_header: dict[str, str]
) -> None:
    input_with_ids = [
        {"role": "user", "content": "hi"},
        {
            "id": "fc_1",
            "type": "function_call",
            "status": "in_progress",
            "call_id": "call_1",
            "name": "f",
            "arguments": "{}",
        },
        {"id": "fco_1", "type": "function_call_output", "call_id": "call_1", "output": "{}"},
    ]

    client.post("/v1/responses", json={**RESPONSES_BODY, "input": input_with_ids}, headers=auth_header)

    body = registry.execute_responses.call_args.args[0]
    raw = body.model_dump(exclude_none=True)
    assert raw["input"][1]["id"] == "fc_1"
    assert raw["input"][1]["status"] == "in_progress"
    assert raw["input"][2]["id"] == "fco_1"


def test_post_responses_with_custom_tool_call_accepted(registry: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    input_with_custom_call = [
        {"type": "custom_tool_call", "call_id": "call_1", "namespace": None, "name": "run", "input": "echo hi"},
    ]

    resp = client.post(
        "/v1/responses",
        json={**RESPONSES_BODY, "input": input_with_custom_call},
        headers=auth_header,
    )

    assert resp.status_code == 200
    body = registry.execute_responses.call_args.args[0]
    raw = body.model_dump(exclude_none=True)
    assert raw["input"][0]["type"] == "custom_tool_call"


CHAT_BODY = {
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hi"}],
}


def test_post_chat_completions_delegates_to_registry(registry: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    resp = client.post("/v1/chat/completions", json=CHAT_BODY, headers=auth_header)

    assert resp.status_code == 200
    assert registry.execute_chat_completion.call_count == 1


def test_post_chat_completions_passes_model_to_registry(registry: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    client.post("/v1/chat/completions", json=CHAT_BODY, headers=auth_header)

    assert registry.execute_chat_completion.call_args.args[0].model == "gpt-4"


def test_post_chat_completions_passes_reasoning_effort_to_registry(
    registry: MagicMock, client: TestClient, auth_header: dict[str, str]
) -> None:
    client.post("/v1/chat/completions", json={**CHAT_BODY, "reasoning_effort": "medium"}, headers=auth_header)

    body = registry.execute_chat_completion.call_args.args[0]
    assert body.reasoning_effort == "medium"
    assert "reasoning_effort" in body.model_dump(exclude_none=True)


def test_post_chat_completions_passes_both_reasoning_fields_unresolved(
    registry: MagicMock, client: TestClient, auth_header: dict[str, str]
) -> None:
    client.post(
        "/v1/chat/completions",
        json={**CHAT_BODY, "reasoning_effort": "low", "reasoning": {"effort": "high"}},
        headers=auth_header,
    )

    body = registry.execute_chat_completion.call_args.args[0]
    raw = body.model_dump(exclude_none=True)
    assert raw["reasoning_effort"] == "low"
    assert raw["reasoning"] == {"effort": "high"}


COMPLETIONS_BODY = {"model": "gpt-3.5", "prompt": "Once upon a time"}


def test_post_completions_delegates_to_registry(registry: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    resp = client.post("/v1/completions", json=COMPLETIONS_BODY, headers=auth_header)

    assert resp.status_code == 200
    assert registry.execute_completion.call_count == 1


def test_post_completions_passes_model_to_registry(registry: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    client.post("/v1/completions", json=COMPLETIONS_BODY, headers=auth_header)

    assert registry.execute_completion.call_args.args[0].model == "gpt-3.5"


OLLAMA_BODY = {
    "model": "llama3",
    "messages": [{"role": "user", "content": "Hello"}],
}


def test_post_ollama_chat_delegates_to_registry(registry: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    resp = client.post("/api/chat", json=OLLAMA_BODY, headers=auth_header)

    assert resp.status_code == 200
    assert registry.execute_ollama_chat.call_count == 1


def test_post_ollama_chat_passes_model_to_registry(registry: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    client.post("/api/chat", json=OLLAMA_BODY, headers=auth_header)

    assert registry.execute_ollama_chat.call_args.args[0].model == "llama3"


EMBEDDINGS_BODY = {"model": "emb-model", "input": "Hello world"}


def test_post_embeddings_delegates_to_registry(registry: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    resp = client.post("/v1/embeddings", json=EMBEDDINGS_BODY, headers=auth_header)

    assert resp.status_code == 200
    assert registry.execute_embeddings.call_count == 1


def test_post_embeddings_passes_model_to_registry(registry: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    client.post("/v1/embeddings", json=EMBEDDINGS_BODY, headers=auth_header)
    assert registry.execute_embeddings.call_args.args[0].model == "emb-model"


SPEECH_BODY = {"model": "tts-model", "input": "Hello"}


def test_post_audio_speech_delegates_to_registry(registry: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    resp = client.post("/v1/audio/speech", json=SPEECH_BODY, headers=auth_header)

    assert resp.status_code == 200
    assert registry.execute_audio_speech.call_count == 1


def test_post_audio_speech_passes_model_to_registry(registry: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    client.post("/v1/audio/speech", json=SPEECH_BODY, headers=auth_header)

    assert registry.execute_audio_speech.call_args.args[0].model == "tts-model"


def test_post_audio_transcriptions_delegates_to_registry(registry: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    resp = client.post(
        "/v1/audio/transcriptions",
        data={"model": "stt-model"},
        files={"file": ("audio.wav", b"fake-audio", "audio/wav")},
        headers=auth_header,
    )

    assert resp.status_code == 200
    assert registry.execute_audio_transcriptions.call_count == 1


def test_post_audio_transcriptions_passes_model_to_registry(registry: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    client.post(
        "/v1/audio/transcriptions",
        data={"model": "stt-model"},
        files={"file": ("audio.wav", b"fake-audio", "audio/wav")},
        headers=auth_header,
    )

    assert registry.execute_audio_transcriptions.call_args.args[0].model == "stt-model"


IMAGES_BODY = {"model": "img-model", "prompt": "A cat"}


def test_post_images_generations_delegates_to_registry(registry: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    resp = client.post("/v1/images/generations", json=IMAGES_BODY, headers=auth_header)

    assert resp.status_code == 200
    assert registry.execute_images_generations.call_count == 1


def test_post_images_generations_passes_model_to_registry(registry: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    client.post("/v1/images/generations", json=IMAGES_BODY, headers=auth_header)

    assert registry.execute_images_generations.call_args.args[0].model == "img-model"


RERANK_BODY = {"model": "rnk-model", "query": "What is AI?", "documents": ["AI is cool", "AI is useful"]}


def test_post_rerank_delegates_to_registry(registry: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    resp = client.post("/v1/rerank", json=RERANK_BODY, headers=auth_header)

    assert resp.status_code == 200
    assert registry.execute_rerank.call_count == 1


def test_post_rerank_passes_model_to_registry(registry: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    client.post("/v1/rerank", json=RERANK_BODY, headers=auth_header)

    assert registry.execute_rerank.call_args.args[0].model == "rnk-model"


SYSTEMONE_BODY = {
    "model": "tev1:0.8b",
    "state": "Our checkout has returned 500 errors since 9am.",
    "questions": {"label": {"type": "choice", "instructions": "Which label?", "criteria": {"bug": None, "billing": None}}},
}


def test_post_systemone_delegates_to_registry(registry: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    resp = client.post("/v1/systemone", json=SYSTEMONE_BODY, headers=auth_header)

    assert resp.status_code == 200
    assert registry.execute_systemone.call_count == 1


def test_post_systemone_passes_model_to_registry(registry: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    client.post("/v1/systemone", json=SYSTEMONE_BODY, headers=auth_header)

    assert registry.execute_systemone.call_args.args[0].model == "tev1:0.8b"


def test_post_systemone_without_questions_is_rejected(registry: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    resp = client.post("/v1/systemone", json={"model": "tev1:0.8b", "state": "x"}, headers=auth_header)

    assert resp.status_code == 400
    assert registry.execute_systemone.call_count == 0


def test_post_systemone_accepts_null_state_and_question_without_instructions(
    registry: MagicMock, client: TestClient, auth_header: dict[str, str]
) -> None:
    resp = client.post(
        "/v1/systemone", json={"model": "tev1:0.8b", "state": None, "questions": {"urgent": {"type": "noul"}}}, headers=auth_header
    )

    assert resp.status_code == 200
    body = registry.execute_systemone.call_args.args[0]
    assert body.state is None
    assert body.questions["urgent"].instructions is None


def test_post_systemone_returns_backend_error_unchanged(registry: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    registry.execute_systemone.return_value = JSONResponse({"error": "too many questions"}, status_code=400)

    resp = client.post("/v1/systemone", json=SYSTEMONE_BODY, headers=auth_header)

    assert resp.status_code == 400
    assert resp.json() == {"error": "too many questions"}


def test_post_systemone_requires_auth() -> None:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(router)
    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.post("/v1/systemone", json=SYSTEMONE_BODY)

    assert resp.status_code in (401, 403)


def test_get_custom_returns_prefix_list(registry: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    ep1 = MagicMock()
    ep1.props.prefix = "my-service"
    ep2 = MagicMock()
    ep2.props.prefix = None
    registry.custom_endpoints = MagicMock()
    registry.custom_endpoints.list_models = MagicMock(return_value=[ep1, ep2])

    resp = client.get("/custom", headers=auth_header)

    assert resp.status_code == 200
    assert resp.json() == ["my-service"]


def test_get_custom_returns_empty_when_no_prefixes(registry: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    registry.custom_endpoints = MagicMock()
    registry.custom_endpoints.list_models = MagicMock(return_value=[])

    resp = client.get("/custom", headers=auth_header)

    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.parametrize("method", ["get", "post", "put", "delete", "patch"])
def test_custom_endpoint_delegates_to_registry(method: str, registry: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    resp = getattr(client, method)("/custom/my-svc/v1/action", headers=auth_header)

    assert resp.status_code == 200
    assert registry.execute_custom_endpoints.call_count == 1


def test_custom_endpoint_passes_full_path_to_registry(registry: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    client.get("/custom/my-svc/v1/action", headers=auth_header)

    assert registry.execute_custom_endpoints.call_args.args[0] == "my-svc/v1/action"


@pytest.mark.parametrize(("route_headers", "expected_route"), [({"X-DF-Route": "inst-1"}, "inst-1"), ({}, None)])
def test_custom_endpoint_passes_route_header_to_registry(
    route_headers: dict[str, str], expected_route: str | None, registry: MagicMock, client: TestClient, auth_header: dict[str, str]
) -> None:
    client.post("/custom/my-svc/jobs", headers=auth_header | route_headers)

    assert registry.execute_custom_endpoints.call_args.args[2] == expected_route


def test_custom_endpoint_returns_route_unavailable_error_body(registry: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    registry.execute_custom_endpoints = AsyncMock(
        side_effect=OpenAIHTTPException(503, "Route x is not available for prefix my-svc", "api_error", "route_unavailable")
    )

    resp = client.post("/custom/my-svc/jobs", headers=auth_header | {"X-DF-Route": "x"})

    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "route_unavailable"
    assert resp.json()["error"]["type"] == "api_error"


@pytest.mark.parametrize("method", ["get", "post", "put", "delete", "patch", "head", "options"])
def test_custom_endpoint_openapi_documents_route_header(method: str, client: TestClient) -> None:
    operation = client.get("/openapi.json").json()["paths"]["/custom/{full_path}"][method]

    assert any(p["in"] == "header" and p["name"] == "X-DF-Route" for p in operation["parameters"])
    assert "X-DF-Route" in operation["responses"]["200"]["headers"]
    assert "route_unavailable" in operation["responses"]["503"]["description"]


def test_custom_endpoint_rejects_path_traversal(registry: MagicMock, auth_header: dict[str, str]) -> None:
    # Use percent-encoded `..` so the HTTP client doesn't normalize the path.
    # FastAPI decodes it and the SafePath validator should reject it.
    with TestClient(_make_app(registry), raise_server_exceptions=False) as client:
        resp = client.get("/custom/foo%2F..%2Fetc%2Fpasswd", headers=auth_header)

    assert resp.status_code == 400
    assert registry.execute_custom_endpoints.call_count == 0


def test_get_mcp_returns_prefix_list(registry: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    ep1 = MagicMock()
    ep1.props.prefix = "mcp-service"
    ep2 = MagicMock()
    ep2.props.prefix = None
    registry.mcp_endpoints = MagicMock()
    registry.mcp_endpoints.list_models = MagicMock(return_value=[ep1, ep2])

    resp = client.get("/mcp", headers=auth_header)

    assert resp.status_code == 200
    assert resp.json() == ["mcp-service"]


def test_get_mcp_returns_empty_when_no_prefixes(registry: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    registry.mcp_endpoints = MagicMock()
    registry.mcp_endpoints.list_models = MagicMock(return_value=[])

    resp = client.get("/mcp", headers=auth_header)

    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.parametrize("method", ["get", "post", "put", "delete", "patch"])
def test_mcp_endpoint_delegates_to_registry(method: str, registry: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    resp = getattr(client, method)("/mcp/my-mcp/sse", headers=auth_header)

    assert resp.status_code == 200
    assert registry.execute_mcp_endpoints.call_count == 1


def test_mcp_endpoint_passes_full_path_to_registry(registry: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    client.get("/mcp/my-mcp/sse", headers=auth_header)

    assert registry.execute_mcp_endpoints.call_args.args[0] == "my-mcp/sse"


def test_mcp_endpoint_rejects_path_traversal(registry: MagicMock, auth_header: dict[str, str]) -> None:
    with TestClient(_make_app(registry), raise_server_exceptions=False) as client:
        resp = client.get("/mcp/foo%2F..%2Fetc%2Fpasswd", headers=auth_header)

    assert resp.status_code == 400
    assert registry.execute_mcp_endpoints.call_count == 0


def test_http_exception_returns_openai_style_error_body(registry: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    registry.execute_chat_completion = AsyncMock(side_effect=HTTPException(400, "Model not found"))

    resp = client.post("/v1/chat/completions", json=CHAT_BODY, headers=auth_header)

    assert resp.status_code == 400
    assert resp.json() == {"error": {"message": "Model not found", "type": "invalid_request_error", "param": None, "code": None}}


def test_validation_error_returns_openai_style_error_body_with_param(
    registry: MagicMock, client: TestClient, auth_header: dict[str, str]
) -> None:
    resp = client.post("/v1/chat/completions", json={"model": "gpt-4"}, headers=auth_header)

    assert resp.status_code == 400
    body = resp.json()
    assert body["error"]["type"] == "invalid_request_error"
    assert body["error"]["param"] == "messages"
    assert registry.execute_chat_completion.call_count == 0


def test_get_model_tokenizer_returns_zip_and_limit(registry: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    registry.get_embeddings_tokenizer = AsyncMock(return_value=(b"PK-zip", 8192))

    resp = client.get("/v1/models/BAAI/bge-m3/tokenizer", headers=auth_header)

    assert resp.status_code == 200
    assert resp.content == b"PK-zip"
    assert resp.headers["content-type"] == "application/zip"
    assert resp.headers["x-max-tokens"] == "8192"
    assert registry.get_embeddings_tokenizer.call_args == call("BAAI/bge-m3")


def test_get_model_tokenizer_omits_unknown_limit(registry: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    registry.get_embeddings_tokenizer = AsyncMock(return_value=(b"PK-zip", None))

    resp = client.get("/v1/models/bge-m3/tokenizer", headers=auth_header)

    assert resp.status_code == 200
    assert "x-max-tokens" not in resp.headers


def test_get_model_tokenizer_passes_through_404(registry: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    registry.get_embeddings_tokenizer = AsyncMock(side_effect=HTTPException(404, "tokenizer_not_available"))

    resp = client.get("/v1/models/text-embedding-3-small/tokenizer", headers=auth_header)

    assert resp.status_code == 404
    assert "tokenizer_not_available" in resp.text


def test_get_model_with_path_id_still_reaches_model_route(registry: MagicMock, client: TestClient, auth_header: dict[str, str]) -> None:
    registry.get_model = MagicMock(return_value=_make_api_model("BAAI/bge-m3"))
    registry.get_embeddings_tokenizer = AsyncMock()

    resp = client.get("/v1/models/BAAI/bge-m3", headers=auth_header)

    assert resp.status_code == 200
    assert resp.json()["id"] == "BAAI/bge-m3"
    assert registry.get_embeddings_tokenizer.call_count == 0
