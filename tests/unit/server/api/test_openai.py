# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for server/api/openai.py endpoints."""

from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock, call

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.testclient import TestClient

from server.api.openai import router
from server.core.dependencies import auth_server, get_endpoint_registry
from server.models.api import ApiModel, ApiModels, ModelProps


def _make_model_props() -> ModelProps:
    return ModelProps(private=False, type="llm", endpoints=["chat"])


def _make_api_model(model_id: str = "gpt-4") -> ApiModel:
    return ApiModel(id=model_id, object="model", created=0, owned_by="local", props=_make_model_props())


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
    mock.execute_custom_endpoints = AsyncMock(return_value=JSONResponse({"ok": True}))
    mock.execute_mcp_endpoints = AsyncMock(return_value=JSONResponse({"ok": True}))
    return mock


def _make_app(registry: MagicMock) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[auth_server] = lambda: "test-key"
    app.dependency_overrides[get_endpoint_registry] = lambda: registry
    return app


@pytest.fixture
def client(registry: MagicMock) -> Generator[TestClient]:
    with TestClient(_make_app(registry)) as c:
        yield c


def test_auth_required_returns_403_without_token() -> None:
    app = FastAPI()
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


def test_custom_endpoint_rejects_path_traversal(registry: MagicMock, auth_header: dict[str, str]) -> None:
    # Use percent-encoded `..` so the HTTP client doesn't normalize the path.
    # FastAPI decodes it and the SafePath validator should reject it.
    with TestClient(_make_app(registry), raise_server_exceptions=False) as client:
        resp = client.get("/custom/foo%2F..%2Fetc%2Fpasswd", headers=auth_header)

    assert resp.status_code in (400, 422)
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

    assert resp.status_code in (400, 422)
    assert registry.execute_mcp_endpoints.call_count == 0
