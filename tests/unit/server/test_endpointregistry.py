# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call, patch

import aiohttp
import pytest
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from server.endpointregistry import (
    ChatCompletionEndpoint,
    CustomEndpoint,
    Endpoint,
    EndpointRegistry,
    McpEndpoint,
    ProxyOptions,
    RegisteredModel,
    RegistrationOptions,
    SimpleEndpoint,
    _classify_error,  # pyright: ignore[reportPrivateUsage]
    post_form,
    post_json,
)
from server.models.api import (
    ChatCompletionRequest,
    CompletionLegacyRequest,
    CreateSpeechRequest,
    CreateTranscriptionRequest,
    EmbeddingRequest,
    ImagesRequest,
    MessagesRequest,
    Model,
    ModelProps,
    OllamaChatRequest,
    RerankRequest,
    ResponsesRequest,
)
from server.utils.core import HttpClientError
from server.websockets.models import UsageChangeRequest


def make_props(
    private: bool = False,
    type: str = "llm",
    endpoints: list[str] | None = None,
    context_window: int | None = None,
    max_context_window: int | None = None,
) -> ModelProps:
    return ModelProps(
        private=private,
        type=type,
        endpoints=endpoints or ["chat"],
        context_window=context_window,
        max_context_window=max_context_window,
    )


def make_parent_infra() -> MagicMock:
    pi = MagicMock()
    pi.send_models_list = MagicMock()
    pi.send_usage = MagicMock()
    return pi


def make_metrics_registry() -> MagicMock:
    mr = MagicMock()
    # Make label chains return a mock that has inc/dec/observe
    counter = MagicMock()
    counter.inc = MagicMock()
    counter.dec = MagicMock()
    counter.observe = MagicMock()
    mr.requests_in_flight.labels.return_value = counter
    mr.request_duration.labels.return_value = counter
    mr.request_total.labels.return_value = counter
    mr.request_errors.labels.return_value = counter
    return mr


def make_registry() -> EndpointRegistry:
    config = MagicMock()
    config.is_log_payloads_enabled.return_value = False
    parent_infra = make_parent_infra()
    model_tester = MagicMock()
    metrics_registry = make_metrics_registry()
    return EndpointRegistry(
        config=config,
        parent_infra=parent_infra,
        model_tester=model_tester,
        metrics_registry=metrics_registry,
    )


def make_endpoint() -> "Endpoint[Any]":
    registry: dict[Any, Any] = {}
    parent_infra = make_parent_infra()
    return Endpoint(registry=registry, parent_infra=parent_infra)  # pyright: ignore[reportMissingTypeArgument]


def make_chat_endpoint(
    on_chat: bool = True,
    on_completion: bool = False,
    on_responses: bool = False,
    on_messages: bool = False,
    on_ollama: bool = False,
) -> ChatCompletionEndpoint:
    ep = ChatCompletionEndpoint()
    if on_chat:
        ep.on_chat_completion = AsyncMock(return_value=MagicMock(spec=StreamingResponse))
    if on_completion:
        ep.on_completion = AsyncMock(return_value=MagicMock(spec=StreamingResponse))
    if on_responses:
        ep.on_responses = AsyncMock(return_value=MagicMock(spec=StreamingResponse))
    if on_messages:
        ep.on_messages = AsyncMock(return_value=MagicMock(spec=StreamingResponse))
    if on_ollama:
        ep.on_ollama_chat = AsyncMock(return_value=MagicMock(spec=StreamingResponse))
    return ep


def test_endpoint_init_stores_registry_and_parent_infra():
    registry = {}
    pi = make_parent_infra()

    ep = Endpoint(registry=registry, parent_infra=pi)

    assert ep.registry is registry
    assert ep.parent_infra is pi
    assert ep.models == {}


def test_add_model_creates_registration_and_returns_id():
    ep = make_endpoint()
    reg_opts = RegistrationOptions(origin="local")

    rid = ep.add_model("gpt-4", make_props(), SimpleEndpoint(on_request=AsyncMock()), "llm", reg_opts)

    assert rid is not None
    assert "gpt-4" in ep.models
    assert rid in ep.models["gpt-4"]


def test_add_model_uses_provided_registration_id():
    ep = make_endpoint()
    reg_opts = RegistrationOptions(origin="local", id="my-id-123")

    rid = ep.add_model("gpt-4", make_props(), SimpleEndpoint(on_request=AsyncMock()), "llm", reg_opts)

    assert rid == "my-id-123"


def test_add_model_registers_in_global_registry():
    ep = make_endpoint()
    reg_opts = RegistrationOptions(origin="local")

    rid = ep.add_model("gpt-4", make_props(), SimpleEndpoint(on_request=AsyncMock()), "llm", reg_opts)

    assert rid in ep.registry


def test_add_model_calls_send_models_list_by_default():
    ep = make_endpoint()
    reg_opts = RegistrationOptions(origin="local")

    ep.add_model("gpt-4", make_props(), SimpleEndpoint(on_request=AsyncMock()), "llm", reg_opts)

    assert ep.parent_infra.send_models_list.call_count == 1  # pyright: ignore[reportAttributeAccessIssue]


def test_add_model_skips_notification_when_send_notification_false():
    ep = make_endpoint()
    reg_opts = RegistrationOptions(origin="local", send_notification=False)

    ep.add_model("gpt-4", make_props(), SimpleEndpoint(on_request=AsyncMock()), "llm", reg_opts)

    assert ep.parent_infra.send_models_list.call_count == 0  # pyright: ignore[reportAttributeAccessIssue]


def test_add_model_sets_usage_from_options():
    ep = make_endpoint()
    reg_opts = RegistrationOptions(origin="local", usage=5)

    rid = ep.add_model("gpt-4", make_props(), SimpleEndpoint(on_request=AsyncMock()), "llm", reg_opts)

    assert ep.models["gpt-4"][rid].usage == 5


def test_add_model_defaults_usage_to_zero_when_none():
    ep = make_endpoint()
    reg_opts = RegistrationOptions(origin="local")

    rid = ep.add_model("gpt-4", make_props(), SimpleEndpoint(on_request=AsyncMock()), "llm", reg_opts)

    assert ep.models["gpt-4"][rid].usage == 0


def test_remove_model_removes_from_models_and_registry():
    ep = make_endpoint()
    reg_opts = RegistrationOptions(origin="local")
    rid = ep.add_model("gpt-4", make_props(), SimpleEndpoint(on_request=AsyncMock()), "llm", reg_opts)

    ep.remove_model("gpt-4", rid)

    assert "gpt-4" not in ep.models
    assert rid not in ep.registry


def test_remove_model_no_op_when_model_id_missing():
    ep = make_endpoint()

    ep.remove_model("nonexistent", "some-id")  # must not raise


def test_remove_model_calls_send_models_list():
    ep = make_endpoint()
    reg_opts = RegistrationOptions(origin="local")
    rid = ep.add_model("gpt-4", make_props(), SimpleEndpoint(on_request=AsyncMock()), "llm", reg_opts)
    ep.parent_infra.send_models_list.reset_mock()  # pyright: ignore[reportAttributeAccessIssue]

    ep.remove_model("gpt-4", rid)

    assert ep.parent_infra.send_models_list.call_count == 1  # pyright: ignore[reportAttributeAccessIssue]


def test_remove_model_skips_notification_when_requested():
    ep = make_endpoint()
    reg_opts = RegistrationOptions(origin="local")
    rid = ep.add_model("gpt-4", make_props(), SimpleEndpoint(on_request=AsyncMock()), "llm", reg_opts)
    ep.parent_infra.send_models_list.reset_mock()  # pyright: ignore[reportAttributeAccessIssue]

    ep.remove_model("gpt-4", rid, send_notification=False)

    assert ep.parent_infra.send_models_list.call_count == 0  # pyright: ignore[reportAttributeAccessIssue]


def test_remove_model_unknown_registration_id_does_not_delete_model():
    ep = make_endpoint()
    ep.add_model("gpt-4", make_props(), SimpleEndpoint(on_request=AsyncMock()), "llm", RegistrationOptions(origin="local", id="id-1"))

    ep.remove_model("gpt-4", "nonexistent-rid")

    assert ep.has_model("gpt-4")


def test_remove_model_leaves_model_when_other_registrations_remain():
    ep = make_endpoint()
    rid1 = ep.add_model(
        "gpt-4", make_props(), SimpleEndpoint(on_request=AsyncMock()), "llm", RegistrationOptions(origin="local", id="id-1")
    )
    ep.add_model("gpt-4", make_props(), SimpleEndpoint(on_request=AsyncMock()), "llm", RegistrationOptions(origin="local", id="id-2"))

    ep.remove_model("gpt-4", rid1)

    assert ep.has_model("gpt-4")


def test_remove_model_with_registration_id_not_in_registry():
    ep = make_endpoint()
    ep.add_model("gpt-4", make_props(), SimpleEndpoint(on_request=AsyncMock()), "llm", RegistrationOptions(origin="local", id="id-1"))
    del ep.registry["id-1"]

    ep.remove_model("gpt-4", "id-1")  # must not raise


def test_has_model_returns_true_when_registered():
    ep = make_endpoint()

    ep.add_model("gpt-4", make_props(), SimpleEndpoint(on_request=AsyncMock()), "llm", RegistrationOptions(origin="local"))

    assert ep.has_model("gpt-4") is True


def test_has_model_returns_false_when_not_registered():
    ep = make_endpoint()

    assert ep.has_model("gpt-4") is False


def test_has_model_returns_false_after_all_removed():
    ep = make_endpoint()
    rid = ep.add_model("gpt-4", make_props(), SimpleEndpoint(on_request=AsyncMock()), "llm", RegistrationOptions(origin="local"))

    ep.remove_model("gpt-4", rid)

    assert ep.has_model("gpt-4") is False


def test_get_model_returns_none_when_model_not_registered():
    ep = make_endpoint()

    assert ep.get_model("gpt-4") is None


def test_get_model_returns_model_with_zero_usage_immediately():
    ep = make_endpoint()
    rid = ep.add_model("gpt-4", make_props(), SimpleEndpoint(on_request=AsyncMock()), "llm", RegistrationOptions(origin="local", usage=0))

    result = ep.get_model("gpt-4")

    assert result is not None
    assert result.id == rid


def test_get_model_returns_lowest_usage_when_all_busy():
    ep = make_endpoint()
    ep.add_model(
        "gpt-4",
        make_props(),
        SimpleEndpoint(on_request=AsyncMock()),
        "llm",
        RegistrationOptions(origin="local", usage=3),
    )
    rid2 = ep.add_model(
        "gpt-4",
        make_props(),
        SimpleEndpoint(on_request=AsyncMock()),
        "llm",
        RegistrationOptions(origin="local", usage=1),
    )

    result = ep.get_model("gpt-4")

    assert result is not None
    assert result.id == rid2


def test_get_model_with_filter_excludes_non_matching():
    ep = make_endpoint()
    chat_ep = make_chat_endpoint(on_chat=True, on_completion=False)
    ep.add_model("gpt-4", make_props(), chat_ep, "llm", RegistrationOptions(origin="local"))

    result = ep.get_model("gpt-4", filter=lambda x: x.on_completion is not None)  # pyright: ignore[reportUnknownLambdaType]

    assert result is None


def test_get_model_with_registration_id_returns_specific():
    ep = make_endpoint()
    ep.add_model(
        "gpt-4",
        make_props(),
        SimpleEndpoint(on_request=AsyncMock()),
        "llm",
        RegistrationOptions(origin="local", id="id-1", usage=1),
    )
    rid2 = ep.add_model(
        "gpt-4",
        make_props(),
        SimpleEndpoint(on_request=AsyncMock()),
        "llm",
        RegistrationOptions(origin="local", id="id-2", usage=2),
    )

    result = ep.get_model("gpt-4", registration_id=rid2)

    assert result is not None
    assert result.id == rid2


def test_is_model_private_returns_true_when_any_private():
    ep = make_endpoint()
    ep.add_model(
        "gpt-4",
        make_props(private=True),
        SimpleEndpoint(on_request=AsyncMock()),
        "llm",
        RegistrationOptions(origin="local"),
    )

    result = ep.is_model_private(ep.models["gpt-4"])

    assert result is True


def test_get_model_type_returns_type():
    ep = make_endpoint()
    ep.add_model("gpt-4", make_props(type="llm"), SimpleEndpoint(on_request=AsyncMock()), "llm", RegistrationOptions(origin="local"))

    result = ep.get_model_type(ep.models["gpt-4"])

    assert result == "llm"


def test_get_model_available_endpoints_aggregates():
    ep = make_endpoint()
    ep.add_model(
        "gpt-4",
        make_props(endpoints=["chat", "completions"]),
        SimpleEndpoint(on_request=AsyncMock()),
        "llm",
        RegistrationOptions(origin="local"),
    )

    endpoints = ep.get_model_available_endpoints(ep.models["gpt-4"])

    assert "chat" in endpoints
    assert "completions" in endpoints


def test_get_model_context_window_returns_max():
    ep = make_endpoint()
    ep.add_model(
        "gpt-4",
        make_props(context_window=4096),
        SimpleEndpoint(on_request=AsyncMock()),
        "llm",
        RegistrationOptions(origin="local", id="id-1"),
    )
    ep.add_model(
        "gpt-4",
        make_props(context_window=8192),
        SimpleEndpoint(on_request=AsyncMock()),
        "llm",
        RegistrationOptions(origin="local", id="id-2"),
    )

    result = ep.get_model_context_window(ep.models["gpt-4"])

    assert result == 8192


def test_get_max_context_window_returns_max():
    ep = make_endpoint()
    ep.add_model(
        "gpt-4",
        make_props(max_context_window=16000),
        SimpleEndpoint(on_request=AsyncMock()),
        "llm",
        RegistrationOptions(origin="local", id="id-1"),
    )

    result = ep.get_max_context_window(ep.models["gpt-4"])

    assert result == 16000


def test_get_models_returns_api_model_list():
    ep = make_endpoint()
    ep.add_model("gpt-4", make_props(), SimpleEndpoint(on_request=AsyncMock()), "llm", RegistrationOptions(origin="local"))

    models = ep.get_models()

    assert len(models) == 1
    assert models[0].id == "gpt-4"


def test_get_models_returns_empty_when_no_models():
    ep = make_endpoint()

    assert ep.get_models() == []


def test_get_models_aggregates_multiple_models():
    ep = make_endpoint()
    ep.add_model("gpt-4", make_props(), SimpleEndpoint(on_request=AsyncMock()), "llm", RegistrationOptions(origin="local"))
    ep.add_model("gpt-3.5", make_props(), SimpleEndpoint(on_request=AsyncMock()), "llm", RegistrationOptions(origin="local"))

    models = ep.get_models()

    assert len(models) == 2


def test_list_models_returns_registered_model_entries():
    ep = make_endpoint()
    ep.add_model("gpt-4", make_props(), SimpleEndpoint(on_request=AsyncMock()), "llm", RegistrationOptions(origin="local", id="rid-1"))

    items = ep.list_models()

    assert len(items) == 1
    assert items[0].name == "gpt-4"


def test_list_models_returns_multiple_registrations_for_same_model():
    ep = make_endpoint()
    ep.add_model("gpt-4", make_props(), SimpleEndpoint(on_request=AsyncMock()), "llm", RegistrationOptions(origin="local", id="id-a"))
    ep.add_model("gpt-4", make_props(), SimpleEndpoint(on_request=AsyncMock()), "llm", RegistrationOptions(origin="local", id="id-b"))

    items = ep.list_models()

    assert len(items) == 2


def test_proxy_options_get_request_headers_no_request():
    opts = ProxyOptions(url="http://example.com", headers={"Authorization": "Bearer tok"})

    headers = opts.get_request_headers(None)

    assert headers == {"Authorization": "Bearer tok"}


def test_proxy_options_get_request_headers_merges_allowed_request_headers():
    opts = ProxyOptions(
        url="http://example.com",
        headers={"Authorization": "Bearer tok"},
        allowed_request_headers=["x-custom"],
    )
    request = MagicMock()
    request.headers = {"x-custom": "val", "x-other": "ignored"}

    headers = opts.get_request_headers(request)

    assert headers["x-custom"] == "val"
    assert headers["Authorization"] == "Bearer tok"
    assert "x-other" not in headers


def test_proxy_options_get_request_headers_no_allowed_headers_returns_static():
    opts = ProxyOptions(url="http://example.com", headers={"Authorization": "Bearer tok"})
    request = MagicMock()
    request.headers = {"x-forward": "val"}

    headers = opts.get_request_headers(request)

    assert "x-forward" not in headers
    assert headers["Authorization"] == "Bearer tok"


def test_endpoint_registry_init_sets_attributes():
    reg = make_registry()

    assert reg.chat_completion_endpoints is not None
    assert reg.embeddings_endpoints is not None
    assert reg.audio_speech_endpoints is not None
    assert reg.audio_transcriptions_endpoints is not None
    assert reg.custom_endpoints is not None
    assert reg.images_generations_endpoints is not None
    assert reg.rerank_endpoints is not None
    assert reg.mcp_endpoints is not None
    assert reg.registry == {}


def test_endpoint_registry_init_assigns_self_to_parent_infra():
    config = MagicMock()
    config.is_log_payloads_enabled.return_value = False
    parent_infra = make_parent_infra()
    reg = EndpointRegistry(
        config=config,
        parent_infra=parent_infra,
        model_tester=MagicMock(),
        metrics_registry=make_metrics_registry(),
    )

    assert parent_infra.endpoint_registry is reg


def test_registry_get_models_aggregates_all_endpoint_types():
    reg = make_registry()
    props = make_props()
    reg.register_embeddings("emb-model", props, SimpleEndpoint(on_request=AsyncMock()), None)

    models = reg.get_models()

    ids = [m.id for m in models.data]
    assert "emb-model" in ids


def test_registry_get_models_returns_empty_when_nothing_registered():
    reg = make_registry()

    result = reg.get_models()

    assert result.data == []


def test_registry_get_models_combines_multiple_types():
    reg = make_registry()
    props = make_props()
    chat_ep = make_chat_endpoint()
    reg.register_chat_completion("chat-model", props, chat_ep, None)
    reg.register_embeddings("emb-model", props, SimpleEndpoint(on_request=AsyncMock()), None)

    models = reg.get_models()

    ids = [m.id for m in models.data]
    assert "chat-model" in ids
    assert "emb-model" in ids


def test_registry_get_models_includes_audio_speech():
    reg = make_registry()

    reg.register_audio_speech("tts-m", make_props(), SimpleEndpoint(on_request=AsyncMock()), None)

    ids = [m.id for m in reg.get_models().data]
    assert "tts-m" in ids


def test_registry_get_models_includes_audio_transcriptions():
    reg = make_registry()

    reg.register_audio_transcriptions("stt-m", make_props(), SimpleEndpoint(on_request=AsyncMock()), None)

    ids = [m.id for m in reg.get_models().data]
    assert "stt-m" in ids


def test_registry_get_models_includes_images_generations():
    reg = make_registry()

    reg.register_image_generations("img-m", make_props(), SimpleEndpoint(on_request=AsyncMock()), None)

    ids = [m.id for m in reg.get_models().data]
    assert "img-m" in ids


def test_registry_get_models_includes_rerank():
    reg = make_registry()

    reg.register_rerank("rnk-m", make_props(), SimpleEndpoint(on_request=AsyncMock()), None)

    ids = [m.id for m in reg.get_models().data]
    assert "rnk-m" in ids


def test_registry_get_model_returns_model_by_id():
    reg = make_registry()

    reg.register_embeddings("emb-model", make_props(), SimpleEndpoint(on_request=AsyncMock()), None)

    model = reg.get_model("emb-model")
    assert model.id == "emb-model"


def test_registry_get_model_raises_404_when_not_found():
    reg = make_registry()

    with pytest.raises(HTTPException) as exc_info:
        reg.get_model("nonexistent")

    assert exc_info.value.status_code == 404


def test_registry_list_models_returns_all():
    reg = make_registry()
    props = make_props()
    reg.register_embeddings("emb", props, SimpleEndpoint(on_request=AsyncMock()), None)
    reg.register_rerank("rnk", props, SimpleEndpoint(on_request=AsyncMock()), None)

    models = reg.list_models()

    names = [m.name for m in models]
    assert "emb" in names
    assert "rnk" in names


def test_register_chat_completion_registers_model():
    reg = make_registry()
    ep = make_chat_endpoint()

    rid = reg.register_chat_completion("gpt-4", make_props(), ep, None)

    assert reg.chat_completion_endpoints.has_model("gpt-4")
    assert rid in reg.registry


def test_register_chat_completion_type_is_llm_for_all_suffixes():
    reg = make_registry()
    ep = make_chat_endpoint(on_chat=True, on_completion=True, on_responses=True, on_messages=True, on_ollama=True)

    reg.register_chat_completion("gpt-4", make_props(), ep, None)

    models = reg.chat_completion_endpoints.get_models()
    assert models[0].props.type == "llm"


def test_register_chat_completion_type_is_llm_partial():
    reg = make_registry()
    ep = make_chat_endpoint(on_chat=True, on_completion=False, on_responses=False, on_messages=False, on_ollama=False)

    reg.register_chat_completion("gpt-4", make_props(), ep, None)

    models = reg.chat_completion_endpoints.get_models()
    assert "v2" in models[0].props.type


@pytest.mark.asyncio
async def test_register_chat_completion_as_proxy_registers_model():
    reg = make_registry()
    opts = ProxyOptions(url="http://example.com/v1/chat/completions")

    rid = reg.register_chat_completion_as_proxy(
        model="gpt-4",
        props=make_props(),
        chat_completions=opts,
        completions=None,
        responses=None,
        messages=None,
        ollama_chat=None,
        registration_options=None,
    )

    assert reg.chat_completion_endpoints.has_model("gpt-4")
    assert rid in reg.registry


def test_register_chat_completion_as_proxy_raises_without_chat_or_completion():
    reg = make_registry()

    with pytest.raises(RuntimeError):
        reg.register_chat_completion_as_proxy(
            model="gpt-4",
            props=make_props(),
            chat_completions=None,
            completions=None,
            responses=None,
            messages=None,
            ollama_chat=None,
            registration_options=None,
        )


def test_register_chat_completion_as_proxy_with_all_options():
    reg = make_registry()
    opts = ProxyOptions(url="http://example.com/")

    reg.register_chat_completion_as_proxy(
        model="gpt-4",
        props=make_props(),
        chat_completions=opts,
        completions=opts,
        responses=opts,
        messages=opts,
        ollama_chat=opts,
        registration_options=None,
    )

    ep = reg.chat_completion_endpoints.get_model("gpt-4")
    assert ep is not None
    assert ep.endpoint.on_chat_completion is not None
    assert ep.endpoint.on_completion is not None
    assert ep.endpoint.on_responses is not None
    assert ep.endpoint.on_messages is not None
    assert ep.endpoint.on_ollama_chat is not None


def test_unregister_chat_completion_removes_model():
    reg = make_registry()
    ep = make_chat_endpoint()

    rid = reg.register_chat_completion("gpt-4", make_props(), ep, None)

    reg.unregister_chat_completion("gpt-4", rid)
    assert not reg.chat_completion_endpoints.has_model("gpt-4")


def test_register_embeddings_registers_model():
    reg = make_registry()

    rid = reg.register_embeddings("emb", make_props(), SimpleEndpoint(on_request=AsyncMock()), None)

    assert reg.embeddings_endpoints.has_model("emb")
    assert rid in reg.registry


def test_register_embeddings_as_proxy_registers_model():
    reg = make_registry()
    opts = ProxyOptions(url="http://example.com/v1/embeddings")

    rid = reg.register_embeddings_as_proxy("emb", make_props(), opts, None)

    assert reg.embeddings_endpoints.has_model("emb")
    assert rid in reg.registry


def test_unregister_embeddings_removes_model():
    reg = make_registry()
    rid = reg.register_embeddings("emb", make_props(), SimpleEndpoint(on_request=AsyncMock()), None)

    reg.unregister_embeddings("emb", rid)

    assert not reg.embeddings_endpoints.has_model("emb")


def test_register_audio_speech_registers_model():
    reg = make_registry()

    reg.register_audio_speech("tts-model", make_props(), SimpleEndpoint(on_request=AsyncMock()), None)

    assert reg.audio_speech_endpoints.has_model("tts-model")


def test_register_audio_speech_as_proxy_registers_model():
    reg = make_registry()
    opts = ProxyOptions(url="http://example.com/v1/audio/speech")

    reg.register_audio_speech_as_proxy("tts-model", make_props(), opts, None)
    assert reg.audio_speech_endpoints.has_model("tts-model")


def test_unregister_audio_speech_removes_model():
    reg = make_registry()

    rid = reg.register_audio_speech("tts-model", make_props(), SimpleEndpoint(on_request=AsyncMock()), None)

    reg.unregister_audio_speech("tts-model", rid)
    assert not reg.audio_speech_endpoints.has_model("tts-model")


def test_register_audio_transcriptions_registers_model():
    reg = make_registry()

    reg.register_audio_transcriptions("stt-model", make_props(), SimpleEndpoint(on_request=AsyncMock()), None)

    assert reg.audio_transcriptions_endpoints.has_model("stt-model")


def test_register_audio_transcriptions_as_proxy_registers_model():
    reg = make_registry()
    opts = ProxyOptions(url="http://example.com/v1/audio/transcriptions")

    reg.register_audio_transcriptions_as_proxy("stt-model", make_props(), opts, None)

    assert reg.audio_transcriptions_endpoints.has_model("stt-model")


def test_unregister_audio_transcriptions_removes_model():
    reg = make_registry()

    rid = reg.register_audio_transcriptions("stt-model", make_props(), SimpleEndpoint(on_request=AsyncMock()), None)

    reg.unregister_audio_transcriptions("stt-model", rid)
    assert not reg.audio_transcriptions_endpoints.has_model("stt-model")


def test_register_image_generations_registers_model():
    reg = make_registry()

    reg.register_image_generations("img-model", make_props(), SimpleEndpoint(on_request=AsyncMock()), None)

    assert reg.images_generations_endpoints.has_model("img-model")


def test_register_image_generations_as_proxy_registers_model():
    reg = make_registry()
    opts = ProxyOptions(url="http://example.com/v1/images/generations")

    reg.register_image_generations_as_proxy("img-model", make_props(), opts, None)

    assert reg.images_generations_endpoints.has_model("img-model")


def test_unregister_image_generations_removes_model():
    reg = make_registry()

    rid = reg.register_image_generations("img-model", make_props(), SimpleEndpoint(on_request=AsyncMock()), None)

    reg.unregister_image_generations("img-model", rid)
    assert not reg.images_generations_endpoints.has_model("img-model")


def test_register_rerank_registers_model():
    reg = make_registry()

    reg.register_rerank("rnk-model", make_props(), SimpleEndpoint(on_request=AsyncMock()), None)

    assert reg.rerank_endpoints.has_model("rnk-model")


def test_register_rerank_as_proxy_registers_model():
    reg = make_registry()
    opts = ProxyOptions(url="http://example.com/v1/rerank")

    reg.register_rerank_as_proxy("rnk-model", make_props(), opts, None)

    assert reg.rerank_endpoints.has_model("rnk-model")


def test_unregister_rerank_removes_model():
    reg = make_registry()

    rid = reg.register_rerank("rnk-model", make_props(), SimpleEndpoint(on_request=AsyncMock()), None)

    reg.unregister_rerank("rnk-model", rid)
    assert not reg.rerank_endpoints.has_model("rnk-model")


def test_register_custom_endpoint_registers():
    reg = make_registry()

    reg.register_custom_endpoint("custom/path", make_props(), CustomEndpoint(on_request=AsyncMock()), None)

    assert reg.custom_endpoints.has_model("custom/path")


def test_register_custom_endpoint_as_proxy_registers():
    reg = make_registry()
    opts = ProxyOptions(url="http://example.com/custom")

    reg.register_custom_endpoint_as_proxy("custom/path", make_props(), opts, None)

    assert reg.custom_endpoints.has_model("custom/path")


def test_unregister_custom_endpoint_removes():
    reg = make_registry()

    rid = reg.register_custom_endpoint("custom/path", make_props(), CustomEndpoint(on_request=AsyncMock()), None)

    reg.unregister_custom_endpoint("custom/path", rid)
    assert not reg.custom_endpoints.has_model("custom/path")


def test_register_mcp_endpoint_registers():
    reg = make_registry()

    reg.register_mcp_endpoint("mcp/url", make_props(), McpEndpoint(on_request=AsyncMock()), None)

    assert reg.mcp_endpoints.has_model("mcp/url")


def test_register_mcp_endpoint_as_proxy_registers():
    reg = make_registry()
    opts = ProxyOptions(url="http://example.com/mcp")

    reg.register_mcp_endpoint_as_proxy("mcp/url", make_props(), opts, None)

    assert reg.mcp_endpoints.has_model("mcp/url")


def test_unregister_mcp_endpoint_removes():
    reg = make_registry()

    rid = reg.register_mcp_endpoint("mcp/url", make_props(), McpEndpoint(on_request=AsyncMock()), None)

    reg.unregister_mcp_endpoint("mcp/url", rid)
    assert not reg.mcp_endpoints.has_model("mcp/url")


def test_update_usage_updates_model_usage():
    reg = make_registry()
    ep = make_chat_endpoint()

    rid = reg.register_chat_completion("gpt-4", make_props(), ep, RegistrationOptions(origin="local", usage=0))

    reg.update_usage(UsageChangeRequest(id=rid, usage=3))
    entry = reg.registry[rid]
    assert entry.registered_model.usage == 3


def test_update_usage_calls_refresh_on_change():
    reg = make_registry()
    ep = make_chat_endpoint()

    rid = reg.register_chat_completion("gpt-4", make_props(), ep, RegistrationOptions(origin="local", usage=0))

    reg.parent_infra.send_usage.reset_mock()  # pyright: ignore[reportAttributeAccessIssue]
    reg.update_usage(UsageChangeRequest(id=rid, usage=5))
    assert reg.parent_infra.send_usage.call_count == 1  # pyright: ignore[reportAttributeAccessIssue]


def test_update_usage_no_op_when_same_value():
    reg = make_registry()
    ep = make_chat_endpoint()

    rid = reg.register_chat_completion("gpt-4", make_props(), ep, RegistrationOptions(origin="local", usage=2))

    reg.parent_infra.send_usage.reset_mock()  # pyright: ignore[reportAttributeAccessIssue]
    reg.update_usage(UsageChangeRequest(id=rid, usage=2))
    assert reg.parent_infra.send_usage.call_count == 0  # pyright: ignore[reportAttributeAccessIssue]


def test_update_usage_no_op_for_unknown_id():
    reg = make_registry()
    reg.update_usage(UsageChangeRequest(id="nonexistent", usage=1))  # must not raise


def test_update_models_registers_new_models():
    reg = make_registry()
    new_model = Model(id="new-id", name="gpt-4", type="llm", props=make_props(), usage=0)

    with patch.object(reg, "_register_proxy") as mock_register:
        reg.update_models([], [new_model], "http://api.example.com/", "mykey")

    assert mock_register.call_count == 1


def test_update_models_removes_old_models():
    reg = make_registry()
    ep = make_chat_endpoint()
    reg.register_chat_completion("gpt-4", make_props(), ep, RegistrationOptions(origin="local", id="old-id"))
    old_model = Model(id="old-id", name="gpt-4", type="llm", props=make_props(), usage=0)

    reg.update_models([old_model], [], "http://api.example.com/", "mykey")

    assert not reg.chat_completion_endpoints.has_model("gpt-4")


def test_update_models_sends_models_list_when_changed():
    reg = make_registry()
    new_model = Model(id="new-id", name="gpt-4", type="llm", props=make_props(), usage=0)
    reg.parent_infra.send_models_list.reset_mock()  # pyright: ignore[reportAttributeAccessIssue]

    with patch.object(reg, "_register_proxy"):
        reg.update_models([], [new_model], "http://api.example.com/", "mykey")

    assert reg.parent_infra.send_models_list.call_count == 1  # pyright: ignore[reportAttributeAccessIssue]


def test_update_models_no_notification_when_nothing_changes():
    reg = make_registry()
    ep = make_chat_endpoint()
    reg.register_chat_completion("gpt-4", make_props(), ep, RegistrationOptions(origin="local", id="exist-id"))
    existing = Model(id="exist-id", name="gpt-4", type="llm", props=make_props(), usage=0)
    reg.parent_infra.send_models_list.reset_mock()  # pyright: ignore[reportAttributeAccessIssue]

    reg.update_models([existing], [existing], "http://api.example.com/", "mykey")

    assert reg.parent_infra.send_models_list.call_count == 0  # pyright: ignore[reportAttributeAccessIssue]


def test_register_proxy_llm_registers_chat_completion():
    reg = make_registry()

    reg._register_proxy("gpt-4", "llm", make_props(), "http://example.com/", "key", RegistrationOptions(origin="http://example.com/"))  # pyright: ignore[reportPrivateUsage]

    assert reg.chat_completion_endpoints.has_model("gpt-4")


def test_register_proxy_tts_registers_audio_speech():
    reg = make_registry()

    reg._register_proxy("tts-model", "tts", make_props(), "http://example.com/", "key", RegistrationOptions(origin="http://example.com/"))  # pyright: ignore[reportPrivateUsage]

    assert reg.audio_speech_endpoints.has_model("tts-model")


def test_register_proxy_stt_registers_audio_transcriptions():
    reg = make_registry()

    reg._register_proxy("stt-model", "stt", make_props(), "http://example.com/", "key", RegistrationOptions(origin="http://example.com/"))  # pyright: ignore[reportPrivateUsage]

    assert reg.audio_transcriptions_endpoints.has_model("stt-model")


def test_register_proxy_txt2img_registers_image_generations():
    reg = make_registry()

    reg._register_proxy(  # pyright: ignore[reportPrivateUsage]
        "img-model", "txt2img", make_props(), "http://example.com/", "key", RegistrationOptions(origin="http://example.com/")
    )

    assert reg.images_generations_endpoints.has_model("img-model")


def test_register_proxy_embedding_registers_embeddings():
    reg = make_registry()

    reg._register_proxy(  # pyright: ignore[reportPrivateUsage]
        "emb-model", "embedding", make_props(), "http://example.com/", "key", RegistrationOptions(origin="http://example.com/")
    )

    assert reg.embeddings_endpoints.has_model("emb-model")


def test_register_proxy_rerank_registers_rerank():
    reg = make_registry()

    reg._register_proxy(  # pyright: ignore[reportPrivateUsage]
        "rnk-model", "rerank", make_props(), "http://example.com/", "key", RegistrationOptions(origin="http://example.com/")
    )

    assert reg.rerank_endpoints.has_model("rnk-model")


def test_register_proxy_custom_registers_custom_endpoint():
    reg = make_registry()

    reg._register_proxy(  # pyright: ignore[reportPrivateUsage]
        "custom/url", "custom", make_props(), "http://example.com/", "key", RegistrationOptions(origin="http://example.com/")
    )

    assert reg.custom_endpoints.has_model("custom/url")


def test_register_proxy_mcp_registers_mcp_endpoint():
    reg = make_registry()

    reg._register_proxy("mcp/url", "mcp", make_props(), "http://example.com/", "key", RegistrationOptions(origin="http://example.com/"))  # pyright: ignore[reportPrivateUsage]

    assert reg.mcp_endpoints.has_model("mcp/url")


def test_register_proxy_llm_partial_v2_only():
    reg = make_registry()

    reg._register_proxy("gpt-4", "llm-v2", make_props(), "http://example.com/", "key", RegistrationOptions(origin="http://example.com/"))  # pyright: ignore[reportPrivateUsage]
    ep = reg.chat_completion_endpoints.get_model("gpt-4")

    assert ep is not None
    assert ep.endpoint.on_chat_completion is not None
    assert ep.endpoint.on_completion is None


def test_register_proxy_unknown_type_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    reg = make_registry()

    with caplog.at_level(logging.WARNING, logger="uvicorn.error"):
        reg._register_proxy(  # pyright: ignore[reportPrivateUsage]
            "x", "unknown-type", make_props(), "http://example.com/", "key", RegistrationOptions(origin="http://example.com/")
        )

    assert "Cannot register proxy" in caplog.text


def test_has_chat_completion_model_true():
    reg = make_registry()
    ep = make_chat_endpoint(on_chat=True)

    reg.register_chat_completion("gpt-4", make_props(), ep, None)

    assert reg.has_chat_completion_model("gpt-4") is True


def test_has_chat_completion_model_false_when_not_registered():
    reg = make_registry()

    assert reg.has_chat_completion_model("gpt-4") is False


def test_has_completion_model_true():
    reg = make_registry()
    ep = make_chat_endpoint(on_chat=True, on_completion=True)

    reg.register_chat_completion("gpt-4", make_props(), ep, None)

    assert reg.has_completion_model("gpt-4") is True


def test_has_completion_model_false_when_no_completion():
    reg = make_registry()
    ep = make_chat_endpoint(on_chat=True, on_completion=False)

    reg.register_chat_completion("gpt-4", make_props(), ep, None)

    assert reg.has_completion_model("gpt-4") is False


def test_has_embeddings_model_true():
    reg = make_registry()

    reg.register_embeddings("emb", make_props(), SimpleEndpoint(on_request=AsyncMock()), None)

    assert reg.has_embeddings_model("emb") is True


def test_has_embeddings_model_false():
    reg = make_registry()

    assert reg.has_embeddings_model("emb") is False


def test_has_audio_speech_model():
    reg = make_registry()

    reg.register_audio_speech("tts", make_props(), SimpleEndpoint(on_request=AsyncMock()), None)

    assert reg.has_audio_speech_model("tts") is True


def test_has_audio_transcriptions_model():
    reg = make_registry()

    reg.register_audio_transcriptions("stt", make_props(), SimpleEndpoint(on_request=AsyncMock()), None)

    assert reg.has_audio_transcriptions_model("stt") is True


def test_has_image_generations_model():
    reg = make_registry()

    reg.register_image_generations("img", make_props(), SimpleEndpoint(on_request=AsyncMock()), None)

    assert reg.has_image_generations_model("img") is True


def test_has_rerank_model():
    reg = make_registry()

    reg.register_rerank("rnk", make_props(), SimpleEndpoint(on_request=AsyncMock()), None)

    assert reg.has_rerank_model("rnk") is True


def test_has_custom_endpoint():
    reg = make_registry()

    reg.register_custom_endpoint("custom/path", make_props(), CustomEndpoint(on_request=AsyncMock()), None)

    assert reg.has_custom_endpoint("custom/path") is True


def test_has_mcp_endpoint():
    reg = make_registry()

    reg.register_mcp_endpoint("mcp/url", make_props(), McpEndpoint(on_request=AsyncMock()), None)

    assert reg.has_mcp_endpoint("mcp/url") is True


@pytest.mark.asyncio
async def test_execute_messages_success():
    reg = make_registry()
    ep = make_chat_endpoint(on_messages=True)
    response = MagicMock()
    ep.on_messages = AsyncMock(return_value=response)
    reg.register_chat_completion("ant-model", make_props(), ep, RegistrationOptions(origin="local"))
    body = MagicMock(spec=MessagesRequest)
    body.model = "ant-model"
    body.model_dump_json.return_value = "{}"

    with patch.object(reg, "with_usage", new_callable=AsyncMock, return_value=response) as mock_usage:
        await reg.execute_messages(body)

    assert mock_usage.call_count == 1


@pytest.mark.asyncio
async def test_execute_messages_raises_400_when_model_missing():
    reg = make_registry()
    body = MagicMock(spec=MessagesRequest)
    body.model = "unknown-model"
    body.model_dump_json.return_value = "{}"

    with pytest.raises(HTTPException) as exc_info:
        await reg.execute_messages(body)

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_execute_messages_raises_400_when_model_has_no_messages_support():
    reg = make_registry()
    ep = make_chat_endpoint(on_chat=True, on_messages=False)
    reg.register_chat_completion("gpt-4", make_props(), ep, RegistrationOptions(origin="local"))

    body = MagicMock(spec=MessagesRequest)
    body.model = "gpt-4"
    body.model_dump_json.return_value = "{}"

    with pytest.raises(HTTPException) as exc_info:
        await reg.execute_messages(body)

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_execute_responses_success():
    reg = make_registry()
    ep = make_chat_endpoint(on_responses=True)
    response = MagicMock()
    reg.register_chat_completion("resp-model", make_props(), ep, RegistrationOptions(origin="local"))

    body = MagicMock(spec=ResponsesRequest)
    body.model = "resp-model"
    body.model_dump_json.return_value = "{}"

    with patch.object(reg, "with_usage", new_callable=AsyncMock, return_value=response):
        result = await reg.execute_responses(body)

    assert result is response


@pytest.mark.asyncio
async def test_execute_responses_raises_400_when_not_registered():
    reg = make_registry()
    body = MagicMock(spec=ResponsesRequest)
    body.model = "unknown"
    body.model_dump_json.return_value = "{}"

    with pytest.raises(HTTPException) as exc_info:
        await reg.execute_responses(body)

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_execute_responses_raises_400_when_model_has_no_responses_support():
    reg = make_registry()
    ep = make_chat_endpoint(on_chat=True, on_responses=False)
    reg.register_chat_completion("gpt-4", make_props(), ep, RegistrationOptions(origin="local"))

    body = MagicMock(spec=ResponsesRequest)
    body.model = "gpt-4"
    body.model_dump_json.return_value = "{}"

    with pytest.raises(HTTPException) as exc_info:
        await reg.execute_responses(body)

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_execute_chat_completion_success():
    reg = make_registry()
    ep = make_chat_endpoint(on_chat=True)
    response = MagicMock()
    reg.register_chat_completion("gpt-4", make_props(), ep, RegistrationOptions(origin="local"))

    body = MagicMock(spec=ChatCompletionRequest)
    body.model = "gpt-4"
    body.model_dump_json.return_value = "{}"

    with patch.object(reg, "with_usage", new_callable=AsyncMock, return_value=response):
        result = await reg.execute_chat_completion(body)

    assert result is response


@pytest.mark.asyncio
async def test_execute_chat_completion_raises_400_when_not_registered():
    reg = make_registry()
    body = MagicMock(spec=ChatCompletionRequest)
    body.model = "unknown"
    body.model_dump_json.return_value = "{}"

    with pytest.raises(HTTPException) as exc_info:
        await reg.execute_chat_completion(body)

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_execute_chat_completion_raises_400_when_model_has_no_chat_support():
    reg = make_registry()
    ep = make_chat_endpoint(on_chat=False, on_completion=True)
    reg.register_chat_completion("gpt-4", make_props(), ep, RegistrationOptions(origin="local"))

    body = MagicMock(spec=ChatCompletionRequest)
    body.model = "gpt-4"
    body.model_dump_json.return_value = "{}"

    with pytest.raises(HTTPException) as exc_info:
        await reg.execute_chat_completion(body)

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_execute_completion_success():
    reg = make_registry()
    ep = make_chat_endpoint(on_completion=True)
    response = MagicMock()
    reg.register_chat_completion("gpt-35", make_props(), ep, RegistrationOptions(origin="local"))

    body = MagicMock(spec=CompletionLegacyRequest)
    body.model = "gpt-35"

    with patch.object(reg, "with_usage", new_callable=AsyncMock, return_value=response):
        result = await reg.execute_completion(body)

    assert result is response


@pytest.mark.asyncio
async def test_execute_completion_raises_400_when_not_registered():
    reg = make_registry()
    body = MagicMock(spec=CompletionLegacyRequest)
    body.model = "unknown"

    with pytest.raises(HTTPException) as exc_info:
        await reg.execute_completion(body)

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_execute_completion_raises_400_when_model_has_no_completion_support():
    reg = make_registry()
    ep = make_chat_endpoint(on_chat=True, on_completion=False)
    reg.register_chat_completion("gpt-4", make_props(), ep, RegistrationOptions(origin="local"))

    body = MagicMock(spec=CompletionLegacyRequest)
    body.model = "gpt-4"

    with pytest.raises(HTTPException) as exc_info:
        await reg.execute_completion(body)

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_execute_ollama_chat_success():
    reg = make_registry()
    ep = make_chat_endpoint(on_ollama=True)
    response = MagicMock()
    reg.register_chat_completion("ollama-model", make_props(), ep, RegistrationOptions(origin="local"))

    body = MagicMock(spec=OllamaChatRequest)
    body.model = "ollama-model"

    with patch.object(reg, "with_usage", new_callable=AsyncMock, return_value=response):
        result = await reg.execute_ollama_chat(body)

    assert result is response


@pytest.mark.asyncio
async def test_execute_ollama_chat_raises_400_when_not_registered():
    reg = make_registry()
    body = MagicMock(spec=OllamaChatRequest)
    body.model = "unknown"

    with pytest.raises(HTTPException) as exc_info:
        await reg.execute_ollama_chat(body)

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_execute_ollama_chat_raises_400_when_model_has_no_ollama_support():
    reg = make_registry()
    ep = make_chat_endpoint(on_chat=True, on_ollama=False)
    reg.register_chat_completion("gpt-4", make_props(), ep, RegistrationOptions(origin="local"))

    body = MagicMock(spec=OllamaChatRequest)
    body.model = "gpt-4"

    with pytest.raises(HTTPException) as exc_info:
        await reg.execute_ollama_chat(body)

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_execute_embeddings_success():
    reg = make_registry()
    response = MagicMock()
    reg.register_embeddings(
        "emb",
        make_props(),
        SimpleEndpoint(on_request=AsyncMock(return_value=response)),
        RegistrationOptions(origin="local"),
    )

    body = MagicMock(spec=EmbeddingRequest)
    body.model = "emb"

    with patch.object(reg, "with_usage", new_callable=AsyncMock, return_value=response):
        result = await reg.execute_embeddings(body)

    assert result is response


@pytest.mark.asyncio
async def test_execute_embeddings_raises_400_when_not_registered():
    reg = make_registry()
    body = MagicMock(spec=EmbeddingRequest)
    body.model = "unknown"

    with pytest.raises(HTTPException) as exc_info:
        await reg.execute_embeddings(body)

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_execute_images_generations_success():
    reg = make_registry()
    response = MagicMock()
    reg.register_image_generations(
        "img",
        make_props(),
        SimpleEndpoint(on_request=AsyncMock(return_value=response)),
        RegistrationOptions(origin="local"),
    )

    body = MagicMock(spec=ImagesRequest)
    body.model = "img"
    body.model_dump_json.return_value = "{}"

    with patch.object(reg, "with_usage", new_callable=AsyncMock, return_value=response):
        result = await reg.execute_images_generations(body)

    assert result is response


@pytest.mark.asyncio
async def test_execute_images_generations_raises_400_when_not_registered():
    reg = make_registry()
    body = MagicMock(spec=ImagesRequest)
    body.model = "unknown"
    body.model_dump_json.return_value = "{}"

    with pytest.raises(HTTPException) as exc_info:
        await reg.execute_images_generations(body)

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_execute_audio_speech_success():
    reg = make_registry()
    response = MagicMock()
    reg.register_audio_speech(
        "tts",
        make_props(),
        SimpleEndpoint(on_request=AsyncMock(return_value=response)),
        RegistrationOptions(origin="local"),
    )

    body = MagicMock(spec=CreateSpeechRequest)
    body.model = "tts"
    body.model_dump_json.return_value = "{}"

    with patch.object(reg, "with_usage", new_callable=AsyncMock, return_value=response):
        result = await reg.execute_audio_speech(body)

    assert result is response


@pytest.mark.asyncio
async def test_execute_audio_speech_raises_400_when_not_registered():
    reg = make_registry()
    body = MagicMock(spec=CreateSpeechRequest)
    body.model = "unknown"
    body.model_dump_json.return_value = "{}"

    with pytest.raises(HTTPException) as exc_info:
        await reg.execute_audio_speech(body)

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_execute_audio_transcriptions_success():
    reg = make_registry()
    response = MagicMock()
    reg.register_audio_transcriptions(
        "stt",
        make_props(),
        SimpleEndpoint(on_request=AsyncMock(return_value=response)),
        RegistrationOptions(origin="local"),
    )

    body = MagicMock(spec=CreateTranscriptionRequest)
    body.model = "stt"

    with patch.object(reg, "with_usage", new_callable=AsyncMock, return_value=response):
        result = await reg.execute_audio_transcriptions(body)

    assert result is response


@pytest.mark.asyncio
async def test_execute_audio_transcriptions_raises_400_when_not_registered():
    reg = make_registry()
    body = MagicMock(spec=CreateTranscriptionRequest)
    body.model = "unknown"

    with pytest.raises(HTTPException) as exc_info:
        await reg.execute_audio_transcriptions(body)

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_execute_rerank_success():
    reg = make_registry()
    response = MagicMock()
    reg.register_rerank(
        "rnk",
        make_props(),
        SimpleEndpoint(on_request=AsyncMock(return_value=response)),
        RegistrationOptions(origin="local"),
    )

    body = MagicMock(spec=RerankRequest)
    body.model = "rnk"
    body.model_dump_json.return_value = "{}"

    with patch.object(reg, "with_usage", new_callable=AsyncMock, return_value=response):
        result = await reg.execute_rerank(body)

    assert result is response


@pytest.mark.asyncio
async def test_execute_rerank_raises_400_when_not_registered():
    reg = make_registry()
    body = MagicMock(spec=RerankRequest)
    body.model = "unknown"
    body.model_dump_json.return_value = "{}"

    with pytest.raises(HTTPException) as exc_info:
        await reg.execute_rerank(body)

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_execute_custom_endpoints_success():
    reg = make_registry()
    response = MagicMock()
    reg.register_custom_endpoint(
        "my-svc",
        make_props(),
        CustomEndpoint(on_request=AsyncMock(return_value=response)),
        RegistrationOptions(origin="local"),
    )

    request = MagicMock()

    with patch.object(reg, "with_usage", new_callable=AsyncMock, return_value=response):
        result = await reg.execute_custom_endpoints("my-svc/some/path", request)

    assert result is response


@pytest.mark.asyncio
async def test_execute_custom_endpoints_raises_400_when_not_found():
    reg = make_registry()
    request = MagicMock()
    with pytest.raises(HTTPException) as exc_info:
        await reg.execute_custom_endpoints("nonexistent/path", request)

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_execute_mcp_endpoints_success():
    reg = make_registry()
    response = MagicMock()
    reg.register_mcp_endpoint(
        "mcp-svc",
        make_props(),
        McpEndpoint(on_request=AsyncMock(return_value=response)),
        RegistrationOptions(origin="local"),
    )

    request = MagicMock()

    with patch.object(reg, "with_usage", new_callable=AsyncMock, return_value=response):
        result = await reg.execute_mcp_endpoints("mcp-svc/some/path", request)

    assert result is response


@pytest.mark.asyncio
async def test_execute_mcp_endpoints_raises_400_when_not_found():
    reg = make_registry()
    request = MagicMock()

    with pytest.raises(HTTPException) as exc_info:
        await reg.execute_mcp_endpoints("nonexistent/path", request)

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_with_usage_skips_metrics_for_non_local_origin():
    reg = make_registry()
    ep = make_chat_endpoint()
    rid = reg.register_chat_completion("gpt-4", make_props(), ep, RegistrationOptions(origin="http://remote.example.com"))
    model = reg.registry[rid].registered_model

    response = MagicMock()
    func = AsyncMock(return_value=response)

    result = await reg.with_usage(model, func)

    assert result is response
    assert reg.metrics_registry.requests_in_flight.labels.call_count == 0  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.asyncio
async def test_with_usage_records_metrics_for_local_non_streaming():
    reg = make_registry()
    ep = make_chat_endpoint()
    rid = reg.register_chat_completion("gpt-4", make_props(), ep, RegistrationOptions(origin="local"))
    model = reg.registry[rid].registered_model

    response = MagicMock()
    response.__class__ = object  # not StreamingResponse  # pyright: ignore[reportAttributeAccessIssue]
    func = AsyncMock(return_value=response)

    result = await reg.with_usage(model, func)

    assert result is response
    assert reg.metrics_registry.requests_in_flight.labels.call_count == 2  # pyright: ignore[reportAttributeAccessIssue]
    assert reg.metrics_registry.request_total.labels.call_count == 1  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.asyncio
async def test_with_usage_records_error_metrics_on_exception():
    reg = make_registry()
    ep = make_chat_endpoint()
    rid = reg.register_chat_completion("gpt-4", make_props(), ep, RegistrationOptions(origin="local"))
    model = reg.registry[rid].registered_model

    func = AsyncMock(side_effect=RuntimeError("boom"))

    with pytest.raises(RuntimeError):
        await reg.with_usage(model, func)

    assert reg.metrics_registry.request_errors.labels.call_count == 1  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.asyncio
async def test_with_usage_wraps_streaming_response():
    reg = make_registry()
    ep = make_chat_endpoint()
    rid = reg.register_chat_completion("gpt-4", make_props(), ep, RegistrationOptions(origin="local"))
    model = reg.registry[rid].registered_model

    async def _gen():
        yield b"chunk1"
        yield b"chunk2"

    streaming = StreamingResponse(_gen(), media_type="text/event-stream", status_code=200)
    func = AsyncMock(return_value=streaming)
    result = await reg.with_usage(model, func)
    assert isinstance(result, StreamingResponse)
    chunks = []

    async for chunk in result.body_iterator:
        chunks.append(chunk)

    assert b"chunk1" in chunks
    assert b"chunk2" in chunks


def test_add_usage_increments_and_notifies():
    reg = make_registry()
    ep = make_chat_endpoint()
    rid = reg.register_chat_completion("gpt-4", make_props(), ep, RegistrationOptions(origin="local", usage=0))
    model = reg.registry[rid].registered_model
    reg.parent_infra.send_usage.reset_mock()  # pyright: ignore[reportAttributeAccessIssue]

    reg._add_usage(model)  # pyright: ignore[reportPrivateUsage]

    assert model.usage == 1
    assert reg.parent_infra.send_usage.call_count == 1  # pyright: ignore[reportAttributeAccessIssue]


def test_remove_usage_decrements_and_notifies():
    reg = make_registry()
    ep = make_chat_endpoint()
    rid = reg.register_chat_completion("gpt-4", make_props(), ep, RegistrationOptions(origin="local", usage=2))
    model = reg.registry[rid].registered_model
    reg.parent_infra.send_usage.reset_mock()  # pyright: ignore[reportAttributeAccessIssue]

    reg._remove_usage(model)  # pyright: ignore[reportPrivateUsage]

    assert model.usage == 1
    assert reg.parent_infra.send_usage.call_count == 1  # pyright: ignore[reportAttributeAccessIssue]


def test_refresh_usage_sends_usage():
    reg = make_registry()
    ep = make_chat_endpoint()
    rid = reg.register_chat_completion("gpt-4", make_props(), ep, RegistrationOptions(origin="local", usage=3))
    model = reg.registry[rid].registered_model
    reg.parent_infra.send_usage.reset_mock()  # pyright: ignore[reportAttributeAccessIssue]

    reg._refresh_usage(model)  # pyright: ignore[reportPrivateUsage]

    assert reg.parent_infra.send_usage.call_count == 1  # pyright: ignore[reportAttributeAccessIssue]
    assert reg.parent_infra.send_usage.call_args == call(UsageChangeRequest(id=model.id, usage=3))  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.asyncio
async def test_test_model_calls_model_tester():
    reg = make_registry()
    ep = make_chat_endpoint()
    rid = reg.register_chat_completion("gpt-4", make_props(), ep, RegistrationOptions(origin="local"))
    reg.model_tester.test_model = AsyncMock(return_value={"status": "ok"})

    result = await reg.test_model(rid)

    assert result == {"status": "ok"}
    assert reg.model_tester.test_model.call_count == 1


@pytest.mark.asyncio
async def test_test_model_raises_400_when_not_found():
    reg = make_registry()
    with pytest.raises(HTTPException) as exc_info:
        await reg.test_model("nonexistent-rid")

    assert exc_info.value.status_code == 400


def test_classify_error_timeout():
    assert _classify_error(TimeoutError()) == "timeout"


def test_classify_error_aiohttp_client_error():
    assert _classify_error(aiohttp.ClientError()) == "connection_error"


def test_classify_error_http_client_error():
    assert _classify_error(HttpClientError(message="err", status_code=500, headers=MagicMock(), body="")) == "http_error"


def test_classify_error_generic():
    assert _classify_error(RuntimeError("boom")) == "model_error"


@pytest.mark.asyncio
async def test_post_json_sends_request_and_returns_streaming_response():
    class FakeBody(BaseModel):
        model: str = "gpt-4"
        text: str = "hello"

    mock_http_response = MagicMock()
    mock_streaming = MagicMock(spec=StreamingResponse)
    mock_http_response.as_streaming_response.return_value = mock_streaming

    opts = ProxyOptions(url="http://example.com/v1/chat/completions")

    with patch("server.endpointregistry.make_http_request", new_callable=AsyncMock, return_value=mock_http_response):
        result = await post_json(FakeBody(), opts)

    assert result is mock_streaming


@pytest.mark.asyncio
async def test_post_json_removes_model_when_requested():
    class FakeBody(BaseModel):
        model: str = "gpt-4"

    mock_http_response = MagicMock()
    mock_http_response.as_streaming_response.return_value = MagicMock()
    opts = ProxyOptions(url="http://example.com/", remove_model=True)

    with patch("server.endpointregistry.make_http_request", new_callable=AsyncMock, return_value=mock_http_response) as mock_req:
        await post_json(FakeBody(), opts)

    called_data = mock_req.call_args.kwargs["data"]
    assert "model" not in json.loads(called_data._value)


@pytest.mark.asyncio
async def test_post_json_rewrites_model_when_requested():
    class FakeBody(BaseModel):
        model: str = "gpt-4"

    mock_http_response = MagicMock()
    mock_http_response.as_streaming_response.return_value = MagicMock()
    opts = ProxyOptions(url="http://example.com/", rewrite_model_to="gpt-3.5-turbo")

    with patch("server.endpointregistry.make_http_request", new_callable=AsyncMock, return_value=mock_http_response) as mock_req:
        await post_json(FakeBody(), opts)

    called_data = mock_req.call_args.kwargs["data"]
    assert json.loads(called_data._value)["model"] == "gpt-3.5-turbo"


@pytest.mark.asyncio
async def test_post_form_sends_request_and_returns_streaming_response():
    mock_http_response = MagicMock()
    mock_streaming = MagicMock(spec=StreamingResponse)
    mock_http_response.as_streaming_response.return_value = mock_streaming
    data = MagicMock()
    data.to_form = AsyncMock(return_value=MagicMock())
    opts = ProxyOptions(url="http://example.com/v1/audio/transcriptions")

    with patch("server.endpointregistry.make_http_request", new_callable=AsyncMock, return_value=mock_http_response):
        result = await post_form(data, opts)

    assert result is mock_streaming


@pytest.mark.asyncio
async def test_register_chat_completion_as_proxy_callback_invokes_post_json():
    reg = make_registry()
    opts = ProxyOptions(url="http://example.com/v1/chat/completions")
    reg.register_chat_completion_as_proxy(
        model="gpt-4",
        props=make_props(),
        chat_completions=opts,
        completions=None,
        responses=None,
        messages=None,
        ollama_chat=None,
        registration_options=None,
    )
    ep = reg.chat_completion_endpoints.get_model("gpt-4")
    assert ep is not None
    mock_resp = MagicMock(spec=StreamingResponse)

    with patch("server.endpointregistry.post_json", new_callable=AsyncMock, return_value=mock_resp):
        body = MagicMock(spec=ChatCompletionRequest)

        result = await ep.endpoint.on_chat_completion(body, None)  # pyright: ignore[reportOptionalCall]
    assert result is mock_resp


@pytest.mark.asyncio
async def test_register_chat_completion_as_proxy_completion_callback_invokes_post_json():
    reg = make_registry()
    opts = ProxyOptions(url="http://example.com/v1/completions")
    reg.register_chat_completion_as_proxy(
        model="gpt-4",
        props=make_props(),
        chat_completions=opts,
        completions=opts,
        responses=None,
        messages=None,
        ollama_chat=None,
        registration_options=None,
    )
    ep = reg.chat_completion_endpoints.get_model("gpt-4")
    mock_resp = MagicMock(spec=StreamingResponse)

    with patch("server.endpointregistry.post_json", new_callable=AsyncMock, return_value=mock_resp):
        body = MagicMock(spec=CompletionLegacyRequest)
        result = await ep.endpoint.on_completion(body, None)  # pyright: ignore[reportOptionalMemberAccess, reportOptionalCall]

    assert result is mock_resp


@pytest.mark.asyncio
async def test_register_chat_completion_as_proxy_responses_callback():
    reg = make_registry()
    opts = ProxyOptions(url="http://example.com/v1/responses")
    reg.register_chat_completion_as_proxy(
        model="gpt-4",
        props=make_props(),
        chat_completions=opts,
        completions=None,
        responses=opts,
        messages=None,
        ollama_chat=None,
        registration_options=None,
    )
    ep = reg.chat_completion_endpoints.get_model("gpt-4")
    mock_resp = MagicMock(spec=StreamingResponse)

    with patch("server.endpointregistry.post_json", new_callable=AsyncMock, return_value=mock_resp):
        body = MagicMock(spec=ResponsesRequest)
        result = await ep.endpoint.on_responses(body, None)  # pyright: ignore[reportOptionalMemberAccess, reportOptionalCall]

    assert result is mock_resp


@pytest.mark.asyncio
async def test_register_chat_completion_as_proxy_messages_callback():
    reg = make_registry()
    opts = ProxyOptions(url="http://example.com/v1/messages")
    reg.register_chat_completion_as_proxy(
        model="gpt-4",
        props=make_props(),
        chat_completions=opts,
        completions=None,
        responses=None,
        messages=opts,
        ollama_chat=None,
        registration_options=None,
    )
    ep = reg.chat_completion_endpoints.get_model("gpt-4")
    mock_resp = MagicMock(spec=StreamingResponse)

    with patch("server.endpointregistry.post_json", new_callable=AsyncMock, return_value=mock_resp):
        body = MagicMock(spec=MessagesRequest)
        result = await ep.endpoint.on_messages(body, None)  # pyright: ignore[reportOptionalMemberAccess, reportOptionalCall]

    assert result is mock_resp


@pytest.mark.asyncio
async def test_register_chat_completion_as_proxy_ollama_callback():
    reg = make_registry()
    opts = ProxyOptions(url="http://example.com/api/chat")
    reg.register_chat_completion_as_proxy(
        model="gpt-4",
        props=make_props(),
        chat_completions=opts,
        completions=None,
        responses=None,
        messages=None,
        ollama_chat=opts,
        registration_options=None,
    )
    ep = reg.chat_completion_endpoints.get_model("gpt-4")
    mock_resp = MagicMock(spec=StreamingResponse)

    with patch("server.endpointregistry.post_json", new_callable=AsyncMock, return_value=mock_resp):
        body = MagicMock(spec=OllamaChatRequest)
        result = await ep.endpoint.on_ollama_chat(body, None)  # pyright: ignore[reportOptionalMemberAccess, reportOptionalCall]

    assert result is mock_resp


@pytest.mark.asyncio
async def test_register_embeddings_as_proxy_callback_invokes_post_json():
    reg = make_registry()
    opts = ProxyOptions(url="http://example.com/v1/embeddings")
    reg.register_embeddings_as_proxy("emb", make_props(), opts, None)
    ep = reg.embeddings_endpoints.get_model("emb")
    mock_resp = MagicMock(spec=StreamingResponse)

    with patch("server.endpointregistry.post_json", new_callable=AsyncMock, return_value=mock_resp):
        body = MagicMock(spec=EmbeddingRequest)
        result = await ep.endpoint.on_request(body, None)  # pyright: ignore[reportOptionalMemberAccess]

    assert result is mock_resp


@pytest.mark.asyncio
async def test_register_audio_speech_as_proxy_callback_invokes_post_json():
    reg = make_registry()
    opts = ProxyOptions(url="http://example.com/v1/audio/speech")
    reg.register_audio_speech_as_proxy("tts", make_props(), opts, None)
    ep = reg.audio_speech_endpoints.get_model("tts")
    mock_resp = MagicMock(spec=StreamingResponse)

    with patch("server.endpointregistry.post_json", new_callable=AsyncMock, return_value=mock_resp):
        body = MagicMock(spec=CreateSpeechRequest)
        result = await ep.endpoint.on_request(body, None)  # pyright: ignore[reportOptionalMemberAccess]

    assert result is mock_resp


@pytest.mark.asyncio
async def test_register_audio_transcriptions_as_proxy_callback_invokes_post_form():
    reg = make_registry()
    opts = ProxyOptions(url="http://example.com/v1/audio/transcriptions")
    reg.register_audio_transcriptions_as_proxy("stt", make_props(), opts, None)
    ep = reg.audio_transcriptions_endpoints.get_model("stt")
    mock_resp = MagicMock(spec=StreamingResponse)

    with patch("server.endpointregistry.post_form", new_callable=AsyncMock, return_value=mock_resp):
        body = MagicMock(spec=CreateTranscriptionRequest)
        result = await ep.endpoint.on_request(body, None)  # pyright: ignore[reportOptionalMemberAccess]

    assert result is mock_resp


@pytest.mark.asyncio
async def test_register_image_generations_as_proxy_callback_invokes_post_json():
    reg = make_registry()
    opts = ProxyOptions(url="http://example.com/v1/images/generations")
    reg.register_image_generations_as_proxy("img", make_props(), opts, None)
    ep = reg.images_generations_endpoints.get_model("img")
    mock_resp = MagicMock(spec=StreamingResponse)

    with patch("server.endpointregistry.post_json", new_callable=AsyncMock, return_value=mock_resp):
        body = MagicMock(spec=ImagesRequest)
        result = await ep.endpoint.on_request(body, None)  # pyright: ignore[reportOptionalMemberAccess]

    assert result is mock_resp


@pytest.mark.asyncio
async def test_register_rerank_as_proxy_callback_invokes_post_json():
    reg = make_registry()
    opts = ProxyOptions(url="http://example.com/v1/rerank")
    reg.register_rerank_as_proxy("rnk", make_props(), opts, None)
    ep = reg.rerank_endpoints.get_model("rnk")
    mock_resp = MagicMock(spec=StreamingResponse)

    with patch("server.endpointregistry.post_json", new_callable=AsyncMock, return_value=mock_resp):
        body = MagicMock(spec=RerankRequest)
        result = await ep.endpoint.on_request(body, None)  # pyright: ignore[reportOptionalMemberAccess]

    assert result is mock_resp


@pytest.mark.asyncio
async def test_register_custom_endpoint_as_proxy_callback_invokes_make_http_request():
    reg = make_registry()
    opts = ProxyOptions(url="http://example.com/custom/")
    reg.register_custom_endpoint_as_proxy("my-svc", make_props(), opts, None)
    ep = reg.custom_endpoints.get_model("my-svc")
    mock_http_response = MagicMock()
    mock_streaming = MagicMock(spec=StreamingResponse)
    mock_http_response.as_streaming_response.return_value = mock_streaming
    request = MagicMock()
    request.headers = {"content-type": "application/json"}
    request.method = "POST"
    request.path_params = {"full_path": "my-svc/sub/path"}
    request.stream.return_value = AsyncMock()

    with patch("server.endpointregistry.make_http_request", new_callable=AsyncMock, return_value=mock_http_response):
        result = await ep.endpoint.on_request(request)  # pyright: ignore[reportOptionalMemberAccess]

    assert result is mock_streaming


@pytest.mark.asyncio
async def test_register_mcp_endpoint_as_proxy_callback_invokes_make_http_request():
    reg = make_registry()
    opts = ProxyOptions(url="http://example.com/mcp/")
    reg.register_mcp_endpoint_as_proxy("mcp-svc", make_props(), opts, None)
    ep = reg.mcp_endpoints.get_model("mcp-svc")
    mock_http_response = MagicMock()
    mock_streaming = MagicMock(spec=StreamingResponse)
    mock_http_response.as_streaming_response.return_value = mock_streaming
    request = MagicMock()
    request.headers = {"content-type": "application/json"}
    request.method = "POST"
    request.path_params = {"full_path": "mcp-svc/sub/path"}
    request.stream.return_value = AsyncMock()

    with patch("server.endpointregistry.make_http_request", new_callable=AsyncMock, return_value=mock_http_response):
        result = await ep.endpoint.on_request(request)  # pyright: ignore[reportOptionalMemberAccess]

    assert result is mock_streaming


@pytest.mark.asyncio
async def test_register_custom_endpoint_as_proxy_callback_no_query_string():
    reg = make_registry()
    opts = ProxyOptions(url="http://example.com/custom/")
    reg.register_custom_endpoint_as_proxy("my-svc", make_props(), opts, None)
    ep = reg.custom_endpoints.get_model("my-svc")
    mock_http_response = MagicMock()
    mock_streaming = MagicMock(spec=StreamingResponse)
    mock_http_response.as_streaming_response.return_value = mock_streaming
    request = MagicMock()
    request.headers = {"content-type": "application/json"}
    request.method = "GET"
    request.path_params = {"full_path": "my-svc/sub/path"}
    request.url.query = ""
    request.stream.return_value = AsyncMock()

    with patch("server.endpointregistry.make_http_request", new_callable=AsyncMock, return_value=mock_http_response):
        result = await ep.endpoint.on_request(request)  # pyright: ignore[reportOptionalMemberAccess]

    assert result is mock_streaming


@pytest.mark.asyncio
async def test_register_mcp_endpoint_as_proxy_callback_no_query_string():
    reg = make_registry()
    opts = ProxyOptions(url="http://example.com/mcp/")
    reg.register_mcp_endpoint_as_proxy("mcp-svc", make_props(), opts, None)
    ep = reg.mcp_endpoints.get_model("mcp-svc")
    mock_http_response = MagicMock()
    mock_http_response.as_streaming_response.return_value = MagicMock(spec=StreamingResponse)
    request = MagicMock()
    request.headers = {"content-type": "application/json"}
    request.method = "GET"
    request.path_params = {"full_path": "mcp-svc/sub/path"}
    request.stream.return_value = AsyncMock()
    request.url.query = ""

    with patch("server.endpointregistry.make_http_request", new_callable=AsyncMock, return_value=mock_http_response) as mock_req:
        await ep.endpoint.on_request(request)  # pyright: ignore[reportOptionalMemberAccess]

    assert "?" not in mock_req.call_args.kwargs["url"]


@pytest.mark.asyncio
async def test_execute_messages_with_log_payload():
    reg = make_registry()
    reg.config.is_log_payloads_enabled.return_value = True  # pyright: ignore[reportAttributeAccessIssue]
    ep = make_chat_endpoint(on_messages=True)
    response = MagicMock()
    reg.register_chat_completion("ant-model", make_props(), ep, RegistrationOptions(origin="local"))
    body = MagicMock(spec=MessagesRequest)
    body.model = "ant-model"
    body.model_dump_json.return_value = "{}"

    with patch.object(reg, "with_usage", new_callable=AsyncMock, return_value=response):
        result = await reg.execute_messages(body)

    assert result is response


@pytest.mark.asyncio
async def test_execute_responses_with_log_payload():
    reg = make_registry()
    reg.config.is_log_payloads_enabled.return_value = True  # pyright: ignore[reportAttributeAccessIssue]
    ep = make_chat_endpoint(on_responses=True)
    response = MagicMock()
    reg.register_chat_completion("resp-model", make_props(), ep, RegistrationOptions(origin="local"))
    body = MagicMock(spec=ResponsesRequest)
    body.model = "resp-model"
    body.model_dump_json.return_value = "{}"

    with patch.object(reg, "with_usage", new_callable=AsyncMock, return_value=response):
        result = await reg.execute_responses(body)

    assert result is response


@pytest.mark.asyncio
async def test_execute_chat_completion_with_log_payload():
    reg = make_registry()
    reg.config.is_log_payloads_enabled.return_value = True  # pyright: ignore[reportAttributeAccessIssue]
    ep = make_chat_endpoint(on_chat=True)
    response = MagicMock()
    reg.register_chat_completion("gpt-4", make_props(), ep, RegistrationOptions(origin="local"))
    body = MagicMock(spec=ChatCompletionRequest)
    body.model = "gpt-4"
    body.model_dump_json.return_value = "{}"

    with patch.object(reg, "with_usage", new_callable=AsyncMock, return_value=response):
        result = await reg.execute_chat_completion(body)

    assert result is response


@pytest.mark.asyncio
async def test_execute_images_generations_with_log_payload():
    reg = make_registry()
    reg.config.is_log_payloads_enabled.return_value = True  # pyright: ignore[reportAttributeAccessIssue]
    response = MagicMock()
    reg.register_image_generations("img", make_props(), SimpleEndpoint(on_request=AsyncMock()), RegistrationOptions(origin="local"))
    body = MagicMock(spec=ImagesRequest)
    body.model = "img"
    body.model_dump_json.return_value = "{}"

    with patch.object(reg, "with_usage", new_callable=AsyncMock, return_value=response):
        result = await reg.execute_images_generations(body)

    assert result is response


@pytest.mark.asyncio
async def test_execute_audio_speech_with_log_payload():
    reg = make_registry()
    reg.config.is_log_payloads_enabled.return_value = True  # pyright: ignore[reportAttributeAccessIssue]
    response = MagicMock()
    reg.register_audio_speech("tts", make_props(), SimpleEndpoint(on_request=AsyncMock()), RegistrationOptions(origin="local"))
    body = MagicMock(spec=CreateSpeechRequest)
    body.model = "tts"
    body.model_dump_json.return_value = "{}"

    with patch.object(reg, "with_usage", new_callable=AsyncMock, return_value=response):
        result = await reg.execute_audio_speech(body)

    assert result is response


@pytest.mark.asyncio
async def test_execute_rerank_with_log_payload():
    reg = make_registry()
    reg.config.is_log_payloads_enabled.return_value = True  # pyright: ignore[reportAttributeAccessIssue]
    response = MagicMock()
    reg.register_rerank("rnk", make_props(), SimpleEndpoint(on_request=AsyncMock()), RegistrationOptions(origin="local"))
    body = MagicMock(spec=RerankRequest)
    body.model = "rnk"
    body.model_dump_json.return_value = "{}"

    with patch.object(reg, "with_usage", new_callable=AsyncMock, return_value=response):
        result = await reg.execute_rerank(body)

    assert result is response


def _make_registered_model_with_endpoint(ep: ChatCompletionEndpoint) -> Any:
    """Build a RegisteredModel whose endpoint is `ep` but has no on_messages/etc."""
    return RegisteredModel(id="fake-id", name="gpt-4", origin="local", props=make_props(), type="llm", endpoint=ep, usage=0)


@pytest.mark.asyncio
async def test_execute_messages_includes_supported_endpoints_in_error():
    reg = make_registry()
    ep = make_chat_endpoint(on_chat=True, on_responses=True, on_completion=True, on_messages=False)
    fake = _make_registered_model_with_endpoint(ep)
    body = MagicMock(spec=MessagesRequest)
    body.model = "gpt-4"
    body.model_dump_json.return_value = "{}"

    with patch.object(reg.chat_completion_endpoints, "get_model", return_value=fake), pytest.raises(HTTPException) as exc_info:
        await reg.execute_messages(body)

    assert "/v1/chat/completions" in exc_info.value.detail or "/v1/responses" in exc_info.value.detail


@pytest.mark.asyncio
async def test_execute_responses_includes_supported_endpoints_in_error():
    reg = make_registry()
    ep = make_chat_endpoint(on_chat=True, on_messages=True, on_completion=True, on_responses=False)
    fake = _make_registered_model_with_endpoint(ep)
    body = MagicMock(spec=ResponsesRequest)
    body.model = "gpt-4"
    body.model_dump_json.return_value = "{}"

    with patch.object(reg.chat_completion_endpoints, "get_model", return_value=fake), pytest.raises(HTTPException) as exc_info:
        await reg.execute_responses(body)

    assert "/v1/chat/completions" in exc_info.value.detail or "/v1/messages" in exc_info.value.detail


@pytest.mark.asyncio
async def test_execute_chat_completion_includes_supported_endpoints_in_error():
    reg = make_registry()
    ep = make_chat_endpoint(on_chat=False, on_messages=True, on_responses=True, on_completion=True)
    fake = _make_registered_model_with_endpoint(ep)
    body = MagicMock(spec=ChatCompletionRequest)
    body.model = "gpt-4"
    body.model_dump_json.return_value = "{}"

    with patch.object(reg.chat_completion_endpoints, "get_model", return_value=fake), pytest.raises(HTTPException) as exc_info:
        await reg.execute_chat_completion(body)

    assert "/v1/messages" in exc_info.value.detail or "/v1/responses" in exc_info.value.detail


@pytest.mark.asyncio
async def test_execute_completion_includes_supported_endpoints_in_error():
    reg = make_registry()
    ep = make_chat_endpoint(on_chat=True, on_messages=True, on_responses=True, on_completion=False)
    fake = _make_registered_model_with_endpoint(ep)
    body = MagicMock(spec=CompletionLegacyRequest)
    body.model = "gpt-4"

    with patch.object(reg.chat_completion_endpoints, "get_model", return_value=fake), pytest.raises(HTTPException) as exc_info:
        await reg.execute_completion(body)

    assert "/v1/chat/completions" in exc_info.value.detail or "/v1/messages" in exc_info.value.detail


@pytest.mark.asyncio
async def test_execute_ollama_chat_includes_supported_endpoints_in_error():
    reg = make_registry()
    ep = make_chat_endpoint(on_chat=True, on_messages=True, on_responses=True, on_completion=True, on_ollama=False)
    fake = _make_registered_model_with_endpoint(ep)
    body = MagicMock(spec=OllamaChatRequest)
    body.model = "gpt-4"

    with patch.object(reg.chat_completion_endpoints, "get_model", return_value=fake), pytest.raises(HTTPException) as exc_info:
        await reg.execute_ollama_chat(body)

    assert "/v1/chat/completions" in exc_info.value.detail or "/v1/messages" in exc_info.value.detail


@pytest.mark.asyncio
async def test_execute_messages_calls_on_messages_directly():
    reg = make_registry()
    response = MagicMock()
    ep = make_chat_endpoint(on_messages=True)
    ep.on_messages = AsyncMock(return_value=response)
    reg.register_chat_completion("ant-model", make_props(), ep, RegistrationOptions(origin="remote"))
    body = MagicMock(spec=MessagesRequest)
    body.model = "ant-model"
    body.model_dump_json.return_value = "{}"

    result = await reg.execute_messages(body)

    assert result is response


@pytest.mark.asyncio
async def test_execute_responses_calls_on_responses_directly():
    reg = make_registry()
    response = MagicMock()
    ep = make_chat_endpoint(on_responses=True)
    ep.on_responses = AsyncMock(return_value=response)
    reg.register_chat_completion("resp-model", make_props(), ep, RegistrationOptions(origin="remote"))
    body = MagicMock(spec=ResponsesRequest)
    body.model = "resp-model"
    body.model_dump_json.return_value = "{}"

    result = await reg.execute_responses(body)

    assert result is response


@pytest.mark.asyncio
async def test_execute_chat_completion_calls_on_chat_directly():
    reg = make_registry()
    response = MagicMock()
    ep = make_chat_endpoint(on_chat=True)
    ep.on_chat_completion = AsyncMock(return_value=response)
    reg.register_chat_completion("gpt-4", make_props(), ep, RegistrationOptions(origin="remote"))
    body = MagicMock(spec=ChatCompletionRequest)
    body.model = "gpt-4"
    body.model_dump_json.return_value = "{}"

    result = await reg.execute_chat_completion(body)

    assert result is response


@pytest.mark.asyncio
async def test_execute_completion_calls_on_completion_directly():
    reg = make_registry()
    response = MagicMock()
    ep = make_chat_endpoint(on_completion=True)
    ep.on_completion = AsyncMock(return_value=response)
    reg.register_chat_completion("gpt-35", make_props(), ep, RegistrationOptions(origin="remote"))
    body = MagicMock(spec=CompletionLegacyRequest)
    body.model = "gpt-35"

    result = await reg.execute_completion(body)

    assert result is response


@pytest.mark.asyncio
async def test_execute_ollama_chat_calls_on_ollama_directly():
    reg = make_registry()
    response = MagicMock()
    ep = make_chat_endpoint(on_ollama=True)
    ep.on_ollama_chat = AsyncMock(return_value=response)
    reg.register_chat_completion("oll-model", make_props(), ep, RegistrationOptions(origin="remote"))
    body = MagicMock(spec=OllamaChatRequest)
    body.model = "oll-model"

    result = await reg.execute_ollama_chat(body)

    assert result is response


@pytest.mark.asyncio
async def test_execute_embeddings_calls_on_request_directly():
    reg = make_registry()
    response = MagicMock()
    reg.register_embeddings(
        "emb",
        make_props(),
        SimpleEndpoint(on_request=AsyncMock(return_value=response)),
        RegistrationOptions(origin="remote"),
    )
    body = MagicMock(spec=EmbeddingRequest)
    body.model = "emb"

    result = await reg.execute_embeddings(body)

    assert result is response


@pytest.mark.asyncio
async def test_execute_images_generations_calls_on_request_directly():
    reg = make_registry()
    response = MagicMock()
    reg.register_image_generations(
        "img",
        make_props(),
        SimpleEndpoint(on_request=AsyncMock(return_value=response)),
        RegistrationOptions(origin="remote"),
    )
    body = MagicMock(spec=ImagesRequest)
    body.model = "img"
    body.model_dump_json.return_value = "{}"

    result = await reg.execute_images_generations(body)

    assert result is response


@pytest.mark.asyncio
async def test_execute_audio_speech_calls_on_request_directly():
    reg = make_registry()
    response = MagicMock()
    reg.register_audio_speech(
        "tts",
        make_props(),
        SimpleEndpoint(on_request=AsyncMock(return_value=response)),
        RegistrationOptions(origin="remote"),
    )
    body = MagicMock(spec=CreateSpeechRequest)
    body.model = "tts"
    body.model_dump_json.return_value = "{}"

    result = await reg.execute_audio_speech(body)

    assert result is response


@pytest.mark.asyncio
async def test_execute_audio_transcriptions_calls_on_request_directly():
    reg = make_registry()
    response = MagicMock()
    reg.register_audio_transcriptions(
        "stt",
        make_props(),
        SimpleEndpoint(on_request=AsyncMock(return_value=response)),
        RegistrationOptions(origin="remote"),
    )
    body = MagicMock(spec=CreateTranscriptionRequest)
    body.model = "stt"

    result = await reg.execute_audio_transcriptions(body)

    assert result is response


@pytest.mark.asyncio
async def test_execute_rerank_calls_on_request_directly():
    reg = make_registry()
    response = MagicMock()
    reg.register_rerank(
        "rnk",
        make_props(),
        SimpleEndpoint(on_request=AsyncMock(return_value=response)),
        RegistrationOptions(origin="remote"),
    )
    body = MagicMock(spec=RerankRequest)
    body.model = "rnk"
    body.model_dump_json.return_value = "{}"

    result = await reg.execute_rerank(body)

    assert result is response


@pytest.mark.asyncio
async def test_execute_custom_endpoints_calls_on_request_directly():
    reg = make_registry()
    response = MagicMock()
    reg.register_custom_endpoint(
        "my-svc",
        make_props(),
        CustomEndpoint(on_request=AsyncMock(return_value=response)),
        RegistrationOptions(origin="remote"),
    )
    request = MagicMock()

    result = await reg.execute_custom_endpoints("my-svc/path", request)

    assert result is response


@pytest.mark.asyncio
async def test_execute_mcp_endpoints_calls_on_request_directly():
    reg = make_registry()
    response = MagicMock()
    reg.register_mcp_endpoint(
        "mcp-svc",
        make_props(),
        McpEndpoint(on_request=AsyncMock(return_value=response)),
        RegistrationOptions(origin="remote"),
    )
    request = MagicMock()

    result = await reg.execute_mcp_endpoints("mcp-svc/path", request)

    assert result is response


@pytest.mark.asyncio
async def test_with_usage_streaming_logs_payload_when_enabled():
    reg = make_registry()
    ep = make_chat_endpoint()
    rid = reg.register_chat_completion("gpt-4", make_props(), ep, RegistrationOptions(origin="local"))
    model = reg.registry[rid].registered_model

    async def _gen():
        yield b"chunk1"
        yield b"chunk2"

    streaming = StreamingResponse(_gen(), media_type="text/event-stream", status_code=200)
    func = AsyncMock(return_value=streaming)
    result = await reg.with_usage(model, func, log_payload=True)
    assert isinstance(result, StreamingResponse)
    chunks = []

    async for chunk in result.body_iterator:
        chunks.append(chunk)

    assert b"chunk1" in chunks


@pytest.mark.asyncio
async def test_register_chat_completion_as_proxy_only_completions_no_chat_completions():
    reg = make_registry()
    opts = ProxyOptions(url="http://example.com/v1/completions")
    reg.register_chat_completion_as_proxy(
        model="legacy-model",
        props=make_props(),
        chat_completions=None,
        completions=opts,
        responses=None,
        messages=None,
        ollama_chat=None,
        registration_options=None,
    )
    ep = reg.chat_completion_endpoints.get_model("legacy-model")
    assert ep is not None
    assert ep.endpoint.on_chat_completion is None
    mock_resp = MagicMock(spec=StreamingResponse)

    with patch("server.endpointregistry.post_json", new_callable=AsyncMock, return_value=mock_resp):
        body = MagicMock(spec=CompletionLegacyRequest)
        result = await ep.endpoint.on_completion(body, None)  # pyright: ignore[reportOptionalCall]

    assert result is mock_resp


def test_update_models_prev_model_not_in_registry_is_skipped():
    reg = make_registry()

    ghost = Model(id="ghost-id", name="ghost", type="llm", props=make_props(), usage=0)
    reg.parent_infra.send_models_list.reset_mock()  # pyright: ignore[reportAttributeAccessIssue]

    reg.update_models([ghost], [], "http://api.example.com/", "mykey")

    assert reg.parent_infra.send_models_list.call_count == 0  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.asyncio
async def test_execute_messages_error_with_no_supported_endpoints():
    reg = make_registry()
    ep = make_chat_endpoint(on_chat=False, on_responses=False, on_completion=False, on_messages=False)
    fake = _make_registered_model_with_endpoint(ep)
    body = MagicMock(spec=MessagesRequest)
    body.model = "gpt-4"
    body.model_dump_json.return_value = "{}"

    with patch.object(reg.chat_completion_endpoints, "get_model", return_value=fake), pytest.raises(HTTPException) as exc_info:
        await reg.execute_messages(body)

    assert "Given model not support" in exc_info.value.detail
    assert "/v1/responses" not in exc_info.value.detail
    assert "/v1/chat/completions" not in exc_info.value.detail
    assert "/v1/completions" not in exc_info.value.detail


@pytest.mark.asyncio
async def test_execute_responses_error_with_no_supported_endpoints():
    reg = make_registry()
    ep = make_chat_endpoint(on_chat=False, on_responses=False, on_completion=False, on_messages=False)
    fake = _make_registered_model_with_endpoint(ep)
    body = MagicMock(spec=ResponsesRequest)
    body.model = "gpt-4"
    body.model_dump_json.return_value = "{}"

    with patch.object(reg.chat_completion_endpoints, "get_model", return_value=fake), pytest.raises(HTTPException) as exc_info:
        await reg.execute_responses(body)

    assert "Given model not support" in exc_info.value.detail
    assert "/v1/messages" not in exc_info.value.detail
    assert "/v1/chat/completions" not in exc_info.value.detail
    assert "/v1/completions" not in exc_info.value.detail


@pytest.mark.asyncio
async def test_execute_chat_completion_error_with_no_supported_endpoints():
    reg = make_registry()
    ep = make_chat_endpoint(on_chat=False, on_responses=False, on_completion=False, on_messages=False)
    fake = _make_registered_model_with_endpoint(ep)
    body = MagicMock(spec=ChatCompletionRequest)
    body.model = "gpt-4"
    body.model_dump_json.return_value = "{}"

    with patch.object(reg.chat_completion_endpoints, "get_model", return_value=fake), pytest.raises(HTTPException) as exc_info:
        await reg.execute_chat_completion(body)

    assert "Given model not support" in exc_info.value.detail
    assert "/v1/messages" not in exc_info.value.detail
    assert "/v1/responses" not in exc_info.value.detail
    assert "/v1/completions" not in exc_info.value.detail


@pytest.mark.asyncio
async def test_execute_completion_error_with_no_supported_endpoints():
    reg = make_registry()
    ep = make_chat_endpoint(on_chat=False, on_responses=False, on_completion=False, on_messages=False)
    fake = _make_registered_model_with_endpoint(ep)
    body = MagicMock(spec=CompletionLegacyRequest)
    body.model = "gpt-4"

    with patch.object(reg.chat_completion_endpoints, "get_model", return_value=fake), pytest.raises(HTTPException) as exc_info:
        await reg.execute_completion(body)

    assert "Given model not support" in exc_info.value.detail
    assert "/v1/messages" not in exc_info.value.detail
    assert "/v1/responses" not in exc_info.value.detail
    assert "/v1/chat/completions" not in exc_info.value.detail


@pytest.mark.asyncio
async def test_execute_ollama_chat_error_with_no_supported_endpoints():
    reg = make_registry()
    ep = make_chat_endpoint(on_chat=False, on_responses=False, on_completion=False, on_messages=False, on_ollama=False)
    fake = _make_registered_model_with_endpoint(ep)
    body = MagicMock(spec=OllamaChatRequest)
    body.model = "gpt-4"

    with patch.object(reg.chat_completion_endpoints, "get_model", return_value=fake), pytest.raises(HTTPException) as exc_info:
        await reg.execute_ollama_chat(body)

    assert "Given model not support" in exc_info.value.detail
    assert "/v1/messages" not in exc_info.value.detail
    assert "/v1/responses" not in exc_info.value.detail
    assert "/v1/chat/completions" not in exc_info.value.detail
    assert "/v1/completions" not in exc_info.value.detail
