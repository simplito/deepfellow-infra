# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

import asyncio
import json
import logging
import time
from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, call, patch

import aiohttp
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry
from pydantic import BaseModel, ValidationError

import server.endpointregistry as _er_module
from server.endpointregistry import (
    ChatCompletionEndpoint,
    CustomEndpoint,
    Endpoint,
    EndpointRegistry,
    McpEndpoint,
    McpSseSessionStore,
    ProxyOptions,
    RegisteredModel,
    RegistrationOptions,
    ResponseItemStore,
    SimpleEndpoint,
    _classify_error,  # pyright: ignore[reportPrivateUsage]
    _make_request_with_reauth,  # pyright: ignore[reportPrivateUsage]
    _read_body_with_limit,  # pyright: ignore[reportPrivateUsage]
    _rewrite_sse_endpoint_events,  # pyright: ignore[reportPrivateUsage]
    _unwrap_json_text,  # pyright: ignore[reportPrivateUsage]
    post_form,
    post_json,
    post_json_rerank_sglang,
    post_json_responses,
)
from server.error_handlers import register_exception_handlers
from server.metrics_registry import MetricsRegistry
from server.models.api import (
    ChatCompletionRequest,
    CompletionLegacyRequest,
    CreateSpeechRequest,
    CreateTranscriptionRequest,
    EmbeddingRequest,
    ImagesRequest,
    ItemReference,
    McpToolInfo,
    MessagesRequest,
    Model,
    ModelProps,
    OllamaChatRequest,
    Reasoning,
    RerankRequest,
    ResponsesRequest,
    SystemOneRequest,
)
from server.utils.core import HttpClientError
from server.utils.sentry import UpstreamError, attached_tags, is_reported
from server.websockets.models import UsageChangeRequest


def make_props(
    private: bool = False,
    type: str = "llm",
    endpoints: list[str] | None = None,
    context_window: int | None = None,
    max_context_window: int | None = None,
    supports_vision: bool | None = None,
    supports_audio: bool | None = None,
) -> ModelProps:
    return ModelProps(
        private=private,
        type=type,
        endpoints=endpoints or ["chat"],
        context_window=context_window,
        max_context_window=max_context_window,
        supports_vision=supports_vision,
        supports_audio=supports_audio,
    )


def make_parent_infra() -> MagicMock:
    pi = MagicMock()
    pi.send_models_list = MagicMock()
    pi.send_usage = MagicMock()
    pi.send_warm = MagicMock()
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


def make_registry(metrics_registry: MetricsRegistry | None = None) -> EndpointRegistry:
    config = MagicMock()
    config.is_log_payloads_enabled.return_value = False
    parent_infra = make_parent_infra()
    model_tester = MagicMock()
    return EndpointRegistry(
        config=config,
        parent_infra=parent_infra,
        model_tester=model_tester,
        metrics_registry=metrics_registry or make_metrics_registry(),
    )


async def _async_byte_stream(data: bytes):
    if data:
        yield data


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


def test_add_model_with_alias_makes_model_reachable_under_both_names():
    ep = make_endpoint()

    rid = ep.add_model(
        "gpt-4", make_props(), SimpleEndpoint(on_request=AsyncMock()), "llm", RegistrationOptions(origin="local", alias="smart")
    )

    original = ep.get_model("gpt-4")
    aliased = ep.get_model("smart")
    assert original is not None
    assert original is aliased
    assert original.id == rid
    assert {m.id for m in ep.get_models()} == {"gpt-4", "smart"}


def test_add_model_alias_equal_to_model_id_is_ignored():
    ep = make_endpoint()

    ep.add_model("gpt-4", make_props(), SimpleEndpoint(on_request=AsyncMock()), "llm", RegistrationOptions(origin="local", alias="gpt-4"))

    assert list(ep.models) == ["gpt-4"]
    assert len(ep.models["gpt-4"]) == 1


def test_remove_model_with_alias_removes_both_names():
    ep = make_endpoint()
    rid = ep.add_model(
        "gpt-4", make_props(), SimpleEndpoint(on_request=AsyncMock()), "llm", RegistrationOptions(origin="local", alias="smart")
    )

    ep.remove_model("gpt-4", rid)

    assert ep.models == {}
    assert rid not in ep.registry


def test_remove_model_with_alias_keeps_alias_for_other_registrations():
    ep = make_endpoint()
    first = ep.add_model(
        "gpt-4", make_props(), SimpleEndpoint(on_request=AsyncMock()), "llm", RegistrationOptions(origin="local", alias="smart")
    )
    second = ep.add_model(
        "gpt-4", make_props(), SimpleEndpoint(on_request=AsyncMock()), "llm", RegistrationOptions(origin="local", alias="smart")
    )

    ep.remove_model("gpt-4", first)

    assert list(ep.models["smart"]) == [second]


def test_list_models_reports_aliased_registration_once_under_primary_name():
    ep = make_endpoint()
    ep.add_model("gpt-4", make_props(), SimpleEndpoint(on_request=AsyncMock()), "llm", RegistrationOptions(origin="local", alias="smart"))

    listed = ep.list_models()

    assert [m.name for m in listed] == ["gpt-4"]


def test_get_models_reports_alias_of_only_on_alias_entry():
    ep = make_endpoint()
    ep.add_model("gpt-4", make_props(), SimpleEndpoint(on_request=AsyncMock()), "llm", RegistrationOptions(origin="local", alias="smart"))
    ep.add_model("llama3", make_props(), SimpleEndpoint(on_request=AsyncMock()), "llm", RegistrationOptions(origin="local"))

    alias_of = {m.id: m.props.alias_of for m in ep.get_models()}

    assert alias_of == {"gpt-4": [], "smart": ["gpt-4"], "llama3": []}


def test_get_models_alias_of_a_renamed_model_reports_the_registered_name():
    ep = make_endpoint()
    ep.add_model(
        "corp-gpt", make_props(), SimpleEndpoint(on_request=AsyncMock()), "llm", RegistrationOptions(origin="local", alias="smart")
    )

    alias_of = {m.id: m.props.alias_of for m in ep.get_models()}

    assert alias_of == {"corp-gpt": [], "smart": ["corp-gpt"]}


def test_get_models_alias_shared_by_models_of_several_services_lists_each_model_once():
    ep = make_endpoint()
    for model_id, owned_by in (("gpt-4", "openai"), ("gpt-4", "openai"), ("llama3", "ollama"), ("gemma", "vllm")):
        ep.add_model(
            model_id,
            make_props(),
            SimpleEndpoint(on_request=AsyncMock()),
            "llm",
            RegistrationOptions(origin="local", alias="smart", owned_by=owned_by),
        )

    smart = next(m for m in ep.get_models() if m.id == "smart")

    assert smart.props.alias_of == ["gemma", "gpt-4", "llama3"]


def test_get_models_alias_taking_name_of_installed_model_lists_only_aliased_models():
    ep = make_endpoint()
    ep.add_model("gpt-4", make_props(), SimpleEndpoint(on_request=AsyncMock()), "llm", RegistrationOptions(origin="local"))
    ep.add_model("llama3", make_props(), SimpleEndpoint(on_request=AsyncMock()), "llm", RegistrationOptions(origin="local", alias="gpt-4"))

    alias_of = {m.id: m.props.alias_of for m in ep.get_models()}

    assert alias_of == {"gpt-4": ["llama3"], "llama3": []}


def test_get_models_alias_of_drops_model_once_its_registration_is_removed():
    ep = make_endpoint()
    ep.add_model("gpt-4", make_props(), SimpleEndpoint(on_request=AsyncMock()), "llm", RegistrationOptions(origin="local", alias="smart"))
    llama = ep.add_model(
        "llama3", make_props(), SimpleEndpoint(on_request=AsyncMock()), "llm", RegistrationOptions(origin="local", alias="smart")
    )

    ep.remove_model("llama3", llama)

    smart = next(m for m in ep.get_models() if m.id == "smart")
    assert smart.props.alias_of == ["gpt-4"]


def test_registry_get_model_reports_alias_of():
    registry = make_registry()
    registry.embeddings_endpoints.add_model(
        "text-embedding-3-small",
        make_props(type="embedding"),
        SimpleEndpoint(on_request=AsyncMock()),
        "embedding",
        RegistrationOptions(origin="local", alias="embedder"),
    )

    assert registry.get_model("embedder").props.alias_of == ["text-embedding-3-small"]
    assert registry.get_model("text-embedding-3-small").props.alias_of == []


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


def test_get_model_prefers_higher_capacity_over_lower_when_not_saturated():
    ep = make_endpoint()
    ep.add_model(
        "gpt-4",
        make_props(),
        SimpleEndpoint(on_request=AsyncMock()),
        "llm",
        RegistrationOptions(origin="local", usage=3, capacity_state=4),
    )
    rid2 = ep.add_model(
        "gpt-4",
        make_props(),
        SimpleEndpoint(on_request=AsyncMock()),
        "llm",
        RegistrationOptions(origin="local", usage=0, capacity_state=10),
    )

    result = ep.get_model("gpt-4")

    assert result is not None
    assert result.id == rid2


def test_get_model_spills_over_when_top_ranked_local_instance_saturated():
    ep = make_endpoint()
    ep.add_model(
        "gpt-4",
        make_props(),
        SimpleEndpoint(on_request=AsyncMock()),
        "llm",
        RegistrationOptions(origin="local", usage=4, capacity_state=4),
    )
    rid2 = ep.add_model(
        "gpt-4",
        make_props(),
        SimpleEndpoint(on_request=AsyncMock()),
        "llm",
        RegistrationOptions(origin="local", usage=1, capacity_state=2),
    )

    result = ep.get_model("gpt-4")

    assert result is not None
    assert result.id == rid2


def test_get_model_prefers_warm_over_higher_capacity_cold():
    ep = make_endpoint()
    ep.add_model(
        "gpt-4",
        make_props(),
        SimpleEndpoint(on_request=AsyncMock()),
        "llm",
        RegistrationOptions(origin="local", capacity_state=100, warm=False),
    )
    rid_warm = ep.add_model(
        "gpt-4",
        make_props(),
        SimpleEndpoint(on_request=AsyncMock()),
        "llm",
        RegistrationOptions(origin="local", capacity_state=1, warm=True),
    )

    result = ep.get_model("gpt-4")

    assert result is not None
    assert result.id == rid_warm


def test_get_model_prefers_higher_capacity_among_equally_warm():
    ep = make_endpoint()
    ep.add_model(
        "gpt-4",
        make_props(),
        SimpleEndpoint(on_request=AsyncMock()),
        "llm",
        RegistrationOptions(origin="local", capacity_state=2, warm=True),
    )
    rid_bigger = ep.add_model(
        "gpt-4",
        make_props(),
        SimpleEndpoint(on_request=AsyncMock()),
        "llm",
        RegistrationOptions(origin="local", capacity_state=8, warm=True),
    )

    result = ep.get_model("gpt-4")

    assert result is not None
    assert result.id == rid_bigger


def test_get_model_packs_onto_same_instance_among_equal_warmth_and_capacity():
    """Requests must pack onto one instance of an equally-ranked tier until it saturates,
    not spread randomly across every instance in that tier - see `_pick_ranked_model`."""
    ep = make_endpoint()
    ep.add_model(
        "gpt-4",
        make_props(),
        SimpleEndpoint(on_request=AsyncMock()),
        "llm",
        RegistrationOptions(origin="local", capacity_state=4, warm=True),
    )
    ep.add_model(
        "gpt-4",
        make_props(),
        SimpleEndpoint(on_request=AsyncMock()),
        "llm",
        RegistrationOptions(origin="local", capacity_state=4, warm=True),
    )

    seen: set[str] = set()
    for _ in range(50):
        result = ep.get_model("gpt-4")
        assert result is not None
        seen.add(result.id)

    assert len(seen) == 1


def test_get_model_local_registration_usable_up_to_full_capacity():
    ep = make_endpoint()
    rid = ep.add_model(
        "gpt-4",
        make_props(),
        SimpleEndpoint(on_request=AsyncMock()),
        "llm",
        RegistrationOptions(origin="local", capacity_state=4, usage=3),
    )

    result = ep.get_model("gpt-4")

    assert result is not None
    assert result.id == rid


def test_get_model_mesh_registration_reserves_headroom_below_full_capacity():
    ep = make_endpoint()
    ep.add_model(
        "gpt-4",
        make_props(),
        SimpleEndpoint(on_request=AsyncMock()),
        "llm",
        RegistrationOptions(origin="https://peer", capacity_state=10, usage=9, owned_by="mesh"),
    )
    rid_fallback = ep.add_model(
        "gpt-4",
        make_props(),
        SimpleEndpoint(on_request=AsyncMock()),
        "llm",
        RegistrationOptions(origin="local", capacity_state=1, usage=0),
    )

    result = ep.get_model("gpt-4")

    assert result is not None
    assert result.id == rid_fallback


def test_get_model_mesh_registration_wins_below_its_headroom_threshold():
    ep = make_endpoint()
    rid_mesh = ep.add_model(
        "gpt-4",
        make_props(),
        SimpleEndpoint(on_request=AsyncMock()),
        "llm",
        RegistrationOptions(origin="https://peer", capacity_state=10, usage=5, owned_by="mesh"),
    )
    ep.add_model(
        "gpt-4",
        make_props(),
        SimpleEndpoint(on_request=AsyncMock()),
        "llm",
        RegistrationOptions(origin="local", capacity_state=1, usage=0),
    )

    result = ep.get_model("gpt-4")

    assert result is not None
    assert result.id == rid_mesh


@pytest.mark.parametrize("bad_capacity", [0, -1])
def test_registered_model_rejects_non_positive_capacity(bad_capacity: int) -> None:
    """Construction-time guard: a zero/negative capacity is a programming error, not "unknown" - it
    must fail loudly here rather than silently saturate the instance (`num_parallel=0` was exactly
    this bug, now fixed upstream in ollama_service, but this is the last line of defense)."""
    with pytest.raises(ValidationError):
        RegisteredModel(
            id="fake-id",
            name="gpt-4",
            origin="local",
            props=make_props(),
            type="llm",
            endpoint=SimpleEndpoint(on_request=AsyncMock()),
            usage=0,
            created=0,
            owned_by="local",
            capacity_state=bad_capacity,
        )


@pytest.mark.parametrize("bad_capacity", [0, -1])
def test_model_wire_schema_coerces_non_positive_capacity_to_unknown(bad_capacity: int) -> None:
    """Unlike the internal `RegisteredModel` boundary, a mesh peer is untrusted input: a
    malformed `capacity` must degrade to "unknown" rather than raise and break mesh sync."""
    model = Model(id="id-1", name="gpt-4", type="llm", props=make_props(), usage=0, capacity=bad_capacity, capacity_known=True)

    assert model.capacity is None
    assert model.capacity_known is False


def test_model_wire_schema_coerces_string_capacity_to_unknown() -> None:
    """A peer sending `capacity` as a non-int (e.g. `"0"`, before Pydantic's own coercion runs)
    must degrade to "unknown" rather than crash the validator itself with a raw TypeError from
    comparing a str to an int."""
    model = Model.model_validate(
        {"id": "id-1", "name": "gpt-4", "type": "llm", "props": make_props().model_dump(), "usage": 0, "capacity": "0"}
    )

    assert model.capacity is None
    assert model.capacity_known is False


def test_model_wire_schema_coerces_non_numeric_string_capacity_to_unknown() -> None:
    """A peer sending a genuinely non-numeric `capacity` string must degrade to "unknown"
    rather than crash the validator itself with a raw ValueError from `float()`."""
    model = Model.model_validate(
        {"id": "id-1", "name": "gpt-4", "type": "llm", "props": make_props().model_dump(), "usage": 0, "capacity": "garbage"}
    )

    assert model.capacity is None
    assert model.capacity_known is False


@pytest.mark.parametrize("bool_capacity", [True, False])
def test_model_wire_schema_coerces_bool_capacity_to_unknown(bool_capacity: bool) -> None:
    """A bool is technically an int subclass but never a legitimate capacity; it must degrade
    to "unknown" rather than silently pass through pydantic's own bool->int coercion."""
    model = Model.model_validate(
        {"id": "id-1", "name": "gpt-4", "type": "llm", "props": make_props().model_dump(), "usage": 0, "capacity": bool_capacity}
    )

    assert model.capacity is None
    assert model.capacity_known is False


def test_model_wire_schema_accepts_whole_number_float_string_capacity() -> None:
    """A numeric string like "5.0" is a legitimate capacity serialized as a whole-number float
    and must reach pydantic's normal int coercion like an equivalent float or int would."""
    model = Model.model_validate(
        {
            "id": "id-1",
            "name": "gpt-4",
            "type": "llm",
            "props": make_props().model_dump(),
            "usage": 0,
            "capacity": "5.0",
            "capacity_known": True,
        }
    )

    assert model.capacity == 5
    assert model.capacity_known is True


def test_model_wire_schema_coerces_absurdly_large_capacity_to_unknown() -> None:
    """A capacity far beyond what any real backend could serve is a corrupted/malicious value,
    not a legitimate capacity, and must degrade to "unknown" rather than be accepted verbatim."""
    model = Model.model_validate(
        {"id": "id-1", "name": "gpt-4", "type": "llm", "props": make_props().model_dump(), "usage": 0, "capacity": 1e20}
    )

    assert model.capacity is None
    assert model.capacity_known is False


def test_model_wire_schema_rejects_non_dict_item() -> None:
    """A peer sending a non-dict item in place of a model object must raise a clean
    ValidationError, not crash the validator itself with a raw AttributeError from calling
    `.get()` on it."""
    with pytest.raises(ValidationError):
        Model.model_validate("notadict")


def test_model_wire_schema_preserves_genuinely_unbounded_capacity() -> None:
    """`capacity=None, capacity_known=True` (e.g. a cloud/proxy peer) must survive the `Model`
    boundary as-is - it must not collapse to "unknown" (capacity_known=False) alongside the
    malformed-input case above, since that would flip it to the worst rather than best rank."""
    model = Model(id="id-1", name="gpt-4", type="llm", props=make_props(), usage=0, capacity=None, capacity_known=True)

    assert model.capacity is None
    assert model.capacity_known is True


def test_get_model_capacity_none_treated_as_unbounded():
    ep = make_endpoint()
    rid = ep.add_model(
        "gpt-4",
        make_props(),
        SimpleEndpoint(on_request=AsyncMock()),
        "llm",
        RegistrationOptions(origin="local", capacity_state="unbounded", usage=1000),
    )

    result = ep.get_model("gpt-4")

    assert result is not None
    assert result.id == rid


def test_get_model_falls_back_to_saturated_candidate_when_only_one_exists():
    ep = make_endpoint()
    rid = ep.add_model(
        "gpt-4",
        make_props(),
        SimpleEndpoint(on_request=AsyncMock()),
        "llm",
        RegistrationOptions(origin="local", capacity_state=2, usage=2),
    )

    result = ep.get_model("gpt-4")

    assert result is not None
    assert result.id == rid


def test_get_model_spills_over_from_saturated_warm_tier_to_available_cold_tier():
    ep = make_endpoint()
    ep.add_model(
        "gpt-4",
        make_props(),
        SimpleEndpoint(on_request=AsyncMock()),
        "llm",
        RegistrationOptions(origin="local", capacity_state=2, usage=2, warm=True),
    )
    rid_cold = ep.add_model(
        "gpt-4",
        make_props(),
        SimpleEndpoint(on_request=AsyncMock()),
        "llm",
        RegistrationOptions(origin="local", capacity_state=2, usage=0, warm=False),
    )

    result = ep.get_model("gpt-4")

    assert result is not None
    assert result.id == rid_cold


def test_get_model_falls_back_to_least_loaded_among_multiple_saturated():
    ep = make_endpoint()
    ep.add_model(
        "gpt-4",
        make_props(),
        SimpleEndpoint(on_request=AsyncMock()),
        "llm",
        RegistrationOptions(origin="local", capacity_state=2, usage=9),
    )
    rid_less_loaded = ep.add_model(
        "gpt-4",
        make_props(),
        SimpleEndpoint(on_request=AsyncMock()),
        "llm",
        RegistrationOptions(origin="local", capacity_state=2, usage=3),
    )

    result = ep.get_model("gpt-4")

    assert result is not None
    assert result.id == rid_less_loaded


def test_get_model_spills_over_within_same_tier_once_lower_id_candidate_saturates():
    """The intra-tier packing loop (`_pick_ranked_model`) must actually advance past a saturated,
    lower-id candidate to the next id-ordered one in the same tier - not just always return the
    first member, which would make the "then spill over" half of packing untested."""
    ep = make_endpoint()
    rid_a = ep.add_model(
        "gpt-4",
        make_props(),
        SimpleEndpoint(on_request=AsyncMock()),
        "llm",
        RegistrationOptions(origin="local", capacity_state=4, usage=0, warm=True),
    )
    rid_b = ep.add_model(
        "gpt-4",
        make_props(),
        SimpleEndpoint(on_request=AsyncMock()),
        "llm",
        RegistrationOptions(origin="local", capacity_state=4, usage=0, warm=True),
    )
    lower_id, higher_id = sorted([rid_a, rid_b])
    ep.models["gpt-4"][lower_id].usage = 4  # saturate whichever candidate packing would try first

    result = ep.get_model("gpt-4")

    assert result is not None
    assert result.id == higher_id


def test_get_model_registration_id_bypasses_ranking_even_when_saturated():
    """A caller pinning a specific `registration_id` (e.g. sticky routing) must get that exact
    registration back regardless of ranking or saturation - the ranking/packing logic only
    applies when no explicit target is given."""
    ep = make_endpoint()
    rid_saturated_cold = ep.add_model(
        "gpt-4",
        make_props(),
        SimpleEndpoint(on_request=AsyncMock()),
        "llm",
        RegistrationOptions(origin="local", capacity_state=2, usage=2, warm=False),
    )
    ep.add_model(
        "gpt-4",
        make_props(),
        SimpleEndpoint(on_request=AsyncMock()),
        "llm",
        RegistrationOptions(origin="local", capacity_state=2, usage=0, warm=True),
    )

    result = ep.get_model("gpt-4", registration_id=rid_saturated_cold)

    assert result is not None
    assert result.id == rid_saturated_cold


def test_get_model_mesh_registration_capacity_one_boundary():
    """A mesh registration with capacity=1 has a saturation threshold of 0.9 (90% of 1); usage=0
    must still be usable (below threshold), while usage=1 must be treated as saturated."""
    ep = make_endpoint()
    rid_mesh = ep.add_model(
        "gpt-4",
        make_props(),
        SimpleEndpoint(on_request=AsyncMock()),
        "llm",
        RegistrationOptions(origin="https://peer", capacity_state=1, usage=0, owned_by="mesh"),
    )

    result = ep.get_model("gpt-4")

    assert result is not None
    assert result.id == rid_mesh

    ep.models["gpt-4"][rid_mesh].usage = 1
    rid_fallback = ep.add_model(
        "gpt-4",
        make_props(),
        SimpleEndpoint(on_request=AsyncMock()),
        "llm",
        RegistrationOptions(origin="local", capacity_state=1, usage=0),
    )

    result = ep.get_model("gpt-4")

    assert result is not None
    assert result.id == rid_fallback


def test_get_model_saturated_fallback_random_tie_break_among_equal_usage():
    ep = make_endpoint()
    rid1 = ep.add_model(
        "gpt-4",
        make_props(),
        SimpleEndpoint(on_request=AsyncMock()),
        "llm",
        RegistrationOptions(origin="local", capacity_state=2, usage=2),
    )
    rid2 = ep.add_model(
        "gpt-4",
        make_props(),
        SimpleEndpoint(on_request=AsyncMock()),
        "llm",
        RegistrationOptions(origin="local", capacity_state=2, usage=2),
    )

    seen: set[str] = set()
    for _ in range(50):
        result = ep.get_model("gpt-4")
        assert result is not None
        seen.add(result.id)

    assert seen == {rid1, rid2}


def test_get_model_saturated_fallback_searches_across_all_tiers():
    ep = make_endpoint()
    ep.add_model(
        "gpt-4",
        make_props(),
        SimpleEndpoint(on_request=AsyncMock()),
        "llm",
        RegistrationOptions(origin="local", capacity_state=2, usage=2, warm=True),
    )
    rid_cold_less_loaded = ep.add_model(
        "gpt-4",
        make_props(),
        SimpleEndpoint(on_request=AsyncMock()),
        "llm",
        RegistrationOptions(origin="local", capacity_state=1, usage=1, warm=False),
    )

    result = ep.get_model("gpt-4")

    assert result is not None
    assert result.id == rid_cold_less_loaded


def test_get_model_prefers_known_capacity_over_unknown_capacity():
    ep = make_endpoint()
    rid_known = ep.add_model(
        "gpt-4",
        make_props(),
        SimpleEndpoint(on_request=AsyncMock()),
        "llm",
        RegistrationOptions(origin="local", capacity_state=1, usage=0),
    )
    ep.add_model(
        "gpt-4",
        make_props(),
        SimpleEndpoint(on_request=AsyncMock()),
        "llm",
        RegistrationOptions(origin="local", capacity_state="unknown", usage=0),
    )

    result = ep.get_model("gpt-4")

    assert result is not None
    assert result.id == rid_known


def test_get_model_prefers_known_finite_capacity_over_genuinely_unbounded():
    """A genuinely unbounded backend (e.g. a cloud/proxy model) must not monopolize traffic ahead
    of a known, finite-capacity local backend - it ranks as overflow, used only once known-capacity
    backends saturate. See `_rank_key`'s docstring for the reasoning and the tradeoff this implies.
    """
    ep = make_endpoint()
    rid_known = ep.add_model(
        "gpt-4",
        make_props(),
        SimpleEndpoint(on_request=AsyncMock()),
        "llm",
        RegistrationOptions(origin="local", capacity_state=1, usage=0),
    )
    ep.add_model(
        "gpt-4",
        make_props(),
        SimpleEndpoint(on_request=AsyncMock()),
        "llm",
        RegistrationOptions(origin="local", capacity_state="unbounded", usage=0),
    )

    result = ep.get_model("gpt-4")

    assert result is not None
    assert result.id == rid_known


def test_get_model_treats_genuinely_unbounded_capacity_as_better_than_unknown():
    """Unbounded still outranks the "failed to determine capacity" tier - it's a real, deliberate
    capability, not a parse failure, so it must not be lumped in with genuine unknowns.
    """
    ep = make_endpoint()
    rid_unbounded = ep.add_model(
        "gpt-4",
        make_props(),
        SimpleEndpoint(on_request=AsyncMock()),
        "llm",
        RegistrationOptions(origin="local", capacity_state="unbounded", usage=0),
    )
    ep.add_model(
        "gpt-4",
        make_props(),
        SimpleEndpoint(on_request=AsyncMock()),
        "llm",
        RegistrationOptions(origin="local", capacity_state="unknown", usage=0),
    )

    result = ep.get_model("gpt-4")

    assert result is not None
    assert result.id == rid_unbounded


def test_rank_key_raises_on_invalid_capacity_state():
    """Defensive fallback: `capacity_state` is pydantic-validated at construction, but `_rank_key`
    must still fail loudly rather than silently mis-ranking if an invalid value ever slips through.
    """
    bogus = SimpleNamespace(capacity_state="bogus", warm=True)

    with pytest.raises(ValueError, match="Unknown capacity_state"):
        _er_module._rank_key(bogus)  # pyright: ignore[reportPrivateUsage, reportArgumentType]


def test_get_model_with_multiple_unbounded_candidates_picks_least_loaded():
    ep = make_endpoint()
    ep.add_model(
        "gpt-4",
        make_props(),
        SimpleEndpoint(on_request=AsyncMock()),
        "llm",
        RegistrationOptions(origin="local", capacity_state="unbounded", usage=5),
    )
    rid_least_loaded = ep.add_model(
        "gpt-4",
        make_props(),
        SimpleEndpoint(on_request=AsyncMock()),
        "llm",
        RegistrationOptions(origin="local", capacity_state="unbounded", usage=1),
    )
    ep.add_model(
        "gpt-4",
        make_props(),
        SimpleEndpoint(on_request=AsyncMock()),
        "llm",
        RegistrationOptions(origin="local", capacity_state="unbounded", usage=3),
    )

    result = ep.get_model("gpt-4")

    assert result is not None
    assert result.id == rid_least_loaded


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


def test_get_model_context_window_returns_none_when_undeclared():
    ep = make_endpoint()
    ep.add_model(
        "custom-model",
        make_props(),
        SimpleEndpoint(on_request=AsyncMock()),
        "custom",
        RegistrationOptions(origin="local", id="id-1"),
    )

    assert ep.get_model_context_window(ep.models["custom-model"]) is None


def test_get_max_context_window_returns_none_when_undeclared():
    """Reporting 0 here made df-server store 0 as its chunk-size ceiling and fail every ingest."""
    ep = make_endpoint()
    ep.add_model(
        "custom-model",
        make_props(),
        SimpleEndpoint(on_request=AsyncMock()),
        "custom",
        RegistrationOptions(origin="local", id="id-1"),
    )

    assert ep.get_max_context_window(ep.models["custom-model"]) is None


def test_get_max_context_window_ignores_registrations_without_a_window():
    ep = make_endpoint()
    ep.add_model(
        "gpt-4",
        make_props(),
        SimpleEndpoint(on_request=AsyncMock()),
        "llm",
        RegistrationOptions(origin="local", id="id-1"),
    )
    ep.add_model(
        "gpt-4",
        make_props(max_context_window=16000),
        SimpleEndpoint(on_request=AsyncMock()),
        "llm",
        RegistrationOptions(origin="local", id="id-2"),
    )

    assert ep.get_max_context_window(ep.models["gpt-4"]) == 16000


def test_get_model_supports_vision_true_when_any_registration_true():
    ep = make_endpoint()
    ep.add_model(
        "vlm",
        make_props(supports_vision=None),
        SimpleEndpoint(on_request=AsyncMock()),
        "llm",
        RegistrationOptions(origin="local", id="id-1"),
    )
    ep.add_model(
        "vlm",
        make_props(supports_vision=True),
        SimpleEndpoint(on_request=AsyncMock()),
        "llm",
        RegistrationOptions(origin="local", id="id-2"),
    )

    assert ep.get_model_supports_vision(ep.models["vlm"]) is True


def test_get_model_supports_vision_false_when_all_registrations_false():
    ep = make_endpoint()
    ep.add_model(
        "text-only",
        make_props(supports_vision=False),
        SimpleEndpoint(on_request=AsyncMock()),
        "llm",
        RegistrationOptions(origin="local", id="id-1"),
    )
    ep.add_model(
        "text-only",
        make_props(supports_vision=False),
        SimpleEndpoint(on_request=AsyncMock()),
        "llm",
        RegistrationOptions(origin="local", id="id-2"),
    )

    assert ep.get_model_supports_vision(ep.models["text-only"]) is False


def test_get_model_supports_vision_none_when_mixed_none_and_false():
    ep = make_endpoint()
    ep.add_model(
        "unknown-model",
        make_props(supports_vision=None),
        SimpleEndpoint(on_request=AsyncMock()),
        "llm",
        RegistrationOptions(origin="local", id="id-1"),
    )
    ep.add_model(
        "unknown-model",
        make_props(supports_vision=False),
        SimpleEndpoint(on_request=AsyncMock()),
        "llm",
        RegistrationOptions(origin="local", id="id-2"),
    )

    assert ep.get_model_supports_vision(ep.models["unknown-model"]) is None


def test_get_model_supports_vision_none_when_undeclared():
    ep = make_endpoint()
    ep.add_model(
        "custom-model",
        make_props(),
        SimpleEndpoint(on_request=AsyncMock()),
        "custom",
        RegistrationOptions(origin="local", id="id-1"),
    )

    assert ep.get_model_supports_vision(ep.models["custom-model"]) is None


def test_get_model_supports_audio_true_when_any_registration_true():
    ep = make_endpoint()
    ep.add_model(
        "alm",
        make_props(supports_audio=None),
        SimpleEndpoint(on_request=AsyncMock()),
        "llm",
        RegistrationOptions(origin="local", id="id-1"),
    )
    ep.add_model(
        "alm",
        make_props(supports_audio=True),
        SimpleEndpoint(on_request=AsyncMock()),
        "llm",
        RegistrationOptions(origin="local", id="id-2"),
    )

    assert ep.get_model_supports_audio(ep.models["alm"]) is True


def test_build_api_model_includes_modality_support():
    ep = make_endpoint()
    ep.add_model(
        "vlm",
        make_props(supports_vision=True, supports_audio=False),
        SimpleEndpoint(on_request=AsyncMock()),
        "llm",
        RegistrationOptions(origin="local", id="id-1"),
    )

    api_model = ep._build_api_model("vlm", ep.models["vlm"])  # pyright: ignore[reportPrivateUsage]

    assert api_model.props.supports_vision is True
    assert api_model.props.supports_audio is False


def test_build_api_model_sums_usage_and_capacity_across_registrations():
    ep = make_endpoint()
    for reg_id, usage in (("id-1", 2), ("id-2", 3)):
        ep.add_model(
            "llm-a",
            make_props(),
            SimpleEndpoint(on_request=AsyncMock()),
            "llm",
            RegistrationOptions(origin="local", id=reg_id, usage=usage, capacity_state=4),
        )

    api_model = ep._build_api_model("llm-a", ep.models["llm-a"])  # pyright: ignore[reportPrivateUsage]

    assert api_model.usage == 5
    assert api_model.capacity == 8


def test_build_api_model_capacity_is_none_when_any_registration_is_not_finite():
    ep = make_endpoint()
    ep.add_model(
        "llm-b",
        make_props(),
        SimpleEndpoint(on_request=AsyncMock()),
        "llm",
        RegistrationOptions(origin="local", id="id-1", capacity_state=4),
    )
    ep.add_model(
        "llm-b",
        make_props(),
        SimpleEndpoint(on_request=AsyncMock()),
        "llm",
        RegistrationOptions(origin="local", id="id-2", capacity_state="unbounded"),
    )

    assert ep._build_api_model("llm-b", ep.models["llm-b"]).capacity is None  # pyright: ignore[reportPrivateUsage]


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


def test_get_models_sets_real_created_timestamp():
    ep = make_endpoint()
    before = int(time.time())
    ep.add_model("gpt-4", make_props(), SimpleEndpoint(on_request=AsyncMock()), "llm", RegistrationOptions(origin="local"))
    after = int(time.time())

    models = ep.get_models()

    assert before <= models[0].created <= after


def test_get_models_sets_owned_by_from_registration_options():
    ep = make_endpoint()
    ep.add_model(
        "gpt-4", make_props(), SimpleEndpoint(on_request=AsyncMock()), "llm", RegistrationOptions(origin="local", owned_by="ollama")
    )

    models = ep.get_models()

    assert models[0].owned_by == "ollama"


def test_get_models_owned_by_defaults_to_local():
    ep = make_endpoint()
    ep.add_model("gpt-4", make_props(), SimpleEndpoint(on_request=AsyncMock()), "llm", None)

    models = ep.get_models()

    assert models[0].owned_by == "local"


def test_list_models_returns_registered_model_entries():
    ep = make_endpoint()
    ep.add_model("gpt-4", make_props(), SimpleEndpoint(on_request=AsyncMock()), "llm", RegistrationOptions(origin="local", id="rid-1"))

    items = ep.list_models()

    assert len(items) == 1
    assert items[0].name == "gpt-4"


def test_list_models_includes_capacity():
    ep = make_endpoint()
    ep.add_model(
        "gpt-4",
        make_props(),
        SimpleEndpoint(on_request=AsyncMock()),
        "llm",
        RegistrationOptions(origin="local", id="rid-1", capacity_state=8),
    )

    items = ep.list_models()

    assert len(items) == 1
    assert items[0].capacity == 8


def test_list_models_includes_capacity_known_and_warm():
    ep = make_endpoint()
    ep.add_model(
        "gpt-4",
        make_props(),
        SimpleEndpoint(on_request=AsyncMock()),
        "llm",
        RegistrationOptions(origin="local", id="rid-1", capacity_state="unknown", warm=False),
    )

    items = ep.list_models()

    assert len(items) == 1
    assert items[0].capacity is None
    assert items[0].capacity_known is False
    assert items[0].warm is False


def test_list_models_excludes_mesh_ancestor_owned_models():
    ep = make_endpoint()
    ep.add_model(
        "local-model", make_props(), SimpleEndpoint(on_request=AsyncMock()), "llm", RegistrationOptions(origin="local", id="rid-local")
    )
    ep.add_model(
        "ancestor-model",
        make_props(),
        SimpleEndpoint(on_request=AsyncMock()),
        "llm",
        RegistrationOptions(origin="https://ancestor", id="rid-ancestor", owned_by="mesh-ancestor"),
    )

    items = ep.list_models()

    assert [item.name for item in items] == ["local-model"]


def test_list_models_custom_exclude_owned_by_also_excludes_mesh():
    ep = make_endpoint()
    ep.add_model(
        "local-model", make_props(), SimpleEndpoint(on_request=AsyncMock()), "llm", RegistrationOptions(origin="local", id="rid-local")
    )
    ep.add_model(
        "child-model",
        make_props(),
        SimpleEndpoint(on_request=AsyncMock()),
        "llm",
        RegistrationOptions(origin="https://child", id="rid-child", owned_by="mesh"),
    )

    items = ep.list_models(frozenset({"mesh", "mesh-ancestor"}))

    assert [item.name for item in items] == ["local-model"]


def test_list_models_strips_subtype_suffix_from_type():
    # Regression: WS clients crash when a model with a composite type like "llm-v2"
    # is sent over the mesh — the remote peer's ModelType literal rejects unknown suffixes.
    ep = make_endpoint()
    ep.add_model(
        "my-model", make_props(), SimpleEndpoint(on_request=AsyncMock()), "llm-v2", RegistrationOptions(origin="local", id="rid-sub")
    )

    items = ep.list_models()

    assert len(items) == 1
    assert items[0].type == "llm"


def test_list_models_returns_multiple_registrations_for_same_model():
    ep = make_endpoint()
    ep.add_model("gpt-4", make_props(), SimpleEndpoint(on_request=AsyncMock()), "llm", RegistrationOptions(origin="local", id="id-a"))
    ep.add_model("gpt-4", make_props(), SimpleEndpoint(on_request=AsyncMock()), "llm", RegistrationOptions(origin="local", id="id-b"))

    items = ep.list_models()

    assert len(items) == 2


@pytest.mark.asyncio
async def test_proxy_options_get_request_headers_no_request():
    opts = ProxyOptions(url="http://example.com", headers={"Authorization": "Bearer tok"})

    headers = await opts.get_request_headers(None)

    assert headers == {"Authorization": "Bearer tok"}


@pytest.mark.asyncio
async def test_proxy_options_get_request_headers_merges_allowed_request_headers():
    opts = ProxyOptions(
        url="http://example.com",
        headers={"Authorization": "Bearer tok"},
        allowed_request_headers=["x-custom"],
    )
    request = MagicMock()
    request.headers = {"x-custom": "val", "x-other": "ignored"}

    headers = await opts.get_request_headers(request)

    assert headers["x-custom"] == "val"
    assert headers["Authorization"] == "Bearer tok"
    assert "x-other" not in headers


@pytest.mark.asyncio
async def test_proxy_options_get_request_headers_no_allowed_headers_returns_static():
    opts = ProxyOptions(url="http://example.com", headers={"Authorization": "Bearer tok"})
    request = MagicMock()
    request.headers = {"x-forward": "val"}

    headers = await opts.get_request_headers(request)

    assert "x-forward" not in headers
    assert headers["Authorization"] == "Bearer tok"


@pytest.mark.asyncio
async def test_proxy_options_get_request_headers_dynamic_headers_override_static():
    opts = ProxyOptions(
        url="http://example.com",
        headers={"Authorization": "Bearer static"},
        dynamic_headers=AsyncMock(return_value={"Authorization": "Bearer live"}),
    )

    headers = await opts.get_request_headers(None)

    assert headers["Authorization"] == "Bearer live"


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


def test_registry_list_own_models_excludes_both_mesh_directions():
    reg = make_registry()
    props = make_props()
    reg.register_chat_completion("local-model", props, make_chat_endpoint(), RegistrationOptions(origin="local", id="rid-local"))
    child_model = Model(id="rid-child", name="child-model", type="llm", props=props, usage=0)
    ancestor_model = Model(id="rid-ancestor", name="ancestor-model", type="llm", props=props, usage=0)
    reg.update_models([], [child_model], "http://child.example.com/", "childkey")
    reg.update_models([], [ancestor_model], "http://ancestor.example.com/", "ancestorkey", owned_by="mesh-ancestor")

    names = [m.name for m in reg.list_own_models()]

    assert names == ["local-model"]


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
    reg.register_chat_completion("gpt-4", make_props(), ep, RegistrationOptions(origin="local", id="old-id", owned_by="mesh"))
    old_model = Model(id="old-id", name="gpt-4", type="llm", props=make_props(), usage=0)

    reg.update_models([old_model], [], "http://api.example.com/", "mykey")

    assert not reg.chat_completion_endpoints.has_model("gpt-4")


def test_update_models_registers_new_models_with_mesh_owned_by():
    reg = make_registry()
    new_model = Model(id="new-id", name="gpt-4", type="llm", props=make_props(), usage=0)

    reg.update_models([], [new_model], "http://api.example.com/", "mykey")

    model = reg.get_model("gpt-4")
    assert model.owned_by == "mesh"


def test_update_models_registers_new_models_with_explicit_owned_by():
    reg = make_registry()
    new_model = Model(id="new-id", name="gpt-4", type="llm", props=make_props(), usage=0)

    reg.update_models([], [new_model], "http://ancestor.example.com/", "mykey", owned_by="mesh-ancestor")

    model = reg.get_model("gpt-4")
    assert model.owned_by == "mesh-ancestor"


def test_update_models_ignores_colliding_id_from_different_owner():
    """An older/misbehaving mesh peer echoing back an id it doesn't own must not shadow the real owner."""
    reg = make_registry()
    ep = make_chat_endpoint()
    reg.register_chat_completion("gpt-4", make_props(), ep, RegistrationOptions(origin="local", id="shared-id", owned_by="local"))
    colliding_model = Model(id="shared-id", name="other-model", type="llm", props=make_props(), usage=0)

    reg.update_models([], [colliding_model], "http://ancestor.example.com/", "mykey", owned_by="mesh-ancestor")

    model = reg.get_model("gpt-4")
    assert model.owned_by == "local"
    assert not reg.chat_completion_endpoints.has_model("other-model")


def test_update_models_does_not_remove_colliding_id_owned_by_someone_else():
    """A stale ancestor entry must not delete a local model that happens to reuse its id."""
    reg = make_registry()
    ep = make_chat_endpoint()
    reg.register_chat_completion("gpt-4", make_props(), ep, RegistrationOptions(origin="local", id="shared-id", owned_by="local"))
    stale_ancestor_model = Model(id="shared-id", name="gpt-4", type="llm", props=make_props(), usage=0)

    reg.update_models([stale_ancestor_model], [], "http://ancestor.example.com/", "mykey", owned_by="mesh-ancestor")

    model = reg.get_model("gpt-4")
    assert model.owned_by == "local"


def test_model_deserializes_missing_capacity_field_as_none():
    """Old peer -> new node: a raw mesh payload without `capacity` degrades to None instead of failing."""
    raw = {"id": "id-1", "name": "gpt-4", "type": "llm", "props": make_props().model_dump(), "usage": 0}

    model = Model.model_validate(raw)

    assert model.capacity is None


def test_model_deserializes_present_capacity_field():
    """New peer -> old node: an explicit `capacity` field parses through untouched."""
    raw = {"id": "id-1", "name": "gpt-4", "type": "llm", "props": make_props().model_dump(), "usage": 0, "capacity": 8}

    model = Model.model_validate(raw)

    assert model.capacity == 8


def test_update_models_preserves_capacity_from_mesh_child():
    reg = make_registry()
    new_model = Model(id="new-id", name="gpt-4", type="llm", props=make_props(), usage=0, capacity=8)

    reg.update_models([], [new_model], "http://api.example.com/", "mykey")

    registered = reg.chat_completion_endpoints.models["gpt-4"]["new-id"]
    assert registered.capacity_state == 8


def test_update_models_defaults_capacity_to_none_when_missing():
    reg = make_registry()
    new_model = Model(id="new-id", name="gpt-4", type="llm", props=make_props(), usage=0)

    reg.update_models([], [new_model], "http://api.example.com/", "mykey")

    registered = reg.chat_completion_endpoints.models["gpt-4"]["new-id"]
    assert registered.capacity_state == "unknown"


def test_model_deserializes_missing_capacity_known_and_warm_as_false():
    """Old peer -> new node: a raw mesh payload missing these fields defaults capacity to unknown and treats the model as cold, not warm."""
    raw = {"id": "id-1", "name": "gpt-4", "type": "llm", "props": make_props().model_dump(), "usage": 0}

    model = Model.model_validate(raw)

    assert model.capacity_known is False
    assert model.warm is False


def test_update_models_preserves_capacity_known_and_warm_from_mesh_child():
    reg = make_registry()
    new_model = Model(id="new-id", name="gpt-4", type="llm", props=make_props(), usage=0, capacity=None, capacity_known=False, warm=False)

    reg.update_models([], [new_model], "http://api.example.com/", "mykey")

    registered = reg.chat_completion_endpoints.models["gpt-4"]["new-id"]
    assert registered.capacity_state == "unknown"
    assert registered.warm is False


def test_update_warm_updates_model_warm_state():
    reg = make_registry()
    ep = make_chat_endpoint()
    rid = reg.register_chat_completion("gpt-4", make_props(), ep, RegistrationOptions(origin="local", warm=True))

    reg.update_warm(rid, False)

    assert reg.registry[rid].registered_model.warm is False


def test_update_warm_no_op_for_unknown_id():
    reg = make_registry()
    reg.update_warm("nonexistent", False)  # must not raise


def test_update_warm_propagates_to_parent_infra_when_changed():
    reg = make_registry()
    ep = make_chat_endpoint()
    rid = reg.register_chat_completion("gpt-4", make_props(), ep, RegistrationOptions(origin="local", warm=True))
    reg.parent_infra.send_warm.reset_mock()  # pyright: ignore[reportAttributeAccessIssue]

    reg.update_warm(rid, False)

    assert reg.parent_infra.send_warm.call_count == 1  # pyright: ignore[reportAttributeAccessIssue]


def test_update_warm_does_not_propagate_when_unchanged():
    reg = make_registry()
    ep = make_chat_endpoint()
    rid = reg.register_chat_completion("gpt-4", make_props(), ep, RegistrationOptions(origin="local", warm=True))
    reg.parent_infra.send_warm.reset_mock()  # pyright: ignore[reportAttributeAccessIssue]

    reg.update_warm(rid, True)

    assert reg.parent_infra.send_warm.call_count == 0  # pyright: ignore[reportAttributeAccessIssue]


def test_update_modality_support_updates_props_in_place():
    reg = make_registry()
    ep = make_chat_endpoint()
    rid = reg.register_chat_completion(
        "gpt-4", make_props(supports_vision=None, supports_audio=None), ep, RegistrationOptions(origin="local")
    )

    reg.update_modality_support(rid, supports_vision=True, supports_audio=False)

    props = reg.registry[rid].registered_model.props
    assert props.supports_vision is True
    assert props.supports_audio is False


def test_update_modality_support_no_op_for_unknown_id():
    reg = make_registry()
    reg.update_modality_support("nonexistent", supports_vision=True, supports_audio=True)  # must not raise


def test_update_modality_support_is_no_op_when_unchanged():
    reg = make_registry()
    ep = make_chat_endpoint()
    rid = reg.register_chat_completion(
        "gpt-4", make_props(supports_vision=True, supports_audio=None), ep, RegistrationOptions(origin="local")
    )
    original_props = reg.registry[rid].registered_model.props

    reg.update_modality_support(rid, supports_vision=True, supports_audio=None)

    assert reg.registry[rid].registered_model.props is original_props


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


def test_update_models_refreshes_proxy_when_api_key_rotates():
    """A rotated peer API key must be re-applied to already-registered proxies, not just to new ones."""
    reg = make_registry()
    existing = Model(id="exist-id", name="gpt-4", type="llm", props=make_props(), usage=0)
    reg.update_models([], [existing], "http://api.example.com/", "oldkey")

    with patch.object(reg, "_register_proxy") as mock_register:
        reg.update_models([existing], [existing], "http://api.example.com/", "newkey")

    assert mock_register.call_count == 1
    assert mock_register.call_args.kwargs["api_key"] == "newkey"


def test_update_models_no_refresh_when_api_key_unchanged():
    reg = make_registry()
    existing = Model(id="exist-id", name="gpt-4", type="llm", props=make_props(), usage=0)
    reg.update_models([], [existing], "http://api.example.com/", "samekey")

    with patch.object(reg, "_register_proxy") as mock_register:
        reg.update_models([existing], [existing], "http://api.example.com/", "samekey")

    assert mock_register.call_count == 0


def test_update_models_key_rotation_does_not_touch_colliding_id_owned_by_someone_else():
    """A rotated ancestor key must not re-register an id that collided with a different owner."""
    reg = make_registry()
    ep = make_chat_endpoint()
    reg.register_chat_completion("gpt-4", make_props(), ep, RegistrationOptions(origin="local", id="shared-id", owned_by="local"))
    colliding_model = Model(id="shared-id", name="other-model", type="llm", props=make_props(), usage=0)
    reg.update_models([], [colliding_model], "http://ancestor.example.com/", "oldkey", owned_by="mesh-ancestor")

    with patch.object(reg, "_register_proxy") as mock_register:
        reg.update_models([colliding_model], [colliding_model], "http://ancestor.example.com/", "newkey", owned_by="mesh-ancestor")

    assert mock_register.call_count == 0
    model = reg.get_model("gpt-4")
    assert model.owned_by == "local"


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


def test_register_proxy_custom_uses_configured_standard_proxy_timeout():
    reg = make_registry()
    reg.config.standard_proxy_timeout_seconds = 1800

    with patch.object(reg, "register_custom_endpoint_as_proxy") as mock_register:
        reg._register_proxy(  # pyright: ignore[reportPrivateUsage]
            "custom/url", "custom", make_props(), "http://example.com/", "key", RegistrationOptions(origin="http://example.com/")
        )

    registered_options = mock_register.call_args.kwargs["options"]
    assert registered_options.read_timeout_seconds == 1800


def test_register_proxy_custom_forwards_full_path():
    reg = make_registry()

    with patch.object(reg, "register_custom_endpoint_as_proxy") as mock_register:
        reg._register_proxy(  # pyright: ignore[reportPrivateUsage]
            "custom/url", "custom", make_props(), "http://example.com/", "key", RegistrationOptions(origin="http://example.com/")
        )

    registered_options = mock_register.call_args.kwargs["options"]
    assert registered_options.forward_full_path is True


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

    with caplog.at_level(logging.WARNING, logger="uvicorn"):
        reg._register_proxy(  # pyright: ignore[reportPrivateUsage]
            "x", "unknown-type", make_props(), "http://example.com/", "key", RegistrationOptions(origin="http://example.com/")
        )

    assert "Cannot register proxy" in caplog.text


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
async def test_execute_messages_raises_404_when_model_missing():
    reg = make_registry()
    body = MagicMock(spec=MessagesRequest)
    body.model = "unknown-model"
    body.model_dump_json.return_value = "{}"

    with pytest.raises(HTTPException) as exc_info:
        await reg.execute_messages(body)

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Model not found"


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
    body.input = None
    body.model_dump_json.return_value = "{}"

    with patch.object(reg, "with_usage", new_callable=AsyncMock, return_value=response):
        result = await reg.execute_responses(body)

    assert result is response


@pytest.mark.asyncio
async def test_execute_responses_raises_404_when_not_registered():
    reg = make_registry()
    body = MagicMock(spec=ResponsesRequest)
    body.model = "unknown"
    body.input = None
    body.model_dump_json.return_value = "{}"

    with pytest.raises(HTTPException) as exc_info:
        await reg.execute_responses(body)

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Model not found"


@pytest.mark.asyncio
async def test_execute_responses_raises_400_when_model_has_no_responses_support():
    reg = make_registry()
    ep = make_chat_endpoint(on_chat=True, on_responses=False)
    reg.register_chat_completion("gpt-4", make_props(), ep, RegistrationOptions(origin="local"))

    body = MagicMock(spec=ResponsesRequest)
    body.model = "gpt-4"
    body.input = None
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
async def test_execute_chat_completion_raises_404_when_not_registered():
    reg = make_registry()
    body = MagicMock(spec=ChatCompletionRequest)
    body.model = "unknown"
    body.model_dump_json.return_value = "{}"

    with pytest.raises(HTTPException) as exc_info:
        await reg.execute_chat_completion(body)

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Model not found"


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
async def test_execute_completion_raises_404_when_not_registered():
    reg = make_registry()
    body = MagicMock(spec=CompletionLegacyRequest)
    body.model = "unknown"

    with pytest.raises(HTTPException) as exc_info:
        await reg.execute_completion(body)

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Model not found"


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
async def test_execute_ollama_chat_raises_404_when_not_registered():
    reg = make_registry()
    body = MagicMock(spec=OllamaChatRequest)
    body.model = "unknown"

    with pytest.raises(HTTPException) as exc_info:
        await reg.execute_ollama_chat(body)

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Model not found"


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
async def test_execute_embeddings_raises_404_when_not_registered():
    reg = make_registry()
    body = MagicMock(spec=EmbeddingRequest)
    body.model = "unknown"

    with pytest.raises(HTTPException) as exc_info:
        await reg.execute_embeddings(body)

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Model not found"


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
async def test_execute_images_generations_raises_404_when_not_registered():
    reg = make_registry()
    body = MagicMock(spec=ImagesRequest)
    body.model = "unknown"
    body.model_dump_json.return_value = "{}"

    with pytest.raises(HTTPException) as exc_info:
        await reg.execute_images_generations(body)

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Model not found"


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
async def test_execute_audio_speech_raises_404_when_not_registered():
    reg = make_registry()
    body = MagicMock(spec=CreateSpeechRequest)
    body.model = "unknown"
    body.model_dump_json.return_value = "{}"

    with pytest.raises(HTTPException) as exc_info:
        await reg.execute_audio_speech(body)

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Model not found"


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
    body.response_format = "json"

    with patch.object(reg, "with_usage", new_callable=AsyncMock, return_value=response):
        result = await reg.execute_audio_transcriptions(body)

    assert result is response


@pytest.mark.asyncio
async def test_execute_audio_transcriptions_raises_404_when_not_registered():
    reg = make_registry()
    body = MagicMock(spec=CreateTranscriptionRequest)
    body.model = "unknown"

    with pytest.raises(HTTPException) as exc_info:
        await reg.execute_audio_transcriptions(body)

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Model not found"


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
async def test_execute_rerank_raises_404_when_not_registered():
    reg = make_registry()
    body = MagicMock(spec=RerankRequest)
    body.model = "unknown"
    body.model_dump_json.return_value = "{}"

    with pytest.raises(HTTPException) as exc_info:
        await reg.execute_rerank(body)

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Model not found"


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


@pytest.fixture
def metrics_registry() -> MetricsRegistry:
    return MetricsRegistry(registry=CollectorRegistry(auto_describe=True))


@pytest.fixture
def metered_registry(metrics_registry: MetricsRegistry) -> EndpointRegistry:
    return make_registry(metrics_registry)


@pytest.fixture
def local_model(metered_registry: EndpointRegistry) -> RegisteredModel[Any]:
    rid = metered_registry.register_chat_completion("gpt-4", make_props(), make_chat_endpoint(), RegistrationOptions(origin="local"))
    return metered_registry.registry[rid].registered_model


def request_metrics(metrics_registry: MetricsRegistry, model: RegisteredModel[Any]) -> dict[str, float]:
    def value(name: str, labels: dict[str, str]) -> float:
        return metrics_registry.registry.get_sample_value(name, labels) or 0.0

    model_labels = {"model_name": model.name, "model_type": model.type}
    return {
        "success": value("deepfellow_requests_total", {**model_labels, "status": "success"}),
        "error": value("deepfellow_requests_total", {**model_labels, "status": "error"}),
        "duration_count": value("deepfellow_request_duration_seconds_count", model_labels),
        "in_flight": value("deepfellow_requests_in_flight", {"model_type": model.type}),
    }


def request_errors(metrics_registry: MetricsRegistry, model: RegisteredModel[Any]) -> dict[str, float]:
    return {
        sample.labels["error_type"]: sample.value
        for metric in metrics_registry.request_errors.collect()
        for sample in metric.samples
        if sample.name == "deepfellow_request_errors_total" and sample.labels["model_name"] == model.name
    }


async def stream_chunks(*chunks: bytes, error: BaseException | None = None) -> AsyncGenerator[bytes]:
    for chunk in chunks:
        yield chunk
    if error is not None:
        raise error


async def read_stream(response: StreamingResponse) -> list[bytes]:
    return [cast("bytes", chunk) async for chunk in response.body_iterator]


SUCCESS = {"success": 1.0, "error": 0.0, "duration_count": 1.0, "in_flight": 0.0}
ERROR = {"success": 0.0, "error": 1.0, "duration_count": 1.0, "in_flight": 0.0}


@pytest.mark.asyncio
async def test_with_usage_counts_non_streamed_5xx_as_http_5xx_error(
    metered_registry: EndpointRegistry, metrics_registry: MetricsRegistry, local_model: RegisteredModel[Any]
):
    response = Response(content=b"model failed", status_code=500)

    result = await metered_registry.with_usage(local_model, AsyncMock(return_value=response))

    assert result is response
    assert request_metrics(metrics_registry, local_model) == ERROR
    assert request_errors(metrics_registry, local_model) == {"http_5xx": 1.0}
    assert local_model.usage == 0


@pytest.mark.asyncio
async def test_with_usage_counts_non_streamed_4xx_as_success(
    metered_registry: EndpointRegistry, metrics_registry: MetricsRegistry, local_model: RegisteredModel[Any]
):
    response = Response(content=b"bad request", status_code=400)

    result = await metered_registry.with_usage(local_model, AsyncMock(return_value=response))

    assert result is response
    assert request_metrics(metrics_registry, local_model) == SUCCESS
    assert request_errors(metrics_registry, local_model) == {}
    assert local_model.usage == 0


@pytest.mark.asyncio
async def test_with_usage_counts_streamed_5xx_as_http_5xx_error(
    metered_registry: EndpointRegistry, metrics_registry: MetricsRegistry, local_model: RegisteredModel[Any]
):
    response = StreamingResponse(stream_chunks(b"model ", b"overloaded"), status_code=503)

    result = await metered_registry.with_usage(local_model, AsyncMock(return_value=response))
    chunks = await read_stream(result)

    assert result.status_code == 503
    assert chunks == [b"model ", b"overloaded"]
    assert request_metrics(metrics_registry, local_model) == ERROR
    assert request_errors(metrics_registry, local_model) == {"http_5xx": 1.0}
    assert local_model.usage == 0


@pytest.mark.parametrize(
    ("error", "expected_error_type"),
    [
        (aiohttp.ClientPayloadError("broken"), "connection_error"),
        (TimeoutError(), "timeout"),
        (RuntimeError("boom"), "model_error"),
    ],
)
@pytest.mark.asyncio
async def test_with_usage_counts_stream_broken_by_model_as_error(
    metered_registry: EndpointRegistry,
    metrics_registry: MetricsRegistry,
    local_model: RegisteredModel[Any],
    error: Exception,
    expected_error_type: str,
):
    response = StreamingResponse(stream_chunks(b"partial", error=error), status_code=200)
    result = await metered_registry.with_usage(local_model, AsyncMock(return_value=response))

    with pytest.raises(type(error)):
        await read_stream(result)

    assert request_metrics(metrics_registry, local_model) == ERROR
    assert request_errors(metrics_registry, local_model) == {expected_error_type: 1.0}
    assert local_model.usage == 0


@pytest.mark.asyncio
async def test_with_usage_counts_broken_5xx_stream_once_as_http_5xx(
    metered_registry: EndpointRegistry, metrics_registry: MetricsRegistry, local_model: RegisteredModel[Any]
):
    response = StreamingResponse(stream_chunks(b"partial", error=aiohttp.ClientPayloadError("broken")), status_code=500)
    result = await metered_registry.with_usage(local_model, AsyncMock(return_value=response))

    with pytest.raises(aiohttp.ClientPayloadError):
        await read_stream(result)

    assert request_metrics(metrics_registry, local_model) == ERROR
    assert request_errors(metrics_registry, local_model) == {"http_5xx": 1.0}
    assert local_model.usage == 0


@pytest.mark.asyncio
async def test_with_usage_counts_successful_stream_as_success(
    metered_registry: EndpointRegistry, metrics_registry: MetricsRegistry, local_model: RegisteredModel[Any]
):
    response = StreamingResponse(stream_chunks(b"hello ", b"world"), status_code=200)

    result = await metered_registry.with_usage(local_model, AsyncMock(return_value=response))
    chunks = await read_stream(result)

    assert chunks == [b"hello ", b"world"]
    assert request_metrics(metrics_registry, local_model) == SUCCESS
    assert request_errors(metrics_registry, local_model) == {}
    assert local_model.usage == 0


@pytest.mark.asyncio
async def test_with_usage_counts_client_closing_stream_as_success(
    metered_registry: EndpointRegistry, metrics_registry: MetricsRegistry, local_model: RegisteredModel[Any]
):
    response = StreamingResponse(stream_chunks(b"first", b"second"), status_code=200)
    result = await metered_registry.with_usage(local_model, AsyncMock(return_value=response))
    body_iterator = cast("AsyncGenerator[bytes]", result.body_iterator)

    first = await anext(body_iterator)
    await body_iterator.aclose()

    assert first == b"first"
    assert request_metrics(metrics_registry, local_model) == SUCCESS
    assert request_errors(metrics_registry, local_model) == {}
    assert local_model.usage == 0


@pytest.mark.asyncio
async def test_with_usage_counts_cancelled_stream_as_success(
    metered_registry: EndpointRegistry, metrics_registry: MetricsRegistry, local_model: RegisteredModel[Any]
):
    response = StreamingResponse(stream_chunks(b"partial", error=asyncio.CancelledError()), status_code=200)
    result = await metered_registry.with_usage(local_model, AsyncMock(return_value=response))

    with pytest.raises(asyncio.CancelledError):
        await read_stream(result)

    assert request_metrics(metrics_registry, local_model) == SUCCESS
    assert request_errors(metrics_registry, local_model) == {}
    assert local_model.usage == 0


@pytest.mark.asyncio
async def test_with_usage_counts_exception_before_response_as_error(
    metered_registry: EndpointRegistry, metrics_registry: MetricsRegistry, local_model: RegisteredModel[Any]
):
    func = AsyncMock(side_effect=aiohttp.ClientConnectionError("refused"))

    with pytest.raises(aiohttp.ClientConnectionError):
        await metered_registry.with_usage(local_model, func)

    assert request_metrics(metrics_registry, local_model) == ERROR
    assert request_errors(metrics_registry, local_model) == {"connection_error": 1.0}
    assert local_model.usage == 0


@pytest.fixture
def vllm_model(metered_registry: EndpointRegistry) -> RegisteredModel[Any]:
    rid = metered_registry.register_chat_completion(
        "qwen3", make_props(), make_chat_endpoint(), RegistrationOptions(origin="local", owned_by="vllm")
    )
    return metered_registry.registry[rid].registered_model


VLLM_TAGS = {"service": "vllm", "model": "qwen3", "model_type": "llm"}


@pytest.mark.parametrize("status_code", [500, 502, 503, 504])
@pytest.mark.asyncio
@patch("server.endpointregistry.logger")
@patch("server.endpointregistry.capture_exception")
async def test_with_usage_reports_non_streamed_5xx_once(
    mock_capture_exception: MagicMock,
    mock_logger: MagicMock,
    metered_registry: EndpointRegistry,
    vllm_model: RegisteredModel[Any],
    status_code: int,
):
    response = Response(content=b"prompt: tell me a secret", status_code=status_code, headers={"x-upstream": "1"})

    result = await metered_registry.with_usage(vllm_model, AsyncMock(return_value=response))

    assert result is response
    assert result.body == b"prompt: tell me a secret"
    assert result.headers["x-upstream"] == "1"
    assert mock_capture_exception.call_count == 1
    exc = mock_capture_exception.call_args.args[0]
    assert isinstance(exc, UpstreamError)
    assert str(exc) == f"vllm upstream returned HTTP {status_code}"
    error_id = mock_capture_exception.call_args.kwargs["error_id"]
    assert mock_capture_exception.call_args == call(
        exc,
        error_id=error_id,
        tags={**VLLM_TAGS, "error_type": "http_5xx", "upstream_status": str(status_code)},
        fingerprint=["upstream-http", "vllm", str(status_code)],
    )
    assert mock_logger.warning.call_args == call(
        "Request to model %r of service %s failed: HTTP %s (error_id=%s)", "qwen3", "vllm", str(status_code), error_id
    )


@pytest.mark.asyncio
@patch("server.endpointregistry.capture_exception")
async def test_with_usage_reports_streamed_5xx_once_and_keeps_stream(
    mock_capture_exception: MagicMock, metered_registry: EndpointRegistry, vllm_model: RegisteredModel[Any]
):
    response = StreamingResponse(stream_chunks(b"model ", b"overloaded"), status_code=503, headers={"x-upstream": "1"})

    result = await metered_registry.with_usage(vllm_model, AsyncMock(return_value=response))
    reported_before_body = mock_capture_exception.call_count
    chunks = await read_stream(result)

    assert reported_before_body == 1
    assert mock_capture_exception.call_count == 1
    assert result.status_code == 503
    assert result.headers["x-upstream"] == "1"
    assert chunks == [b"model ", b"overloaded"]
    assert mock_capture_exception.call_args.kwargs["tags"] == {**VLLM_TAGS, "error_type": "http_5xx", "upstream_status": "503"}


@pytest.mark.asyncio
@patch("server.endpointregistry.capture_exception")
async def test_with_usage_reports_broken_5xx_stream_once_and_marks_exception_reported(
    mock_capture_exception: MagicMock, metered_registry: EndpointRegistry, vllm_model: RegisteredModel[Any]
):
    error = aiohttp.ClientPayloadError("broken")
    response = StreamingResponse(stream_chunks(b"partial", error=error), status_code=502)
    result = await metered_registry.with_usage(vllm_model, AsyncMock(return_value=response))

    with pytest.raises(aiohttp.ClientPayloadError):
        await read_stream(result)

    assert mock_capture_exception.call_count == 1
    assert is_reported(error) is True
    assert attached_tags(error) == {**VLLM_TAGS, "error_type": "connection_error"}


@pytest.mark.parametrize("status_code", [200, 201, 400, 401, 404, 422, 429, 499])
@pytest.mark.asyncio
@patch("server.endpointregistry.capture_exception")
async def test_with_usage_does_not_report_non_5xx_response(
    mock_capture_exception: MagicMock, metered_registry: EndpointRegistry, vllm_model: RegisteredModel[Any], status_code: int
):
    response = Response(content=b"body", status_code=status_code)

    result = await metered_registry.with_usage(vllm_model, AsyncMock(return_value=response))

    assert result is response
    assert mock_capture_exception.call_count == 0


@pytest.mark.parametrize("status_code", [200, 429])
@pytest.mark.asyncio
@patch("server.endpointregistry.capture_exception")
async def test_with_usage_does_not_report_non_5xx_stream(
    mock_capture_exception: MagicMock, metered_registry: EndpointRegistry, vllm_model: RegisteredModel[Any], status_code: int
):
    response = StreamingResponse(stream_chunks(b"hello ", b"world"), status_code=status_code)

    result = await metered_registry.with_usage(vllm_model, AsyncMock(return_value=response))
    chunks = await read_stream(result)

    assert chunks == [b"hello ", b"world"]
    assert mock_capture_exception.call_count == 0


@pytest.mark.asyncio
@patch("server.endpointregistry.capture_exception")
async def test_with_usage_does_not_report_non_response_result(
    mock_capture_exception: MagicMock, metered_registry: EndpointRegistry, vllm_model: RegisteredModel[Any]
):
    result = await metered_registry.with_usage(vllm_model, AsyncMock(return_value={"status_code": 500}))

    assert result == {"status_code": 500}
    assert mock_capture_exception.call_count == 0


@pytest.mark.parametrize("status_code", [500, 503])
@pytest.mark.asyncio
@patch("server.endpointregistry.capture_exception")
async def test_with_usage_reports_5xx_http_exception_from_service(
    mock_capture_exception: MagicMock, metered_registry: EndpointRegistry, vllm_model: RegisteredModel[Any], status_code: int
):
    error = HTTPException(status_code, "Something went wrong")

    with pytest.raises(HTTPException) as exc_info:
        await metered_registry.with_usage(vllm_model, AsyncMock(side_effect=error))

    assert exc_info.value is error
    assert mock_capture_exception.call_count == 1
    assert mock_capture_exception.call_args == call(
        error,
        error_id=mock_capture_exception.call_args.kwargs["error_id"],
        tags={**VLLM_TAGS, "error_type": "model_error", "upstream_status": str(status_code)},
        fingerprint=None,
    )
    assert is_reported(error) is True


@pytest.mark.parametrize("status_code", [400, 404, 429])
@pytest.mark.asyncio
@patch("server.endpointregistry.capture_exception")
async def test_with_usage_does_not_report_4xx_http_exception_from_service(
    mock_capture_exception: MagicMock, metered_registry: EndpointRegistry, vllm_model: RegisteredModel[Any], status_code: int
):
    error = HTTPException(status_code, "Bad request")

    with pytest.raises(HTTPException):
        await metered_registry.with_usage(vllm_model, AsyncMock(side_effect=error))

    assert mock_capture_exception.call_count == 0
    assert is_reported(error) is False
    assert attached_tags(error) == {**VLLM_TAGS, "error_type": "model_error"}


@pytest.mark.parametrize(
    ("error", "expected_error_type"),
    [
        (aiohttp.ClientConnectionError("refused"), "connection_error"),
        (TimeoutError(), "timeout"),
        (RuntimeError("boom"), "model_error"),
    ],
)
@pytest.mark.asyncio
@patch("server.endpointregistry.capture_exception")
async def test_with_usage_tags_exception_before_response_for_global_handler(
    mock_capture_exception: MagicMock,
    metered_registry: EndpointRegistry,
    vllm_model: RegisteredModel[Any],
    error: Exception,
    expected_error_type: str,
):
    with pytest.raises(type(error)):
        await metered_registry.with_usage(vllm_model, AsyncMock(side_effect=error))

    assert mock_capture_exception.call_count == 0
    assert attached_tags(error) == {**VLLM_TAGS, "error_type": expected_error_type}
    assert is_reported(error) is False


@pytest.mark.asyncio
@patch("server.endpointregistry.capture_exception")
async def test_with_usage_tags_exception_breaking_successful_stream(
    mock_capture_exception: MagicMock, metered_registry: EndpointRegistry, vllm_model: RegisteredModel[Any]
):
    error = aiohttp.ClientPayloadError("broken")
    response = StreamingResponse(stream_chunks(b"partial", error=error), status_code=200)
    result = await metered_registry.with_usage(vllm_model, AsyncMock(return_value=response))

    with pytest.raises(aiohttp.ClientPayloadError):
        await read_stream(result)

    assert mock_capture_exception.call_count == 0
    assert attached_tags(error) == {**VLLM_TAGS, "error_type": "connection_error"}
    assert is_reported(error) is False


@pytest.mark.asyncio
@patch("server.endpointregistry.capture_exception")
async def test_with_usage_does_not_report_or_tag_cancelled_stream(
    mock_capture_exception: MagicMock, metered_registry: EndpointRegistry, vllm_model: RegisteredModel[Any]
):
    error = asyncio.CancelledError()
    response = StreamingResponse(stream_chunks(b"partial", error=error), status_code=200)
    result = await metered_registry.with_usage(vllm_model, AsyncMock(return_value=response))

    with pytest.raises(asyncio.CancelledError):
        await read_stream(result)

    assert mock_capture_exception.call_count == 0
    assert attached_tags(error) == {}


@pytest.mark.asyncio
@patch("server.endpointregistry.capture_exception")
async def test_with_usage_reports_5xx_stream_closed_by_client_once(
    mock_capture_exception: MagicMock, metered_registry: EndpointRegistry, vllm_model: RegisteredModel[Any]
):
    response = StreamingResponse(stream_chunks(b"first", b"second"), status_code=500)
    result = await metered_registry.with_usage(vllm_model, AsyncMock(return_value=response))
    body_iterator = cast("AsyncGenerator[bytes]", result.body_iterator)

    await anext(body_iterator)
    await body_iterator.aclose()

    assert mock_capture_exception.call_count == 1


@pytest.mark.asyncio
@patch("server.endpointregistry.capture_exception")
async def test_with_usage_does_not_report_or_tag_mesh_model(mock_capture_exception: MagicMock, metered_registry: EndpointRegistry):
    rid = metered_registry.register_chat_completion("gpt-4", make_props(), make_chat_endpoint(), RegistrationOptions(origin="http://peer"))
    model = metered_registry.registry[rid].registered_model
    error = aiohttp.ClientConnectionError("refused")

    result = await metered_registry.with_usage(model, AsyncMock(return_value=Response(status_code=500)))
    with pytest.raises(aiohttp.ClientConnectionError):
        await metered_registry.with_usage(model, AsyncMock(side_effect=error))

    assert result.status_code == 500
    assert mock_capture_exception.call_count == 0
    assert attached_tags(error) == {}


def _app_proxying(registry: EndpointRegistry, model: RegisteredModel[Any], func: AsyncMock) -> TestClient:
    app = FastAPI()
    register_exception_handlers(app)

    async def proxied() -> Any:
        return await registry.with_usage(model, func)

    app.add_api_route("/proxied", proxied, methods=["GET"])
    return TestClient(app, raise_server_exceptions=False)


@patch("server.error_handlers.capture_exception")
@patch("server.endpointregistry.capture_exception")
def test_with_usage_connection_error_is_reported_once_by_global_handler_with_tags(
    mock_registry_capture: MagicMock, mock_handler_capture: MagicMock, metered_registry: EndpointRegistry, vllm_model: RegisteredModel[Any]
):
    client = _app_proxying(metered_registry, vllm_model, AsyncMock(side_effect=aiohttp.ClientConnectionError("refused")))

    resp = client.get("/proxied")

    assert resp.status_code == 500
    assert mock_registry_capture.call_count == 0
    assert mock_handler_capture.call_count == 1
    assert mock_handler_capture.call_args.kwargs["error_id"] == resp.headers["X-Error-Id"]
    assert mock_handler_capture.call_args.kwargs["handled"] is False
    assert mock_handler_capture.call_args.kwargs["tags"] == {**VLLM_TAGS, "error_type": "connection_error"}


@patch("server.error_handlers.capture_exception")
@patch("server.endpointregistry.capture_exception")
def test_with_usage_broken_2xx_stream_is_reported_once_by_global_handler_with_tags(
    mock_registry_capture: MagicMock, mock_handler_capture: MagicMock, metered_registry: EndpointRegistry, vllm_model: RegisteredModel[Any]
):
    response = StreamingResponse(stream_chunks(b"partial", error=aiohttp.ClientPayloadError("broken")), status_code=200)
    client = _app_proxying(metered_registry, vllm_model, AsyncMock(return_value=response))

    resp = client.get("/proxied")

    assert resp.status_code == 200
    assert mock_registry_capture.call_count == 0
    assert mock_handler_capture.call_count == 1
    assert mock_handler_capture.call_args.kwargs["handled"] is False
    assert mock_handler_capture.call_args.kwargs["tags"] == {**VLLM_TAGS, "error_type": "connection_error"}


@pytest.mark.parametrize("status_code", [200, 503])
@patch("server.error_handlers.capture_exception")
@patch("server.endpointregistry.capture_exception")
def test_with_usage_5xx_http_exception_mid_stream_is_reported_once_end_to_end(
    mock_registry_capture: MagicMock,
    mock_handler_capture: MagicMock,
    metered_registry: EndpointRegistry,
    vllm_model: RegisteredModel[Any],
    status_code: int,
):
    response = StreamingResponse(stream_chunks(b"partial", error=HTTPException(502, "Upstream gone")), status_code=status_code)
    client = _app_proxying(metered_registry, vllm_model, AsyncMock(return_value=response))

    resp = client.get("/proxied")

    assert resp.status_code == status_code
    assert mock_registry_capture.call_count == 1
    assert mock_handler_capture.call_count == 0


@patch("server.error_handlers.capture_exception")
@patch("server.endpointregistry.capture_exception")
def test_with_usage_4xx_http_exception_mid_stream_is_reported_by_global_handler_with_tags(
    mock_registry_capture: MagicMock, mock_handler_capture: MagicMock, metered_registry: EndpointRegistry, vllm_model: RegisteredModel[Any]
):
    response = StreamingResponse(stream_chunks(b"partial", error=HTTPException(400, "Bad chunk")), status_code=200)
    client = _app_proxying(metered_registry, vllm_model, AsyncMock(return_value=response))

    client.get("/proxied")

    assert mock_registry_capture.call_count == 0
    assert mock_handler_capture.call_count == 1
    assert mock_handler_capture.call_args.kwargs["tags"] == {**VLLM_TAGS, "error_type": "model_error"}


@patch("server.error_handlers.capture_exception")
@patch("server.endpointregistry.capture_exception")
def test_with_usage_broken_5xx_stream_is_reported_once_end_to_end(
    mock_registry_capture: MagicMock, mock_handler_capture: MagicMock, metered_registry: EndpointRegistry, vllm_model: RegisteredModel[Any]
):
    response = StreamingResponse(stream_chunks(b"partial", error=aiohttp.ClientPayloadError("broken")), status_code=500)
    client = _app_proxying(metered_registry, vllm_model, AsyncMock(return_value=response))

    resp = client.get("/proxied")

    assert resp.status_code == 500
    assert mock_registry_capture.call_count == 1
    assert mock_handler_capture.call_count == 0


@patch("server.error_handlers.capture_exception")
@patch("server.endpointregistry.capture_exception")
def test_with_usage_5xx_http_exception_is_reported_once_end_to_end(
    mock_registry_capture: MagicMock, mock_handler_capture: MagicMock, metered_registry: EndpointRegistry, vllm_model: RegisteredModel[Any]
):
    client = _app_proxying(metered_registry, vllm_model, AsyncMock(side_effect=HTTPException(503, "Model unavailable")))

    resp = client.get("/proxied")

    assert resp.status_code == 503
    assert resp.json()["error"]["message"] == "Model unavailable"
    assert "X-Error-Id" not in resp.headers
    assert mock_registry_capture.call_count == 1
    assert mock_handler_capture.call_count == 0


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


# Hard-coded on purpose: iterating over the production constants would not catch a typo in them.
_CHAT_EXTRA_FIELDS_PAYLOAD: dict[str, object] = {
    "thinking_token_budget": 300,
    "chat_template_kwargs": {"enable_thinking": True},
    "repetition_detection": {"max_pattern_size": 20},
    "top_k": 40,
    "min_p": 0.05,
    "repetition_penalty": 1.1,
    "presence_penalty": 0.5,
    "frequency_penalty": -0.5,
    "logprobs": True,
    "parallel_tool_calls": False,
}
_LEGACY_EXTRA_FIELDS_PAYLOAD: dict[str, object] = {
    "thinking_token_budget": 300,
    "repetition_detection": {"max_pattern_size": 20},
    "top_k": 40,
    "min_p": 0.05,
    "repetition_penalty": 1.1,
}


async def _post_chat_extra_fields(options: ProxyOptions) -> dict[str, object]:
    body = ChatCompletionRequest(  # pyright: ignore[reportCallIssue]
        model="m",
        messages=[{"role": "user", "content": "hi"}],  # pyright: ignore[reportArgumentType]
        reasoning_effort="low",
        **_CHAT_EXTRA_FIELDS_PAYLOAD,  # pyright: ignore[reportArgumentType]
    )
    mock_http_response = MagicMock()
    mock_http_response.as_streaming_response.return_value = MagicMock()
    with patch("server.endpointregistry.make_http_request", new_callable=AsyncMock, return_value=mock_http_response) as mock_req:
        await post_json(body, options)
    return json.loads(mock_req.call_args.kwargs["data"]._value)


@pytest.mark.asyncio
async def test_post_json_forwards_extra_chat_fields_when_enabled():
    sent = await _post_chat_extra_fields(ProxyOptions(url="http://vllm/", forward_extra_fields=True))

    assert {field: sent[field] for field in _CHAT_EXTRA_FIELDS_PAYLOAD} == _CHAT_EXTRA_FIELDS_PAYLOAD


@pytest.mark.asyncio
async def test_post_json_strips_extra_chat_fields_by_default():
    sent = await _post_chat_extra_fields(ProxyOptions(url="http://api.openai.com/"))

    assert not set(_CHAT_EXTRA_FIELDS_PAYLOAD) & set(sent)
    assert sent["reasoning_effort"] == "low"


@pytest.mark.asyncio
async def test_post_json_keeps_legacy_openai_fields_for_remote_completions():
    body = CompletionLegacyRequest(model="m", prompt="hi", presence_penalty=0.5, logprobs=2, **_LEGACY_EXTRA_FIELDS_PAYLOAD)  # pyright: ignore[reportArgumentType]
    mock_http_response = MagicMock()
    mock_http_response.as_streaming_response.return_value = MagicMock()

    with patch("server.endpointregistry.make_http_request", new_callable=AsyncMock, return_value=mock_http_response) as mock_req:
        await post_json(body, ProxyOptions(url="http://api.openai.com/"))

    sent = json.loads(mock_req.call_args.kwargs["data"]._value)
    assert sent["presence_penalty"] == 0.5
    assert sent["logprobs"] == 2
    assert not set(_LEGACY_EXTRA_FIELDS_PAYLOAD) & set(sent)


@pytest.mark.asyncio
async def test_post_json_forwards_legacy_extra_fields_when_enabled():
    body = CompletionLegacyRequest(model="m", prompt="hi", **_LEGACY_EXTRA_FIELDS_PAYLOAD)  # pyright: ignore[reportArgumentType]
    mock_http_response = MagicMock()
    mock_http_response.as_streaming_response.return_value = MagicMock()

    with patch("server.endpointregistry.make_http_request", new_callable=AsyncMock, return_value=mock_http_response) as mock_req:
        await post_json(body, ProxyOptions(url="http://vllm/", forward_extra_fields=True))

    sent = json.loads(mock_req.call_args.kwargs["data"]._value)
    assert {field: sent[field] for field in _LEGACY_EXTRA_FIELDS_PAYLOAD} == _LEGACY_EXTRA_FIELDS_PAYLOAD


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

    with patch("server.endpointregistry.post_json_responses", new_callable=AsyncMock, return_value=mock_resp):
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
async def test_register_rerank_as_proxy_normalize_sglang_invokes_post_json_rerank_sglang():
    reg = make_registry()
    opts = ProxyOptions(url="http://example.com/v1/rerank")
    reg.register_rerank_as_proxy("rnk-sglang", make_props(), opts, None, normalize_sglang_response=True)
    ep = reg.rerank_endpoints.get_model("rnk-sglang")
    mock_resp = MagicMock(spec=StreamingResponse)

    with patch("server.endpointregistry.post_json_rerank_sglang", new_callable=AsyncMock, return_value=mock_resp) as mock_call:
        body = MagicMock(spec=RerankRequest)
        result = await ep.endpoint.on_request(body, None)  # pyright: ignore[reportOptionalMemberAccess]

    assert result is mock_resp
    mock_call.assert_awaited_once_with(body, opts, None, timeout=reg._proxy_timeout)  # type: ignore[reportPrivateUsage]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("read_timeout_seconds", "expected_sock_read"),
    [
        pytest.param(None, 300, id="default"),
        pytest.param(1800, 1800, id="configured"),
    ],
)
async def test_register_custom_endpoint_as_proxy_callback_invokes_make_http_request(
    read_timeout_seconds: int | None, expected_sock_read: int
) -> None:
    reg = make_registry()
    opts = (
        ProxyOptions(url="http://example.com/custom/")
        if read_timeout_seconds is None
        else ProxyOptions(url="http://example.com/custom/", read_timeout_seconds=read_timeout_seconds)
    )
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

    with patch("server.endpointregistry.make_http_request", new_callable=AsyncMock, return_value=mock_http_response) as mock_call:
        result = await ep.endpoint.on_request(request)  # pyright: ignore[reportOptionalMemberAccess]

    assert result is mock_streaming
    assert mock_call.call_args.kwargs["timeout"] == aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=expected_sock_read)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("forward_full_path", "expected_url"),
    [
        pytest.param(False, "http://example.com/custom/sub/path", id="strips-prefix-for-local-container"),
        pytest.param(True, "http://example.com/custom/my-svc/sub/path", id="forwards-full-path-for-mesh"),
    ],
)
async def test_register_custom_endpoint_as_proxy_callback_forward_full_path(forward_full_path: bool, expected_url: str) -> None:
    reg = make_registry()
    opts = ProxyOptions(url="http://example.com/custom", forward_full_path=forward_full_path)
    reg.register_custom_endpoint_as_proxy("my-svc", make_props(), opts, None)
    ep = reg.custom_endpoints.get_model("my-svc")
    mock_http_response = MagicMock()
    mock_streaming = MagicMock(spec=StreamingResponse)
    mock_http_response.as_streaming_response.return_value = mock_streaming
    request = MagicMock()
    request.headers = {"content-type": "application/json"}
    request.method = "POST"
    request.path_params = {"full_path": "my-svc/sub/path"}
    request.url.query = ""
    request.stream.return_value = AsyncMock()

    with patch("server.endpointregistry.make_http_request", new_callable=AsyncMock, return_value=mock_http_response) as mock_call:
        result = await ep.endpoint.on_request(request)  # pyright: ignore[reportOptionalMemberAccess]

    assert result is mock_streaming
    assert mock_call.call_args.kwargs["url"] == expected_url


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
    request.stream = lambda: _async_byte_stream(b"{}")

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
    request.stream = lambda: _async_byte_stream(b"")
    request.url.query = ""

    with patch("server.endpointregistry.make_http_request", new_callable=AsyncMock, return_value=mock_http_response) as mock_req:
        await ep.endpoint.on_request(request)  # pyright: ignore[reportOptionalMemberAccess]

    assert "?" not in mock_req.call_args.kwargs["url"]


@pytest.mark.asyncio
async def test_register_mcp_endpoint_as_proxy_streams_body_when_no_on_reauth():
    """Without `on_reauth` (the common, non-OAuth case) the request body must stream straight through, not buffer."""
    reg = make_registry()
    opts = ProxyOptions(url="http://example.com/mcp/")
    reg.register_mcp_endpoint_as_proxy("mcp-svc", make_props(), opts, None)
    ep = reg.mcp_endpoints.get_model("mcp-svc")
    mock_http_response = MagicMock()
    mock_http_response.as_streaming_response.return_value = MagicMock(spec=StreamingResponse)
    stream = _async_byte_stream(b"{}")
    request = MagicMock()
    request.headers = {"content-type": "application/json"}
    request.method = "POST"
    request.path_params = {"full_path": "mcp-svc"}
    request.stream = MagicMock(return_value=stream)
    request.url.query = ""

    with patch("server.endpointregistry.make_http_request", new_callable=AsyncMock, return_value=mock_http_response) as mock_req:
        await ep.endpoint.on_request(request)  # pyright: ignore[reportOptionalMemberAccess]

    assert mock_req.call_args.kwargs["data"] is stream
    request.stream.assert_called_once()


@pytest.mark.asyncio
async def test_register_mcp_endpoint_as_proxy_buffers_body_when_on_reauth_set():
    """With `on_reauth` configured (OAuth-enabled proxies) the body must be buffered so a 401 retry can replay it."""
    reg = make_registry()
    opts = ProxyOptions(url="http://example.com/mcp/", on_reauth=AsyncMock())
    reg.register_mcp_endpoint_as_proxy("mcp-svc", make_props(), opts, None)
    ep = reg.mcp_endpoints.get_model("mcp-svc")
    mock_http_response = MagicMock()
    mock_http_response.response.status = 200
    mock_http_response.as_streaming_response.return_value = MagicMock(spec=StreamingResponse)
    request = MagicMock()
    request.headers = {"content-type": "application/json"}
    request.method = "POST"
    request.path_params = {"full_path": "mcp-svc"}
    request.stream = lambda: _async_byte_stream(b'{"jsonrpc":"2.0"}')
    request.url.query = ""

    with patch("server.endpointregistry.make_http_request", new_callable=AsyncMock, return_value=mock_http_response) as mock_req:
        await ep.endpoint.on_request(request)  # pyright: ignore[reportOptionalMemberAccess]

    assert mock_req.call_args.kwargs["data"] == b'{"jsonrpc":"2.0"}'


@pytest.mark.asyncio
async def test_make_request_with_reauth_passes_through_non_401():
    on_reauth = AsyncMock()
    opts = ProxyOptions(url="http://example.com/mcp/", on_reauth=on_reauth)
    ok_response = MagicMock()
    ok_response.response.status = 200

    with patch("server.endpointregistry.make_http_request", new_callable=AsyncMock, return_value=ok_response) as mock_req:
        result = await _make_request_with_reauth("http://example.com/mcp/", "POST", b"{}", {}, opts)

    assert result is ok_response
    assert mock_req.await_count == 1
    assert on_reauth.call_count == 0


@pytest.mark.asyncio
async def test_make_request_with_reauth_401_without_on_reauth_passes_through():
    opts = ProxyOptions(url="http://example.com/mcp/")
    unauthorized_response = MagicMock()
    unauthorized_response.response.status = 401
    unauthorized_response.discard = AsyncMock()

    with patch("server.endpointregistry.make_http_request", new_callable=AsyncMock, return_value=unauthorized_response) as mock_req:
        result = await _make_request_with_reauth("http://example.com/mcp/", "POST", b"{}", {}, opts)

    assert result is unauthorized_response
    mock_req.assert_awaited_once()
    unauthorized_response.discard.assert_not_called()


@pytest.mark.asyncio
async def test_make_request_with_reauth_401_on_reauth_returns_none_passes_through():
    opts = ProxyOptions(url="http://example.com/mcp/", on_reauth=AsyncMock(return_value=None))
    unauthorized_response = MagicMock()
    unauthorized_response.response.status = 401
    unauthorized_response.discard = AsyncMock()

    with patch("server.endpointregistry.make_http_request", new_callable=AsyncMock, return_value=unauthorized_response) as mock_req:
        result = await _make_request_with_reauth("http://example.com/mcp/", "POST", b"{}", {"x": "1"}, opts)

    assert result is unauthorized_response
    mock_req.assert_awaited_once()
    unauthorized_response.discard.assert_not_called()


@pytest.mark.asyncio
async def test_make_request_with_reauth_401_retries_with_refreshed_headers():
    opts = ProxyOptions(url="http://example.com/mcp/", on_reauth=AsyncMock(return_value={"Authorization": "Bearer new"}))
    unauthorized_response = MagicMock()
    unauthorized_response.response.status = 401
    unauthorized_response.discard = AsyncMock()
    retried_response = MagicMock()
    retried_response.response.status = 200

    with patch(
        "server.endpointregistry.make_http_request",
        new_callable=AsyncMock,
        side_effect=[unauthorized_response, retried_response],
    ) as mock_req:
        result = await _make_request_with_reauth("http://example.com/mcp/", "POST", b"{}", {"Authorization": "Bearer old"}, opts)

    assert result is retried_response
    assert mock_req.await_count == 2
    unauthorized_response.discard.assert_awaited_once()
    second_call_headers = mock_req.call_args_list[1].kwargs["headers"]
    assert second_call_headers["Authorization"] == "Bearer new"


@pytest.mark.asyncio
async def test_read_body_with_limit_returns_full_body_under_limit():
    request = MagicMock()
    request.stream = lambda: _async_byte_stream(b"hello world")

    body = await _read_body_with_limit(request, 1024)

    assert body == b"hello world"


@pytest.mark.asyncio
async def test_read_body_with_limit_raises_413_over_limit():
    request = MagicMock()
    request.stream = lambda: _async_byte_stream(b"way too much body")

    with pytest.raises(HTTPException) as exc_info:
        await _read_body_with_limit(request, 4)

    assert exc_info.value.status_code == 413


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
    body.input = None
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
    return RegisteredModel(
        id="fake-id", name="gpt-4", origin="local", props=make_props(), type="llm", endpoint=ep, usage=0, created=0, owned_by="local"
    )


@pytest.mark.asyncio
async def test_execute_messages_includes_supported_endpoints_in_error():
    reg = make_registry()
    ep = make_chat_endpoint(on_chat=True, on_responses=True, on_completion=True, on_messages=False)
    fake = _make_registered_model_with_endpoint(ep)
    reg.register_chat_completion("gpt-4", make_props(), ep, RegistrationOptions(origin="local"))
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
    reg.register_chat_completion("gpt-4", make_props(), ep, RegistrationOptions(origin="local"))
    body = MagicMock(spec=ResponsesRequest)
    body.model = "gpt-4"
    body.input = None
    body.model_dump_json.return_value = "{}"

    with patch.object(reg.chat_completion_endpoints, "get_model", return_value=fake), pytest.raises(HTTPException) as exc_info:
        await reg.execute_responses(body)

    assert "/v1/chat/completions" in exc_info.value.detail or "/v1/messages" in exc_info.value.detail


@pytest.mark.asyncio
async def test_execute_chat_completion_includes_supported_endpoints_in_error():
    reg = make_registry()
    ep = make_chat_endpoint(on_chat=False, on_messages=True, on_responses=True, on_completion=True)
    fake = _make_registered_model_with_endpoint(ep)
    reg.register_chat_completion("gpt-4", make_props(), ep, RegistrationOptions(origin="local"))
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
    reg.register_chat_completion("gpt-4", make_props(), ep, RegistrationOptions(origin="local"))
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
    reg.register_chat_completion("gpt-4", make_props(), ep, RegistrationOptions(origin="local"))
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
    body.input = None
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
    body.response_format = "json"

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
    reg.register_chat_completion("gpt-4", make_props(), ep, RegistrationOptions(origin="local"))
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
    reg.register_chat_completion("gpt-4", make_props(), ep, RegistrationOptions(origin="local"))
    body = MagicMock(spec=ResponsesRequest)
    body.model = "gpt-4"
    body.input = None
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
    reg.register_chat_completion("gpt-4", make_props(), ep, RegistrationOptions(origin="local"))
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
    reg.register_chat_completion("gpt-4", make_props(), ep, RegistrationOptions(origin="local"))
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
    reg.register_chat_completion("gpt-4", make_props(), ep, RegistrationOptions(origin="local"))
    body = MagicMock(spec=OllamaChatRequest)
    body.model = "gpt-4"

    with patch.object(reg.chat_completion_endpoints, "get_model", return_value=fake), pytest.raises(HTTPException) as exc_info:
        await reg.execute_ollama_chat(body)

    assert "Given model not support" in exc_info.value.detail
    assert "/v1/messages" not in exc_info.value.detail
    assert "/v1/responses" not in exc_info.value.detail
    assert "/v1/chat/completions" not in exc_info.value.detail
    assert "/v1/completions" not in exc_info.value.detail


def test_session_store_add_and_get() -> None:
    store = McpSseSessionStore()
    store.add("sid1", "https://upstream/msg?sessionId=sid1")
    assert store.get("sid1") == "https://upstream/msg?sessionId=sid1"


def test_session_store_get_unknown_returns_none() -> None:
    store = McpSseSessionStore()
    assert store.get("nonexistent") is None


def test_session_store_remove() -> None:
    store = McpSseSessionStore()
    store.add("sid1", "https://upstream/msg?sessionId=sid1")
    store.remove("sid1")
    assert store.get("sid1") is None


def test_session_store_remove_unknown_is_noop() -> None:
    store = McpSseSessionStore()
    store.remove("nonexistent")  # must not raise


def test_session_store_evicts_expired(monkeypatch: pytest.MonkeyPatch) -> None:
    store = McpSseSessionStore(ttl_seconds=10)
    real_now = time.monotonic()
    store.add("sid1", "url1")
    # Make the store think 20 seconds have passed
    monkeypatch.setattr(_er_module.time, "monotonic", lambda: real_now + 20)
    assert store.get("sid1") is None


def test_model_exists_returns_true_when_registered():
    reg = make_registry()
    reg.register_embeddings("emb", make_props(), SimpleEndpoint(on_request=AsyncMock()), None)

    result = reg.model_exists("emb")

    assert result is True


def test_model_exists_returns_false_when_not_registered():
    reg = make_registry()

    result = reg.model_exists("unknown")

    assert result is False


def test_model_exists_ignores_custom_and_mcp_endpoints():
    reg = make_registry()
    reg.register_custom_endpoint("custom/path", make_props(), CustomEndpoint(on_request=AsyncMock()), None)
    reg.register_mcp_endpoint("mcp/url", make_props(), McpEndpoint(on_request=AsyncMock()), None)

    assert reg.model_exists("custom/path") is False
    assert reg.model_exists("mcp/url") is False


@pytest.mark.asyncio
async def test_execute_embeddings_raises_400_when_model_registered_for_other_endpoint():
    reg = make_registry()
    reg.register_chat_completion("shared", make_props(), make_chat_endpoint(), RegistrationOptions(origin="local"))
    body = MagicMock(spec=EmbeddingRequest)
    body.model = "shared"

    with pytest.raises(HTTPException) as exc_info:
        await reg.execute_embeddings(body)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Given model is not supported"


@pytest.mark.asyncio
async def test_execute_images_generations_raises_400_when_model_registered_for_other_endpoint():
    reg = make_registry()
    reg.register_embeddings("shared", make_props(), SimpleEndpoint(on_request=AsyncMock()), RegistrationOptions(origin="local"))
    body = MagicMock(spec=ImagesRequest)
    body.model = "shared"
    body.model_dump_json.return_value = "{}"

    with pytest.raises(HTTPException) as exc_info:
        await reg.execute_images_generations(body)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Given model is not supported"


@pytest.mark.asyncio
async def test_execute_audio_speech_raises_400_when_model_registered_for_other_endpoint():
    reg = make_registry()
    reg.register_embeddings("shared", make_props(), SimpleEndpoint(on_request=AsyncMock()), RegistrationOptions(origin="local"))
    body = MagicMock(spec=CreateSpeechRequest)
    body.model = "shared"
    body.model_dump_json.return_value = "{}"

    with pytest.raises(HTTPException) as exc_info:
        await reg.execute_audio_speech(body)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Given model is not supported"


@pytest.mark.asyncio
async def test_execute_audio_transcriptions_raises_400_when_model_registered_for_other_endpoint():
    reg = make_registry()
    reg.register_embeddings("shared", make_props(), SimpleEndpoint(on_request=AsyncMock()), RegistrationOptions(origin="local"))
    body = MagicMock(spec=CreateTranscriptionRequest)
    body.model = "shared"

    with pytest.raises(HTTPException) as exc_info:
        await reg.execute_audio_transcriptions(body)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Given model is not supported"


@pytest.mark.asyncio
async def test_execute_rerank_raises_400_when_model_registered_for_other_endpoint():
    reg = make_registry()
    reg.register_embeddings("shared", make_props(), SimpleEndpoint(on_request=AsyncMock()), RegistrationOptions(origin="local"))
    body = MagicMock(spec=RerankRequest)
    body.model = "shared"
    body.model_dump_json.return_value = "{}"

    with pytest.raises(HTTPException) as exc_info:
        await reg.execute_rerank(body)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Given model is not supported"


@pytest.mark.asyncio
async def test_execute_messages_raises_400_when_model_registered_for_other_endpoint():
    reg = make_registry()
    reg.register_embeddings("shared", make_props(), SimpleEndpoint(on_request=AsyncMock()), RegistrationOptions(origin="local"))
    body = MagicMock(spec=MessagesRequest)
    body.model = "shared"
    body.model_dump_json.return_value = "{}"

    with pytest.raises(HTTPException) as exc_info:
        await reg.execute_messages(body)

    assert exc_info.value.status_code == 400
    assert "Given model not support this endpoint" in exc_info.value.detail


def test_session_store_evicts_oldest_when_at_capacity() -> None:
    store = McpSseSessionStore(max_sessions=2)
    store.add("sid1", "url1")
    store.add("sid2", "url2")
    store.add("sid3", "url3")
    assert store.get("sid1") is None  # evicted
    assert store.get("sid2") == "url2"
    assert store.get("sid3") == "url3"


def test_session_store_add_overwrites_existing() -> None:
    store = McpSseSessionStore()
    store.add("sid1", "url-v1")
    store.add("sid1", "url-v2")
    assert store.get("sid1") == "url-v2"


async def _collect(gen: object) -> bytes:
    """Consume an async generator and return all bytes concatenated."""
    result = b""
    async for chunk in gen:  # type: ignore[union-attr]
        result += chunk
    return result


@pytest.mark.asyncio
async def test_rewrite_rewrites_endpoint_data_url() -> None:
    store = McpSseSessionStore()

    async def source():  # type: ignore[return]
        yield b"event: endpoint\ndata: https://upstream.example.com/msg?sessionId=abc123\n\n"

    result = await _collect(
        _rewrite_sse_endpoint_events(source(), "https://upstream.example.com/sse", "https://proxy.example.com/mcp", store)
    )

    assert b"data: https://proxy.example.com/mcp?sessionId=abc123" in result
    assert store.get("abc123") == "https://upstream.example.com/msg?sessionId=abc123"


@pytest.mark.asyncio
async def test_rewrite_passes_through_non_endpoint_events() -> None:
    store = McpSseSessionStore()

    async def source():  # type: ignore[return]
        yield b'event: message\ndata: {"jsonrpc":"2.0"}\n\n'

    result = await _collect(_rewrite_sse_endpoint_events(source(), "https://up.example.com/sse", "https://proxy.example.com/mcp", store))

    assert b'data: {"jsonrpc":"2.0"}' in result
    assert b"proxy" not in result.split(b"data:")[1]


@pytest.mark.asyncio
async def test_rewrite_resolves_relative_endpoint_url() -> None:
    store = McpSseSessionStore()

    async def source():  # type: ignore[return]
        yield b"event: endpoint\ndata: /msg?sessionId=rel123\n\n"

    result = await _collect(
        _rewrite_sse_endpoint_events(source(), "https://upstream.example.com/sse", "https://proxy.example.com/mcp", store)
    )

    assert b"sessionId=rel123" in result
    assert store.get("rel123") is not None


@pytest.mark.asyncio
async def test_rewrite_handles_chunked_delivery() -> None:
    """SSE data split across multiple chunks must be reassembled correctly."""
    store = McpSseSessionStore()

    async def source():  # type: ignore[return]
        yield b"event: endpoi"
        yield b"nt\ndata: https://up.example.com/msg?sessionId=chunk1\n\n"

    result = await _collect(_rewrite_sse_endpoint_events(source(), "https://up.example.com/sse", "https://proxy.example.com/mcp", store))

    assert b"sessionId=chunk1" in result
    assert store.get("chunk1") is not None


@pytest.mark.asyncio
async def test_rewrite_passes_through_non_event_non_data_lines() -> None:
    store = McpSseSessionStore()

    async def source():  # type: ignore[return]
        yield b"retry: 3000\n\n"

    result = await _collect(_rewrite_sse_endpoint_events(source(), "https://up.example.com/sse", "https://proxy.example.com/mcp", store))

    assert b"retry: 3000" in result


@pytest.mark.asyncio
async def test_rewrite_emits_remaining_buffer_without_newline() -> None:
    """Incomplete final line (no trailing newline) must still be emitted."""
    store = McpSseSessionStore()

    async def source():  # type: ignore[return]
        yield b"id: 42"  # no trailing \n

    result = await _collect(_rewrite_sse_endpoint_events(source(), "https://up.example.com/sse", "https://proxy.example.com/mcp", store))

    assert b"id: 42" in result


@pytest.mark.asyncio
async def test_mcp_sse_proxy_get_streams_rewritten_events() -> None:
    reg = make_registry()
    reg.config.mcp_sse_session_ttl_seconds = 300
    reg.config.mcp_sse_max_sessions = 128
    opts = ProxyOptions(url="https://upstream.example.com/sse")
    reg.register_mcp_sse_endpoint_as_proxy("sse-svc", make_props(), opts, None)
    ep = reg.mcp_endpoints.get_model("sse-svc")

    async def fake_content():  # type: ignore[return]
        yield b"event: endpoint\ndata: https://upstream.example.com/msg?sessionId=sess1\n\n"

    mock_upstream = MagicMock()
    mock_upstream.content = fake_content()
    mock_upstream.response = AsyncMock()
    mock_upstream.response.release = AsyncMock()

    request = MagicMock()
    request.method = "GET"
    request.headers = {}
    request.url = MagicMock()
    request.url.__str__ = MagicMock(return_value="https://proxy.example.com/mcp/sse-svc/mcp")

    with patch("server.endpointregistry.make_http_request", new_callable=AsyncMock, return_value=mock_upstream):
        response = await ep.endpoint.on_request(request)  # pyright: ignore[reportOptionalMemberAccess]

    assert isinstance(response, StreamingResponse)
    assert response.status_code == 200

    # Consume the streaming body to verify URL rewriting
    chunks: list[bytes] = []
    async for chunk in response.body_iterator:  # type: ignore[union-attr]
        chunks.append(chunk if isinstance(chunk, bytes) else chunk.encode())  # type: ignore[union-attr]
    body = b"".join(chunks)
    assert b"sessionId=sess1" in body
    assert b"proxy.example.com" in body


@pytest.mark.asyncio
async def test_mcp_sse_proxy_post_missing_session_id_raises_400() -> None:
    reg = make_registry()
    reg.config.mcp_sse_session_ttl_seconds = 300
    reg.config.mcp_sse_max_sessions = 128
    opts = ProxyOptions(url="https://upstream.example.com/sse")
    reg.register_mcp_sse_endpoint_as_proxy("sse-svc", make_props(), opts, None)
    ep = reg.mcp_endpoints.get_model("sse-svc")

    request = MagicMock()
    request.method = "POST"
    request.headers = {}
    request.query_params = {}

    with pytest.raises(HTTPException) as exc_info:
        await ep.endpoint.on_request(request)  # pyright: ignore[reportOptionalMemberAccess]
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_mcp_sse_proxy_post_unknown_session_id_raises_404() -> None:
    reg = make_registry()
    reg.config.mcp_sse_session_ttl_seconds = 300
    reg.config.mcp_sse_max_sessions = 128
    opts = ProxyOptions(url="https://upstream.example.com/sse")
    reg.register_mcp_sse_endpoint_as_proxy("sse-svc", make_props(), opts, None)
    ep = reg.mcp_endpoints.get_model("sse-svc")

    request = MagicMock()
    request.method = "POST"
    request.headers = {}
    request.query_params = {"sessionId": "unknown-session"}

    with pytest.raises(HTTPException) as exc_info:
        await ep.endpoint.on_request(request)  # pyright: ignore[reportOptionalMemberAccess]
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_mcp_sse_proxy_post_valid_session_forwards_to_upstream() -> None:
    mock_store = MagicMock(spec=McpSseSessionStore)
    mock_store.get.return_value = "https://upstream.example.com/msg?sessionId=valid1"

    opts = ProxyOptions(url="https://upstream.example.com/sse", allowed_response_headers=["content-type"])
    with patch("server.endpointregistry.McpSseSessionStore", return_value=mock_store):
        reg = make_registry()
        reg.config.mcp_sse_session_ttl_seconds = 300
        reg.config.mcp_sse_max_sessions = 128
        reg.register_mcp_sse_endpoint_as_proxy("sse-svc2", make_props(), opts, None)

    ep = reg.mcp_endpoints.get_model("sse-svc2")

    mock_upstream_post = MagicMock()
    mock_streaming = MagicMock(spec=StreamingResponse)
    mock_upstream_post.as_streaming_response.return_value = mock_streaming

    post_request = MagicMock()
    post_request.method = "POST"
    post_request.headers = {"content-type": "application/json"}
    post_request.query_params = {"sessionId": "valid1"}
    post_request.stream = MagicMock()

    with patch("server.endpointregistry.make_http_request", new_callable=AsyncMock, return_value=mock_upstream_post):
        result = await ep.endpoint.on_request(post_request)  # pyright: ignore[reportOptionalMemberAccess]

    assert result is mock_streaming
    mock_upstream_post.as_streaming_response.assert_called_once_with(["content-type"])


def test_session_store_evict_expired_removes_stale_session() -> None:
    """Line 258: del self._sessions[sid] in _evict_expired loop."""
    store = McpSseSessionStore(ttl_seconds=1, max_sessions=128)
    store.add("stale", "url-stale")
    # Manually backdate the timestamp so the session appears expired
    store._sessions["stale"] = ("url-stale", store._sessions["stale"][1] - 2)  # pyright: ignore[reportPrivateUsage]
    # Calling add() triggers _evict_expired, which should delete "stale"
    store.add("fresh", "url-fresh")
    assert store.get("stale") is None
    assert store.get("fresh") == "url-fresh"


@pytest.mark.asyncio
async def test_rewrite_endpoint_data_without_session_id_passes_through() -> None:
    """Line 319: data line in endpoint event with no sessionId is yielded as-is."""
    store = McpSseSessionStore()

    async def source():  # type: ignore[return]
        yield b"event: endpoint\ndata: https://upstream.example.com/no-session-here\n\n"

    result = await _collect(
        _rewrite_sse_endpoint_events(source(), "https://upstream.example.com/sse", "https://proxy.example.com/mcp", store)
    )

    assert b"data: https://upstream.example.com/no-session-here" in result
    assert b"proxy.example.com" not in result


@pytest.mark.asyncio
async def test_mcp_sse_proxy_get_finally_no_session_id() -> None:
    """Lines 733->735: finally block with session_id=None skips remove() call."""
    reg = make_registry()
    reg.config.mcp_sse_session_ttl_seconds = 300
    reg.config.mcp_sse_max_sessions = 128
    opts = ProxyOptions(url="https://upstream.example.com/sse")
    reg.register_mcp_sse_endpoint_as_proxy("sse-nosession", make_props(), opts, None)
    ep = reg.mcp_endpoints.get_model("sse-nosession")

    async def fake_content():  # type: ignore[return]
        # SSE stream with no endpoint event → no sessionId captured
        yield b"event: message\ndata: hello\n\n"

    mock_upstream = MagicMock()
    mock_upstream.content = fake_content()
    mock_upstream.response = AsyncMock()
    mock_upstream.response.release = AsyncMock()

    request = MagicMock()
    request.method = "GET"
    request.headers = {}
    request.url = MagicMock()
    request.url.__str__ = MagicMock(return_value="https://proxy.example.com/mcp/sse-nosession/mcp")

    with patch("server.endpointregistry.make_http_request", new_callable=AsyncMock, return_value=mock_upstream):
        response = await ep.endpoint.on_request(request)  # pyright: ignore[reportOptionalMemberAccess]

    # Consume the full stream — finally block fires with session_id=None
    chunks: list[bytes] = []
    async for chunk in response.body_iterator:  # type: ignore[union-attr]
        chunks.append(chunk if isinstance(chunk, bytes) else chunk.encode())  # type: ignore[union-attr]

    mock_upstream.response.release.assert_awaited_once()


# ---------------------------------------------------------------------------
# New Endpoint methods: get_model_prefix, get_model_transport, get_model_tools
# ---------------------------------------------------------------------------


def test_get_model_prefix_returns_prefix_when_set() -> None:
    ep = make_endpoint()
    props = make_props()
    props.prefix = "my-prefix"
    ep.add_model("model1", props, SimpleEndpoint(on_request=AsyncMock()), "mcp", RegistrationOptions(origin="local"))

    result = ep.get_model_prefix(ep.models["model1"])

    assert result == "my-prefix"


def test_get_model_prefix_returns_none_when_no_prefix() -> None:
    ep = make_endpoint()
    ep.add_model("model1", make_props(), SimpleEndpoint(on_request=AsyncMock()), "mcp", RegistrationOptions(origin="local"))

    result = ep.get_model_prefix(ep.models["model1"])

    assert result is None


def test_get_model_transport_returns_transport_when_set() -> None:
    ep = make_endpoint()
    props = make_props()
    props.transport = "streamable_http"
    ep.add_model("model1", props, SimpleEndpoint(on_request=AsyncMock()), "mcp", RegistrationOptions(origin="local"))

    result = ep.get_model_transport(ep.models["model1"])

    assert result == "streamable_http"


def test_get_model_transport_returns_none_when_no_transport() -> None:
    ep = make_endpoint()
    ep.add_model("model1", make_props(), SimpleEndpoint(on_request=AsyncMock()), "mcp", RegistrationOptions(origin="local"))

    result = ep.get_model_transport(ep.models["model1"])

    assert result is None


def test_get_model_tools_returns_tools_when_set() -> None:
    ep = make_endpoint()
    props = make_props()
    props.tools = [McpToolInfo(name="search", description="web search")]
    ep.add_model("model1", props, SimpleEndpoint(on_request=AsyncMock()), "mcp", RegistrationOptions(origin="local"))

    result = ep.get_model_tools(ep.models["model1"])

    assert len(result) == 1
    assert result[0].name == "search"


def test_get_model_tools_returns_empty_when_no_tools() -> None:
    ep = make_endpoint()
    ep.add_model("model1", make_props(), SimpleEndpoint(on_request=AsyncMock()), "mcp", RegistrationOptions(origin="local"))

    result = ep.get_model_tools(ep.models["model1"])

    assert result == []


def test_get_healthy_models_returns_only_healthy() -> None:
    ep = make_endpoint()
    ep.add_model("healthy-m", make_props(), SimpleEndpoint(on_request=AsyncMock()), "mcp", RegistrationOptions(origin="local", id="r1"))
    ep.add_model("unhealthy-m", make_props(), SimpleEndpoint(on_request=AsyncMock()), "mcp", RegistrationOptions(origin="local", id="r2"))

    for reg_model in ep.models["healthy-m"].values():
        reg_model.healthy = True

    result = ep.get_healthy_models()

    ids = [m.id for m in result]
    assert "healthy-m" in ids
    assert "unhealthy-m" not in ids


def test_get_healthy_models_returns_empty_when_none_healthy() -> None:
    ep = make_endpoint()
    ep.add_model("m1", make_props(), SimpleEndpoint(on_request=AsyncMock()), "mcp", RegistrationOptions(origin="local"))

    result = ep.get_healthy_models()

    assert result == []


def test_registry_get_models_includes_healthy_mcp_models() -> None:
    reg = make_registry()
    props = make_props(type="mcp", endpoints=["/mcp/my-mcp/mcp"])
    reg.register_mcp_endpoint("my-mcp", props, McpEndpoint(on_request=AsyncMock()), RegistrationOptions(origin="local", id="r1"))

    for reg_model in reg.mcp_endpoints.models["my-mcp"].values():
        reg_model.healthy = True

    ids = [m.id for m in reg.get_models().data]

    assert "my-mcp" in ids


def test_registry_get_models_excludes_unhealthy_mcp_models() -> None:
    reg = make_registry()
    props = make_props(type="mcp", endpoints=["/mcp/unhealthy/mcp"])
    reg.register_mcp_endpoint("unhealthy", props, McpEndpoint(on_request=AsyncMock()), RegistrationOptions(origin="local"))

    ids = [m.id for m in reg.get_models().data]

    assert "unhealthy" not in ids


def test_registry_get_models_includes_custom_models() -> None:
    reg = make_registry()
    reg.register_custom_endpoint("my-custom", make_props(type="custom"), CustomEndpoint(on_request=AsyncMock()), None)

    ids = [m.id for m in reg.get_models().data]

    assert "my-custom" in ids


def test_response_item_store_put_and_get() -> None:
    store = ResponseItemStore()
    store.put({"id": "rs_1", "type": "reasoning", "summary": []})
    assert store.get("rs_1") == {"id": "rs_1", "type": "reasoning", "summary": []}


def test_response_item_store_get_unknown_returns_none() -> None:
    store = ResponseItemStore()
    assert store.get("nonexistent") is None


def test_response_item_store_put_ignores_item_without_id() -> None:
    store = ResponseItemStore()
    store.put({"type": "reasoning", "summary": []})
    assert store.get("rs_1") is None


def test_response_item_store_evicts_expired(monkeypatch: pytest.MonkeyPatch) -> None:
    store = ResponseItemStore(ttl_seconds=10)
    real_now = time.monotonic()
    store.put({"id": "rs_1", "type": "reasoning", "summary": []})
    monkeypatch.setattr(_er_module.time, "monotonic", lambda: real_now + 20)
    assert store.get("rs_1") is None


def test_response_item_store_put_prunes_expired_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    store = ResponseItemStore(ttl_seconds=10)
    real_now = time.monotonic()
    store.put({"id": "rs_1", "type": "reasoning", "summary": []})
    monkeypatch.setattr(_er_module.time, "monotonic", lambda: real_now + 20)
    store.put({"id": "rs_2", "type": "reasoning", "summary": []})
    assert "rs_1" not in store._items  # pyright: ignore[reportPrivateUsage]


def test_response_item_store_evicts_oldest_when_at_capacity() -> None:
    store = ResponseItemStore(max_items=2)
    store.put({"id": "rs_1", "type": "reasoning", "summary": []})
    store.put({"id": "rs_2", "type": "reasoning", "summary": []})
    store.put({"id": "rs_3", "type": "reasoning", "summary": []})
    assert store.get("rs_1") is None  # evicted
    assert store.get("rs_2") is not None
    assert store.get("rs_3") is not None


@pytest.mark.asyncio
async def test_post_json_responses_captures_output_items_and_returns_body_unchanged() -> None:
    response_body = json.dumps({"id": "resp_1", "output": [{"id": "rs_871851", "type": "reasoning", "summary": []}]}).encode()

    async def content_gen():  # type: ignore[return]
        yield response_body

    mock_http_response = MagicMock()
    mock_http_response.response.content_type = "application/json"
    mock_http_response.response.status = 200
    mock_http_response.response.headers = {}
    mock_http_response.content = content_gen()

    store = ResponseItemStore()
    opts = ProxyOptions(url="http://example.com/v1/responses")

    with patch("server.endpointregistry.make_http_request", new_callable=AsyncMock, return_value=mock_http_response):
        result = await post_json_responses(ResponsesRequest(model="gemma4:e4b", stream=False), opts, store)

    assert store.get("rs_871851") == {"id": "rs_871851", "type": "reasoning", "summary": []}
    assert await _collect(result.body_iterator) == response_body


@pytest.mark.asyncio
async def test_post_json_responses_skips_capture_when_streaming() -> None:
    mock_http_response = MagicMock()
    mock_streaming = MagicMock(spec=StreamingResponse)
    mock_http_response.as_streaming_response.return_value = mock_streaming

    store = ResponseItemStore()
    opts = ProxyOptions(url="http://example.com/v1/responses")

    with patch("server.endpointregistry.make_http_request", new_callable=AsyncMock, return_value=mock_http_response):
        result = await post_json_responses(ResponsesRequest(model="gemma4:e4b", stream=True), opts, store)

    assert result is mock_streaming
    mock_http_response.as_streaming_response.assert_called_once()


@pytest.mark.asyncio
async def test_post_json_responses_logs_and_passes_through_on_invalid_json(caplog: pytest.LogCaptureFixture) -> None:
    async def content_gen():  # type: ignore[return]
        yield b"not json"

    mock_http_response = MagicMock()
    mock_http_response.response.content_type = "application/json"
    mock_http_response.response.status = 200
    mock_http_response.response.headers = {}
    mock_http_response.content = content_gen()

    store = ResponseItemStore()
    opts = ProxyOptions(url="http://example.com/v1/responses")

    with (
        caplog.at_level(logging.WARNING, logger="uvicorn"),
        patch("server.endpointregistry.make_http_request", new_callable=AsyncMock, return_value=mock_http_response),
    ):
        result = await post_json_responses(ResponsesRequest(model="gemma4:e4b", stream=False), opts, store)

    assert await _collect(result.body_iterator) == b"not json"
    assert "Failed to parse" in caplog.text


@pytest.mark.asyncio
async def test_post_json_rerank_sglang_rewrites_array_response_to_cohere_shape() -> None:
    sglang_body = json.dumps(
        [
            {"score": 2.484375, "document": "A cat is an animal.", "index": 2, "meta_info": {}},
            {"score": 0.91796875, "document": "The Yeti is a legendary creature.", "index": 1, "meta_info": {}},
        ]
    ).encode()

    async def content_gen():  # type: ignore[return]
        yield sglang_body

    mock_http_response = MagicMock()
    mock_http_response.response.content_type = "application/json"
    mock_http_response.response.status = 200
    mock_http_response.response.headers = {}
    mock_http_response.content = content_gen()

    opts = ProxyOptions(url="http://example.com/v1/rerank")

    with patch("server.endpointregistry.make_http_request", new_callable=AsyncMock, return_value=mock_http_response):
        result = await post_json_rerank_sglang(RerankRequest(model="m", query="q", documents=[]), opts)

    body = await _collect(result.body_iterator)
    assert json.loads(body) == {
        "results": [
            {"index": 2, "relevance_score": 2.484375, "document": {"text": "A cat is an animal."}},
            {"index": 1, "relevance_score": 0.91796875, "document": {"text": "The Yeti is a legendary creature."}},
        ]
    }


@pytest.mark.asyncio
async def test_post_json_rerank_sglang_passes_through_non_json_response() -> None:
    mock_http_response = MagicMock()
    mock_streaming = MagicMock(spec=StreamingResponse)
    mock_http_response.response.content_type = "text/plain"
    mock_http_response.as_streaming_response.return_value = mock_streaming

    opts = ProxyOptions(url="http://example.com/v1/rerank")

    with patch("server.endpointregistry.make_http_request", new_callable=AsyncMock, return_value=mock_http_response):
        result = await post_json_rerank_sglang(RerankRequest(model="m", query="q", documents=[]), opts)

    assert result is mock_streaming


@pytest.mark.asyncio
async def test_post_json_rerank_sglang_removes_model_when_requested() -> None:
    mock_http_response = MagicMock()
    mock_http_response.response.content_type = "text/plain"
    mock_http_response.as_streaming_response.return_value = MagicMock()
    opts = ProxyOptions(url="http://example.com/v1/rerank", remove_model=True)

    with patch("server.endpointregistry.make_http_request", new_callable=AsyncMock, return_value=mock_http_response) as mock_req:
        await post_json_rerank_sglang(RerankRequest(model="m", query="q", documents=[]), opts)

    called_data = mock_req.call_args.kwargs["data"]
    assert "model" not in json.loads(called_data._value)


@pytest.mark.asyncio
async def test_post_json_rerank_sglang_rewrites_model_when_requested() -> None:
    mock_http_response = MagicMock()
    mock_http_response.response.content_type = "text/plain"
    mock_http_response.as_streaming_response.return_value = MagicMock()
    opts = ProxyOptions(url="http://example.com/v1/rerank", rewrite_model_to="served-name")

    with patch("server.endpointregistry.make_http_request", new_callable=AsyncMock, return_value=mock_http_response) as mock_req:
        await post_json_rerank_sglang(RerankRequest(model="m", query="q", documents=[]), opts)

    called_data = mock_req.call_args.kwargs["data"]
    assert json.loads(called_data._value)["model"] == "served-name"


@pytest.mark.asyncio
async def test_post_json_rerank_sglang_logs_and_passes_through_on_malformed_body(caplog: pytest.LogCaptureFixture) -> None:
    async def content_gen():  # type: ignore[return]
        yield b"not json"

    mock_http_response = MagicMock()
    mock_http_response.response.content_type = "application/json"
    mock_http_response.response.status = 200
    mock_http_response.response.headers = {}
    mock_http_response.content = content_gen()

    opts = ProxyOptions(url="http://example.com/v1/rerank")

    with (
        caplog.at_level(logging.WARNING, logger="uvicorn"),
        patch("server.endpointregistry.make_http_request", new_callable=AsyncMock, return_value=mock_http_response),
    ):
        result = await post_json_rerank_sglang(RerankRequest(model="m", query="q", documents=[]), opts)

    assert await _collect(result.body_iterator) == b"not json"
    assert "Failed to normalize SGLang rerank response" in caplog.text


@pytest.mark.asyncio
async def test_execute_responses_resolves_item_reference_from_store() -> None:
    reg = make_registry()
    reg.response_item_store.put({"id": "rs_871851", "type": "reasoning", "summary": []})
    ep = make_chat_endpoint(on_responses=True)
    ep.on_responses = AsyncMock(return_value=MagicMock())
    reg.register_chat_completion("gemma4:e4b", make_props(), ep, RegistrationOptions(origin="local"))

    body = ResponsesRequest(model="gemma4:e4b", input=[ItemReference(id="rs_871851")])

    await reg.execute_responses(body)

    forwarded_input = ep.on_responses.call_args.args[0].input  # pyright: ignore[reportOptionalMemberAccess]
    assert len(forwarded_input) == 1
    assert isinstance(forwarded_input[0], Reasoning)
    assert forwarded_input[0].id == "rs_871851"


@pytest.mark.asyncio
async def test_execute_responses_leaves_unresolved_item_reference_untouched() -> None:
    reg = make_registry()
    ep = make_chat_endpoint(on_responses=True)
    ep.on_responses = AsyncMock(return_value=MagicMock())
    reg.register_chat_completion("gpt-5.4-nano", make_props(), ep, RegistrationOptions(origin="remote"))

    body = ResponsesRequest(model="gpt-5.4-nano", input=[ItemReference(id="rs_unknown")])

    await reg.execute_responses(body)

    forwarded_input = ep.on_responses.call_args.args[0].input  # pyright: ignore[reportOptionalMemberAccess]
    assert len(forwarded_input) == 1
    assert isinstance(forwarded_input[0], ItemReference)
    assert forwarded_input[0].id == "rs_unknown"


def test_resolve_item_reference_leaves_non_reference_items_unchanged() -> None:
    reg = make_registry()
    item = Reasoning(summary=[])

    result = reg._resolve_item_reference(item)  # pyright: ignore[reportPrivateUsage]

    assert result is item


def test_resolve_item_reference_falls_back_when_cached_item_fails_validation() -> None:
    reg = make_registry()
    reg.response_item_store.put({"id": "rs_bad", "type": "reasoning"})  # missing required "summary" field

    reference = ItemReference(id="rs_bad")
    result = reg._resolve_item_reference(reference)  # pyright: ignore[reportPrivateUsage]

    assert result is reference


@pytest.mark.asyncio
async def test_post_json_responses_removes_model_when_requested() -> None:
    mock_http_response = MagicMock()
    mock_http_response.response.content_type = "application/json"
    mock_http_response.response.status = 200
    mock_http_response.response.headers = {}

    async def content_gen():  # type: ignore[return]
        yield json.dumps({"output": []}).encode()

    mock_http_response.content = content_gen()
    opts = ProxyOptions(url="http://example.com/v1/responses", remove_model=True)
    store = ResponseItemStore()

    with patch("server.endpointregistry.make_http_request", new_callable=AsyncMock, return_value=mock_http_response) as mock_req:
        await post_json_responses(ResponsesRequest(model="gemma4:e4b", stream=False), opts, store)

    called_data = mock_req.call_args.kwargs["data"]
    assert "model" not in json.loads(called_data._value)


@pytest.mark.asyncio
async def test_post_json_responses_rewrites_model_when_requested() -> None:
    mock_http_response = MagicMock()
    mock_http_response.response.content_type = "application/json"
    mock_http_response.response.status = 200
    mock_http_response.response.headers = {}

    async def content_gen():  # type: ignore[return]
        yield json.dumps({"output": []}).encode()

    mock_http_response.content = content_gen()
    opts = ProxyOptions(url="http://example.com/v1/responses", rewrite_model_to="gemma4:internal")
    store = ResponseItemStore()

    with patch("server.endpointregistry.make_http_request", new_callable=AsyncMock, return_value=mock_http_response) as mock_req:
        await post_json_responses(ResponsesRequest(model="gemma4:e4b", stream=False), opts, store)

    called_data = mock_req.call_args.kwargs["data"]
    assert json.loads(called_data._value)["model"] == "gemma4:internal"


@pytest.mark.asyncio
async def test_post_json_responses_captures_only_dict_items_from_output_list() -> None:
    response_body = json.dumps(
        {
            "id": "resp_1",
            "output": [
                "not-a-dict-item",
                {"id": "rs_871851", "type": "reasoning", "summary": []},
                {"id": "fc_1", "type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"},
            ],
        }
    ).encode()

    async def content_gen():  # type: ignore[return]
        yield response_body

    mock_http_response = MagicMock()
    mock_http_response.response.content_type = "application/json"
    mock_http_response.response.status = 200
    mock_http_response.response.headers = {}
    mock_http_response.content = content_gen()

    store = ResponseItemStore()
    opts = ProxyOptions(url="http://example.com/v1/responses")

    with patch("server.endpointregistry.make_http_request", new_callable=AsyncMock, return_value=mock_http_response):
        await post_json_responses(ResponsesRequest(model="gemma4:e4b", stream=False), opts, store)

    assert store.get("rs_871851") is not None
    assert store.get("fc_1") is not None


@pytest.mark.asyncio
async def test_register_mcp_endpoint_as_proxy_forwards_full_path_when_requested():
    reg = make_registry()
    opts = ProxyOptions(url="http://child.example.com/mcp", forward_full_path=True)
    reg.register_mcp_endpoint_as_proxy("duck", make_props(), opts, None)
    ep = reg.mcp_endpoints.get_model("duck")
    request = MagicMock()
    request.headers = {}
    request.method = "POST"
    request.path_params = {"full_path": "duck/mcp"}
    request.stream = lambda: _async_byte_stream(b"")
    request.url.query = ""

    with patch("server.endpointregistry.make_http_request", new_callable=AsyncMock, return_value=MagicMock()) as mock_req:
        await ep.endpoint.on_request(request)  # pyright: ignore[reportOptionalMemberAccess]

    assert mock_req.call_args.kwargs["url"] == "http://child.example.com/mcp/duck/mcp"


def test_register_proxy_mcp_forwards_protocol_headers_and_full_path():
    reg = make_registry()

    with patch.object(reg, "register_mcp_endpoint_as_proxy") as mock_register:
        reg._register_proxy("duck", "mcp", make_props(), "http://child.example.com/", "key", None)  # pyright: ignore[reportPrivateUsage]

    options = mock_register.call_args.kwargs["options"]
    assert options.url == "http://child.example.com/mcp"
    assert options.forward_full_path is True
    assert options.allowed_request_headers == ["accept", "mcp-session-id"]
    assert options.allowed_response_headers == ["accept", "mcp-session-id", "www-authenticate"]


def test_register_proxy_mcp_uses_sse_proxy_for_sse_transport():
    reg = make_registry()
    props = make_props(type="mcp")
    props.transport = "sse"

    with (
        patch.object(reg, "register_mcp_sse_endpoint_as_proxy") as mock_sse,
        patch.object(reg, "register_mcp_endpoint_as_proxy") as mock_http,
    ):
        reg._register_proxy("duck", "mcp", props, "http://child.example.com/", "key", None)  # pyright: ignore[reportPrivateUsage]

    assert mock_http.call_count == 0
    assert mock_sse.call_args.kwargs["options"].url == "http://child.example.com/mcp/duck/sse"


def test_list_models_reports_registration_health():
    reg = make_registry()
    reg.register_mcp_endpoint(
        "duck", make_props(type="mcp"), McpEndpoint(on_request=AsyncMock()), RegistrationOptions(origin="local", healthy=True)
    )

    assert [m.healthy for m in reg.list_models()] == [True]


def test_update_models_registers_mcp_as_healthy_from_peer():
    reg = make_registry()
    model = Model(id="mcp-id", name="duck", type="mcp", props=make_props(type="mcp"), usage=0, healthy=True)

    reg.update_models([], [model], "http://child.example.com/", "key")

    assert [m.id for m in reg.mcp_endpoints.get_healthy_models()] == ["duck"]


def test_update_models_applies_health_and_props_change_in_place():
    reg = make_registry()
    old = Model(id="mcp-id", name="duck", type="mcp", props=make_props(type="mcp"), usage=0)
    reg.update_models([], [old], "http://child.example.com/", "key")
    reg.parent_infra.send_models_list.reset_mock()  # pyright: ignore[reportAttributeAccessIssue]
    new = old.model_copy(update={"healthy": True, "props": make_props(type="mcp", endpoints=["/mcp/duck/mcp"])})

    with patch.object(reg, "_register_proxy") as mock_register:
        reg.update_models([old], [new], "http://child.example.com/", "key")

    assert mock_register.call_count == 0
    assert [m.id for m in reg.mcp_endpoints.get_healthy_models()] == ["duck"]
    assert reg.registry["mcp-id"].registered_model.props.endpoints == ["/mcp/duck/mcp"]
    assert reg.parent_infra.send_models_list.call_count == 1  # pyright: ignore[reportAttributeAccessIssue]


def test_update_models_reregisters_mcp_proxy_when_transport_changes():
    reg = make_registry()
    old = Model(id="mcp-id", name="duck", type="mcp", props=make_props(type="mcp"), usage=0, healthy=True)
    reg.update_models([], [old], "http://child.example.com/", "key")
    sse_props = make_props(type="mcp")
    sse_props.transport = "sse"
    new = old.model_copy(update={"props": sse_props})

    with patch.object(reg, "_register_proxy") as mock_register:
        reg.update_models([old], [new], "http://child.example.com/", "key")

    assert mock_register.call_count == 1
    assert mock_register.call_args.kwargs["props"].transport == "sse"


@pytest.mark.asyncio
async def test_register_mcp_endpoint_as_proxy_rejects_parent_dir_segments_in_full_path():
    reg = make_registry()
    opts = ProxyOptions(url="http://child.example.com/mcp", forward_full_path=True)
    reg.register_mcp_endpoint_as_proxy("duck", make_props(), opts, None)
    ep = reg.mcp_endpoints.get_model("duck")
    request = MagicMock()
    request.headers = {}
    request.method = "POST"
    request.path_params = {"full_path": "duck/../../admin"}
    request.url.query = ""

    with (
        patch("server.endpointregistry.make_http_request", new_callable=AsyncMock) as mock_req,
        pytest.raises(HTTPException) as exc,
    ):
        await ep.endpoint.on_request(request)  # pyright: ignore[reportOptionalMemberAccess]

    assert exc.value.status_code == 400
    assert mock_req.call_count == 0


def test_register_proxy_mcp_sse_url_uses_model_name_prefix():
    reg = make_registry()
    props = make_props(type="mcp")
    props.transport = "sse"

    with patch.object(reg, "register_mcp_sse_endpoint_as_proxy") as mock_sse:
        reg._register_proxy("my-prefix", "mcp", props, "http://child.example.com", "key", None)  # pyright: ignore[reportPrivateUsage]

    assert mock_sse.call_args.kwargs["options"].url == "http://child.example.com/mcp/my-prefix/sse"
    assert mock_sse.call_args.kwargs["url"] == "my-prefix"


async def _transcribe(
    response_format: str | None, response: StreamingResponse, origin: str = "remote", stream: bool = False
) -> tuple[EndpointRegistry, Any]:
    reg = make_registry()
    reg.register_audio_transcriptions(
        "stt",
        make_props(),
        SimpleEndpoint(on_request=AsyncMock(return_value=response)),
        RegistrationOptions(origin=origin),
    )
    body = MagicMock(spec=CreateTranscriptionRequest)
    body.model = "stt"
    body.response_format = response_format
    body.stream = stream
    return reg, await reg.execute_audio_transcriptions(body)


async def _body_of(response: Any) -> bytes:
    if isinstance(response, StreamingResponse):
        return b"".join([cast("bytes", chunk) async for chunk in response.body_iterator])
    return response.body


@pytest.mark.parametrize("origin", ["remote", "local"])
@pytest.mark.asyncio
async def test_execute_audio_transcriptions_text_unwraps_json_reply(origin: str):
    backend = StreamingResponse(
        _async_byte_stream(b'{"text": "hello world", "usage": {"seconds": 3}}'), media_type="application/json", status_code=200
    )

    reg, result = await _transcribe("text", backend, origin)

    assert result.media_type == "text/plain"
    assert await _body_of(result) == b"hello world"
    assert all(model.usage == 0 for model in reg.audio_transcriptions_endpoints.models["stt"].values())


@pytest.mark.parametrize("origin", ["remote", "local"])
@pytest.mark.asyncio
async def test_execute_audio_transcriptions_text_passes_plain_text_reply_through(origin: str):
    backend = StreamingResponse(_async_byte_stream(b"hello world"), media_type="text/plain", status_code=200)

    _, result = await _transcribe("text", backend, origin)

    assert result.media_type == "text/plain"
    assert await _body_of(result) == b"hello world"


@pytest.mark.parametrize("origin", ["remote", "local"])
@pytest.mark.asyncio
async def test_execute_audio_transcriptions_text_passes_error_through(origin: str):
    payload = b'{"text": "nope", "error": "boom"}'
    backend = StreamingResponse(_async_byte_stream(payload), media_type="application/json", status_code=500)

    reg, result = await _transcribe("text", backend, origin)

    assert result.status_code == 500
    assert await _body_of(result) == payload
    assert all(model.usage == 0 for model in reg.audio_transcriptions_endpoints.models["stt"].values())


@pytest.mark.parametrize("origin", ["remote", "local"])
@pytest.mark.parametrize("response_format", ["json", "verbose_json", "srt", "vtt", None])
@pytest.mark.asyncio
async def test_execute_audio_transcriptions_other_formats_pass_through(response_format: str | None, origin: str):
    backend = StreamingResponse(_async_byte_stream(b'{"text": "hello"}'), media_type="application/json", status_code=200)

    _, result = await _transcribe(response_format, backend, origin)

    assert result.status_code == 200
    assert result.media_type == "application/json"
    assert await _body_of(result) == b'{"text": "hello"}'


@pytest.mark.parametrize("origin", ["remote", "local"])
@pytest.mark.asyncio
async def test_execute_audio_transcriptions_text_with_stream_passes_through(origin: str):
    payload = b'data: {"text": "hello"}\n\n'
    backend = StreamingResponse(_async_byte_stream(payload), media_type="text/event-stream", status_code=200)

    _, result = await _transcribe("text", backend, origin, stream=True)

    assert result.media_type == "text/event-stream"
    assert await _body_of(result) == payload


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", [b"not json at all", b'{"other": 1}', b'{"text": 5}', b'["text"]'])
async def test_unwrap_json_text_replays_json_reply_without_text_field(raw: bytes) -> None:
    response = StreamingResponse(_async_byte_stream(raw), media_type="application/json", headers={"x-test": "1"})
    response.headers["content-length"] = str(len(raw))

    result = await _unwrap_json_text(response)

    assert isinstance(result, StreamingResponse)
    assert result.media_type == "application/json"
    assert result.status_code == 200
    assert result.headers["x-test"] == "1"
    assert await _collect(result.body_iterator) == raw


def _systemone_body(model: str = "tev1") -> SystemOneRequest:
    return SystemOneRequest.model_validate(
        {
            "model": model,
            "state": "Our checkout has returned 500 errors since 9am.",
            "questions": {"label": {"type": "choice", "instructions": "Which label?", "criteria": {"bug": None, "billing": "Payments"}}},
        }
    )


def test_register_systemone_registers_model_listed_as_decision():
    reg = make_registry()

    reg.register_systemone("tev1", make_props(type="decision", endpoints=["/v1/systemone"]), SimpleEndpoint(on_request=AsyncMock()), None)

    assert reg.systemone_endpoints.has_model("tev1")
    model = reg.get_model("tev1")
    assert model.props.type == "decision"
    assert model.props.endpoints == ["/v1/systemone"]
    assert [m.type for m in reg.list_models() if m.name == "tev1"] == ["decision"]
    assert "tev1" in [m.id for m in reg.get_models().data]
    assert reg.model_exists("tev1")


def test_list_own_models_includes_local_systemone_model():
    reg = make_registry()

    reg.register_systemone("tev1", make_props(type="decision"), SimpleEndpoint(on_request=AsyncMock()), RegistrationOptions(origin="local"))

    assert [m.name for m in reg.list_own_models()] == ["tev1"]


def test_register_systemone_as_proxy_registers_model():
    reg = make_registry()

    reg.register_systemone_as_proxy("tev1", make_props(type="decision"), ProxyOptions(url="http://example.com/v1/systemone"), None)

    assert reg.systemone_endpoints.has_model("tev1")


def test_unregister_systemone_removes_model():
    reg = make_registry()
    rid = reg.register_systemone("tev1", make_props(type="decision"), SimpleEndpoint(on_request=AsyncMock()), None)

    reg.unregister_systemone("tev1", rid)

    assert not reg.systemone_endpoints.has_model("tev1")
    assert "tev1" not in [m.id for m in reg.get_models().data]


@pytest.mark.asyncio
@patch("server.endpointregistry.make_http_request", new_callable=AsyncMock)
async def test_register_systemone_as_proxy_forwards_rewritten_request_and_returns_backend_response(mock_make_http_request: AsyncMock):
    reg = make_registry()
    reg.register_systemone_as_proxy(
        "tev1", make_props(type="decision"), ProxyOptions(url="http://ollama:11434/v1/systemone", rewrite_model_to="tev1:0.8b"), None
    )
    ep = reg.systemone_endpoints.get_model("tev1")
    backend_error = StreamingResponse(_async_byte_stream(b'{"error":"bad"}'), status_code=400)
    _mock_http_response = MagicMock()
    _mock_http_response.as_streaming_response.return_value = backend_error
    mock_make_http_request.return_value = _mock_http_response

    result = await ep.endpoint.on_request(_systemone_body(), None)  # pyright: ignore[reportOptionalMemberAccess]

    assert result is backend_error
    assert mock_make_http_request.call_count == 1
    assert mock_make_http_request.call_args.kwargs["url"] == "http://ollama:11434/v1/systemone"
    sent = json.loads(mock_make_http_request.call_args.kwargs["data"]._value)
    assert sent["model"] == "tev1:0.8b"
    assert sent["questions"]["label"]["criteria"] == {"bug": None, "billing": "Payments"}


@pytest.mark.asyncio
@patch("server.endpointregistry.make_http_request", new_callable=AsyncMock)
async def test_register_systemone_as_proxy_keeps_null_state_in_sent_payload(mock_make_http_request: AsyncMock):
    reg = make_registry()
    reg.register_systemone_as_proxy("tev1", make_props(type="decision"), ProxyOptions(url="http://ollama:11434/v1/systemone"), None)
    ep = reg.systemone_endpoints.get_model("tev1")
    mock_make_http_request.return_value = MagicMock()
    body = SystemOneRequest.model_validate({"model": "tev1", "state": None, "questions": {"urgent": {"type": "noul"}}})

    await ep.endpoint.on_request(body, None)  # pyright: ignore[reportOptionalMemberAccess]

    sent = json.loads(mock_make_http_request.call_args.kwargs["data"]._value)
    assert "state" in sent
    assert sent["state"] is None
    assert sent["questions"] == {"urgent": {"type": "noul"}}


@pytest.mark.asyncio
@patch.object(EndpointRegistry, "with_usage", new_callable=AsyncMock)
async def test_execute_systemone_wraps_local_model_with_usage(mock_with_usage: AsyncMock):
    reg = make_registry()
    _mock_response = MagicMock()
    mock_with_usage.return_value = _mock_response
    reg.register_systemone(
        "tev1",
        make_props(type="decision"),
        SimpleEndpoint(on_request=AsyncMock(return_value=_mock_response)),
        RegistrationOptions(origin="local"),
    )

    result = await reg.execute_systemone(_systemone_body())

    assert result is _mock_response
    assert mock_with_usage.call_count == 1


@pytest.mark.asyncio
async def test_execute_systemone_calls_on_request_directly_for_remote_model():
    reg = make_registry()
    _mock_response = MagicMock()
    _mock_on_request = AsyncMock(return_value=_mock_response)
    reg.register_systemone(
        "tev1", make_props(type="decision"), SimpleEndpoint(on_request=_mock_on_request), RegistrationOptions(origin="remote")
    )
    body = _systemone_body()

    result = await reg.execute_systemone(body)

    assert result is _mock_response
    assert _mock_on_request.call_count == 1
    assert _mock_on_request.call_args == call(body, None)


@pytest.mark.asyncio
async def test_execute_systemone_logs_payload_when_enabled(caplog: pytest.LogCaptureFixture):
    reg = make_registry()
    reg.config.is_log_payloads_enabled.return_value = True  # pyright: ignore[reportAttributeAccessIssue]
    reg.register_systemone(
        "tev1", make_props(type="decision"), SimpleEndpoint(on_request=AsyncMock()), RegistrationOptions(origin="remote")
    )

    with caplog.at_level(logging.INFO):
        await reg.execute_systemone(_systemone_body())

    assert "DUMP REQUEST PAYLOAD /v1/systemone" in caplog.text


@pytest.mark.asyncio
async def test_execute_systemone_raises_404_when_not_registered():
    reg = make_registry()

    with pytest.raises(HTTPException) as exc_info:
        await reg.execute_systemone(_systemone_body("unknown"))

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Model not found"


@pytest.mark.asyncio
async def test_execute_systemone_raises_400_when_model_is_not_a_decision_model():
    reg = make_registry()
    reg.register_chat_completion("llama", make_props(), make_chat_endpoint(), RegistrationOptions(origin="local"))

    with pytest.raises(HTTPException) as exc_info:
        await reg.execute_systemone(_systemone_body("llama"))

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Given model is not supported"


@pytest.mark.asyncio
async def test_execute_chat_completion_rejects_decision_model_without_forwarding():
    reg = make_registry()
    _mock_on_request = AsyncMock()
    reg.register_systemone(
        "tev1", make_props(type="decision"), SimpleEndpoint(on_request=_mock_on_request), RegistrationOptions(origin="local")
    )
    body = MagicMock(spec=ChatCompletionRequest)
    body.model = "tev1"
    body.model_dump_json.return_value = "{}"

    with pytest.raises(HTTPException) as exc_info:
        await reg.execute_chat_completion(body)

    assert exc_info.value.status_code == 400
    assert _mock_on_request.call_count == 0


@pytest.mark.asyncio
@patch("server.endpointregistry.make_http_request", new_callable=AsyncMock)
async def test_register_proxy_decision_proxies_to_child_systemone(mock_make_http_request: AsyncMock):
    reg = make_registry()
    reg._register_proxy(  # pyright: ignore[reportPrivateUsage]
        "tev1", "decision", make_props(type="decision"), "http://child:8086/", "child-key", RegistrationOptions(origin="http://child:8086/")
    )
    ep = reg.systemone_endpoints.get_model("tev1")
    _mock_http_response = MagicMock()
    _mock_http_response.as_streaming_response.return_value = MagicMock()
    mock_make_http_request.return_value = _mock_http_response

    await ep.endpoint.on_request(_systemone_body(), None)  # pyright: ignore[reportOptionalMemberAccess]

    assert reg.systemone_endpoints.has_model("tev1")
    assert mock_make_http_request.call_count == 1
    assert mock_make_http_request.call_args.kwargs["url"] == "http://child:8086/v1/systemone"
    assert mock_make_http_request.call_args.kwargs["headers"]["Authorization"] == "Bearer child-key"
