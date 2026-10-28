# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Endpoint registry holds callbacks for given endpoints and models."""

import asyncio
import json
import random
import time
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, TypeVar, cast
from urllib.parse import parse_qs, urljoin, urlparse

import aiohttp
from aiohttp import ClientTimeout, JsonPayload
from fastapi import HTTPException, Request
from fastapi.responses import PlainTextResponse, Response, StreamingResponse
from pydantic import BaseModel, TypeAdapter, ValidationError, field_validator
from starlette.exceptions import HTTPException as StarletteHTTPException

from server.config import AppSettings
from server.metrics_registry import MetricsRegistry
from server.models.api import (
    ALL_LLM_SUFFIXES,
    CHAT_COMPLETION_EXTRA_FIELDS,
    COMPLETION_LEGACY_EXTRA_FIELDS,
    LLM_SUFFIX_MAP,
    ApiModel,
    ApiModelProps,
    ApiModels,
    ChatCompletionRequest,
    CompletionLegacyRequest,
    CreateSpeechRequest,
    CreateTranscriptionRequest,
    EmbeddingRequest,
    FormSerializable,
    ImagesRequest,
    Input,
    ItemReference,
    McpToolInfo,
    MessagesRequest,
    Model,
    ModelId,
    ModelProps,
    ModelType,
    OllamaChatRequest,
    RerankRequest,
    ResponsesRequest,
    SystemOneRequest,
)
from server.models.common import JsonSerializable, StarletteResponse
from server.utils.core import HttpClientError, HttpResponse, Utils, make_http_request
from server.utils.sentry import UpstreamError, attach_tags, capture_exception, mark_reported
from server.websockets.models import RegistrationId, UsageChangeRequest, WarmChangeRequest
from server.websockets.parent_infra_group import ParentInfraGroup

if TYPE_CHECKING:
    from server.model_tester import ModelTester

from server.utils.logger import logger

T = TypeVar("T")

type EndpointCallback[T] = Callable[[T, Request | None], Awaitable[StarletteResponse]]
type CustomEndpointCallback = Callable[[Request], Awaitable[StarletteResponse]]
type McpEndpointCallback = Callable[[Request], Awaitable[StarletteResponse]]

type CapacityState = int | Literal["unbounded", "unknown"]
"""A backend's routing concurrency limit: a positive int (known, finite), "unbounded" (no
concurrency concept, e.g. a cloud/proxy backend), or "unknown" (we tried and failed to
determine it, e.g. a vLLM log-parse miss). Kept internal-only: the wire/mesh schema
(`server/models/api.py::Model`) still serializes this as a `capacity`/`capacity_known` pair for
rolling-upgrade compatibility across mesh peers on different versions; `_capacity_state_from_pair`
and `_capacity_pair_from_state` convert at that boundary."""

_ALL_SATURATED_WARNING_INTERVAL_SECONDS = 60.0  # avoid flooding logs while a model stays saturated under peak load


def _capacity_state_from_pair(capacity: int | None, capacity_known: bool) -> CapacityState:
    if capacity is not None:
        return capacity
    return "unbounded" if capacity_known else "unknown"


def _capacity_pair_from_state(state: CapacityState) -> tuple[int | None, bool]:
    if isinstance(state, int):
        return state, True
    return None, state == "unbounded"


_MCP_PROXY_MAX_BODY_BYTES = 10 * 1024 * 1024  # generous for MCP JSON-RPC payloads; bounds the buffering needed for a 401-retry replay


class SimpleEndpoint[T](NamedTuple):
    on_request: EndpointCallback[T]


class CustomEndpoint(NamedTuple):
    on_request: CustomEndpointCallback


class McpEndpoint(NamedTuple):
    on_request: McpEndpointCallback


class ChatCompletionEndpoint:
    on_chat_completion: EndpointCallback[ChatCompletionRequest] | None = None
    on_completion: EndpointCallback[CompletionLegacyRequest] | None = None
    on_responses: EndpointCallback[ResponsesRequest] | None = None
    on_messages: EndpointCallback[MessagesRequest] | None = None
    on_ollama_chat: EndpointCallback[OllamaChatRequest] | None = None


class RegisteredModel[T](BaseModel):
    id: RegistrationId
    name: ModelId
    origin: str
    props: ModelProps
    type: str
    endpoint: T
    usage: int
    created: int
    owned_by: str
    healthy: bool = False
    capacity_state: CapacityState = "unbounded"
    warm: bool = True
    alias: ModelId | None = None

    @field_validator("capacity_state")
    @classmethod
    def _validate_capacity_state(cls, v: CapacityState) -> CapacityState:
        if isinstance(v, int) and v <= 0:
            raise ValueError("capacity_state must be a positive integer, 'unbounded', or 'unknown' (zero/negative is not a valid cap)")
        return v


class RegistrationOptions(NamedTuple):
    """Options for `Endpoint.add_model`.

    `capacity_state="unbounded"` (the default) means the registration is genuinely unbounded (no
    concurrency concept, e.g. a cloud/proxy backend). A service that tries and fails to determine
    a real capacity (e.g. a vLLM log-parse miss) must explicitly pass `capacity_state="unknown"`
    to rank in the worst routing tier instead of being treated as unbounded (see `_rank_key`).
    """

    origin: str
    id: RegistrationId | None = None
    usage: int | None = None
    send_notification: bool = True
    owned_by: str = "local"
    capacity_state: CapacityState = "unbounded"
    warm: bool = True
    alias: ModelId | None = None
    healthy: bool = False


class RegistryEntry(NamedTuple):
    model_id: ModelId
    endpoint: "Endpoint[Any]"
    registered_model: RegisteredModel[Any]


def _rank_key(x: RegisteredModel[Any]) -> tuple[int, float]:
    """Sort key ranking warm before cold, then higher capacity before lower.

    `capacity_state` distinguishes "no explicit cap" for backends with no concurrency concept at
    all (cloud/proxy, genuinely `"unbounded"`) from backends whose capacity we *tried* and failed
    to determine (e.g. a vLLM log-parse miss, `"unknown"`). These rank differently: a genuinely
    unbounded backend (`0.0`) ranks below every known finite capacity but above the
    unknown/failed tier (`+inf`), so it acts as an overflow valve for finite backends without
    ever being preferred over a real parse failure.
    """
    match x.capacity_state:
        case int() as capacity:
            capacity_key = -capacity
        case "unbounded":
            capacity_key = 0.0
        case "unknown":
            capacity_key = float("inf")
        case _:
            msg = f"Unknown capacity_state: {x.capacity_state!r}"
            raise ValueError(msg)
    return (0 if x.warm else 1, capacity_key)


def _saturation_threshold(x: RegisteredModel[Any]) -> float | None:
    """Usage level at which `x` is considered saturated, or None if it has no explicit cap."""
    if not isinstance(x.capacity_state, int):
        return None
    return x.capacity_state if x.origin == "local" else x.capacity_state * 0.9


def _group_by_rank(candidates: list[RegisteredModel[Any]]) -> list[list[RegisteredModel[Any]]]:
    """Group ranked candidates into consecutive equal-rank tiers, preserving rank order."""
    groups: list[list[RegisteredModel[Any]]] = []
    for x in sorted(candidates, key=_rank_key):
        if groups and _rank_key(groups[-1][0]) == _rank_key(x):
            groups[-1].append(x)
        else:
            groups.append([x])
    return groups


def _pick_ranked_model[T](
    model_id: ModelId, candidates: list[RegisteredModel[T]], all_saturated_warning_last_logged: dict[ModelId, float]
) -> RegisteredModel[T]:
    """Pick a candidate: pack onto the top-ranked tier until saturated, then spill over to the next tier.

    Packing within a tier is deterministic (by registration id), not randomized - shuffling
    would spread requests evenly instead of packing, defeating warmth-aware routing. Only the
    all-saturated fallback below, which no longer benefits from packing, uses a random
    tie-break among equally-loaded candidates instead of a deterministic one.
    """
    groups = _group_by_rank(candidates)
    for group in groups:
        if _saturation_threshold(group[0]) is None:
            # No candidate in this tier has an explicit cap (all share the same rank key,
            # so either all or none do) - there's no saturation threshold to pack against,
            # so fall back to least-loaded instead of an arbitrary pick.
            return min(group, key=lambda x: x.usage)
        # Pack onto the same candidate (by id, for a stable order across calls) until it
        # saturates, then spill over to the next one.
        for x in sorted(group, key=lambda c: c.id):
            # Every member of a tier shares the same rank key, so either all of them have a
            # saturation threshold or none do; the `is None` branch above already handled the
            # latter case, so `threshold` here is always a real number.
            threshold = _saturation_threshold(x)
            if threshold is not None and x.usage < threshold:
                return x
    shuffled = list(candidates)
    random.shuffle(shuffled)
    least_loaded = min(shuffled, key=lambda x: x.usage)
    last_logged = all_saturated_warning_last_logged.get(model_id, 0.0)
    now = time.time()
    if now - last_logged >= _ALL_SATURATED_WARNING_INTERVAL_SECONDS:
        all_saturated_warning_last_logged[model_id] = now
        logger.warning(
            "All %d candidate(s) for model %r are saturated; routing to least-loaded anyway (id=%s usage=%s capacity_state=%s)",
            len(candidates),
            least_loaded.name,
            least_loaded.id,
            least_loaded.usage,
            least_loaded.capacity_state,
        )
    return least_loaded


class Endpoint[T]:
    def __init__(
        self,
        registry: dict[RegistrationId, RegistryEntry],
        parent_infra: ParentInfraGroup,
    ):
        self.registry = registry
        self.parent_infra = parent_infra
        self.models = dict[ModelId, dict[RegistrationId, RegisteredModel[T]]]()
        self._all_saturated_warning_last_logged: dict[ModelId, float] = {}

    def add_model(
        self, model_id: ModelId, props: ModelProps, endpoint: T, type: str, options: RegistrationOptions | None
    ) -> RegistrationId:
        """Add model to registry."""
        registered_model = RegisteredModel(
            id=options.id if options and options.id is not None else str(uuid.uuid4()),
            name=model_id,
            origin=options.origin if options else "local",
            props=props,
            endpoint=endpoint,
            type=type,
            usage=options.usage if options and options.usage is not None else 0,
            created=int(time.time()),
            owned_by=options.owned_by if options else "local",
            capacity_state=options.capacity_state if options else "unbounded",
            warm=options.warm if options else True,
            alias=options.alias if options and options.alias and options.alias != model_id else None,
            healthy=options.healthy if options else False,
        )
        if model_id not in self.models:
            self.models[model_id] = {}
        self.models[model_id][registered_model.id] = registered_model
        if registered_model.alias:
            # The alias bucket shares the very same RegisteredModel, so usage, warmth and capacity are counted once.
            self.models.setdefault(registered_model.alias, {})[registered_model.id] = registered_model
        self.registry[registered_model.id] = RegistryEntry(model_id=model_id, endpoint=self, registered_model=registered_model)
        if not options or options.send_notification:
            self.parent_infra.send_models_list()
        return registered_model.id

    def remove_model(self, model_id: ModelId, registration_id: RegistrationId, send_notification: bool = True) -> None:
        """Remove model from registry."""
        if model_id not in self.models:
            return
        removed = self.models[model_id].pop(registration_id, None)
        if len(self.models[model_id]) == 0:
            del self.models[model_id]
        if removed and removed.alias and removed.alias in self.models:
            self.models[removed.alias].pop(registration_id, None)
            if len(self.models[removed.alias]) == 0:
                del self.models[removed.alias]
        if registration_id in self.registry:
            del self.registry[registration_id]
        if send_notification:
            self.parent_infra.send_models_list()

    def has_model(self, model_id: ModelId) -> bool:
        """Have model in registry."""
        return model_id in self.models and len(self.models[model_id]) > 0

    def get_model(
        self,
        model_id: ModelId,
        filter: Callable[[T], bool] | None = None,
        registration_id: RegistrationId | None = None,
    ) -> RegisteredModel[T] | None:
        """Get model from registry.

        Selection among candidates ranks warm registrations before cold ones, then higher
        capacity before lower (see `_rank_key` for how `capacity_state` is ranked). Requests
        pack onto the top-ranked registration (deterministically, by registration id) until it
        crosses its saturation threshold (100% of capacity for locally-registered instances,
        ~90% for mesh-sourced ones, to cover mesh usage propagation lag) and only then spill
        over to the next-ranked registration. If every candidate is saturated, routes to
        whichever has the lowest usage instead of refusing the request, with a random tie-break
        among equally-loaded candidates in that fallback case.
        """
        if model_id not in self.models:
            return None
        candidates = [x for x in self.models[model_id].values() if not filter or filter(x.endpoint)]
        if not candidates:
            return None
        if registration_id:
            return next((x for x in candidates if x.id == registration_id), None)

        chosen = _pick_ranked_model(model_id, candidates, self._all_saturated_warning_last_logged)
        logger.debug(
            "Choosen model origin=%s id=%s name=%s usage=%s capacity_state=%s",
            chosen.origin,
            chosen.id,
            chosen.name,
            chosen.usage,
            chosen.capacity_state,
        )
        return chosen

    def is_model_private(self, model: dict[RegistrationId, RegisteredModel[T]]) -> bool:
        """Return is model private."""
        return any(item.props.private for item in model.values())

    def get_model_type(self, model: dict[RegistrationId, RegisteredModel[T]]) -> str:
        """Return unique model types."""
        return next(iter({item.type for item in model.values()}))

    def get_model_created(self, model: dict[RegistrationId, RegisteredModel[T]]) -> int:
        """Return the earliest registration timestamp for a model."""
        return min(item.created for item in model.values())

    def get_model_owned_by(self, model: dict[RegistrationId, RegisteredModel[T]]) -> str:
        """Return the owning service type for a model."""
        return next(iter(model.values())).owned_by

    def get_model_available_endpoints(self, model: dict[RegistrationId, RegisteredModel[T]]) -> list[str]:
        """Get model available endpoints."""
        return list({endpoint for item in model.values() for endpoint in item.props.endpoints})

    def get_model_context_window(self, model: dict[RegistrationId, RegisteredModel[T]]) -> int | None:
        """Get model context window, or None when no registration declares one."""
        return max((window for item in model.values() if (window := item.props.context_window) is not None), default=None)

    def get_max_context_window(self, model: dict[RegistrationId, RegisteredModel[T]]) -> int | None:
        """Get model max context window, or None when no registration declares one.

        Reported as None rather than 0: consumers use this as a ceiling for their own request sizing,
        and 0 is not a window a model could have. Any registration may leave the window undeclared -
        custom-service models never declare one - and seeding a max() with 0 used to publish 0 for all
        of them.
        """
        return max((window for item in model.values() if (window := item.props.max_context_window) is not None), default=None)

    def get_model_prefix(self, model: dict[RegistrationId, RegisteredModel[T]]) -> str | None:
        """Get model prefix."""
        for item in model.values():
            if item.props.prefix:
                return item.props.prefix
        return None

    def get_model_transport(self, model: dict[RegistrationId, RegisteredModel[T]]) -> str | None:
        """Get model transport."""
        for item in model.values():
            if item.props.transport:
                return item.props.transport
        return None

    def get_model_tools(self, model: dict[RegistrationId, RegisteredModel[T]]) -> list[McpToolInfo]:
        """Get model tools from first registration that has them."""
        for item in model.values():
            if item.props.tools:
                return item.props.tools
        return []

    @staticmethod
    def _aggregate_modality_support(values: list[bool | None]) -> bool | None:
        """Aggregate a modality flag (`supports_vision`/`supports_audio`) across a model's registrations.

        `True` if any registration reports `True` (the model can do it on at least one backend),
        `False` if every registration explicitly reports `False` (confirmed unsupported everywhere),
        otherwise `None` (unknown, or a mix of `None` and `False` with no confirmed `True`).
        """
        if any(v is True for v in values):
            return True
        if values and all(v is False for v in values):
            return False
        return None

    def get_model_supports_vision(self, model: dict[RegistrationId, RegisteredModel[T]]) -> bool | None:
        """Aggregate `supports_vision` across a model's registrations. See `_aggregate_modality_support`."""
        return self._aggregate_modality_support([item.props.supports_vision for item in model.values()])

    def get_model_supports_audio(self, model: dict[RegistrationId, RegisteredModel[T]]) -> bool | None:
        """Aggregate `supports_audio` across a model's registrations. See `_aggregate_modality_support`."""
        return self._aggregate_modality_support([item.props.supports_audio for item in model.values()])

    def get_model_alias_of(self, model_id: ModelId, model: dict[RegistrationId, RegisteredModel[T]]) -> list[str]:
        """Names of the models registered under `model_id` as their alias."""
        return sorted({item.name for item in model.values() if item.alias == model_id})

    @staticmethod
    def _aggregate_capacity(model: dict[RegistrationId, RegisteredModel[T]]) -> int | None:
        """Sum finite capacities across registrations; `None` if any registration is unbounded or unknown."""
        total = 0
        for item in model.values():
            if not isinstance(item.capacity_state, int):
                return None
            total += item.capacity_state
        return total

    def _build_api_model(self, model_id: ModelId, model: dict[RegistrationId, RegisteredModel[T]]) -> ApiModel:
        """Build an ApiModel from a model's registrations."""
        return ApiModel(
            id=model_id,
            object="model",
            created=self.get_model_created(model),
            owned_by=self.get_model_owned_by(model),
            props=ApiModelProps(
                private=self.is_model_private(model),
                type=self.get_model_type(model),
                endpoints=self.get_model_available_endpoints(model),
                context_window=self.get_model_context_window(model),
                max_context_window=self.get_max_context_window(model),
                prefix=self.get_model_prefix(model),
                transport=self.get_model_transport(model),
                tools=self.get_model_tools(model),
                supports_vision=self.get_model_supports_vision(model),
                supports_audio=self.get_model_supports_audio(model),
                alias_of=self.get_model_alias_of(model_id, model),
            ),
            usage=sum(item.usage for item in model.values()),
            capacity=self._aggregate_capacity(model),
        )

    def get_models(self) -> list[ApiModel]:
        """List models from registry."""
        return [self._build_api_model(model_id, model) for model_id, model in self.models.items()]

    def get_healthy_models(self) -> list[ApiModel]:
        """List only models that have at least one healthy registration."""
        return [
            self._build_api_model(model_id, model) for model_id, model in self.models.items() if any(reg.healthy for reg in model.values())
        ]

    def list_models(self, exclude_owned_by: frozenset[str] = frozenset({"mesh-ancestor"})) -> list[Model]:
        """List models from registry, for reporting this node's own aggregate to the rest of the mesh.

        By default excludes only models proxied in from a mesh ancestor (`owned_by="mesh-ancestor"`):
        those are already reported by that ancestor itself further up the `AncestorInfo` chain, so
        echoing them back here would re-offer them to the very peer that sent them (upward) or
        duplicate them alongside their real ancestor entry (downward). Callers that need this node's
        genuinely-own models — e.g. to report downward to children — must also exclude `"mesh"`
        (models proxied up from a child), otherwise a child's own model gets echoed back down to it
        under the same registration id as its local entry, corrupting that entry on removal.
        """
        res = list[Model]()
        for model_id, map in self.models.items():
            for item in map.values():
                if item.owned_by in exclude_owned_by or model_id != item.name:
                    # `model_id != item.name` skips alias buckets: the registration is reported once, under its primary name.
                    continue
                capacity, capacity_known = _capacity_pair_from_state(item.capacity_state)
                res.append(
                    Model(
                        id=item.id,
                        name=model_id,
                        type=cast("ModelType", item.type.split("-")[0]),
                        props=item.props,
                        usage=item.usage,
                        capacity=capacity,
                        capacity_known=capacity_known,
                        warm=item.warm,
                        healthy=item.healthy,
                    )
                )
        return res


class McpSseSessionStore:
    """Maps session IDs to upstream message endpoint URLs for SSE transport.

    All mutations are plain dict operations with no await points inside them,
    so they are atomic from asyncio's single-threaded event loop perspective.
    No asyncio.Lock is needed here — that would only be required with threads or
    if an await were introduced inside a mutation.
    """

    def __init__(self, ttl_seconds: int = 300, max_sessions: int = 128) -> None:
        self._sessions: dict[str, tuple[str, float]] = {}
        self._ttl = ttl_seconds
        self._max = max_sessions

    def _evict_expired(self) -> None:
        """Remove all sessions whose TTL has elapsed."""
        now = time.monotonic()
        expired = [sid for sid, (_, ts) in self._sessions.items() if now - ts >= self._ttl]
        for sid in expired:
            del self._sessions[sid]

    def add(self, session_id: str, upstream_url: str) -> None:
        """Store session → upstream messages URL mapping, evicting expired/oldest if at capacity."""
        self._evict_expired()
        if len(self._sessions) >= self._max:
            # evict oldest (first inserted) to make room
            oldest = next(iter(self._sessions))
            del self._sessions[oldest]
        self._sessions[session_id] = (upstream_url, time.monotonic())

    def get(self, session_id: str) -> str | None:
        """Return upstream messages URL for a session, or None if unknown or expired."""
        entry = self._sessions.get(session_id)
        if entry is None:
            return None
        url, ts = entry
        if time.monotonic() - ts >= self._ttl:
            del self._sessions[session_id]
            return None
        return url

    def remove(self, session_id: str) -> None:
        """Remove a session mapping."""
        self._sessions.pop(session_id, None)


_input_item_adapter = TypeAdapter(Input)


class ResponseItemStore:
    """Caches stateful output items (e.g. reasoning) from this gateway's own /v1/responses replies.

    Backends like Ollama mint an id on stateful output items but have no real storage to resolve an
    `item_reference` to that id against, since this gateway does not implement `store`/`previous_response_id`.
    This lets a later request's `item_reference` be resolved locally (splicing the full item back into the
    outgoing request) instead of forwarding an unresolvable reference upstream.

    Same no-lock reasoning as McpSseSessionStore: plain dict mutations, no await points inside them.
    """

    def __init__(self, ttl_seconds: int = 600, max_items: int = 512) -> None:
        self._items: dict[str, tuple[dict[str, Any], float]] = {}
        self._ttl = ttl_seconds
        self._max = max_items

    def _evict_expired(self) -> None:
        now = time.monotonic()
        expired = [item_id for item_id, (_, ts) in self._items.items() if now - ts >= self._ttl]
        for item_id in expired:
            del self._items[item_id]

    def put(self, item: dict[str, Any]) -> None:
        """Store an output item by its id, evicting expired/oldest if at capacity."""
        item_id = item.get("id")
        if not isinstance(item_id, str):
            return
        self._evict_expired()
        if len(self._items) >= self._max and item_id not in self._items:
            oldest = next(iter(self._items))
            del self._items[oldest]
        self._items[item_id] = (item, time.monotonic())

    def get(self, item_id: str) -> dict[str, Any] | None:
        """Return a stored item, or None if unknown or expired."""
        entry = self._items.get(item_id)
        if entry is None:
            return None
        item, ts = entry
        if time.monotonic() - ts >= self._ttl:
            del self._items[item_id]
            return None
        return item


async def _rewrite_sse_endpoint_events(
    content: AsyncGenerator[bytes],
    upstream_sse_url: str,
    proxy_endpoint_url: str,
    session_store: McpSseSessionStore,
) -> AsyncGenerator[bytes]:
    """Stream SSE events, rewriting 'endpoint' event data URLs to point through the proxy."""
    buffer = ""
    in_endpoint_event = False

    async for chunk in content:
        buffer += chunk.decode("utf-8", errors="replace")

        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.rstrip("\r")

            if not line:
                in_endpoint_event = False
                yield b"\n"
                continue

            if line.startswith("event:"):
                in_endpoint_event = line[6:].strip() == "endpoint"
                yield (line + "\n").encode()
            elif line.startswith("data:") and in_endpoint_event:
                raw = line[5:].strip()
                upstream_url = raw if raw.startswith("http") else urljoin(upstream_sse_url, raw)
                parsed = urlparse(upstream_url)
                session_id = (parse_qs(parsed.query).get("sessionId") or [None])[0]
                if session_id:
                    session_store.add(session_id, upstream_url)
                    yield f"data: {proxy_endpoint_url}?sessionId={session_id}\n".encode()
                else:
                    yield (line + "\n").encode()
            else:
                yield (line + "\n").encode()

    if buffer:
        yield buffer.encode()


@dataclass
class ProxyOptions:
    url: str
    rewrite_model_to: str | None = None
    remove_model: bool = False
    headers: dict[str, str] | None = None
    allowed_response_headers: list[str] | None = None
    allowed_request_headers: list[str] | None = None
    dynamic_headers: Callable[[], Awaitable[dict[str, str]]] | None = None
    on_reauth: Callable[[], Awaitable[dict[str, str] | None]] | None = None
    # Max idle time between reads from the backend, not a total-request cap. Only applied by
    # register_custom_endpoint_as_proxy; ignored by every other proxy path. This is deliberate —
    # the longer timeout is meant for doc-chunker and other custom services only.
    read_timeout_seconds: int = 300
    # Custom-endpoint and streamable-HTTP MCP proxies only. True forwards the full path (prefix included) — for proxying to
    # another DF-Infra node's /custom/{full_path} or /mcp/{full_path}. False (default) strips the prefix — for
    # proxying straight to a backing container, which doesn't know about it.
    forward_full_path: bool = False
    # Chat/legacy completions proxies only. True forwards backend-specific fields (see CHAT_COMPLETION_EXTRA_FIELDS) — for
    # self-hosted backends and other DF-Infra nodes. False (default) strips them — for remote providers that reject unknown fields.
    forward_extra_fields: bool = False

    async def get_request_headers(self, request: Request | None) -> dict[str, str]:
        """Get request headers, merging in freshly-computed dynamic headers (e.g. a live OAuth token) last."""
        static_headers = self.headers or {}
        if request:
            allowed_request_headers = self.allowed_request_headers or []
            request_headers = {k: v for k, v in dict(request.headers).items() if k in allowed_request_headers}
            headers = request_headers | static_headers
        else:
            headers = dict(static_headers)
        if self.dynamic_headers:
            headers |= await self.dynamic_headers()
        return headers


async def _make_request_with_reauth(
    url: str,
    method: str,
    data: bytes | None,
    headers: dict[str, str],
    options: ProxyOptions,
    timeout: ClientTimeout | None = None,
) -> HttpResponse:
    """Issue a request; on a 401 with `on_reauth` configured, refresh headers and retry once.

    If `on_reauth` isn't set, or returns None (nothing to refresh, e.g. no refresh token),
    the original 401 response is returned untouched so it streams through to the client.
    """
    response = await make_http_request(url=url, method=method, data=data, headers=headers, timeout=timeout)
    if response.response.status != 401 or not options.on_reauth:
        return response
    new_headers = await options.on_reauth()
    if new_headers is None:
        return response
    await response.discard()
    return await make_http_request(url=url, method=method, data=data, headers=headers | new_headers, timeout=timeout)


async def _read_body_with_limit(request: Request, max_bytes: int) -> bytes:
    """Buffer `request`'s body, raising 413 instead of growing past `max_bytes`."""
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(413, f"Request body exceeds the {max_bytes}-byte limit for this MCP proxy endpoint.")
        chunks.append(chunk)
    return b"".join(chunks)


def _mcp_transport_changed(existing: RegisteredModel[Any], model: Model) -> bool:
    """Whether an MCP registration's transport differs from the peer's, which needs a different proxy."""
    return model.type == "mcp" and existing.props.transport != model.props.transport


class ModelInfo(NamedTuple):
    id: str
    type: str


class EndpointRegistry:
    def __init__(
        self,
        config: AppSettings,
        parent_infra: ParentInfraGroup,
        model_tester: "ModelTester",
        metrics_registry: MetricsRegistry,
    ):
        self.config = config
        self.parent_infra = parent_infra
        self.parent_infra.endpoint_registry = self
        self.model_tester = model_tester
        self.metrics_registry = metrics_registry
        self.registry = dict[RegistrationId, RegistryEntry]()
        self.chat_completion_endpoints = Endpoint[ChatCompletionEndpoint](self.registry, self.parent_infra)
        self.embeddings_endpoints = Endpoint[SimpleEndpoint[EmbeddingRequest]](self.registry, self.parent_infra)
        self.audio_speech_endpoints = Endpoint[SimpleEndpoint[CreateSpeechRequest]](self.registry, self.parent_infra)
        self.audio_transcriptions_endpoints = Endpoint[SimpleEndpoint[CreateTranscriptionRequest]](self.registry, self.parent_infra)
        self.custom_endpoints = Endpoint[CustomEndpoint](self.registry, self.parent_infra)
        self.images_generations_endpoints = Endpoint[SimpleEndpoint[ImagesRequest]](self.registry, self.parent_infra)
        self.rerank_endpoints = Endpoint[SimpleEndpoint[RerankRequest]](self.registry, self.parent_infra)
        self.systemone_endpoints = Endpoint[SimpleEndpoint[SystemOneRequest]](self.registry, self.parent_infra)
        self.mcp_endpoints = Endpoint[McpEndpoint](self.registry, self.parent_infra)
        self.response_item_store = ResponseItemStore()
        self._proxy_api_keys = dict[str, str]()

    @property
    def _proxy_timeout(self) -> ClientTimeout:
        """Default timeout for outbound proxy requests, driven by `DF_STANDARD_PROXY_TIMEOUT_SECONDS`.

        This is a total-duration cap (`ClientTimeout.total`) here - the value used for chat
        completions, embeddings, audio, images, rerank, and MCP proxying. Custom-service proxying
        (doc-chunker, GLiNER, etc.) reads the same setting but interprets it differently: as a
        per-chunk idle-read window with no total cap - see `ProxyOptions.read_timeout_seconds`,
        and `CustomService.get_proxy_timeout_field` for the per-service override.
        """
        return ClientTimeout(total=self.config.standard_proxy_timeout_seconds, sock_connect=30)

    def get_models(self) -> ApiModels:
        """Get models for api."""
        models: list[ApiModel] = []
        for model in self.chat_completion_endpoints.get_models():
            models.append(model)
        for model in self.embeddings_endpoints.get_models():
            models.append(model)
        for model in self.audio_speech_endpoints.get_models():
            models.append(model)
        for model in self.audio_transcriptions_endpoints.get_models():
            models.append(model)
        for model in self.images_generations_endpoints.get_models():
            models.append(model)
        for model in self.rerank_endpoints.get_models():
            models.append(model)
        for model in self.systemone_endpoints.get_models():
            models.append(model)
        for model in self.mcp_endpoints.get_healthy_models():
            models.append(model)
        for model in self.custom_endpoints.get_models():
            models.append(model)
        return ApiModels(data=models)

    def get_model(self, model_id: str) -> ApiModel:
        """Get model for api."""
        models = self.get_models()
        model = next((x for x in models.data if x.id == model_id), None)
        if model is None:
            raise HTTPException(404, f"Model not found {model_id}")
        return model

    def list_models(self) -> list[Model]:
        """List models for load balancing."""
        models = list[Model]()
        models.extend(self.chat_completion_endpoints.list_models())
        models.extend(self.embeddings_endpoints.list_models())
        models.extend(self.audio_speech_endpoints.list_models())
        models.extend(self.audio_transcriptions_endpoints.list_models())
        models.extend(self.images_generations_endpoints.list_models())
        models.extend(self.rerank_endpoints.list_models())
        models.extend(self.systemone_endpoints.list_models())
        models.extend(self.custom_endpoints.list_models())
        models.extend(self.mcp_endpoints.list_models())
        return models

    def list_own_models(self) -> list[Model]:
        """List only this node's genuinely-own models — excludes both mesh directions.

        Unlike `list_models()`, which reports everything routable through this node (including
        models proxied up from children) for this node's own upward report, this excludes
        `owned_by in {"mesh", "mesh-ancestor"}` entirely. Used to build what this node exposes
        downward to its children as "its own" models — reusing `list_models()` there would echo
        a child's model back down to it under the same registration id as its local entry.
        """
        exclude = frozenset({"mesh", "mesh-ancestor"})
        models = list[Model]()
        models.extend(self.chat_completion_endpoints.list_models(exclude))
        models.extend(self.embeddings_endpoints.list_models(exclude))
        models.extend(self.audio_speech_endpoints.list_models(exclude))
        models.extend(self.audio_transcriptions_endpoints.list_models(exclude))
        models.extend(self.images_generations_endpoints.list_models(exclude))
        models.extend(self.rerank_endpoints.list_models(exclude))
        models.extend(self.systemone_endpoints.list_models(exclude))
        models.extend(self.custom_endpoints.list_models(exclude))
        models.extend(self.mcp_endpoints.list_models(exclude))
        return models

    def model_exists(self, model_id: ModelId) -> bool:
        """Check if model exists in any endpoint."""
        for reg in (
            self.chat_completion_endpoints,
            self.embeddings_endpoints,
            self.audio_speech_endpoints,
            self.audio_transcriptions_endpoints,
            self.images_generations_endpoints,
            self.rerank_endpoints,
            self.systemone_endpoints,
        ):
            if model_id in reg.models:
                return True
        return False

    def register_chat_completion(
        self,
        model: str,
        props: ModelProps,
        endpoint: ChatCompletionEndpoint,
        registration_options: RegistrationOptions | None,
    ) -> RegistrationId:
        """Register chat completion endpoint for given model."""
        suffixes = [suffix for field, suffix in LLM_SUFFIX_MAP if getattr(endpoint, field)]

        model_type = "llm" if len(suffixes) == len(ALL_LLM_SUFFIXES) else "llm-" + "-".join(suffixes)
        return self.chat_completion_endpoints.add_model(model, props, endpoint, model_type, registration_options)

    def register_chat_completion_as_proxy(  # noqa: C901
        self,
        model: str,
        props: ModelProps,
        chat_completions: ProxyOptions | None,
        completions: ProxyOptions | None,
        responses: ProxyOptions | None,
        messages: ProxyOptions | None,
        ollama_chat: ProxyOptions | None,
        registration_options: RegistrationOptions | None,
    ) -> RegistrationId:
        """Register chat completion for given model as a proxy."""
        endpoint = ChatCompletionEndpoint()
        if not chat_completions and not completions:
            raise RuntimeError("Chat completions nor completions registered " + model)
        if chat_completions:

            async def on_chat_completions_request(body: ChatCompletionRequest, request: Request | None) -> StreamingResponse:
                return await post_json(body, chat_completions, request, timeout=self._proxy_timeout)

            endpoint.on_chat_completion = on_chat_completions_request
        if completions:

            async def on_completion_request(body: CompletionLegacyRequest, request: Request | None) -> StreamingResponse:
                return await post_json(body, completions, request, timeout=self._proxy_timeout)

            endpoint.on_completion = on_completion_request
        if responses:

            async def on_responses_request(body: ResponsesRequest, request: Request | None) -> StreamingResponse:
                return await post_json_responses(body, responses, self.response_item_store, request, timeout=self._proxy_timeout)

            endpoint.on_responses = on_responses_request

        if messages:

            async def on_messages_request(body: MessagesRequest, request: Request | None) -> StreamingResponse:
                return await post_json(body, messages, request, timeout=self._proxy_timeout)

            endpoint.on_messages = on_messages_request

        if ollama_chat:

            async def on_ollama_chat_request(body: OllamaChatRequest, request: Request | None) -> StreamingResponse:
                return await post_json(body, ollama_chat, request, timeout=self._proxy_timeout)

            endpoint.on_ollama_chat = on_ollama_chat_request

        return self.register_chat_completion(model, props, endpoint, registration_options)

    def unregister_chat_completion(self, model: str, registration_id: RegistrationId) -> None:
        """Unregister chat completion for given model."""
        self.chat_completion_endpoints.remove_model(model, registration_id)

    def register_embeddings(
        self,
        model: str,
        props: ModelProps,
        endpoint: SimpleEndpoint[EmbeddingRequest],
        registration_options: RegistrationOptions | None,
    ) -> RegistrationId:
        """Register embeddings endpoint for given model."""
        return self.embeddings_endpoints.add_model(model, props, endpoint, "embedding", registration_options)

    def register_embeddings_as_proxy(
        self,
        model: str,
        props: ModelProps,
        options: ProxyOptions,
        registration_options: RegistrationOptions | None,
    ) -> RegistrationId:
        """Register embeddings for given model as a proxy."""

        async def on_request(body: EmbeddingRequest, request: Request | None) -> StreamingResponse:
            return await post_json(body, options, request, timeout=self._proxy_timeout)

        return self.register_embeddings(model, props, SimpleEndpoint(on_request=on_request), registration_options)

    def unregister_embeddings(self, model: str, registration_id: RegistrationId) -> None:
        """Unregister embeddings for given model."""
        self.embeddings_endpoints.remove_model(model, registration_id)

    def register_audio_speech(
        self,
        model: str,
        props: ModelProps,
        endpoint: SimpleEndpoint[CreateSpeechRequest],
        registration_options: RegistrationOptions | None,
    ) -> RegistrationId:
        """Register audio speech endpoint for given model."""
        return self.audio_speech_endpoints.add_model(model, props, endpoint, "tts", registration_options)

    def register_audio_speech_as_proxy(
        self,
        model: str,
        props: ModelProps,
        options: ProxyOptions,
        registration_options: RegistrationOptions | None,
    ) -> RegistrationId:
        """Register audio speech for given model as a proxy."""

        async def on_request(body: CreateSpeechRequest, request: Request | None) -> StreamingResponse:
            return await post_json(body, options, request, timeout=self._proxy_timeout)

        return self.register_audio_speech(model, props, SimpleEndpoint(on_request=on_request), registration_options)

    def unregister_audio_speech(self, model: str, registration_id: RegistrationId) -> None:
        """Unregister audio speech for given model."""
        self.audio_speech_endpoints.remove_model(model, registration_id)

    def register_audio_transcriptions(
        self,
        model: str,
        props: ModelProps,
        endpoint: SimpleEndpoint[CreateTranscriptionRequest],
        registration_options: RegistrationOptions | None,
    ) -> RegistrationId:
        """Register audio transcriptions endpoint for given model."""
        return self.audio_transcriptions_endpoints.add_model(model, props, endpoint, "stt", registration_options)

    def register_audio_transcriptions_as_proxy(
        self,
        model: str,
        props: ModelProps,
        options: ProxyOptions,
        registration_options: RegistrationOptions | None,
    ) -> RegistrationId:
        """Register audio transcriptions for given model as a proxy."""

        async def on_request(body: CreateTranscriptionRequest, request: Request | None) -> StreamingResponse:
            return await post_form(body, options, request, timeout=self._proxy_timeout)

        return self.register_audio_transcriptions(model, props, SimpleEndpoint(on_request=on_request), registration_options)

    def unregister_audio_transcriptions(self, model: str, registration_id: RegistrationId) -> None:
        """Unregister audio transcriptions for given model."""
        self.audio_transcriptions_endpoints.remove_model(model, registration_id)

    def register_image_generations(
        self,
        model: str,
        props: ModelProps,
        endpoint: SimpleEndpoint[ImagesRequest],
        registration_options: RegistrationOptions | None,
    ) -> RegistrationId:
        """Register image generations endpoint for given model."""
        return self.images_generations_endpoints.add_model(model, props, endpoint, "txt2img", registration_options)

    def register_image_generations_as_proxy(
        self,
        model: str,
        props: ModelProps,
        options: ProxyOptions,
        registration_options: RegistrationOptions | None,
    ) -> RegistrationId:
        """Register image generations for given model as a proxy."""

        async def on_request(body: ImagesRequest, request: Request | None) -> StreamingResponse:
            return await post_json(body, options, request, timeout=self._proxy_timeout)

        return self.register_image_generations(model, props, SimpleEndpoint(on_request=on_request), registration_options)

    def unregister_image_generations(self, model: str, registration_id: RegistrationId) -> None:
        """Unregister image generations for given model."""
        self.images_generations_endpoints.remove_model(model, registration_id)

    def register_rerank(
        self,
        model: str,
        props: ModelProps,
        endpoint: SimpleEndpoint[RerankRequest],
        registration_options: RegistrationOptions | None,
    ) -> RegistrationId:
        """Register rerank endpoint for given model."""
        return self.rerank_endpoints.add_model(model, props, endpoint, "rerank", registration_options)

    def register_rerank_as_proxy(
        self,
        model: str,
        props: ModelProps,
        options: ProxyOptions,
        registration_options: RegistrationOptions | None,
        normalize_sglang_response: bool = False,
    ) -> RegistrationId:
        """Register rerank for given model as a proxy.

        `normalize_sglang_response` rewrites SGLang's native rerank response (a plain
        `[{"score", "document", "index", ...}]` array) into the Cohere-style `{"results": [...]}`
        shape the rest of this API expects; other backends (vLLM, the dedicated rerank service)
        already return that shape natively and don't need it.
        """

        async def on_request(body: RerankRequest, request: Request | None) -> StreamingResponse:
            if normalize_sglang_response:
                return await post_json_rerank_sglang(body, options, request, timeout=self._proxy_timeout)
            return await post_json(body, options, request, timeout=self._proxy_timeout)

        return self.register_rerank(model, props, SimpleEndpoint(on_request=on_request), registration_options)

    def unregister_rerank(self, model: str, registration_id: RegistrationId) -> None:
        """Unregister rerank for given model."""
        self.rerank_endpoints.remove_model(model, registration_id)

    def register_systemone(
        self,
        model: str,
        props: ModelProps,
        endpoint: SimpleEndpoint[SystemOneRequest],
        registration_options: RegistrationOptions | None,
    ) -> RegistrationId:
        """Register systemone (decision model) endpoint for given model."""
        return self.systemone_endpoints.add_model(model, props, endpoint, "decision", registration_options)

    def register_systemone_as_proxy(
        self,
        model: str,
        props: ModelProps,
        options: ProxyOptions,
        registration_options: RegistrationOptions | None,
    ) -> RegistrationId:
        """Register systemone (decision model) for given model as a proxy."""

        async def on_request(body: SystemOneRequest, request: Request | None) -> StreamingResponse:
            return await post_json(body, options, request, timeout=self._proxy_timeout)

        return self.register_systemone(model, props, SimpleEndpoint(on_request=on_request), registration_options)

    def unregister_systemone(self, model: str, registration_id: RegistrationId) -> None:
        """Unregister systemone (decision model) for given model."""
        self.systemone_endpoints.remove_model(model, registration_id)

    def register_custom_endpoint(
        self,
        url: str,
        props: ModelProps,
        endpoint: CustomEndpoint,
        registration_options: RegistrationOptions | None,
    ) -> RegistrationId:
        """Register custom endpoint."""
        return self.custom_endpoints.add_model(url, props, endpoint, "custom", registration_options)

    def register_custom_endpoint_as_proxy(
        self,
        url: str,
        props: ModelProps,
        options: ProxyOptions,
        registration_options: RegistrationOptions | None,
    ) -> RegistrationId:
        """Register custom endpoint as a proxy."""

        async def on_request(request: Request) -> StreamingResponse:
            headers = await options.get_request_headers(request)
            headers["content-type"] = request.headers.get("content-type") or "application/octet-stream"
            if options.forward_full_path:
                sub_path = request.path_params["full_path"]
            else:
                _, _, sub_path = request.path_params["full_path"].partition("/")
            full_url = Utils.join_url(options.url, sub_path) if sub_path else options.url
            if request.url.query:
                full_url = f"{full_url}?{request.url.query}"
            return (
                await make_http_request(
                    url=full_url,
                    method=request.method,
                    data=request.stream(),
                    headers=headers,
                    timeout=ClientTimeout(total=None, sock_connect=30, sock_read=options.read_timeout_seconds),
                )
            ).as_streaming_response(options.allowed_response_headers)

        return self.register_custom_endpoint(url, props, CustomEndpoint(on_request=on_request), registration_options)

    def unregister_custom_endpoint(self, model: str, registration_id: RegistrationId) -> None:
        """Unregister custom endpoint."""
        self.custom_endpoints.remove_model(model, registration_id)

    def register_mcp_endpoint(
        self,
        url: str,
        props: ModelProps,
        endpoint: McpEndpoint,
        registration_options: RegistrationOptions | None,
    ) -> RegistrationId:
        """Register mcp endpoint."""
        return self.mcp_endpoints.add_model(url, props, endpoint, "mcp", registration_options)

    def register_mcp_endpoint_as_proxy(
        self,
        url: str,
        props: ModelProps,
        options: ProxyOptions,
        registration_options: RegistrationOptions | None,
    ) -> RegistrationId:
        """Register mcp endpoint as a proxy."""

        async def on_request(request: Request) -> StreamingResponse:
            headers = await options.get_request_headers(request)
            headers["content-type"] = request.headers.get("content-type") or "application/octet-stream"
            # Mesh proxy only: the child routes by the `{prefix}/...` path, so the whole path must be forwarded.
            full_url = options.url
            if options.forward_full_path:
                full_path: str = request.path_params["full_path"]
                if ".." in full_path.split("/"):
                    raise HTTPException(400, "Invalid MCP path.")
                full_url = Utils.join_url(options.url, full_path)
            if request.url.query:
                full_url = f"{full_url}?{request.url.query}"
            logger.debug("MCP proxy: %s %s -> %s", request.method, request.path_params["full_path"], full_url)
            if options.on_reauth:
                # Buffered (not streamed) so a 401-triggered retry can replay the body; size-capped to bound memory use.
                # Only needed for OAuth-enabled proxies — everything else keeps streaming the body straight through.
                body = await _read_body_with_limit(request, _MCP_PROXY_MAX_BODY_BYTES)
                response = await _make_request_with_reauth(full_url, request.method, body, headers, options, timeout=self._proxy_timeout)
            else:
                response = await make_http_request(
                    url=full_url, method=request.method, data=request.stream(), headers=headers, timeout=self._proxy_timeout
                )
            logger.debug("MCP proxy response: %s", response.response.status)
            return response.as_streaming_response(options.allowed_response_headers)

        return self.register_mcp_endpoint(url, props, McpEndpoint(on_request=on_request), registration_options)

    def register_mcp_sse_endpoint_as_proxy(
        self,
        url: str,
        props: ModelProps,
        options: ProxyOptions,
        registration_options: RegistrationOptions | None,
    ) -> RegistrationId:
        """Register an SSE-transport MCP endpoint as a proxy.

        GET  → streams upstream SSE, rewriting the 'endpoint' event URL to route
               through this proxy so the client never speaks directly to upstream.
        POST → forwards the client message to the upstream messages URL stored in
               the session store that was populated during the GET phase.
        """
        session_store = McpSseSessionStore(
            ttl_seconds=self.config.mcp_sse_session_ttl_seconds,
            max_sessions=self.config.mcp_sse_max_sessions,
        )

        async def on_request(request: Request) -> StreamingResponse:
            headers = await options.get_request_headers(request)

            if request.method == "GET":
                headers["accept"] = "text/event-stream"
                headers.pop("content-type", None)
                proxy_endpoint_url = str(request.url).split("?")[0]
                # A mid-SSE-stream token expiry can't be transparently retried (no client body to
                # replay mid-stream); only this initial GET goes through the reauth helper. Proactive
                # background refresh (see McpService) mitigates the gap; the client must reconnect
                # if the token expires after the stream is already established.
                upstream_response = await _make_request_with_reauth(options.url, "GET", None, headers, options, timeout=self._proxy_timeout)

                async def rewritten() -> AsyncGenerator[bytes]:
                    session_id: str | None = None
                    try:
                        async for chunk in _rewrite_sse_endpoint_events(
                            upstream_response.content,
                            options.url,
                            proxy_endpoint_url,
                            session_store,
                        ):
                            if session_id is None and b"sessionId=" in chunk:
                                session_id = chunk.decode(errors="replace").split("sessionId=")[-1].strip()
                            yield chunk
                    finally:
                        if session_id:
                            session_store.remove(session_id)
                        await upstream_response.response.release()

                return StreamingResponse(
                    rewritten(),
                    media_type="text/event-stream",
                    status_code=200,
                    headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
                )

            session_id = request.query_params.get("sessionId")
            if not session_id:
                raise HTTPException(400, "Missing sessionId query parameter")
            upstream_url = session_store.get(session_id)
            if upstream_url is None:
                raise HTTPException(404, f"Unknown session: {session_id}")
            headers["content-type"] = request.headers.get("content-type") or "application/json"
            response = await make_http_request(
                url=upstream_url, method="POST", data=request.stream(), headers=headers, timeout=self._proxy_timeout
            )
            return response.as_streaming_response(options.allowed_response_headers)

        return self.register_mcp_endpoint(url, props, McpEndpoint(on_request=on_request), registration_options)

    def unregister_mcp_endpoint(self, model: str, registration_id: RegistrationId) -> None:
        """Unregister mcp endpoint."""
        self.mcp_endpoints.remove_model(model, registration_id)

    def update_usage(self, usage: UsageChangeRequest) -> None:
        """Update model usage."""
        entry = self.registry.get(usage.id, None)
        if entry and entry.registered_model.usage != usage.usage:
            entry.registered_model.usage = usage.usage
            self._refresh_usage(entry.registered_model)

    def update_warm(self, registration_id: RegistrationId, warm: bool) -> None:
        """Update whether a registration's model is currently loaded/warm on its backend."""
        entry = self.registry.get(registration_id, None)
        if entry and entry.registered_model.warm != warm:
            entry.registered_model.warm = warm
            self.parent_infra.send_warm(WarmChangeRequest(id=registration_id, warm=warm))

    def update_modality_support(
        self, registration_id: RegistrationId, *, supports_vision: bool | None, supports_audio: bool | None
    ) -> None:
        """Update a registration's resolved `supports_vision`/`supports_audio` in place.

        Lets a service keep an already-registered model's props in sync when its catalog is
        refreshed after install (e.g. a `classify_modality_support` fix or a regenerated static
        catalog correcting a previously-unknown or wrong value), without unregistering and losing
        the registration's id, usage counters, or warm/health state.
        """
        entry = self.registry.get(registration_id, None)
        if entry is None:
            return
        props = entry.registered_model.props
        if props.supports_vision != supports_vision or props.supports_audio != supports_audio:
            entry.registered_model.props = props.model_copy(update={"supports_vision": supports_vision, "supports_audio": supports_audio})

    def update_models(self, prev_list: list[Model], new_list: list[Model], api_url: str, api_key: str, owned_by: str = "mesh") -> None:
        """Update models, it remove old models and add new ones.

        `owned_by` distinguishes the direction the models came from: `"mesh"` for models proxied
        up from a subinfra (the default, existing behavior), `"mesh-ancestor"` for models proxied
        down from a parent/ancestor (see `ParentInfra._apply_ancestors`). `list_models()` excludes
        the latter so they aren't echoed back to whoever they came from.

        Defensive against a mesh peer on a version that doesn't (yet, or anymore) filter what it
        reports: a registration id is only ever registered or removed here if the existing registry
        entry for that id (if any) has the same `owned_by` as this call. This stops a peer from
        overwriting or deleting an entry that actually belongs to a different origin (e.g. this
        node's own local model, or one proxied from a different direction) just because it happens
        to reuse the same id.

        `api_key` is compared against the last key seen for `api_url`: an unchanged model list
        alone would otherwise never re-run `_register_proxy`, so a peer rotating its API key
        (e.g. `infra_api_key`) would leave every already-registered proxy endpoint calling out
        with the stale `Authorization` header until the model happened to be re-registered for an
        unrelated reason (e.g. a reconnect).
        """
        key_changed = self._proxy_api_keys.get(api_url) != api_key
        self._proxy_api_keys[api_url] = api_key
        registered = set[RegistrationId]()
        changed = False
        for model in new_list:
            registered.add(model.id)
            existing = self.registry.get(model.id)
            if existing is not None and existing.registered_model.owned_by != owned_by:
                logger.warning(
                    "Ignoring model id collision: id=%s from origin=%s owned_by=%r conflicts with "
                    "existing registration owned_by=%r origin=%s",
                    model.id,
                    api_url,
                    owned_by,
                    existing.registered_model.owned_by,
                    existing.registered_model.origin,
                )
                continue
            if existing is None:
                logger.info(f"Register new model origin={api_url} id={model.id} name={model.name} type={model.type}")
            elif key_changed or _mcp_transport_changed(existing.registered_model, model):
                logger.info(f"Refresh proxy origin={api_url} id={model.id} name={model.name} type={model.type}")
                existing.endpoint.remove_model(model_id=existing.model_id, registration_id=model.id, send_notification=False)
            else:
                # Same proxy, but the peer may have changed MCP props (tools) or health since the last sync.
                registered_model = existing.registered_model
                if model.type == "mcp" and (registered_model.props != model.props or registered_model.healthy != model.healthy):
                    registered_model.props = model.props
                    registered_model.healthy = model.healthy
                    changed = True
                continue
            changed = True
            self._register_proxy(
                model_id=model.name,
                type=model.type,
                props=model.props,
                url=api_url,
                api_key=api_key,
                registration_options=RegistrationOptions(
                    id=model.id,
                    origin=api_url,
                    usage=model.usage,
                    send_notification=False,
                    owned_by=owned_by,
                    capacity_state=_capacity_state_from_pair(model.capacity, model.capacity_known),
                    warm=model.warm,
                    healthy=model.healthy,
                ),
            )
        for model in prev_list:
            if model.id not in registered:
                prev = self.registry.get(model.id, None)
                if prev and prev.registered_model.owned_by == owned_by:
                    changed = True
                    logger.info(f"Remove old model origin={api_url} id={model.id} name={model.name} type={model.type}")
                    prev.endpoint.remove_model(model_id=prev.model_id, registration_id=model.id, send_notification=False)
        if changed:
            self.parent_infra.send_models_list()

    def _register_mcp_proxy(
        self, model_id: ModelId, props: ModelProps, url: str, api_key: str, registration_options: RegistrationOptions | None
    ) -> None:
        """Register a proxy to an MCP server on a mesh peer, over the transport the peer reported."""
        options = ProxyOptions(
            url=urljoin(url, "mcp"),
            headers={"Authorization": f"Bearer {api_key}"},
            allowed_request_headers=["accept", "mcp-session-id"],
            allowed_response_headers=["accept", "mcp-session-id", "www-authenticate"],
        )
        if props.transport == "sse":
            # The SSE proxy always targets `options.url`, so point it at this prefix on the peer.
            options.url = Utils.join_url(options.url, f"{model_id}/sse")
            self.register_mcp_sse_endpoint_as_proxy(url=model_id, props=props, options=options, registration_options=registration_options)
        else:
            options.forward_full_path = True
            self.register_mcp_endpoint_as_proxy(url=model_id, props=props, options=options, registration_options=registration_options)

    def _register_proxy(  # noqa: C901
        self,
        model_id: ModelId,
        type: str,
        props: ModelProps,
        url: str,
        api_key: str,
        registration_options: RegistrationOptions | None,
    ) -> None:
        """Register proxy."""
        if type == "llm" or type.startswith("llm-"):
            headers = {"Authorization": f"Bearer {api_key}"}
            parts = set(ALL_LLM_SUFFIXES) if type == "llm" else set(type.split("-")[1:])

            def _proxy(path: str) -> ProxyOptions:
                # The peer decides what to strip, since only it knows its backend.
                return ProxyOptions(url=urljoin(url, path), headers=headers, forward_extra_fields=True)

            self.register_chat_completion_as_proxy(
                model=model_id,
                props=props,
                completions=_proxy("v1/completions") if "v1" in parts else None,
                chat_completions=_proxy("v1/chat/completions") if "v2" in parts else None,
                responses=_proxy("v1/responses") if "v3" in parts else None,
                messages=_proxy("v1/messages") if "ant" in parts else None,
                ollama_chat=_proxy("api/chat") if "ollama" in parts else None,
                registration_options=registration_options,
            )
        elif type == "tts":
            self.register_audio_speech_as_proxy(
                model=model_id,
                props=props,
                options=ProxyOptions(
                    url=urljoin(url, "v1/audio/speech"),
                    headers={"Authorization": f"Bearer {api_key}"},
                ),
                registration_options=registration_options,
            )
        elif type == "stt":
            self.register_audio_transcriptions_as_proxy(
                model=model_id,
                props=props,
                options=ProxyOptions(
                    url=urljoin(url, "v1/audio/transcriptions"),
                    headers={"Authorization": f"Bearer {api_key}"},
                ),
                registration_options=registration_options,
            )
        elif type == "txt2img":
            self.register_image_generations_as_proxy(
                model=model_id,
                props=props,
                options=ProxyOptions(
                    url=urljoin(url, "v1/images/generations"),
                    headers={"Authorization": f"Bearer {api_key}"},
                ),
                registration_options=registration_options,
            )
        elif type == "embedding":
            self.register_embeddings_as_proxy(
                model=model_id,
                props=props,
                options=ProxyOptions(
                    url=urljoin(url, "v1/embeddings"),
                    headers={"Authorization": f"Bearer {api_key}"},
                ),
                registration_options=registration_options,
            )
        elif type == "rerank":
            self.register_rerank_as_proxy(
                model=model_id,
                props=props,
                options=ProxyOptions(
                    url=urljoin(url, "v1/rerank"),
                    headers={"Authorization": f"Bearer {api_key}"},
                ),
                registration_options=registration_options,
            )
        elif type == "decision":
            self.register_systemone_as_proxy(
                model=model_id,
                props=props,
                options=ProxyOptions(
                    url=urljoin(url, "v1/systemone"),
                    headers={"Authorization": f"Bearer {api_key}"},
                ),
                registration_options=registration_options,
            )
        elif type == "custom":
            self.register_custom_endpoint_as_proxy(
                url=model_id,
                props=props,
                options=ProxyOptions(
                    url=urljoin(url, "custom"),
                    headers={"Authorization": f"Bearer {api_key}"},
                    read_timeout_seconds=self.config.standard_proxy_timeout_seconds,
                    forward_full_path=True,
                ),
                registration_options=registration_options,
            )
        elif type == "mcp":
            self._register_mcp_proxy(model_id, props, url, api_key, registration_options)
        else:
            logger.warning(f"Cannot register proxy with model_type={type}")

    async def execute_messages(
        self,
        body: MessagesRequest,
        request: Request | None = None,
        registration_id: RegistrationId | None = None,
    ) -> StarletteResponse:
        """Process messages request."""
        if self.config.is_log_payloads_enabled():
            logger.info(f"DUMP REQUEST PAYLOAD /v1/messages {body.model_dump_json(exclude_none=True)}")

        endpoint = self.chat_completion_endpoints.get_model(
            body.model,
            filter=lambda x: x.on_messages is not None,
            registration_id=registration_id,
        )
        on_messages = endpoint.endpoint.on_messages if endpoint else None
        if not endpoint or not on_messages:
            if not self.model_exists(body.model):
                raise HTTPException(404, "Model not found")
            msg = "Given model not support this endpoint.\n"
            if endpoint:
                supported_endpoints = []
                if endpoint.endpoint.on_responses:
                    supported_endpoints.append("/v1/responses")
                if endpoint.endpoint.on_chat_completion:
                    supported_endpoints.append("/v1/chat/completions")
                if endpoint.endpoint.on_completion:
                    supported_endpoints.append("/v1/completions")

                msg = msg + "Supported endpoints:\n" + ", ".join(supported_endpoints)

            raise HTTPException(400, msg)

        async def func() -> StarletteResponse:
            return await on_messages(body, request)

        return await self.with_usage(endpoint, func, self.config.is_log_payloads_enabled())

    async def execute_responses(
        self,
        body: ResponsesRequest,
        request: Request | None = None,
        registration_id: RegistrationId | None = None,
    ) -> StarletteResponse:
        """Process responses request."""
        if isinstance(body.input, list):
            body.input = [self._resolve_item_reference(item) for item in body.input]

        if self.config.is_log_payloads_enabled():
            logger.info(f"DUMP REQUEST PAYLOAD /v1/responses {body.model_dump_json(exclude_none=True)}")

        endpoint = self.chat_completion_endpoints.get_model(
            body.model,
            filter=lambda x: x.on_responses is not None,
            registration_id=registration_id,
        )
        on_responses = endpoint.endpoint.on_responses if endpoint else None
        if not endpoint or not on_responses:
            if not self.model_exists(body.model):
                raise HTTPException(404, "Model not found")
            msg = "Given model not support this endpoint.\n"
            if endpoint:
                supported_endpoints = []
                if endpoint.endpoint.on_messages:
                    supported_endpoints.append("/v1/messages")
                if endpoint.endpoint.on_chat_completion:
                    supported_endpoints.append("/v1/chat/completions")
                if endpoint.endpoint.on_completion:
                    supported_endpoints.append("/v1/completions")

                msg = msg + "Supported endpoints:\n" + ", ".join(supported_endpoints)

            raise HTTPException(400, msg)

        async def func() -> StarletteResponse:
            return await on_responses(body, request)

        return await self.with_usage(endpoint, func, self.config.is_log_payloads_enabled())

    def _resolve_item_reference(self, item: Input) -> Input:
        """Splice a cached item in place of an `item_reference`, if this gateway produced it earlier."""
        if not isinstance(item, ItemReference):
            return item
        stored = self.response_item_store.get(item.id)
        if stored is None:
            return item
        try:
            return _input_item_adapter.validate_python(stored)
        except ValidationError:
            logger.warning("Cached item for id=%s failed to validate as an Input item", item.id)
            return item

    async def execute_chat_completion(
        self,
        body: ChatCompletionRequest,
        request: Request | None = None,
        registration_id: RegistrationId | None = None,
    ) -> StarletteResponse:
        """Process chat completion request."""
        if self.config.is_log_payloads_enabled():
            logger.info(f"DUMP REQUEST PAYLOAD /v1/chat/completions {body.model_dump_json(exclude_none=True)}")

        endpoint = self.chat_completion_endpoints.get_model(
            body.model,
            filter=lambda x: x.on_chat_completion is not None,
            registration_id=registration_id,
        )
        on_chat_completion = endpoint.endpoint.on_chat_completion if endpoint else None
        if not endpoint or not on_chat_completion:
            if not self.model_exists(body.model):
                raise HTTPException(404, "Model not found")
            msg = "Given model not support this endpoint.\n"
            if endpoint:
                supported_endpoints = []
                if endpoint.endpoint.on_messages:
                    supported_endpoints.append("/v1/messages")
                if endpoint.endpoint.on_responses:
                    supported_endpoints.append("/v1/responses")
                if endpoint.endpoint.on_completion:
                    supported_endpoints.append("/v1/completions")

                msg = msg + "Supported endpoints:\n" + ", ".join(supported_endpoints)

            raise HTTPException(400, msg)

        async def func() -> StarletteResponse:
            return await on_chat_completion(body, request)

        return await self.with_usage(endpoint, func, self.config.is_log_payloads_enabled())

    async def execute_completion(
        self,
        body: CompletionLegacyRequest,
        request: Request | None = None,
        registration_id: RegistrationId | None = None,
    ) -> StarletteResponse:
        """Process completion request."""
        endpoint = self.chat_completion_endpoints.get_model(
            body.model,
            filter=lambda x: x.on_completion is not None,
            registration_id=registration_id,
        )
        on_completion = endpoint.endpoint.on_completion if endpoint else None
        if not endpoint or not on_completion:
            if not self.model_exists(body.model):
                raise HTTPException(404, "Model not found")
            msg = "Given model not support this endpoint.\n"
            if endpoint:
                supported_endpoints = []
                if endpoint.endpoint.on_messages:
                    supported_endpoints.append("/v1/messages")
                if endpoint.endpoint.on_responses:
                    supported_endpoints.append("/v1/responses")
                if endpoint.endpoint.on_chat_completion:
                    supported_endpoints.append("/v1/chat/completions")

                msg = msg + "Supported endpoints:\n" + ", ".join(supported_endpoints)

            raise HTTPException(400, msg)

        async def func() -> StarletteResponse:
            return await on_completion(body, request)

        return await self.with_usage(endpoint, func)

    async def execute_ollama_chat(
        self,
        body: OllamaChatRequest,
        request: Request | None = None,
        registration_id: RegistrationId | None = None,
    ) -> StarletteResponse:
        """Process ollama chat."""
        endpoint = self.chat_completion_endpoints.get_model(
            body.model,
            filter=lambda x: x.on_ollama_chat is not None,
            registration_id=registration_id,
        )
        on_ollama_chat = endpoint.endpoint.on_ollama_chat if endpoint else None
        if not endpoint or not on_ollama_chat:
            if not self.model_exists(body.model):
                raise HTTPException(404, "Model not found")
            msg = "Given model not support this endpoint.\n"
            if endpoint:
                supported_endpoints = []
                if endpoint.endpoint.on_messages:
                    supported_endpoints.append("/v1/messages")
                if endpoint.endpoint.on_responses:
                    supported_endpoints.append("/v1/responses")
                if endpoint.endpoint.on_chat_completion:
                    supported_endpoints.append("/v1/chat/completions")
                if endpoint.endpoint.on_completion:
                    supported_endpoints.append("/v1/completions")

                msg = msg + "Supported endpoints:\n" + ", ".join(supported_endpoints)

            raise HTTPException(400, msg)

        async def func() -> StarletteResponse:
            return await on_ollama_chat(body, request)

        return await self.with_usage(endpoint, func)

    async def execute_embeddings(
        self,
        body: EmbeddingRequest,
        request: Request | None = None,
        registration_id: RegistrationId | None = None,
    ) -> StarletteResponse:
        """Process embeddings request."""
        endpoint = self.embeddings_endpoints.get_model(body.model, registration_id=registration_id)
        if not endpoint:
            if not self.model_exists(body.model):
                raise HTTPException(404, "Model not found")
            raise HTTPException(400, "Given model is not supported")

        async def func() -> StarletteResponse:
            return await endpoint.endpoint.on_request(body, request)

        return await self.with_usage(endpoint, func)

    async def execute_images_generations(
        self,
        body: ImagesRequest,
        request: Request | None = None,
        registration_id: RegistrationId | None = None,
    ) -> StarletteResponse:
        """Process images generations request."""
        if self.config.is_log_payloads_enabled():
            logger.info(f"DUMP REQUEST PAYLOAD /v1/images/generations {body.model_dump_json(exclude_none=True)}")
        endpoint = self.images_generations_endpoints.get_model(body.model, registration_id=registration_id)
        if not endpoint:
            if not self.model_exists(body.model):
                raise HTTPException(404, "Model not found")
            raise HTTPException(400, "Given model is not supported")

        async def func() -> StarletteResponse:
            return await endpoint.endpoint.on_request(body, request)

        return await self.with_usage(endpoint, func)

    async def execute_audio_speech(
        self,
        body: CreateSpeechRequest,
        request: Request | None = None,
        registration_id: RegistrationId | None = None,
    ) -> StarletteResponse:
        """Process audio speech request."""
        if self.config.is_log_payloads_enabled():
            logger.info(f"DUMP REQUEST PAYLOAD /v1/audio/speech {body.model_dump_json(exclude_none=True)}")
        endpoint = self.audio_speech_endpoints.get_model(body.model, registration_id=registration_id)
        if not endpoint:
            if not self.model_exists(body.model):
                raise HTTPException(404, "Model not found")
            raise HTTPException(400, "Given model is not supported")

        async def func() -> StarletteResponse:
            return await endpoint.endpoint.on_request(body, request)

        return await self.with_usage(endpoint, func)

    async def execute_audio_transcriptions(
        self,
        body: CreateTranscriptionRequest,
        request: Request | None = None,
        registration_id: RegistrationId | None = None,
    ) -> StarletteResponse:
        """Process audio transcriptions request."""
        endpoint = self.audio_transcriptions_endpoints.get_model(body.model, registration_id=registration_id)
        if not endpoint:
            if not self.model_exists(body.model):
                raise HTTPException(404, "Model not found")
            raise HTTPException(400, "Given model is not supported")

        async def func() -> StarletteResponse:
            return await endpoint.endpoint.on_request(body, request)

        response = await self.with_usage(endpoint, func)
        if body.response_format == "text" and not body.stream:
            return await _unwrap_json_text(response)
        return response

    async def execute_rerank(
        self,
        body: RerankRequest,
        request: Request | None = None,
        registration_id: RegistrationId | None = None,
    ) -> StarletteResponse:
        """Process rerank request."""
        if self.config.is_log_payloads_enabled():
            logger.info(f"DUMP REQUEST PAYLOAD /v1/rerank {body.model_dump_json(exclude_none=True)}")
        endpoint = self.rerank_endpoints.get_model(body.model, registration_id=registration_id)
        if not endpoint:
            if not self.model_exists(body.model):
                raise HTTPException(404, "Model not found")
            raise HTTPException(400, "Given model is not supported")

        async def func() -> StarletteResponse:
            return await endpoint.endpoint.on_request(body, request)

        return await self.with_usage(endpoint, func)

    async def execute_systemone(
        self,
        body: SystemOneRequest,
        request: Request | None = None,
        registration_id: RegistrationId | None = None,
    ) -> StarletteResponse:
        """Process systemone (decision model) request."""
        if self.config.is_log_payloads_enabled():
            logger.info(f"DUMP REQUEST PAYLOAD /v1/systemone {body.model_dump_json(exclude_none=True)}")
        endpoint = self.systemone_endpoints.get_model(body.model, registration_id=registration_id)
        if not endpoint:
            if not self.model_exists(body.model):
                raise HTTPException(404, "Model not found")
            raise HTTPException(400, "Given model is not supported")

        async def func() -> StarletteResponse:
            return await endpoint.endpoint.on_request(body, request)

        return await self.with_usage(endpoint, func)

    async def execute_custom_endpoints(self, url: str, request: Request) -> StarletteResponse:
        """Process custom endpoint request."""
        endpoint = self.custom_endpoints.get_model(url.split("/")[0])
        if not endpoint:
            raise HTTPException(400, "Given url is not supported")

        async def func() -> StarletteResponse:
            return await endpoint.endpoint.on_request(request)

        return await self.with_usage(endpoint, func)

    async def execute_mcp_endpoints(self, url: str, request: Request) -> StarletteResponse:
        """Process mcp endpoint request."""
        endpoint = self.mcp_endpoints.get_model(url.split("/")[0])
        if not endpoint:
            raise HTTPException(400, "Given url is not supported")

        async def func() -> StarletteResponse:
            return await endpoint.endpoint.on_request(request)

        return await self.with_usage(endpoint, func)

    async def with_usage(self, model: RegisteredModel[Any], func: Callable[[], Awaitable[T]], log_payload: bool = False) -> T:
        """With usage."""
        if model.origin != "local":
            return await func()
        self._add_usage(model)
        self.metrics_registry.requests_in_flight.labels(model_type=model.type).inc()
        start_time = time.perf_counter()

        try:
            resp = await func()
        except Exception as e:
            self._record_request_metrics(model, start_time, _classify_error(e))
            self._report_upstream_exception(model, e)
            raise

        # A 5xx from the model is passed through to the client unchanged, so it has to be counted from its status.
        status_code = resp.status_code if isinstance(resp, Response) else None
        error_type = "http_5xx" if status_code is not None and status_code >= 500 else None
        if status_code is not None and error_type is not None:
            self._report_upstream_response(model, status_code)

        if isinstance(resp, StreamingResponse):

            async def create_generator() -> AsyncGenerator[Any]:
                """Add usage for response."""
                stream_error_type = error_type
                chunks = list[bytes]()
                try:
                    async for chunk in resp.body_iterator:
                        if log_payload and isinstance(chunk, bytes):
                            chunks.append(chunk)
                        yield chunk
                except Exception as e:
                    # A client disconnect arrives as CancelledError/GeneratorExit (not Exception) and stays a success.
                    stream_error_type = stream_error_type or _classify_error(e)
                    self._report_upstream_exception(model, e, already_reported=error_type is not None)
                    raise
                finally:
                    if log_payload:
                        logger.info(f"DUMP RESPONSE PAYLOAD {b''.join(chunks).decode('utf-8')}")
                    self._record_request_metrics(model, start_time, stream_error_type)

            return StreamingResponse(create_generator(), media_type=resp.media_type, status_code=resp.status_code, headers=resp.headers)  # pyright: ignore[reportReturnType]
        self._record_request_metrics(model, start_time, error_type)
        return resp

    def _record_request_metrics(self, model: RegisteredModel[Any], start_time: float, error_type: str | None) -> None:
        """Record a finished request to a local model; `error_type` is None for a success."""
        duration = time.perf_counter() - start_time
        self.metrics_registry.request_duration.labels(model_name=model.name, model_type=model.type).observe(duration)
        status = "success" if error_type is None else "error"
        self.metrics_registry.request_total.labels(model_name=model.name, model_type=model.type, status=status).inc()
        if error_type is not None:
            self.metrics_registry.request_errors.labels(model_name=model.name, error_type=error_type).inc()
        self.metrics_registry.requests_in_flight.labels(model_type=model.type).dec()
        self._remove_usage(model)

    @staticmethod
    def _upstream_tags(model: RegisteredModel[Any], error_type: str, status_code: int | None = None) -> dict[str, str]:
        """Sentry tags of a failed local model request."""
        tags = {"service": model.owned_by, "model": model.name, "model_type": model.props.type, "error_type": error_type}
        if status_code is not None:
            tags["upstream_status"] = str(status_code)
        return tags

    @staticmethod
    def _capture_upstream_failure(
        model: RegisteredModel[Any], exc: BaseException, tags: dict[str, str], fingerprint: list[str] | None = None
    ) -> None:
        """Log and report a failed local model request under a new error ID."""
        error_id = str(uuid.uuid4())
        logger.warning(
            "Request to model %r of service %s failed: HTTP %s (error_id=%s)",
            model.name,
            model.owned_by,
            tags.get("upstream_status"),
            error_id,
        )
        capture_exception(exc, error_id=error_id, tags=tags, fingerprint=fingerprint)

    def _report_upstream_response(self, model: RegisteredModel[Any], status_code: int) -> None:
        """Report an upstream 5xx status to Sentry, without the response body."""
        failure = UpstreamError(f"{model.owned_by} upstream returned HTTP {status_code}")
        tags = self._upstream_tags(model, "http_5xx", status_code)
        self._capture_upstream_failure(model, failure, tags, fingerprint=["upstream-http", model.owned_by, str(status_code)])

    def _report_upstream_exception(self, model: RegisteredModel[Any], exc: Exception, *, already_reported: bool = False) -> None:
        """Report a new 5xx `HTTPException`; otherwise tag `exc` for the catch-all handler and mark it if `already_reported`."""
        if isinstance(exc, StarletteHTTPException) and exc.status_code >= 500 and not already_reported:
            self._capture_upstream_failure(model, exc, self._upstream_tags(model, _classify_error(exc), exc.status_code))
            mark_reported(exc)
            return
        attach_tags(exc, self._upstream_tags(model, _classify_error(exc)))
        if already_reported:
            mark_reported(exc)

    def _add_usage(self, model: RegisteredModel[Any]) -> None:
        model.usage += 1
        self._refresh_usage(model)

    def _remove_usage(self, model: RegisteredModel[Any]) -> None:
        model.usage -= 1
        self._refresh_usage(model)

    def _refresh_usage(self, model: RegisteredModel[Any]) -> None:
        logger.debug(f"Model usage origin={model.origin} id={model.id} name={model.name} usage={model.usage}")
        self.parent_infra.send_usage(UsageChangeRequest(id=model.id, usage=model.usage))

    async def test_model(self, registration_id: RegistrationId) -> JsonSerializable:
        """Test model of given id."""
        entry = self.registry.get(registration_id, None)
        if not entry:
            raise HTTPException(400, "Model not installed")
        return await self.model_tester.test_model(entry)


def _classify_error(e: Exception) -> str:
    """Classify an exception into an error type for metrics."""
    if isinstance(e, asyncio.TimeoutError):
        return "timeout"
    if isinstance(e, aiohttp.ClientError):
        return "connection_error"
    if isinstance(e, HttpClientError):
        return "http_error"
    return "model_error"


async def post_json(
    data: BaseModel, options: ProxyOptions, request: Request | None = None, timeout: ClientTimeout | None = None
) -> StreamingResponse:
    """Make HTTP POST request sending data as JSON."""
    raw = data.model_dump(exclude_none=True)
    if not options.forward_extra_fields:
        extra_fields = (
            CHAT_COMPLETION_EXTRA_FIELDS
            if isinstance(data, ChatCompletionRequest)
            else COMPLETION_LEGACY_EXTRA_FIELDS
            if isinstance(data, CompletionLegacyRequest)
            else frozenset()
        )
        for field in extra_fields:
            raw.pop(field, None)
    if options.remove_model:
        del raw["model"]
    if options.rewrite_model_to:
        raw["model"] = options.rewrite_model_to
    return (
        await make_http_request(
            url=options.url,
            method="POST",
            data=JsonPayload(raw),
            headers=await options.get_request_headers(request),
            timeout=timeout,
        )
    ).as_streaming_response(options.allowed_response_headers)


async def post_json_responses(
    data: ResponsesRequest,
    options: ProxyOptions,
    store: ResponseItemStore,
    request: Request | None = None,
    timeout: ClientTimeout | None = None,
) -> StreamingResponse:
    """Make HTTP POST request for /v1/responses, caching stateful output items for later item_reference resolution."""
    raw = data.model_dump(exclude_none=True)
    if options.remove_model:
        del raw["model"]
    if options.rewrite_model_to:
        raw["model"] = options.rewrite_model_to
    http_response = await make_http_request(
        url=options.url,
        method="POST",
        data=JsonPayload(raw),
        headers=await options.get_request_headers(request),
        timeout=timeout,
    )
    if data.stream or not (http_response.response.content_type or "").startswith("application/json"):
        return http_response.as_streaming_response(options.allowed_response_headers)

    body = b"".join([chunk async for chunk in http_response.content])
    try:
        parsed = json.loads(body)
        for item in parsed.get("output", []):
            if isinstance(item, dict):
                store.put(item)
    except json.JSONDecodeError:
        logger.warning("Failed to parse /v1/responses body for item_reference caching", exc_info=True)

    allowed_response_headers = options.allowed_response_headers or []
    response_headers = {k: v for k, v in dict(http_response.response.headers).items() if k in allowed_response_headers}

    async def replay() -> AsyncGenerator[bytes]:
        yield body

    return StreamingResponse(
        replay(),
        media_type=http_response.response.content_type,
        status_code=http_response.response.status,
        headers=response_headers,
    )


def _sglang_rerank_response_to_cohere(raw_body: bytes) -> bytes:
    """Rewrite SGLang's native `/v1/rerank` array response into the Cohere-style `{"results": [...]}` shape."""
    parsed = json.loads(raw_body)
    results = [
        {
            "index": item["index"],
            "relevance_score": item["score"],
            "document": {"text": item["document"]} if item.get("document") is not None else None,
        }
        for item in parsed
    ]
    return json.dumps({"results": results}).encode()


async def post_json_rerank_sglang(
    data: RerankRequest, options: ProxyOptions, request: Request | None = None, timeout: ClientTimeout | None = None
) -> StreamingResponse:
    """Make HTTP POST request to SGLang's `/v1/rerank`, rewriting its native array response into Cohere-style shape."""
    raw = data.model_dump(exclude_none=True)
    if options.remove_model:
        del raw["model"]
    if options.rewrite_model_to:
        raw["model"] = options.rewrite_model_to
    http_response = await make_http_request(
        url=options.url,
        method="POST",
        data=JsonPayload(raw),
        headers=await options.get_request_headers(request),
        timeout=timeout,
    )
    if not (http_response.response.content_type or "").startswith("application/json"):
        return http_response.as_streaming_response(options.allowed_response_headers)

    body = b"".join([chunk async for chunk in http_response.content])
    try:
        body = _sglang_rerank_response_to_cohere(body)
    except (json.JSONDecodeError, KeyError, TypeError):
        logger.warning("Failed to normalize SGLang rerank response into Cohere shape", exc_info=True)

    allowed_response_headers = options.allowed_response_headers or []
    response_headers = {k: v for k, v in dict(http_response.response.headers).items() if k in allowed_response_headers}

    async def replay() -> AsyncGenerator[bytes]:
        yield body

    return StreamingResponse(
        replay(),
        media_type=http_response.response.content_type,
        status_code=http_response.response.status,
        headers=response_headers,
    )


async def _unwrap_json_text(response: StarletteResponse) -> StarletteResponse:
    """Turn a JSON `{"text": ...}` transcription reply into plain text.

    Some backends (e.g. vLLM) answer `response_format=text` with JSON. Errors and replies that are
    not such JSON are returned unchanged.
    """
    if not isinstance(response, StreamingResponse) or not 200 <= response.status_code < 300 or response.media_type != "application/json":
        return response
    raw = b"".join([chunk if isinstance(chunk, bytes) else str(chunk).encode() async for chunk in response.body_iterator])
    try:
        data = json.loads(raw)
    except ValueError:
        data = None
    if isinstance(data, dict) and isinstance(data.get("text"), str):
        headers = {k: v for k, v in response.headers.items() if k.lower() not in ("content-length", "content-type")}
        return PlainTextResponse(data["text"], status_code=response.status_code, headers=headers)

    async def replay() -> AsyncGenerator[bytes]:
        yield raw

    headers = {k: v for k, v in response.headers.items() if k.lower() != "content-length"}
    return StreamingResponse(replay(), media_type=response.media_type, status_code=response.status_code, headers=headers)


async def post_form(
    data: FormSerializable, options: ProxyOptions, request: Request | None = None, timeout: ClientTimeout | None = None
) -> StreamingResponse:
    """Make HTTP POST request sending data as form."""
    return (
        await make_http_request(
            url=options.url,
            method="POST",
            data=await data.to_form(options.remove_model, options.rewrite_model_to),
            headers=await options.get_request_headers(request),
            timeout=timeout,
        )
    ).as_streaming_response(options.allowed_response_headers)
