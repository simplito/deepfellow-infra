# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

import functools
import logging
from collections.abc import Generator
from typing import TYPE_CHECKING, Any, cast
from unittest import mock

import pytest
import pytest_asyncio
import sentry_sdk
from fastapi.responses import Response
from opentelemetry import trace
from opentelemetry.trace import NonRecordingSpan, SpanContext
from prometheus_client import CollectorRegistry
from pydantic import SecretStr
from sentry_sdk.envelope import Envelope
from sentry_sdk.integrations.atexit import AtexitIntegration
from sentry_sdk.integrations.dedupe import DedupeIntegration
from sentry_sdk.transport import Transport

from server.config import AppSettings
from server.dynamic_config import DynamicSettings
from server.endpointregistry import ChatCompletionEndpoint, EndpointRegistry, RegistrationOptions
from server.metrics_registry import MetricsRegistry
from server.models.api import ModelProps
from server.models.services import InstallServiceIn
from server.services.openai_service import OpenAIService
from server.services.remote_service import DefaultRemoteServiceOptions, InstalledInfo
from server.services_manager import ServicesManager
from server.utils.sentry import (
    SDK_LOGGER_NAME,
    UpstreamError,
    _forward_sdk_warnings,  # pyright: ignore[reportPrivateUsage]
    _SdkWarningForwarder,  # pyright: ignore[reportPrivateUsage]
    attach_tags,
    attached_tags,
    capture_exception,
    is_reported,
    mark_reported,
    parse_dsn,
    sentry,
)
from server.utils.version import get_infra_version

if TYPE_CHECKING:
    from sentry_sdk.types import Event

DSN = "https://publickey@sentry.example.com/42"


class RecordingTransport(Transport):
    def __init__(self) -> None:
        super().__init__()
        self.envelopes: list[Envelope] = []

    def capture_envelope(self, envelope: Envelope) -> None:
        self.envelopes.append(envelope)

    def events(self) -> list[dict[str, Any]]:
        return [cast("dict[str, Any]", event) for envelope in self.envelopes if (event := envelope.get_event()) is not None]

    def serialized(self) -> str:
        return "".join(envelope.serialize().decode() for envelope in self.envelopes)


@pytest.fixture(autouse=True)
def reset_sentry() -> Generator[None]:
    yield
    sentry_sdk.get_global_scope().set_client(None)
    sentry.config = None
    sentry._applied = None  # pyright: ignore[reportPrivateUsage]
    sentry._client_enabled = False  # pyright: ignore[reportPrivateUsage]
    sentry._secret_sources.clear()  # pyright: ignore[reportPrivateUsage]


@pytest.fixture
def config(monkeypatch: pytest.MonkeyPatch) -> AppSettings:
    # Set here, not relying on .test.env: it isn't in the repo, so CI's image build runs the tests without it.
    monkeypatch.setenv("DF_INFRA_ADMIN_API_KEY", "admin-key")
    settings = AppSettings()  # pyright: ignore[reportCallIssue]
    settings.name = "node-a"
    settings.sentry_enabled = True
    settings.sentry_dsn = SecretStr(DSN)
    settings.sentry_environment = "staging"
    return settings


@pytest.fixture
def transport() -> RecordingTransport:
    return RecordingTransport()


@pytest_asyncio.fixture
async def active_sentry(config: AppSettings, transport: RecordingTransport) -> RecordingTransport:
    with mock.patch("server.utils.sentry.sentry_sdk.init", functools.partial(sentry_sdk.init, transport=transport)):
        await sentry.reconfigure(config)
    return transport


@pytest.fixture
def span_context() -> SpanContext:
    return SpanContext(trace_id=0x0AF7651916CD43DD8448EB211C80319C, span_id=0x00F067AA0BA902B7, is_remote=False)


def _fail(message: str) -> None:
    raise RuntimeError(message)


def _raise_and_capture(message: str, **kwargs: Any) -> None:
    try:
        _fail(message)
    except RuntimeError as e:
        capture_exception(e, error_id="error-1", **kwargs)


# --- parse_dsn ---


def test_parse_dsn_returns_host_and_project():
    dsn = parse_dsn(DSN)

    assert dsn.host == "sentry.example.com"
    assert dsn.project_id == "42"


@pytest.mark.parametrize(
    "dsn",
    ["not-a-dsn", "https://sentry.example.com/42", "https://key@sentry.example.com:notaport/42", "https://key@[::1/42"],
)
def test_parse_dsn_raises_value_error_on_malformed_dsn(dsn: str):
    with pytest.raises(ValueError):  # noqa: PT011
        parse_dsn(dsn)


# --- reconfigure ---


@pytest.mark.asyncio
@mock.patch("server.utils.sentry.sentry_sdk.init")
async def test_reconfigure_initializes_client_with_privacy_options(mock_init: mock.MagicMock, config: AppSettings):
    await sentry.reconfigure(config)

    assert mock_init.call_count == 1
    assert mock_init.call_args == mock.call(
        dsn=DSN,
        environment="staging",
        release=get_infra_version(),
        default_integrations=False,
        integrations=mock.ANY,
        include_local_variables=False,
        send_default_pii=False,
        before_send=sentry.before_send,
        debug=False,
        keep_alive=False,
        spotlight=False,
    )
    assert [type(integration) for integration in mock_init.call_args.kwargs["integrations"]] == [DedupeIntegration, AtexitIntegration]


@pytest.mark.asyncio
async def test_reconfigure_enables_only_dedupe_and_atexit_integrations(active_sentry: RecordingTransport):
    integrations = sentry_sdk.get_client().integrations

    assert sorted(integrations) == ["atexit", "dedupe"]


@pytest.mark.asyncio
async def test_reconfigure_ignores_sentry_debug_environment_variables(
    monkeypatch: pytest.MonkeyPatch, config: AppSettings, transport: RecordingTransport
):
    monkeypatch.setenv("SENTRY_DEBUG", "true")
    monkeypatch.setenv("SENTRY_KEEP_ALIVE", "true")
    monkeypatch.setenv("SENTRY_SPOTLIGHT", "true")

    with mock.patch("server.utils.sentry.sentry_sdk.init", functools.partial(sentry_sdk.init, transport=transport)):
        await sentry.reconfigure(config)

    client = sentry_sdk.get_client()
    assert client.options["debug"] is False
    assert client.options["keep_alive"] is False
    assert getattr(client, "spotlight", None) is None


@pytest.mark.asyncio
async def test_reconfigure_reports_active_only_while_client_is_set_up(active_sentry: RecordingTransport, config: AppSettings):
    was_active = sentry.is_active

    await sentry.shutdown()

    assert was_active is True
    assert sentry.is_active is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("enabled", "dsn"),
    [
        (False, DSN),
        (True, ""),
    ],
)
@mock.patch("server.utils.sentry.sentry_sdk.init")
async def test_reconfigure_skips_init_when_inactive(mock_init: mock.MagicMock, config: AppSettings, enabled: bool, dsn: str):
    config.sentry_enabled = enabled
    config.sentry_dsn = SecretStr(dsn)

    await sentry.reconfigure(config)

    assert mock_init.call_count == 0
    assert sentry.is_active is False
    assert sentry.needs_reconfigure(config) is False


@pytest.mark.asyncio
@mock.patch("server.utils.sentry.sentry_sdk.init")
async def test_reconfigure_defaults_environment_to_production(mock_init: mock.MagicMock, config: AppSettings):
    config.sentry_environment = ""

    await sentry.reconfigure(config)

    assert mock_init.call_args.kwargs["environment"] == "production"


@pytest.mark.asyncio
@mock.patch("server.utils.sentry.sentry_sdk.init")
@mock.patch("server.utils.sentry.sentry_sdk.get_client")
async def test_reconfigure_detaches_and_closes_previous_client(
    mock_get_client: mock.MagicMock, mock_init: mock.MagicMock, config: AppSettings
):
    previous_client = mock_get_client.return_value
    previous_client.is_active.return_value = True
    _mock_manager = mock.MagicMock()
    _mock_manager.attach_mock(previous_client.close, "close")
    _mock_manager.attach_mock(mock_init, "init")

    await sentry.reconfigure(config)

    assert previous_client.close.call_count == 1
    assert _mock_manager.mock_calls[0] == mock.call.close()
    assert sentry_sdk.get_global_scope().client.is_active() is False


@pytest.mark.asyncio
@mock.patch("server.utils.sentry.logger")
@mock.patch("server.utils.sentry.sentry_sdk.init")
async def test_reconfigure_turns_sentry_off_on_invalid_dsn_without_retrying(
    mock_init: mock.MagicMock, mock_logger: mock.MagicMock, config: AppSettings
):
    config.sentry_dsn = SecretStr("not-a-dsn")

    await sentry.reconfigure(config)

    assert mock_init.call_count == 0
    assert mock_logger.warning.call_count == 1
    assert "sentry_dsn is invalid" in mock_logger.warning.call_args.args[0]
    assert sentry.needs_reconfigure(config) is False
    assert sentry.is_active is False
    assert sentry_sdk.get_client().is_active() is False


@pytest.mark.asyncio
@mock.patch("server.utils.sentry.sentry_sdk.init")
async def test_reconfigure_leaves_change_pending_when_init_fails(mock_init: mock.MagicMock, config: AppSettings):
    mock_init.side_effect = RuntimeError("init failed")

    with pytest.raises(RuntimeError):
        await sentry.reconfigure(config)

    assert sentry.needs_reconfigure(config) is True


@pytest.mark.asyncio
@mock.patch("server.utils.sentry.sentry_sdk.init")
async def test_reconfigure_leaves_previous_config_pending_when_init_fails(mock_init: mock.MagicMock, config: AppSettings):
    await sentry.reconfigure(config)
    previous = DynamicSettings(sentry_enabled=True, sentry_dsn=SecretStr(DSN), sentry_environment="staging")
    config.sentry_environment = "other"
    mock_init.side_effect = RuntimeError("init failed")

    with pytest.raises(RuntimeError):
        await sentry.reconfigure(config)

    assert sentry.needs_reconfigure(previous) is True
    assert sentry.is_active is False


@pytest.mark.asyncio
@mock.patch("server.utils.sentry.logger")
@mock.patch("server.utils.sentry.sentry_sdk.init")
async def test_reconfigure_logs_dsn_host_and_project_without_key(
    mock_init: mock.MagicMock, mock_logger: mock.MagicMock, config: AppSettings
):
    await sentry.reconfigure(config)

    assert mock_logger.info.call_count == 1
    assert mock_logger.info.call_args.args[1:] == ("sentry.example.com", "42", "staging")
    assert "publickey" not in str(mock_logger.info.call_args)


@pytest.mark.asyncio
async def test_reconfigure_off_closes_active_client(active_sentry: RecordingTransport, config: AppSettings):
    config.sentry_enabled = False

    await sentry.reconfigure(config)

    assert sentry_sdk.get_client().is_active() is False
    assert sentry.is_active is False


@pytest.mark.asyncio
async def test_shutdown_closes_client(active_sentry: RecordingTransport):
    await sentry.shutdown()

    assert sentry_sdk.get_client().is_active() is False
    assert sentry.is_active is False


def test_needs_reconfigure_is_true_before_first_apply(config: AppSettings):
    config.sentry_enabled = False

    result = sentry.needs_reconfigure(config)

    assert result is True


@pytest.mark.asyncio
@mock.patch("server.utils.sentry.sentry_sdk.init")
async def test_needs_reconfigure_ignores_environment_change_while_off(mock_init: mock.MagicMock, config: AppSettings):
    config.sentry_enabled = False
    await sentry.reconfigure(config)

    result = sentry.needs_reconfigure(DynamicSettings(sentry_enabled=False, sentry_dsn=SecretStr(DSN), sentry_environment="other"))

    assert result is False


@pytest.mark.asyncio
@mock.patch("server.utils.sentry.sentry_sdk.init")
async def test_needs_reconfigure_detects_sentry_setting_changes(mock_init: mock.MagicMock, config: AppSettings):
    await sentry.reconfigure(config)

    result = sentry.needs_reconfigure(DynamicSettings(sentry_enabled=True, sentry_dsn=SecretStr(DSN), sentry_environment="other"))

    assert result is True


# --- capture_exception ---


@pytest.mark.asyncio
async def test_capture_exception_sends_unhandled_event_with_correlation_tags(active_sentry: RecordingTransport, span_context: SpanContext):
    _raise_and_capture("boom", handled=False, span_context=span_context)

    events = active_sentry.events()
    assert len(events) == 1
    event = events[0]
    assert event["exception"]["values"][0]["type"] == "RuntimeError"
    assert event["exception"]["values"][0]["mechanism"] == {"type": "generic", "handled": False}
    assert event["tags"] == {
        "error_id": "error-1",
        "trace_id": "0af7651916cd43dd8448eb211c80319c",
        "span_id": "00f067aa0ba902b7",
        "infra_name": "node-a",
    }
    assert event["release"] == get_infra_version()
    assert event["environment"] == "staging"
    assert event["server_name"] == "node-a"


@pytest.mark.asyncio
async def test_capture_exception_marks_event_handled_by_default(active_sentry: RecordingTransport):
    _raise_and_capture("boom")

    assert active_sentry.events()[0]["exception"]["values"][0]["mechanism"]["handled"] is True


@pytest.mark.asyncio
async def test_capture_exception_uses_current_span_without_span_context(active_sentry: RecordingTransport, span_context: SpanContext):
    with trace.use_span(NonRecordingSpan(span_context)):
        _raise_and_capture("boom")

    assert active_sentry.events()[0]["tags"]["trace_id"] == "0af7651916cd43dd8448eb211c80319c"


@pytest.mark.asyncio
async def test_capture_exception_omits_trace_tags_without_valid_span(active_sentry: RecordingTransport):
    _raise_and_capture("boom")

    assert active_sentry.events()[0]["tags"] == {"error_id": "error-1", "infra_name": "node-a"}


@pytest.mark.asyncio
async def test_capture_exception_adds_caller_tags(active_sentry: RecordingTransport):
    _raise_and_capture("boom", tags={"task": "install"})

    assert active_sentry.events()[0]["tags"]["task"] == "install"


@pytest.mark.asyncio
async def test_capture_exception_sends_no_server_name_when_name_empty(active_sentry: RecordingTransport, config: AppSettings):
    config.name = ""

    _raise_and_capture("boom")

    event = active_sentry.events()[0]
    assert "server_name" not in event
    assert "infra_name" not in event["tags"]


@pytest.mark.asyncio
async def test_capture_exception_uses_name_changed_at_runtime(active_sentry: RecordingTransport, config: AppSettings):
    config.name = "node-b"

    _raise_and_capture("boom")

    event = active_sentry.events()[0]
    assert event["server_name"] == "node-b"
    assert event["tags"]["infra_name"] == "node-b"


@pytest.mark.asyncio
async def test_capture_exception_sends_no_local_variables(active_sentry: RecordingTransport):
    request_body = "user prompt"

    _raise_and_capture(f"boom {len(request_body)}")

    frames = active_sentry.events()[0]["exception"]["values"][0]["stacktrace"]["frames"]
    assert frames
    assert all("vars" not in frame for frame in frames)
    assert "user prompt" not in active_sentry.serialized()


@pytest.mark.asyncio
async def test_capture_exception_sends_nothing_when_inactive(active_sentry: RecordingTransport, config: AppSettings):
    config.sentry_enabled = False
    await sentry.reconfigure(config)

    _raise_and_capture("boom")

    assert active_sentry.envelopes == []


@pytest.mark.asyncio
@mock.patch("server.utils.sentry.logger")
@mock.patch("server.utils.sentry.sentry_sdk.capture_event")
async def test_capture_exception_logs_and_swallows_errors(
    mock_capture_event: mock.MagicMock, mock_logger: mock.MagicMock, active_sentry: RecordingTransport
):
    mock_capture_event.side_effect = RuntimeError("transport down")

    _raise_and_capture("boom")

    assert mock_capture_event.call_count == 1
    assert mock_logger.exception.call_count == 1
    assert mock_logger.exception.call_args == mock.call("Failed to report exception to Sentry (error_id=%s)", "error-1")


@pytest.mark.asyncio
async def test_capture_exception_sets_fingerprint(active_sentry: RecordingTransport):
    capture_exception(UpstreamError("vllm upstream returned HTTP 500"), error_id="error-1", fingerprint=["upstream-http", "vllm", "500"])

    event = active_sentry.events()[0]
    assert event["fingerprint"] == ["upstream-http", "vllm", "500"]
    assert event["exception"]["values"][0]["type"] == "UpstreamError"
    assert event["exception"]["values"][0]["value"] == "vllm upstream returned HTTP 500"


@pytest.mark.asyncio
async def test_capture_exception_leaves_default_fingerprint(active_sentry: RecordingTransport):
    _raise_and_capture("boom")

    assert "fingerprint" not in active_sentry.events()[0]


@pytest.mark.asyncio
async def test_before_send_keeps_fingerprint_with_short_secret(active_sentry: RecordingTransport):
    sentry.register_secret_source(lambda: ["vllm"])

    capture_exception(UpstreamError("boom"), error_id="error-1", fingerprint=["upstream-http", "vllm", "500"])

    assert active_sentry.events()[0]["fingerprint"] == ["upstream-http", "vllm", "500"]


def test_attached_tags_empty_for_untouched_exception():
    assert attached_tags(RuntimeError("boom")) == {}


def test_attach_tags_round_trip():
    exc = RuntimeError("boom")

    attach_tags(exc, {"service": "vllm", "model": "m1"})

    assert attached_tags(exc) == {"service": "vllm", "model": "m1"}


def test_attach_tags_merges_over_previous_tags():
    exc = RuntimeError("boom")
    attach_tags(exc, {"service": "vllm", "error_type": "http_5xx"})

    attach_tags(exc, {"error_type": "connection_error", "model": "m1"})

    assert attached_tags(exc) == {"service": "vllm", "error_type": "connection_error", "model": "m1"}


def test_attached_tags_returns_copy():
    exc = RuntimeError("boom")
    attach_tags(exc, {"service": "vllm"})

    attached_tags(exc)["service"] = "changed"

    assert attached_tags(exc) == {"service": "vllm"}


def test_attached_tags_include_tags_of_cause():
    cause = RuntimeError("upstream")
    attach_tags(cause, {"service": "vllm", "error_type": "http_5xx"})
    wrapper = RuntimeError("wrapped")
    wrapper.__cause__ = cause
    attach_tags(wrapper, {"error_type": "model_error"})

    assert attached_tags(wrapper) == {"service": "vllm", "error_type": "model_error"}


def test_attached_tags_ignore_implicit_context():
    context = RuntimeError("unrelated")
    attach_tags(context, {"service": "vllm"})
    exc = RuntimeError("later")
    exc.__context__ = context

    assert attached_tags(exc) == {}


def test_cause_chain_stops_on_cycle():
    first = RuntimeError("first")
    second = RuntimeError("second")
    first.__cause__ = second
    second.__cause__ = first
    mark_reported(second)

    assert is_reported(first) is True
    assert attached_tags(first) == {}


def test_is_reported_true_when_cause_reported():
    cause = RuntimeError("upstream")
    mark_reported(cause)
    wrapper = RuntimeError("wrapped")
    wrapper.__cause__ = cause

    assert is_reported(wrapper) is True


def test_is_reported_false_for_untouched_exception():
    assert is_reported(RuntimeError("boom")) is False


def test_mark_reported_sets_flag():
    exc = RuntimeError("boom")

    mark_reported(exc)

    assert is_reported(exc) is True


# --- secret scrubbing ---


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field_name", "value", "secret"),
    [
        ("hugging_face_token", SecretStr("hf_secret_token_value"), "hf_secret_token_value"),
        ("mesh_key", SecretStr("mesh-secret-value"), "mesh-secret-value"),
        ("docker_hub_token", "dckr_pat_secret_value", "dckr_pat_secret_value"),
        ("sentry_dsn", SecretStr(DSN), DSN),
    ],
)
async def test_before_send_scrubs_config_secret_from_event(
    active_sentry: RecordingTransport, config: AppSettings, field_name: str, value: SecretStr | str, secret: str
):
    setattr(config, field_name, value)

    _raise_and_capture(f"request failed with {secret} in it")

    event = active_sentry.events()[0]
    assert event["exception"]["values"][0]["value"] == "request failed with [Filtered] in it"
    assert secret not in active_sentry.serialized()


@pytest.mark.asyncio
async def test_before_send_scrubs_values_from_registered_secret_source(active_sentry: RecordingTransport):
    sentry.register_secret_source(lambda: ["sk-provider-key-value"])

    _raise_and_capture("provider rejected sk-provider-key-value")

    assert active_sentry.events()[0]["exception"]["values"][0]["value"] == "provider rejected [Filtered]"


@pytest.mark.asyncio
async def test_before_send_scrubs_installed_provider_api_key(active_sentry: RecordingTransport):
    manager = ServicesManager()
    openai = OpenAIService(
        config=mock.MagicMock(),
        endpoint_registry=mock.MagicMock(),
        service_provider=mock.MagicMock(),
        model_downloader=mock.MagicMock(),
        docker_service=mock.MagicMock(),
        hardware=mock.MagicMock(gpus=[]),
        services_manager=manager,
    )
    openai.instances_info["default"].installed = InstalledInfo(
        models={},
        options=InstallServiceIn(spec={"api_url": "https://api.openai.com", "api_key": "sk-proj-abc123"}),
        parsed_options=DefaultRemoteServiceOptions(api_url="https://api.openai.com", api_key="sk-proj-abc123"),
    )
    manager.register_service(openai)
    sentry.register_secret_source(manager.secret_values)

    _raise_and_capture("GET https://api.openai.com/v1/models?key=sk-proj-abc123 failed")

    assert active_sentry.events()[0]["exception"]["values"][0]["value"] == "GET https://api.openai.com/v1/models?key=[Filtered] failed"
    assert "sk-proj-abc123" not in active_sentry.serialized()


@pytest.mark.asyncio
async def test_upstream_5xx_event_carries_no_response_body(active_sentry: RecordingTransport):
    registry = EndpointRegistry(
        config=mock.MagicMock(),
        parent_infra=mock.MagicMock(),
        model_tester=mock.MagicMock(),
        metrics_registry=MetricsRegistry(registry=CollectorRegistry()),
    )
    rid = registry.register_chat_completion(
        "qwen3",
        ModelProps(private=False, type="llm", endpoints=["chat"]),
        ChatCompletionEndpoint(),
        RegistrationOptions(origin="local", owned_by="vllm"),
    )
    upstream = Response(content=b"prompt too long: tell me the launch codes", status_code=500)

    result = await registry.with_usage(registry.registry[rid].registered_model, mock.AsyncMock(return_value=upstream))

    assert result.body == b"prompt too long: tell me the launch codes"
    event = active_sentry.events()[0]
    assert event["exception"]["values"][0]["value"] == "vllm upstream returned HTTP 500"
    assert event["tags"]["upstream_status"] == "500"
    assert "launch codes" not in active_sentry.serialized()


@pytest.mark.asyncio
async def test_shutdown_drops_registered_secret_sources(active_sentry: RecordingTransport):
    sentry.register_secret_source(lambda: ["sk-provider-key-value"])

    await sentry.shutdown()

    assert "sk-provider-key-value" not in sentry._secret_values()  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_before_send_scrubs_secret_changed_after_init(active_sentry: RecordingTransport, config: AppSettings):
    config.civitai_token = SecretStr("civitai-new-token")

    _raise_and_capture("download failed for civitai-new-token")

    assert "civitai-new-token" not in active_sentry.serialized()


@pytest.mark.asyncio
async def test_before_send_replaces_secret_containing_another_whole(active_sentry: RecordingTransport, config: AppSettings):
    config.hugging_face_token = SecretStr("abcdef")
    config.civitai_token = SecretStr("abcdefghij")

    _raise_and_capture("token abcdefghij")

    assert active_sentry.events()[0]["exception"]["values"][0]["value"] == "token [Filtered]"


@pytest.mark.asyncio
async def test_before_send_keeps_event_structure_with_short_secret(active_sentry: RecordingTransport, config: AppSettings):
    config.civitai_token = SecretStr("ion")

    _raise_and_capture("failed at ion")

    event = active_sentry.events()[0]
    assert event["exception"]["values"][0]["value"] == "failed at [Filtered]"
    assert event["environment"] == "staging"
    assert event["level"] == "error"
    assert event["exception"]["values"][0]["type"] == "RuntimeError"


@pytest.mark.asyncio
async def test_before_send_keeps_stack_frame_source_fields_with_short_secret(active_sentry: RecordingTransport, config: AppSettings):
    config.civitai_token = SecretStr("unit")

    _raise_and_capture("failed at unit")

    event = active_sentry.events()[0]
    frame = event["exception"]["values"][0]["stacktrace"]["frames"][-1]
    assert event["exception"]["values"][0]["value"] == "failed at [Filtered]"
    assert frame["module"] == "tests.unit.utils.test_sentry"
    assert frame["filename"].endswith("tests/unit/utils/test_sentry.py")
    assert frame["function"] == "_fail"


def test_before_send_keeps_trace_context_ids_with_short_secret(config: AppSettings):
    config.civitai_token = SecretStr("1234")
    sentry.config = config
    trace_context = {"trace_id": "abcd1234abcd1234abcd1234abcd1234", "span_id": "1234abcd1234abcd", "parent_span_id": "abcd12341234abcd"}
    event = cast("Event", {"message": "failed at 1234", "contexts": {"trace": dict(trace_context)}})

    result = sentry.before_send(event, {})

    assert result == {
        "message": "failed at [Filtered]",
        "contexts": {"trace": trace_context},
        "server_name": "node-a",
        "tags": {"infra_name": "node-a"},
    }


def test_before_send_without_config_drops_server_name_and_scrubs_nothing():
    event = cast("Event", {"server_name": "container-host", "message": "plain message"})

    result = sentry.before_send(event, {})

    assert result == {"message": "plain message"}


@pytest.mark.asyncio
@mock.patch("server.utils.sentry.logger")
@mock.patch("server.utils.sentry._scrub")
async def test_before_send_drops_event_when_scrubbing_fails(
    mock_scrub: mock.MagicMock, mock_logger: mock.MagicMock, active_sentry: RecordingTransport
):
    mock_scrub.side_effect = RuntimeError("scrub failed")

    _raise_and_capture("boom")

    assert active_sentry.envelopes == []
    assert mock_logger.warning.call_count == 1


# --- SDK warning forwarding ---


@mock.patch("server.utils.sentry.logger")
def test_forward_sdk_warnings_relogs_sdk_warning_on_infra_logger(mock_logger: mock.MagicMock):
    _forward_sdk_warnings()

    logging.getLogger(SDK_LOGGER_NAME).warning("Rate-limited via 429")

    assert mock_logger.log.call_count == 1
    assert mock_logger.log.call_args == mock.call(logging.WARNING, "Sentry SDK: %s", "Rate-limited via 429", exc_info=None)


@mock.patch("server.utils.sentry.logger")
def test_forward_sdk_warnings_ignores_sdk_debug_records(mock_logger: mock.MagicMock):
    _forward_sdk_warnings()

    logging.getLogger(SDK_LOGGER_NAME).debug("Sending envelope")

    assert mock_logger.log.call_count == 0


def test_forward_sdk_warnings_installs_one_handler():
    _forward_sdk_warnings()
    _forward_sdk_warnings()

    handlers = logging.getLogger(SDK_LOGGER_NAME).handlers
    assert len([handler for handler in handlers if isinstance(handler, _SdkWarningForwarder)]) == 1
