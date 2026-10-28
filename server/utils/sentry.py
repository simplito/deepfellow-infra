# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Sentry error tracking: (re)configure the SDK client, report exceptions to it and scrub secrets from its events."""

import asyncio
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import sentry_sdk
from opentelemetry import trace
from opentelemetry.trace import SpanContext
from pydantic import SecretStr
from sentry_sdk.integrations.atexit import AtexitIntegration
from sentry_sdk.integrations.dedupe import DedupeIntegration
from sentry_sdk.types import Event, Hint
from sentry_sdk.utils import Dsn, event_from_exception

from server.config import AppSettings
from server.dynamic_config import DynamicSettings
from server.utils.logger import logger
from server.utils.version import get_infra_version

# Used when `sentry_environment` is empty. Passed explicitly because `sentry_sdk.init` would otherwise
# fall back to the `SENTRY_ENVIRONMENT` environment variable, and Sentry is configured only through config.json.
DEFAULT_ENVIRONMENT = "production"

FILTERED = "[Filtered]"

# The SDK reports delivery failures (network errors, 401/403 for a revoked key, 429 rate limits)
# only to this logger.
SDK_LOGGER_NAME = "sentry_sdk.errors"

SecretSource = Callable[[], Iterable[str]]

_ATTACHED_TAGS_ATTR = "_df_sentry_tags"
_REPORTED_ATTR = "_df_sentry_reported"


class UpstreamError(Exception):
    """Stands in for a model failure that raised nothing (upstream 5xx, dead container); never raised."""


def attach_tags(exc: BaseException, tags: Mapping[str, str]) -> None:
    """Attach Sentry tags to `exc` for a later report."""
    setattr(exc, _ATTACHED_TAGS_ATTR, {**attached_tags(exc), **tags})


def _cause_chain(exc: BaseException) -> list[BaseException]:
    """Return `exc` and the exceptions it was raised from, outermost first."""
    chain: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and all(current is not seen for seen in chain):
        chain.append(current)
        current = current.__cause__
    return chain


def attached_tags(exc: BaseException) -> dict[str, str]:
    """Return the Sentry tags attached to `exc` or to an exception it was raised from."""
    tags: dict[str, str] = {}
    for link in reversed(_cause_chain(exc)):
        tags.update(cast("dict[str, str]", getattr(link, _ATTACHED_TAGS_ATTR, {})))
    return tags


def mark_reported(exc: BaseException) -> None:
    """Mark `exc` as already reported to Sentry."""
    setattr(exc, _REPORTED_ATTR, True)


def is_reported(exc: BaseException) -> bool:
    """Whether `exc`, or an exception it was raised from, was already reported to Sentry."""
    return any(getattr(link, _REPORTED_ATTR, False) is True for link in _cause_chain(exc))


@dataclass(frozen=True)
class _Target:
    """The Sentry configuration to apply: off, or on with a DSN and an environment."""

    active: bool
    dsn: str = ""
    environment: str = ""


_OFF = _Target(active=False)


def parse_dsn(dsn: str) -> Dsn:
    """Parse a Sentry DSN; the one place that decides whether a DSN is valid.

    Raises:
        ValueError: If `dsn` is not a valid Sentry DSN. That covers the SDK's `BadDsn`, a `ValueError`
            subclass, and urllib's own `ValueError` (bad port, bad IPv6 host), which `Dsn` lets escape unwrapped.
    """
    return Dsn(dsn)


def _target_for(settings: AppSettings | DynamicSettings) -> _Target:
    """Sentry is on only when enabled with a non-empty DSN; every inactive combination is the same `_OFF`."""
    dsn = settings.sentry_dsn.get_secret_value()
    if not settings.sentry_enabled or not dsn:
        return _OFF
    return _Target(active=True, dsn=dsn, environment=settings.sentry_environment or DEFAULT_ENVIRONMENT)


class _SdkWarningForwarder(logging.Handler):
    """Re-log WARNING and above from the SDK's logger through the infra logger."""

    def __init__(self) -> None:
        super().__init__(logging.WARNING)

    def emit(self, record: logging.LogRecord) -> None:
        """Log `record`'s message on the infra logger, keeping its level and traceback."""
        logger.log(record.levelno, "Sentry SDK: %s", record.getMessage(), exc_info=record.exc_info)


def _forward_sdk_warnings() -> None:
    """Make SDK delivery failures visible in the infra log; idempotent.

    On import the SDK puts a filter on its logger that drops every record unless the client has
    `debug=True`, so a handler added next to it would never see anything. The filters are moved onto
    the SDK's own stderr handler instead (keeping that one debug-only), and propagation is turned off
    so the SDK's debug records still don't reach the root logger.
    """
    sdk_logger = logging.getLogger(SDK_LOGGER_NAME)
    if any(isinstance(handler, _SdkWarningForwarder) for handler in sdk_logger.handlers):
        return
    for log_filter in list(sdk_logger.filters):
        sdk_logger.removeFilter(log_filter)
        for handler in sdk_logger.handlers:
            handler.addFilter(log_filter)
    sdk_logger.propagate = False
    sdk_logger.addHandler(_SdkWarningForwarder())


# Event fields left out of scrubbing, at any depth: they hold SDK- or infra-generated identifiers and enums
# (IDs, levels, mechanism, our correlation tags, the SDK's trace context) or come from the source code (stack-frame
# paths, modules, functions, source lines), never secrets. A short secret value occurring inside them (e.g. a 5-character
# key matching part of "production", of a module path or of a hex trace ID) would corrupt the event's structure and
# break grouping and trace linking.
_UNSCRUBBED_KEYS = frozenset(
    {
        *("event_id", "level", "platform", "timestamp", "release", "environment", "server_name", "sdk", "tags", "mechanism", "type"),
        "fingerprint",
        *("trace_id", "span_id", "parent_span_id"),
        *("filename", "abs_path", "module", "function", "context_line", "pre_context", "post_context"),
    }
)


def _scrub(value: object, secrets: list[str]) -> object:
    """Return `value` with every occurrence of each of `secrets` replaced by `FILTERED`, recursing into containers.

    Only content is scrubbed: mapping keys and the values of `_UNSCRUBBED_KEYS` are kept as they are.
    """
    if isinstance(value, str):
        for secret in secrets:
            value = value.replace(secret, FILTERED)
        return value
    if isinstance(value, Mapping):
        items = cast("Mapping[object, object]", value).items()
        return {key: item if key in _UNSCRUBBED_KEYS else _scrub(item, secrets) for key, item in items}
    if isinstance(value, list | tuple):
        scrubbed = [_scrub(item, secrets) for item in cast("Iterable[object]", value)]
        return tuple(scrubbed) if isinstance(value, tuple) else scrubbed
    return value


class SentryManager:
    """Owns the process-wide Sentry client: applies the Sentry settings and filters every event it sends.

    A module-level singleton (`sentry`), like `tracer`: the SDK client is process-global, and code
    without a request or `app` (background tasks) must be able to report through it.
    """

    def __init__(self) -> None:
        # The shared `AppSettings` instance, mutated in place by `/admin/config`, so the event filter
        # always reads the current instance name and secrets.
        self.config: AppSettings | None = None
        # Last confirmed-applied configuration, not `config` itself: `config` gets overwritten regardless
        # of success, which would hide a still-pending change from a retry. None until first applied.
        self._applied: _Target | None = None
        # Whether a client is set up: false also when the applied configuration has a malformed DSN.
        self._client_enabled = False
        self._secret_sources: list[SecretSource] = []

    @property
    def is_active(self) -> bool:
        """Whether a Sentry client is set up and sending events."""
        return self._client_enabled

    def needs_reconfigure(self, settings: AppSettings | DynamicSettings) -> bool:
        """Whether `settings` differ from what was last applied, including a change that failed to apply."""
        return _target_for(settings) != self._applied

    def register_secret_source(self, source: SecretSource) -> None:
        """Add a callable returning more secret values to scrub from every event, read at send time."""
        self._secret_sources.append(source)

    async def reconfigure(self, config: AppSettings) -> None:
        """(Re)apply `sentry_enabled`/`sentry_dsn`/`sentry_environment` without a restart.

        Called at startup and from `PUT /admin/config` when those settings changed (or a previous
        apply failed). The current client is detached first, so no new event reaches it, then closed
        in a thread: `close()` flushes queued events and can block up to the client's `shutdown_timeout`.

        A malformed DSN turns Sentry off with a warning and still counts as applied, so later config
        saves don't keep retrying a value that can never work. Any other failure propagates and leaves
        the change pending.

        Only error capture is set up: no automatic integrations (they would attach request data), no
        stack-frame local variables, no PII and no performance tracing.
        """
        self.config = config
        _forward_sdk_warnings()
        target = _target_for(config)
        # Cleared before the old client is closed: if the steps below fail, nothing is applied any more,
        # so any configuration (including the previous one) must count as pending.
        self._applied = None
        await self._close_client()

        if not target.active:
            self._applied = target
            return

        try:
            dsn = parse_dsn(target.dsn)
        except ValueError as e:
            logger.warning(
                "Sentry error tracking is off: sentry_dsn is invalid (%s). Set a valid (or empty) sentry_dsn via PUT /admin/config.",
                e,
            )
            self._applied = target
            return

        sentry_sdk.init(
            dsn=target.dsn,
            environment=target.environment,
            release=get_infra_version(),
            default_integrations=False,
            integrations=[DedupeIntegration(), AtexitIntegration()],
            include_local_variables=False,
            send_default_pii=False,
            before_send=self.before_send,
            # Passed explicitly so the SDK's `SENTRY_DEBUG`/`SENTRY_KEEP_ALIVE`/`SENTRY_SPOTLIGHT` env fallbacks
            # never apply; Spotlight would also send every event to a local sidecar.
            debug=False,
            keep_alive=False,
            spotlight=False,
        )
        self._client_enabled = True
        # A well-formed but wrong DSN is otherwise indistinguishable from a working one.
        logger.info("Sentry error tracking enabled (host=%s, project=%s, environment=%s)", dsn.host, dsn.project_id, target.environment)
        self._applied = target

    async def shutdown(self) -> None:
        """Flush pending events, close the client and drop the secret sources."""
        await self._close_client()
        self._applied = None
        self._secret_sources.clear()

    async def _close_client(self) -> None:
        previous_client = sentry_sdk.get_client()
        sentry_sdk.get_global_scope().set_client(None)
        self._client_enabled = False
        if previous_client.is_active():
            await asyncio.to_thread(previous_client.close)

    def _secret_values(self) -> list[str]:
        """Return the current non-empty secret values, longest first so a secret containing another is replaced whole."""
        values: set[str] = set()
        if self.config is not None:
            for field_name in AppSettings.model_fields:
                value = getattr(self.config, field_name)
                if isinstance(value, SecretStr):
                    values.add(value.get_secret_value())
            values.add(self.config.docker_hub_token)
        for source in self._secret_sources:
            values.update(source())
        values.discard("")
        return sorted(values, key=len, reverse=True)

    def before_send(self, event: Event, _hint: Hint) -> Event | None:
        """Set the instance identity on `event` and scrub every config secret from it.

        Runs for every event, whatever reported it. If scrubbing fails the event is dropped, never
        sent unscrubbed.
        """
        try:
            name = self.config.name if self.config is not None else ""
            if name:
                event["server_name"] = name
                event["tags"] = {**event.get("tags", {}), "infra_name": name}
            else:
                event.pop("server_name", None)
            secrets = self._secret_values()
            return cast("Event", _scrub(event, secrets)) if secrets else event
        except Exception:
            logger.warning("Dropped a Sentry event because scrubbing secrets from it failed", exc_info=True)
            return None


sentry = SentryManager()


def capture_exception(
    exc: BaseException,
    *,
    error_id: str,
    handled: bool = True,
    tags: Mapping[str, str] | None = None,
    span_context: SpanContext | None = None,
    fingerprint: Sequence[str] | None = None,
) -> None:
    """Send `exc` to Sentry with correlation tags; never raises, and does nothing while Sentry is off.

    Builds the event with `event_from_exception` rather than `sentry_sdk.capture_exception`, which
    can't mark an exception unhandled.

    Args:
        exc: The exception to report.
        error_id: The identifier already logged (and, for a request, returned to the client) for this failure.
        handled: False for exceptions nothing caught (the catch-all 500 handler), True for caught-and-logged ones.
        tags: Extra event tags.
        span_context: The OTel span the failure belongs to; defaults to the current span. Its IDs are
            added as `trace_id`/`span_id` when valid.
        fingerprint: Sentry grouping key replacing the default one.
    """
    try:
        client = sentry_sdk.get_client()
        if not sentry.is_active or not client.is_active():
            return
        event, hint = event_from_exception(exc, client_options=client.options, mechanism={"type": "generic", "handled": handled})
        event_tags = {"error_id": error_id}
        context = span_context if span_context is not None and span_context.is_valid else trace.get_current_span().get_span_context()
        if context.is_valid:
            event_tags["trace_id"] = format(context.trace_id, "032x")
            event_tags["span_id"] = format(context.span_id, "016x")
        event["tags"] = {**event_tags, **(tags or {})}
        if fingerprint is not None:
            event["fingerprint"] = list(fingerprint)
        sentry_sdk.capture_event(event, hint=hint)
    except Exception:
        logger.exception("Failed to report exception to Sentry (error_id=%s)", error_id)
