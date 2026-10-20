# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Infra config API."""

import asyncio
import typing
from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import SecretStr, ValidationError

from server import dynamic_config
from server.config import AppSettings
from server.core.dependencies import (
    auth_admin,
    get_config,
    get_config_lock,
    get_infra_websocket_server,
    get_model_downloader,
    get_otlp_logging,
    get_parent_infra,
    get_task_manager,
)
from server.dynamic_config import DynamicSettings
from server.models.config import ConfigEntry, ConfigOut, ConfigRevealOut
from server.task_manager import TaskManager
from server.utils.logger import logger
from server.utils.model_downloader import ModelDownloader
from server.utils.sentry import parse_dsn, sentry
from server.utils.tracing import OtlpLoggingManager, tracer
from server.websockets.infra_websocket_server import InfraWebsocketServer
from server.websockets.parent_infra_group import ParentInfraGroup

router = APIRouter(prefix="/admin/config", tags=["Config"])

_MASKED = "••••••••"
_ENV_PREFIX: str = AppSettings.model_config.get("env_prefix", "DF_")  # type: ignore[call-overload]
_SENTRY_FIELDS = frozenset({"sentry_enabled", "sentry_dsn", "sentry_environment"})


def _is_editable(field_name: str) -> bool:
    return field_name in DynamicSettings.model_fields


def _is_secret(annotation: type | None) -> bool:
    if annotation is SecretStr:
        return True
    return SecretStr in typing.get_args(annotation or ())


def _to_env_key(field_name: str) -> str:
    return f"{_ENV_PREFIX}{field_name.upper()}"


def _to_field_name(env_key: str) -> str:
    return env_key.upper().removeprefix(_ENV_PREFIX).lower()


def _validate_container_name_suffix(config: AppSettings, current: DynamicSettings, new_settings: DynamicSettings) -> None:
    """Reject a new container_name_suffix with 400 while the legacy DF_CONTAINER_NAME_PREFIX is set.

    The legacy prefix takes priority in get_docker_container_name(), so a new suffix would be
    accepted and persisted but silently do nothing. Clearing it, or re-sending the current one, is fine.
    """
    if (
        config.container_name_prefix
        and new_settings.container_name_suffix
        and new_settings.container_name_suffix != current.container_name_suffix
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "container_name_suffix has no effect while the legacy DF_CONTAINER_NAME_PREFIX is set. Remove "
                "DF_CONTAINER_NAME_PREFIX from .env and restart first - installed containers keep their names."
            ),
        )


def _validate_sentry_dsn(updates: dict[str, Any], new_settings: DynamicSettings) -> None:
    """Reject with 400 a request that sets a malformed `sentry_dsn`, or that changes Sentry settings and leaves it on with one.

    So enabling Sentry over a malformed stored DSN is refused instead of answering 200 while Sentry stays off.
    Turning Sentry off, and every update that doesn't touch the Sentry settings, always pass: a bad stored DSN
    (Sentry is then off with a warning) must not make unrelated updates fail. An empty DSN is valid and means off.

    Not a `DynamicSettings` validator: `read_envelope` would then abort startup on a hand-edited bad DSN.
    """
    dsn = new_settings.sentry_dsn.get_secret_value()
    if not dsn:
        return
    if "sentry_dsn" not in updates and not (new_settings.sentry_enabled and not _SENTRY_FIELDS.isdisjoint(updates)):
        return
    try:
        parse_dsn(dsn)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid Sentry DSN: {e}") from e


@router.get("", summary="List infra configuration entries.")
async def get_config_entries(
    config: Annotated[AppSettings, Depends(get_config)],
    _: Annotated[str, Depends(auth_admin)],
) -> ConfigOut:
    """Return all configuration keys; secret values are masked."""
    entries: list[ConfigEntry] = []
    for field_name, field_info in AppSettings.model_fields.items():
        is_secret = _is_secret(field_info.annotation)
        if is_secret:
            value = _MASKED
        else:
            raw = getattr(config, field_name)
            value = str(raw) if raw is not None else ""
        entries.append(
            ConfigEntry(
                key=_to_env_key(field_name),
                value=value,
                is_secret=is_secret,
                field_name=field_name,
                is_editable=_is_editable(field_name),
            )
        )
    return ConfigOut(entries=entries)


@router.get("/{key}/reveal", summary="Reveal a secret configuration value.")
async def reveal_config_entry(
    key: str,
    config: Annotated[AppSettings, Depends(get_config)],
    _: Annotated[str, Depends(auth_admin)],
) -> ConfigRevealOut:
    """Return the plain-text value of a secret config entry."""
    if not key.upper().startswith(_ENV_PREFIX):
        raise HTTPException(status_code=404, detail="Config key not found")

    field_name = _to_field_name(key)
    field_info = AppSettings.model_fields.get(field_name)
    if field_info is None:
        raise HTTPException(status_code=404, detail="Config key not found")

    if not _is_secret(field_info.annotation):
        raise HTTPException(status_code=400, detail="Key is not a secret")

    value = getattr(config, field_name)
    if isinstance(value, SecretStr):
        return ConfigRevealOut(key=_to_env_key(field_name), value=value.get_secret_value())
    if value is None:
        return ConfigRevealOut(key=_to_env_key(field_name), value="")

    raise HTTPException(status_code=500, detail="Unexpected value type")


@router.put("", summary="Update dynamic infra configuration.")
async def update_dynamic_config(
    updates: Annotated[dict[str, Any], Body()],
    config: Annotated[AppSettings, Depends(get_config)],
    parent_infra: Annotated[ParentInfraGroup, Depends(get_parent_infra)],
    task_manager: Annotated[TaskManager, Depends(get_task_manager)],
    config_lock: Annotated[asyncio.Lock, Depends(get_config_lock)],
    otlp_logging: Annotated[OtlpLoggingManager, Depends(get_otlp_logging)],
    infra_websocket_server: Annotated[InfraWebsocketServer, Depends(get_infra_websocket_server)],
    model_downloader: Annotated[ModelDownloader, Depends(get_model_downloader)],
    auth: Annotated[str, Depends(auth_admin)],
) -> ConfigOut:
    """Apply a partial update to the dynamic settings backed by `config.json`.

    Only fields defined on `DynamicSettings` may be changed here — bootstrap-only fields
    (`docker_subnet`, `external_only`, `storage_dir`, `storage_services_dir`, the legacy
    `container_name_prefix`, `compose_prefix`, `infra_admin_api_key`) are immutable at runtime. Validated, persisted
    atomically, applied in-memory, and any required side effects (mesh reconnect, OTEL and Sentry
    reconfigure, model downloader token refresh) are triggered — all without a process restart.

    `config.json` has exactly one writer (this handler); the lock serializes read-merge-write-apply
    cycles so two concurrent PUTs can't each read the same snapshot and clobber each other's change.
    """
    unknown = set(updates) - set(DynamicSettings.model_fields)
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown or non-dynamic config field(s): {sorted(unknown)}")

    async with config_lock:
        current = dynamic_config.snapshot_dynamic_settings(config)
        # `.to_persistable_dict()`, not `.model_dump(mode="json")` — the latter masks every SecretStr
        # field to "**********", which would then get baked into `new_settings` for any secret field
        # not present in `updates`, wiping it.
        merged_data = current.to_persistable_dict() | updates
        try:
            new_settings = DynamicSettings.model_validate(merged_data)
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=jsonable_encoder(e.errors())) from e
        _validate_container_name_suffix(config, current, new_settings)
        _validate_sentry_dsn(updates, new_settings)

        # Diffed against what parent_infra/otlp_logging/sentry last confirmed applied, not `current` —
        # `apply_dynamic_settings()` below overwrites `config` regardless of success.
        mesh_changed = new_settings.connect_to_mesh_url != parent_infra.configured_mesh_url or (
            new_settings.connect_to_mesh_key.get_secret_value() != parent_infra.configured_mesh_key.get_secret_value()
        )
        otel_logging_changed = (
            new_settings.otel_logging_enabled != otlp_logging.configured_enabled
            or new_settings.otel_exporter_otlp_endpoint != otlp_logging.configured_endpoint
        )
        sentry_changed = sentry.needs_reconfigure(new_settings)
        downloaders_changed = (
            new_settings.hugging_face_token.get_secret_value() != current.hugging_face_token.get_secret_value()
            or new_settings.civitai_token.get_secret_value() != current.civitai_token.get_secret_value()
        )

        # Persist before mutating in-memory state: the file is the source of truth.
        try:
            dynamic_config.persist_settings(config, new_settings)
        except Exception as e:
            logger.exception("Failed to persist config.json")
            raise HTTPException(status_code=502, detail="Configuration could not be saved to config.json.") from e
        dynamic_config.apply_dynamic_settings(config, new_settings)

        try:
            if mesh_changed:
                await parent_infra.reconfigure(config, task_manager)
            if otel_logging_changed:
                await otlp_logging.reconfigure(config)
            if sentry_changed:
                await sentry.reconfigure(config)
            if not new_settings.otel_tracing_enabled and tracer.is_active:
                # Only a disable needs an explicit stop; endpoint changes are picked up lazily
                # elsewhere. Checked against live state, not a flag, so a retry isn't skipped.
                await tracer.shutdown()
            # Stateless and cheap — run unconditionally rather than gating on a flag that could go
            # stale like the ones above used to.
            infra_websocket_server.broadcast_ancestors_to_children()
            if downloaders_changed:
                model_downloader.create_downloaders(config)
        except Exception as e:
            logger.exception("Config saved, but applying it live failed")
            raise HTTPException(
                status_code=502,
                detail="Configuration was saved but could not be applied live. A restart may be required.",
            ) from e

    return await get_config_entries(config, auth)
