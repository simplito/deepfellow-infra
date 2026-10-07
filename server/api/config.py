# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Infra config API."""

import asyncio
import logging
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
from server.utils.model_downloader import ModelDownloader
from server.utils.tracing import OtlpLoggingManager, tracer
from server.websockets.infra_websocket_server import InfraWebsocketServer
from server.websockets.parent_infra_group import ParentInfraGroup

logger = logging.getLogger("uvicorn.error")

router = APIRouter(prefix="/admin/config", tags=["Config"])

_MASKED = "••••••••"
_ENV_PREFIX: str = AppSettings.model_config.get("env_prefix", "DF_")  # type: ignore[call-overload]


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
    (`docker_subnet`, `external_only`, `storage_dir`, `storage_services_dir`,
    `container_name_prefix`, `compose_prefix`, `infra_admin_api_key`) are immutable at runtime. Validated, persisted
    atomically, applied in-memory, and any required side effects (mesh reconnect, OTEL
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

        mesh_changed = new_settings.connect_to_mesh_url != current.connect_to_mesh_url or (
            new_settings.connect_to_mesh_key.get_secret_value() != current.connect_to_mesh_key.get_secret_value()
        )
        otel_logging_changed = (
            new_settings.otel_logging_enabled != current.otel_logging_enabled
            or new_settings.otel_exporter_otlp_endpoint != current.otel_exporter_otlp_endpoint
        )
        otel_tracing_disabled = current.otel_tracing_enabled and not new_settings.otel_tracing_enabled
        ancestor_broadcast_needed = (
            new_settings.share_models_downstream != current.share_models_downstream
            or new_settings.infra_api_key.get_secret_value() != current.infra_api_key.get_secret_value()
            or new_settings.name != current.name
            or new_settings.infra_url != current.infra_url
        )
        downloaders_changed = (
            new_settings.hugging_face_token.get_secret_value() != current.hugging_face_token.get_secret_value()
            or new_settings.civitai_token.get_secret_value() != current.civitai_token.get_secret_value()
            or new_settings.adapter_registry_url != current.adapter_registry_url
            or new_settings.adapter_registry_secret.get_secret_value() != current.adapter_registry_secret.get_secret_value()
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
            if otel_tracing_disabled:
                # Endpoint changes are already picked up lazily by tracer._get_tracer()/
                # _get_mcp_instruments(); only a disable transition needs an explicit stop, since
                # nothing calls those getters again to shut down the now-unreferenced providers.
                await tracer.shutdown()
            if ancestor_broadcast_needed:
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
