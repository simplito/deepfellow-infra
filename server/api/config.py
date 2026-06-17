# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Infra config API."""

import typing
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import SecretStr

from server.config import AppSettings
from server.core.dependencies import auth_admin, get_config
from server.models.config import ConfigEntry, ConfigOut, ConfigRevealOut

router = APIRouter(prefix="/admin/config", tags=["Config"])

_MASKED = "••••••••"
_ENV_PREFIX: str = AppSettings.model_config.get("env_prefix", "DF_")  # type: ignore[call-overload]


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
        entries.append(ConfigEntry(key=_to_env_key(field_name), value=value, is_secret=is_secret))
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
