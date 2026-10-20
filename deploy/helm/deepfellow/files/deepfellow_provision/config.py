# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Everything a run needs, read from the environment the chart sets, exactly once."""

import json
import os
from dataclasses import dataclass
from typing import Any

from deepfellow_provision.errors import ProvisionError

Backend = dict[str, Any]

# Keys every backend descriptor in BACKENDS_JSON must carry. Checked once, here, so a mismatch
# between chart and script fails at startup with a name instead of a KeyError mid-registration.
REQUIRED_BACKEND_KEYS = ("id", "api_url")


@dataclass(frozen=True)
class Config:
    """Where the services are, who the admin is, and which backends to register."""

    namespace: str
    state_secret: str
    infra_url: str
    infra_admin_key: str
    infra_api_key: str
    server_url: str
    admin_name: str
    admin_email: str
    admin_password: str
    org_name: str
    project_name: str
    key_name: str
    backends: tuple[Backend, ...]


def env(name: str, default: str | None = None, *, required: bool = False) -> str:
    """Read an environment variable, failing when a required one is missing or empty."""
    value = os.environ.get(name, default)
    if required and not value:
        msg = f"missing required env {name}"
        raise ProvisionError(msg)
    return value or ""


def parse_backends(raw: str) -> list[Backend]:
    """Parse and validate BACKENDS_JSON."""
    try:
        parsed: object = json.loads(raw)
    except json.JSONDecodeError as e:
        msg = f"BACKENDS_JSON is not valid JSON: {e}"
        raise ProvisionError(msg) from e
    if not isinstance(parsed, list):
        msg = f"BACKENDS_JSON must be a JSON array, got {type(parsed).__name__}"
        raise ProvisionError(msg)
    backends: list[Backend] = []
    for i, item in enumerate(parsed):  # pyright: ignore[reportUnknownVariableType]
        if not isinstance(item, dict):
            msg = f"BACKENDS_JSON[{i}] must be an object, got {type(item).__name__}"
            raise ProvisionError(msg)
        missing = [k for k in REQUIRED_BACKEND_KEYS if not item.get(k)]
        if missing:
            msg = f"BACKENDS_JSON[{i}] is missing required key(s): {missing}"
            raise ProvisionError(msg)
        backends.append(item)  # pyright: ignore[reportUnknownArgumentType]
    return backends


def load() -> Config:
    """Build the Config from the environment."""
    return Config(
        namespace=env("POD_NAMESPACE", required=True),
        state_secret=env("STATE_SECRET_NAME", required=True),
        infra_url=env("DF_INFRA_URL", required=True),
        infra_admin_key=env("DF_INFRA_ADMIN_API_KEY", required=True),
        infra_api_key=env("DF_INFRA_API_KEY", required=True),
        server_url=env("DF_SERVER_URL", required=True),
        admin_name=env("ADMIN_NAME", "admin"),
        admin_email=env("ADMIN_EMAIL", required=True),
        admin_password=env("ADMIN_PASSWORD", required=True),
        org_name=env("PROJECT_ORG_NAME", "Workspace"),
        project_name=env("PROJECT_NAME", "Default"),
        key_name=env("PROJECT_KEY_NAME", "app"),
        backends=tuple(parse_backends(env("BACKENDS_JSON", "[]"))),
    )
