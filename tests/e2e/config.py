# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""E2e suite settings, read from `DF_E2E_*` environment variables."""

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class E2ESettings(BaseSettings):
    """Where the target instance is and how the suite talks to it."""

    model_config = SettingsConfigDict(env_prefix="DF_E2E_", extra="ignore")

    base_url: str = ""
    admin_api_key: SecretStr = SecretStr("")
    api_key: SecretStr = SecretStr("")
    openai_api_key: SecretStr = SecretStr("")

    install_timeout: float = 1800
    inference_timeout: float = 300
    ready_timeout: float = 120

    allow_dirty: bool = False
    # Fail instead of skip when a `gpu` scenario finds no GPU on the target - set where a GPU is expected (CI nightly).
    require_gpu: bool = False
    # Fail instead of skip when the upgrade test cannot run - set in the CI upgrade jobs, where a skip would pass a
    # blocking job that tested nothing.
    require_upgrade: bool = False
    # Where to save service container logs when a scenario fails; nothing is saved when empty.
    log_dir: str = ""
    purge: bool = False
    expected_version: str = ""

    # Upgrade test: a command that replaces the target's Infra container (e.g. `tests/e2e/instance.sh upgrade <image> <id>`),
    # and the version the target must report afterwards.
    upgrade_command: str = ""
    upgrade_version: str = ""
