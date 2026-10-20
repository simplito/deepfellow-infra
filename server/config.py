# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Config."""

import os
from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr, ValidationError
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict


class ConfigError(Exception):
    """Exception raised when there is an error in the configuration."""


class AppSettings(BaseSettings):
    """Bootstrap settings, sourced from `.env`.

    Only fields needed before Docker/networking can come up are loaded from the environment
    here. Everything else is dynamic: it is defaulted below and then populated/overwritten
    at startup from `config.json` by `server.dynamic_config`, and mutated in place
    thereafter by the `/admin/config` API. Because this same `AppSettings` instance is
    passed by reference into every long-lived component, mutating an attribute here is
    immediately visible everywhere without extra plumbing.
    """

    infra_admin_api_key: SecretStr  # key to connect to marketplace

    docker_subnet: str = ""
    storage_dir: str = ""
    storage_services_dir: str = ""
    container_name_prefix: str = ""
    compose_prefix: str = "df_"
    external_only: bool = False

    docker_hub_token: str = ""

    ollama_vram_overhead_factor: float = Field(default=1.0, validation_alias=AliasChoices("OLLAMA_VRAM_OVERHEAD_FACTOR"))

    # --- Everything below is dynamic: populated from config.json, mutable at runtime. ---

    name: str = ""
    infra_url: str = ""
    container_name_suffix: str = ""

    mesh_key: SecretStr = SecretStr("")  # key to connect subinfra through ws
    infra_api_key: SecretStr = SecretStr("")  # key to call /v1/ endpoints

    connect_to_mesh_url: str = ""
    connect_to_mesh_key: SecretStr = SecretStr("")
    share_models_downstream: bool = True  # expose own + ancestor models to connecting subinfras

    hugging_face_token: SecretStr = SecretStr("")
    civitai_token: SecretStr = SecretStr("")
    adapter_registry_secret: SecretStr = SecretStr("")
    log_payloads: str = ""
    stop_containers_on_shutdown: str = ""

    mcp_sse_session_ttl_seconds: int = 300
    mcp_sse_max_sessions: int = 128

    standard_proxy_timeout_seconds: int = 600

    # metrics are authorized by HTTPBasicAuth
    metrics_username: str = ""
    metrics_password: SecretStr = SecretStr("")

    otel_exporter_otlp_endpoint: str = "http://localhost:4317"
    otel_tracing_enabled: bool = False
    otel_logging_enabled: bool = False

    # Sentry is configured only through config.json / `PUT /admin/config`: the aliases can't match any
    # real environment variable, so `DF_SENTRY_*` is never read, not even to seed a fresh config.json.
    sentry_enabled: bool = Field(default=False, validation_alias="__DF_SENTRY_ENABLED_NOT_SETTABLE_VIA_ENV__")
    sentry_dsn: SecretStr = Field(default=SecretStr(""), validation_alias="__DF_SENTRY_DSN_NOT_SETTABLE_VIA_ENV__")
    sentry_environment: str = Field(default="", validation_alias="__DF_SENTRY_ENVIRONMENT_NOT_SETTABLE_VIA_ENV__")

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        env_nested_delimiter="__",
        env_prefix="DF_",
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],  # noqa: ARG003
        init_settings: PydanticBaseSettingsSource,  # noqa: ARG003
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,  # noqa: ARG003
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Customize the settings sources for Pydantic's BaseSettings.

        This class method overrides the default settings sources configuration in Pydantic's
        BaseSettings to define a custom priority order for settings sources. It specifies which
        sources are used to load configuration values and in what order they are processed.

        The customized configuration prioritizes:
        1. Environment variables (highest priority)
        2. Values from .env files

        Init values and file secrets are deliberately excluded from the sources, as indicated
        by the noqa comments.
        """
        # Detect if running under pytest (or set your own condition)
        if "PYTEST_CURRENT_TEST" in os.environ or "PYTEST_VERSION" in os.environ:
            # Only use env from (.test.env)
            return (env_settings,)

        # Default: use env, dotenv, and TOML
        return (env_settings, dotenv_settings)  # pragma: no cover

    def get_storage_dir(self) -> Path:
        """Get storage dir."""
        return Path(self.storage_dir) if self.storage_dir else get_main_dir() / "./storage"

    def get_storage_services_dir(self) -> Path:
        """Get storage dir."""
        return Path(self.storage_services_dir) if self.storage_services_dir else self.get_storage_dir() / "services"

    def is_log_payloads_enabled(self) -> bool:
        """Is log payloads enabled."""
        return self.log_payloads == "true"

    def is_stop_containers_on_shutdown_enabled(self) -> bool:
        """Is stop containers on shutdown enabled."""
        return self.stop_containers_on_shutdown != "false"


def get_main_dir() -> Path:
    """Get main dir of the application."""
    return Path(__file__).resolve().parent.parent


def get_services_json_path(config: AppSettings) -> Path:
    """Return the path of services.json, the installed-services state persisted by ServiceProvider."""
    return (config.get_storage_dir() / "./services.json").resolve()


def load_config() -> AppSettings:
    """Load bootstrap config from `.env`. Dynamic fields are populated afterwards from config.json."""
    try:
        return AppSettings()  # type: ignore
    except ValidationError as e:
        has_unknown = False
        messages = ["DeepFellow Infra config error:"]
        for error in e.errors():
            if error["type"] == "missing":
                name = get_name_from_loc(error["loc"])
                messages.append(f"Missing config value for {name}")
            else:
                has_unknown = True
        message = "\n    ".join(messages)
        if has_unknown:
            raise ConfigError(message) from e
        raise ConfigError(message)  # noqa: B904


def get_name_from_loc(loc: tuple[int | str, ...]) -> str:
    """Get the environment variable name for the given location."""
    name = "DF"
    first = True
    for ele in loc:
        name += ("_" if first else "__") + str(ele).upper()
        first = False
    return name
