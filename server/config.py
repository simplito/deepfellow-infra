"""Config."""

import os

from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict


class ConfigError(SystemExit):
    """Exception raised when there is an error in the configuration."""


class AppSettings(BaseSettings):
    api_key: str

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
