# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Config."""

import os
from pathlib import Path

from pydantic import SecretStr, ValidationError
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict


class ConfigError(Exception):
    """Exception raised when there is an error in the configuration."""


class AppSettings(BaseSettings):
    name: str
    infra_url: str
    infra_admin_api_key: SecretStr  # key to connect to marketplace
    mesh_key: SecretStr  # key to connect subinfra through ws
    infra_api_key: SecretStr  # key to call /v1/ endpoints
    connect_to_mesh_url: str = ""
    connect_to_mesh_key: SecretStr = SecretStr("")

    docker_subnet: str = ""
    storage_dir: str = ""
    storage_services_dir: str = ""
    hugging_face_token: str = ""
    civitai_token: str = ""
    log_payloads: str = ""
    container_name_prefix: str = ""
    compose_prefix: str = "df_"
    stop_containers_on_shutdown: str = ""
    nvidia_gpus_count: int = 1

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


def load_config() -> AppSettings:
    """Load config."""
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
