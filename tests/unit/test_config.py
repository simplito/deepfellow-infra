# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

import pytest

from server.config import AppSettings, ConfigError, get_main_dir, get_name_from_loc, load_config

_REQUIRED_ENV = {
    "DF_NAME": "test-node",
    "DF_INFRA_URL": "http://infra.local",
    "DF_INFRA_ADMIN_API_KEY": "admin-key",
    "DF_MESH_KEY": "mesh-key",
    "DF_INFRA_API_KEY": "api-key",
}


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> AppSettings:
    for k, v in _REQUIRED_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("DF_CONNECT_TO_MESH_URL", raising=False)
    monkeypatch.delenv("DF_CONNECT_TO_MESH_URLS", raising=False)

    return AppSettings()  # pyright: ignore[reportCallIssue]


def test_app_settings_loads(settings: AppSettings) -> None:
    assert settings.name == "test-node"
    assert settings.infra_url == "http://infra.local"


def test_get_storage_dir_default(settings: AppSettings) -> None:
    result = settings.get_storage_dir()

    assert result == get_main_dir() / "./storage"


def test_get_storage_dir_custom(monkeypatch: pytest.MonkeyPatch) -> None:
    for k, v in _REQUIRED_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("DF_STORAGE_DIR", "/custom/storage")

    s = AppSettings()  # pyright: ignore[reportCallIssue]

    assert s.get_storage_dir() == Path("/custom/storage")


def test_get_storage_services_dir_default(settings: AppSettings) -> None:
    result = settings.get_storage_services_dir()

    assert result == settings.get_storage_dir() / "services"


def test_get_storage_services_dir_custom(monkeypatch: pytest.MonkeyPatch) -> None:
    for k, v in _REQUIRED_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("DF_STORAGE_SERVICES_DIR", "/custom/services")

    s = AppSettings()  # pyright: ignore[reportCallIssue]

    assert s.get_storage_services_dir() == Path("/custom/services")


def test_is_log_payloads_enabled_true(monkeypatch: pytest.MonkeyPatch) -> None:
    for k, v in _REQUIRED_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("DF_LOG_PAYLOADS", "true")

    s = AppSettings()  # pyright: ignore[reportCallIssue]

    assert s.is_log_payloads_enabled() is True


def test_is_log_payloads_enabled_false(settings: AppSettings) -> None:
    assert settings.is_log_payloads_enabled() is False


def test_is_stop_containers_on_shutdown_enabled_default(monkeypatch: pytest.MonkeyPatch) -> None:
    for k, v in _REQUIRED_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("DF_STOP_CONTAINERS_ON_SHUTDOWN", raising=False)
    s = AppSettings()  # pyright: ignore[reportCallIssue]
    assert s.is_stop_containers_on_shutdown_enabled() is True


def test_is_stop_containers_on_shutdown_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    for k, v in _REQUIRED_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("DF_STOP_CONTAINERS_ON_SHUTDOWN", "false")

    s = AppSettings()  # pyright: ignore[reportCallIssue]

    assert s.is_stop_containers_on_shutdown_enabled() is False


def test_load_config_success(monkeypatch: pytest.MonkeyPatch) -> None:
    for k, v in _REQUIRED_ENV.items():
        monkeypatch.setenv(k, v)

    result = load_config()

    assert result.name == "test-node"


def test_load_config_missing_fields_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in _REQUIRED_ENV:
        monkeypatch.delenv(k, raising=False)

    with pytest.raises(ConfigError, match="Missing config value"):
        load_config()


def test_load_config_invalid_field_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    for k, v in _REQUIRED_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("DF_OTEL_TRACING_ENABLED", "not_a_bool")

    with pytest.raises(ConfigError):
        load_config()


@pytest.mark.parametrize(
    ("loc", "expected"),
    [
        (("name",), "DF_NAME"),
        (("infra", "url"), "DF_INFRA__URL"),
        (("a", "b", "c"), "DF_A__B__C"),
        ((0,), "DF_0"),
    ],
)
def test_get_name_from_loc(loc: tuple[str, str], expected: str) -> None:
    assert get_name_from_loc(loc) == expected


def test_get_main_dir_returns_existing_path() -> None:
    result = get_main_dir()
    assert isinstance(result, Path)
    assert result.is_dir()


def test_connect_to_mesh_url_empty_by_default(settings: AppSettings) -> None:
    assert settings.connect_to_mesh_url == ""


def test_connect_to_mesh_url_set(monkeypatch: pytest.MonkeyPatch) -> None:
    for k, v in _REQUIRED_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("DF_CONNECT_TO_MESH_URL", "http://mesh.url")

    s = AppSettings()  # pyright: ignore[reportCallIssue]

    assert s.connect_to_mesh_url == "http://mesh.url"
