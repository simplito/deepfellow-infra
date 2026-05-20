# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from server.serviceprovider import ACTUAL_CONFIG_VERSION, ServiceProvider


def make_provider(tmp_path: Path) -> ServiceProvider:
    config = MagicMock()
    config.get_storage_dir.return_value = tmp_path
    return ServiceProvider(config)


def test_get_file_path(tmp_path: Path):
    provider = make_provider(tmp_path)

    path = provider._get_file_path()  # pyright: ignore[reportPrivateUsage]

    assert path == (tmp_path / "services.json").resolve()


@pytest.mark.parametrize(
    ("data", "expected_cloud_enabled"),
    [
        ({"version": "v2", "services": {"ollama": {"something": True}}}, False),
        ({"version": "v2", "services": {"claude": {"api_key": "abc"}, "ollama": {}}}, True),
        ({"version": "v2", "services": {"claude": {}}}, False),
        ({"version": "v2"}, False),
    ],
)
def test_convert_v2_to_v3_cloud_enabled(tmp_path: Path, data: dict[str, Any], expected_cloud_enabled: bool):
    provider = make_provider(tmp_path)

    result = provider.convert_v2_to_v3_config(data)

    assert result["cloud_enabled"] is expected_cloud_enabled


def test_convert_v2_to_v3_does_not_mutate_original(tmp_path: Path):
    provider = make_provider(tmp_path)
    data: dict[str, Any] = {"version": "v2", "services": {"openai": {"key": "x"}}}

    result = provider.convert_v2_to_v3_config(data)

    assert "cloud_enabled" not in data
    assert result["cloud_enabled"] is True


def test_convert_v1_to_v2_moves_options_models_custom(tmp_path: Path):
    provider = make_provider(tmp_path)
    data: dict[str, Any] = {
        "version": "v1",
        "services": {
            "ollama": {
                "options": {"host": "localhost"},
                "models": ["llama3"],
                "custom": ["my-model"],
            }
        },
    }

    result = provider.convert_v1_to_v2_config(data)

    svc = result["services"]["ollama"]
    inst = svc["instances"]["default"]
    assert inst["options"] == {"host": "localhost"}
    assert inst["models"] == ["llama3"]
    assert inst["custom"] == ["my-model"]
    assert "options" not in svc
    assert "models" not in svc
    assert "custom" not in svc


def test_convert_v1_to_v2_non_dict_service_raises(tmp_path: Path):
    provider = make_provider(tmp_path)
    data: dict[str, Any] = {"version": "v1", "services": {"ollama": "some_string"}}
    with pytest.raises(TypeError):
        provider.convert_v1_to_v2_config(data)


def test_convert_v1_to_v2_partial_fields(tmp_path: Path):
    provider = make_provider(tmp_path)
    data: dict[str, Any] = {
        "version": "v1",
        "services": {"ollama": {}},
    }

    result = provider.convert_v1_to_v2_config(data)

    inst = result["services"]["ollama"]["instances"]["default"]
    assert inst["options"] == {}
    assert inst["models"] == []
    assert inst["custom"] == []


def test_convert_v1_to_v2_no_services(tmp_path: Path):
    provider = make_provider(tmp_path)
    data: dict[str, Any] = {"version": "v1"}

    result = provider.convert_v1_to_v2_config(data)

    assert "services" not in result


@pytest.mark.asyncio
async def test_load_file_not_found_returns_defaults(tmp_path: Path):
    provider = make_provider(tmp_path)

    result = await provider.load()

    assert result["version"] == ACTUAL_CONFIG_VERSION
    assert result["services"] == {}
    assert result["cloud_enabled"] is False


@pytest.mark.asyncio
async def test_load_current_version(tmp_path: Path):
    provider = make_provider(tmp_path)
    content: dict[str, Any] = {"version": "v3", "services": {}, "cloud_enabled": False}
    (tmp_path / "services.json").write_text(json.dumps(content))

    result = await provider.load()

    assert result["version"] == "v3"
    assert result["services"] == {}


@pytest.mark.asyncio
async def test_load_v2_migrates_to_v3(tmp_path: Path):
    provider = make_provider(tmp_path)
    content: dict[str, Any] = {"version": "v2", "services": {"openai": {"key": "abc"}}}
    (tmp_path / "services.json").write_text(json.dumps(content))

    result = await provider.load()

    assert result["version"] == ACTUAL_CONFIG_VERSION
    assert result["cloud_enabled"] is True
    saved = json.loads((tmp_path / "services.json").read_text())
    assert saved["version"] == ACTUAL_CONFIG_VERSION


@pytest.mark.asyncio
async def test_load_v1_migrates_through_v2_to_v3(tmp_path: Path):
    provider = make_provider(tmp_path)
    content: dict[str, Any] = {
        "version": "v1",
        "services": {"ollama": {"options": {"host": "localhost"}, "models": [], "custom": []}},
    }
    (tmp_path / "services.json").write_text(json.dumps(content))

    result = await provider.load()

    assert result["version"] == ACTUAL_CONFIG_VERSION
    saved = json.loads((tmp_path / "services.json").read_text())
    assert saved["version"] == ACTUAL_CONFIG_VERSION
    assert "instances" in saved["services"]["ollama"]


@pytest.mark.asyncio
async def test_load_missing_services_key_adds_empty(tmp_path: Path):
    provider = make_provider(tmp_path)
    content: dict[str, Any] = {"version": "v3", "cloud_enabled": False}
    (tmp_path / "services.json").write_text(json.dumps(content))

    result = await provider.load()

    assert result["services"] == {}


@pytest.mark.asyncio
async def test_load_version_mismatch_triggers_save(tmp_path: Path):
    provider = make_provider(tmp_path)
    content: dict[str, Any] = {"version": "v2", "services": {}, "cloud_enabled": False}
    (tmp_path / "services.json").write_text(json.dumps(content))

    await provider.load()

    saved = json.loads((tmp_path / "services.json").read_text())
    assert saved["version"] == ACTUAL_CONFIG_VERSION


@pytest.mark.asyncio
async def test_save_writes_json(tmp_path: Path):
    provider = make_provider(tmp_path)
    content = {"version": "v3", "services": {"ollama": {}}, "cloud_enabled": False}

    await provider.save(content)  # type: ignore[arg-type]

    saved = json.loads((tmp_path / "services.json").read_text())
    assert saved["services"] == {"ollama": {}}


@pytest.mark.asyncio
async def test_write_file_creates_directories(tmp_path: Path):
    provider = make_provider(tmp_path)
    deep_path = tmp_path / "a" / "b" / "c" / "file.json"

    await provider._write_file(str(deep_path), '{"ok": true}')  # pyright: ignore[reportPrivateUsage]

    assert deep_path.exists()
    assert json.loads(deep_path.read_text()) == {"ok": True}


@pytest.mark.asyncio
async def test_modify_with_sync_handler(tmp_path: Path):
    provider = make_provider(tmp_path)
    (tmp_path / "services.json").write_text(json.dumps({"version": "v3", "services": {}, "cloud_enabled": False}))

    def handler(content: Any) -> Any:
        content["cloud_enabled"] = True
        return content

    await provider._modify(handler)  # pyright: ignore[reportPrivateUsage]

    saved = json.loads((tmp_path / "services.json").read_text())
    assert saved["cloud_enabled"] is True


@pytest.mark.asyncio
async def test_modify_with_async_handler(tmp_path: Path):
    provider = make_provider(tmp_path)
    (tmp_path / "services.json").write_text(json.dumps({"version": "v3", "services": {}, "cloud_enabled": False}))

    async def handler(content: Any) -> Any:
        content["cloud_enabled"] = True
        return content

    await provider._modify(handler)  # pyright: ignore[reportPrivateUsage]

    saved = json.loads((tmp_path / "services.json").read_text())
    assert saved["cloud_enabled"] is True


@pytest.mark.asyncio
async def test_modify_handler_returns_false_skips_save(tmp_path: Path):
    provider = make_provider(tmp_path)
    original = {"version": "v3", "services": {}, "cloud_enabled": False}
    (tmp_path / "services.json").write_text(json.dumps(original))

    provider.save = AsyncMock()

    def handler(content: Any) -> Any:
        return False

    await provider._modify(handler)  # pyright: ignore[reportPrivateUsage]

    assert provider.save.await_count == 0


@pytest.mark.asyncio
async def test_save_service_config(tmp_path: Path):
    provider = make_provider(tmp_path)
    (tmp_path / "services.json").write_text(json.dumps({"version": "v3", "services": {}, "cloud_enabled": False}))
    await provider.save_service_config("ollama", {"options": {"host": "localhost"}})
    saved = json.loads((tmp_path / "services.json").read_text())
    assert saved["services"]["ollama"] == {"options": {"host": "localhost"}}


@pytest.mark.asyncio
async def test_get_cloud_enabled_false(tmp_path: Path):
    provider = make_provider(tmp_path)
    (tmp_path / "services.json").write_text(json.dumps({"version": "v3", "services": {}, "cloud_enabled": False}))

    result = await provider.get_cloud_enabled()

    assert result is False


@pytest.mark.asyncio
async def test_get_cloud_enabled_true(tmp_path: Path):
    provider = make_provider(tmp_path)
    (tmp_path / "services.json").write_text(json.dumps({"version": "v3", "services": {}, "cloud_enabled": True}))

    result = await provider.get_cloud_enabled()

    assert result is True


@pytest.mark.asyncio
async def test_set_cloud_enabled(tmp_path: Path):
    provider = make_provider(tmp_path)
    (tmp_path / "services.json").write_text(json.dumps({"version": "v3", "services": {}, "cloud_enabled": False}))

    await provider.set_cloud_enabled(True)

    saved = json.loads((tmp_path / "services.json").read_text())
    assert saved["cloud_enabled"] is True


@pytest.mark.asyncio
async def test_clear_service_config_existing(tmp_path: Path):
    provider = make_provider(tmp_path)
    (tmp_path / "services.json").write_text(json.dumps({"version": "v3", "services": {"ollama": {}}, "cloud_enabled": False}))

    await provider.clear_service_config("ollama")

    saved = json.loads((tmp_path / "services.json").read_text())
    assert "ollama" not in saved["services"]


@pytest.mark.asyncio
async def test_clear_service_config_nonexistent(tmp_path: Path):
    provider = make_provider(tmp_path)
    (tmp_path / "services.json").write_text(json.dumps({"version": "v3", "services": {}, "cloud_enabled": False}))

    await provider.clear_service_config("nonexistent")

    saved = json.loads((tmp_path / "services.json").read_text())
    assert saved["services"] == {}
