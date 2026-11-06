# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from server.serviceprovider import ACTUAL_CONFIG_VERSION, ServiceProvider, ServicesConfigError, ServicesMigrationError


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
    content: dict[str, Any] = {"version": "v4", "services": {}, "cloud_enabled": False, "warnings": []}
    (tmp_path / "services.json").write_text(json.dumps(content))

    result = await provider.load()

    assert result["version"] == "v4"
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
@pytest.mark.parametrize(
    ("content", "match"),
    [
        ("[]", "top level, got list"),
        ("null", "top level, got NoneType"),
        ('{"version": "v4", "services": []}', "'services' to be a JSON object"),
        ('{"version": "v4", "services": {}, "warnings": {}}', "'warnings' to be a JSON array"),
    ],
)
async def test_load_wrong_structure_raises_clear_error(tmp_path: Path, content: str, match: str):
    provider = make_provider(tmp_path)
    (tmp_path / "services.json").write_text(content)

    with pytest.raises(ServicesConfigError, match=match):
        await provider.load()

    assert (tmp_path / "services.json").read_text() == content


@pytest.mark.asyncio
async def test_load_truncated_file_raises_and_leaves_file_untouched(tmp_path: Path):
    provider = make_provider(tmp_path)
    content = '{"version": "v4", "serv'
    (tmp_path / "services.json").write_text(content)

    with pytest.raises(json.JSONDecodeError):
        await provider.load()

    assert (tmp_path / "services.json").read_text() == content


@pytest.mark.asyncio
async def test_load_raises_migration_error_when_saving_migrated_file_fails(tmp_path: Path):
    provider = make_provider(tmp_path)
    content = json.dumps({"version": "v3", "services": {}, "cloud_enabled": False})
    (tmp_path / "services.json").write_text(content)
    error = OSError(30, "Read-only file system")
    provider._save_locked = AsyncMock(side_effect=error)  # pyright: ignore[reportPrivateUsage]

    with pytest.raises(ServicesMigrationError) as exc_info:
        await provider.load()

    assert exc_info.value.cause is error
    assert (tmp_path / "services.json").read_text() == content


@pytest.mark.asyncio
async def test_save_service_config_on_truncated_file_raises_and_leaves_file_untouched(tmp_path: Path):
    provider = make_provider(tmp_path)
    content = '{"version": "v4", "serv'
    (tmp_path / "services.json").write_text(content)

    with pytest.raises(json.JSONDecodeError):
        await provider.save_service_config("ollama", {})

    assert (tmp_path / "services.json").read_text() == content


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
async def test_write_file_removes_tmp_file_on_failure(tmp_path: Path):
    provider = make_provider(tmp_path)
    target = tmp_path / "file.json"

    with (
        patch("server.serviceprovider.os.replace", side_effect=OSError("disk full")),
        pytest.raises(OSError, match="disk full"),
    ):
        await provider._write_file(str(target), '{"ok": true}')  # pyright: ignore[reportPrivateUsage]

    assert not target.with_name("file.json.tmp").exists()
    assert not target.exists()


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
    original = {"version": "v4", "services": {}, "cloud_enabled": False, "warnings": []}
    (tmp_path / "services.json").write_text(json.dumps(original))

    provider._save_locked = AsyncMock()  # pyright: ignore[reportPrivateUsage]

    def handler(content: Any) -> Any:
        return False

    await provider._modify(handler)  # pyright: ignore[reportPrivateUsage]

    assert provider._save_locked.await_count == 0  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_save_service_config(tmp_path: Path):
    provider = make_provider(tmp_path)
    (tmp_path / "services.json").write_text(json.dumps({"version": "v3", "services": {}, "cloud_enabled": False}))
    await provider.save_service_config("ollama", {"options": {"host": "localhost"}})
    saved = json.loads((tmp_path / "services.json").read_text())
    assert saved["services"]["ollama"] == {"options": {"host": "localhost"}}


@pytest.mark.asyncio
async def test_save_service_config_skipped_and_file_untouched_when_load_failed(tmp_path: Path):
    provider = make_provider(tmp_path)
    provider.load_error = "Could not read the services configuration"
    content = json.dumps(
        {
            "version": "v4",
            "services": {"ollama": {"instances": {"default": {}, "second": {}}}},
            "cloud_enabled": False,
            "warnings": [],
        }
    )
    (tmp_path / "services.json").write_text(content)

    await provider.save_service_config("ollama", {"instances": {"default": {}}})

    assert (tmp_path / "services.json").read_text() == content


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


def test_convert_v3_to_v4_adds_warnings(tmp_path: Path):
    provider = make_provider(tmp_path)
    data: dict[str, Any] = {"version": "v3", "services": {}, "cloud_enabled": False}

    result = provider.convert_v3_to_v4_config(data)

    assert result["warnings"] == []
    assert "warnings" not in data


@pytest.mark.asyncio
async def test_load_v3_migrates_to_v4(tmp_path: Path):
    provider = make_provider(tmp_path)
    content: dict[str, Any] = {"version": "v3", "services": {}, "cloud_enabled": False}
    (tmp_path / "services.json").write_text(json.dumps(content))

    result = await provider.load()

    assert result["version"] == ACTUAL_CONFIG_VERSION
    assert result["warnings"] == []
    saved = json.loads((tmp_path / "services.json").read_text())
    assert saved["warnings"] == []


@pytest.mark.asyncio
async def test_add_warning_appends_entry(tmp_path: Path):
    provider = make_provider(tmp_path)
    (tmp_path / "services.json").write_text(json.dumps({"version": "v4", "services": {}, "cloud_enabled": False, "warnings": []}))

    await provider.add_warning("ollama", "boom", instance="default", model_id="m1")

    saved = json.loads((tmp_path / "services.json").read_text())
    assert len(saved["warnings"]) == 1
    entry = saved["warnings"][0]
    assert entry["service_id"] == "ollama"
    assert entry["message"] == "boom"
    assert entry["instance"] == "default"
    assert entry["model_id"] == "m1"
    assert entry["id"]
    assert entry["created_at"]


@pytest.mark.asyncio
async def test_list_warnings_returns_all(tmp_path: Path):
    provider = make_provider(tmp_path)
    (tmp_path / "services.json").write_text(json.dumps({"version": "v4", "services": {}, "cloud_enabled": False, "warnings": []}))

    await provider.add_warning("ollama", "first")
    await provider.add_warning("vllm", "second")

    result = await provider.list_warnings()

    assert [w["message"] for w in result] == ["first", "second"]


@pytest.mark.asyncio
async def test_dismiss_warning_removes_matching_entry(tmp_path: Path):
    provider = make_provider(tmp_path)
    (tmp_path / "services.json").write_text(json.dumps({"version": "v4", "services": {}, "cloud_enabled": False, "warnings": []}))

    await provider.add_warning("ollama", "first")
    await provider.add_warning("vllm", "second")
    warning_id = (await provider.list_warnings())[0]["id"]

    removed = await provider.dismiss_warning(warning_id)

    assert removed is True
    result = await provider.list_warnings()
    assert [w["message"] for w in result] == ["second"]


@pytest.mark.asyncio
async def test_dismiss_warning_nonexistent_is_noop(tmp_path: Path):
    provider = make_provider(tmp_path)
    (tmp_path / "services.json").write_text(json.dumps({"version": "v4", "services": {}, "cloud_enabled": False, "warnings": []}))
    await provider.add_warning("ollama", "first")

    provider._save_locked = AsyncMock()  # pyright: ignore[reportPrivateUsage]
    removed = await provider.dismiss_warning("nonexistent")

    assert removed is False
    assert provider._save_locked.await_count == 0  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_dismiss_all_warnings_removes_everything(tmp_path: Path):
    provider = make_provider(tmp_path)
    (tmp_path / "services.json").write_text(json.dumps({"version": "v4", "services": {}, "cloud_enabled": False, "warnings": []}))

    await provider.add_warning("ollama", "first")
    await provider.add_warning("vllm", "second")

    await provider.dismiss_all_warnings()

    result = await provider.list_warnings()
    assert result == []


@pytest.mark.asyncio
async def test_dismiss_all_warnings_empty_is_noop(tmp_path: Path):
    provider = make_provider(tmp_path)
    (tmp_path / "services.json").write_text(json.dumps({"version": "v4", "services": {}, "cloud_enabled": False, "warnings": []}))

    provider._save_locked = AsyncMock()  # pyright: ignore[reportPrivateUsage]
    await provider.dismiss_all_warnings()

    assert provider._save_locked.await_count == 0  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_dismiss_warnings_matching_removes_exact_match(tmp_path: Path):
    provider = make_provider(tmp_path)
    (tmp_path / "services.json").write_text(json.dumps({"version": "v4", "services": {}, "cloud_enabled": False, "warnings": []}))

    await provider.add_warning("ollama", "model failed", instance="default", model_id="m1")
    await provider.add_warning("ollama", "other model failed", instance="default", model_id="m2")
    await provider.add_warning("ollama", "service failed")

    await provider.dismiss_warnings_matching("ollama", instance="default", model_id="m1")

    result = await provider.list_warnings()
    assert [w["message"] for w in result] == ["other model failed", "service failed"]


@pytest.mark.asyncio
async def test_dismiss_warnings_matching_service_level_only(tmp_path: Path):
    provider = make_provider(tmp_path)
    (tmp_path / "services.json").write_text(json.dumps({"version": "v4", "services": {}, "cloud_enabled": False, "warnings": []}))

    await provider.add_warning("ollama", "model failed", instance="default", model_id="m1")
    await provider.add_warning("ollama", "service failed")

    await provider.dismiss_warnings_matching("ollama")

    result = await provider.list_warnings()
    assert [w["message"] for w in result] == ["model failed"]


@pytest.mark.asyncio
async def test_dismiss_warnings_matching_no_match_is_noop(tmp_path: Path):
    provider = make_provider(tmp_path)
    (tmp_path / "services.json").write_text(json.dumps({"version": "v4", "services": {}, "cloud_enabled": False, "warnings": []}))
    await provider.add_warning("ollama", "model failed", instance="default", model_id="m1")

    provider._save_locked = AsyncMock()  # pyright: ignore[reportPrivateUsage]
    await provider.dismiss_warnings_matching("ollama", instance="default", model_id="does-not-exist")

    assert provider._save_locked.await_count == 0  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_dismiss_warnings_matching_any_removes_both_keys_in_one_write(tmp_path: Path):
    provider = make_provider(tmp_path)
    (tmp_path / "services.json").write_text(json.dumps({"version": "v4", "services": {}, "cloud_enabled": False, "warnings": []}))

    await provider.add_warning("ollama", "service failed")
    await provider.add_warning("ollama", "instance failed", instance="default")
    await provider.add_warning("ollama", "model failed", instance="default", model_id="m1")

    with patch.object(provider, "_save_locked", wraps=provider._save_locked) as save_locked:  # pyright: ignore[reportPrivateUsage]
        await provider.dismiss_warnings_matching_any("ollama", [(None, None), ("default", None)])

    save_locked.assert_awaited_once()
    result = await provider.list_warnings()
    assert [w["message"] for w in result] == ["model failed"]


@pytest.mark.asyncio
async def test_dismiss_warnings_matching_any_no_match_is_noop(tmp_path: Path):
    provider = make_provider(tmp_path)
    (tmp_path / "services.json").write_text(json.dumps({"version": "v4", "services": {}, "cloud_enabled": False, "warnings": []}))
    await provider.add_warning("ollama", "model failed", instance="default", model_id="m1")

    provider._save_locked = AsyncMock()  # pyright: ignore[reportPrivateUsage]
    await provider.dismiss_warnings_matching_any("ollama", [(None, None), ("other-instance", None)])

    assert provider._save_locked.await_count == 0  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_dismiss_warnings_for_instance_removes_all_models(tmp_path: Path):
    provider = make_provider(tmp_path)
    (tmp_path / "services.json").write_text(json.dumps({"version": "v4", "services": {}, "cloud_enabled": False, "warnings": []}))

    await provider.add_warning("ollama", "m1 failed", instance="default", model_id="m1")
    await provider.add_warning("ollama", "m2 failed", instance="default", model_id="m2")
    await provider.add_warning("ollama", "other instance failed", instance="other", model_id="m1")
    await provider.add_warning("llamacpp", "unrelated service failed", instance="default", model_id="m1")

    await provider.dismiss_warnings_for_instance("ollama", "default")

    result = await provider.list_warnings()
    assert [w["message"] for w in result] == ["other instance failed", "unrelated service failed"]


@pytest.mark.asyncio
async def test_dismiss_warnings_for_instance_no_match_is_noop(tmp_path: Path):
    provider = make_provider(tmp_path)
    (tmp_path / "services.json").write_text(json.dumps({"version": "v4", "services": {}, "cloud_enabled": False, "warnings": []}))
    await provider.add_warning("ollama", "m1 failed", instance="default", model_id="m1")

    provider._save_locked = AsyncMock()  # pyright: ignore[reportPrivateUsage]
    await provider.dismiss_warnings_for_instance("ollama", "does-not-exist")

    assert provider._save_locked.await_count == 0  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_add_warning_replaces_existing_entry_for_same_key(tmp_path: Path):
    provider = make_provider(tmp_path)
    (tmp_path / "services.json").write_text(json.dumps({"version": "v4", "services": {}, "cloud_enabled": False, "warnings": []}))

    await provider.add_warning("ollama", "first failure", instance="default", model_id="m1")
    await provider.add_warning("ollama", "second failure", instance="default", model_id="m1")

    result = await provider.list_warnings()
    assert len(result) == 1
    assert result[0]["message"] == "second failure"


@pytest.mark.asyncio
async def test_clear_service_config_drops_its_warnings(tmp_path: Path):
    provider = make_provider(tmp_path)
    (tmp_path / "services.json").write_text(
        json.dumps({"version": "v4", "services": {"ollama": {"foo": "bar"}}, "cloud_enabled": False, "warnings": []})
    )

    await provider.add_warning("ollama", "model failed", instance="default", model_id="m1")
    await provider.add_warning("llamacpp", "unrelated service failed", instance="default", model_id="m1")

    await provider.clear_service_config("ollama")

    result = await provider.list_warnings()
    assert [w["message"] for w in result] == ["unrelated service failed"]
