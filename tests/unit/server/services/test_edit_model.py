# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Tests for Base2Service.edit_model / edit_model_install_options (DFINFRA-271)."""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from server.docker import DockerOptions
from server.models.models import AddCustomModelIn, InstallModelIn, UninstallModelIn
from server.models.services import InstallServiceIn
from server.services.base2_service import CustomModel
from server.services.custom_service import CustomService, InstalledInfo, ModelInstalledInfo

_CUSTOM_MODEL_DATA: dict[str, Any] = {
    "id": "my-custom",
    "default_prefix": "my-custom",
    "size": "1GB",
    "image": "test/image:latest",
    "image_port": 8000,
}


@pytest.fixture
def deps() -> dict[str, Any]:
    docker_svc = MagicMock()
    docker_svc.get_docker_subnet.return_value = "172.20.0.0/16"
    docker_svc.get_docker_container_name.side_effect = lambda name: f"df-{name}"  # pyright: ignore[reportUnknownLambdaType]
    docker_svc.get_image_warnings = AsyncMock(return_value=[])
    docker_svc.is_docker_image_pulled = AsyncMock(return_value=True)
    docker_svc.install_and_run_docker = AsyncMock(return_value=(8090, True, False))
    docker_svc.get_container_host.return_value = "172.20.0.1"
    docker_svc.get_container_port.return_value = 8090
    docker_svc.uninstall_docker = AsyncMock()
    service_provider = MagicMock()
    service_provider.add_warning = AsyncMock()
    service_provider.dismiss_warnings_matching = AsyncMock()
    service_provider.dismiss_warnings_matching_any = AsyncMock()
    service_provider.dismiss_warnings_for_instance = AsyncMock()
    endpoint_registry = MagicMock()
    endpoint_registry.register_custom_endpoint_as_proxy.side_effect = lambda **kwargs: kwargs["registration_options"].id  # pyright: ignore[reportUnknownLambdaType, reportUnknownMemberType]
    return {
        "config": MagicMock(),
        "endpoint_registry": endpoint_registry,
        "service_provider": service_provider,
        "model_downloader": MagicMock(),
        "docker_service": docker_svc,
        "hardware": MagicMock(gpus=[]),
        "services_manager": MagicMock(),
    }


@pytest.fixture
def svc(deps: dict[str, Any]) -> CustomService:
    service = CustomService(**deps)
    service.service_provider.save_service_config = AsyncMock()
    service.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    return service


async def _add_and_install(svc: CustomService, custom_model_id: str = "cm-1", data: dict[str, Any] | None = None) -> CustomModel:
    data = dict(data or _CUSTOM_MODEL_DATA)
    model = CustomModel(id=custom_model_id, data=data)
    svc._add_custom_model("default", model)  # pyright: ignore[reportPrivateUsage]
    svc.instances_info["default"].config.custom = [model]

    promise = await svc.install_model("default", data["id"], InstallModelIn(spec={"prefix": data["default_prefix"]}))
    await promise.wait()
    return model


def _custom_definitions(svc: CustomService) -> list[CustomModel]:
    custom = svc.instances_info["default"].config.custom
    assert custom is not None
    return custom


def _installed_models(svc: CustomService) -> dict[str, ModelInstalledInfo]:
    installed = svc.instances_info["default"].installed
    assert installed is not None
    return installed.models


@pytest.mark.asyncio
async def test_edit_model_not_found_raises_404(svc: CustomService) -> None:
    with pytest.raises(HTTPException) as exc:
        await svc.edit_model("default", "ghost", AddCustomModelIn(spec={"name": "x"}))

    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_edit_model_definition_without_id_raises_400(svc: CustomService) -> None:
    data_without_id = {k: v for k, v in _CUSTOM_MODEL_DATA.items() if k != "id"}
    model = CustomModel(id="cm-1", data=data_without_id)
    svc.instances_info["default"].config.custom = [model]

    with pytest.raises(HTTPException) as exc:
        await svc.edit_model("default", "cm-1", AddCustomModelIn(spec=_CUSTOM_MODEL_DATA))

    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_edit_model_update_failure_skips_rollback_install_when_not_installed(svc: CustomService) -> None:
    """The update-failure except branch only attempts a rollback reinstall `if was_installed` - for a
    model that was never installed to begin with, it must just re-raise, not try to (re)install it."""
    model = CustomModel(id="cm-1", data=dict(_CUSTOM_MODEL_DATA))
    svc._add_custom_model("default", model)  # pyright: ignore[reportPrivateUsage]
    svc.instances_info["default"].config.custom = [model]

    # `default_prefix` violates the required pattern, so "apply new definition" fails validation,
    # with the model not installed (id stays the same - it's not the id change being rejected here).
    invalid_data = {**_CUSTOM_MODEL_DATA, "default_prefix": "bad prefix!"}
    with pytest.raises(HTTPException):
        await svc.edit_model("default", "cm-1", AddCustomModelIn(spec=invalid_data))

    assert svc.instances_info["default"].config.custom[0].data["id"] == "my-custom"
    installed = svc.instances_info["default"].installed
    assert installed is not None
    assert installed.models == {}


@pytest.mark.asyncio
async def test_edit_model_rejects_id_change_without_uninstalling(svc: CustomService, deps: dict[str, Any]) -> None:
    """Changing the model's `id` mid-edit would leave `edit_model` reinstalling under a now-stale id
    (the id it captured from the OLD definition) - reject it upfront, before any uninstall happens,
    rather than letting the uninstall run for nothing and then failing on reinstall."""
    await _add_and_install(svc)

    with pytest.raises(HTTPException) as exc:
        await svc.edit_model("default", "cm-1", AddCustomModelIn(spec={**_CUSTOM_MODEL_DATA, "id": "other-id"}))

    assert exc.value.status_code == 400
    deps["docker_service"].uninstall_docker.assert_not_called()
    assert _custom_definitions(svc)[0].data["id"] == "my-custom"
    assert "my-custom" in _installed_models(svc)


@pytest.mark.asyncio
async def test_edit_model_skips_reinstall_when_not_installed(svc: CustomService) -> None:
    model = CustomModel(id="cm-1", data=dict(_CUSTOM_MODEL_DATA))
    svc._add_custom_model("default", model)  # pyright: ignore[reportPrivateUsage]
    svc.instances_info["default"].config.custom = [model]

    new_data = {**_CUSTOM_MODEL_DATA, "image": "test/updated:latest"}
    result = await svc.edit_model("default", "cm-1", AddCustomModelIn(spec=new_data))

    assert result is None
    options = svc.models["default"]["my-custom"].options
    assert isinstance(options, DockerOptions)
    assert options.image == "test/updated:latest"


@pytest.mark.asyncio
async def test_edit_model_reinstalls_when_installed(svc: CustomService, deps: dict[str, Any]) -> None:
    await _add_and_install(svc)
    assert "my-custom" in _installed_models(svc)

    new_data = {**_CUSTOM_MODEL_DATA, "image": "test/updated:latest"}
    promise = await svc.edit_model("default", "cm-1", AddCustomModelIn(spec=new_data))
    assert promise is not None
    result = await promise.wait()

    assert result.status == "OK"
    assert _custom_definitions(svc)[0].data["image"] == "test/updated:latest"
    assert "my-custom" in _installed_models(svc)
    deps["docker_service"].uninstall_docker.assert_called_once()
    assert deps["docker_service"].install_and_run_docker.call_count == 2


@pytest.mark.asyncio
async def test_edit_model_reinstalls_with_updated_default_prefix(svc: CustomService, deps: dict[str, Any]) -> None:
    """Changing `default_prefix` during an edit must carry through to the reinstalled model's
    install-time `prefix` option - otherwise the definition changes but the actually-registered
    endpoint prefix (and the WebUI's "Configuration" column, which reads `options.spec`) stays stale."""
    await _add_and_install(svc)

    new_data = {**_CUSTOM_MODEL_DATA, "default_prefix": "new-prefix"}
    promise = await svc.edit_model("default", "cm-1", AddCustomModelIn(spec=new_data))
    assert promise is not None
    result = await promise.wait()

    assert result.status == "OK"
    installed = _installed_models(svc)["my-custom"]
    assert installed.options.spec is not None
    assert installed.options.spec["prefix"] == "new-prefix"


@pytest.mark.asyncio
async def test_edit_model_self_heals_previously_drifted_prefix(svc: CustomService, deps: dict[str, Any]) -> None:
    """`prefix` must always follow `default_prefix` on edit - even if it had already drifted away from
    it for some other reason (e.g. a past edit under a since-fixed bug) - since a custom-backed model
    has no user-facing way to intentionally diverge the two, there's nothing worth "preserving" here,
    and permanently locking the prefix in on the first mismatch would be its own (unrecoverable) bug."""
    model = CustomModel(id="cm-1", data=dict(_CUSTOM_MODEL_DATA))
    svc._add_custom_model("default", model)  # pyright: ignore[reportPrivateUsage]
    svc.instances_info["default"].config.custom = [model]
    install_promise = await svc.install_model("default", "my-custom", InstallModelIn(spec={"prefix": "already-drifted"}))
    await install_promise.wait()

    new_data = {**_CUSTOM_MODEL_DATA, "default_prefix": "new-prefix"}
    promise = await svc.edit_model("default", "cm-1", AddCustomModelIn(spec=new_data))
    assert promise is not None
    await promise.wait()

    installed = _installed_models(svc)["my-custom"]
    assert installed.options.spec is not None
    assert installed.options.spec["prefix"] == "new-prefix"


@pytest.mark.asyncio
async def test_edit_model_rollback_keeps_old_prefix_after_default_prefix_change_failure(svc: CustomService, deps: dict[str, Any]) -> None:
    """The reinstall attempt uses a recomputed prefix, but a failed reinstall must roll back using the
    original (pre-edit) install options - including its original `prefix` - not the half-applied one."""
    await _add_and_install(svc)
    deps["docker_service"].install_and_run_docker = AsyncMock(side_effect=[RuntimeError("boom"), (8090, True, False)])

    with pytest.raises(RuntimeError, match="boom"):
        await svc.edit_model("default", "cm-1", AddCustomModelIn(spec={**_CUSTOM_MODEL_DATA, "default_prefix": "new-prefix"}))

    installed = _installed_models(svc)["my-custom"]
    assert installed.options.spec is not None
    assert installed.options.spec["prefix"] == "my-custom"


@pytest.mark.asyncio
async def test_edit_model_restores_definition_and_install_on_reinstall_failure(svc: CustomService, deps: dict[str, Any]) -> None:
    await _add_and_install(svc)

    deps["docker_service"].install_and_run_docker = AsyncMock(side_effect=[RuntimeError("boom"), (8090, True, False)])

    with pytest.raises(RuntimeError, match="boom"):
        await svc.edit_model("default", "cm-1", AddCustomModelIn(spec={**_CUSTOM_MODEL_DATA, "image": "test/updated:latest"}))

    assert _custom_definitions(svc)[0].data["image"] == _CUSTOM_MODEL_DATA["image"]
    assert "my-custom" in _installed_models(svc)


@pytest.mark.asyncio
async def test_edit_model_logs_original_failure_when_reinstall_fails_but_rollback_succeeds(
    svc: CustomService, deps: dict[str, Any]
) -> None:
    """Same as the update-failure case above, but for the reinstall step: the reinstall itself fails,
    the best-effort rollback (restore old definition + reinstall) succeeds, and the original failure
    must still be logged."""
    await _add_and_install(svc)
    deps["docker_service"].install_and_run_docker = AsyncMock(side_effect=[RuntimeError("boom"), (8090, True, False)])

    with patch("server.services.base2_service.logger") as mock_logger:
        with pytest.raises(RuntimeError, match="boom"):
            await svc.edit_model("default", "cm-1", AddCustomModelIn(spec={**_CUSTOM_MODEL_DATA, "image": "test/updated:latest"}))
        mock_logger.exception.assert_called_once()
        assert "failed to reinstall model" in mock_logger.exception.call_args.args[0]


@pytest.mark.asyncio
async def test_edit_model_restores_definition_when_update_fails(svc: CustomService, deps: dict[str, Any]) -> None:
    await _add_and_install(svc)

    # `default_prefix` violates the required pattern, so `_add_custom_model` (inside `_update_custom_model`)
    # rejects it after "my-custom" has already been removed from the live registry and uninstalled.
    invalid_data = {**_CUSTOM_MODEL_DATA, "default_prefix": "bad prefix!"}
    with pytest.raises(HTTPException):
        await svc.edit_model("default", "cm-1", AddCustomModelIn(spec=invalid_data))

    # Definition and install state are restored to the original ("my-custom").
    assert _custom_definitions(svc)[0].data["id"] == "my-custom"
    assert "my-custom" in _installed_models(svc)


@pytest.mark.asyncio
async def test_edit_model_logs_original_failure_even_when_rollback_succeeds(svc: CustomService, deps: dict[str, Any]) -> None:
    """The common case - the edit itself fails, but the best-effort rollback succeeds and restores the
    model - must still log the original failure. Nothing else logs it: `error_handlers.py` only
    translates exceptions to HTTP responses, so without this the admin sees a bare 400 with no trace
    in the server logs of what actually went wrong."""
    await _add_and_install(svc)

    invalid_data = {**_CUSTOM_MODEL_DATA, "default_prefix": "bad prefix!"}
    with patch("server.services.base2_service.logger") as mock_logger:
        with pytest.raises(HTTPException):
            await svc.edit_model("default", "cm-1", AddCustomModelIn(spec=invalid_data))
        mock_logger.exception.assert_called_once()
        assert "failed to apply new definition" in mock_logger.exception.call_args.args[0]


@pytest.mark.asyncio
async def test_edit_model_logs_when_rollback_reinstall_fails_after_update_failure(svc: CustomService, deps: dict[str, Any]) -> None:
    """When the update itself fails and the best-effort rollback reinstall ALSO fails, that second
    failure must be logged - not silently swallowed - since the admin otherwise only sees the original
    error with no trace the model wasn't actually restored."""
    await _add_and_install(svc)
    deps["docker_service"].install_and_run_docker = AsyncMock(side_effect=RuntimeError("rollback boom"))

    invalid_data = {**_CUSTOM_MODEL_DATA, "default_prefix": "bad prefix!"}
    with patch("server.services.base2_service.logger") as mock_logger:
        with pytest.raises(HTTPException):
            await svc.edit_model("default", "cm-1", AddCustomModelIn(spec=invalid_data))
        assert mock_logger.exception.call_count == 2
        logged = [call.args[0] for call in mock_logger.exception.call_args_list]
        assert any("failed to apply new definition" in msg for msg in logged)
        assert any("failed to roll back model" in msg for msg in logged)


@pytest.mark.asyncio
async def test_edit_model_logs_when_rollback_fails_after_reinstall_failure(svc: CustomService, deps: dict[str, Any]) -> None:
    """When the reinstall itself fails and the best-effort rollback (restore old definition + reinstall)
    ALSO fails, that second failure must be logged - not silently swallowed."""
    await _add_and_install(svc)
    deps["docker_service"].install_and_run_docker = AsyncMock(side_effect=RuntimeError("boom"))

    with patch("server.services.base2_service.logger") as mock_logger:
        with pytest.raises(RuntimeError, match="boom"):
            await svc.edit_model("default", "cm-1", AddCustomModelIn(spec={**_CUSTOM_MODEL_DATA, "image": "test/updated:latest"}))
        assert mock_logger.exception.call_count == 2
        logged = [call.args[0] for call in mock_logger.exception.call_args_list]
        assert any("failed to reinstall model" in msg for msg in logged)
        assert any("failed to roll back model" in msg for msg in logged)


@pytest.mark.asyncio
async def test_edit_model_install_options_logs_when_rollback_fails(svc: CustomService, deps: dict[str, Any]) -> None:
    """When edit_model_install_options's reinstall fails and the best-effort rollback ALSO fails, that
    second failure must be logged - not silently swallowed."""
    await _add_and_install(svc)
    deps["docker_service"].install_and_run_docker = AsyncMock(side_effect=RuntimeError("boom"))

    with patch("server.services.base2_service.logger") as mock_logger:
        with pytest.raises(RuntimeError, match="boom"):
            await svc.edit_model_install_options("default", "my-custom", InstallModelIn(spec={"prefix": "new-prefix"}))
        assert mock_logger.exception.call_count == 2
        logged = [call.args[0] for call in mock_logger.exception.call_args_list]
        assert any("failed to reinstall model" in msg for msg in logged)
        assert any("failed to roll back model" in msg for msg in logged)


@pytest.mark.asyncio
async def test_remove_custom_model_clears_edit_lock(svc: CustomService) -> None:
    """`_edit_locks` entries must not accumulate forever - removing a custom model should drop its
    lock, the same way `_failed_models`/`_crash_poll_state` get cleaned up elsewhere in this class."""
    await _add_and_install(svc)
    async with svc._get_edit_lock("default", "my-custom"):  # pyright: ignore[reportPrivateUsage]
        pass
    assert ("default", "my-custom") in svc._edit_locks  # pyright: ignore[reportPrivateUsage]

    await svc.uninstall_model("default", "my-custom", UninstallModelIn(purge=False))
    await svc.remove_custom_model("default", "cm-1")

    assert ("default", "my-custom") not in svc._edit_locks  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_edit_model_serializes_concurrent_edits(svc: CustomService, deps: dict[str, Any]) -> None:
    await _add_and_install(svc)

    call_order: list[str] = []
    real_install_and_run_docker = deps["docker_service"].install_and_run_docker

    async def slow_install(*args: Any, **kwargs: Any) -> tuple[int, bool]:
        call_order.append("install-start")
        await asyncio.sleep(0.01)
        call_order.append("install-end")
        return await real_install_and_run_docker(*args, **kwargs)

    deps["docker_service"].install_and_run_docker = AsyncMock(side_effect=slow_install)

    async def run_edit(image: str) -> None:
        promise = await svc.edit_model("default", "cm-1", AddCustomModelIn(spec={**_CUSTOM_MODEL_DATA, "image": image}))
        assert promise is not None
        await promise.wait()

    await asyncio.gather(run_edit("test/a:latest"), run_edit("test/b:latest"))

    # If the two edits interleaved, "install-start" would appear twice before any "install-end".
    starts_before_first_end = call_order[: call_order.index("install-end")].count("install-start")
    assert starts_before_first_end == 1


@pytest.mark.asyncio
async def test_edit_model_install_options_rejects_colliding_prefix(svc: CustomService) -> None:
    """Base2Service.edit_model_install_options previously had no prefix-collision check at all (only
    the add-model paths did) - editing a catalog model's install-time prefix to match another
    model's would silently register both endpoints under the same path."""
    model = CustomModel(id="cm-1", data=dict(_CUSTOM_MODEL_DATA))
    svc._add_custom_model("default", model)  # pyright: ignore[reportPrivateUsage]
    other = CustomModel(id="cm-2", data={**_CUSTOM_MODEL_DATA, "id": "other-custom", "default_prefix": "other-custom"})
    svc._add_custom_model("default", other)  # pyright: ignore[reportPrivateUsage]

    with pytest.raises(HTTPException) as exc:
        await svc.edit_model_install_options("default", "my-custom", InstallModelIn(spec={"prefix": "other-custom"}))

    assert exc.value.status_code == 400
    assert _installed_models(svc) == {}


@pytest.mark.asyncio
async def test_edit_model_install_options_unknown_model_reaches_install_error(svc: CustomService) -> None:
    """With no spec prefix and no matching model to fall back to a default_prefix from, the upfront
    collision check is skipped entirely (nothing to check) - the underlying install still catches an
    unknown model_id and raises its own 'Model not found'."""
    with pytest.raises(HTTPException) as exc:
        await svc.edit_model_install_options("default", "ghost", InstallModelIn())

    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_edit_model_install_options_not_installed_still_installs(svc: CustomService) -> None:
    model = CustomModel(id="cm-1", data=dict(_CUSTOM_MODEL_DATA))
    svc._add_custom_model("default", model)  # pyright: ignore[reportPrivateUsage]
    svc.instances_info["default"].config.custom = [model]

    was_installed, promise = await svc.edit_model_install_options("default", "my-custom", InstallModelIn(spec={"prefix": "new-prefix"}))
    result = await promise.wait()

    assert result.status == "OK"
    assert was_installed is False
    assert "my-custom" in _installed_models(svc)


@pytest.mark.asyncio
async def test_edit_model_install_options_failure_skips_rollback_when_not_installed(svc: CustomService, deps: dict[str, Any]) -> None:
    """The failure-except branch only attempts a rollback reinstall `if was_installed` - for a model
    that was never installed to begin with, it must just re-raise, not try to reinstall the old options
    (there's no previous install to restore)."""
    model = CustomModel(id="cm-1", data=dict(_CUSTOM_MODEL_DATA))
    svc._add_custom_model("default", model)  # pyright: ignore[reportPrivateUsage]
    deps["docker_service"].get_image_warnings = AsyncMock(return_value=["some warning"])

    with pytest.raises(HTTPException):
        await svc.edit_model_install_options("default", "my-custom", InstallModelIn(spec={"prefix": "my-custom"}))

    assert _installed_models(svc) == {}


@pytest.mark.asyncio
async def test_edit_model_install_options_reinstalls_with_new_options(svc: CustomService, deps: dict[str, Any]) -> None:
    await _add_and_install(svc)

    was_installed, promise = await svc.edit_model_install_options("default", "my-custom", InstallModelIn(spec={"prefix": "new-prefix"}))
    result = await promise.wait()

    assert result.status == "OK"
    assert was_installed is True
    deps["docker_service"].uninstall_docker.assert_called_once()
    assert deps["docker_service"].install_and_run_docker.call_count == 2
    installed = _installed_models(svc)["my-custom"]
    assert installed.options.spec is not None
    assert installed.options.spec["prefix"] == "new-prefix"


@pytest.mark.asyncio
async def test_edit_model_install_options_restores_old_options_on_failure(svc: CustomService, deps: dict[str, Any]) -> None:
    await _add_and_install(svc)

    deps["docker_service"].install_and_run_docker = AsyncMock(side_effect=[RuntimeError("boom"), (8090, True, False)])

    with pytest.raises(RuntimeError, match="boom"):
        await svc.edit_model_install_options("default", "my-custom", InstallModelIn(spec={"prefix": "new-prefix"}))

    installed = _installed_models(svc)["my-custom"]
    assert installed.options.spec is not None
    assert installed.options.spec["prefix"] == "my-custom"


@pytest.mark.asyncio
async def test_edit_model_install_options_logs_original_failure_when_rollback_succeeds(svc: CustomService, deps: dict[str, Any]) -> None:
    """Same as edit_model's rollback-succeeds cases above, but for edit_model_install_options: the
    reinstall fails, the best-effort rollback succeeds, and the original failure must still be logged."""
    await _add_and_install(svc)
    deps["docker_service"].install_and_run_docker = AsyncMock(side_effect=[RuntimeError("boom"), (8090, True, False)])

    with patch("server.services.base2_service.logger") as mock_logger:
        with pytest.raises(RuntimeError, match="boom"):
            await svc.edit_model_install_options("default", "my-custom", InstallModelIn(spec={"prefix": "new-prefix"}))
        mock_logger.exception.assert_called_once()
        assert "failed to reinstall model" in mock_logger.exception.call_args.args[0]


@pytest.mark.asyncio
async def test_edit_model_install_options_reinstalls_with_carried_over_options(svc: CustomService) -> None:
    """The `_carry_over_edited_install_options` hook sees the old options and its result is what gets reinstalled."""
    await _add_and_install(svc)
    carried = InstallModelIn(spec={"prefix": "new-prefix", "kept": "yes"})

    with patch.object(svc, "_carry_over_edited_install_options", return_value=carried) as hook:
        _, promise = await svc.edit_model_install_options("default", "my-custom", InstallModelIn(spec={"prefix": "new-prefix"}))
        await promise.wait()

    old_options, new_options = hook.call_args.args
    assert old_options.spec == {"prefix": "my-custom"}
    assert new_options.spec == {"prefix": "new-prefix"}
    assert _installed_models(svc)["my-custom"].options.spec == {"prefix": "new-prefix", "kept": "yes"}


async def _edit_capturing_kept_utilization(svc: CustomService, old_spec: dict[str, Any], new_spec: dict[str, Any]) -> float | None:
    await _add_and_install(svc)
    installed = _installed_models(svc)["my-custom"]
    installed.options = InstallModelIn(spec=old_spec)
    installed.gpu_memory_utilization = 0.4  # pyright: ignore[reportAttributeAccessIssue]
    seen: list[float | None] = []

    async def fake_install(instance: str, model_id: str, options: InstallModelIn) -> MagicMock:
        seen.append(svc._edit_kept_gpu_utilization.get((instance, model_id)))  # pyright: ignore[reportPrivateUsage]
        return MagicMock()

    with patch.object(svc, "_install_model_and_wait", side_effect=fake_install):
        await svc.edit_model_install_options("default", "my-custom", InstallModelIn(spec=new_spec))

    assert ("default", "my-custom") not in svc._edit_kept_gpu_utilization  # pyright: ignore[reportPrivateUsage]
    return seen[0]


@pytest.mark.asyncio
async def test_edit_model_install_options_keeps_gpu_utilization_when_priority_unchanged(svc: CustomService) -> None:
    kept = await _edit_capturing_kept_utilization(svc, {"gpu_priority": "0.5"}, {"gpu_priority": 0.5, "prefix": "new-prefix"})

    assert kept == 0.4


@pytest.mark.asyncio
async def test_edit_model_install_options_drops_gpu_utilization_when_priority_changed(svc: CustomService) -> None:
    kept = await _edit_capturing_kept_utilization(svc, {"gpu_priority": 0.5}, {"gpu_priority": 0.8})

    assert kept is None
