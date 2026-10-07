# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, call

import pytest
from fastapi import HTTPException

from server.models.models import (
    AddCustomModelIn,
    InstallModelIn,
    ListModelsFilters,
    ListModelsOut,
    ModelSpecification,
    RetrieveModelOut,
    UninstallModelIn,
)
from server.models.services import (
    CatalogRefreshOut,
    InstallServiceIn,
    ListAllModelsFilters,
    ListServicesFilters,
    RetrieveServiceOut,
    UninstallServiceIn,
)
from server.services.base2_service import Base2Service
from server.services.mcp_service import McpService
from server.services.ollama_service import OllamaService
from server.services_manager import ServicesManager
from server.utils.core import PromiseWithProgress
from tests.unit.server.fakes import FakeService

if TYPE_CHECKING:
    from server.serviceprovider import ServiceRawConfig


@pytest.fixture
def mcp_svc() -> McpService:
    docker_svc = MagicMock()
    docker_svc.get_docker_subnet.return_value = "172.20.0.0/16"
    docker_svc.get_docker_container_name.side_effect = lambda name: f"df-{name}"  # pyright: ignore[reportUnknownLambdaType]
    return McpService(
        config=MagicMock(),
        endpoint_registry=MagicMock(),
        service_provider=MagicMock(),
        model_downloader=MagicMock(),
        docker_service=docker_svc,
        hardware=MagicMock(gpus=[], nvidia_gpus=[]),
    )


@pytest.mark.parametrize(
    ("input_id", "expected_tid", "expected_inst"),
    [
        ("my-service|inst_01", "my-service", "inst_01"),
        ("simple-service", "simple-service", "default"),
        ("123-456", "123-456", "default"),
        ("A-z_0-9", "A-z_0-9", "default"),
        ("a" * 64 + "|" + "b" * 64, "a" * 64, "b" * 64),
    ],
)
def test_split_success(input_id: str, expected_tid: str, expected_inst: str, services_manager: ServicesManager):
    """Test various valid formats and boundary lengths."""

    tid, inst = services_manager.split_service_type_and_instance(input_id)

    assert tid == expected_tid
    assert inst == expected_inst


@pytest.mark.parametrize(
    ("invalid_id", "expected_status", "error_part"),
    [
        # Character violations
        ("service.name", 400, "invalid characters"),
        ("service name|inst1", 400, "invalid characters"),
        ("serviceID|inst!", 400, "invalid characters"),
        ("service@domain", 400, "invalid characters"),
        # Length violations
        ("a" * 65, 400, "exceeds maximum length"),
        ("valid-id|" + "b" * 65, 400, "exceeds maximum length"),
        # Format violations
        ("part1|part2|part3", 404, "Incorrect service_id"),
        ("", 400, "invalid characters"),  # Empty string fails regex
    ],
)
def test_split_failures(invalid_id: str, expected_status: int, error_part: str, services_manager: ServicesManager):
    """Test that invalid inputs raise the correct HTTPException."""
    with pytest.raises(HTTPException) as exc:
        services_manager.split_service_type_and_instance(invalid_id)

    assert exc.value.status_code == expected_status
    assert error_part in exc.value.detail


def test_register_service(services_manager: ServicesManager):
    svc = FakeService("ollama")
    services_manager.register_service(svc)

    assert "ollama" in services_manager.services


def test_register_service_duplicate_raises(services_manager: ServicesManager):
    svc = FakeService("ollama")
    services_manager.register_service(svc)

    with pytest.raises(RuntimeError):
        services_manager.register_service(svc)


@pytest.mark.asyncio
async def test_drain_warning_tasks_awaits_base2_services_only(services_manager: ServicesManager):
    base2_svc = MagicMock(spec=Base2Service)
    base2_svc.drain_warning_tasks = AsyncMock()
    services_manager.services["llamacpp"] = base2_svc
    services_manager.services["ollama"] = FakeService("ollama")

    await services_manager.drain_warning_tasks()

    base2_svc.drain_warning_tasks.assert_awaited_once()


@pytest.mark.asyncio
async def test_drain_warning_tasks_noop_when_no_services(services_manager: ServicesManager):
    await services_manager.drain_warning_tasks()


@pytest.mark.asyncio
async def test_load_service_existing(services_manager: ServicesManager):
    svc = FakeService("ollama")
    svc.load_service = AsyncMock()
    services_manager.register_service(svc)
    cfg: ServiceRawConfig = {}

    await services_manager.load_service("ollama", cfg)

    assert svc.load_service.await_count == 1
    assert svc.load_service.await_args == call(cfg)


@pytest.mark.asyncio
async def test_load_service_missing_is_noop(services_manager: ServicesManager):
    # Should not raise even when service is unknown
    await services_manager.load_service("nonexistent", {})


def test_get_service_missing_raises(services_manager: ServicesManager):
    with pytest.raises(HTTPException) as exc:
        services_manager._get_service("missing")  # pyright: ignore[reportPrivateUsage]

    assert exc.value.status_code == 404


def test_get_service_returns_service(services_manager: ServicesManager):
    svc = FakeService("ollama")
    services_manager.register_service(svc)

    result = services_manager._get_service("ollama")  # pyright: ignore[reportPrivateUsage]

    assert result is svc  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_list_services_no_filter(services_manager: ServicesManager):
    services_manager.register_service(FakeService("svc-a"))
    services_manager.register_service(FakeService("svc-b", installed=False))

    result = await services_manager.list_services(ListServicesFilters(installed=None))

    assert len(result.list) == 2


@pytest.mark.asyncio
async def test_list_services_installed_filter(services_manager: ServicesManager):
    services_manager.register_service(FakeService("svc-a", installed=True))
    services_manager.register_service(FakeService("svc-b", installed=False))

    result = await services_manager.list_services(ListServicesFilters(installed=True))

    assert len(result.list) == 1
    assert result.list[0].type == "svc-a"


@pytest.mark.asyncio
async def test_get_service(services_manager: ServicesManager):
    svc = FakeService("ollama")
    services_manager.register_service(svc)

    out = await services_manager.get_service("ollama")

    assert isinstance(out, RetrieveServiceOut)
    assert out.type == "ollama"


@pytest.mark.asyncio
async def test_install_service(services_manager: ServicesManager):
    svc = FakeService("ollama")
    svc.install_instance = AsyncMock(return_value=MagicMock())
    services_manager.register_service(svc)
    options = InstallServiceIn(spec={})

    await services_manager.install_service("ollama", options)

    assert svc.install_instance.await_count == 1
    assert svc.install_instance.await_args == call("default", options)


@pytest.mark.asyncio
async def test_update_service(services_manager: ServicesManager):
    svc = FakeService("ollama")
    svc.update_instance = AsyncMock(return_value=MagicMock())
    services_manager.register_service(svc)
    options = InstallServiceIn(spec={})

    await services_manager.update_service("ollama", options)

    assert svc.update_instance.await_count == 1
    assert svc.update_instance.await_args == call("default", options)


@pytest.mark.asyncio
async def test_uninstall_service(services_manager: ServicesManager):
    svc = FakeService("ollama")
    svc.uninstall_instance = AsyncMock()
    services_manager.register_service(svc)
    options = UninstallServiceIn(purge=False)

    await services_manager.uninstall_service("ollama", options)

    assert svc.uninstall_instance.await_count == 1
    assert svc.uninstall_instance.await_args == call("default", options)


@pytest.mark.asyncio
async def test_get_docker_tags_for_service(services_manager: ServicesManager):
    svc = FakeService("ollama")
    svc.get_docker_tags = AsyncMock(return_value=["v1", "v2"])
    services_manager.register_service(svc)

    result = await services_manager.get_docker_tags_for_service("ollama", "gpu")

    svc.get_docker_tags.assert_awaited_once_with("gpu")
    assert result == ["v1", "v2"]


def test_get_default_docker_tag_for_service(services_manager: ServicesManager):
    svc = FakeService("ollama")
    svc.get_default_docker_tag = MagicMock(return_value="v1")
    services_manager.register_service(svc)

    result = services_manager.get_default_docker_tag_for_service("ollama", "gpu")

    svc.get_default_docker_tag.assert_called_once_with("gpu")
    assert result == "v1"


def test_get_docker_image_repo_for_service(services_manager: ServicesManager):
    svc = FakeService("ollama")
    svc.get_docker_image_repo = MagicMock(return_value="public.ecr.aws/q9t5s3a7/vllm-cpu-release-repo")
    services_manager.register_service(svc)

    result = services_manager.get_docker_image_repo_for_service("ollama", "cpu")

    svc.get_docker_image_repo.assert_called_once_with("cpu")
    assert result == "public.ecr.aws/q9t5s3a7/vllm-cpu-release-repo"


@pytest.mark.asyncio
async def test_list_models_from_all_services_installed_only(services_manager: ServicesManager):
    svc_installed = FakeService("svc-a", installed=True)
    svc_not_installed = FakeService("svc-b", installed=False)
    svc_installed.list_models = AsyncMock(
        return_value=ListModelsOut(
            list=[
                RetrieveModelOut(
                    id="m1",
                    service="svc-a",
                    type="llm",
                    installed=True,
                    downloaded=True,
                    size="small",
                    spec=ModelSpecification(fields=[]),
                    has_docker=False,
                )
            ]
        )
    )
    svc_not_installed.list_models = AsyncMock(return_value=ListModelsOut(list=[]))
    services_manager.register_service(svc_installed)
    services_manager.register_service(svc_not_installed)

    result = await services_manager.list_models_from_all_services(ListAllModelsFilters())

    assert svc_not_installed.list_models.await_count == 0
    assert len(result.list) == 1


@pytest.mark.asyncio
async def test_list_models_from_all_services_service_id_filter(services_manager: ServicesManager):
    svc_a = FakeService("svc-a", installed=True)
    svc_b = FakeService("svc-b", installed=True)
    svc_a.list_models = AsyncMock(
        return_value=ListModelsOut(
            list=[
                RetrieveModelOut(
                    id="m1",
                    service="svc-a",
                    type="llm",
                    installed=True,
                    downloaded=True,
                    size="small",
                    spec=ModelSpecification(fields=[]),
                    has_docker=False,
                )
            ]
        )
    )
    svc_b.list_models = AsyncMock(
        return_value=ListModelsOut(
            list=[
                RetrieveModelOut(
                    id="m2",
                    service="svc-b",
                    type="llm",
                    installed=True,
                    downloaded=True,
                    size="small",
                    spec=ModelSpecification(fields=[]),
                    has_docker=False,
                )
            ]
        )
    )
    services_manager.register_service(svc_a)
    services_manager.register_service(svc_b)

    result = await services_manager.list_models_from_all_services(ListAllModelsFilters(service_id="svc-a"))

    assert svc_b.list_models.await_count == 0
    assert len(result.list) == 1


@pytest.mark.asyncio
async def test_list_models_from_service(services_manager: ServicesManager):
    svc = FakeService("ollama")
    svc.list_models = AsyncMock(return_value=ListModelsOut(list=[]))
    services_manager.register_service(svc)
    filters = ListModelsFilters()

    await services_manager.list_models_from_service("ollama", filters)

    assert svc.list_models.await_count == 1
    assert svc.list_models.await_args == call("default", filters)


def _custom_model_out(custom_spec: dict[str, Any] | None) -> RetrieveModelOut:
    return RetrieveModelOut(
        id="my-custom",
        service="ollama",
        type="llm",
        installed=False,
        downloaded=True,
        size="small",
        spec=ModelSpecification(fields=[]),
        has_docker=True,
        custom="cm-1",
        custom_spec=custom_spec,
    )


@pytest.mark.asyncio
async def test_list_models_from_service_enriches_custom_spec(services_manager: ServicesManager):
    svc = FakeService("ollama")
    svc.list_models = AsyncMock(return_value=ListModelsOut(list=[_custom_model_out(None)]))
    svc.get_custom_model_definition = MagicMock(return_value={"id": "my-custom", "image": "test/image"})
    services_manager.register_service(svc)

    result = await services_manager.list_models_from_service("ollama", ListModelsFilters())

    svc.get_custom_model_definition.assert_called_once_with("cm-1")
    assert result.list[0].custom_spec == {"id": "my-custom", "image": "test/image"}


@pytest.mark.asyncio
async def test_list_models_from_service_default_definition_is_none(services_manager: ServicesManager):
    svc = FakeService("ollama")
    svc.list_models = AsyncMock(return_value=ListModelsOut(list=[_custom_model_out(None)]))
    services_manager.register_service(svc)

    result = await services_manager.list_models_from_service("ollama", ListModelsFilters())

    assert result.list[0].custom_spec is None


@pytest.mark.asyncio
async def test_list_models_from_service_preserves_existing_custom_spec(services_manager: ServicesManager):
    existing = {"kind": "user", "id": "my-custom"}
    svc = FakeService("ollama")
    svc.list_models = AsyncMock(return_value=ListModelsOut(list=[_custom_model_out(existing)]))
    svc.get_custom_model_definition = MagicMock(return_value={"should": "not be used"})
    services_manager.register_service(svc)

    result = await services_manager.list_models_from_service("ollama", ListModelsFilters())

    svc.get_custom_model_definition.assert_not_called()
    assert result.list[0].custom_spec == existing


@pytest.mark.asyncio
async def test_get_model_from_service(services_manager: ServicesManager):
    svc = FakeService("ollama")
    svc.get_model = AsyncMock(return_value=MagicMock())
    services_manager.register_service(svc)

    await services_manager.get_model_from_service("ollama", "llama3")

    assert svc.get_model.await_count == 1
    assert svc.get_model.await_args == call("default", "llama3")


@pytest.mark.asyncio
async def test_get_model_install_progress(services_manager: ServicesManager):
    svc = FakeService("ollama")
    progress = MagicMock()
    svc.get_model_install_progress = AsyncMock(return_value=progress)
    services_manager.register_service(svc)

    result = await services_manager.get_model_install_progress("ollama", "llama3")

    assert result is progress
    assert svc.get_model_install_progress.call_count == 1
    assert svc.get_model_install_progress.call_args == call("default", "llama3")


@pytest.mark.asyncio
async def test_get_service_install_progress(services_manager: ServicesManager):
    svc = FakeService("ollama")
    progress = MagicMock()
    svc.get_instance_install_progress = AsyncMock(return_value=progress)
    services_manager.register_service(svc)

    result = await services_manager.get_service_install_progress("ollama")

    assert result is progress
    assert svc.get_instance_install_progress.call_count == 1
    assert svc.get_instance_install_progress.call_args == call("default")


@pytest.mark.asyncio
async def test_cancel_service_install(services_manager: ServicesManager):
    svc = FakeService("ollama")
    svc.cancel_instance_install = AsyncMock()
    services_manager.register_service(svc)

    await services_manager.cancel_service_install("ollama")

    assert svc.cancel_instance_install.await_count == 1
    assert svc.cancel_instance_install.await_args == call("default")


@pytest.mark.asyncio
async def test_install_model_in_service(services_manager: ServicesManager):
    svc = FakeService("ollama")
    svc.install_model = AsyncMock(return_value=MagicMock())
    services_manager.register_service(svc)
    options = InstallModelIn()

    await services_manager.install_model_in_service("ollama", "llama3", options)

    assert svc.install_model.await_count == 1
    assert svc.install_model.await_args == call("default", "llama3", options)


@pytest.mark.asyncio
async def test_uninstall_model_from_service(services_manager: ServicesManager):
    svc = FakeService("ollama")
    svc.uninstall_model = AsyncMock()
    services_manager.register_service(svc)
    options = UninstallModelIn()

    await services_manager.uninstall_model_from_service("ollama", "llama3", options)

    assert svc.uninstall_model.await_count == 1
    assert svc.uninstall_model.await_args == call("default", "llama3", options)


@pytest.mark.asyncio
async def test_cancel_model_install(services_manager: ServicesManager):
    svc = FakeService("ollama")
    svc.cancel_model_install = AsyncMock()
    services_manager.register_service(svc)

    await services_manager.cancel_model_install("ollama", "llama3")

    assert svc.cancel_model_install.await_count == 1
    assert svc.cancel_model_install.await_args == call("default", "llama3")


@pytest.mark.asyncio
async def test_add_custom_model(services_manager: ServicesManager):
    svc = FakeService("ollama")
    svc.add_custom_model = AsyncMock(return_value="my-custom-id")
    services_manager.register_service(svc)
    options = AddCustomModelIn(spec={"name": "my-model"})

    result = await services_manager.add_custom_model("ollama", options)

    assert result == "my-custom-id"
    assert svc.add_custom_model.await_count == 1
    assert svc.add_custom_model.await_args == call("default", options)


@pytest.mark.asyncio
async def test_remove_custom_model(services_manager: ServicesManager):
    svc = FakeService("ollama")
    svc.remove_custom_model = AsyncMock()
    services_manager.register_service(svc)

    await services_manager.remove_custom_model("ollama", "my-custom-id")

    assert svc.remove_custom_model.await_count == 1
    assert svc.remove_custom_model.await_args == call("default", "my-custom-id")


@pytest.mark.asyncio
async def test_update_custom_model(services_manager: ServicesManager):
    svc = FakeService("ollama")
    svc.update_custom_model = AsyncMock()
    services_manager.register_service(svc)
    options = AddCustomModelIn(spec={"name": "my-model"})

    await services_manager.update_custom_model("ollama", "my-custom-id", options)

    assert svc.update_custom_model.await_count == 1
    assert svc.update_custom_model.await_args == call("default", "my-custom-id", options)


@pytest.mark.asyncio
async def test_edit_model(services_manager: ServicesManager):
    svc = FakeService("ollama")
    svc.edit_model = AsyncMock(return_value=None)
    services_manager.register_service(svc)
    new_definition = AddCustomModelIn(spec={"name": "my-model"})

    result = await services_manager.edit_model("ollama", "my-custom-id", new_definition)

    assert result is None
    assert svc.edit_model.await_count == 1
    assert svc.edit_model.await_args == call("default", "my-custom-id", new_definition)


@pytest.mark.asyncio
async def test_edit_model_install_options(services_manager: ServicesManager):
    svc = FakeService("ollama")
    promise = MagicMock()
    svc.edit_model_install_options = AsyncMock(return_value=(True, promise))
    services_manager.register_service(svc)
    new_options = InstallModelIn(spec={"prefix": "my-model"})

    result = await services_manager.edit_model_install_options("ollama", "my-model", new_options)

    assert result == (True, promise)
    assert svc.edit_model_install_options.await_count == 1
    assert svc.edit_model_install_options.await_args == call("default", "my-model", new_options)


@pytest.mark.asyncio
async def test_get_duplicate_spec(services_manager: ServicesManager):
    svc = FakeService("ollama")
    svc.get_duplicate_spec = AsyncMock(return_value={"id": "my-model"})
    services_manager.register_service(svc)

    result = await services_manager.get_duplicate_spec("ollama", "my-model")

    assert result == {"id": "my-model"}
    assert svc.get_duplicate_spec.await_count == 1
    assert svc.get_duplicate_spec.await_args == call("default", "my-model")


@pytest.mark.asyncio
async def test_sync_models_in_service(services_manager: ServicesManager):
    svc = FakeService("ollama")
    svc.sync_models = AsyncMock()
    services_manager.register_service(svc)

    await services_manager.sync_models_in_service("ollama")

    assert svc.sync_models.await_count == 1
    assert svc.sync_models.await_args == call("default")


@pytest.mark.asyncio
async def test_get_docker_logs(services_manager: ServicesManager):
    svc = FakeService("ollama")
    svc.get_docker_logs = AsyncMock(return_value="log output")
    services_manager.register_service(svc)

    result = await services_manager.get_docker_logs("ollama", None)

    assert result == "log output"
    assert svc.get_docker_logs.await_count == 1
    assert svc.get_docker_logs.await_args == call("default", None)


@pytest.mark.asyncio
async def test_get_docker_compose_file(services_manager: ServicesManager):
    svc = FakeService("ollama")
    svc.get_docker_compose_file = AsyncMock(return_value="compose yaml")
    services_manager.register_service(svc)

    result = await services_manager.get_docker_compose_file("ollama", None)

    assert result == "compose yaml"
    assert svc.get_docker_compose_file.await_count == 1
    assert svc.get_docker_compose_file.await_args == call("default", None)


@pytest.mark.asyncio
async def test_restart_docker(services_manager: ServicesManager):
    svc = FakeService("ollama")
    svc.restart_docker = AsyncMock()
    services_manager.register_service(svc)

    await services_manager.restart_docker("ollama", None)

    assert svc.restart_docker.await_count == 1
    assert svc.restart_docker.await_args == call("default", None)


@pytest.mark.asyncio
async def test_stop_all_services_stops_installed(services_manager: ServicesManager):
    svc = FakeService("ollama", installed=True)
    svc.stop_instance = AsyncMock()
    services_manager.register_service(svc)

    await services_manager.stop_all_services()

    assert svc.stop_instance.await_count == 1
    assert svc.stop_instance.await_args == call("default")


@pytest.mark.asyncio
async def test_stop_all_services_skips_not_installed(services_manager: ServicesManager):
    svc = FakeService("ollama", installed=False)
    svc.stop_instance = AsyncMock()
    services_manager.register_service(svc)

    await services_manager.stop_all_services()

    assert svc.stop_instance.await_count == 0


@pytest.mark.asyncio
async def test_stop_all_services_skips_empty_instances(services_manager: ServicesManager):
    svc = FakeService("ollama")
    svc.instances_info = {}
    svc.stop_instance = AsyncMock()
    services_manager.register_service(svc)

    await services_manager.stop_all_services()

    assert svc.stop_instance.await_count == 0


@pytest.mark.asyncio
async def test_stop_all_services_continues_on_error(services_manager: ServicesManager):
    svc_a = FakeService("svc-a", installed=True)
    svc_b = FakeService("svc-b", installed=True)
    svc_a.stop_instance = AsyncMock(side_effect=RuntimeError("boom"))
    svc_b.stop_instance = AsyncMock()
    services_manager.register_service(svc_a)
    services_manager.register_service(svc_b)

    # Should not raise even when one service fails
    await services_manager.stop_all_services()

    assert svc_b.stop_instance.await_count == 1
    assert svc_b.stop_instance.await_args == call("default")


@pytest.mark.asyncio
async def test_refresh_catalog_calls_refresh_catalog(services_manager: ServicesManager):
    svc = MagicMock(spec=OllamaService)
    svc.get_type = MagicMock(return_value="ollama")
    svc.refresh_catalog = AsyncMock(return_value=PromiseWithProgress(value=CatalogRefreshOut(added=5, total=700)))
    services_manager.register_service(svc)

    promise = await services_manager.refresh_catalog("ollama")
    result = await promise.wait()

    svc.refresh_catalog.assert_awaited_once()
    assert result.added == 5
    assert result.total == 700


@pytest.mark.asyncio
async def test_refresh_catalog_raises_405_for_non_ollama(services_manager: ServicesManager):
    svc = FakeService("vllm")
    services_manager.register_service(svc)

    with pytest.raises(HTTPException) as exc_info:
        await services_manager.refresh_catalog("vllm")

    assert exc_info.value.status_code == 405


@pytest.mark.asyncio
async def test_external_only_disabled_service_returns_409() -> None:
    services_manager = ServicesManager(external_only=True)
    services_manager.register_service(FakeService("openai"))

    assert "openai" in services_manager.services

    with pytest.raises(HTTPException) as exc_info:
        await services_manager.get_service("llamacpp")

    assert exc_info.value.status_code == 409
    assert "DF_EXTERNAL_ONLY" in exc_info.value.detail


@pytest.mark.asyncio
async def test_non_external_only_unknown_service_returns_404() -> None:
    services_manager = ServicesManager()

    with pytest.raises(HTTPException) as exc_info:
        await services_manager.get_service("llamacpp")

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_external_only_docker_tags_returns_409() -> None:
    services_manager = ServicesManager(external_only=True)
    services_manager.register_service(FakeService("openai"))

    with pytest.raises(HTTPException) as exc_info:
        await services_manager.get_docker_tags_for_service("openai", "cpu")

    assert exc_info.value.status_code == 409
    assert "DF_EXTERNAL_ONLY" in exc_info.value.detail
