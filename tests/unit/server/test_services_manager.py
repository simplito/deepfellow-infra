# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING
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
    InstallServiceIn,
    ListAllModelsFilters,
    ListServicesFilters,
    RetrieveServiceOut,
    UninstallServiceIn,
)
from server.services_manager import ServicesManager
from tests.unit.server.fakes import FakeService

if TYPE_CHECKING:
    from server.serviceprovider import ServiceRawConfig


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
    svc.get_model_install_progress = MagicMock(return_value=progress)
    services_manager.register_service(svc)

    result = await services_manager.get_model_install_progress("ollama", "llama3")

    assert result is progress
    assert svc.get_model_install_progress.call_count == 1
    assert svc.get_model_install_progress.call_args == call("default", "llama3")


@pytest.mark.asyncio
async def test_get_service_install_progress(services_manager: ServicesManager):
    svc = FakeService("ollama")
    progress = MagicMock()
    svc.get_instance_install_progress = MagicMock(return_value=progress)
    services_manager.register_service(svc)

    result = await services_manager.get_service_install_progress("ollama")

    assert result is progress
    assert svc.get_instance_install_progress.call_count == 1
    assert svc.get_instance_install_progress.call_args == call("default")


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
