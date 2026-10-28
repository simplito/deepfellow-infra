# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

import asyncio
import json
import time
from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import ANY, AsyncMock, MagicMock, call, patch

import aiohttp
import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from scripts.get_huggingface_models import HfRepoAccessError, ModelIncompatibleError
from scripts.utils.hf_common import HfErrorKind
from server.docker import ContainerStatus
from server.models.models import AddCustomModelIn, InstallModelIn, ListModelsFilters, UninstallModelIn
from server.models.services import GpuStats, InstallServiceIn, UninstallServiceIn
from server.services.base2_service import CustomModel, Instance, InstanceConfig, ModelConfig
from server.services.finetune_service import FinetuneService
from server.services.model_catalog_refresh import huggingface_refresh_guard
from server.services.vllm_service import (
    AdapterMetadata,
    DownloadedInfo,
    InstalledInfo,
    ModelInstalledInfo,
    VllmModel,
    VllmModelOptions,
    VllmOptions,
    VllmService,
    _const,  # pyright: ignore[reportPrivateUsage]
    _read_models,  # pyright: ignore[reportPrivateUsage]
    _round_up_lora_rank,  # pyright: ignore[reportPrivateUsage]
)
from server.utils.core import (
    CommandResult,
    DownloadedPacket,
    PreDownloadPacket,
    Stream,
    StreamChunk,
    StreamChunkProgress,
    SuccessDownloadPacket,
)
from server.utils.docker_image_version import newest_docker_tag
from server.utils.exceptions import AppError
from server.utils.hardware import NvidiaGpuInfo
from server.utils.sentry import UpstreamError
from tests.unit.server.real_docker_tags import real_tags


@pytest.fixture
def deps() -> dict[str, Any]:
    hw = MagicMock()
    hw.cpu.avx512 = True
    hw.gpus = []
    hw.nvidia_gpus = []
    hw.intel_gpus = []
    hw.gpu_load_lock = asyncio.Lock()
    finetune_service = MagicMock(spec=FinetuneService)
    finetune_service.get_installed_base_url.return_value = "http://df-finetune:8333"
    services_manager = MagicMock()
    services_manager.services = {"finetune": finetune_service}
    return {
        "config": MagicMock(),
        "endpoint_registry": MagicMock(),
        "service_provider": MagicMock(),
        "model_downloader": MagicMock(),
        "docker_service": MagicMock(),
        "hardware": hw,
        "services_manager": services_manager,
    }


@pytest.fixture
def svc(deps: dict[str, Any]) -> VllmService:
    return VllmService(**deps)


def _make_installed_info(hardware: str | bool | None = False) -> InstalledInfo:
    return InstalledInfo(
        models={},
        options=InstallServiceIn(spec={}),
        parsed_options=VllmOptions(hardware=hardware),
    )


def _make_model_installed_info(
    model_id: str = "test-model",
    registration_id: str = "reg-1",
    gpu_memory_utilization: float | None = None,
    model_type: str = "llm",
) -> ModelInstalledInfo:
    docker = MagicMock()
    docker.name = f"df-vllm-{model_id}"
    docker.container_name = f"df-vllm-{model_id}"
    return ModelInstalledInfo(
        id=model_id,
        registered_name=model_id,
        options=InstallModelIn(spec={}),
        docker=docker,
        container_host="localhost",
        container_port=8000,
        docker_exposed_port=8000,
        registration_id=registration_id,
        model_path=Path("/tmp/model"),
        base_url="http://localhost:8000",
        gpu_memory_utilization=gpu_memory_utilization,
        model_type=model_type,  # type: ignore[arg-type]
    )


def _mock_adapter_registry_session_cls(status: int, json_data: dict[str, Any] | None = None) -> MagicMock:
    """Build a mock `aiohttp.ClientSession` class whose `session.get(url)` yields a mock response."""
    mock_response = AsyncMock()
    mock_response.status = status
    mock_response.raise_for_status = MagicMock()
    mock_response.json = AsyncMock(return_value=json_data or {})
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.get = MagicMock(return_value=mock_response)

    return MagicMock(return_value=mock_session)


def _setup_install_mocks(svc: VllmService, deps: dict[str, Any], hardware: str | bool | None = False) -> InstalledInfo:
    installed = _make_installed_info(hardware=hardware)
    svc.instances_info["default"].installed = installed
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=(8000, True, False))
    deps["docker_service"].get_docker_subnet.return_value = None
    deps["docker_service"].get_docker_container_name.return_value = "container"
    deps["docker_service"].get_container_host.return_value = "localhost"
    deps["docker_service"].get_container_port.return_value = 8000
    return installed


def test_get_type(svc: VllmService) -> None:
    assert svc.get_type() == "vllm"


def test_get_description_not_empty(svc: VllmService) -> None:
    assert svc.get_description()


def test_default_models_loaded(svc: VllmService) -> None:
    assert "default" in svc.models
    assert len(svc.models["default"]) > 0


def test_get_size_cpu_only(deps: dict[str, Any]) -> None:
    deps["hardware"].cpu.avx512 = True
    deps["hardware"].gpus = []
    svc = VllmService(**deps)

    sizes = svc.get_size()

    assert "cpu" in sizes
    assert "gpu" not in sizes


def test_get_size_with_gpu(deps: dict[str, Any]) -> None:
    nvidia = MagicMock(spec=NvidiaGpuInfo)
    deps["hardware"].gpus = [nvidia]
    svc = VllmService(**deps)

    sizes = svc.get_size()

    assert "gpu" in sizes


def test_get_spec_has_hardware_field(svc: VllmService) -> None:
    spec = svc.get_spec()

    field_names = [f.name for f in spec.fields]
    assert "hardware" in field_names


def test_const_has_cpu_and_gpu_images() -> None:
    assert "cpu" in _const.images
    assert "gpu" in _const.images
    assert _const.images["cpu"].name
    assert _const.images["gpu"].name


def test_get_model_type_spec_baseline_fields(svc: VllmService) -> None:
    installed = _make_installed_info(hardware=False)
    svc.instances_info["default"].installed = installed

    spec = svc.get_model_type_spec("default", "llm")

    field_names = [f.name for f in spec.fields]
    for expected in ("rename", "max_model_length", "quantization", "extra_args", "extra_envs"):
        assert expected in field_names


def test_get_model_type_spec_adds_gpu_priority_for_gpu_hardware(svc: VllmService) -> None:
    svc.instances_info["default"].config.options = InstallServiceIn(spec={"hardware": True})

    spec = svc.get_model_type_spec("default", "llm")

    field = next(f for f in spec.fields if f.name == "gpu_priority")
    assert field.type == "gpu-priority"


def test_get_model_type_spec_omits_gpu_priority_for_cpu_hardware(svc: VllmService) -> None:
    spec = svc.get_model_type_spec("default", "llm")

    field_names = [f.name for f in spec.fields]
    assert "gpu_priority" not in field_names


def test_get_model_type_spec_reranker_default_extra_args(svc: VllmService) -> None:
    spec = svc.get_model_type_spec("default", "reranker")

    extra_args_field = next(f for f in spec.fields if f.name == "extra_args")
    assert extra_args_field.default is not None
    assert "--trust-remote-code" in (extra_args_field.default or "")


def test_get_model_spec_stt_has_no_max_model_length(svc: VllmService) -> None:
    stt_names = [f.name for f in svc.get_model_type_spec("default", "stt").fields]
    llm_names = [f.name for f in svc.get_model_type_spec("default", "llm").fields]

    assert "max_model_length" not in stt_names
    assert "max_model_length" in llm_names
    assert {"rename", "alias", "extra_args", "extra_envs"} <= set(stt_names)


def test_get_model_install_fields_returns_spec_fields_of_model_type(svc: VllmService) -> None:
    svc.models["default"]["reranker-model"] = VllmModel(hf_id="BAAI/bge-reranker-v2-m3", size="1 GB", model_type="reranker")

    result = svc._get_model_install_fields("default", "reranker-model")  # pyright: ignore[reportPrivateUsage]

    assert result == svc.get_model_type_spec("default", "reranker").fields


def test_get_model_install_fields_returns_empty_list_for_unknown_model(svc: VllmService) -> None:
    result = svc._get_model_install_fields("default", "ghost-model")  # pyright: ignore[reportPrivateUsage]

    assert result == []


def test_get_model_install_fields_returns_empty_list_for_purged_instance(svc: VllmService) -> None:
    svc.models["gpu2"] = {"m": VllmModel(hf_id="google/gemma-3-270m-it", size="1 GB")}

    result = svc._get_model_install_fields("gpu2", "m")  # pyright: ignore[reportPrivateUsage]

    assert result == []


def test_get_custom_model_spec_not_none(svc: VllmService) -> None:
    result = svc.get_custom_model_spec()

    assert result is not None
    field_names = [f.name for f in result.fields]
    assert "id" in field_names
    assert "hf_id" in field_names
    assert "revision" in field_names
    assert "size" in field_names


def test_model_installed_info_get_info() -> None:
    model = _make_model_installed_info(registration_id="reg-42")

    info = model.get_info()

    assert info.registration_id == "reg-42"


def test_model_installed_info_get_info_keeps_legacy_gpu_memory_utilization_spec_key() -> None:
    model = _make_model_installed_info()
    model.options.spec = {"gpu_memory_utilization": 0.5}

    info = model.get_info()

    assert info.spec == {"gpu_memory_utilization": 0.5}


def test_carry_over_edited_install_options_keeps_legacy_absolute_utilization(svc: VllmService) -> None:
    """The edit form only renders gpu_priority, so a legacy absolute value must survive an unrelated edit."""
    old = InstallModelIn(spec={"gpu_memory_utilization": 0.3, "prefix": "old"})
    new = InstallModelIn(spec={"prefix": "new"})

    result = svc._carry_over_edited_install_options(old, new)  # pyright: ignore[reportPrivateUsage]

    assert result.spec == {"prefix": "new", "gpu_memory_utilization": 0.3}
    assert new.spec == {"prefix": "new"}


def test_carry_over_edited_install_options_lets_new_priority_replace_legacy_value(svc: VllmService) -> None:
    old = InstallModelIn(spec={"gpu_memory_utilization": 0.3})
    new = InstallModelIn(spec={"gpu_priority": 0.5})

    assert svc._carry_over_edited_install_options(old, new) is new  # pyright: ignore[reportPrivateUsage]


def test_get_installed_info_returns_spec_when_installed(svc: VllmService) -> None:
    installed = _make_installed_info()
    installed.options.spec["hardware"] = False
    svc.instances_info["default"].installed = installed

    result = svc.get_installed_info("default")

    assert result == installed.options.spec


def test_get_installed_info_delegates_when_not_installed(svc: VllmService) -> None:
    svc.instances_info["default"].installed = None

    with patch.object(svc, "_get_service_installed_info", return_value=False) as mock:  # pyright: ignore[reportPrivateUsage]
        result = svc.get_installed_info("default")

    assert mock.call_count == 1
    assert mock.call_args == call("default")
    assert result is False


def test_generate_instance_config_none_info(svc: VllmService) -> None:
    config = svc._generate_instance_config("default", None, None)  # pyright: ignore[reportPrivateUsage]

    assert config.options is None
    assert config.models == []


def test_generate_instance_config_with_info(svc: VllmService) -> None:
    info = _make_installed_info()
    info.models["m1"] = _make_model_installed_info("m1")

    config = svc._generate_instance_config("default", info, None)  # pyright: ignore[reportPrivateUsage]

    assert config.options == info.options
    assert len(config.models or []) == 1


def test_generate_instance_config_persists_resolved_container_name(svc: VllmService) -> None:
    """Resolved container name round-trips into the persisted ModelConfig."""
    info = _make_installed_info()
    info.models["m1"] = _make_model_installed_info("m1")
    info.models["m1"].docker.container_name = "pinned-container-name"

    config = svc._generate_instance_config("default", info, None)  # pyright: ignore[reportPrivateUsage]

    assert (config.models or [])[0].container_name == "pinned-container-name"


def test_generate_instance_config_embeds_definition_when_in_registry(svc: VllmService) -> None:
    info = _make_installed_info()
    info.models["m1"] = _make_model_installed_info("m1")
    svc.models["default"]["m1"] = VllmModel(hf_id="m1", size="1 GB")

    config = svc._generate_instance_config("default", info, None)  # pyright: ignore[reportPrivateUsage]

    assert (config.models or [])[0].definition == {
        "hf_id": "m1",
        "revision": None,
        "env_vars": None,
        "quantization": None,
        "dtype": "auto",
        "shm_size": "16gb",
        "ulimits": None,
        "max_model_len": None,
        "gpu_memory_utilization": None,
        "size": "1 GB",
        "custom": None,
        "model_type": "llm",
        "adapter_id": None,
        "lora_rank": None,
        "supports_vision": None,
        "supports_audio": None,
    }


def test_generate_instance_config_omits_definition_when_missing_from_registry(svc: VllmService) -> None:
    info = _make_installed_info()
    info.models["ghost-model"] = _make_model_installed_info("ghost-model")

    config = svc._generate_instance_config("default", info, None)  # pyright: ignore[reportPrivateUsage]

    assert (config.models or [])[0].definition is None


def test_restore_model_definition_reinstates_model(svc: VllmService) -> None:
    definition = {"hf_id": "ghost-model", "size": "2 GB"}

    svc._restore_model_definition("default", "ghost-model", definition)  # pyright: ignore[reportPrivateUsage]

    assert svc.models["default"]["ghost-model"] == VllmModel(hf_id="ghost-model", size="2 GB")


def test_reload_instance_models_syncs_installed_registration_modality_support(svc: VllmService, deps: dict[str, Any]) -> None:
    """A catalog refresh must re-sync an already-installed model's registered modality support.

    Otherwise a fix that corrects `classify_modality_support`'s output (or a regenerated static
    catalog) never reaches a model that was already installed before the refresh - it would keep
    whatever was captured at install time forever.
    """
    installed = _make_installed_info()
    model_id = next(iter(svc.models["default"]))
    installed.models[model_id] = _make_model_installed_info(model_id, registration_id="reg-1")
    svc.instances_info["default"].installed = installed

    refreshed_model = svc.models["default"][model_id].model_copy(update={"supports_vision": True, "supports_audio": True})

    with patch.dict(_const.models, {model_id: refreshed_model}):
        svc._reload_instance_models("default")  # pyright: ignore[reportPrivateUsage]

    deps["endpoint_registry"].update_modality_support.assert_called_once_with("reg-1", supports_vision=True, supports_audio=True)


def test_reload_instance_models_skips_sync_when_not_registered(svc: VllmService, deps: dict[str, Any]) -> None:
    """No registration id (e.g. a model mid-install) means nothing to update yet."""
    installed = _make_installed_info()
    model_id = next(iter(svc.models["default"]))
    installed.models[model_id] = _make_model_installed_info(model_id, registration_id="")
    svc.instances_info["default"].installed = installed

    svc._reload_instance_models("default")  # pyright: ignore[reportPrivateUsage]

    deps["endpoint_registry"].update_modality_support.assert_not_called()


def test_load_download_info(svc: VllmService) -> None:
    result = svc._load_download_info({"model_path": "/tmp/x"})  # pyright: ignore[reportPrivateUsage]

    assert isinstance(result, DownloadedInfo)
    assert result.model_path == "/tmp/x"


@pytest.fixture(autouse=True)
def _x86_host() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    with patch("platform.machine", return_value="x86_64"):
        yield


@pytest.mark.parametrize("machine", ["arm64", "aarch64"])
def test_arm_host_without_gpu_is_supported_despite_no_avx512(deps: dict[str, Any], machine: str) -> None:
    deps["hardware"].cpu.avx512 = False
    deps["hardware"].gpus = []
    svc = VllmService(**deps)

    with patch("platform.machine", return_value=machine):
        assert svc.get_hardware_unsupported_reason() is None
        assert "cpu" in svc.get_size()


def _hardware_option_values(svc: VllmService) -> list[str]:
    field = next(f for f in svc.get_spec().fields if f.name == "hardware")
    return [v if isinstance(v, str) else v.value for v in field.values or []]


def test_get_spec_hides_cpu_option_on_x86_without_avx512_even_with_gpu(deps: dict[str, Any]) -> None:
    deps["hardware"].cpu.avx512 = False
    deps["hardware"].gpus = [MagicMock(spec=NvidiaGpuInfo)]
    svc = VllmService(**deps)

    options = _hardware_option_values(svc)

    assert "CPU" not in options


@pytest.mark.parametrize("machine", ["arm64", "aarch64"])
@patch("platform.machine")
def test_get_spec_offers_cpu_option_on_arm_without_avx512(mock_machine: MagicMock, deps: dict[str, Any], machine: str) -> None:
    mock_machine.return_value = machine
    deps["hardware"].cpu.avx512 = False
    deps["hardware"].gpus = []
    svc = VllmService(**deps)

    options = _hardware_option_values(svc)

    assert "CPU" in options


def test_is_given_hardware_support_gpu_raises_400_no_avx512_no_gpu(deps: dict[str, Any]) -> None:
    deps["hardware"].cpu.avx512 = False
    deps["hardware"].gpus = []
    svc = VllmService(**deps)

    with pytest.raises(HTTPException) as exc_info:
        svc.is_given_hardware_support_gpu(None)

    assert exc_info.value.status_code == 400


def test_get_specified_hardware_parts_raises_400_no_avx512_no_gpu(deps: dict[str, Any]) -> None:
    deps["hardware"].cpu.avx512 = False
    deps["hardware"].gpus = []
    svc = VllmService(**deps)

    with pytest.raises(HTTPException) as exc_info:
        svc.get_specified_hardware_parts(None)

    assert exc_info.value.status_code == 400


def test_is_given_hardware_support_gpu_with_gpu_does_not_raise(deps: dict[str, Any]) -> None:
    nvidia = MagicMock(spec=NvidiaGpuInfo)
    deps["hardware"].gpus = [nvidia]
    svc = VllmService(**deps)

    result = svc.is_given_hardware_support_gpu(True)

    assert result is True


def test_get_docker_compose_file_path_raises_400_no_model_id(svc: VllmService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    with pytest.raises(HTTPException) as exc_info:
        svc.get_docker_compose_file_path("default", None)

    assert exc_info.value.status_code == 400


def test_get_docker_compose_file_path_raises_400_model_not_installed(svc: VllmService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    with pytest.raises(HTTPException) as exc_info:
        svc.get_docker_compose_file_path("default", "some-model")

    assert exc_info.value.status_code == 400


def test_get_docker_compose_file_path_returns_path_for_installed_model(svc: VllmService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    installed.models["test-model"] = _make_model_installed_info("test-model")
    svc.instances_info["default"].installed = installed
    expected = Path("/some/path/docker-compose.yml")
    deps["docker_service"].get_docker_compose_file_path.return_value = expected

    result = svc.get_docker_compose_file_path("default", "test-model")

    assert result == expected


def test_get_docker_options_returns_docker_options_for_installed_model(svc: VllmService) -> None:
    installed = _make_installed_info()
    model_installed = _make_model_installed_info("test-model")
    installed.models["test-model"] = model_installed
    svc.instances_info["default"].installed = installed

    result = svc.get_docker_options("default", "test-model")

    assert result is model_installed.docker


@pytest.mark.asyncio
async def test_restart_docker_forwards_installed_model_docker_options(svc: VllmService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    model_installed = _make_model_installed_info("test-model")
    installed.models["test-model"] = model_installed
    svc.instances_info["default"].installed = installed
    deps["docker_service"].restart_docker_compose = AsyncMock()

    await svc.restart_docker("default", "test-model")

    deps["docker_service"].restart_docker_compose.assert_called_once_with(model_installed.docker)


def test_add_custom_model_registers_entry(svc: VllmService) -> None:
    custom = CustomModel(id="c-1", data={"id": "my-model", "hf_id": "google/gemma-3-270m-it", "size": "1GB"})

    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    assert "my-model" in svc.models["default"]
    assert svc.models["default"]["my-model"].custom == "c-1"
    assert svc.models["default"]["my-model"].hf_id == "google/gemma-3-270m-it"
    assert svc.models["default"]["my-model"].revision is None


def test_add_custom_model_registers_revision(svc: VllmService) -> None:
    custom = CustomModel(
        id="c-1b", data={"id": "my-model-rev", "hf_id": "google/gemma-3-270m-it", "revision": "quantized-awq", "size": "1GB"}
    )

    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    assert svc.models["default"]["my-model-rev"].revision == "quantized-awq"


def test_add_custom_model_registers_adapter_fields(svc: VllmService) -> None:
    custom = CustomModel(
        id="c-1c",
        data={
            "id": "pirate-bot",
            "hf_id": "Qwen/Qwen3-8B",
            "size": "16GB",
            "adapter_id": "pirate-llama",
            "lora_rank": 16,
        },
    )

    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    assert svc.models["default"]["pirate-bot"].adapter_id == "pirate-llama"
    assert svc.models["default"]["pirate-bot"].lora_rank == 16


def test_add_custom_model_duplicate_raises_400(svc: VllmService) -> None:
    custom = CustomModel(id="c-2", data={"id": "dup", "hf_id": "google/model", "size": "1GB"})

    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    with pytest.raises(HTTPException) as exc_info:
        svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]
    assert exc_info.value.status_code == 400


def test_add_custom_model_creates_dict_for_new_instance(svc: VllmService) -> None:
    svc.instances_info["extra"] = Instance(None, None, {}, InstanceConfig())
    custom = CustomModel(id="c-3", data={"id": "new-model", "hf_id": "google/model", "size": "1GB"})

    svc._add_custom_model("extra", custom)  # pyright: ignore[reportPrivateUsage]

    assert "new-model" in svc.models["extra"]


def test_remove_custom_model_deletes_entry(svc: VllmService) -> None:
    custom = CustomModel(id="c-4", data={"id": "to-remove", "hf_id": "google/model", "size": "1GB"})
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]
    svc.instances_info["default"].installed = _make_installed_info()

    svc._remove_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    assert "to-remove" not in svc.models["default"]


def test_remove_custom_model_raises_400_when_in_use(svc: VllmService) -> None:
    custom = CustomModel(id="c-5", data={"id": "in-use", "hf_id": "google/model", "size": "1GB"})
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]
    installed = _make_installed_info()
    installed.models["in-use"] = _make_model_installed_info("in-use")
    svc.instances_info["default"].installed = installed

    with pytest.raises(HTTPException) as exc_info:
        svc._remove_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_list_models_raises_404_for_unknown_instance(svc: VllmService) -> None:
    with pytest.raises(HTTPException) as exc_info:
        await svc.list_models("nonexistent", ListModelsFilters())

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_list_models_returns_all_models_no_filter(svc: VllmService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    result = await svc.list_models("default", ListModelsFilters())

    assert len(result.list) == len(svc.models["default"])


@pytest.mark.asyncio
async def test_list_models_filters_installed_only(svc: VllmService) -> None:
    installed = _make_installed_info()
    model_id = next(iter(svc.models["default"]))
    installed.models[model_id] = _make_model_installed_info(model_id)
    svc.instances_info["default"].installed = installed

    result = await svc.list_models("default", ListModelsFilters(installed=True))

    assert len(result.list) >= 1
    assert all(bool(m.installed) for m in result.list)


@pytest.mark.asyncio
async def test_get_model_raises_400_for_unknown_model_id(svc: VllmService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    with pytest.raises(HTTPException) as exc_info:
        await svc.get_model("default", "nonexistent-model")

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_get_model_returns_correct_model(svc: VllmService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    model_id = next(iter(svc.models["default"]))

    result = await svc.get_model("default", model_id)

    assert result.id == model_id


@pytest.mark.asyncio
async def test_download_model_emits_initial_and_final_progress(svc: VllmService, deps: dict[str, Any], tmp_path: Path) -> None:
    model = VllmModel(hf_id="google/test-model", size="1GB")
    stream = MagicMock()
    local_path = tmp_path / "model-files"
    local_path.mkdir()

    async def mock_download(*args: object, **kwargs: object):  # type: ignore[misc]
        yield SuccessDownloadPacket(local_path=local_path, filename="model-files")

    deps["model_downloader"].download = mock_download

    await svc._download_model(stream, model, tmp_path)  # pyright: ignore[reportPrivateUsage]

    assert stream.emit.call_count >= 2


@pytest.mark.asyncio
async def test_download_model_updates_progress_on_downloaded_packet(svc: VllmService, deps: dict[str, Any], tmp_path: Path) -> None:
    model = VllmModel(hf_id="google/test-model", size="100MB")
    stream = MagicMock()
    local_path = tmp_path / "model-files"
    local_path.mkdir()

    async def mock_download(*args: object, **kwargs: object):  # type: ignore[misc]
        yield DownloadedPacket(downloaded_bytes_size=1024 * 1024)
        yield SuccessDownloadPacket(local_path=local_path, filename="model-files")

    deps["model_downloader"].download = mock_download

    await svc._download_model(stream, model, tmp_path)  # pyright: ignore[reportPrivateUsage]

    progress_calls = [call for call in stream.emit.call_args_list if hasattr(call.args[0], "get") and call.args[0].get("value", 0) > 0]
    assert len(progress_calls) >= 1


@pytest.mark.asyncio
async def test_download_model_or_set_progress_starts_new_download(svc: VllmService, tmp_path: Path) -> None:
    stream = MagicMock()
    model_id = "google/test-model"
    model = VllmModel(hf_id=model_id, size="1GB")
    svc.models["default"][model_id] = model

    with patch.object(  # pyright: ignore[reportPrivateUsage]
        svc, "_download_model", new_callable=AsyncMock, return_value=tmp_path / "model"
    ) as mock_dl:
        await svc._download_model_or_set_progress(stream, model_id, model, tmp_path)  # pyright: ignore[reportPrivateUsage]

    assert mock_dl.call_count == 1
    assert model_id not in svc.models_download_progress


@pytest.mark.asyncio
async def test_download_model_or_set_progress_cleans_up_on_failure(svc: VllmService, tmp_path: Path) -> None:
    stream = MagicMock()
    model_id = "google/test-model"
    model = VllmModel(hf_id=model_id, size="1GB")
    svc.models["default"][model_id] = model

    with (
        patch.object(svc, "_download_model", new_callable=AsyncMock, side_effect=HTTPException(400, "boom")),  # pyright: ignore[reportPrivateUsage]
        pytest.raises(HTTPException),
    ):
        await svc._download_model_or_set_progress(stream, model_id, model, tmp_path)  # pyright: ignore[reportPrivateUsage]

    assert model_id not in svc.models_download_progress


@pytest.mark.asyncio
async def test_download_model_or_set_progress_retries_download_after_failure(svc: VllmService, tmp_path: Path) -> None:
    stream1 = MagicMock()
    stream2 = MagicMock()
    model_id = "google/test-model"
    model = VllmModel(hf_id=model_id, size="1GB")
    svc.models["default"][model_id] = model

    mock_dl = AsyncMock(side_effect=[HTTPException(400, "boom"), tmp_path / "model"])
    with patch.object(svc, "_download_model", mock_dl):  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(HTTPException):
            await svc._download_model_or_set_progress(stream1, model_id, model, tmp_path)  # pyright: ignore[reportPrivateUsage]
        await svc._download_model_or_set_progress(stream2, model_id, model, tmp_path)  # pyright: ignore[reportPrivateUsage]

    assert mock_dl.call_count == 2


@pytest.mark.asyncio
async def test_download_model_or_set_progress_forwards_existing_stream(svc: VllmService, tmp_path: Path) -> None:
    existing_stream: Stream[StreamChunk] = Stream()  # type: ignore[type-arg]
    chunk = StreamChunkProgress(type="progress", stage="download", value=0.5, data={"local_model_path": str(tmp_path / "model")})
    existing_stream.emit(chunk)
    existing_stream.close()
    model_id = "google/test-model"
    model = VllmModel(hf_id=model_id, size="1GB")
    svc.models_download_progress[model_id] = existing_stream  # type: ignore[assignment]
    output_stream = MagicMock()

    await svc._download_model_or_set_progress(output_stream, model_id, model, tmp_path)  # pyright: ignore[reportPrivateUsage]

    assert output_stream.emit.call_count == 1
    assert output_stream.emit.call_args == call(chunk)


@pytest.mark.asyncio
async def test_get_gpu_memory_utilization_increments_total(svc: VllmService) -> None:
    svc.gpu_memory_utilization = 0.0
    model = VllmModel(hf_id="google/test", size="1GB")
    opts = VllmModelOptions(gpu_memory_utilization=0.5)

    result = await svc._get_gpu_memory_utilization("default", "google/test", opts, model)  # pyright: ignore[reportPrivateUsage]

    assert result == 0.5
    assert svc.gpu_memory_utilization == 0.5


@pytest.mark.asyncio
async def test_get_gpu_memory_utilization_raises_422_when_sum_exceeds_1(svc: VllmService) -> None:
    svc.gpu_memory_utilization = 0.8
    model = VllmModel(hf_id="google/test", size="1GB")
    opts = VllmModelOptions(gpu_memory_utilization=0.5)

    with pytest.raises(HTTPException) as exc_info:
        await svc._get_gpu_memory_utilization("default", "google/test", opts, model)  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 422


@pytest.mark.asyncio
async def test_get_default_gpu_memory_utilization_uses_free_fraction_with_margin(svc: VllmService, deps: dict[str, Any]) -> None:
    deps["hardware"].get_realtime_stats = AsyncMock(return_value=GpuStats(total_vram_gb=10.0, used_vram_gb=5.0, gpus=None))

    result = await svc._get_default_gpu_memory_utilization()  # pyright: ignore[reportPrivateUsage]

    assert result == 0.45


@pytest.mark.asyncio
async def test_get_default_gpu_memory_utilization_caps_at_default_when_gpu_is_free(svc: VllmService, deps: dict[str, Any]) -> None:
    deps["hardware"].get_realtime_stats = AsyncMock(return_value=GpuStats(total_vram_gb=10.0, used_vram_gb=0.0, gpus=None))

    result = await svc._get_default_gpu_memory_utilization()  # pyright: ignore[reportPrivateUsage]

    assert result == 0.9


@pytest.mark.asyncio
async def test_get_default_gpu_memory_utilization_falls_back_when_stats_unavailable(svc: VllmService, deps: dict[str, Any]) -> None:
    deps["hardware"].get_realtime_stats = AsyncMock(return_value=None)

    result = await svc._get_default_gpu_memory_utilization()  # pyright: ignore[reportPrivateUsage]

    assert result == 0.9


@pytest.mark.asyncio
async def test_get_gpu_memory_utilization_uses_dynamic_default_when_unset(svc: VllmService, deps: dict[str, Any]) -> None:
    svc.gpu_memory_utilization = 0.0
    deps["hardware"].get_realtime_stats = AsyncMock(return_value=GpuStats(total_vram_gb=10.0, used_vram_gb=5.0, gpus=None))
    model = VllmModel(hf_id="google/test", size="1GB")
    opts = VllmModelOptions()

    result = await svc._get_gpu_memory_utilization("default", "google/test", opts, model)  # pyright: ignore[reportPrivateUsage]

    assert result == 0.45
    assert svc.gpu_memory_utilization == 0.45


@pytest.mark.asyncio
async def test_get_gpu_memory_utilization_reuses_persisted_value_instead_of_live_default(svc: VllmService, deps: dict[str, Any]) -> None:
    """A persisted auto-tuned value must be reused so a reload doesn't drift with currently-free VRAM."""
    svc.gpu_memory_utilization = 0.0
    svc.instances_info["default"].config = InstanceConfig(
        models=[ModelConfig(model_id="google/test", options=InstallModelIn(spec={}), gpu_memory_utilization=0.33)]
    )
    model = VllmModel(hf_id="google/test", size="1GB")
    opts = VllmModelOptions()

    result = await svc._get_gpu_memory_utilization("default", "google/test", opts, model)  # pyright: ignore[reportPrivateUsage]

    assert result == 0.33
    deps["hardware"].get_realtime_stats.assert_not_called()


@pytest.mark.asyncio
async def test_get_gpu_memory_utilization_applies_priority_to_available_vram(svc: VllmService, deps: dict[str, Any]) -> None:
    svc.gpu_memory_utilization = 0.0
    deps["hardware"].get_realtime_stats = AsyncMock(return_value=GpuStats(total_vram_gb=10.0, used_vram_gb=0.0, gpus=None))
    model = VllmModel(hf_id="google/test", size="1GB")
    opts = VllmModelOptions(gpu_priority=0.5)

    result = await svc._get_gpu_memory_utilization("default", "google/test", opts, model)  # pyright: ignore[reportPrivateUsage]

    assert result == 0.45
    assert svc.gpu_memory_utilization == 0.45


@pytest.mark.asyncio
async def test_get_gpu_memory_utilization_applies_priority_to_currently_free_vram(svc: VllmService, deps: dict[str, Any]) -> None:
    svc.gpu_memory_utilization = 0.4
    deps["hardware"].get_realtime_stats = AsyncMock(return_value=GpuStats(total_vram_gb=10.0, used_vram_gb=4.0, gpus=None))
    model = VllmModel(hf_id="google/test", size="1GB")
    opts = VllmModelOptions(gpu_priority=0.5)

    result = await svc._get_gpu_memory_utilization("default", "google/test", opts, model)  # pyright: ignore[reportPrivateUsage]

    assert result == 0.27
    assert svc.gpu_memory_utilization == pytest.approx(0.67)


@pytest.mark.asyncio
async def test_get_gpu_memory_utilization_priority_path_still_enforces_total_limit(svc: VllmService, deps: dict[str, Any]) -> None:
    svc.gpu_memory_utilization = 0.95
    deps["hardware"].get_realtime_stats = AsyncMock(return_value=GpuStats(total_vram_gb=10.0, used_vram_gb=0.0, gpus=None))
    model = VllmModel(hf_id="google/test", size="1GB")
    opts = VllmModelOptions(gpu_priority=0.5)

    with pytest.raises(HTTPException) as exc:
        await svc._get_gpu_memory_utilization("default", "google/test", opts, model)  # pyright: ignore[reportPrivateUsage]

    assert exc.value.status_code == 422
    assert svc.gpu_memory_utilization == 0.95


@pytest.mark.asyncio
async def test_get_gpu_memory_utilization_rejects_priority_that_rounds_below_minimum(svc: VllmService, deps: dict[str, Any]) -> None:
    """A tiny priority (or almost no free VRAM) must not start the engine with 0 utilization."""
    svc.gpu_memory_utilization = 0.0
    deps["hardware"].get_realtime_stats = AsyncMock(return_value=GpuStats(total_vram_gb=10.0, used_vram_gb=9.5, gpus=None))
    model = VllmModel(hf_id="google/test", size="1GB")
    opts = VllmModelOptions(gpu_priority=0.01)

    with pytest.raises(HTTPException) as exc:
        await svc._get_gpu_memory_utilization("default", "google/test", opts, model)  # pyright: ignore[reportPrivateUsage]

    assert exc.value.status_code == 422
    assert svc.gpu_memory_utilization == 0.0


@pytest.mark.asyncio
async def test_get_gpu_memory_utilization_priority_wins_over_model_default(svc: VllmService, deps: dict[str, Any]) -> None:
    svc.gpu_memory_utilization = 0.0
    deps["hardware"].get_realtime_stats = AsyncMock(return_value=GpuStats(total_vram_gb=10.0, used_vram_gb=0.0, gpus=None))
    model = VllmModel(hf_id="google/test", size="1GB", gpu_memory_utilization=0.8)
    opts = VllmModelOptions(gpu_priority=0.5)

    result = await svc._get_gpu_memory_utilization("default", "google/test", opts, model)  # pyright: ignore[reportPrivateUsage]

    assert result == 0.45


@pytest.mark.asyncio
async def test_get_gpu_memory_utilization_explicit_absolute_value_wins_over_priority(svc: VllmService, deps: dict[str, Any]) -> None:
    svc.gpu_memory_utilization = 0.0
    model = VllmModel(hf_id="google/test", size="1GB")
    opts = VllmModelOptions(gpu_memory_utilization=0.3, gpu_priority=0.5)

    result = await svc._get_gpu_memory_utilization("default", "google/test", opts, model)  # pyright: ignore[reportPrivateUsage]

    assert result == 0.3
    deps["hardware"].get_realtime_stats.assert_not_called()


@pytest.mark.parametrize("priority", [0.0, 0.005, -0.1, 1.5])
def test_model_options_reject_out_of_range_gpu_priority(priority: float) -> None:
    with pytest.raises(ValidationError):
        VllmModelOptions(gpu_priority=priority)


@pytest.mark.asyncio
async def test_get_gpu_memory_utilization_priority_never_exceeds_available_vram(svc: VllmService, deps: dict[str, Any]) -> None:
    svc.gpu_memory_utilization = 0.0
    deps["hardware"].get_realtime_stats = AsyncMock(return_value=GpuStats(total_vram_gb=10.0, used_vram_gb=0.0, gpus=None))
    model = VllmModel(hf_id="google/test", size="1GB")
    opts = VllmModelOptions(gpu_priority=1.0)

    result = await svc._get_gpu_memory_utilization("default", "google/test", opts, model)  # pyright: ignore[reportPrivateUsage]

    assert result == 0.9


@pytest.mark.asyncio
async def test_get_gpu_memory_utilization_reuses_persisted_value_over_priority(svc: VllmService, deps: dict[str, Any]) -> None:
    """A persisted value must win over recomputing from priority, for the same reload-stability reason as the default."""
    svc.gpu_memory_utilization = 0.0
    svc.instances_info["default"].config = InstanceConfig(
        models=[ModelConfig(model_id="google/test", options=InstallModelIn(spec={"gpu_priority": 1.0}), gpu_memory_utilization=0.33)]
    )
    model = VllmModel(hf_id="google/test", size="1GB")
    opts = VllmModelOptions(gpu_priority=1.0)

    result = await svc._get_gpu_memory_utilization("default", "google/test", opts, model)  # pyright: ignore[reportPrivateUsage]

    assert result == 0.33
    deps["hardware"].get_realtime_stats.assert_not_called()


@pytest.mark.asyncio
async def test_get_gpu_memory_utilization_recomputes_when_priority_changed(svc: VllmService, deps: dict[str, Any]) -> None:
    """A persisted value computed for a different priority must not silently override the new one."""
    svc.gpu_memory_utilization = 0.0
    deps["hardware"].get_realtime_stats = AsyncMock(return_value=GpuStats(total_vram_gb=10.0, used_vram_gb=0.0, gpus=None))
    svc.instances_info["default"].config = InstanceConfig(
        models=[ModelConfig(model_id="google/test", options=InstallModelIn(spec={"gpu_priority": 1.0}), gpu_memory_utilization=0.9)]
    )
    model = VllmModel(hf_id="google/test", size="1GB")
    opts = VllmModelOptions(gpu_priority=0.5)

    result = await svc._get_gpu_memory_utilization("default", "google/test", opts, model)  # pyright: ignore[reportPrivateUsage]

    assert result == 0.45


@pytest.mark.asyncio
async def test_get_gpu_memory_utilization_reuses_persisted_value_for_string_priority(svc: VllmService, deps: dict[str, Any]) -> None:
    """The persisted spec is the raw request body, so a priority sent as a string must still match its parsed value."""
    svc.gpu_memory_utilization = 0.0
    svc.instances_info["default"].config = InstanceConfig(
        models=[ModelConfig(model_id="google/test", options=InstallModelIn(spec={"gpu_priority": "0.5"}), gpu_memory_utilization=0.33)]
    )
    model = VllmModel(hf_id="google/test", size="1GB")
    opts = VllmModelOptions(gpu_priority=0.5)

    result = await svc._get_gpu_memory_utilization("default", "google/test", opts, model)  # pyright: ignore[reportPrivateUsage]

    assert result == 0.33
    deps["hardware"].get_realtime_stats.assert_not_called()


@pytest.mark.asyncio
async def test_get_gpu_memory_utilization_treats_legacy_zero_as_unset(svc: VllmService, deps: dict[str, Any]) -> None:
    """The old free-text field allowed 0, which used to mean "unset" - it must not launch the engine with no memory."""
    svc.gpu_memory_utilization = 0.0
    deps["hardware"].get_realtime_stats = AsyncMock(return_value=GpuStats(total_vram_gb=10.0, used_vram_gb=0.0, gpus=None))
    model = VllmModel(hf_id="google/test", size="1GB")
    opts = VllmModelOptions(gpu_memory_utilization=0)

    result = await svc._get_gpu_memory_utilization("default", "google/test", opts, model)  # pyright: ignore[reportPrivateUsage]

    assert result == 0.9


def test_get_persisted_model_gpu_memory_utilization_returns_none_for_unparseable_priority(svc: VllmService) -> None:
    svc.instances_info["default"].config = InstanceConfig(
        models=[ModelConfig(model_id="google/test", options=InstallModelIn(spec={"gpu_priority": "high"}), gpu_memory_utilization=0.33)]
    )

    assert svc._get_persisted_model_gpu_memory_utilization("default", "google/test", None) is None  # pyright: ignore[reportPrivateUsage]


def test_get_persisted_model_gpu_memory_utilization_returns_none_when_priority_was_reset(svc: VllmService) -> None:
    """Resetting a priority back to unset must re-derive the default, not reuse the value sized for the old priority."""
    svc.instances_info["default"].config = InstanceConfig(
        models=[ModelConfig(model_id="google/test", options=InstallModelIn(spec={"gpu_priority": 0.5}), gpu_memory_utilization=0.33)]
    )

    assert svc._get_persisted_model_gpu_memory_utilization("default", "google/test", None) is None  # pyright: ignore[reportPrivateUsage]


def test_get_persisted_model_gpu_memory_utilization_returns_none_for_untracked_instance(svc: VllmService) -> None:
    assert svc._get_persisted_model_gpu_memory_utilization("no-such-instance", "model", None) is None  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_get_quantization_returns_valid_value(svc: VllmService) -> None:
    model = VllmModel(hf_id="google/test", size="1GB")
    opts = VllmModelOptions(quantization="fp8")

    result = await svc._get_quantization(opts, model)  # pyright: ignore[reportPrivateUsage]

    assert result == "fp8"


@pytest.mark.asyncio
async def test_get_quantization_raises_422_on_invalid_characters(svc: VllmService) -> None:
    model = VllmModel(hf_id="google/test", size="1GB")
    opts = VllmModelOptions(quantization=None)
    opts.quantization = "fp8!@#"  # bypass pydantic

    with pytest.raises(HTTPException) as exc_info:
        await svc._get_quantization(opts, model)  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 422


@pytest.mark.asyncio
async def test_get_quantization_returns_none_when_not_set(svc: VllmService) -> None:
    model = VllmModel(hf_id="google/test", size="1GB")
    opts = VllmModelOptions()

    result = await svc._get_quantization(opts, model)  # pyright: ignore[reportPrivateUsage]

    assert result is None


def test_register_model_endpoint_llm_calls_chat_completion_proxy(svc: VllmService, deps: dict[str, Any]) -> None:
    model = VllmModel(hf_id="google/test", size="1GB", model_type="llm")
    model_info = _make_model_installed_info("google/test", model_type="llm")

    svc._register_model_endpoint(  # pyright: ignore[reportPrivateUsage]
        model_info=model_info,
        model=model,
        registered_name="google/test",
        model_id="google/test",
        context_window=4096,
        max_context_window=4096,
        capacity=None,
    )

    assert deps["endpoint_registry"].register_chat_completion_as_proxy.call_count == 1
    assert deps["endpoint_registry"].register_rerank_as_proxy.call_count == 0


def test_register_model_endpoint_forwards_modality_support(svc: VllmService, deps: dict[str, Any]) -> None:
    model = VllmModel(hf_id="org/vlm", size="1GB", model_type="llm", supports_vision=True, supports_audio=None)
    model_info = _make_model_installed_info("org/vlm", model_type="llm")

    svc._register_model_endpoint(  # pyright: ignore[reportPrivateUsage]
        model_info=model_info,
        model=model,
        registered_name="org/vlm",
        model_id="org/vlm",
        context_window=4096,
        max_context_window=4096,
        capacity=None,
    )

    call_kwargs = deps["endpoint_registry"].register_chat_completion_as_proxy.call_args.kwargs
    assert call_kwargs["props"].supports_vision is True
    assert call_kwargs["props"].supports_audio is None


def test_register_model_endpoint_reranker_calls_rerank_proxy(svc: VllmService, deps: dict[str, Any]) -> None:
    model = VllmModel(hf_id="google/reranker", size="1GB", model_type="reranker")
    model_info = _make_model_installed_info("google/reranker", model_type="reranker")

    svc._register_model_endpoint(  # pyright: ignore[reportPrivateUsage]
        model_info=model_info,
        model=model,
        registered_name="google/reranker",
        model_id="google/reranker",
        context_window=None,
        max_context_window=None,
        capacity=None,
    )

    assert deps["endpoint_registry"].register_rerank_as_proxy.call_count == 1
    assert deps["endpoint_registry"].register_chat_completion_as_proxy.call_count == 0


def test_register_model_endpoint_embedding_calls_embeddings_proxy(svc: VllmService, deps: dict[str, Any]) -> None:
    model = VllmModel(hf_id="google/embedder", size="1GB", model_type="embedding")
    model_info = _make_model_installed_info("google/embedder", model_type="embedding")

    svc._register_model_endpoint(  # pyright: ignore[reportPrivateUsage]
        model_info=model_info,
        model=model,
        registered_name="google/embedder",
        model_id="google/embedder",
        context_window=None,
        max_context_window=None,
        capacity=None,
    )

    assert deps["endpoint_registry"].register_embeddings_as_proxy.call_count == 1
    assert deps["endpoint_registry"].register_chat_completion_as_proxy.call_count == 0
    assert deps["endpoint_registry"].register_rerank_as_proxy.call_count == 0


def test_register_model_endpoint_stt_calls_audio_transcriptions_proxy(svc: VllmService, deps: dict[str, Any]) -> None:
    model = VllmModel(hf_id="openai/whisper-small", size="1GB", model_type="stt")
    model_info = _make_model_installed_info("openai/whisper-small", model_type="stt")

    svc._register_model_endpoint(  # pyright: ignore[reportPrivateUsage]
        model_info=model_info,
        model=model,
        registered_name="openai/whisper-small",
        model_id="openai/whisper-small",
        context_window=None,
        max_context_window=None,
        capacity=None,
    )

    registry = deps["endpoint_registry"]
    assert registry.register_audio_transcriptions_as_proxy.call_count == 1
    kwargs = registry.register_audio_transcriptions_as_proxy.call_args.kwargs
    assert kwargs["props"].type == "stt"
    assert kwargs["props"].endpoints == ["/v1/audio/transcriptions"]
    assert kwargs["options"].url.endswith("/v1/audio/transcriptions")
    assert kwargs["options"].rewrite_model_to == "openai/whisper-small"
    assert registry.register_chat_completion_as_proxy.call_count == 0


def test_register_model_endpoint_passes_capacity_through(svc: VllmService, deps: dict[str, Any]) -> None:
    model = VllmModel(hf_id="google/test", size="1GB", model_type="llm")
    model_info = _make_model_installed_info("google/test", model_type="llm")

    svc._register_model_endpoint(  # pyright: ignore[reportPrivateUsage]
        model_info=model_info,
        model=model,
        registered_name="google/test",
        model_id="google/test",
        context_window=4096,
        max_context_window=4096,
        capacity=8,
    )

    call_kwargs = deps["endpoint_registry"].register_chat_completion_as_proxy.call_args.kwargs
    assert call_kwargs["registration_options"].capacity_state == 8


def test_register_model_endpoint_marks_capacity_unknown_when_none(svc: VllmService, deps: dict[str, Any]) -> None:
    """capacity=None from vLLM always means "couldn't determine it", never "no concurrency concept"."""
    model = VllmModel(hf_id="google/test", size="1GB", model_type="llm")
    model_info = _make_model_installed_info("google/test", model_type="llm")

    svc._register_model_endpoint(  # pyright: ignore[reportPrivateUsage]
        model_info=model_info,
        model=model,
        registered_name="google/test",
        model_id="google/test",
        context_window=4096,
        max_context_window=4096,
        capacity=None,
    )

    call_kwargs = deps["endpoint_registry"].register_chat_completion_as_proxy.call_args.kwargs
    assert call_kwargs["registration_options"].capacity_state == "unknown"


def test_get_image_true_returns_gpu_image(svc: VllmService) -> None:
    image = svc._get_image(True)  # pyright: ignore[reportPrivateUsage]

    assert image == _const.images["gpu"]


def test_get_image_false_returns_cpu_image(svc: VllmService) -> None:
    image = svc._get_image(False)  # pyright: ignore[reportPrivateUsage]

    assert image == _const.images["cpu"]


def test_get_default_docker_image_name_no_gpu_returns_cpu_image(svc: VllmService) -> None:
    assert svc.get_default_docker_image_name() == _const.images["cpu"].name


def test_get_default_docker_image_name_with_gpu_returns_gpu_image(deps: dict[str, Any]) -> None:
    deps["hardware"].gpus = [MagicMock(spec=NvidiaGpuInfo)]
    svc = VllmService(**deps)

    assert svc.get_default_docker_image_name() == _const.images["gpu"].name


@pytest.mark.asyncio
async def test_install_model_returns_already_installed(svc: VllmService) -> None:
    installed = _make_installed_info()
    model_id = next(iter(svc.models["default"]))
    installed.models[model_id] = _make_model_installed_info(model_id)
    svc.instances_info["default"].installed = installed

    promise = await svc._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]

    result = await promise.wait()
    assert result.status == "OK"
    assert "Already installed" in result.details


@pytest.mark.asyncio
async def test_install_model_raises_400_for_unknown_model(svc: VllmService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    with pytest.raises(HTTPException) as exc_info:
        await svc._install_model("default", "nonexistent-model", InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_install_model_calls_docker_install(svc: VllmService, deps: dict[str, Any], tmp_path: Path) -> None:
    _setup_install_mocks(svc, deps)
    model_id = next(iter(svc.models["default"]))

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.vllm_service.get_model_dir_context_window", new_callable=AsyncMock, return_value=4096),
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert deps["docker_service"].install_and_run_docker.call_count == 1


@pytest.mark.asyncio
async def test_install_model_appends_revision_suffix_to_model_dir(svc: VllmService, deps: dict[str, Any], tmp_path: Path) -> None:
    _setup_install_mocks(svc, deps)
    svc.models["default"]["custom-model"] = VllmModel(hf_id="google/gemma-3-270m-it", revision="quantized/awq", size="1GB")

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.vllm_service.get_model_dir_context_window", new_callable=AsyncMock, return_value=4096),
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
    ):
        promise = await svc._install_model("default", "custom-model", InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    docker_options = deps["docker_service"].install_and_run_docker.call_args.args[0]
    assert "google-gemma-3-270m-it--quantized_awq" in docker_options.volumes[0]


@pytest.mark.asyncio
@patch.object(VllmService, "get_specified_hardware_parts", return_value=[])
@patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000")
@patch("server.services.vllm_service.get_model_dir_context_window", new_callable=AsyncMock, return_value=4096)
@patch.object(VllmService, "_download_model_or_set_progress", new_callable=AsyncMock)
async def test_install_model_passes_json_extra_arg_verbatim(
    mock_download_model_or_set_progress: AsyncMock,
    mock_get_model_dir_context_window: AsyncMock,
    mock_get_base_url: MagicMock,
    mock_get_specified_hardware_parts: MagicMock,
    svc: VllmService,
    deps: dict[str, Any],
    tmp_path: Path,
):
    _setup_install_mocks(svc, deps)
    mock_download_model_or_set_progress.return_value = tmp_path / "model"
    model_id = next(iter(svc.models["default"]))
    json_value = '{"method":"mtp","num_speculative_tokens":3}'
    spec = {"extra_args": {"--speculative-config": json_value}}

    promise = await svc._install_model("default", model_id, InstallModelIn(spec=spec))  # pyright: ignore[reportPrivateUsage]
    await promise.wait()

    assert deps["docker_service"].install_and_run_docker.call_count == 1
    docker_options = deps["docker_service"].install_and_run_docker.call_args.args[0]
    assert isinstance(docker_options.command, list)
    idx = docker_options.command.index("--speculative-config")
    assert docker_options.command[idx + 1] == json_value


@pytest.mark.asyncio
async def test_install_model_dedups_download_by_revision_aware_key(svc: VllmService, deps: dict[str, Any], tmp_path: Path) -> None:
    """Two custom models sharing an hf_id but differing only by revision must not share a download dedup key.

    Otherwise the second install would see the first's (bare hf_id) key already "in progress" and be
    handed the first revision's model_dir instead of downloading its own.
    """
    _setup_install_mocks(svc, deps)
    svc.models["default"]["custom-model"] = VllmModel(hf_id="google/gemma-3-270m-it", revision="quantized/awq", size="1GB")

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model") as mock_download,  # pyright: ignore[reportPrivateUsage]
        patch("server.services.vllm_service.get_model_dir_context_window", new_callable=AsyncMock, return_value=4096),
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
    ):
        promise = await svc._install_model("default", "custom-model", InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    download_key = mock_download.call_args.args[1]
    assert download_key == "google-gemma-3-270m-it--quantized_awq"


@pytest.mark.asyncio
async def test_install_model_uses_container_name_override_when_given(svc: VllmService, deps: dict[str, Any], tmp_path: Path) -> None:
    _setup_install_mocks(svc, deps)
    model_id = next(iter(svc.models["default"]))

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.vllm_service.get_model_dir_context_window", new_callable=AsyncMock, return_value=4096),
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn(spec={}), "pinned-name")  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    docker_options = deps["docker_service"].install_and_run_docker.call_args.args[0]
    assert docker_options.container_name == "pinned-name"
    deps["docker_service"].get_docker_container_name.assert_not_called()


@pytest.mark.asyncio
async def test_install_model_falls_back_to_generated_container_name(svc: VllmService, deps: dict[str, Any], tmp_path: Path) -> None:
    _setup_install_mocks(svc, deps)
    model_id = next(iter(svc.models["default"]))

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.vllm_service.get_model_dir_context_window", new_callable=AsyncMock, return_value=4096),
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    docker_options = deps["docker_service"].install_and_run_docker.call_args.args[0]
    assert docker_options.container_name == deps["docker_service"].get_docker_container_name.return_value


@pytest.mark.asyncio
async def test_install_model_downloads_and_mounts_lora_adapter(svc: VllmService, deps: dict[str, Any], tmp_path: Path) -> None:
    """A model with `adapter_id` set must fetch the two safetensors files from the adapter registry,
    mount them into the container, and pass the resolved rank/path to the vLLM command."""
    _setup_install_mocks(svc, deps)
    svc.models["default"]["pirate-bot"] = VllmModel(hf_id="Qwen/Qwen3-8B", size="16GB", adapter_id="pirate-llama", lora_rank=16)
    deps["services_manager"].services["finetune"].get_installed_base_url.return_value = "http://df-finetune:8333"
    download_calls: list[tuple[str, str | None]] = []

    def mock_adapter_downloader_cls(secret: str) -> MagicMock:
        downloader = MagicMock()

        async def mock_download(  # type: ignore[misc]
            download_url: str, model_dir: Path, temp_dir: Path, filename: str | None = None, revision: str | None = None
        ):
            download_calls.append((download_url, filename))
            yield SuccessDownloadPacket(local_path=model_dir / (filename or "file"), filename=filename or "file")

        downloader.download = mock_download
        return downloader

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_get_working_dir", return_value=tmp_path),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.vllm_service.get_model_dir_context_window", new_callable=AsyncMock, return_value=4096),
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch("server.services.vllm_service.AdapterRegistryDownloader", side_effect=mock_adapter_downloader_cls),
        patch.object(
            svc,
            "_fetch_adapter_metadata",
            new_callable=AsyncMock,
            return_value=AdapterMetadata(base_model="Qwen/Qwen3-8B", rank=16, created=1_700_000_000),
        ),
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
    ):
        promise = await svc._install_model("default", "pirate-bot", InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    urls = {url for url, _ in download_calls}
    assert urls == {
        "http://df-finetune:8333/v1/adapters/pirate-llama/safetensors/adapter_config.json",
        "http://df-finetune:8333/v1/adapters/pirate-llama/safetensors/adapter_model.safetensors",
    }

    docker_options = deps["docker_service"].install_and_run_docker.call_args.args[0]
    assert any("adapter-pirate_llama" in volume and "/mnt/hf/loras/pirate_llama" in volume for volume in docker_options.volumes)
    assert "--enable-lora" in docker_options.command
    command = docker_options.command
    assert isinstance(command, list)
    assert command[command.index("--lora-modules") + 1] == "pirate-bot=/mnt/hf/loras/pirate_llama"
    assert command[command.index("--max-lora-rank") + 1] == "16"
    assert svc.models_downloaded["pirate-bot"].adapter_path is not None


@pytest.mark.asyncio
async def test_install_model_raises_400_when_adapter_registry_not_configured(
    svc: VllmService, deps: dict[str, Any], tmp_path: Path
) -> None:
    _setup_install_mocks(svc, deps)
    svc.models["default"]["pirate-bot"] = VllmModel(hf_id="Qwen/Qwen3-8B", size="16GB", adapter_id="pirate-llama", lora_rank=16)
    deps["services_manager"].services["finetune"].get_installed_base_url.return_value = None

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_get_working_dir", return_value=tmp_path),  # pyright: ignore[reportPrivateUsage]
    ):
        promise = await svc._install_model("default", "pirate-bot", InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(HTTPException) as exc_info:
            await promise.wait()

    assert exc_info.value.status_code == 400


def test_purge_stale_download_removes_old_directory_when_path_changed(svc: VllmService, tmp_path: Path) -> None:
    old_dir = tmp_path / "old-model"
    old_dir.mkdir()
    new_dir = tmp_path / "new-model"
    svc.models_downloaded["custom-model"] = DownloadedInfo(model_path=str(old_dir))

    svc._purge_stale_download("custom-model", new_dir)  # pyright: ignore[reportPrivateUsage]

    assert not old_dir.exists()


def test_purge_stale_download_keeps_directory_when_path_unchanged(svc: VllmService, tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    svc.models_downloaded["custom-model"] = DownloadedInfo(model_path=str(model_dir))

    svc._purge_stale_download("custom-model", model_dir)  # pyright: ignore[reportPrivateUsage]

    assert model_dir.exists()


def test_purge_stale_download_noop_when_nothing_downloaded_yet(svc: VllmService, tmp_path: Path) -> None:
    svc._purge_stale_download("custom-model", tmp_path / "model")  # pyright: ignore[reportPrivateUsage]


def test_purge_stale_download_removes_old_adapter_directory_when_adapter_changed(svc: VllmService, tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    old_adapter_dir = tmp_path / "adapter-old"
    old_adapter_dir.mkdir()
    new_adapter_dir = tmp_path / "adapter-new"
    svc.models_downloaded["custom-model"] = DownloadedInfo(model_path=str(model_dir), adapter_path=str(old_adapter_dir))

    svc._purge_stale_download("custom-model", model_dir, new_adapter_dir)  # pyright: ignore[reportPrivateUsage]

    assert not old_adapter_dir.exists()


def test_purge_stale_download_keeps_adapter_directory_when_unchanged(svc: VllmService, tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    svc.models_downloaded["custom-model"] = DownloadedInfo(model_path=str(model_dir), adapter_path=str(adapter_dir))

    svc._purge_stale_download("custom-model", model_dir, adapter_dir)  # pyright: ignore[reportPrivateUsage]

    assert adapter_dir.exists()


@pytest.mark.asyncio
async def test_install_model_edit_cleans_up_previous_download_directory(svc: VllmService, deps: dict[str, Any], tmp_path: Path) -> None:
    """Reinstalling a custom model under a new hf_id/revision must delete its old download directory.

    Otherwise the old directory is orphaned forever - nothing references it once
    `models_downloaded[model_id]` is overwritten with the new path.
    """
    _setup_install_mocks(svc, deps)
    svc.models["default"]["custom-model"] = VllmModel(hf_id="google/gemma-3-270m-it", size="1GB")
    old_dir = tmp_path / "old-model"
    old_dir.mkdir()
    svc.models_downloaded["custom-model"] = DownloadedInfo(model_path=str(old_dir))
    new_dir = tmp_path / "new-model"

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=new_dir),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.vllm_service.get_model_dir_context_window", new_callable=AsyncMock, return_value=4096),
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
    ):
        promise = await svc._install_model("default", "custom-model", InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert not old_dir.exists()
    assert svc.models_downloaded["custom-model"].model_path == str(new_dir)


@pytest.mark.asyncio
async def test_install_model_invalidates_log_cache_before_reading_capacity(svc: VllmService, deps: dict[str, Any], tmp_path: Path) -> None:
    """A reinstall on the same deterministic container name must not read the previous container's cached logs."""
    _setup_install_mocks(svc, deps)
    model_id = next(iter(svc.models["default"]))
    svc._log_cache["container"] = (time.monotonic(), "Maximum concurrency for 4096 tokens per request: 2.00x", 8.0)  # pyright: ignore[reportPrivateUsage]

    mock_result = MagicMock()
    mock_result.stdout = "Maximum concurrency for 4096 tokens per request: 16.00x"
    mock_result.stderr = ""

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.vllm_service.get_model_dir_context_window", new_callable=AsyncMock, return_value=4096),
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
        patch("server.services.base2_service.Utils.run_command", new_callable=AsyncMock, return_value=mock_result),
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    call_kwargs = deps["endpoint_registry"].register_chat_completion_as_proxy.call_args.kwargs
    assert call_kwargs["registration_options"].capacity_state == 16


@pytest.mark.asyncio
async def test_install_model_reuses_persisted_capacity_when_container_not_restarted(
    svc: VllmService, deps: dict[str, Any], tmp_path: Path
) -> None:
    """A no-op reinstall (container already running, config unchanged) must reuse the last known capacity, not re-read logs."""
    _setup_install_mocks(svc, deps)
    model_id = next(iter(svc.models["default"]))
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=(8000, False, False))
    svc.instances_info["default"].config = InstanceConfig(
        models=[ModelConfig(model_id=model_id, options=InstallModelIn(spec={}), capacity=16, capacity_known=True)]
    )

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.vllm_service.get_model_dir_context_window", new_callable=AsyncMock, return_value=4096),
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
        patch("server.services.base2_service.Utils.run_command", new_callable=AsyncMock) as mock_run_command,
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    mock_run_command.assert_not_called()
    call_kwargs = deps["endpoint_registry"].register_chat_completion_as_proxy.call_args.kwargs
    assert call_kwargs["registration_options"].capacity_state == 16


@pytest.mark.asyncio
async def test_install_model_capacity_unknown_when_not_restarted_and_never_persisted(
    svc: VllmService, deps: dict[str, Any], tmp_path: Path
) -> None:
    """A no-op reinstall with no prior known capacity and no recoverable log line must leave capacity unknown."""
    _setup_install_mocks(svc, deps)
    model_id = next(iter(svc.models["default"]))
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=(8000, False, False))

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.vllm_service.get_model_dir_context_window", new_callable=AsyncMock, return_value=4096),
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
        patch(
            "server.services.base2_service.Utils.run_command",
            new_callable=AsyncMock,
            return_value=CommandResult(exit_code=0, stdout="", stderr=""),
        ) as mock_run_command,
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    mock_run_command.assert_called_once()
    call_kwargs = deps["endpoint_registry"].register_chat_completion_as_proxy.call_args.kwargs
    assert call_kwargs["registration_options"].capacity_state == "unknown"


@pytest.mark.asyncio
async def test_install_model_capacity_recovered_from_logs_when_not_restarted_and_never_persisted(
    svc: VllmService, deps: dict[str, Any], tmp_path: Path
) -> None:
    """An adopted orphan container with no persisted capacity should still recover it from its startup logs."""
    _setup_install_mocks(svc, deps)
    model_id = next(iter(svc.models["default"]))
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=(8000, False, False))

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.vllm_service.get_model_dir_context_window", new_callable=AsyncMock, return_value=4096),
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
        patch(
            "server.services.base2_service.Utils.run_command",
            new_callable=AsyncMock,
            return_value=CommandResult(exit_code=0, stdout="Maximum concurrency for 4096 tokens per request: 2.00x", stderr=""),
        ),
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    call_kwargs = deps["endpoint_registry"].register_chat_completion_as_proxy.call_args.kwargs
    assert call_kwargs["registration_options"].capacity_state == 2


def test_get_persisted_model_capacity_returns_unknown_for_untracked_instance(svc: VllmService) -> None:
    assert svc._get_persisted_model_capacity("no-such-instance", "model") == (None, False)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_resolve_model_capacity_not_restarted_reuses_known_none_without_reading_logs(svc: VllmService) -> None:
    """A persisted `capacity=None, capacity_known=True` (a genuinely-checked-but-unbounded reading) must round-trip as-is."""
    svc.instances_info["default"].config = InstanceConfig(
        models=[ModelConfig(model_id="m", options=InstallModelIn(spec={}), capacity=None, capacity_known=True)]
    )

    with patch("server.services.base2_service.Utils.run_command", new_callable=AsyncMock) as mock_run_command:
        capacity, capacity_known = await svc._resolve_model_capacity(  # pyright: ignore[reportPrivateUsage]
            "default",
            "m",
            "vLLM",
            "container",
            False,
            svc._parse_max_concurrency,  # pyright: ignore[reportPrivateUsage]
        )

    mock_run_command.assert_not_called()
    assert capacity is None
    assert capacity_known is True


@pytest.mark.asyncio
async def test_install_model_registers_chat_completion_endpoint(svc: VllmService, deps: dict[str, Any], tmp_path: Path) -> None:
    _setup_install_mocks(svc, deps)
    model_id = next(iter(svc.models["default"]))

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.vllm_service.get_model_dir_context_window", new_callable=AsyncMock, return_value=4096),
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert deps["endpoint_registry"].register_chat_completion_as_proxy.call_count == 1


@pytest.mark.asyncio
async def test_install_model_records_model_as_downloaded(svc: VllmService, deps: dict[str, Any], tmp_path: Path) -> None:
    _setup_install_mocks(svc, deps)
    model_id = next(iter(svc.models["default"]))

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.vllm_service.get_model_dir_context_window", new_callable=AsyncMock, return_value=4096),
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert model_id in svc.models_downloaded


@pytest.mark.asyncio
async def test_install_model_uses_rename_as_registered_name(svc: VllmService, deps: dict[str, Any], tmp_path: Path) -> None:
    installed = _setup_install_mocks(svc, deps)
    model_id = next(iter(svc.models["default"]))

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.vllm_service.get_model_dir_context_window", new_callable=AsyncMock, return_value=4096),
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn(spec={"rename": "my-alias"}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert installed.models[model_id].registered_name == "my-alias"


@pytest.mark.asyncio
async def test_install_model_registers_reranker_endpoint_for_reranker_model(svc: VllmService, deps: dict[str, Any], tmp_path: Path) -> None:
    _setup_install_mocks(svc, deps)
    reranker_id = "reranker-model"
    svc.models["default"][reranker_id] = VllmModel(hf_id=reranker_id, size="1GB", model_type="reranker")

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
    ):
        promise = await svc._install_model("default", reranker_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert deps["endpoint_registry"].register_rerank_as_proxy.call_count == 1
    assert deps["endpoint_registry"].register_chat_completion_as_proxy.call_count == 0


@pytest.mark.asyncio
async def test_install_model_registers_transcriptions_for_stt_model(svc: VllmService, deps: dict[str, Any], tmp_path: Path) -> None:
    _setup_install_mocks(svc, deps)
    stt_id = "whisper-model"
    svc.models["default"][stt_id] = VllmModel(hf_id=stt_id, size="1GB", model_type="stt")

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
    ):
        promise = await svc._install_model("default", stt_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert deps["endpoint_registry"].register_audio_transcriptions_as_proxy.call_count == 1
    assert deps["endpoint_registry"].register_chat_completion_as_proxy.call_count == 0
    assert svc.get_instance_installed_info("default").models[stt_id].model_type == "stt"


@pytest.mark.asyncio
async def test_install_model_with_empty_spec_runs_reranker_with_default_extra_args(
    svc: VllmService, deps: dict[str, Any], tmp_path: Path
) -> None:
    _setup_install_mocks(svc, deps)
    deps["service_provider"].save_service_config = AsyncMock()
    deps["service_provider"].dismiss_warnings_matching_any = AsyncMock()
    reranker_id = "reranker-model"
    svc.models["default"][reranker_id] = VllmModel(hf_id=reranker_id, size="1GB", model_type="reranker")

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
        patch("server.services.vllm_service.validate_model_compatibility", new=AsyncMock(return_value=None)),
    ):
        promise = await svc.install_model("default", reranker_id, InstallModelIn(spec={}))
        await promise.wait()

    installed = svc.get_instance_installed_info("default").models[reranker_id]
    assert installed.options.spec is not None
    assert installed.options.spec["extra_args"] == {"--trust-remote-code": ""}
    docker_options = deps["docker_service"].install_and_run_docker.call_args.args[0]
    assert "--trust-remote-code" in docker_options.command


@pytest.mark.asyncio
async def test_install_model_docker_failure_decrements_gpu_memory(svc: VllmService, deps: dict[str, Any], tmp_path: Path) -> None:
    nvidia = MagicMock(spec=NvidiaGpuInfo)
    deps["hardware"].gpus = [nvidia]
    svc2 = VllmService(**deps)
    installed = _make_installed_info(hardware=True)
    svc2.instances_info["default"].installed = installed
    deps["docker_service"].install_and_run_docker = AsyncMock(side_effect=RuntimeError("docker failed"))
    deps["docker_service"].get_docker_subnet.return_value = None
    deps["docker_service"].get_docker_container_name.return_value = "container"
    deps["docker_service"].stop_docker = AsyncMock()
    model_id = next(iter(svc2.models["default"]))

    with (
        patch.object(svc2, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc2, "get_specified_hardware_parts", return_value=[nvidia]),
        patch.object(svc2, "is_given_hardware_support_gpu", return_value=True),
    ):
        promise = await svc2._install_model("default", model_id, InstallModelIn(spec={"gpu_memory_utilization": 0.5}))  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(RuntimeError):
            await promise.wait()

    assert svc2.gpu_memory_utilization == 0.0


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("the estimated maximum model length is 110256.", 110256),
        ("estimated maximum model length is 4096", 4096),
        ("docker failed for some other reason", None),
    ],
)
def test_parse_kv_cache_max_len_suggestion(raw: str, expected: int | None) -> None:
    assert VllmService._parse_kv_cache_max_len_suggestion(raw) == expected  # pyright: ignore[reportPrivateUsage]


def test_check_prefix_collision_skips_models_without_default_prefix(svc: VllmService) -> None:
    """`VllmModel` has no `default_prefix` field (DFINFRA-309) - unlike MCP/Custom models, whose
    runtime model type is always addressed by a prefix. The collision scan over every catalog model
    of the instance must not crash reading `default_prefix` off one that was never installed."""
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    assert len(svc.models["default"]) > 1

    svc._check_prefix_collision("default", "some-new-prefix", exclude_model_id=None)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_edit_model_install_options_without_prefix_does_not_crash(svc: VllmService, deps: dict[str, Any], tmp_path: Path) -> None:
    """DFINFRA-309: editing a vLLM catalog model's install options without an explicit `prefix` in the
    spec used to unconditionally read `model.default_prefix`, which `VllmModel` doesn't define, and
    crashed with a 500 AttributeError before any uninstall/reinstall was attempted."""
    _setup_install_mocks(svc, deps)
    deps["service_provider"].save_service_config = AsyncMock()
    model_id = next(iter(svc.models["default"]))

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.vllm_service.get_model_dir_context_window", new_callable=AsyncMock, return_value=4096),
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
    ):
        was_installed, promise = await svc.edit_model_install_options(
            "default", model_id, InstallModelIn(spec={"gpu_memory_utilization": 0.6})
        )
        result = await promise.wait()

    assert was_installed is False
    assert result.status == "OK"


@pytest.mark.asyncio
async def test_edit_model_on_installed_custom_model_does_not_crash(svc: VllmService, deps: dict[str, Any], tmp_path: Path) -> None:
    """DFINFRA-309: `VllmModel` (the runtime type behind both catalog and custom vLLM models) has no
    `default_prefix` field. Editing an installed custom model used to unconditionally read
    `updated_model.default_prefix` while computing the reinstall prefix, crashing before the
    definition change could actually be applied."""
    _setup_install_mocks(svc, deps)
    deps["service_provider"].save_service_config = AsyncMock()
    deps["service_provider"].dismiss_warnings_matching = AsyncMock()
    deps["service_provider"].dismiss_warnings_matching_any = AsyncMock()
    deps["docker_service"].uninstall_docker = AsyncMock()
    custom_data = {"id": "my-model", "hf_id": "google/gemma-3-270m-it", "size": "1GB"}
    custom = CustomModel(id="c-1", data=custom_data)
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]
    svc.instances_info["default"].config.custom = [custom]

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.vllm_service.get_model_dir_context_window", new_callable=AsyncMock, return_value=4096),
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
        patch("server.services.vllm_service.validate_model_compatibility", new=AsyncMock(return_value=None)),
    ):
        install_promise = await svc.install_model("default", "my-model", InstallModelIn(spec={}))
        await install_promise.wait()

        promise = await svc.edit_model("default", "c-1", AddCustomModelIn(spec={**custom_data, "revision": "quantized-awq"}))
        assert promise is not None
        result = await promise.wait()

    assert result.status == "OK"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("GPU KV cache size: 100,000 tokens, Maximum concurrency for 4096 tokens per request: 12.34x", 12),
        ("Maximum concurrency for 4096 tokens per request: 1.00x", 1),
        ("docker started successfully with no relevant log line", None),
        ("Maximum concurrency for 4096 tokens per request: 0.73x", 1),
        (
            "Maximum concurrency for 4096 tokens per request: 4.00x\n"
            "restarting with smaller max-model-len\n"
            "Maximum concurrency for 4096 tokens per request: 8.00x",
            8,
        ),
        ("Maximum concurrency for 4096 tokens per request: 1.2.3x", None),
        # A number long enough that float() rounds it to inf must not raise OverflowError out of round().
        (f"Maximum concurrency for 4096 tokens per request: {'9' * 400}.0x", None),
    ],
)
def test_parse_max_concurrency(raw: str, expected: int | None) -> None:
    assert VllmService._parse_max_concurrency(raw) == expected  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_get_max_concurrency_from_logs_returns_parsed_value(svc: VllmService) -> None:
    with patch.object(
        svc, "_get_docker_logs", new_callable=AsyncMock, return_value="Maximum concurrency for 4096 tokens per request: 8.00x"
    ):  # pyright: ignore[reportPrivateUsage]
        result = await svc._get_max_concurrency_from_logs("my-container", "vLLM", svc._parse_max_concurrency)  # pyright: ignore[reportPrivateUsage]

    assert result == 8


@pytest.mark.asyncio
async def test_get_max_concurrency_from_logs_returns_none_on_failure(svc: VllmService) -> None:
    with patch.object(svc, "_get_docker_logs", new_callable=AsyncMock, side_effect=RuntimeError("boom")):  # pyright: ignore[reportPrivateUsage]
        result = await svc._get_max_concurrency_from_logs("my-container", "vLLM", svc._parse_max_concurrency)  # pyright: ignore[reportPrivateUsage]

    assert result is None


@pytest.mark.asyncio
async def test_install_model_retries_with_suggested_max_len_on_kv_cache_error(
    svc: VllmService, deps: dict[str, Any], tmp_path: Path
) -> None:
    nvidia = MagicMock(spec=NvidiaGpuInfo)
    deps["hardware"].gpus = [nvidia]
    svc2 = VllmService(**deps)
    installed = _make_installed_info(hardware=True)
    svc2.instances_info["default"].installed = installed
    kv_cache_error = AppError(
        "ValueError: To serve at least one request with the models's max seq len (131072), "
        "(4.0 GiB KV cache is needed, which is larger than the available KV cache memory (3.37 GiB). "
        "Based on the available memory, the estimated maximum model length is 110256."
    )
    deps["docker_service"].install_and_run_docker = AsyncMock(side_effect=[kv_cache_error, (8000, True, False)])
    deps["docker_service"].get_docker_subnet.return_value = None
    deps["docker_service"].get_docker_container_name.return_value = "container"
    deps["docker_service"].get_container_host.return_value = "localhost"
    deps["docker_service"].get_container_port.return_value = 8000
    deps["docker_service"].stop_docker = AsyncMock()
    model_id = next(iter(svc2.models["default"]))

    with (
        patch.object(svc2, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.vllm_service.get_model_dir_context_window", new_callable=AsyncMock, return_value=131072),
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc2, "get_specified_hardware_parts", return_value=[nvidia]),
        patch.object(svc2, "is_given_hardware_support_gpu", return_value=True),
    ):
        promise = await svc2._install_model("default", model_id, InstallModelIn(spec={"gpu_memory_utilization": 0.5}))  # pyright: ignore[reportPrivateUsage]
        result = await promise.wait()

    assert result.status == "OK"
    assert deps["docker_service"].install_and_run_docker.call_count == 2
    register_kwargs = deps["endpoint_registry"].register_chat_completion_as_proxy.call_args.kwargs
    assert register_kwargs["props"].context_window == 110256


@pytest.mark.asyncio
@patch.object(VllmService, "get_specified_hardware_parts")
@patch.object(VllmService, "is_given_hardware_support_gpu", return_value=True)
@patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000")
@patch("server.services.vllm_service.get_model_dir_context_window", new_callable=AsyncMock, return_value=131072)
@patch.object(VllmService, "_download_model_or_set_progress", new_callable=AsyncMock)
async def test_install_model_keeps_json_extra_arg_verbatim_on_kv_cache_retry(
    mock_download_model_or_set_progress: AsyncMock,
    mock_get_model_dir_context_window: AsyncMock,
    mock_get_base_url: MagicMock,
    mock_is_given_hardware_support_gpu: MagicMock,
    mock_get_specified_hardware_parts: MagicMock,
    deps: dict[str, Any],
    tmp_path: Path,
):
    nvidia = MagicMock(spec=NvidiaGpuInfo)
    deps["hardware"].gpus = [nvidia]
    gpu_svc = VllmService(**deps)
    _setup_install_mocks(gpu_svc, deps, hardware=True)
    kv_cache_error = AppError("the estimated maximum model length is 110256.")
    deps["docker_service"].install_and_run_docker = AsyncMock(side_effect=[kv_cache_error, (8000, True, False)])
    deps["docker_service"].stop_docker = AsyncMock()
    mock_download_model_or_set_progress.return_value = tmp_path / "model"
    mock_get_specified_hardware_parts.return_value = [nvidia]
    model_id = next(iter(gpu_svc.models["default"]))
    json_value = '{"method":"mtp","num_speculative_tokens":3}'
    spec = {"gpu_memory_utilization": 0.5, "extra_args": {"--speculative-config": json_value}}

    promise = await gpu_svc._install_model("default", model_id, InstallModelIn(spec=spec))  # pyright: ignore[reportPrivateUsage]
    await promise.wait()

    assert deps["docker_service"].install_and_run_docker.call_count == 2
    retried_command = deps["docker_service"].install_and_run_docker.call_args.args[0].command
    assert isinstance(retried_command, list)
    assert retried_command[retried_command.index("--max-model-len") + 1] == "110256"
    assert retried_command[retried_command.index("--speculative-config") + 1] == json_value


@pytest.mark.asyncio
async def test_install_model_does_not_retry_when_max_model_length_is_explicit(
    svc: VllmService, deps: dict[str, Any], tmp_path: Path
) -> None:
    nvidia = MagicMock(spec=NvidiaGpuInfo)
    deps["hardware"].gpus = [nvidia]
    svc2 = VllmService(**deps)
    installed = _make_installed_info(hardware=True)
    svc2.instances_info["default"].installed = installed
    kv_cache_error = AppError("the estimated maximum model length is 110256.")
    deps["docker_service"].install_and_run_docker = AsyncMock(side_effect=kv_cache_error)
    deps["docker_service"].get_docker_subnet.return_value = None
    deps["docker_service"].get_docker_container_name.return_value = "container"
    deps["docker_service"].stop_docker = AsyncMock()
    model_id = next(iter(svc2.models["default"]))

    with (
        patch.object(svc2, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc2, "get_specified_hardware_parts", return_value=[nvidia]),
        patch.object(svc2, "is_given_hardware_support_gpu", return_value=True),
    ):
        promise = await svc2._install_model(  # pyright: ignore[reportPrivateUsage]
            "default", model_id, InstallModelIn(spec={"gpu_memory_utilization": 0.5, "max_model_length": 4096})
        )
        with pytest.raises(AppError):
            await promise.wait()

    assert deps["docker_service"].install_and_run_docker.call_count == 1


@pytest.mark.asyncio
async def test_install_model_reraises_when_retry_also_fails(svc: VllmService, deps: dict[str, Any], tmp_path: Path) -> None:
    nvidia = MagicMock(spec=NvidiaGpuInfo)
    deps["hardware"].gpus = [nvidia]
    svc2 = VllmService(**deps)
    installed = _make_installed_info(hardware=True)
    svc2.instances_info["default"].installed = installed
    kv_cache_error = AppError("the estimated maximum model length is 110256.")
    retry_error = AppError("still failing")
    deps["docker_service"].install_and_run_docker = AsyncMock(side_effect=[kv_cache_error, retry_error])
    deps["docker_service"].get_docker_subnet.return_value = None
    deps["docker_service"].get_docker_container_name.return_value = "container"
    deps["docker_service"].stop_docker = AsyncMock()
    model_id = next(iter(svc2.models["default"]))

    with (
        patch.object(svc2, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc2, "get_specified_hardware_parts", return_value=[nvidia]),
        patch.object(svc2, "is_given_hardware_support_gpu", return_value=True),
    ):
        promise = await svc2._install_model("default", model_id, InstallModelIn(spec={"gpu_memory_utilization": 0.5}))  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(AppError, match="still failing"):
            await promise.wait()

    assert deps["docker_service"].install_and_run_docker.call_count == 2
    assert deps["docker_service"].stop_docker.call_count == 2


@pytest.mark.parametrize(
    "raw",
    [
        "ValueError: ... is larger than the available KV cache memory (3.37 GiB) ...",
        "No available memory for the cache blocks.",
        "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB",
        "CUDA error: out of memory",
    ],
)
def test_diagnose_insufficient_vram_matches_known_patterns(raw: str) -> None:
    msg = VllmService._diagnose_insufficient_vram("my-model", raw)  # pyright: ignore[reportPrivateUsage]
    assert msg is not None
    assert "my-model" in msg
    assert "VRAM" in msg


def test_diagnose_insufficient_vram_returns_none_for_unrelated_error() -> None:
    assert VllmService._diagnose_insufficient_vram("my-model", "connection refused") is None  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize(
    "raw",
    [
        "huggingface_hub.errors.LocalEntryNotFoundError: Cannot find the requested files in the disk cache "
        "and outgoing traffic has been disabled.",
        "OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.",
    ],
)
def test_diagnose_missing_model_files_matches_known_patterns(raw: str) -> None:
    msg = VllmService._diagnose_missing_model_files("my-model", raw)  # pyright: ignore[reportPrivateUsage]
    assert msg is not None
    assert "my-model" in msg
    assert "missing files" in msg


def test_diagnose_missing_model_files_returns_none_for_unrelated_error() -> None:
    assert VllmService._diagnose_missing_model_files("my-model", "connection refused") is None  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_install_model_raises_friendly_error_when_retry_fails_on_vram(svc: VllmService, deps: dict[str, Any], tmp_path: Path) -> None:
    nvidia = MagicMock(spec=NvidiaGpuInfo)
    deps["hardware"].gpus = [nvidia]
    svc2 = VllmService(**deps)
    installed = _make_installed_info(hardware=True)
    svc2.instances_info["default"].installed = installed
    kv_cache_error = RuntimeError("the estimated maximum model length is 4096.")
    vram_error = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
    deps["docker_service"].install_and_run_docker = AsyncMock(side_effect=[kv_cache_error, vram_error])
    deps["docker_service"].get_docker_subnet.return_value = None
    deps["docker_service"].get_docker_container_name.return_value = "container"
    deps["docker_service"].stop_docker = AsyncMock()
    model_id = next(iter(svc2.models["default"]))

    with (
        patch.object(svc2, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc2, "get_specified_hardware_parts", return_value=[nvidia]),
        patch.object(svc2, "is_given_hardware_support_gpu", return_value=True),
    ):
        promise = await svc2._install_model("default", model_id, InstallModelIn(spec={"gpu_memory_utilization": 0.5}))  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(AppError, match="Not enough GPU VRAM"):
            await promise.wait()

    assert deps["docker_service"].install_and_run_docker.call_count == 2
    assert deps["docker_service"].stop_docker.call_count == 2


@pytest.mark.asyncio
async def test_install_model_raises_friendly_error_on_vram_when_max_model_length_is_explicit(
    svc: VllmService, deps: dict[str, Any], tmp_path: Path
) -> None:
    nvidia = MagicMock(spec=NvidiaGpuInfo)
    deps["hardware"].gpus = [nvidia]
    svc2 = VllmService(**deps)
    installed = _make_installed_info(hardware=True)
    svc2.instances_info["default"].installed = installed
    vram_error = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
    deps["docker_service"].install_and_run_docker = AsyncMock(side_effect=vram_error)
    deps["docker_service"].get_docker_subnet.return_value = None
    deps["docker_service"].get_docker_container_name.return_value = "container"
    deps["docker_service"].stop_docker = AsyncMock()
    model_id = next(iter(svc2.models["default"]))

    with (
        patch.object(svc2, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc2, "get_specified_hardware_parts", return_value=[nvidia]),
        patch.object(svc2, "is_given_hardware_support_gpu", return_value=True),
    ):
        promise = await svc2._install_model(  # pyright: ignore[reportPrivateUsage]
            "default", model_id, InstallModelIn(spec={"gpu_memory_utilization": 0.5, "max_model_length": 4096})
        )
        with pytest.raises(AppError, match="Not enough GPU VRAM"):
            await promise.wait()

    assert deps["docker_service"].install_and_run_docker.call_count == 1
    assert deps["docker_service"].stop_docker.call_count == 1


@pytest.mark.asyncio
async def test_install_model_raises_friendly_error_when_retry_fails_on_missing_files(
    svc: VllmService, deps: dict[str, Any], tmp_path: Path
) -> None:
    nvidia = MagicMock(spec=NvidiaGpuInfo)
    deps["hardware"].gpus = [nvidia]
    svc2 = VllmService(**deps)
    installed = _make_installed_info(hardware=True)
    svc2.instances_info["default"].installed = installed
    kv_cache_error = RuntimeError("the estimated maximum model length is 4096.")
    missing_files_error = RuntimeError(
        "huggingface_hub.errors.LocalEntryNotFoundError: Cannot find the requested files in the disk cache "
        "and outgoing traffic has been disabled."
    )
    deps["docker_service"].install_and_run_docker = AsyncMock(side_effect=[kv_cache_error, missing_files_error])
    deps["docker_service"].get_docker_subnet.return_value = None
    deps["docker_service"].get_docker_container_name.return_value = "container"
    deps["docker_service"].stop_docker = AsyncMock()
    model_id = next(iter(svc2.models["default"]))

    with (
        patch.object(svc2, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc2, "get_specified_hardware_parts", return_value=[nvidia]),
        patch.object(svc2, "is_given_hardware_support_gpu", return_value=True),
    ):
        promise = await svc2._install_model("default", model_id, InstallModelIn(spec={"gpu_memory_utilization": 0.5}))  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(AppError, match="missing files"):
            await promise.wait()

    assert deps["docker_service"].install_and_run_docker.call_count == 2
    assert deps["docker_service"].stop_docker.call_count == 2


@pytest.mark.asyncio
async def test_install_model_raises_friendly_error_when_model_files_are_missing(
    svc: VllmService, deps: dict[str, Any], tmp_path: Path
) -> None:
    nvidia = MagicMock(spec=NvidiaGpuInfo)
    deps["hardware"].gpus = [nvidia]
    svc2 = VllmService(**deps)
    installed = _make_installed_info(hardware=True)
    svc2.instances_info["default"].installed = installed
    missing_files_error = RuntimeError(
        "huggingface_hub.errors.LocalEntryNotFoundError: Cannot find the requested files in the disk cache "
        "and outgoing traffic has been disabled."
    )
    deps["docker_service"].install_and_run_docker = AsyncMock(side_effect=missing_files_error)
    deps["docker_service"].get_docker_subnet.return_value = None
    deps["docker_service"].get_docker_container_name.return_value = "container"
    deps["docker_service"].stop_docker = AsyncMock()
    model_id = next(iter(svc2.models["default"]))

    with (
        patch.object(svc2, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc2, "get_specified_hardware_parts", return_value=[nvidia]),
        patch.object(svc2, "is_given_hardware_support_gpu", return_value=True),
    ):
        promise = await svc2._install_model("default", model_id, InstallModelIn(spec={"gpu_memory_utilization": 0.5}))  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(AppError, match="missing files"):
            await promise.wait()

    assert deps["docker_service"].install_and_run_docker.call_count == 1
    assert deps["docker_service"].stop_docker.call_count == 1


@pytest.mark.asyncio
async def test_uninstall_model_removes_from_installed_and_unregisters_endpoint(svc: VllmService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    installed.models["test-model"] = _make_model_installed_info("test-model", "reg-1")
    svc.instances_info["default"].installed = installed
    deps["docker_service"].uninstall_docker = AsyncMock()

    await svc._uninstall_model("default", "test-model", UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert "test-model" not in installed.models
    assert deps["endpoint_registry"].unregister_chat_completion.call_count == 1
    assert deps["endpoint_registry"].unregister_chat_completion.call_args == call("test-model", "reg-1")


@pytest.mark.asyncio
async def test_uninstall_model_decrements_gpu_memory_utilization(svc: VllmService, deps: dict[str, Any]) -> None:
    svc.gpu_memory_utilization = 0.5
    installed = _make_installed_info()
    installed.models["test-model"] = _make_model_installed_info("test-model", gpu_memory_utilization=0.5)
    svc.instances_info["default"].installed = installed
    deps["docker_service"].uninstall_docker = AsyncMock()

    await svc._uninstall_model("default", "test-model", UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert svc.gpu_memory_utilization == 0.0


@pytest.mark.asyncio
async def test_uninstall_model_floors_gpu_memory_at_zero(svc: VllmService, deps: dict[str, Any]) -> None:
    svc.gpu_memory_utilization = 0.1
    installed = _make_installed_info()
    installed.models["test-model"] = _make_model_installed_info("test-model", gpu_memory_utilization=0.5)
    svc.instances_info["default"].installed = installed
    deps["docker_service"].uninstall_docker = AsyncMock()

    await svc._uninstall_model("default", "test-model", UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert svc.gpu_memory_utilization == 0.0


@pytest.mark.asyncio
async def test_uninstall_model_purges_files_and_downloaded_entry(svc: VllmService, deps: dict[str, Any], tmp_path: Path) -> None:
    installed = _make_installed_info()
    model_dir = tmp_path / "model-dir"
    model_dir.mkdir()
    model_info = _make_model_installed_info("test-model")
    model_info.model_path = model_dir
    installed.models["test-model"] = model_info
    svc.instances_info["default"].installed = installed
    svc.models_downloaded["test-model"] = DownloadedInfo(model_path=str(model_dir))
    deps["docker_service"].uninstall_docker = AsyncMock()

    await svc._uninstall_model("default", "test-model", UninstallModelIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert not model_dir.exists()
    assert "test-model" not in svc.models_downloaded


@pytest.mark.asyncio
async def test_uninstall_model_purges_adapter_directory(svc: VllmService, deps: dict[str, Any], tmp_path: Path) -> None:
    installed = _make_installed_info()
    model_dir = tmp_path / "model-dir"
    model_dir.mkdir()
    adapter_dir = tmp_path / "adapter-dir"
    adapter_dir.mkdir()
    model_info = _make_model_installed_info("pirate-bot")
    model_info.model_path = model_dir
    installed.models["pirate-bot"] = model_info
    svc.instances_info["default"].installed = installed
    svc.models_downloaded["pirate-bot"] = DownloadedInfo(model_path=str(model_dir), adapter_path=str(adapter_dir))
    deps["docker_service"].uninstall_docker = AsyncMock()

    await svc._uninstall_model("default", "pirate-bot", UninstallModelIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert not adapter_dir.exists()
    assert "pirate-bot" not in svc.models_downloaded


@pytest.mark.asyncio
async def test_uninstall_model_reranker_calls_unregister_rerank(svc: VllmService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    installed.models["reranker-model"] = _make_model_installed_info("reranker-model", "reg-rerank", model_type="reranker")
    svc.instances_info["default"].installed = installed
    deps["docker_service"].uninstall_docker = AsyncMock()

    await svc._uninstall_model("default", "reranker-model", UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert deps["endpoint_registry"].unregister_rerank.call_count == 1
    assert deps["endpoint_registry"].unregister_rerank.call_args == call("reranker-model", "reg-rerank")
    assert deps["endpoint_registry"].unregister_chat_completion.call_count == 0


@pytest.mark.asyncio
async def test_uninstall_model_stt_calls_unregister_audio_transcriptions(svc: VllmService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    installed.models["whisper-model"] = _make_model_installed_info("whisper-model", "reg-stt", model_type="stt")
    svc.instances_info["default"].installed = installed
    deps["docker_service"].uninstall_docker = AsyncMock()

    await svc._uninstall_model("default", "whisper-model", UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert deps["endpoint_registry"].unregister_audio_transcriptions.call_args == call("whisper-model", "reg-stt")
    assert deps["endpoint_registry"].unregister_chat_completion.call_count == 0
    assert "whisper-model" not in installed.models


@pytest.mark.asyncio
async def test_uninstall_model_embedding_calls_unregister_embeddings(svc: VllmService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    installed.models["embedder-model"] = _make_model_installed_info("embedder-model", "reg-embed", model_type="embedding")
    svc.instances_info["default"].installed = installed
    deps["docker_service"].uninstall_docker = AsyncMock()

    await svc._uninstall_model("default", "embedder-model", UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert deps["endpoint_registry"].unregister_embeddings.call_count == 1
    assert deps["endpoint_registry"].unregister_embeddings.call_args == call("embedder-model", "reg-embed")
    assert deps["endpoint_registry"].unregister_chat_completion.call_count == 0
    assert deps["endpoint_registry"].unregister_rerank.call_count == 0


@pytest.mark.asyncio
async def test_uninstall_model_ignores_unknown_model_id(svc: VllmService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    deps["docker_service"].uninstall_docker = AsyncMock()

    await svc._uninstall_model("default", "unknown-model", UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert deps["endpoint_registry"].unregister_chat_completion.call_count == 0
    assert deps["docker_service"].uninstall_docker.call_count == 0


@pytest.mark.asyncio
async def test_install_instance_returns_installed_info_with_empty_models(svc: VllmService) -> None:
    options = InstallServiceIn(spec={"hardware": False})

    with (
        patch.object(svc, "_download_image_or_set_progress", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_verify_docker_image", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
    ):
        promise = await svc._install_instance("default", options)  # pyright: ignore[reportPrivateUsage]
        result = await promise.wait()

    assert isinstance(result, InstalledInfo)
    assert result.models == {}


@pytest.mark.asyncio
async def test_uninstall_instance_calls_uninstall_model_for_each_model(svc: VllmService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    installed.models["m1"] = _make_model_installed_info("m1")
    installed.models["m2"] = _make_model_installed_info("m2")
    svc.instances_info["default"].installed = installed
    deps["docker_service"].uninstall_docker = AsyncMock()

    with patch.object(svc, "_uninstall_model", new_callable=AsyncMock) as mock_uninstall:  # pyright: ignore[reportPrivateUsage]
        await svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert mock_uninstall.call_count == 2


@pytest.mark.asyncio
async def test_uninstall_instance_purge_clears_service_state(svc: VllmService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    deps["docker_service"].remove_image = AsyncMock()
    deps["docker_service"].uninstall_docker = AsyncMock()

    with (
        patch.object(svc, "_uninstall_model", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_clear_working_dir", new_callable=AsyncMock) as mock_clear,  # pyright: ignore[reportPrivateUsage]
    ):
        await svc._uninstall_instance("default", UninstallServiceIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert svc.service_downloaded is False
    assert mock_clear.call_count == 1


@pytest.mark.asyncio
async def test_uninstall_instance_purge_with_other_instance_installed_does_not_remove_image(svc: VllmService, deps: dict[str, Any]) -> None:
    svc.instances_info["extra"] = Instance(None, None, {}, InstanceConfig())
    svc.instances_info["extra"].installed = _make_installed_info()
    svc.instances_info["default"].installed = _make_installed_info()
    deps["docker_service"].remove_image = AsyncMock()

    with (
        patch.object(svc, "_uninstall_model", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_clear_working_dir", new_callable=AsyncMock) as mock_clear,  # pyright: ignore[reportPrivateUsage]
    ):
        await svc._uninstall_instance("default", UninstallServiceIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert deps["docker_service"].remove_image.call_count == 0
    assert mock_clear.call_count == 0
    assert svc.instances_info["default"].installed is None
    assert "extra" in svc.instances_info


@pytest.mark.asyncio
async def test_uninstall_instance_purge_removes_per_model_docker_images(svc: VllmService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    model_info = _make_model_installed_info("test-model")
    model_info.docker.image = "vllm-model-image:tag"
    installed.models["test-model"] = model_info
    svc.instances_info["default"].installed = installed
    deps["docker_service"].remove_image = AsyncMock()

    with (
        patch.object(svc, "_uninstall_model", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_clear_working_dir", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
    ):
        await svc._uninstall_instance("default", UninstallServiceIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    removed_images = {c.args[0] for c in deps["docker_service"].remove_image.call_args_list}
    assert "vllm-model-image:tag" in removed_images


@pytest.mark.asyncio
async def test_stop_instance_does_nothing_when_not_installed(svc: VllmService) -> None:
    svc.instances_info["default"].installed = None

    with patch.object(svc, "_stop_dockers_parallel", new_callable=AsyncMock) as mock_stop:  # pyright: ignore[reportPrivateUsage]
        await svc.stop_instance("default")

    assert mock_stop.call_count == 0


@pytest.mark.asyncio
async def test_stop_instance_stops_all_containers(svc: VllmService) -> None:
    installed = _make_installed_info()
    installed.models["m1"] = _make_model_installed_info("m1")
    installed.models["m2"] = _make_model_installed_info("m2")
    svc.instances_info["default"].installed = installed

    with patch.object(svc, "_stop_dockers_parallel", new_callable=AsyncMock) as mock_stop:  # pyright: ignore[reportPrivateUsage]
        await svc.stop_instance("default")

    assert mock_stop.call_count == 1
    called_dockers = mock_stop.call_args[0][0]
    assert len(called_dockers) == 2


def test_get_specified_hardware_parts_delegates_to_super_when_hardware_ok(svc: VllmService) -> None:
    svc.hardware.cpu.avx512 = True

    with patch("server.services.base2_service.Base2Service.get_specified_hardware_parts", return_value=[]) as mock_super:
        result = svc.get_specified_hardware_parts(False)

    assert mock_super.call_count == 1
    assert result == []


def test_get_model_type_spec_adds_hardware_key_when_missing(svc: VllmService, deps: dict[str, Any]) -> None:
    deps["docker_service"].has_gpu_support = False
    svc.instances_info["default"].config = InstanceConfig(options=InstallServiceIn(spec={}))

    result = svc.get_model_type_spec("default", "llm")

    assert result is not None


@pytest.mark.asyncio
async def test_install_instance_loads_default_models_for_new_instance(svc: VllmService, deps: dict[str, Any]) -> None:
    svc.instances_info["inst2"] = Instance(None, None, {}, InstanceConfig())
    options = InstallServiceIn(spec={"hardware": False})
    svc.hardware.cpu.avx512 = True

    with (
        patch.object(svc, "_download_image_or_set_progress", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_verify_docker_image", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "load_default_models") as mock_load,
    ):
        promise = await svc._install_instance("inst2", options)  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert mock_load.call_count == 1
    assert mock_load.call_args == call("inst2")


@pytest.mark.asyncio
async def test_install_instance_adds_hardware_key_when_missing(svc: VllmService, deps: dict[str, Any]) -> None:
    options = InstallServiceIn(spec={})
    deps["docker_service"].has_gpu_support = False
    svc.hardware.cpu.avx512 = True

    with (
        patch.object(svc, "_download_image_or_set_progress", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_verify_docker_image", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
    ):
        await svc._install_instance("default", options)  # pyright: ignore[reportPrivateUsage]

    assert "hardware" in options.spec


@pytest.mark.asyncio
async def test_uninstall_instance_purge_removes_non_default_instance(svc: VllmService, deps: dict[str, Any]) -> None:
    svc.instances_info["inst2"] = Instance(None, None, {}, InstanceConfig())
    installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}), parsed_options=VllmOptions())
    svc.instances_info["inst2"].installed = installed
    deps["docker_service"].remove_image = AsyncMock()

    with (
        patch.object(svc, "_uninstall_model", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_clear_working_dir", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
    ):
        await svc._uninstall_instance("inst2", UninstallServiceIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert "inst2" not in svc.instances_info


@pytest.mark.asyncio
async def test_list_models_skips_instances_not_in_filter(svc: VllmService) -> None:
    svc.instances_info["inst2"] = Instance(None, None, {}, InstanceConfig())
    svc.models["inst2"] = {"extra-model": VllmModel(hf_id="extra/model", size="1GB")}
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}), parsed_options=VllmOptions())

    result = await svc.list_models("default", ListModelsFilters())

    assert not any(m.id == "extra-model" for m in result.list)


@pytest.mark.asyncio
async def test_download_model_handles_pre_download_packet_with_size(svc: VllmService, deps: dict[str, Any], tmp_path: Path) -> None:
    model = VllmModel(hf_id="google/test", size="100MB")
    stream = MagicMock()

    async def mock_download(*args: object, **kwargs: object):  # type: ignore[misc]
        yield PreDownloadPacket(file_bytes_size=50 * 1024 * 1024)
        yield SuccessDownloadPacket(local_path=tmp_path / "model", filename="model")

    deps["model_downloader"].download = mock_download

    with patch.object(svc, "_get_working_dir", return_value=tmp_path):  # pyright: ignore[reportPrivateUsage]
        local_path = await svc._download_model(stream, model, tmp_path)  # pyright: ignore[reportPrivateUsage]

    assert local_path is not None


@pytest.mark.asyncio
async def test_download_model_or_set_progress_forwards_chunk_with_data(svc: VllmService, tmp_path: Path) -> None:
    existing_stream: Stream[StreamChunk] = Stream()  # type: ignore[type-arg]
    chunk = StreamChunkProgress(type="progress", stage="download", value=0.9, data={"local_model_path": str(tmp_path / "model")})
    existing_stream.emit(chunk)
    existing_stream.close()
    model = VllmModel(hf_id="google/test", size="1GB")
    svc.models_download_progress["google/test"] = existing_stream  # type: ignore[assignment]
    output_stream = MagicMock()

    result_path = await svc._download_model_or_set_progress(output_stream, "google/test", model, tmp_path)  # pyright: ignore[reportPrivateUsage]

    assert result_path == tmp_path / "model"


@pytest.mark.asyncio
async def test_download_model_or_set_progress_breaks_on_non_download_chunk(svc: VllmService, tmp_path: Path) -> None:
    existing_stream: Stream[StreamChunk] = Stream()  # type: ignore[type-arg]
    finish_chunk: StreamChunk = {"type": "finish", "status": "ok"}  # type: ignore[assignment]
    progress_chunk = StreamChunkProgress(type="progress", stage="download", value=1.0, data={"local_model_path": str(tmp_path / "model")})
    existing_stream.emit(progress_chunk)
    existing_stream.emit(finish_chunk)
    existing_stream.close()
    model = VllmModel(hf_id="google/test", size="1GB")
    svc.models_download_progress["google/test"] = existing_stream  # type: ignore[assignment]
    output_stream = MagicMock()

    result_path = await svc._download_model_or_set_progress(output_stream, "google/test", model, tmp_path)  # pyright: ignore[reportPrivateUsage]

    assert result_path is not None


def test_build_vllm_command_use_gpu_with_model_length(svc: VllmService, tmp_path: Path) -> None:
    opts = VllmModelOptions.model_construct(extra_args={})

    cmd = svc._build_vllm_command(  # pyright: ignore[reportPrivateUsage]
        docker_model_path=tmp_path / "model",
        model_id="google/test",
        opts=opts,
        quantization=None,
        gpu_memory_utilization=None,
        user_model_length=4096,
        use_gpu=True,
    )

    assert "--max-model-len" in cmd
    assert "4096" in cmd


def test_build_vllm_command_no_gpu_no_model_length_adds_disable_sliding_window(svc: VllmService, tmp_path: Path) -> None:
    opts = VllmModelOptions.model_construct(extra_args={})

    cmd = svc._build_vllm_command(  # pyright: ignore[reportPrivateUsage]
        docker_model_path=tmp_path / "model",
        model_id="google/test",
        opts=opts,
        quantization=None,
        gpu_memory_utilization=None,
        user_model_length=None,
        use_gpu=False,
    )

    assert "--disable-sliding-window" in cmd


def test_build_vllm_command_extra_args_with_value(svc: VllmService, tmp_path: Path) -> None:
    opts = VllmModelOptions.model_construct(extra_args={"--dtype": "float16"})

    cmd = svc._build_vllm_command(  # pyright: ignore[reportPrivateUsage]
        docker_model_path=tmp_path / "model",
        model_id="google/test",
        opts=opts,
        quantization=None,
        gpu_memory_utilization=None,
        user_model_length=None,
        use_gpu=True,
    )

    assert "--dtype" in cmd
    assert "float16" in cmd


def test_build_vllm_command_extra_args_flag_only(svc: VllmService, tmp_path: Path) -> None:
    opts = VllmModelOptions.model_construct(extra_args={"--enforce-eager": None})

    cmd = svc._build_vllm_command(  # pyright: ignore[reportPrivateUsage]
        docker_model_path=tmp_path / "model",
        model_id="google/test",
        opts=opts,
        quantization=None,
        gpu_memory_utilization=None,
        user_model_length=None,
        use_gpu=True,
    )

    assert "--enforce-eager" in cmd


def test_build_vllm_command_no_gpu_with_model_length(svc: VllmService, tmp_path: Path) -> None:
    opts = VllmModelOptions.model_construct(extra_args={})

    cmd = svc._build_vllm_command(  # pyright: ignore[reportPrivateUsage]
        docker_model_path=tmp_path / "model",
        model_id="google/test",
        opts=opts,
        quantization=None,
        gpu_memory_utilization=None,
        user_model_length=2048,
        use_gpu=False,
    )

    assert "--max-model-len" in cmd
    assert "2048" in cmd


def test_build_vllm_command_adds_lora_flags_when_adapter_given(svc: VllmService, tmp_path: Path) -> None:
    opts = VllmModelOptions.model_construct(extra_args={})
    adapter_path = tmp_path / "adapter"

    cmd = svc._build_vllm_command(  # pyright: ignore[reportPrivateUsage]
        docker_model_path=tmp_path / "model",
        model_id="pirate-bot",
        opts=opts,
        quantization=None,
        gpu_memory_utilization=None,
        user_model_length=None,
        use_gpu=True,
        docker_adapter_path=adapter_path,
        lora_rank=16,
    )

    assert "--enable-lora" in cmd
    assert "--lora-modules" in cmd
    assert f"pirate-bot={adapter_path}" in cmd
    assert "--max-lora-rank" in cmd
    assert "16" in cmd


def test_build_vllm_command_adds_lora_flags_without_rank_when_adapter_given_but_no_rank(svc: VllmService, tmp_path: Path) -> None:
    opts = VllmModelOptions.model_construct(extra_args={})
    adapter_path = tmp_path / "adapter"

    cmd = svc._build_vllm_command(  # pyright: ignore[reportPrivateUsage]
        docker_model_path=tmp_path / "model",
        model_id="pirate-bot",
        opts=opts,
        quantization=None,
        gpu_memory_utilization=None,
        user_model_length=None,
        use_gpu=True,
        docker_adapter_path=adapter_path,
        lora_rank=None,
    )

    assert "--enable-lora" in cmd
    assert "--lora-modules" in cmd
    assert "--max-lora-rank" not in cmd


def test_build_vllm_command_omits_lora_flags_when_no_adapter(svc: VllmService, tmp_path: Path) -> None:
    opts = VllmModelOptions.model_construct(extra_args={})

    cmd = svc._build_vllm_command(  # pyright: ignore[reportPrivateUsage]
        docker_model_path=tmp_path / "model",
        model_id="google/test",
        opts=opts,
        quantization=None,
        gpu_memory_utilization=None,
        user_model_length=None,
        use_gpu=True,
    )

    assert "--enable-lora" not in cmd
    assert "--lora-modules" not in cmd
    assert "--max-lora-rank" not in cmd


@pytest.mark.asyncio
async def test_download_model_raises_500_when_no_success_packet(svc: VllmService, deps: dict[str, Any], tmp_path: Path) -> None:
    model = VllmModel(hf_id="google/test", size="100MB")
    stream = MagicMock()

    async def mock_download(*args: object, **kwargs: object):  # type: ignore[misc]
        yield DownloadedPacket(downloaded_bytes_size=1024)

    deps["model_downloader"].download = mock_download

    with (
        patch.object(svc, "_get_working_dir", return_value=tmp_path),  # pyright: ignore[reportPrivateUsage]
        pytest.raises(HTTPException) as exc_info,
    ):
        await svc._download_model(stream, model, tmp_path)  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 500


@pytest.mark.asyncio
async def test_download_model_or_set_progress_raises_500_after_break_with_no_path(svc: VllmService, tmp_path: Path) -> None:
    existing_stream: Stream[StreamChunk] = Stream()  # type: ignore[type-arg]
    finish_chunk: StreamChunk = {"type": "finish", "status": "ok"}  # type: ignore[assignment]
    existing_stream.emit(finish_chunk)
    existing_stream.close()
    model = VllmModel(hf_id="google/test", size="1GB")
    svc.models_download_progress["google/test"] = existing_stream  # type: ignore[assignment]
    output_stream = MagicMock()

    with pytest.raises(HTTPException) as exc_info:
        await svc._download_model_or_set_progress(output_stream, "google/test", model, tmp_path)  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 500


def test_get_size_no_cpu_without_avx512(deps: dict[str, Any]) -> None:
    deps["hardware"].cpu.avx512 = False
    nvidia = MagicMock(spec=NvidiaGpuInfo)
    deps["hardware"].gpus = [nvidia]
    svc = VllmService(**deps)

    sizes = svc.get_size()

    assert "cpu" not in sizes
    assert "gpu" in sizes


@pytest.mark.asyncio
async def test_uninstall_instance_does_nothing_when_not_installed(svc: VllmService, deps: dict[str, Any]) -> None:
    svc.instances_info["default"].installed = None

    with patch.object(svc, "_uninstall_model", new_callable=AsyncMock) as mock_uninstall:  # pyright: ignore[reportPrivateUsage]
        await svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert mock_uninstall.call_count == 0
    assert svc.instances_info["default"].installed is None


@pytest.mark.asyncio
async def test_uninstall_instance_skips_model_installed_in_other_instance(svc: VllmService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    installed.models["m1"] = _make_model_installed_info("m1")
    installed.models["m2"] = _make_model_installed_info("m2")
    svc.instances_info["default"].installed = installed
    deps["docker_service"].uninstall_docker = AsyncMock()

    def mock_is_in_other(instance: str, model_id: str) -> bool:
        return model_id == "m1"

    with (
        patch.object(svc, "is_model_installed_in_other_instance", side_effect=mock_is_in_other),
        patch.object(svc, "_uninstall_model", new_callable=AsyncMock) as mock_uninstall,  # pyright: ignore[reportPrivateUsage]
    ):
        await svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert mock_uninstall.call_count == 1
    called_ids = [call.args[1] for call in mock_uninstall.call_args_list]
    assert "m1" not in called_ids
    assert "m2" in called_ids


@pytest.mark.asyncio
async def test_download_model_handles_pre_download_packet_without_file_size(svc: VllmService, deps: dict[str, Any], tmp_path: Path) -> None:
    model = VllmModel(hf_id="google/test", size="100MB")
    stream = MagicMock()
    local_path = tmp_path / "model-files"
    local_path.mkdir()

    async def mock_download(*args: object, **kwargs: object):  # type: ignore[misc]
        yield PreDownloadPacket(file_bytes_size=None)  # pyright: ignore[reportArgumentType]
        yield SuccessDownloadPacket(local_path=local_path, filename="model-files")

    deps["model_downloader"].download = mock_download
    result = await svc._download_model(stream, model, tmp_path)  # pyright: ignore[reportPrivateUsage]

    assert result == local_path


@pytest.mark.asyncio
async def test_download_model_ignores_zero_bytes_downloaded_packet(svc: VllmService, deps: dict[str, Any], tmp_path: Path) -> None:
    model = VllmModel(hf_id="google/test", size="100MB")
    stream = MagicMock()
    local_path = tmp_path / "model-files"
    local_path.mkdir()

    async def mock_download(*args: object, **kwargs: object):  # type: ignore[misc]
        yield DownloadedPacket(downloaded_bytes_size=0)
        yield SuccessDownloadPacket(local_path=local_path, filename="model-files")

    deps["model_downloader"].download = mock_download
    result = await svc._download_model(stream, model, tmp_path)  # pyright: ignore[reportPrivateUsage]

    assert result == local_path


@pytest.mark.asyncio
async def test_download_model_or_set_progress_emits_chunk_with_empty_data(svc: VllmService, tmp_path: Path) -> None:
    existing_stream: Stream[StreamChunk] = Stream()  # type: ignore[type-arg]
    chunk_empty_data = StreamChunkProgress(type="progress", stage="download", value=0.3, data={})
    chunk_with_data = StreamChunkProgress(
        type="progress",
        stage="download",
        value=1.0,
        data={"local_model_path": str(tmp_path / "model")},
    )
    existing_stream.emit(chunk_empty_data)
    existing_stream.emit(chunk_with_data)
    existing_stream.close()

    model_id = "google/test"
    model = VllmModel(hf_id=model_id, size="1GB")
    svc.models_download_progress[model_id] = existing_stream  # type: ignore[assignment]
    output_stream = MagicMock()

    result_path = await svc._download_model_or_set_progress(output_stream, model_id, model, tmp_path)  # pyright: ignore[reportPrivateUsage]

    assert result_path == tmp_path / "model"
    assert output_stream.emit.call_count == 2


@pytest.mark.asyncio
async def test_install_model_docker_failure_no_decrement_when_model_gpu_utilization_zero(
    svc: VllmService, deps: dict[str, Any], tmp_path: Path
) -> None:
    nvidia = MagicMock(spec=NvidiaGpuInfo)
    deps["hardware"].gpus = [nvidia]
    svc2 = VllmService(**deps)
    installed = _make_installed_info(hardware=True)
    svc2.instances_info["default"].installed = installed
    deps["docker_service"].install_and_run_docker = AsyncMock(side_effect=RuntimeError("docker failed"))
    deps["docker_service"].get_docker_subnet.return_value = None
    deps["docker_service"].get_docker_container_name.return_value = "container"
    deps["docker_service"].stop_docker = AsyncMock()

    model_id = "zero-gpu-model"
    svc2.models["default"][model_id] = VllmModel(hf_id=model_id, size="1GB", gpu_memory_utilization=0.0)

    with (
        patch.object(svc2, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc2, "get_specified_hardware_parts", return_value=[nvidia]),
        patch.object(svc2, "is_given_hardware_support_gpu", return_value=True),
    ):
        promise = await svc2._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(RuntimeError):
            await promise.wait()

    assert svc2.gpu_memory_utilization >= 0


@pytest.mark.asyncio
async def test_install_model_docker_failure_no_floor_when_utilization_stays_positive(
    svc: VllmService, deps: dict[str, Any], tmp_path: Path
) -> None:
    nvidia = MagicMock(spec=NvidiaGpuInfo)
    deps["hardware"].gpus = [nvidia]
    svc2 = VllmService(**deps)
    svc2.gpu_memory_utilization = 0.5
    installed = _make_installed_info(hardware=True)
    svc2.instances_info["default"].installed = installed
    deps["docker_service"].install_and_run_docker = AsyncMock(side_effect=RuntimeError("docker failed"))
    deps["docker_service"].get_docker_subnet.return_value = None
    deps["docker_service"].get_docker_container_name.return_value = "container"
    deps["docker_service"].stop_docker = AsyncMock()

    model_id = "small-gpu-model"
    svc2.models["default"][model_id] = VllmModel(hf_id=model_id, size="1GB", gpu_memory_utilization=0.3)

    with (
        patch.object(svc2, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc2, "get_specified_hardware_parts", return_value=[nvidia]),
        patch.object(svc2, "is_given_hardware_support_gpu", return_value=True),
    ):
        promise = await svc2._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(RuntimeError):
            await promise.wait()

    assert svc2.gpu_memory_utilization == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_uninstall_model_purge_with_none_model_path_removes_downloaded_entry(svc: VllmService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    installed.models["test-model"] = _make_model_installed_info("test-model")
    svc.instances_info["default"].installed = installed
    svc.models_downloaded["test-model"] = DownloadedInfo(model_path=None)
    deps["docker_service"].uninstall_docker = AsyncMock()

    await svc._uninstall_model("default", "test-model", UninstallModelIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert "test-model" not in svc.models_downloaded


@pytest.mark.asyncio
async def test_resolve_custom_model_size_returns_formatted_size(svc: VllmService, deps: dict[str, Any]) -> None:
    with patch("server.services.vllm_service.fetch_huggingface_model_size", new=AsyncMock(return_value="4.0 GB")):
        result = await svc._resolve_custom_model_size({"hf_id": "google/gemma-3-270m-it"})  # pyright: ignore[reportPrivateUsage]

    assert result == "4.0 GB"


@pytest.mark.asyncio
async def test_resolve_custom_model_size_returns_none_on_exception(svc: VllmService, deps: dict[str, Any]) -> None:
    with patch("server.services.vllm_service.fetch_huggingface_model_size", new=AsyncMock(side_effect=Exception("fail"))):
        result = await svc._resolve_custom_model_size({"hf_id": "google/gemma"})  # pyright: ignore[reportPrivateUsage]

    assert result is None


@pytest.mark.asyncio
@patch("server.services.vllm_service.fetch_huggingface_model_size", new_callable=AsyncMock)
async def test_resolve_custom_model_size_passes_hf_token(
    mock_fetch_huggingface_model_size: AsyncMock, svc: VllmService, deps: dict[str, Any]
) -> None:
    deps["config"].hugging_face_token.get_secret_value.return_value = "hf_test_token"
    mock_fetch_huggingface_model_size.return_value = "4.0 GB"

    await svc._resolve_custom_model_size({"hf_id": "org/private-model", "revision": "main"})  # pyright: ignore[reportPrivateUsage]

    assert mock_fetch_huggingface_model_size.call_count == 1
    assert mock_fetch_huggingface_model_size.call_args == call("org/private-model", "main", token="hf_test_token")


@pytest.mark.asyncio
async def test_validate_custom_model_raises_when_neither_hf_id_nor_adapter_id(svc: VllmService, deps: dict[str, Any]) -> None:
    with (
        patch("server.services.vllm_service.validate_model_compatibility", new=AsyncMock()) as mock_validate,
        pytest.raises(HTTPException) as exc_info,
    ):
        await svc._validate_custom_model({})  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400
    mock_validate.assert_not_called()


@pytest.mark.asyncio
async def test_validate_custom_model_resolves_hf_id_and_rank_from_adapter_registry(svc: VllmService, deps: dict[str, Any]) -> None:
    deps["services_manager"].services["finetune"].get_installed_base_url.return_value = "http://df-finetune:8333"
    deps["config"].adapter_registry_secret.get_secret_value.return_value = "secret-token"
    mock_session_cls = _mock_adapter_registry_session_cls(
        200, {"base_model": "Qwen/Qwen3-8B", "formats": ["safetensors"], "lora": {"rank": 16}}
    )
    spec: dict[str, Any] = {"adapter_id": "pirate-llama", "skip_validation": True}

    with (
        patch("server.services.vllm_service.aiohttp.ClientSession", mock_session_cls),
        patch("server.services.vllm_service.fetch_model_metadata", new=AsyncMock(return_value=None)),
    ):
        await svc._validate_custom_model(spec)  # pyright: ignore[reportPrivateUsage]

    assert spec["hf_id"] == "Qwen/Qwen3-8B"
    assert spec["lora_rank"] == 16
    mock_session_cls.return_value.get.assert_called_once_with("http://df-finetune:8333/v1/adapters/pirate-llama")


@pytest.mark.asyncio
async def test_validate_custom_model_returns_when_adapter_resolution_leaves_hf_id_unset(svc: VllmService) -> None:
    spec: dict[str, Any] = {"adapter_id": "pirate-llama"}

    with patch.object(svc, "_resolve_adapter_metadata", new=AsyncMock()):
        await svc._validate_custom_model(spec)  # pyright: ignore[reportPrivateUsage]

    assert spec == {"adapter_id": "pirate-llama"}


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [aiohttp.ClientError("connection refused"), TimeoutError()])
async def test_resolve_adapter_metadata_raises_on_client_error(svc: VllmService, deps: dict[str, Any], error: Exception) -> None:
    deps["services_manager"].services["finetune"].get_installed_base_url.return_value = "http://df-finetune:8333"
    mock_session_cls = MagicMock()
    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.get = MagicMock(side_effect=error)
    mock_session_cls.return_value = mock_session

    with (
        patch("server.services.vllm_service.aiohttp.ClientSession", mock_session_cls),
        pytest.raises(HTTPException) as exc_info,
    ):
        await svc._resolve_adapter_metadata({}, "pirate-llama")  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_resolve_adapter_metadata_raises_when_finetune_not_installed(svc: VllmService, deps: dict[str, Any]) -> None:
    deps["services_manager"].services["finetune"].get_installed_base_url.return_value = None

    with pytest.raises(HTTPException) as exc_info:
        await svc._resolve_adapter_metadata({}, "pirate-llama")  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_resolve_adapter_metadata_raises_on_404(svc: VllmService, deps: dict[str, Any]) -> None:
    deps["services_manager"].services["finetune"].get_installed_base_url.return_value = "http://df-finetune:8333"
    mock_session_cls = _mock_adapter_registry_session_cls(404)

    with (
        patch("server.services.vllm_service.aiohttp.ClientSession", mock_session_cls),
        pytest.raises(HTTPException) as exc_info,
    ):
        await svc._resolve_adapter_metadata({}, "missing-adapter")  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_resolve_adapter_metadata_raises_when_no_safetensors_format(svc: VllmService, deps: dict[str, Any]) -> None:
    deps["services_manager"].services["finetune"].get_installed_base_url.return_value = "http://df-finetune:8333"
    mock_session_cls = _mock_adapter_registry_session_cls(200, {"base_model": "Qwen/Qwen3-8B", "formats": ["gguf"], "lora": {"rank": 16}})

    with (
        patch("server.services.vllm_service.aiohttp.ClientSession", mock_session_cls),
        pytest.raises(HTTPException) as exc_info,
    ):
        await svc._resolve_adapter_metadata({}, "gguf-only-adapter")  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"formats": ["safetensors"], "lora": {}},
        {"base_model": "Qwen/Qwen3-8B", "formats": ["safetensors"], "lora": {"rank": "16"}},
        {"base_model": "Qwen/Qwen3-8B", "formats": ["safetensors"], "lora": {"rank": True}},
        {"base_model": "Qwen/Qwen3-8B", "formats": ["safetensors"], "lora": {"rank": 0}},
    ],
)
async def test_resolve_adapter_metadata_raises_when_missing_base_model_or_invalid_rank(
    svc: VllmService, deps: dict[str, Any], payload: dict[str, Any]
) -> None:
    deps["services_manager"].services["finetune"].get_installed_base_url.return_value = "http://df-finetune:8333"
    mock_session_cls = _mock_adapter_registry_session_cls(200, payload)

    with (
        patch("server.services.vllm_service.aiohttp.ClientSession", mock_session_cls),
        pytest.raises(HTTPException) as exc_info,
    ):
        await svc._resolve_adapter_metadata({}, "incomplete-adapter")  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_validate_custom_model_does_not_call_adapter_registry_when_adapter_id_absent(svc: VllmService, deps: dict[str, Any]) -> None:
    mock_session_cls = _mock_adapter_registry_session_cls(200, {})

    with (
        patch("server.services.vllm_service.aiohttp.ClientSession", mock_session_cls),
        patch("server.services.vllm_service.validate_model_compatibility", new=AsyncMock(return_value=None)),
    ):
        await svc._validate_custom_model({"hf_id": "org/model"})  # pyright: ignore[reportPrivateUsage]

    mock_session_cls.return_value.get.assert_not_called()


@pytest.mark.parametrize("skip_validation", [True, "true", "True", "1", "yes"])
@pytest.mark.asyncio
async def test_validate_custom_model_skips_heuristics_when_skip_validation_flag_set(
    svc: VllmService, deps: dict[str, Any], skip_validation: object
) -> None:
    with (
        patch("server.services.vllm_service.validate_model_compatibility", new=AsyncMock()) as mock_validate,
        patch("server.services.vllm_service.fetch_model_metadata", new=AsyncMock(return_value=None)) as mock_fetch_metadata,
    ):
        await svc._validate_custom_model(  # pyright: ignore[reportPrivateUsage]
            {"hf_id": "org/model", "skip_validation": skip_validation}
        )

    mock_validate.assert_not_called()
    mock_fetch_metadata.assert_called_once()


@pytest.mark.parametrize("skip_validation", [False, "false", "False", "0", "no", "No", "off", "Off", "", None])
@pytest.mark.asyncio
async def test_validate_custom_model_runs_when_skip_validation_flag_unset(
    svc: VllmService, deps: dict[str, Any], skip_validation: object
) -> None:
    spec: dict[str, Any] = {"hf_id": "org/model"}
    if skip_validation is not None:
        spec["skip_validation"] = skip_validation

    with patch("server.services.vllm_service.validate_model_compatibility", new=AsyncMock(return_value=None)) as mock_validate:
        await svc._validate_custom_model(spec)  # pyright: ignore[reportPrivateUsage]

    mock_validate.assert_called_once()


@pytest.mark.asyncio
async def test_validate_custom_model_passes_hf_id_revision_and_model_type(svc: VllmService, deps: dict[str, Any]) -> None:
    with patch("server.services.vllm_service.validate_model_compatibility", new=AsyncMock(return_value=None)) as mock_validate:
        await svc._validate_custom_model(  # pyright: ignore[reportPrivateUsage]
            {"hf_id": "org/model", "revision": "main", "model_type": "reranker"}
        )

    args = mock_validate.call_args.args
    assert args[1:] == ("org/model", "main", "reranker")


@pytest.mark.asyncio
async def test_validate_custom_model_defaults_model_type_to_llm(svc: VllmService, deps: dict[str, Any]) -> None:
    with patch("server.services.vllm_service.validate_model_compatibility", new=AsyncMock(return_value=None)) as mock_validate:
        await svc._validate_custom_model({"hf_id": "org/model"})  # pyright: ignore[reportPrivateUsage]

    args = mock_validate.call_args.args
    assert args[1:] == ("org/model", None, "llm")


@pytest.mark.asyncio
async def test_validate_custom_model_defaults_empty_model_type_to_llm(svc: VllmService, deps: dict[str, Any]) -> None:
    with patch("server.services.vllm_service.validate_model_compatibility", new=AsyncMock(return_value=None)) as mock_validate:
        await svc._validate_custom_model({"hf_id": "org/model", "model_type": ""})  # pyright: ignore[reportPrivateUsage]

    args = mock_validate.call_args.args
    assert args[1:] == ("org/model", None, "llm")


@pytest.mark.asyncio
async def test_validate_custom_model_raises_400_on_incompatible_model(svc: VllmService, deps: dict[str, Any]) -> None:
    with (
        patch(
            "server.services.vllm_service.validate_model_compatibility",
            new=AsyncMock(side_effect=ModelIncompatibleError("org/model is a GGUF repo")),
        ),
        pytest.raises(HTTPException) as exc_info,
    ):
        await svc._validate_custom_model({"hf_id": "org/model"})  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400
    assert "GGUF" in exc_info.value.detail


@pytest.mark.parametrize(
    ("token", "expected_authenticated"),
    [
        ("hf_test_token", True),
        ("", False),
    ],
    ids=("with_token", "without_token"),
)
@pytest.mark.asyncio
@patch("server.services.vllm_service.validate_model_compatibility", new_callable=AsyncMock)
async def test_validate_custom_model_passes_authenticated_to_compatibility_check(
    mock_validate_model_compatibility: AsyncMock, svc: VllmService, deps: dict[str, Any], token: str, expected_authenticated: bool
) -> None:
    deps["config"].hugging_face_token.get_secret_value.return_value = token
    mock_validate_model_compatibility.return_value = None

    await svc._validate_custom_model({"hf_id": "org/model", "revision": "v2"})  # pyright: ignore[reportPrivateUsage]

    assert mock_validate_model_compatibility.call_count == 1
    assert mock_validate_model_compatibility.call_args == call(ANY, "org/model", "v2", "llm", authenticated=expected_authenticated)


@pytest.mark.asyncio
@patch("server.services.vllm_service.validate_model_compatibility", new_callable=AsyncMock)
async def test_validate_custom_model_raises_400_with_access_error_message(
    mock_validate_model_compatibility: AsyncMock, svc: VllmService, deps: dict[str, Any]
) -> None:
    mock_validate_model_compatibility.side_effect = HfRepoAccessError("org/gated-model", None, HfErrorKind.GATED, authenticated=False)

    with pytest.raises(HTTPException) as exc_info:
        await svc._validate_custom_model({"hf_id": "org/gated-model"})  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400
    assert "org/gated-model is gated" in exc_info.value.detail


@pytest.mark.asyncio
@patch("server.services.vllm_service.validate_model_compatibility", new_callable=AsyncMock)
@patch("server.services.vllm_service.fetch_model_metadata", new_callable=AsyncMock)
async def test_validate_custom_model_skip_validation_checks_repo_access_with_token(
    mock_fetch_model_metadata: AsyncMock, mock_validate_model_compatibility: AsyncMock, svc: VllmService, deps: dict[str, Any]
) -> None:
    deps["config"].hugging_face_token.get_secret_value.return_value = "hf_test_token"
    mock_fetch_model_metadata.return_value = None

    await svc._validate_custom_model(  # pyright: ignore[reportPrivateUsage]
        {"hf_id": "org/model", "revision": "v2", "skip_validation": True}
    )

    assert mock_validate_model_compatibility.call_count == 0
    assert mock_fetch_model_metadata.call_count == 1
    assert mock_fetch_model_metadata.call_args == call(ANY, "org/model", "v2", authenticated=True)


@pytest.mark.parametrize(
    "kind",
    [HfErrorKind.NOT_FOUND, HfErrorKind.REVISION_NOT_FOUND, HfErrorKind.NO_ACCESS, HfErrorKind.GATED],
)
@pytest.mark.asyncio
@patch("server.services.vllm_service.fetch_model_metadata", new_callable=AsyncMock)
async def test_validate_custom_model_skip_validation_still_raises_400_on_inaccessible_repo(
    mock_fetch_model_metadata: AsyncMock, svc: VllmService, deps: dict[str, Any], kind: HfErrorKind
) -> None:
    error = HfRepoAccessError("org/model", "v2", kind, authenticated=False)
    mock_fetch_model_metadata.side_effect = error

    with pytest.raises(HTTPException) as exc_info:
        await svc._validate_custom_model(  # pyright: ignore[reportPrivateUsage]
            {"hf_id": "org/model", "revision": "v2", "skip_validation": True}
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == str(error)


@pytest.mark.asyncio
@patch("server.services.vllm_service.fetch_model_metadata", new_callable=AsyncMock)
async def test_validate_custom_model_skip_validation_fills_size_from_metadata(
    mock_fetch_model_metadata: AsyncMock, svc: VllmService, deps: dict[str, Any]
) -> None:
    mock_fetch_model_metadata.return_value = {"siblings": [{"rfilename": "model.safetensors", "size": 4 * 1024**3}]}
    spec: dict[str, Any] = {"hf_id": "org/model", "skip_validation": True}

    await svc._validate_custom_model(spec)  # pyright: ignore[reportPrivateUsage]

    assert spec["size"] == "4.0 GB"


@pytest.mark.parametrize(
    ("previous", "spec"),
    [
        ({"hf_id": "org/model", "quantization": ""}, {"hf_id": "org/model", "quantization": "fp8"}),
        ({"hf_id": "org/model", "revision": "v2"}, {"hf_id": "org/model", "revision": "v2", "extra_args": {"--dtype": "half"}}),
        ({"hf_id": "org/model"}, {"hf_id": "org/model", "revision": "", "model_type": "llm", "quantization": "fp8"}),
        ({"hf_id": "org/model", "skip_validation": True}, {"hf_id": "org/model", "skip_validation": False}),
    ],
    ids=("quantization", "extra_args_with_revision", "absent_vs_default_fields", "skip_validation_toggled"),
)
@pytest.mark.asyncio
@patch("server.services.vllm_service.validate_model_compatibility", new_callable=AsyncMock)
@patch("server.services.vllm_service.fetch_model_metadata", new_callable=AsyncMock)
async def test_validate_custom_model_skips_huggingface_on_edit_keeping_repo(
    mock_fetch_model_metadata: AsyncMock,
    mock_validate_model_compatibility: AsyncMock,
    svc: VllmService,
    deps: dict[str, Any],
    previous: dict[str, Any],
    spec: dict[str, Any],
) -> None:
    await svc._validate_custom_model(spec, previous=previous)  # pyright: ignore[reportPrivateUsage]

    assert mock_fetch_model_metadata.call_count == 0
    assert mock_validate_model_compatibility.call_count == 0


@pytest.mark.parametrize(
    ("previous", "spec"),
    [
        ({"hf_id": "org/model"}, {"hf_id": "org/other-model"}),
        ({"hf_id": "org/model"}, {"hf_id": "org/model", "revision": "v2"}),
        ({"hf_id": "org/model", "model_type": "llm"}, {"hf_id": "org/model", "model_type": "embedding"}),
    ],
    ids=("hf_id", "revision", "model_type"),
)
@pytest.mark.asyncio
@patch("server.services.vllm_service.validate_model_compatibility", new_callable=AsyncMock)
async def test_validate_custom_model_validates_on_edit_changing_repo(
    mock_validate_model_compatibility: AsyncMock,
    svc: VllmService,
    deps: dict[str, Any],
    previous: dict[str, Any],
    spec: dict[str, Any],
) -> None:
    mock_validate_model_compatibility.side_effect = HfRepoAccessError(spec["hf_id"], None, HfErrorKind.GATED, authenticated=False)

    with pytest.raises(HTTPException) as exc_info:
        await svc._validate_custom_model(spec, previous=previous)  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400
    assert mock_validate_model_compatibility.call_count == 1


@pytest.mark.asyncio
@patch("server.services.vllm_service.fetch_model_metadata", new_callable=AsyncMock)
async def test_validate_custom_model_skip_validation_checks_access_on_edit_changing_repo(
    mock_fetch_model_metadata: AsyncMock, svc: VllmService, deps: dict[str, Any]
) -> None:
    mock_fetch_model_metadata.return_value = None

    await svc._validate_custom_model(  # pyright: ignore[reportPrivateUsage]
        {"hf_id": "org/other-model", "skip_validation": True}, previous={"hf_id": "org/model", "skip_validation": True}
    )

    assert mock_fetch_model_metadata.call_count == 1


@pytest.mark.asyncio
@patch("server.services.vllm_service.validate_model_compatibility", new_callable=AsyncMock)
async def test_validate_custom_model_skips_huggingface_on_edit_keeping_adapter_base_model(
    mock_validate_model_compatibility: AsyncMock, svc: VllmService, deps: dict[str, Any]
) -> None:
    previous = {"adapter_id": "pirate-llama", "hf_id": "org/base-model", "lora_rank": 8}
    spec: dict[str, Any] = {"adapter_id": "pirate-llama", "quantization": "fp8"}

    async def _resolve(spec: dict[str, Any], adapter_id: str) -> None:
        spec["hf_id"] = "org/base-model"
        spec["lora_rank"] = 8

    with patch.object(svc, "_resolve_adapter_metadata", new=AsyncMock(side_effect=_resolve)):
        await svc._validate_custom_model(spec, previous=previous)  # pyright: ignore[reportPrivateUsage]

    assert mock_validate_model_compatibility.call_count == 0
    assert spec["hf_id"] == "org/base-model"


@pytest.mark.asyncio
async def test_validate_custom_model_sends_configured_hf_token_as_bearer_header(svc: VllmService, deps: dict[str, Any]) -> None:
    deps["config"].hugging_face_token.get_secret_value.return_value = "hf_test_token"
    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session_cls = MagicMock(return_value=mock_session)

    with (
        patch("server.services.vllm_service.aiohttp.ClientSession", mock_session_cls),
        patch("server.services.vllm_service.validate_model_compatibility", new=AsyncMock(return_value=None)),
    ):
        await svc._validate_custom_model({"hf_id": "org/model"})  # pyright: ignore[reportPrivateUsage]

    assert mock_session_cls.call_args.kwargs["headers"] == {"Authorization": "Bearer hf_test_token"}


@pytest.mark.asyncio
async def test_validate_custom_model_sends_no_auth_header_when_token_unset(svc: VllmService, deps: dict[str, Any]) -> None:
    deps["config"].hugging_face_token.get_secret_value.return_value = ""
    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session_cls = MagicMock(return_value=mock_session)

    with (
        patch("server.services.vllm_service.aiohttp.ClientSession", mock_session_cls),
        patch("server.services.vllm_service.validate_model_compatibility", new=AsyncMock(return_value=None)),
    ):
        await svc._validate_custom_model({"hf_id": "org/model"})  # pyright: ignore[reportPrivateUsage]

    assert mock_session_cls.call_args.kwargs["headers"] == {}


@pytest.mark.asyncio
async def test_validate_custom_model_fills_size_from_fetched_data(svc: VllmService, deps: dict[str, Any]) -> None:
    fetched_data = {"siblings": [{"rfilename": "model.safetensors", "size": 4 * 1024**3}]}
    spec: dict[str, Any] = {"hf_id": "org/model"}

    with patch("server.services.vllm_service.validate_model_compatibility", new=AsyncMock(return_value=fetched_data)):
        await svc._validate_custom_model(spec)  # pyright: ignore[reportPrivateUsage]

    assert spec["size"] == "4.0 GB"


@pytest.mark.asyncio
async def test_validate_custom_model_does_not_overwrite_explicit_size(svc: VllmService, deps: dict[str, Any]) -> None:
    fetched_data = {"siblings": [{"rfilename": "model.safetensors", "size": 4 * 1024**3}]}
    spec: dict[str, Any] = {"hf_id": "org/model", "size": "1GB"}

    with patch("server.services.vllm_service.validate_model_compatibility", new=AsyncMock(return_value=fetched_data)):
        await svc._validate_custom_model(spec)  # pyright: ignore[reportPrivateUsage]

    assert spec["size"] == "1GB"


@pytest.mark.asyncio
async def test_validate_custom_model_leaves_size_unset_when_siblings_report_no_size(svc: VllmService, deps: dict[str, Any]) -> None:
    fetched_data = {"siblings": [{"rfilename": "README.md"}]}
    spec: dict[str, Any] = {"hf_id": "org/model"}

    with patch("server.services.vllm_service.validate_model_compatibility", new=AsyncMock(return_value=fetched_data)):
        await svc._validate_custom_model(spec)  # pyright: ignore[reportPrivateUsage]

    assert "size" not in spec


@pytest.mark.asyncio
async def test_validate_custom_model_leaves_size_unset_when_fetch_failed(svc: VllmService, deps: dict[str, Any]) -> None:
    spec: dict[str, Any] = {"hf_id": "org/model"}

    with patch("server.services.vllm_service.validate_model_compatibility", new=AsyncMock(return_value=None)):
        await svc._validate_custom_model(spec)  # pyright: ignore[reportPrivateUsage]

    assert "size" not in spec


@pytest.mark.asyncio
async def test_add_custom_model_size_fill_skips_redundant_size_resolution_call(svc: VllmService, deps: dict[str, Any]) -> None:
    # The whole point of filling `size` inside `_validate_custom_model` is to spare
    # `add_custom_model`'s `if not spec.get("size")` guard a second, near-identical HF request via
    # `_resolve_custom_model_size` - verify that guard is actually skipped end-to-end.
    fetched_data = {"siblings": [{"rfilename": "model.safetensors", "size": 4 * 1024**3}]}
    deps["service_provider"].save_service_config = AsyncMock()
    with (
        patch("server.services.vllm_service.validate_model_compatibility", new=AsyncMock(return_value=fetched_data)),
        patch.object(svc, "_resolve_custom_model_size", new=AsyncMock()) as mock_resolve_size,
    ):
        await svc.add_custom_model("default", AddCustomModelIn(spec={"id": "my-model", "hf_id": "org/model"}))

    mock_resolve_size.assert_not_called()


@pytest.mark.asyncio
async def test_get_docker_logs_cache_hit(svc: VllmService) -> None:
    svc._log_cache["my-container"] = (time.monotonic(), "cached logs", 8.0)  # pyright: ignore[reportPrivateUsage]

    result = await svc._get_docker_logs("my-container")  # pyright: ignore[reportPrivateUsage]

    assert result == "cached logs"


@pytest.mark.asyncio
async def test_get_docker_logs_fetches_and_caches(svc: VllmService) -> None:
    mock_result = MagicMock()
    mock_result.exit_code = 0
    mock_result.stdout = "stdout logs"
    mock_result.stderr = "stderr logs"

    with patch("server.services.base2_service.Utils.run_command", new_callable=AsyncMock, return_value=mock_result):
        result = await svc._get_docker_logs("my-container")  # pyright: ignore[reportPrivateUsage]

    assert result == "stdout logsstderr logs"
    assert "my-container" in svc._log_cache  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_get_docker_logs_caches_failed_command_briefly(svc: VllmService) -> None:
    """A failed read is still cached, but only for the short failed-read TTL - long enough to
    avoid spawning a fresh `docker logs` subprocess on every poll tick of a stuck container,
    short enough to pick up the real startup log soon after it becomes available."""
    mock_result = MagicMock()
    mock_result.exit_code = 1
    mock_result.stdout = ""
    mock_result.stderr = "Error: No such container"

    with patch("server.services.base2_service.Utils.run_command", new_callable=AsyncMock, return_value=mock_result):
        result = await svc._get_docker_logs("my-container")  # pyright: ignore[reportPrivateUsage]

    assert result == "Error: No such container"
    cached = svc._log_cache["my-container"]  # pyright: ignore[reportPrivateUsage]
    assert cached[1] == "Error: No such container"
    assert cached[2] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_get_docker_logs_caches_empty_output_briefly(svc: VllmService) -> None:
    mock_result = MagicMock()
    mock_result.exit_code = 0
    mock_result.stdout = ""
    mock_result.stderr = ""

    with patch("server.services.base2_service.Utils.run_command", new_callable=AsyncMock, return_value=mock_result):
        result = await svc._get_docker_logs("my-container")  # pyright: ignore[reportPrivateUsage]

    assert result == ""
    cached = svc._log_cache["my-container"]  # pyright: ignore[reportPrivateUsage]
    assert cached[1] == ""
    assert cached[2] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Total (incl. non-KV cache overhead): 8.50 GiB\n", 8.5),
        ("Model weights: 6.00 GiB\nKV Cache: 2.00 GiB\n", 8.0),
        ("model weights take 5.00 GiB\nKV cache memory: 1.50 GiB\n", 6.5),
        ("Model weights: 4.00 GiB\n", 4.0),
        ("KV cache: 3.00 GiB\n", 3.0),
        ("no useful info here", None),
    ],
)
def test_parse_vllm_vram_gb(raw: str, expected: float | None) -> None:
    result = VllmService._parse_vllm_vram_gb(raw)  # pyright: ignore[reportPrivateUsage]

    assert result == expected


@pytest.mark.asyncio
async def test_get_vram_from_logs_returns_parsed_value(svc: VllmService) -> None:
    installed = _make_installed_info()
    model_info = _make_model_installed_info("test-model")
    model_info.docker.container_name = "vllm-container"
    installed.models["test-model"] = model_info
    svc.instances_info["default"].installed = installed

    with patch.object(svc, "_get_docker_logs", new_callable=AsyncMock, return_value="Total (incl. non-KV cache overhead): 7.00 GiB"):  # pyright: ignore[reportPrivateUsage]
        result = await svc._get_vram_from_logs("default", "test-model")  # pyright: ignore[reportPrivateUsage]

    assert result == 7.0


@pytest.mark.asyncio
async def test_get_vram_from_logs_returns_none_when_no_container(svc: VllmService) -> None:
    installed = _make_installed_info()
    model_info = _make_model_installed_info("test-model")
    model_info.docker.container_name = ""
    installed.models["test-model"] = model_info
    svc.instances_info["default"].installed = installed

    result = await svc._get_vram_from_logs("default", "test-model")  # pyright: ignore[reportPrivateUsage]

    assert result is None


@pytest.mark.asyncio
async def test_get_vram_from_logs_returns_none_on_docker_logs_error(svc: VllmService) -> None:
    installed = _make_installed_info()
    model_info = _make_model_installed_info("test-model")
    model_info.docker.container_name = "vllm-container"
    installed.models["test-model"] = model_info
    svc.instances_info["default"].installed = installed

    with patch.object(svc, "_get_docker_logs", new_callable=AsyncMock, side_effect=Exception("boom")):  # pyright: ignore[reportPrivateUsage]
        result = await svc._get_vram_from_logs("default", "test-model")  # pyright: ignore[reportPrivateUsage]

    assert result is None


def test_get_vram_estimate_returns_value(svc: VllmService) -> None:
    svc.hardware.total_vram_gb = 24.0  # pyright: ignore[reportAttributeAccessIssue]
    model_info = _make_model_installed_info("test-model", gpu_memory_utilization=0.5)

    result = svc._get_vram_estimate(model_info)  # pyright: ignore[reportPrivateUsage]

    assert result == 12.0


def test_get_vram_estimate_returns_none_when_no_utilization(svc: VllmService) -> None:
    model_info = _make_model_installed_info("test-model", gpu_memory_utilization=None)

    result = svc._get_vram_estimate(model_info)  # pyright: ignore[reportPrivateUsage]

    assert result is None


def test_get_vram_estimate_returns_none_when_total_vram_zero(svc: VllmService) -> None:
    svc.hardware.total_vram_gb = 0.0  # pyright: ignore[reportAttributeAccessIssue]
    model_info = _make_model_installed_info("test-model", gpu_memory_utilization=0.5)

    result = svc._get_vram_estimate(model_info)  # pyright: ignore[reportPrivateUsage]

    assert result is None


@pytest.mark.asyncio
async def test_get_cached_vram_estimate_cache_hit(svc: VllmService) -> None:
    svc._vram_cache[("default", "test-model")] = 4.0  # pyright: ignore[reportPrivateUsage]

    result = await svc._get_cached_vram_estimate("default", "test-model", is_loaded=True)  # pyright: ignore[reportPrivateUsage]

    assert result == 4.0


@pytest.mark.asyncio
async def test_get_cached_vram_estimate_from_logs_non_none(svc: VllmService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    with patch.object(svc, "_get_vram_from_logs", new_callable=AsyncMock, return_value=5.5):  # pyright: ignore[reportPrivateUsage]
        result = await svc._get_cached_vram_estimate("default", "test-model", is_loaded=True)  # pyright: ignore[reportPrivateUsage]

    assert result == 5.5
    assert svc._vram_cache[("default", "test-model")] == 5.5  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_get_cached_vram_estimate_model_not_in_installed(svc: VllmService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed  # no models in installed

    with patch.object(svc, "_get_vram_from_logs", new_callable=AsyncMock, return_value=None):  # pyright: ignore[reportPrivateUsage]
        result = await svc._get_cached_vram_estimate("default", "missing-model", is_loaded=True)  # pyright: ignore[reportPrivateUsage]

    assert result is None
    assert ("default", "missing-model") not in svc._vram_cache  # pyright: ignore[reportPrivateUsage]


def test_release_gpu_utilization_with_none_does_not_change_value(svc: VllmService) -> None:
    svc.gpu_memory_utilization = 0.5
    svc._release_gpu_utilization(None)  # pyright: ignore[reportPrivateUsage]
    assert svc.gpu_memory_utilization == pytest.approx(0.5)


def test_release_gpu_utilization_floors_at_zero_when_result_would_be_negative(svc: VllmService) -> None:
    svc.gpu_memory_utilization = 0.1
    svc._release_gpu_utilization(0.5)  # pyright: ignore[reportPrivateUsage]
    assert svc.gpu_memory_utilization == 0


@pytest.mark.asyncio
async def test_install_model_releases_gpu_on_cancelled_error(svc: VllmService, deps: dict[str, Any], tmp_path: Path) -> None:
    nvidia = MagicMock(spec=NvidiaGpuInfo)
    deps["hardware"].gpus = [nvidia]
    svc2 = VllmService(**deps)
    installed = _make_installed_info(hardware=True)
    svc2.instances_info["default"].installed = installed
    deps["docker_service"].install_and_run_docker = AsyncMock(side_effect=asyncio.CancelledError())
    deps["docker_service"].get_docker_subnet.return_value = None
    deps["docker_service"].get_docker_container_name.return_value = "container"
    deps["docker_service"].stop_docker = AsyncMock()

    model_id = "cancel-model"
    svc2.models["default"][model_id] = VllmModel(hf_id=model_id, size="1GB", gpu_memory_utilization=0.5)

    with (
        patch.object(svc2, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc2, "get_specified_hardware_parts", return_value=[nvidia]),
        patch.object(svc2, "is_given_hardware_support_gpu", return_value=True),
    ):
        promise = await svc2._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(asyncio.CancelledError):
            await promise.wait()

    assert svc2.gpu_memory_utilization == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_install_model_registration_failure_rolls_back_model(svc: VllmService, deps: dict[str, Any], tmp_path: Path) -> None:
    installed = _setup_install_mocks(svc, deps)
    deps["docker_service"].stop_docker = AsyncMock()
    model_id = next(iter(svc.models["default"]))
    deps["endpoint_registry"].register_chat_completion_as_proxy.side_effect = RuntimeError("registry down")

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.vllm_service.get_model_dir_context_window", new_callable=AsyncMock, return_value=4096),
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(RuntimeError):
            await promise.wait()

    assert model_id not in installed.models
    deps["docker_service"].stop_docker.assert_called_once()


@pytest.mark.asyncio
async def test_install_model_releases_gpu_when_option_parsing_fails(svc: VllmService, deps: dict[str, Any]) -> None:
    nvidia = MagicMock(spec=NvidiaGpuInfo)
    deps["hardware"].gpus = [nvidia]
    svc2 = VllmService(**deps)
    svc2.instances_info["default"].installed = _make_installed_info(hardware=True)
    model_id = "parse-fail-model"
    svc2.models["default"][model_id] = VllmModel(hf_id=model_id, size="1GB", gpu_memory_utilization=0.5)

    with (
        patch.object(svc2, "is_given_hardware_support_gpu", return_value=True),
        patch.object(svc2, "_get_quantization", new_callable=AsyncMock, side_effect=RuntimeError("bad quantization")),  # pyright: ignore[reportPrivateUsage]
        pytest.raises(RuntimeError),
    ):
        await svc2._install_model("default", model_id, InstallModelIn(spec={"gpu_memory_utilization": 0.5}))  # pyright: ignore[reportPrivateUsage]

    assert svc2.gpu_memory_utilization == 0.0
    assert ("default", model_id) not in svc2._installing  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_install_model_releases_gpu_on_cancelled_error_before_docker_start(svc: VllmService, deps: dict[str, Any]) -> None:
    nvidia = MagicMock(spec=NvidiaGpuInfo)
    deps["hardware"].gpus = [nvidia]
    svc2 = VllmService(**deps)
    svc2.instances_info["default"].installed = _make_installed_info(hardware=True)
    model_id = "cancel-before-docker-model"
    svc2.models["default"][model_id] = VllmModel(hf_id=model_id, size="1GB", gpu_memory_utilization=0.5)

    with (
        patch.object(svc2, "is_given_hardware_support_gpu", return_value=True),
        patch.object(svc2, "_get_quantization", new_callable=AsyncMock, side_effect=asyncio.CancelledError()),  # pyright: ignore[reportPrivateUsage]
        pytest.raises(asyncio.CancelledError),
    ):
        await svc2._install_model("default", model_id, InstallModelIn(spec={"gpu_memory_utilization": 0.5}))  # pyright: ignore[reportPrivateUsage]

    assert svc2.gpu_memory_utilization == 0.0
    assert ("default", model_id) not in svc2._installing  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_install_model_releases_gpu_and_stops_container_when_post_start_fails(
    svc: VllmService, deps: dict[str, Any], tmp_path: Path
) -> None:
    nvidia = MagicMock(spec=NvidiaGpuInfo)
    deps["hardware"].gpus = [nvidia]
    svc2 = VllmService(**deps)
    installed = _make_installed_info(hardware=True)
    svc2.instances_info["default"].installed = installed
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=(8000, True, False))
    deps["docker_service"].get_docker_subnet.return_value = None
    deps["docker_service"].get_docker_container_name.return_value = "container"
    deps["docker_service"].get_container_host.return_value = "localhost"
    deps["docker_service"].get_container_port.return_value = 8000
    deps["docker_service"].stop_docker = AsyncMock()
    model_id = "post-start-model"
    svc2.models["default"][model_id] = VllmModel(hf_id=model_id, size="1GB", gpu_memory_utilization=0.5)

    with (
        patch.object(svc2, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch(
            "server.services.vllm_service.get_model_dir_context_window",
            new_callable=AsyncMock,
            side_effect=RuntimeError("context read failed"),
        ),
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc2, "get_specified_hardware_parts", return_value=[nvidia]),
        patch.object(svc2, "is_given_hardware_support_gpu", return_value=True),
    ):
        promise = await svc2._install_model("default", model_id, InstallModelIn(spec={"gpu_memory_utilization": 0.5}))  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(RuntimeError):
            await promise.wait()

    assert svc2.gpu_memory_utilization == 0.0
    assert model_id not in installed.models
    deps["docker_service"].stop_docker.assert_called_once()


@pytest.mark.asyncio
async def test_install_model_does_not_stop_adopted_container_when_post_start_fails(
    svc: VllmService, deps: dict[str, Any], tmp_path: Path
) -> None:
    """An adopted container pre-dates this install attempt, so a later failure must not tear it down (DFINFRA-297)."""
    nvidia = MagicMock(spec=NvidiaGpuInfo)
    deps["hardware"].gpus = [nvidia]
    svc2 = VllmService(**deps)
    installed = _make_installed_info(hardware=True)
    svc2.instances_info["default"].installed = installed
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=(8000, False, True))
    deps["docker_service"].get_docker_subnet.return_value = None
    deps["docker_service"].get_docker_container_name.return_value = "container"
    deps["docker_service"].get_container_host.return_value = "localhost"
    deps["docker_service"].get_container_port.return_value = 8000
    deps["docker_service"].stop_docker = AsyncMock()
    model_id = "adopted-post-start-model"
    svc2.models["default"][model_id] = VllmModel(hf_id=model_id, size="1GB", gpu_memory_utilization=0.5)

    with (
        patch.object(svc2, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch(
            "server.services.vllm_service.get_model_dir_context_window",
            new_callable=AsyncMock,
            side_effect=RuntimeError("context read failed"),
        ),
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc2, "get_specified_hardware_parts", return_value=[nvidia]),
        patch.object(svc2, "is_given_hardware_support_gpu", return_value=True),
    ):
        promise = await svc2._install_model("default", model_id, InstallModelIn(spec={"gpu_memory_utilization": 0.5}))  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(RuntimeError):
            await promise.wait()

    assert svc2.gpu_memory_utilization == 0.0
    assert model_id not in installed.models
    deps["docker_service"].stop_docker.assert_not_called()


@pytest.mark.asyncio
async def test_install_model_releases_gpu_without_stopping_container_when_download_fails(svc: VllmService, deps: dict[str, Any]) -> None:
    nvidia = MagicMock(spec=NvidiaGpuInfo)
    deps["hardware"].gpus = [nvidia]
    svc2 = VllmService(**deps)
    svc2.instances_info["default"].installed = _make_installed_info(hardware=True)
    deps["docker_service"].stop_docker = AsyncMock()
    model_id = "download-fail-model"
    svc2.models["default"][model_id] = VllmModel(hf_id=model_id, size="1GB", gpu_memory_utilization=0.5)

    with (
        patch.object(
            svc2,
            "_download_model_or_set_progress",
            new_callable=AsyncMock,
            side_effect=RuntimeError("download failed"),  # pyright: ignore[reportPrivateUsage]
        ),
        patch.object(svc2, "get_specified_hardware_parts", return_value=[nvidia]),
        patch.object(svc2, "is_given_hardware_support_gpu", return_value=True),
    ):
        promise = await svc2._install_model("default", model_id, InstallModelIn(spec={"gpu_memory_utilization": 0.5}))  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(RuntimeError):
            await promise.wait()

    assert svc2.gpu_memory_utilization == 0.0
    deps["docker_service"].stop_docker.assert_not_called()


def test_get_image_with_version_overrides_tag_gpu(svc: VllmService) -> None:
    image = svc._get_image(gpu=True, image_version="v0.9.0")  # pyright: ignore[reportPrivateUsage]
    base = _const.images["gpu"].name.split(":")[0]
    assert image.name == f"{base}:v0.9.0"
    assert image.size == _const.images["gpu"].size


def test_get_image_with_version_overrides_tag_cpu(svc: VllmService) -> None:
    image = svc._get_image(gpu=False, image_version="v0.9.0")  # pyright: ignore[reportPrivateUsage]
    base = _const.images["cpu"].name.split(":")[0]
    assert image.name == f"{base}:v0.9.0"
    assert image.size == _const.images["cpu"].size


@pytest.mark.asyncio
async def test_get_docker_tags_returns_filtered_tags(svc: VllmService) -> None:
    mock_client = AsyncMock()
    mock_client.get_tags = AsyncMock(return_value=["v0.9.0-cu130", "v0.9.0-cpu"])
    with (
        patch("server.services.vllm_service.registry_for", return_value=mock_client),
        patch("server.services.vllm_service.image_without_registry_prefix", return_value="vllm/vllm-openai"),
    ):
        tags = await svc.get_docker_tags("GPU")
    assert "v0.9.0-cu130" in tags
    assert "v0.9.0-cpu" not in tags


@pytest.mark.asyncio
async def test_get_docker_tags_cpu_hardware_returns_tags_unfiltered(svc: VllmService) -> None:
    mock_client = AsyncMock()
    mock_client.get_tags = AsyncMock(return_value=["v0.19.0", "v0.19.1"])
    cpu_base = _const.images["cpu"].name.split(":")[0]
    with (
        patch("server.services.vllm_service.registry_for", return_value=mock_client),
        patch("server.services.vllm_service.image_without_registry_prefix", return_value="cpu/vllm-openai") as mock_img,
    ):
        tags = await svc.get_docker_tags("cpu")
    assert tags == ["v0.19.0", "v0.19.1"]
    mock_img.assert_called_once_with(cpu_base)


def test_get_default_docker_tag_gpu(svc: VllmService) -> None:
    assert svc.get_default_docker_tag("GPU") == _const.images["gpu"].name.split(":")[1]


def test_get_default_docker_tag_cpu(svc: VllmService) -> None:
    assert svc.get_default_docker_tag("cpu") == _const.images["cpu"].name.split(":")[1]


def test_get_default_docker_tag_defaults_to_gpu_when_no_hardware(svc: VllmService) -> None:
    assert svc.get_default_docker_tag(None) == _const.images["gpu"].name.split(":")[1]


def test_get_docker_image_repo_gpu(svc: VllmService) -> None:
    assert svc.get_docker_image_repo("GPU") == _const.images["gpu"].name.split(":")[0]


def test_get_docker_image_repo_cpu(svc: VllmService) -> None:
    assert svc.get_docker_image_repo("cpu") == _const.images["cpu"].name.split(":")[0]


def test_get_docker_image_repo_defaults_to_gpu_when_no_hardware(svc: VllmService) -> None:
    assert svc.get_docker_image_repo(None) == _const.images["gpu"].name.split(":")[0]


@pytest.mark.asyncio
async def test_install_model_registration_failure_skips_rollback_when_already_removed(
    svc: VllmService, deps: dict[str, Any], tmp_path: Path
) -> None:
    installed = _setup_install_mocks(svc, deps)
    deps["docker_service"].stop_docker = AsyncMock()
    model_id = next(iter(svc.models["default"]))

    def side_effect(*args: object, **kwargs: object) -> None:
        installed.models.pop(model_id, None)
        raise RuntimeError("registry down")

    deps["endpoint_registry"].register_chat_completion_as_proxy.side_effect = side_effect

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.vllm_service.get_model_dir_context_window", new_callable=AsyncMock, return_value=4096),
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(RuntimeError):
            await promise.wait()

    assert model_id not in installed.models


@pytest.mark.asyncio
async def test_install_model_pre_func_exception_discards_installing_key(svc: VllmService, deps: dict[str, Any]) -> None:
    _setup_install_mocks(svc, deps)
    model_id = next(iter(svc.models["default"]))

    with (
        patch.object(svc, "_get_quantization", new_callable=AsyncMock, side_effect=RuntimeError("quantization failed")),  # pyright: ignore[reportPrivateUsage]
        pytest.raises(RuntimeError),
    ):
        await svc._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]

    assert ("default", model_id) not in svc._installing  # pyright: ignore[reportPrivateUsage]


def _status(
    exists: bool = True,
    state: str = "running",
    health: str = "healthy",
    restart_count: int = 0,
    exit_code: int = 0,
    oom_killed: bool = False,
) -> ContainerStatus:
    return ContainerStatus(
        exists=exists, state=state, health=health, restart_count=restart_count, exit_code=exit_code, oom_killed=oom_killed
    )


@pytest.mark.asyncio
async def test_reconcile_instance_models_running_healthy_does_not_release(svc: VllmService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    model_info = _make_model_installed_info(model_id="m1", gpu_memory_utilization=0.5)
    installed.models["m1"] = model_info
    svc.gpu_memory_utilization = 0.5
    deps["docker_service"].get_container_status = AsyncMock(return_value=_status(state="running", health="healthy"))

    await svc._reconcile_instance_models("default", installed)  # pyright: ignore[reportPrivateUsage]

    assert "m1" in installed.models
    assert svc.gpu_memory_utilization == 0.5
    deps["endpoint_registry"].unregister_chat_completion.assert_not_called()


@pytest.mark.asyncio
async def test_reconcile_instance_models_recovers_within_threshold_resets_counter(svc: VllmService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    model_info = _make_model_installed_info(model_id="m1", gpu_memory_utilization=0.5)
    installed.models["m1"] = model_info
    svc.gpu_memory_utilization = 0.5
    deps["docker_service"].get_container_status = AsyncMock(return_value=_status(state="exited", health=""))

    await svc._reconcile_instance_models("default", installed)  # pyright: ignore[reportPrivateUsage]
    await svc._reconcile_instance_models("default", installed)  # pyright: ignore[reportPrivateUsage]

    assert svc._crash_poll_state[("default", "m1")] == 2  # pyright: ignore[reportPrivateUsage]

    deps["docker_service"].get_container_status = AsyncMock(return_value=_status(state="running", health="healthy"))
    await svc._reconcile_instance_models("default", installed)  # pyright: ignore[reportPrivateUsage]

    assert ("default", "m1") not in svc._crash_poll_state  # pyright: ignore[reportPrivateUsage]
    assert "m1" in installed.models
    assert svc.gpu_memory_utilization == 0.5


@pytest.mark.asyncio
async def test_reconcile_instance_models_releases_after_threshold_bad_polls(svc: VllmService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    model_info = _make_model_installed_info(model_id="m1", registration_id="reg-1", gpu_memory_utilization=0.5, model_type="llm")
    installed.models["m1"] = model_info
    svc.gpu_memory_utilization = 0.5
    deps["docker_service"].get_container_status = AsyncMock(return_value=_status(exists=False, state="", health=""))
    deps["docker_service"].uninstall_docker = AsyncMock()

    with patch.object(svc, "_save", new_callable=AsyncMock) as mock_save:
        for _ in range(3):
            await svc._reconcile_instance_models("default", installed)  # pyright: ignore[reportPrivateUsage]

    assert "m1" not in installed.models
    assert svc.gpu_memory_utilization == pytest.approx(0.0)
    assert ("default", "m1") not in svc._crash_poll_state  # pyright: ignore[reportPrivateUsage]
    deps["endpoint_registry"].unregister_chat_completion.assert_called_once_with("m1", "reg-1")
    deps["docker_service"].uninstall_docker.assert_called_once_with(model_info.docker)
    mock_save.assert_awaited_once()


@pytest.mark.asyncio
async def test_reconcile_instance_models_keeps_bookkeeping_when_teardown_fails_and_retries(svc: VllmService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    model_info = _make_model_installed_info(model_id="m1", registration_id="reg-1", gpu_memory_utilization=0.5)
    installed.models["m1"] = model_info
    svc.gpu_memory_utilization = 0.5
    deps["docker_service"].get_container_status = AsyncMock(return_value=_status(exists=False, state="", health=""))
    deps["docker_service"].uninstall_docker = AsyncMock(side_effect=RuntimeError("docker daemon busy"))

    for _ in range(3):
        await svc._reconcile_instance_models("default", installed)  # pyright: ignore[reportPrivateUsage]

    # Teardown kept failing: bookkeeping stays untouched, nothing released yet.
    assert "m1" in installed.models
    assert svc.gpu_memory_utilization == pytest.approx(0.5)
    deps["endpoint_registry"].unregister_chat_completion.assert_not_called()

    # Teardown finally succeeds on a later tick, without needing 3 more bad polls first.
    deps["docker_service"].uninstall_docker = AsyncMock()
    with patch.object(svc, "_save", new_callable=AsyncMock) as mock_save:
        await svc._reconcile_instance_models("default", installed)  # pyright: ignore[reportPrivateUsage]

    assert "m1" not in installed.models
    assert svc.gpu_memory_utilization == pytest.approx(0.0)
    deps["endpoint_registry"].unregister_chat_completion.assert_called_once_with("m1", "reg-1")
    mock_save.assert_awaited_once()


@pytest.mark.asyncio
@patch("server.services.base2_service.logger")
@patch("server.services.base2_service.capture_exception")
async def test_reconcile_instance_models_reports_dead_container_once(
    mock_capture_exception: MagicMock, mock_logger: MagicMock, svc: VllmService, deps: dict[str, Any]
) -> None:
    installed = _make_installed_info()
    installed.models["m1"] = _make_model_installed_info(model_id="m1", registration_id="reg-1", gpu_memory_utilization=0.5)
    svc.gpu_memory_utilization = 0.5
    status = _status(state="exited", health="unhealthy", restart_count=3, exit_code=137, oom_killed=True)
    deps["docker_service"].get_container_status = AsyncMock(return_value=status)
    deps["docker_service"].uninstall_docker = AsyncMock()

    with patch.object(svc, "_save", new_callable=AsyncMock):
        for _ in range(4):
            await svc._reconcile_instance_models("default", installed)  # pyright: ignore[reportPrivateUsage]

    assert mock_capture_exception.call_count == 1
    failure = mock_capture_exception.call_args.args[0]
    assert isinstance(failure, UpstreamError)
    assert str(failure) == "vllm model container died"
    error_id = mock_capture_exception.call_args.kwargs["error_id"]
    assert mock_capture_exception.call_args == call(
        failure,
        error_id=error_id,
        tags={
            "service": "vllm",
            "model": "m1",
            "container_state": "exited",
            "container_health": "unhealthy",
            "restart_count": "3",
            "exit_code": "137",
            "oom_killed": "true",
        },
        fingerprint=["container-died", "vllm"],
    )
    assert any(error_id in logged.args[0] for logged in mock_logger.warning.call_args_list)


@pytest.mark.asyncio
@patch("server.services.base2_service.capture_exception")
async def test_reconcile_instance_models_reports_missing_container_as_missing(
    mock_capture_exception: MagicMock, svc: VllmService, deps: dict[str, Any]
) -> None:
    installed = _make_installed_info()
    installed.models["m1"] = _make_model_installed_info(model_id="m1", gpu_memory_utilization=0.5)
    svc.gpu_memory_utilization = 0.5
    deps["docker_service"].get_container_status = AsyncMock(return_value=_status(exists=False, state="", health=""))
    deps["docker_service"].uninstall_docker = AsyncMock()

    with patch.object(svc, "_save", new_callable=AsyncMock):
        for _ in range(3):
            await svc._reconcile_instance_models("default", installed)  # pyright: ignore[reportPrivateUsage]

    tags = mock_capture_exception.call_args.kwargs["tags"]
    assert tags == {"service": "vllm", "model": "m1", "container_state": "missing"}


@pytest.mark.asyncio
@patch("server.services.base2_service.capture_exception")
async def test_reconcile_instance_models_does_not_report_transient_restart(
    mock_capture_exception: MagicMock, svc: VllmService, deps: dict[str, Any]
) -> None:
    installed = _make_installed_info()
    installed.models["m1"] = _make_model_installed_info(model_id="m1", gpu_memory_utilization=0.5)
    svc.gpu_memory_utilization = 0.5
    deps["docker_service"].get_container_status = AsyncMock(return_value=_status(state="restarting", health=""))

    await svc._reconcile_instance_models("default", installed)  # pyright: ignore[reportPrivateUsage]
    await svc._reconcile_instance_models("default", installed)  # pyright: ignore[reportPrivateUsage]
    deps["docker_service"].get_container_status = AsyncMock(return_value=_status(state="running", health="healthy"))
    await svc._reconcile_instance_models("default", installed)  # pyright: ignore[reportPrivateUsage]

    assert "m1" in installed.models
    assert mock_capture_exception.call_count == 0


@pytest.mark.asyncio
@patch("server.services.base2_service.capture_exception")
async def test_reconcile_instance_models_reports_once_at_threshold_even_if_teardown_fails(
    mock_capture_exception: MagicMock, svc: VllmService, deps: dict[str, Any]
) -> None:
    installed = _make_installed_info()
    installed.models["m1"] = _make_model_installed_info(model_id="m1", gpu_memory_utilization=0.5)
    svc.gpu_memory_utilization = 0.5
    deps["docker_service"].get_container_status = AsyncMock(return_value=_status(exists=False, state="", health=""))
    deps["docker_service"].uninstall_docker = AsyncMock(side_effect=RuntimeError("docker daemon busy"))

    for _ in range(2):
        await svc._reconcile_instance_models("default", installed)  # pyright: ignore[reportPrivateUsage]
    reported_before_threshold = mock_capture_exception.call_count
    for _ in range(3):
        await svc._reconcile_instance_models("default", installed)  # pyright: ignore[reportPrivateUsage]
    reported_while_teardown_fails = mock_capture_exception.call_count
    deps["docker_service"].uninstall_docker = AsyncMock()
    with patch.object(svc, "_save", new_callable=AsyncMock):
        await svc._reconcile_instance_models("default", installed)  # pyright: ignore[reportPrivateUsage]

    assert reported_before_threshold == 0
    assert reported_while_teardown_fails == 1
    assert mock_capture_exception.call_count == 1
    assert "m1" not in installed.models


@pytest.mark.asyncio
async def test_reconcile_instance_models_recovers_while_teardown_was_failing(svc: VllmService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    model_info = _make_model_installed_info(model_id="m1", gpu_memory_utilization=0.5)
    installed.models["m1"] = model_info
    svc.gpu_memory_utilization = 0.5
    deps["docker_service"].get_container_status = AsyncMock(return_value=_status(exists=False, state="", health=""))
    deps["docker_service"].uninstall_docker = AsyncMock(side_effect=RuntimeError("docker daemon busy"))

    for _ in range(3):
        await svc._reconcile_instance_models("default", installed)  # pyright: ignore[reportPrivateUsage]

    # Container comes back healthy on its own (e.g. manually restarted) before teardown ever succeeded.
    deps["docker_service"].get_container_status = AsyncMock(return_value=_status(state="running", health="healthy"))
    await svc._reconcile_instance_models("default", installed)  # pyright: ignore[reportPrivateUsage]

    assert "m1" in installed.models
    assert svc.gpu_memory_utilization == pytest.approx(0.5)
    assert ("default", "m1") not in svc._crash_poll_state  # pyright: ignore[reportPrivateUsage]
    deps["endpoint_registry"].unregister_chat_completion.assert_not_called()


@pytest.mark.asyncio
async def test_reconcile_instance_models_unhealthy_running_container_counts_as_bad(svc: VllmService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    model_info = _make_model_installed_info(model_id="m1", gpu_memory_utilization=0.5)
    installed.models["m1"] = model_info
    svc.gpu_memory_utilization = 0.5
    deps["docker_service"].get_container_status = AsyncMock(return_value=_status(state="running", health="unhealthy"))
    deps["docker_service"].uninstall_docker = AsyncMock()

    with patch.object(svc, "_save", new_callable=AsyncMock):
        for _ in range(3):
            await svc._reconcile_instance_models("default", installed)  # pyright: ignore[reportPrivateUsage]

    assert "m1" not in installed.models
    assert svc.gpu_memory_utilization == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_reconcile_instance_models_skips_models_currently_installing(svc: VllmService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    model_info = _make_model_installed_info(model_id="m1", gpu_memory_utilization=0.5)
    installed.models["m1"] = model_info
    svc._installing.add(("default", "m1"))  # pyright: ignore[reportPrivateUsage]
    deps["docker_service"].get_container_status = AsyncMock(return_value=_status(exists=False, state="", health=""))

    await svc._reconcile_instance_models("default", installed)  # pyright: ignore[reportPrivateUsage]

    deps["docker_service"].get_container_status.assert_not_called()
    assert "m1" in installed.models


@pytest.mark.asyncio
async def test_reconcile_instance_models_skips_model_uninstalled_concurrently(svc: VllmService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    model_info = _make_model_installed_info(model_id="m1", gpu_memory_utilization=0.5)
    installed.models["m1"] = model_info
    svc.gpu_memory_utilization = 0.5
    deps["docker_service"].get_container_status = AsyncMock(return_value=_status(exists=False, state="", health=""))

    await svc._reconcile_instance_models("default", installed)  # pyright: ignore[reportPrivateUsage]
    await svc._reconcile_instance_models("default", installed)  # pyright: ignore[reportPrivateUsage]
    assert svc._crash_poll_state[("default", "m1")] == 2  # pyright: ignore[reportPrivateUsage]

    async def _status_and_uninstall_concurrently(_container_name: str) -> ContainerStatus:
        # Simulate a concurrent manual uninstall completing while this (3rd) poll is in flight.
        del installed.models["m1"]
        return _status(exists=False, state="", health="")

    deps["docker_service"].get_container_status = AsyncMock(side_effect=_status_and_uninstall_concurrently)

    with patch.object(svc, "_save", new_callable=AsyncMock) as mock_save:
        await svc._reconcile_instance_models("default", installed)  # pyright: ignore[reportPrivateUsage]

    deps["endpoint_registry"].unregister_chat_completion.assert_not_called()
    mock_save.assert_not_awaited()


@pytest.mark.asyncio
async def test_reconcile_instance_models_releases_reranker_model(svc: VllmService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    model_info = _make_model_installed_info(model_id="m1", registration_id="reg-1", gpu_memory_utilization=0.5, model_type="reranker")
    installed.models["m1"] = model_info
    svc.gpu_memory_utilization = 0.5
    deps["docker_service"].get_container_status = AsyncMock(return_value=_status(exists=False, state="", health=""))
    deps["docker_service"].uninstall_docker = AsyncMock()

    with patch.object(svc, "_save", new_callable=AsyncMock):
        for _ in range(3):
            await svc._reconcile_instance_models("default", installed)  # pyright: ignore[reportPrivateUsage]

    deps["endpoint_registry"].unregister_rerank.assert_called_once_with("m1", "reg-1")
    deps["endpoint_registry"].unregister_chat_completion.assert_not_called()


@pytest.mark.asyncio
async def test_start_reconciliation_task_cancels_existing_and_stop_cancels_it(svc: VllmService) -> None:
    svc.instances_info["default"].installed = _make_installed_info()

    svc._start_reconciliation_task("default")  # pyright: ignore[reportPrivateUsage]
    first_task = svc._reconciliation_tasks["default"]  # pyright: ignore[reportPrivateUsage]
    svc._start_reconciliation_task("default")  # pyright: ignore[reportPrivateUsage]
    second_task = svc._reconciliation_tasks["default"]  # pyright: ignore[reportPrivateUsage]

    await asyncio.gather(first_task, return_exceptions=True)
    assert first_task is not second_task
    assert first_task.cancelled()

    svc._stop_reconciliation_task("default")  # pyright: ignore[reportPrivateUsage]
    await asyncio.gather(second_task, return_exceptions=True)
    assert second_task.cancelled()
    assert "default" not in svc._reconciliation_tasks  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_reconciliation_loop_ticks_and_stops_once_instance_uninstalled(svc: VllmService) -> None:
    svc.instances_info["default"].installed = _make_installed_info()
    calls = 0

    async def fake_reconcile(_instance: str, _info: InstalledInfo) -> None:
        nonlocal calls
        calls += 1
        svc.instances_info["default"].installed = None

    with (
        patch("server.services.base2_service.asyncio.sleep", new_callable=AsyncMock),
        patch.object(svc, "_reconcile_instance_models", new=AsyncMock(side_effect=fake_reconcile)),
    ):
        svc._start_reconciliation_task("default")  # pyright: ignore[reportPrivateUsage]
        task = svc._reconciliation_tasks["default"]  # pyright: ignore[reportPrivateUsage]
        await asyncio.wait_for(task, timeout=1)

    assert calls == 1


@pytest.mark.asyncio
async def test_reconciliation_loop_logs_and_continues_on_tick_exception(svc: VllmService) -> None:
    svc.instances_info["default"].installed = _make_installed_info()
    calls = 0

    async def fake_reconcile(_instance: str, _info: InstalledInfo) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("boom")
        svc.instances_info["default"].installed = None

    with (
        patch("server.services.base2_service.asyncio.sleep", new_callable=AsyncMock),
        patch.object(svc, "_reconcile_instance_models", new=AsyncMock(side_effect=fake_reconcile)),
    ):
        svc._start_reconciliation_task("default")  # pyright: ignore[reportPrivateUsage]
        task = svc._reconciliation_tasks["default"]  # pyright: ignore[reportPrivateUsage]
        await asyncio.wait_for(task, timeout=1)

    assert calls == 2


async def _fake_fetch_registry_entries(
    _session: Any,
    _top_by_downloads: int,
    _top_by_likes: int,
    _top_by_trending: int,
    model_type: str,
    log: Any,
    allow_generative_rerankers: bool = False,
) -> tuple[str, list[dict[str, Any]]]:
    if model_type == "llm":
        return "llms", [{"name": "new/model", "size": "2GB"}]
    if model_type == "reranker":
        return "rerankers", []
    if model_type == "stt":
        return "stts", [{"name": "new/whisper", "size": "3GB"}]
    return "embeddings", []


@pytest.fixture(autouse=True)
def _reset_catalog_refresh_guard() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    huggingface_refresh_guard.release()
    yield
    huggingface_refresh_guard.release()


def test_get_catalog_refresh_unavailable_reason_none_when_supported(svc: VllmService) -> None:
    with patch("server.services.vllm_service.model_catalog_refresh.catalog_refresh_supported", True):
        assert svc.get_catalog_refresh_unavailable_reason() is None


def test_get_catalog_refresh_unavailable_reason_set_when_unsupported(svc: VllmService) -> None:
    with patch("server.services.vllm_service.model_catalog_refresh.catalog_refresh_supported", False):
        assert svc.get_catalog_refresh_unavailable_reason() is not None


@pytest.mark.asyncio
async def test_refresh_catalog_writes_registry_and_reloads_const(svc: VllmService, tmp_path: Path) -> None:
    (tmp_path / "static").mkdir()
    (tmp_path / "static" / "vllm-min.json").write_text(json.dumps({"llms": [{"name": "old/model", "size": "1GB"}]}), encoding="utf-8")

    original_models = dict(_const.models)
    try:
        with (
            patch("server.services.vllm_service.model_catalog_refresh.catalog_refresh_supported", True),
            patch("server.services.vllm_service.get_main_dir", return_value=tmp_path),
            patch("server.services.vllm_service.fetch_registry_entries", new=AsyncMock(side_effect=_fake_fetch_registry_entries)),
        ):
            promise = await svc.refresh_catalog()
            result = await promise.wait()

        assert result.added == 2
        assert "new/model" in _const.models
        assert _const.models["new/whisper"].model_type == "stt"
        assert "old/model" not in _const.models
        written = json.loads((tmp_path / "static" / "vllm-min.json").read_text(encoding="utf-8"))
        assert written["llms"] == [{"name": "new/model", "size": "2GB"}]
        assert written["stts"] == [{"name": "new/whisper", "size": "3GB"}]
        assert set(svc.models["default"].keys()) == set(_const.models.keys())
    finally:
        _const.models = original_models


@pytest.mark.asyncio
async def test_refresh_catalog_raises_when_unsupported(svc: VllmService) -> None:
    with (
        patch("server.services.vllm_service.model_catalog_refresh.catalog_refresh_supported", False),
        pytest.raises(HTTPException) as exc_info,
    ):
        await svc.refresh_catalog()
    assert exc_info.value.status_code == 405


@pytest.mark.asyncio
async def test_refresh_catalog_rejects_concurrent_refresh(svc: VllmService) -> None:
    huggingface_refresh_guard.acquire("llamacpp")
    try:
        with (
            patch("server.services.vllm_service.model_catalog_refresh.catalog_refresh_supported", True),
            pytest.raises(HTTPException) as exc_info,
        ):
            await svc.refresh_catalog()
        assert exc_info.value.status_code == 429
    finally:
        huggingface_refresh_guard.release()


@pytest.mark.asyncio
async def test_refresh_catalog_leaves_catalog_unchanged_on_failure(svc: VllmService, tmp_path: Path) -> None:
    (tmp_path / "static").mkdir()
    original_content = json.dumps({"llms": [{"name": "old/model", "size": "1GB"}]})
    (tmp_path / "static" / "vllm-min.json").write_text(original_content, encoding="utf-8")

    original_models = dict(_const.models)
    try:
        with (
            patch("server.services.vllm_service.model_catalog_refresh.catalog_refresh_supported", True),
            patch("server.services.vllm_service.get_main_dir", return_value=tmp_path),
            patch("server.services.vllm_service.fetch_registry_entries", new=AsyncMock(side_effect=RuntimeError("boom"))),
        ):
            promise = await svc.refresh_catalog()
            with pytest.raises(RuntimeError):
                await promise.wait()

        assert (tmp_path / "static" / "vllm-min.json").read_text(encoding="utf-8") == original_content
        assert _const.models == original_models
    finally:
        _const.models = original_models


@pytest.mark.asyncio
async def test_refresh_catalog_raises_502_when_catalog_shrinks_too_much(svc: VllmService, tmp_path: Path) -> None:
    (tmp_path / "static").mkdir()
    original_content = json.dumps({"llms": [{"name": f"old/model-{i}", "size": "1GB"} for i in range(10)]})
    (tmp_path / "static" / "vllm-min.json").write_text(original_content, encoding="utf-8")

    async def _degraded_fetch(
        _session: Any,
        _top_by_downloads: int,
        _top_by_likes: int,
        _top_by_trending: int,
        model_type: str,
        log: Any,
        allow_generative_rerankers: bool = False,
    ) -> tuple[str, list[dict[str, Any]]]:
        if model_type == "llm":
            return "llms", [{"name": "new/model", "size": "2GB"}]
        return ("rerankers" if model_type == "reranker" else "embeddings"), []

    original_models = dict(_const.models)
    try:
        with (
            patch("server.services.vllm_service.model_catalog_refresh.catalog_refresh_supported", True),
            patch("server.services.vllm_service.get_main_dir", return_value=tmp_path),
            patch("server.services.vllm_service.fetch_registry_entries", new=AsyncMock(side_effect=_degraded_fetch)),
        ):
            promise = await svc.refresh_catalog()
            with pytest.raises(HTTPException) as exc_info:
                await promise.wait()
            assert exc_info.value.status_code == 502

        assert (tmp_path / "static" / "vllm-min.json").read_text(encoding="utf-8") == original_content
        assert _const.models == original_models
    finally:
        _const.models = original_models


@pytest.mark.asyncio
async def test_refresh_catalog_restores_installed_model_dropped_from_catalog(svc: VllmService, tmp_path: Path) -> None:
    (tmp_path / "static").mkdir()
    (tmp_path / "static" / "vllm-min.json").write_text(json.dumps({"llms": [{"name": "old/model", "size": "1GB"}]}), encoding="utf-8")

    installed = _make_installed_info()
    installed.models["old/model"] = _make_model_installed_info("old/model")
    svc.instances_info["default"].installed = installed
    svc.instances_info["default"].config = InstanceConfig(
        models=[ModelConfig(model_id="old/model", options=InstallModelIn(spec={}), definition={"hf_id": "old/model", "size": "1GB"})]
    )

    original_models = dict(_const.models)
    try:
        with (
            patch("server.services.vllm_service.model_catalog_refresh.catalog_refresh_supported", True),
            patch("server.services.vllm_service.get_main_dir", return_value=tmp_path),
            patch("server.services.vllm_service.fetch_registry_entries", new=AsyncMock(side_effect=_fake_fetch_registry_entries)),
        ):
            promise = await svc.refresh_catalog()
            await promise.wait()

        assert "old/model" not in _const.models
        assert svc.models["default"]["old/model"] == VllmModel(hf_id="old/model", size="1GB")
    finally:
        _const.models = original_models


@pytest.mark.asyncio
async def test_refresh_catalog_drops_installed_model_with_no_persisted_definition(svc: VllmService, tmp_path: Path) -> None:
    (tmp_path / "static").mkdir()
    (tmp_path / "static" / "vllm-min.json").write_text(json.dumps({"llms": [{"name": "old/model", "size": "1GB"}]}), encoding="utf-8")

    installed = _make_installed_info()
    installed.models["old/model"] = _make_model_installed_info("old/model")
    svc.instances_info["default"].installed = installed

    original_models = dict(_const.models)
    try:
        with (
            patch("server.services.vllm_service.model_catalog_refresh.catalog_refresh_supported", True),
            patch("server.services.vllm_service.get_main_dir", return_value=tmp_path),
            patch("server.services.vllm_service.fetch_registry_entries", new=AsyncMock(side_effect=_fake_fetch_registry_entries)),
        ):
            promise = await svc.refresh_catalog()
            await promise.wait()

        assert "old/model" not in svc.models["default"]
    finally:
        _const.models = original_models


@pytest.mark.asyncio
async def test_sync_models_triggers_refresh_without_awaiting_completion(svc: VllmService) -> None:
    with patch.object(svc, "refresh_catalog", new=AsyncMock()) as mock_refresh:
        await svc.sync_models("default")
    mock_refresh.assert_awaited_once()


@pytest.mark.asyncio
async def test_sync_models_suppresses_unsupported_error(svc: VllmService) -> None:
    with patch("server.services.vllm_service.model_catalog_refresh.catalog_refresh_supported", False):
        await svc.sync_models("default")  # must not raise


@pytest.mark.asyncio
async def test_sync_models_logs_warning_when_background_refresh_fails(svc: VllmService) -> None:
    class _FailingPromise:
        async def wait(self) -> None:
            raise RuntimeError("boom")

    with (
        patch.object(svc, "refresh_catalog", new=AsyncMock(return_value=_FailingPromise())),
        patch("server.services.vllm_service.logger") as mock_logger,
    ):
        await svc.sync_models("default")
        await asyncio.sleep(0)  # let the fire-and-forget task run

    mock_logger.warning.assert_called_once()


@pytest.mark.asyncio
async def test_edit_model_install_options_keeps_legacy_gpu_utilization(svc: VllmService, deps: dict[str, Any], tmp_path: Path) -> None:
    """An edit through the form (which only renders gpu_priority) must not drop a model's legacy absolute utilization."""
    _setup_install_mocks(svc, deps)
    deps["service_provider"].save_service_config = AsyncMock()
    deps["service_provider"].dismiss_warnings_matching = AsyncMock()
    deps["service_provider"].dismiss_warnings_matching_any = AsyncMock()
    deps["docker_service"].uninstall_docker = AsyncMock()
    model_id = next(iter(svc.models["default"]))

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.vllm_service.get_model_dir_context_window", new_callable=AsyncMock, return_value=4096),
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
        patch("server.services.vllm_service.validate_model_compatibility", new=AsyncMock(return_value=None)),
    ):
        install_promise = await svc.install_model("default", model_id, InstallModelIn(spec={"gpu_memory_utilization": 0.3}))
        await install_promise.wait()

        was_installed, promise = await svc.edit_model_install_options("default", model_id, InstallModelIn(spec={"alias": "renamed"}))
        result = await promise.wait()

    assert was_installed is True
    assert result.status == "OK"
    installed = svc.get_instance_installed_info("default").models[model_id]
    assert installed.options.spec == {
        "alias": "renamed",
        "gpu_memory_utilization": 0.3,
        "extra_args": {},
        "extra_envs": {},
        "quantization": "",
    }


def _write_cached_adapter(tmp_path: Path, created: str | None = None) -> Path:
    adapter_dir = tmp_path / "adapter-pirate_llama"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}")
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"old-weights")
    if created is not None:
        (adapter_dir / ".registry_created").write_text(created)
    return adapter_dir


def _mock_adapter_downloader(download_calls: list[str]) -> Callable[..., AsyncIterator[SuccessDownloadPacket]]:
    async def mock_download(  # type: ignore[misc]
        download_url: str, model_dir: Path, temp_dir: Path, filename: str | None = None, revision: str | None = None
    ) -> AsyncIterator[SuccessDownloadPacket]:
        download_calls.append(download_url)
        yield SuccessDownloadPacket(local_path=model_dir / (filename or "file"), filename=filename or "file")

    return mock_download


def _adapter_model(lora_rank: int = 16) -> VllmModel:
    return VllmModel(hf_id="Qwen/Qwen3-8B", size="16GB", adapter_id="pirate-llama", lora_rank=lora_rank)


_ADAPTER_BASE_URL = "http://df-finetune:8333/v1/adapters/pirate-llama/safetensors"


@pytest.mark.asyncio
async def test_download_adapter_uses_on_disk_copy_when_registry_unavailable(svc: VllmService, deps: dict[str, Any], tmp_path: Path) -> None:
    """A reload at startup can run before the Finetune service has loaded - it must not need it."""
    adapter_dir = _write_cached_adapter(tmp_path, created="1700000000")
    deps["services_manager"].services["finetune"].get_installed_base_url.return_value = None
    model = _adapter_model()

    with patch("server.services.vllm_service.AdapterRegistryDownloader") as downloader_cls:
        local_dir, docker_dir = await svc._download_adapter(model, "pirate-llama", tmp_path)  # pyright: ignore[reportPrivateUsage]

    assert local_dir == adapter_dir
    assert docker_dir == Path(svc.hugging_face_cache_path) / "loras" / "pirate_llama"
    assert (adapter_dir / "adapter_model.safetensors").read_bytes() == b"old-weights"
    assert model.lora_rank == 16
    downloader_cls.assert_not_called()


@pytest.mark.asyncio
async def test_download_adapter_keeps_on_disk_copy_when_registry_created_matches(svc: VllmService, tmp_path: Path) -> None:
    _write_cached_adapter(tmp_path, created="1700000000")
    metadata = AdapterMetadata(base_model="Qwen/Qwen3-8B", rank=16, created=1_700_000_000)

    with (
        patch.object(svc, "_fetch_adapter_metadata", new_callable=AsyncMock, return_value=metadata),
        patch("server.services.vllm_service.AdapterRegistryDownloader") as downloader_cls,
    ):
        await svc._download_adapter(_adapter_model(), "pirate-llama", tmp_path)  # pyright: ignore[reportPrivateUsage]

    downloader_cls.assert_not_called()


@pytest.mark.asyncio
async def test_download_adapter_redownloads_all_files_and_refreshes_rank_when_registry_created_differs(
    svc: VllmService, tmp_path: Path
) -> None:
    """An adapter retrained under the same id must not be served from stale weights with a stale rank."""
    adapter_dir = _write_cached_adapter(tmp_path, created="1700000000")
    metadata = AdapterMetadata(base_model="Qwen/Qwen3-8B", rank=64, created=1_800_000_000)
    download_calls: list[str] = []
    model = _adapter_model(lora_rank=16)

    with (
        patch.object(svc, "_fetch_adapter_metadata", new_callable=AsyncMock, return_value=metadata),
        patch("server.services.vllm_service.AdapterRegistryDownloader") as downloader_cls,
    ):
        downloader_cls.return_value.download = _mock_adapter_downloader(download_calls)
        await svc._download_adapter(model, "pirate-llama", tmp_path)  # pyright: ignore[reportPrivateUsage]

    assert download_calls == [f"{_ADAPTER_BASE_URL}/adapter_config.json", f"{_ADAPTER_BASE_URL}/adapter_model.safetensors"]
    # the real downloader skips existing files, so the stale ones must be removed before downloading
    assert not (adapter_dir / "adapter_model.safetensors").exists()
    assert not (adapter_dir / "adapter_config.json").exists()
    assert (adapter_dir / ".registry_created").read_text() == "1800000000"
    assert model.lora_rank == 64


@pytest.mark.asyncio
async def test_download_adapter_redownloads_when_marker_missing(svc: VllmService, tmp_path: Path) -> None:
    adapter_dir = _write_cached_adapter(tmp_path)
    metadata = AdapterMetadata(base_model="Qwen/Qwen3-8B", rank=16, created=1_700_000_000)
    download_calls: list[str] = []

    with (
        patch.object(svc, "_fetch_adapter_metadata", new_callable=AsyncMock, return_value=metadata),
        patch("server.services.vllm_service.AdapterRegistryDownloader") as downloader_cls,
    ):
        downloader_cls.return_value.download = _mock_adapter_downloader(download_calls)
        await svc._download_adapter(_adapter_model(), "pirate-llama", tmp_path)  # pyright: ignore[reportPrivateUsage]

    assert len(download_calls) == 2
    assert (adapter_dir / ".registry_created").read_text() == "1700000000"


@pytest.mark.asyncio
async def test_download_adapter_trusts_on_disk_copy_when_registry_created_unknown(svc: VllmService, tmp_path: Path) -> None:
    adapter_dir = _write_cached_adapter(tmp_path)
    metadata = AdapterMetadata(base_model="Qwen/Qwen3-8B", rank=16, created=None)

    with (
        patch.object(svc, "_fetch_adapter_metadata", new_callable=AsyncMock, return_value=metadata),
        patch("server.services.vllm_service.AdapterRegistryDownloader") as downloader_cls,
    ):
        await svc._download_adapter(_adapter_model(), "pirate-llama", tmp_path)  # pyright: ignore[reportPrivateUsage]

    downloader_cls.assert_not_called()
    assert not (adapter_dir / ".registry_created").exists()


@pytest.mark.asyncio
async def test_download_adapter_fetches_only_missing_files(svc: VllmService, tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter-pirate_llama"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}")
    (adapter_dir / ".registry_created").write_text("1700000000")
    metadata = AdapterMetadata(base_model="Qwen/Qwen3-8B", rank=16, created=1_700_000_000)
    download_calls: list[str] = []

    with (
        patch.object(svc, "_fetch_adapter_metadata", new_callable=AsyncMock, return_value=metadata),
        patch("server.services.vllm_service.AdapterRegistryDownloader") as downloader_cls,
    ):
        downloader_cls.return_value.download = _mock_adapter_downloader(download_calls)
        await svc._download_adapter(_adapter_model(), "pirate-llama", tmp_path)  # pyright: ignore[reportPrivateUsage]

    assert download_calls == [f"{_ADAPTER_BASE_URL}/adapter_model.safetensors"]


@pytest.mark.asyncio
async def test_download_adapter_raises_when_files_missing_and_registry_unavailable(
    svc: VllmService, deps: dict[str, Any], tmp_path: Path
) -> None:
    deps["services_manager"].services["finetune"].get_installed_base_url.return_value = None

    with pytest.raises(HTTPException) as exc_info:
        await svc._download_adapter(_adapter_model(), "pirate-llama", tmp_path)  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_download_adapter_raises_when_base_model_changed(svc: VllmService, tmp_path: Path) -> None:
    _write_cached_adapter(tmp_path, created="1700000000")
    metadata = AdapterMetadata(base_model="meta-llama/Llama-3.1-8B", rank=16, created=1_800_000_000)

    with (
        patch.object(svc, "_fetch_adapter_metadata", new_callable=AsyncMock, return_value=metadata),
        pytest.raises(HTTPException) as exc_info,
    ):
        await svc._download_adapter(_adapter_model(), "pirate-llama", tmp_path)  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400
    assert "meta-llama/Llama-3.1-8B" in exc_info.value.detail


@pytest.mark.asyncio
@pytest.mark.parametrize(("created", "expected"), [(1_700_000_000, 1_700_000_000), (0, None), (None, None), (True, None)])
async def test_fetch_adapter_metadata_normalizes_created(
    svc: VllmService, deps: dict[str, Any], created: object, expected: int | None
) -> None:
    deps["services_manager"].services["finetune"].get_installed_base_url.return_value = "http://df-finetune:8333"
    payload = {"formats": ["safetensors"], "base_model": "Qwen/Qwen3-8B", "lora": {"rank": 16}, "created": created}
    mock_session_cls = _mock_adapter_registry_session_cls(200, payload)

    with patch("server.services.vllm_service.aiohttp.ClientSession", mock_session_cls):
        metadata = await svc._fetch_adapter_metadata("pirate-llama")  # pyright: ignore[reportPrivateUsage]

    assert metadata == AdapterMetadata(base_model="Qwen/Qwen3-8B", rank=16, created=expected)


def test_purge_stale_download_keeps_old_adapter_directory_used_by_another_model(svc: VllmService, tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    shared_adapter_dir = tmp_path / "adapter-shared"
    shared_adapter_dir.mkdir()
    svc.models_downloaded["custom-model"] = DownloadedInfo(model_path=str(model_dir), adapter_path=str(shared_adapter_dir))
    svc.models_downloaded["other-model"] = DownloadedInfo(model_path=str(model_dir), adapter_path=str(shared_adapter_dir))

    svc._purge_stale_download("custom-model", model_dir, tmp_path / "adapter-new")  # pyright: ignore[reportPrivateUsage]

    assert shared_adapter_dir.exists()


def test_purge_stale_download_keeps_old_model_directory_used_by_another_model(svc: VllmService, tmp_path: Path) -> None:
    shared_model_dir = tmp_path / "shared-model"
    shared_model_dir.mkdir()
    svc.models_downloaded["custom-model"] = DownloadedInfo(model_path=str(shared_model_dir))
    svc.models_downloaded["other-model"] = DownloadedInfo(model_path=str(shared_model_dir))

    svc._purge_stale_download("custom-model", tmp_path / "new-model")  # pyright: ignore[reportPrivateUsage]

    assert shared_model_dir.exists()


@pytest.mark.asyncio
async def test_uninstall_model_purge_keeps_directories_used_by_another_model(
    svc: VllmService, deps: dict[str, Any], tmp_path: Path
) -> None:
    """Two adapters on the same base model share its weights; purging one must not break the other."""
    shared_model_dir = tmp_path / "model-dir"
    shared_model_dir.mkdir()
    shared_adapter_dir = tmp_path / "adapter-dir"
    shared_adapter_dir.mkdir()
    svc.instances_info["default"].installed = _make_installed_info()
    svc.models_downloaded["pirate-bot"] = DownloadedInfo(model_path=str(shared_model_dir), adapter_path=str(shared_adapter_dir))
    svc.models_downloaded["pirate-bot-2"] = DownloadedInfo(model_path=str(shared_model_dir), adapter_path=str(shared_adapter_dir))
    deps["docker_service"].uninstall_docker = AsyncMock()

    await svc._uninstall_model("default", "pirate-bot", UninstallModelIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert shared_model_dir.exists()
    assert shared_adapter_dir.exists()
    assert "pirate-bot" not in svc.models_downloaded
    assert "pirate-bot-2" in svc.models_downloaded


@pytest.mark.parametrize(
    ("rank", "expected"),
    [(1, 1), (4, 8), (8, 8), (12, 16), (16, 16), (300, 320), (512, 512), (1024, 1024)],
)
def test_round_up_lora_rank(rank: int, expected: int) -> None:
    assert _round_up_lora_rank(rank) == expected


def test_build_vllm_command_rounds_lora_rank_up_to_vllm_allowed_value(svc: VllmService, tmp_path: Path) -> None:
    cmd = svc._build_vllm_command(  # pyright: ignore[reportPrivateUsage]
        docker_model_path=tmp_path / "model",
        model_id="pirate-bot",
        opts=VllmModelOptions.model_construct(extra_args={}),
        quantization=None,
        gpu_memory_utilization=None,
        user_model_length=None,
        use_gpu=True,
        docker_adapter_path=tmp_path / "adapter",
        lora_rank=12,
    )

    assert cmd[cmd.index("--max-lora-rank") + 1] == "16"


@pytest.mark.asyncio
async def test_resolve_adapter_metadata_raises_when_rank_above_vllm_maximum(svc: VllmService) -> None:
    mock_session_cls = _mock_adapter_registry_session_cls(
        200, {"base_model": "Qwen/Qwen3-8B", "formats": ["safetensors"], "lora": {"rank": 1024}}
    )

    with (
        patch("server.services.vllm_service.aiohttp.ClientSession", mock_session_cls),
        pytest.raises(HTTPException) as exc_info,
    ):
        await svc._resolve_adapter_metadata({}, "huge-adapter")  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400


@pytest.mark.parametrize("utilization", [-0.1, 1.5, float("nan"), float("inf")])
def test_model_options_reject_invalid_legacy_gpu_memory_utilization(utilization: float) -> None:
    with pytest.raises(ValidationError):
        VllmModelOptions(gpu_memory_utilization=utilization)


@pytest.mark.asyncio
async def test_get_default_gpu_memory_utilization_warns_when_stats_unavailable(svc: VllmService, deps: dict[str, Any]) -> None:
    deps["hardware"].get_realtime_stats = AsyncMock(return_value=None)

    with patch("server.services.vllm_service.logger") as mock_logger:
        result = await svc._get_default_gpu_memory_utilization()  # pyright: ignore[reportPrivateUsage]

    assert result == 0.9
    mock_logger.warning.assert_called_once()


def test_get_persisted_model_gpu_memory_utilization_prefers_value_kept_across_an_edit(svc: VllmService) -> None:
    svc._edit_kept_gpu_utilization[("default", "google/test")] = 0.4  # pyright: ignore[reportPrivateUsage]

    assert svc._get_persisted_model_gpu_memory_utilization("default", "google/test", 0.5) == 0.4  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_second_install_is_sized_only_after_the_first_container_is_healthy(deps: dict[str, Any], tmp_path: Path) -> None:
    nvidia = MagicMock(spec=NvidiaGpuInfo)
    deps["hardware"].gpus = [nvidia]
    svc = VllmService(**deps)
    loaded = False
    first_started = asyncio.Event()
    release_first = asyncio.Event()

    async def stats(*_args: Any, **_kwargs: Any) -> GpuStats:
        return GpuStats(total_vram_gb=10.0, used_vram_gb=4.0 if loaded else 0.0, gpus=None)

    async def run_docker(options: Any) -> tuple[int, bool, bool]:
        nonlocal loaded
        if not first_started.is_set():
            first_started.set()
            await release_first.wait()  # the first container is still loading its weights
            loaded = True
        return 8000, True, False

    deps["hardware"].get_realtime_stats = stats
    deps["docker_service"].install_and_run_docker = run_docker
    deps["docker_service"].get_docker_subnet.return_value = None
    deps["docker_service"].get_docker_container_name.return_value = "container"
    svc.instances_info["default"].installed = _make_installed_info(hardware=True)
    first_id, second_id = list(svc.models["default"])[:2]

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_get_quantization", new=AsyncMock(return_value=None)),
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc, "get_specified_hardware_parts", return_value=[nvidia]),
        patch.object(svc, "is_given_hardware_support_gpu", return_value=True),
    ):
        first = await svc._install_model("default", first_id, InstallModelIn(spec={"gpu_memory_utilization": 0.4}))  # pyright: ignore[reportPrivateUsage]
        second = await svc._install_model("default", second_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        first_task = asyncio.create_task(first.wait())
        second_task = asyncio.create_task(second.wait())
        await first_started.wait()
        for _ in range(20):
            await asyncio.sleep(0)  # the second one downloads meanwhile, but must wait for the first before sizing
        assert svc.gpu_memory_utilization == 0.4  # nothing reserved for the second yet: no double counting while loading
        release_first.set()
        await asyncio.gather(first_task, second_task)

    models = svc.instances_info["default"].installed.models  # pyright: ignore[reportOptionalMemberAccess]
    assert models[second_id].gpu_memory_utilization == 0.54  # 0.9 * the 0.6 left once the first model is loaded
    assert svc.gpu_memory_utilization == pytest.approx(0.94)


@pytest.mark.asyncio
async def test_cpu_install_does_not_wait_for_the_gpu_load_lock(deps: dict[str, Any], tmp_path: Path) -> None:
    svc = VllmService(**deps)

    async def run_docker(options: Any) -> tuple[int, bool, bool]:
        return 8000, True, False

    deps["docker_service"].install_and_run_docker = run_docker
    deps["docker_service"].get_docker_subnet.return_value = None
    deps["docker_service"].get_docker_container_name.return_value = "container"
    svc.instances_info["default"].installed = _make_installed_info(hardware=False)
    model_id = next(iter(svc.models["default"]))

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_get_quantization", new=AsyncMock(return_value=None)),
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc, "is_given_hardware_support_gpu", return_value=False),
    ):
        async with deps["hardware"].gpu_load_lock:  # a GPU model is loading right now
            promise = await svc._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
            await asyncio.wait_for(promise.wait(), timeout=5)

    assert model_id in svc.instances_info["default"].installed.models  # pyright: ignore[reportOptionalMemberAccess]


def test_vllm_and_sglang_share_the_gpu_load_lock_of_hardware(deps: dict[str, Any]) -> None:
    assert VllmService(**deps).hardware.gpu_load_lock is deps["hardware"].gpu_load_lock


@pytest.mark.asyncio
async def test_get_gpu_memory_utilization_raises_422_when_default_is_below_minimum(svc: VllmService, deps: dict[str, Any]) -> None:
    svc.gpu_memory_utilization = 0.0
    deps["hardware"].get_realtime_stats = AsyncMock(return_value=GpuStats(total_vram_gb=10.0, used_vram_gb=10.0, gpus=None))
    model = VllmModel(hf_id="google/test", size="1GB")
    opts = VllmModelOptions()

    with pytest.raises(HTTPException) as exc_info:
        await svc._get_gpu_memory_utilization("default", "google/test", opts, model)  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 422
    assert "Not enough free GPU memory" in exc_info.value.detail
    assert svc.gpu_memory_utilization == 0.0


@pytest.mark.parametrize(
    ("installed", "max_cuda", "expected"),
    [
        ("v0.29.0-cu129-ubuntu2404", None, "v0.31.0-cu129-ubuntu2404"),
        ("v0.29.0-cu129-ubuntu2404", (13, 4), "v0.31.0-cu129-ubuntu2404"),
        ("v0.29.0-cu129", None, "v0.31.0-cu129"),
    ],
)
def test_newest_tag_in_the_real_docker_hub_list(svc: VllmService, installed: str, max_cuda: tuple[int, int] | None, expected: str) -> None:
    tags = svc.filter_docker_tags(real_tags("vllm"), "GPU")
    assert newest_docker_tag(tags, installed, max_cuda) == expected


def test_read_models_reads_stt_entries_from_catalog(tmp_path: Path) -> None:
    (tmp_path / "static").mkdir()
    (tmp_path / "static" / "vllm-min.json").write_text(json.dumps({"stts": [{"name": "openai/whisper-small", "size": "1 GB"}]}))

    with patch("server.services.vllm_service.get_main_dir", return_value=tmp_path):
        models = _read_models()

    assert models["openai/whisper-small"].model_type == "stt"
    assert models["openai/whisper-small"].hf_id == "openai/whisper-small"
