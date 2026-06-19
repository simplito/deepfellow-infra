# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call, patch

import aiohttp
import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from server.models.models import InstallModelIn, ListModelsFilters, UninstallModelIn
from server.models.services import InstallServiceIn, MemoryLoadOut, MemoryLoadSession, UninstallServiceIn
from server.services.base2_service import CustomModel, Instance, InstanceConfig
from server.services.ollama_service import (
    DownloadedInfo,
    InstalledInfo,
    ModelInstalledInfo,
    OllamaModel,
    OllamaModelFile,
    OllamaModelFileLine,
    OllamaModelOptions,
    OllamaOptions,
    OllamaService,
    _const,  # pyright: ignore[reportPrivateUsage]
)
from server.utils.core import DownloadedPacket, FetchResult, PreDownloadPacket, Stream, StreamChunkProgress
from server.utils.hardware import IntelGpuInfo, NvidiaGpuInfo


def _make_installed_info(svc: OllamaService, instance: str = "default") -> InstalledInfo:
    docker = MagicMock()
    docker.name = f"ollama-{instance}"
    return InstalledInfo(
        instance_name=instance,
        docker=docker,
        models={},
        options=InstallServiceIn(spec={}),
        parsed_options=OllamaOptions(),
        container_host="localhost",
        container_port=11434,
        docker_exposed_port=11434,
        base_url="http://localhost:11434",
    )


@pytest.fixture
def deps() -> dict[str, Any]:
    return {
        "config": MagicMock(),
        "endpoint_registry": MagicMock(),
        "service_provider": MagicMock(),
        "model_downloader": MagicMock(),
        "docker_service": MagicMock(),
        "hardware": MagicMock(gpus=[], total_vram_gb=8.0),
    }


@pytest.fixture
def svc(deps: dict[str, Any]) -> OllamaService:
    return OllamaService(**deps)


def test_get_type(svc: OllamaService) -> None:
    assert svc.get_type() == "ollama"


@pytest.mark.parametrize(("id", "result"), [("default", "ollama"), ("gpu-1", "ollama|gpu-1")])
def test_get_id_default_instance(id: str, result: str, svc: OllamaService) -> None:
    assert svc.get_id(id) == result


def test_service_has_docker(svc: OllamaService) -> None:
    assert svc.service_has_docker() is True


def test_is_not_cloud_service(svc: OllamaService) -> None:
    assert svc.is_cloud_service() is False


def test_spec_contains_all_required_fields(svc: OllamaService) -> None:
    spec = svc.get_spec()

    field_names = {f.name for f in spec.fields}
    assert {
        "hardware",
        "num_parallel",
        "keep_alive",
        "is_flash_attention",
        "max_loaded_models",
        "kv_cache_type",
        "context_length",
    } <= field_names


def test_spec_num_parallel_is_optional_with_default(svc: OllamaService) -> None:
    spec = svc.get_spec()

    field = next(f for f in spec.fields if f.name == "num_parallel")
    assert field.required is False
    assert field.default == "3"


def test_spec_all_optional_fields_not_required(svc: OllamaService) -> None:
    spec = svc.get_spec()

    optional_names = {"num_parallel", "keep_alive", "is_flash_attention", "max_loaded_models", "kv_cache_type", "context_length"}
    for field in spec.fields:
        if field.name in optional_names:
            assert field.required is False, f"Field {field.name!r} should be optional"


def test_model_spec_has_alias_alive_time_context(svc: OllamaService) -> None:
    spec = svc.get_model_spec()

    field_names = {f.name for f in spec.fields}
    assert {"alias", "alive_time", "context_length"} <= field_names


def test_custom_model_spec_not_none(svc: OllamaService) -> None:
    assert svc.get_custom_model_spec() is not None


def test_custom_model_spec_has_required_fields(svc: OllamaService) -> None:
    spec = svc.get_custom_model_spec()

    assert spec is not None
    field_names = {f.name for f in spec.fields}
    assert {"id", "size", "type"} <= field_names


@pytest.mark.parametrize("alive_time", ["5m", "60s", "2h", "100s", ""])
def test_model_options_valid_alive_time(alive_time: str) -> None:
    opts = OllamaModelOptions(alive_time=alive_time)

    assert opts.alive_time == alive_time


def test_model_options_alive_time_accepts_int() -> None:
    opts = OllamaModelOptions(alive_time=30)

    assert opts.alive_time == 30


@pytest.mark.parametrize("bad_value", ["5minutes", "1x", "2 h", "half hour"])
def test_model_options_invalid_alive_time_raises(bad_value: str) -> None:
    with pytest.raises(ValidationError):
        OllamaModelOptions(alive_time=bad_value)


def test_model_options_alias_defaults_none() -> None:
    opts = OllamaModelOptions()

    assert opts.alias is None


def test_model_options_context_length_defaults_none() -> None:
    opts = OllamaModelOptions()

    assert opts.context_length is None


def test_ollama_options_all_defaults_none_or_empty() -> None:
    opts = OllamaOptions()

    assert opts.hardware is None
    assert opts.num_parallel is None
    assert opts.keep_alive == ""
    assert opts.is_flash_attention is None
    assert opts.max_loaded_models is None
    assert opts.kv_cache_type == ""
    assert opts.context_length is None


def test_build_env_vars_no_cloud_always_set(svc: OllamaService) -> None:
    envs = svc._build_env_vars(OllamaOptions(), [])  # pyright: ignore[reportPrivateUsage]
    assert envs["OLLAMA_NO_CLOUD"] == "1"


def test_build_env_vars_num_parallel(svc: OllamaService) -> None:
    envs = svc._build_env_vars(OllamaOptions(num_parallel=4), [])  # pyright: ignore[reportPrivateUsage]
    assert envs["OLLAMA_NUM_PARALLEL"] == "4"


def test_build_env_vars_num_parallel_none_absent(svc: OllamaService) -> None:
    envs = svc._build_env_vars(OllamaOptions(num_parallel=None), [])  # pyright: ignore[reportPrivateUsage]
    assert "OLLAMA_NUM_PARALLEL" not in envs


def test_build_env_vars_keep_alive(svc: OllamaService) -> None:
    envs = svc._build_env_vars(OllamaOptions(keep_alive="60m"), [])  # pyright: ignore[reportPrivateUsage]
    assert envs["OLLAMA_KEEP_ALIVE"] == "60m"


def test_build_env_vars_keep_alive_empty_absent(svc: OllamaService) -> None:
    envs = svc._build_env_vars(OllamaOptions(keep_alive=""), [])  # pyright: ignore[reportPrivateUsage]
    assert "OLLAMA_KEEP_ALIVE" not in envs


def test_build_env_vars_flash_attention_true(svc: OllamaService) -> None:
    envs = svc._build_env_vars(OllamaOptions(is_flash_attention=True), [])  # pyright: ignore[reportPrivateUsage]
    assert envs["OLLAMA_FLASH_ATTENTION"] == "1"


def test_build_env_vars_flash_attention_false(svc: OllamaService) -> None:
    envs = svc._build_env_vars(OllamaOptions(is_flash_attention=False), [])  # pyright: ignore[reportPrivateUsage]
    assert envs["OLLAMA_FLASH_ATTENTION"] == "0"


def test_build_env_vars_flash_attention_none_absent(svc: OllamaService) -> None:
    envs = svc._build_env_vars(OllamaOptions(is_flash_attention=None), [])  # pyright: ignore[reportPrivateUsage]
    assert "OLLAMA_FLASH_ATTENTION" not in envs


def test_build_env_vars_max_loaded_models(svc: OllamaService) -> None:
    envs = svc._build_env_vars(OllamaOptions(max_loaded_models=2), [])  # pyright: ignore[reportPrivateUsage]
    assert envs["OLLAMA_MAX_LOADED_MODELS"] == "2"


def test_build_env_vars_kv_cache_type(svc: OllamaService) -> None:
    envs = svc._build_env_vars(OllamaOptions(kv_cache_type="q8_0"), [])  # pyright: ignore[reportPrivateUsage]
    assert envs["OLLAMA_KV_CACHE_TYPE"] == "q8_0"


def test_build_env_vars_kv_cache_empty_absent(svc: OllamaService) -> None:
    envs = svc._build_env_vars(OllamaOptions(kv_cache_type=""), [])  # pyright: ignore[reportPrivateUsage]
    assert "OLLAMA_KV_CACHE_TYPE" not in envs


def test_build_env_vars_context_length(svc: OllamaService) -> None:
    envs = svc._build_env_vars(OllamaOptions(context_length=8192), [])  # pyright: ignore[reportPrivateUsage]
    assert envs["OLLAMA_CONTEXT_LENGTH"] == "8192"


def test_build_env_vars_context_length_none_absent(svc: OllamaService) -> None:
    envs = svc._build_env_vars(OllamaOptions(context_length=None), [])  # pyright: ignore[reportPrivateUsage]
    assert "OLLAMA_CONTEXT_LENGTH" not in envs


def test_build_env_vars_intel_gpu_sets_vulkan(svc: OllamaService) -> None:
    intel_gpu = MagicMock(spec=IntelGpuInfo)
    envs = svc._build_env_vars(OllamaOptions(), [intel_gpu])  # pyright: ignore[reportPrivateUsage]
    assert envs["OLLAMA_VULKAN"] == "1"


def test_build_env_vars_nvidia_gpu_no_vulkan(svc: OllamaService) -> None:
    nvidia_gpu = MagicMock(spec=NvidiaGpuInfo)
    envs = svc._build_env_vars(OllamaOptions(), [nvidia_gpu])  # pyright: ignore[reportPrivateUsage]
    assert "OLLAMA_VULKAN" not in envs


def test_build_env_vars_all_options_set(svc: OllamaService) -> None:
    opts = OllamaOptions(
        num_parallel=3,
        keep_alive="5m",
        is_flash_attention=True,
        max_loaded_models=1,
        kv_cache_type="f16",
        context_length=4096,
    )

    envs = svc._build_env_vars(opts, [])  # pyright: ignore[reportPrivateUsage]

    assert envs["OLLAMA_NUM_PARALLEL"] == "3"
    assert envs["OLLAMA_KEEP_ALIVE"] == "5m"
    assert envs["OLLAMA_FLASH_ATTENTION"] == "1"
    assert envs["OLLAMA_MAX_LOADED_MODELS"] == "1"
    assert envs["OLLAMA_KV_CACHE_TYPE"] == "f16"
    assert envs["OLLAMA_CONTEXT_LENGTH"] == "4096"
    assert envs["OLLAMA_NO_CLOUD"] == "1"


@pytest.mark.parametrize(
    ("vram_gb", "expected"),
    [
        (4.0, 4096),
        (16.0, 4096),
        (23.9, 4096),
        (24.0, 32768),
        (128.0, 32768),
        (256.0, 32768),
        (256.1, 262144),
        (512.0, 262144),
    ],
)
def test_get_default_context_value(deps: dict[str, Any], vram_gb: float, expected: int) -> None:
    deps["hardware"].total_vram_gb = vram_gb
    svc = OllamaService(**deps)

    result = svc.get_default_context_value()

    assert result == expected


@pytest.mark.parametrize(
    ("model_context", "service_context", "expected"),
    [
        (2048, 4096, 2048),  # model smaller
        (8192, 4096, 4096),  # service smaller
        (None, 4096, 4096),  # no model context
    ],
)
def test_get_default_context_window(svc: OllamaService, model_context: int, service_context: int, expected: int) -> None:
    assert svc.get_default_context_window(model_context, service_context) == expected


@pytest.mark.parametrize(
    ("model_context", "service_context_length", "expected"),
    [
        (32768, None, 4096),
        (32768, 4096, 4096),
        (32768, 16384, 16384),
        (2048, 4096, 2048),
        (None, None, 4096),
    ],
)
def test_effective_context_caps_native_window_by_runtime(
    svc: OllamaService, model_context: int | None, service_context_length: int | None, expected: int
) -> None:
    parsed_options = OllamaOptions(context_length=service_context_length)

    assert svc._effective_context(model_context, parsed_options) == expected  # pyright: ignore[reportPrivateUsage]


def test_modelfile_parse_from_line() -> None:
    mf = OllamaModelFile.parse("FROM llama3\nPARAMETER num_ctx 4096")
    from_lines = [line for line in mf.lines if isinstance(line, OllamaModelFileLine) and line.instruction == "FROM"]

    assert len(from_lines) == 1
    assert from_lines[0].value == "llama3"


def test_modelfile_parse_adapter_line() -> None:
    mf = OllamaModelFile.parse("FROM llama3\nADAPTER /path/adapter.bin")
    adapter_lines = [line for line in mf.lines if isinstance(line, OllamaModelFileLine) and line.instruction == "ADAPTER"]

    assert len(adapter_lines) == 1
    assert adapter_lines[0].value == "/path/adapter.bin"


def test_modelfile_parse_non_directive_line_preserved() -> None:
    mf = OllamaModelFile.parse("FROM llama3\nPARAMETER num_ctx 4096")
    plain_lines = [line for line in mf.lines if isinstance(line, str)]

    assert any("PARAMETER" in line for line in plain_lines)


def test_modelfile_render_round_trip() -> None:
    original = "FROM llama3\nPARAMETER num_ctx 4096"

    rendered = OllamaModelFile.parse(original).render()

    assert "FROM llama3" in rendered
    assert "PARAMETER num_ctx 4096" in rendered


def test_modelfile_render_multiple_lines() -> None:
    src = "FROM gemma3:1b\nADAPTER ./lora.bin\nPARAMETER num_ctx 8192"

    rendered = OllamaModelFile.parse(src).render()

    assert "FROM gemma3:1b" in rendered
    assert "ADAPTER ./lora.bin" in rendered


def test_modelfile_line_render_from() -> None:
    assert OllamaModelFileLine(instruction="FROM", value="my-model").render() == "FROM my-model"


def test_modelfile_line_render_adapter() -> None:
    assert OllamaModelFileLine(instruction="ADAPTER", value="./lora.bin").render() == "ADAPTER ./lora.bin"


def test_add_custom_llm_model(svc: OllamaService) -> None:
    custom = CustomModel(id="c-1", data={"id": "my-llm", "size": "2GB", "type": "llm"})

    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    assert "my-llm" in svc.models["default"]
    assert svc.models["default"]["my-llm"].type == "llm"


def test_add_custom_embedding_model(svc: OllamaService) -> None:
    custom = CustomModel(id="c-2", data={"id": "my-emb", "size": "500MB", "type": "embedding"})

    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    assert svc.models["default"]["my-emb"].type == "embedding"


def test_add_custom_model_stores_custom_id(svc: OllamaService) -> None:
    custom = CustomModel(id="uuid-123", data={"id": "my-model", "size": "1GB", "type": "llm"})

    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    assert svc.models["default"]["my-model"].custom == "uuid-123"


def test_add_custom_model_duplicate_raises_http_400(svc: OllamaService) -> None:
    custom = CustomModel(id="c-3", data={"id": "dup", "size": "1GB", "type": "llm"})

    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    with pytest.raises(HTTPException) as exc_info:
        svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400


def test_add_custom_model_invalid_type_raises_http_400(svc: OllamaService) -> None:
    custom = CustomModel(id="c-4", data={"id": "bad-type", "size": "1GB", "type": "invalid_type"})

    with pytest.raises(HTTPException) as exc_info:
        svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400


def test_get_description_not_empty(svc: OllamaService) -> None:
    assert svc.get_description()


def test_get_size_not_empty(svc: OllamaService) -> None:
    assert svc.get_size()


def test_get_image_returns_const_image(svc: OllamaService) -> None:
    assert svc._get_image() == _const.image  # pyright: ignore[reportPrivateUsage]


def test_load_download_info_returns_dataclass(svc: OllamaService) -> None:
    result = svc._load_download_info({})  # pyright: ignore[reportPrivateUsage]

    assert isinstance(result, DownloadedInfo)


def test_generate_instance_config_no_info(svc: OllamaService) -> None:
    config = svc._generate_instance_config(None, None)  # pyright: ignore[reportPrivateUsage]

    assert config.options is None
    assert config.models == []


def test_generate_instance_config_with_info(svc: OllamaService) -> None:
    info = _make_installed_info(svc)
    info.models["my-llm"] = ModelInstalledInfo(
        id="my-llm",
        registered_name="my-llm",
        type="llm",
        options=InstallModelIn(spec={}),
        registration_id="reg-1",
        internal_name=None,
    )

    config = svc._generate_instance_config(info, None)  # pyright: ignore[reportPrivateUsage]

    assert config.options == info.options
    assert len(config.models or []) == 1


def test_model_installed_info_get_info() -> None:
    model = ModelInstalledInfo(
        id="my-llm",
        registered_name="my-llm",
        type="llm",
        options=InstallModelIn(spec={}),
        registration_id="reg-42",
        internal_name=None,
    )

    info = model.get_info()

    assert info.registration_id == "reg-42"


def test_get_local_modelfile_path(svc: OllamaService) -> None:
    path = svc._get_local_modelfile_path("my-model", "default")  # pyright: ignore[reportPrivateUsage]

    assert str(path).replace("\\", "/").endswith("custom/my-model/default/Modelfile")


def test_get_docker_modelfile_path(svc: OllamaService) -> None:
    path = svc._get_docker_modelfile_path("my-model", "default")  # pyright: ignore[reportPrivateUsage]

    assert str(path).replace("\\", "/").endswith("custom/my-model/default/Modelfile")


def test_get_modelfile_file_name_from_url_gguf(svc: OllamaService) -> None:
    name = svc._get_modelfile_file_name_from_url("https://example.com/model.gguf")  # pyright: ignore[reportPrivateUsage]

    assert name.endswith(".gguf")


def test_get_modelfile_file_name_from_url_non_gguf(svc: OllamaService) -> None:
    name = svc._get_modelfile_file_name_from_url("https://example.com/model")  # pyright: ignore[reportPrivateUsage]

    assert not name.endswith(".gguf")


def test_remove_paths_removes_file(svc: OllamaService, tmp_path: Path) -> None:
    f = tmp_path / "model.bin"
    f.write_text("data")
    svc._remove_paths([f])  # pyright: ignore[reportPrivateUsage]

    assert not f.exists()


def test_remove_paths_removes_directory(svc: OllamaService, tmp_path: Path) -> None:
    d = tmp_path / "model_dir"
    d.mkdir()
    (d / "file.bin").write_text("data")
    svc._remove_paths([d])  # pyright: ignore[reportPrivateUsage]

    assert not d.exists()


def test_remove_paths_ignores_nonexistent_path(svc: OllamaService, tmp_path: Path) -> None:
    svc._remove_paths([tmp_path / "nonexistent"])  # pyright: ignore[reportPrivateUsage]
    # should not raise


def test_get_installed_info_returns_spec_when_installed(svc: OllamaService) -> None:
    installed = _make_installed_info(svc)
    installed.options.spec["hardware"] = True
    svc.instances_info["default"].installed = installed

    result = svc.get_installed_info("default")

    assert result == installed.options.spec


def test_get_installed_info_calls_get_service_installed_info_when_none(svc: OllamaService) -> None:
    svc.instances_info["default"].installed = None

    with patch.object(svc, "_get_service_installed_info", return_value=False) as mock:  # pyright: ignore[reportPrivateUsage]
        result = svc.get_installed_info("default")

    assert mock.call_count == 1
    assert mock.call_args == call("default")
    assert result is False


def test_get_docker_compose_file_path_raises_400_with_model_id(svc: OllamaService) -> None:
    installed = _make_installed_info(svc)
    svc.instances_info["default"].installed = installed

    with pytest.raises(HTTPException) as exc_info:
        svc.get_docker_compose_file_path("default", "some-model")

    assert exc_info.value.status_code == 400


def test_get_docker_compose_file_path_returns_path_without_model_id(svc: OllamaService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info(svc)
    svc.instances_info["default"].installed = installed
    expected = Path("/some/path/docker-compose.yml")
    deps["docker_service"].get_docker_compose_file_path.return_value = expected

    result = svc.get_docker_compose_file_path("default", None)

    assert result == expected


def test_add_custom_model_creates_models_dict_for_new_instance(svc: OllamaService) -> None:
    svc.instances_info["extra"] = Instance(None, None, {}, InstanceConfig())
    custom = CustomModel(id="c-10", data={"id": "new-model", "size": "1GB", "type": "llm"})

    svc._add_custom_model("extra", custom)  # pyright: ignore[reportPrivateUsage]

    assert "new-model" in svc.models["extra"]


def test_remove_custom_model_deletes_entry(svc: OllamaService) -> None:
    custom = CustomModel(id="c-5", data={"id": "to-remove", "size": "1GB", "type": "llm"})

    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]
    svc._remove_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    assert "to-remove" not in svc.models["default"]


def test_remove_custom_model_raises_400_when_in_use(svc: OllamaService) -> None:
    custom = CustomModel(id="c-6", data={"id": "in-use", "size": "1GB", "type": "llm"})
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]
    installed = _make_installed_info(svc)
    installed.models["in-use"] = ModelInstalledInfo(
        id="in-use",
        registered_name="in-use",
        type="llm",
        options=InstallModelIn(spec={}),
        registration_id="",
        internal_name=None,
    )
    svc.instances_info["default"].installed = installed

    with pytest.raises(HTTPException) as exc_info:
        svc._remove_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_stop_instance_does_nothing_when_not_installed(svc: OllamaService) -> None:
    svc.instances_info["default"].installed = None

    with patch.object(svc, "_stop_docker", new_callable=AsyncMock) as mock_stop:  # pyright: ignore[reportPrivateUsage]
        await svc.stop_instance("default")

    assert mock_stop.call_count == 0


@pytest.mark.asyncio
async def test_stop_instance_calls_stop_docker_when_installed(svc: OllamaService) -> None:
    installed = _make_installed_info(svc)
    svc.instances_info["default"].installed = installed

    with patch.object(svc, "_stop_docker", new_callable=AsyncMock) as mock_stop:  # pyright: ignore[reportPrivateUsage]
        await svc.stop_instance("default")

    assert mock_stop.call_count == 1
    assert mock_stop.call_args == call(installed.docker)


@pytest.mark.asyncio
async def test_is_model_installed_true() -> None:
    with patch("server.services.ollama_service.fetch_from", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = FetchResult(status_code=200, data="")

        result = await OllamaService.is_model_installed("http://localhost:11434", "llama3")

    assert result is True


@pytest.mark.asyncio
async def test_is_model_installed_false() -> None:
    with patch("server.services.ollama_service.fetch_from", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = FetchResult(status_code=404, data="")

        result = await OllamaService.is_model_installed("http://localhost:11434", "unknown")

    assert result is False


@pytest.mark.asyncio
async def test_create_model_from_modelfile_calls_docker_compose_run(svc: OllamaService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info(svc)
    deps["docker_service"].run_command_docker_compose = AsyncMock(return_value="")
    deps["docker_service"].get_docker_compose_file_path.return_value = Path("/path/docker-compose.yml")

    await svc.create_model_from_modelfile(installed, "my-model", "/root/.ollama/Modelfile", None)

    cmd = deps["docker_service"].run_command_docker_compose.call_args[0][2]
    assert "ollama create" in cmd


@pytest.mark.asyncio
async def test_create_model_from_modelfile_includes_quantization_flag(svc: OllamaService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info(svc)
    deps["docker_service"].run_command_docker_compose = AsyncMock(return_value="")
    deps["docker_service"].get_docker_compose_file_path.return_value = Path("/path/docker-compose.yml")

    await svc.create_model_from_modelfile(installed, "my-model", "/root/.ollama/Modelfile", "q4_0")

    cmd = deps["docker_service"].run_command_docker_compose.call_args[0][2]
    assert "-q q4_0" in cmd


@pytest.mark.asyncio
async def test_remove_modelfile_deletes_file_and_empty_parent(svc: OllamaService, tmp_path: Path) -> None:
    model_dir = tmp_path / "main" / "custom" / "my-model" / "default"
    model_dir.mkdir(parents=True)
    modelfile = model_dir / "Modelfile"
    modelfile.write_text("FROM llama3")

    with patch.object(svc, "_get_working_dir", return_value=tmp_path):  # pyright: ignore[reportPrivateUsage]
        await svc.remove_modelfile("my-model", "default")

    assert not modelfile.exists()
    assert not model_dir.exists()


@pytest.mark.asyncio
async def test_remove_modelfile_does_not_delete_nonempty_parent(svc: OllamaService, tmp_path: Path) -> None:
    model_dir = tmp_path / "main" / "custom" / "my-model" / "default"
    model_dir.mkdir(parents=True)
    modelfile = model_dir / "Modelfile"
    modelfile.write_text("FROM llama3")
    (model_dir / "other.bin").write_text("data")

    with patch.object(svc, "_get_working_dir", return_value=tmp_path):  # pyright: ignore[reportPrivateUsage]
        await svc.remove_modelfile("my-model", "default")

    assert not modelfile.exists()
    assert model_dir.exists()


@pytest.mark.asyncio
async def test_save_modelfile_writes_when_file_absent(svc: OllamaService, tmp_path: Path) -> None:
    with patch.object(svc, "_get_working_dir", return_value=tmp_path):  # pyright: ignore[reportPrivateUsage]
        result = await svc.save_modelfile("my-model", "default", "FROM llama3")

    local_path = tmp_path / "main" / "custom" / "my-model" / "default" / "Modelfile"
    assert local_path.exists()
    assert local_path.read_text() == "FROM llama3"
    assert "Modelfile" in result


@pytest.mark.asyncio
async def test_save_modelfile_skips_write_when_content_unchanged(svc: OllamaService, tmp_path: Path) -> None:
    local_path = tmp_path / "main" / "custom" / "my-model" / "default" / "Modelfile"
    local_path.parent.mkdir(parents=True)
    local_path.write_text("FROM llama3")

    with patch.object(svc, "_get_working_dir", return_value=tmp_path):  # pyright: ignore[reportPrivateUsage]
        result = await svc.save_modelfile("my-model", "default", "FROM llama3")

    # file should still have original content (not re-written)
    assert local_path.read_text() == "FROM llama3"
    assert "Modelfile" in result


@pytest.mark.asyncio
async def test_list_models_returns_all_models_for_valid_instance(svc: OllamaService) -> None:
    installed = _make_installed_info(svc)
    svc.instances_info["default"].installed = installed

    result = await svc.list_models("default", ListModelsFilters())

    assert len(result.list) == len(svc.models["default"])


@pytest.mark.asyncio
async def test_list_models_raises_404_for_unknown_instance(svc: OllamaService) -> None:
    with pytest.raises(HTTPException) as exc_info:
        await svc.list_models("nonexistent", ListModelsFilters())

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_list_models_filters_to_installed_only(svc: OllamaService) -> None:
    installed = _make_installed_info(svc)
    some_model_id = next(iter(svc.models["default"]))
    installed.models[some_model_id] = ModelInstalledInfo(
        id=some_model_id,
        registered_name=some_model_id,
        type="llm",
        options=InstallModelIn(spec={}),
        registration_id="reg-1",
        internal_name=None,
    )
    svc.instances_info["default"].installed = installed

    result = await svc.list_models("default", ListModelsFilters(installed=True))

    assert len(result.list) >= 1
    assert all(bool(m.installed) for m in result.list)


@pytest.mark.asyncio
async def test_get_model_raises_400_for_unknown_model_id(svc: OllamaService) -> None:
    installed = _make_installed_info(svc)
    svc.instances_info["default"].installed = installed

    with pytest.raises(HTTPException) as exc_info:
        await svc.get_model("default", "nonexistent-model")

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_get_model_returns_retrieve_model_out_for_known_model(svc: OllamaService) -> None:
    installed = _make_installed_info(svc)
    svc.instances_info["default"].installed = installed
    model_id = next(iter(svc.models["default"]))

    result = await svc.get_model("default", model_id)

    assert result.id == model_id


@pytest.mark.asyncio
async def test_download_with_ollama_emits_progress_events(svc: OllamaService) -> None:
    stream = MagicMock()
    success_data = json.dumps({"status": "success"})

    async def mock_stream_fetch(*args: object, **kwargs: object):  # type: ignore[misc]
        yield FetchResult(status_code=200, data=success_data)

    with patch("server.services.ollama_service.stream_fetch_from", side_effect=mock_stream_fetch):
        await svc._download_with_ollama(stream, "http://localhost:11434", "llama3", "1GB")  # pyright: ignore[reportPrivateUsage]

    assert stream.emit.call_count >= 2


@pytest.mark.asyncio
async def test_download_with_ollama_raises_400_on_error_response(svc: OllamaService) -> None:
    stream = MagicMock()
    error_data = json.dumps({"error": "model not found"})

    async def mock_stream_fetch(*args: object, **kwargs: object):  # type: ignore[misc]
        yield FetchResult(status_code=200, data=error_data)

    with patch("server.services.ollama_service.stream_fetch_from", side_effect=mock_stream_fetch), pytest.raises(HTTPException) as exc_info:
        await svc._download_with_ollama(stream, "http://localhost:11434", "bad-model", "1GB")  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_download_with_downloader_uses_ollama_for_non_url(svc: OllamaService) -> None:
    stream = MagicMock()
    with patch.object(svc, "_download_with_ollama", new_callable=AsyncMock) as mock_ollama:  # pyright: ignore[reportPrivateUsage]
        await svc._download_with_downloader(stream, "http://localhost:11434", "llama3", "llama3", "1GB", 0, 1)  # pyright: ignore[reportPrivateUsage]

    assert mock_ollama.call_count == 1


@pytest.mark.asyncio
async def test_download_with_downloader_uses_model_downloader_for_https_url(svc: OllamaService, deps: dict[str, Any]) -> None:
    stream = MagicMock()

    async def empty_gen(*args: object):  # type: ignore[misc]
        return
        yield

    deps["model_downloader"].download.return_value = empty_gen()

    with patch.object(svc, "_get_working_dir", return_value=Path("/tmp/test")):  # pyright: ignore[reportPrivateUsage]
        await svc._download_with_downloader(  # pyright: ignore[reportPrivateUsage]
            stream, "http://localhost:11434", "https://example.com/model.gguf", "my-model", "1GB", 0, 1
        )

    assert deps["model_downloader"].download.call_count == 1


@pytest.mark.asyncio
async def test_download_model_or_set_progress_starts_download(svc: OllamaService) -> None:
    stream = MagicMock()

    with patch.object(svc, "_download_with_downloader", new_callable=AsyncMock) as mock_dl:  # pyright: ignore[reportPrivateUsage]
        await svc._download_model_or_set_progress(stream, "http://localhost:11434", "llama3", "llama3", "1GB")  # pyright: ignore[reportPrivateUsage]

    assert mock_dl.call_count == 1
    assert "llama3" not in svc.models_download_progress


@pytest.mark.asyncio
async def test_download_model_or_set_progress_cleans_up_on_failure(svc: OllamaService) -> None:
    stream = MagicMock()

    with (
        patch.object(svc, "_download_with_downloader", new_callable=AsyncMock, side_effect=HTTPException(400, "boom")),  # pyright: ignore[reportPrivateUsage]
        pytest.raises(HTTPException),
    ):
        await svc._download_model_or_set_progress(stream, "http://localhost:11434", "llama3", "llama3", "1GB")  # pyright: ignore[reportPrivateUsage]

    assert "llama3" not in svc.models_download_progress


@pytest.mark.asyncio
async def test_download_model_or_set_progress_retries_download_after_failure(svc: OllamaService) -> None:
    stream1 = MagicMock()
    stream2 = MagicMock()

    mock_dl = AsyncMock(side_effect=[HTTPException(400, "boom"), None])
    with patch.object(svc, "_download_with_downloader", mock_dl):  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(HTTPException):
            await svc._download_model_or_set_progress(stream1, "http://localhost:11434", "llama3", "llama3", "1GB")  # pyright: ignore[reportPrivateUsage]
        await svc._download_model_or_set_progress(stream2, "http://localhost:11434", "llama3", "llama3", "1GB")  # pyright: ignore[reportPrivateUsage]

    assert mock_dl.call_count == 2


@pytest.mark.asyncio
async def test_download_model_or_set_progress_forwards_when_in_progress(svc: OllamaService) -> None:
    existing_stream: Stream[StreamChunkProgress] = Stream()  # type: ignore[type-arg]
    chunk = StreamChunkProgress(type="progress", stage="download", value=0.5, data={})
    existing_stream.emit(chunk)
    existing_stream.close()
    svc.models_download_progress["llama3"] = existing_stream  # type: ignore[assignment]

    output_stream = MagicMock()

    await svc._download_model_or_set_progress(  # pyright: ignore[reportPrivateUsage]
        output_stream, "http://localhost:11434", "llama3", "llama3", "1GB"
    )

    assert output_stream.emit.call_count == 1
    assert output_stream.emit.call_args == call(chunk)


@pytest.mark.asyncio
async def test_install_from_modelfile_downloads_https_asset(svc: OllamaService, tmp_path: Path) -> None:
    model_id = "my-model"
    model = OllamaModel(id=model_id, size="1GB", type="llm", hash="abc", context=4096)
    instance_info = _make_installed_info(svc)

    with (
        patch.object(svc, "_get_working_dir", return_value=tmp_path),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock) as mock_dl,  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "save_modelfile", new_callable=AsyncMock, return_value="/root/.ollama/Modelfile"),
        patch.object(svc, "create_model_from_modelfile", new_callable=AsyncMock, return_value=""),
        patch("server.services.ollama_service.detect_context_window_from_path", new_callable=AsyncMock, return_value=None),
    ):
        await svc._install_from_modelfile(  # pyright: ignore[reportPrivateUsage]
            MagicMock(), model_id, "FROM https://example.com/model.gguf", model_id, instance_info, model
        )

    assert mock_dl.call_count == 1


@pytest.mark.asyncio
async def test_install_from_modelfile_raises_400_for_missing_local_path(svc: OllamaService, tmp_path: Path) -> None:
    model_id = "my-model"
    model = OllamaModel(id=model_id, size="1GB", type="llm", hash="abc", context=4096)
    instance_info = _make_installed_info(svc)

    with patch.object(svc, "_get_working_dir", return_value=tmp_path), pytest.raises(HTTPException) as exc_info:
        await svc._install_from_modelfile(  # pyright: ignore[reportPrivateUsage]
            MagicMock(), model_id, "FROM ./missing.gguf", model_id, instance_info, model
        )

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_install_from_modelfile_downloads_via_ollama_when_not_installed(svc: OllamaService, tmp_path: Path) -> None:
    model_id = "my-model"
    model = OllamaModel(id=model_id, size="1GB", type="llm", hash="abc", context=4096)
    instance_info = _make_installed_info(svc)

    with (
        patch.object(svc, "_get_working_dir", return_value=tmp_path),  # pyright: ignore[reportPrivateUsage]
        patch.object(OllamaService, "is_model_installed", new_callable=AsyncMock, return_value=False),
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock) as mock_dl,  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "save_modelfile", new_callable=AsyncMock, return_value="/root/.ollama/Modelfile"),
        patch.object(svc, "create_model_from_modelfile", new_callable=AsyncMock, return_value=""),
    ):
        await svc._install_from_modelfile(  # pyright: ignore[reportPrivateUsage]
            MagicMock(), model_id, "FROM llama3", model_id, instance_info, model
        )

    assert mock_dl.call_count == 1


@pytest.mark.asyncio
async def test_install_from_modelfile_wraps_unexpected_exception(svc: OllamaService, tmp_path: Path) -> None:
    model_id = "my-model"
    model = OllamaModel(id=model_id, size="1GB", type="llm", hash="abc", context=4096)
    instance_info = _make_installed_info(svc)

    with (
        patch.object(svc, "_get_working_dir", return_value=tmp_path),  # pyright: ignore[reportPrivateUsage]
        patch.object(OllamaService, "is_model_installed", new_callable=AsyncMock, return_value=True),
        patch.object(svc, "save_modelfile", new_callable=AsyncMock, side_effect=RuntimeError("boom")),
        pytest.raises(HTTPException) as exc_info,
    ):
        await svc._install_from_modelfile(  # pyright: ignore[reportPrivateUsage]
            MagicMock(), model_id, "FROM llama3", model_id, instance_info, model
        )

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_install_model_returns_already_installed_when_model_in_info(svc: OllamaService) -> None:
    installed = _make_installed_info(svc)
    model_id = next(iter(svc.models["default"]))
    installed.models[model_id] = ModelInstalledInfo(
        id=model_id,
        registered_name=model_id,
        type="llm",
        options=InstallModelIn(spec={}),
        registration_id="",
        internal_name=None,
    )
    svc.instances_info["default"].installed = installed

    promise = await svc._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
    result = await promise.wait()

    assert result.status == "OK"
    assert "Already installed" in result.details


@pytest.mark.asyncio
async def test_install_model_raises_400_for_unknown_model(svc: OllamaService) -> None:
    installed = _make_installed_info(svc)
    svc.instances_info["default"].installed = installed

    with pytest.raises(HTTPException) as exc_info:
        await svc._install_model("default", "nonexistent-model", InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_install_model_registers_llm_endpoint(svc: OllamaService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info(svc)
    svc.instances_info["default"].installed = installed
    svc.models["default"]["test-llm"] = OllamaModel(id="test-llm", size="1GB", type="llm", hash="abc", context=4096)

    with patch("server.services.ollama_service.fetch_from", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = FetchResult(status_code=200, data="")
        promise = await svc._install_model("default", "test-llm", InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert deps["endpoint_registry"].register_chat_completion_as_proxy.call_count == 1


@pytest.mark.asyncio
async def test_install_model_registers_embedding_endpoint(svc: OllamaService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info(svc)
    svc.instances_info["default"].installed = installed
    svc.models["default"]["test-emb"] = OllamaModel(id="test-emb", size="500MB", type="embedding", hash="def", context=None)

    with patch("server.services.ollama_service.fetch_from", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = FetchResult(status_code=200, data="")
        promise = await svc._install_model("default", "test-emb", InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert deps["endpoint_registry"].register_embeddings_as_proxy.call_count == 1


@pytest.mark.asyncio
async def test_install_model_registers_txt2img_endpoint(svc: OllamaService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info(svc)
    svc.instances_info["default"].installed = installed
    svc.models["default"]["test-img"] = OllamaModel(id="test-img", size="2GB", type="txt2img", hash="ghi", context=None)

    with patch("server.services.ollama_service.fetch_from", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = FetchResult(status_code=200, data="")
        promise = await svc._install_model("default", "test-img", InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert deps["endpoint_registry"].register_image_generations_as_proxy.call_count == 1


@pytest.mark.asyncio
async def test_install_model_marks_model_as_downloaded(svc: OllamaService) -> None:
    installed = _make_installed_info(svc)
    svc.instances_info["default"].installed = installed
    svc.models["default"]["test-llm"] = OllamaModel(id="test-llm", size="1GB", type="llm", hash="abc", context=4096)

    with patch("server.services.ollama_service.fetch_from", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = FetchResult(status_code=200, data="")
        promise = await svc._install_model("default", "test-llm", InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert "test-llm" in svc.models_downloaded


@pytest.mark.asyncio
async def test_install_model_uses_alias_as_registered_name(svc: OllamaService) -> None:
    installed = _make_installed_info(svc)
    svc.instances_info["default"].installed = installed
    svc.models["default"]["test-llm"] = OllamaModel(id="test-llm", size="1GB", type="llm", hash="abc", context=4096)

    with patch("server.services.ollama_service.fetch_from", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = FetchResult(status_code=200, data="")
        promise = await svc._install_model(  # pyright: ignore[reportPrivateUsage]
            "default", "test-llm", InstallModelIn(spec={"alias": "my-alias"})
        )
        await promise.wait()

    assert installed.models["test-llm"].registered_name == "my-alias"


@pytest.mark.asyncio
async def test_install_model_generates_context_modelfile_for_llm_with_custom_context(svc: OllamaService) -> None:
    installed = _make_installed_info(svc)
    svc.instances_info["default"].installed = installed
    svc.models["default"]["test-llm"] = OllamaModel(id="test-llm", size="1GB", type="llm", hash="abc", context=4096, modelfile=None)

    with patch.object(svc, "_install_from_modelfile", new_callable=AsyncMock, return_value=None) as mock_install:  # pyright: ignore[reportPrivateUsage]
        promise = await svc._install_model(  # pyright: ignore[reportPrivateUsage]
            "default", "test-llm", InstallModelIn(spec={"context_length": 8192})
        )
        await promise.wait()

    modelfile_recipe: str = mock_install.call_args.args[2]
    assert "num_ctx 8192" in modelfile_recipe
    assert "FROM test-llm" in modelfile_recipe


@pytest.mark.asyncio
async def test_uninstall_model_unregisters_llm_and_removes_from_info(svc: OllamaService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info(svc)
    installed.models["test-llm"] = ModelInstalledInfo(
        id="test-llm",
        registered_name="test-llm",
        type="llm",
        options=InstallModelIn(spec={}),
        registration_id="reg-1",
        internal_name=None,
    )
    svc.instances_info["default"].installed = installed
    svc.models["default"]["test-llm"] = OllamaModel(id="test-llm", size="1GB", type="llm", hash="abc", context=4096)

    await svc._uninstall_model("default", "test-llm", UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert "test-llm" not in installed.models
    assert deps["endpoint_registry"].unregister_chat_completion.call_count == 1
    assert deps["endpoint_registry"].unregister_chat_completion.call_args == call("test-llm", "reg-1")


@pytest.mark.asyncio
async def test_uninstall_model_purges_model_data(svc: OllamaService) -> None:
    installed = _make_installed_info(svc)
    installed.models["test-llm"] = ModelInstalledInfo(
        id="test-llm",
        registered_name="test-llm",
        type="llm",
        options=InstallModelIn(spec={}),
        registration_id="",
        internal_name=None,
    )
    svc.instances_info["default"].installed = installed
    svc.models["default"]["test-llm"] = OllamaModel(id="test-llm", size="1GB", type="llm", hash="abc", context=4096)
    svc.models_downloaded["test-llm"] = DownloadedInfo()

    with (
        patch("server.services.ollama_service.fetch_from", new_callable=AsyncMock) as mock_fetch,
        patch.object(svc, "remove_modelfile", new_callable=AsyncMock),
    ):
        mock_fetch.return_value = FetchResult(status_code=200, data="")
        await svc._uninstall_model("default", "test-llm", UninstallModelIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert mock_fetch.call_count == 1
    assert "test-llm" not in svc.models_downloaded


@pytest.mark.asyncio
async def test_uninstall_model_does_nothing_for_unknown_model_id(svc: OllamaService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info(svc)
    svc.instances_info["default"].installed = installed

    await svc._uninstall_model("default", "unknown-model", UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert deps["endpoint_registry"].unregister_chat_completion.call_count == 0


@pytest.mark.asyncio
async def test_install_instance_calls_docker_and_returns_installed_info(svc: OllamaService, deps: dict[str, Any]) -> None:
    deps["docker_service"].get_docker_subnet.return_value = None
    deps["docker_service"].get_docker_container_name.return_value = "df-ollama"
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=11434)
    deps["docker_service"].get_container_host.return_value = "localhost"
    deps["docker_service"].get_container_port.return_value = 11434
    options = InstallServiceIn(spec={})

    with (
        patch.object(svc, "_download_image_or_set_progress", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_verify_docker_image", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
    ):
        promise = await svc._install_instance("default", options)  # pyright: ignore[reportPrivateUsage]
        result = await promise.wait()

    assert isinstance(result, InstalledInfo)
    assert deps["docker_service"].install_and_run_docker.call_count == 1


@pytest.mark.asyncio
async def test_uninstall_instance_unregisters_all_model_endpoints(svc: OllamaService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info(svc)
    installed.models["llm-model"] = ModelInstalledInfo(
        id="llm-model",
        registered_name="llm-model",
        type="llm",
        options=InstallModelIn(spec={}),
        registration_id="reg-llm",
        internal_name=None,
    )
    installed.models["emb-model"] = ModelInstalledInfo(
        id="emb-model",
        registered_name="emb-model",
        type="embedding",
        options=InstallModelIn(spec={}),
        registration_id="reg-emb",
        internal_name=None,
    )
    svc.instances_info["default"].installed = installed
    svc.models["default"]["llm-model"] = OllamaModel(id="llm-model", size="1GB", type="llm", hash="a", context=4096)
    svc.models["default"]["emb-model"] = OllamaModel(id="emb-model", size="500MB", type="embedding", hash="b", context=None)
    deps["docker_service"].uninstall_docker = AsyncMock()

    with patch.object(svc, "_uninstall_model", new_callable=AsyncMock):  # pyright: ignore[reportPrivateUsage]
        await svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert deps["endpoint_registry"].unregister_chat_completion.call_count == 1
    assert deps["endpoint_registry"].unregister_chat_completion.call_args == call("llm-model", "reg-llm")
    assert deps["endpoint_registry"].unregister_embeddings.call_count == 1
    assert deps["endpoint_registry"].unregister_embeddings.call_args == call("emb-model", "reg-emb")
    assert deps["docker_service"].uninstall_docker.call_count == 1


@pytest.mark.asyncio
async def test_uninstall_instance_purges_working_dir_on_purge_single_instance(svc: OllamaService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info(svc)
    svc.instances_info["default"].installed = installed
    deps["docker_service"].uninstall_docker = AsyncMock()
    deps["docker_service"].remove_image = AsyncMock()

    with (
        patch.object(svc, "_uninstall_model", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_clear_working_dir", new_callable=AsyncMock) as mock_clear,  # pyright: ignore[reportPrivateUsage]
    ):
        await svc._uninstall_instance("default", UninstallServiceIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert svc.service_downloaded is False
    assert mock_clear.call_count == 1


@pytest.mark.asyncio
async def test_install_instance_loads_default_models_for_new_instance(svc: OllamaService, deps: dict[str, Any]) -> None:
    svc.instances_info["gpu-1"] = Instance(None, None, {}, InstanceConfig())
    deps["docker_service"].get_docker_subnet.return_value = None
    deps["docker_service"].get_docker_container_name.return_value = "df-ollama-gpu1"
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=11434)
    deps["docker_service"].get_container_host.return_value = "localhost"
    deps["docker_service"].get_container_port.return_value = 11434

    with (
        patch.object(svc, "_download_image_or_set_progress", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_verify_docker_image", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
    ):
        promise = await svc._install_instance("gpu-1", InstallServiceIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert "gpu-1" in svc.models


@pytest.mark.asyncio
async def test_install_instance_does_not_override_hardware_when_already_in_spec(svc: OllamaService, deps: dict[str, Any]) -> None:
    deps["docker_service"].get_docker_subnet.return_value = None
    deps["docker_service"].get_docker_container_name.return_value = "df-ollama"
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=11434)
    deps["docker_service"].get_container_host.return_value = "localhost"
    deps["docker_service"].get_container_port.return_value = 11434
    options = InstallServiceIn(spec={"hardware": False})

    with (
        patch.object(svc, "_download_image_or_set_progress", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_verify_docker_image", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
    ):
        promise = await svc._install_instance("default", options)  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert options.spec["hardware"] is False  # was not overwritten


@pytest.mark.asyncio
async def test_uninstall_instance_no_op_when_not_installed(svc: OllamaService, deps: dict[str, Any]) -> None:
    svc.instances_info["default"].installed = None

    await svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert deps["endpoint_registry"].unregister_chat_completion.call_count == 0
    assert deps["docker_service"].uninstall_docker.call_count == 0


@pytest.mark.asyncio
async def test_uninstall_instance_unregisters_txt2img_model(svc: OllamaService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info(svc)
    installed.models["img-model"] = ModelInstalledInfo(
        id="img-model",
        registered_name="img-model",
        type="txt2img",
        options=InstallModelIn(spec={}),
        registration_id="reg-img",
        internal_name=None,
    )
    svc.instances_info["default"].installed = installed
    deps["docker_service"].uninstall_docker = AsyncMock()

    with patch.object(svc, "_uninstall_model", new_callable=AsyncMock):  # pyright: ignore[reportPrivateUsage]
        await svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert deps["endpoint_registry"].unregister_image_generations.call_count == 1
    assert deps["endpoint_registry"].unregister_image_generations.call_args == call("img-model", "reg-img")


@pytest.mark.asyncio
async def test_uninstall_instance_skips_model_uninstall_when_shared_with_other_instance(svc: OllamaService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info(svc)
    installed.models["shared-model"] = ModelInstalledInfo(
        id="shared-model",
        registered_name="shared-model",
        type="llm",
        options=InstallModelIn(spec={}),
        registration_id="reg-1",
        internal_name=None,
    )
    svc.instances_info["default"].installed = installed
    deps["docker_service"].uninstall_docker = AsyncMock()

    with (
        patch.object(svc, "is_model_installed_in_other_instance", return_value=True),
        patch.object(svc, "_uninstall_model", new_callable=AsyncMock) as mock_uninstall,  # pyright: ignore[reportPrivateUsage]
    ):
        await svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert mock_uninstall.call_count == 0


@pytest.mark.asyncio
async def test_uninstall_instance_purge_with_multiple_instances_does_not_remove_image(svc: OllamaService, deps: dict[str, Any]) -> None:
    svc.instances_info["gpu-1"] = Instance(None, None, {}, InstanceConfig())
    installed = _make_installed_info(svc)
    svc.instances_info["default"].installed = installed
    deps["docker_service"].uninstall_docker = AsyncMock()

    with (
        patch.object(svc, "_uninstall_model", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_clear_working_dir", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
    ):
        await svc._uninstall_instance("default", UninstallServiceIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert deps["docker_service"].remove_image.call_count == 0


@pytest.mark.asyncio
async def test_uninstall_instance_purge_removes_non_default_instance(svc: OllamaService, deps: dict[str, Any]) -> None:
    svc.instances_info["gpu-1"] = Instance(None, None, {}, InstanceConfig())
    installed = _make_installed_info(svc)
    svc.instances_info["gpu-1"].installed = installed
    deps["docker_service"].uninstall_docker = AsyncMock()

    with (
        patch.object(svc, "_uninstall_model", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_clear_working_dir", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
    ):
        await svc._uninstall_instance("gpu-1", UninstallServiceIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert "gpu-1" not in svc.instances_info


@pytest.mark.asyncio
async def test_list_models_excludes_models_from_other_instances(svc: OllamaService) -> None:
    svc.instances_info["gpu-1"] = Instance(None, None, {}, InstanceConfig())
    svc.models["gpu-1"] = {"gpu-model": OllamaModel(id="gpu-model", size="1GB", type="llm", hash="x", context=4096)}
    installed = _make_installed_info(svc)
    svc.instances_info["default"].installed = installed

    result = await svc.list_models("default", ListModelsFilters())

    assert all(m.id != "gpu-model" for m in result.list)


@pytest.mark.asyncio
async def test_get_model_initializes_models_dict_when_missing(svc: OllamaService) -> None:
    del svc.models["default"]
    installed = _make_installed_info(svc)
    svc.instances_info["default"].installed = installed

    with pytest.raises(HTTPException) as exc_info:
        await svc.get_model("default", "nonexistent")

    assert exc_info.value.status_code == 400
    assert "default" in svc.models  # dict was created


@pytest.mark.asyncio
async def test_download_with_ollama_tracks_completed_and_success_progress(svc: OllamaService) -> None:
    stream = MagicMock()
    records = "\n".join(
        [
            json.dumps({"status": "pulling", "completed": 500000, "digest": "sha256:abc"}),
            json.dumps({"status": "pulling", "completed": 1000000, "digest": "sha256:abc"}),
            json.dumps({"status": "success"}),
        ]
    )

    async def mock_stream(*args: object, **kwargs: object):  # type: ignore[misc]
        yield FetchResult(status_code=200, data=records)

    with patch("server.services.ollama_service.stream_fetch_from", side_effect=mock_stream):
        await svc._download_with_ollama(stream, "http://localhost:11434", "llama3", "1GB")  # pyright: ignore[reportPrivateUsage]

    assert stream.emit.call_count >= 2


@pytest.mark.asyncio
async def test_download_with_downloader_uses_dest_path_parent_as_dir(svc: OllamaService, deps: dict[str, Any], tmp_path: Path) -> None:
    stream = MagicMock()
    dest_path = tmp_path / "subdir" / "model.gguf"

    async def empty_gen(*args: object, **kwargs: object):  # type: ignore[misc]
        return
        yield

    deps["model_downloader"].download.return_value = empty_gen()

    with patch.object(svc, "_get_working_dir", return_value=tmp_path):  # pyright: ignore[reportPrivateUsage]
        await svc._download_with_downloader(  # pyright: ignore[reportPrivateUsage]
            stream, "http://localhost:11434", "https://example.com/model.gguf", "my-model", "1GB", 0, 1, dest_path=dest_path
        )

    call_dir = deps["model_downloader"].download.call_args[0][1]
    assert call_dir == dest_path.parent


@pytest.mark.asyncio
async def test_download_with_downloader_uses_model_subdir_for_non_gguf_url(
    svc: OllamaService, deps: dict[str, Any], tmp_path: Path
) -> None:
    stream = MagicMock()

    async def empty_gen(*args: object, **kwargs: object):  # type: ignore[misc]
        return
        yield

    deps["model_downloader"].download.return_value = empty_gen()

    with patch.object(svc, "_get_working_dir", return_value=tmp_path):  # pyright: ignore[reportPrivateUsage]
        await svc._download_with_downloader(  # pyright: ignore[reportPrivateUsage]
            stream, "http://localhost:11434", "https://example.com/model-repo", "my-model", "1GB", 0, 1
        )

    call_dir = deps["model_downloader"].download.call_args[0][1]
    assert str(call_dir).endswith("model")


@pytest.mark.asyncio
async def test_download_with_downloader_handles_download_packets(svc: OllamaService, deps: dict[str, Any], tmp_path: Path) -> None:
    stream = MagicMock()

    async def gen_packets(*args: object, **kwargs: object):  # type: ignore[misc]
        yield PreDownloadPacket(file_bytes_size=1000000)
        yield DownloadedPacket(downloaded_bytes_size=500000)
        yield DownloadedPacket(downloaded_bytes_size=500000)

    deps["model_downloader"].download.side_effect = gen_packets

    with patch.object(svc, "_get_working_dir", return_value=tmp_path):  # pyright: ignore[reportPrivateUsage]
        await svc._download_with_downloader(  # pyright: ignore[reportPrivateUsage]
            stream, "http://localhost:11434", "https://example.com/model.gguf", "my-model", "1MB", 0, 1
        )

    assert stream.emit.call_count >= 2


@pytest.mark.asyncio
async def test_save_modelfile_removes_legacy_modelfile_at_old_path(svc: OllamaService, tmp_path: Path) -> None:
    old_dir = tmp_path / "main" / "custom" / "my-model"
    old_dir.mkdir(parents=True)
    old_path = old_dir / "Modelfile"
    old_path.write_text("FROM old-model")

    with patch.object(svc, "_get_working_dir", return_value=tmp_path):  # pyright: ignore[reportPrivateUsage]
        await svc.save_modelfile("my-model", "default", "FROM new-model")

    assert not old_path.exists()


@pytest.mark.asyncio
async def test_save_modelfile_overwrites_when_content_differs_and_parent_exists(svc: OllamaService, tmp_path: Path) -> None:
    local_path = tmp_path / "main" / "custom" / "my-model" / "default" / "Modelfile"
    local_path.parent.mkdir(parents=True)
    local_path.write_text("FROM old-model")

    with patch.object(svc, "_get_working_dir", return_value=tmp_path):  # pyright: ignore[reportPrivateUsage]
        await svc.save_modelfile("my-model", "default", "FROM new-model")

    assert local_path.read_text() == "FROM new-model"


@pytest.mark.asyncio
async def test_download_model_or_set_progress_breaks_on_non_download_chunk(svc: OllamaService) -> None:
    existing_stream: Stream[StreamChunkProgress] = Stream()  # type: ignore[type-arg]
    non_download_chunk = {"type": "complete", "stage": "install", "value": 1, "data": {}}
    existing_stream.emit(non_download_chunk)  # type: ignore[arg-type]
    existing_stream.close()
    svc.models_download_progress["llama3"] = existing_stream  # type: ignore[assignment]

    output_stream = MagicMock()

    await svc._download_model_or_set_progress(  # pyright: ignore[reportPrivateUsage]
        output_stream, "http://localhost:11434", "llama3", "llama3", "1GB"
    )

    assert output_stream.emit.call_count == 0


@pytest.mark.asyncio
async def test_install_from_modelfile_skips_download_when_local_path_exists(svc: OllamaService, tmp_path: Path) -> None:
    model_id = "my-model"
    url = "https://example.com/model.gguf"
    file_name = hashlib.sha1(url.encode()).hexdigest() + ".gguf"
    local_path = tmp_path / "main" / "custom" / model_id / file_name
    local_path.parent.mkdir(parents=True)
    local_path.write_bytes(b"")
    model = OllamaModel(id=model_id, size="1GB", type="llm", hash="abc", context=4096)
    instance_info = _make_installed_info(svc)
    modelfile_recipe = f"FROM {url}\nPARAMETER num_ctx 4096"

    with (
        patch.object(svc, "_get_working_dir", return_value=tmp_path),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock) as mock_dl,  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "save_modelfile", new_callable=AsyncMock, return_value="/root/.ollama/Modelfile"),
        patch.object(svc, "create_model_from_modelfile", new_callable=AsyncMock, return_value=""),
        patch("server.services.ollama_service.detect_context_window_from_path", new_callable=AsyncMock, return_value=None),
    ):
        await svc._install_from_modelfile(  # pyright: ignore[reportPrivateUsage]
            MagicMock(), model_id, modelfile_recipe, model_id, instance_info, model
        )

    assert mock_dl.call_count == 0


@pytest.mark.asyncio
async def test_install_from_modelfile_does_not_detect_context_for_adapter_https_line(svc: OllamaService, tmp_path: Path) -> None:
    model_id = "my-model"
    model = OllamaModel(id=model_id, size="1GB", type="llm", hash="abc", context=4096)
    instance_info = _make_installed_info(svc)

    with (
        patch.object(svc, "_get_working_dir", return_value=tmp_path),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "save_modelfile", new_callable=AsyncMock, return_value="/root/.ollama/Modelfile"),
        patch.object(svc, "create_model_from_modelfile", new_callable=AsyncMock, return_value=""),
        patch("server.services.ollama_service.detect_context_window_from_path", new_callable=AsyncMock, return_value=9999) as mock_detect,
    ):
        result = await svc._install_from_modelfile(  # pyright: ignore[reportPrivateUsage]
            MagicMock(), model_id, "ADAPTER https://example.com/adapter.bin", model_id, instance_info, model
        )

    # ADAPTER line does not trigger context detection
    assert mock_detect.call_count == 0
    assert result is None


@pytest.mark.asyncio
async def test_install_from_modelfile_detects_context_for_local_relative_from_path(svc: OllamaService, tmp_path: Path) -> None:
    model_id = "my-model"
    model = OllamaModel(id=model_id, size="1GB", type="llm", hash="abc", context=4096)
    instance_info = _make_installed_info(svc)
    local_file = tmp_path / "main" / "model.gguf"
    local_file.parent.mkdir(parents=True)
    local_file.write_bytes(b"")

    with (
        patch.object(svc, "_get_working_dir", return_value=tmp_path),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "save_modelfile", new_callable=AsyncMock, return_value="/root/.ollama/Modelfile"),
        patch.object(svc, "create_model_from_modelfile", new_callable=AsyncMock, return_value=""),
        patch("server.services.ollama_service.detect_context_window_from_path", new_callable=AsyncMock, return_value=8192) as mock_detect,
    ):
        result = await svc._install_from_modelfile(  # pyright: ignore[reportPrivateUsage]
            MagicMock(), model_id, "FROM ./model.gguf", model_id, instance_info, model
        )

    assert mock_detect.call_count == 1
    assert result == 8192


@pytest.mark.asyncio
async def test_install_from_modelfile_raises_400_on_client_connector_error(svc: OllamaService, tmp_path: Path) -> None:
    model_id = "my-model"
    model = OllamaModel(id=model_id, size="1GB", type="llm", hash="abc", context=4096)
    instance_info = _make_installed_info(svc)
    connector_error = aiohttp.ClientConnectorError(MagicMock(), OSError("connection refused"))

    with (
        patch.object(svc, "_get_working_dir", return_value=tmp_path),  # pyright: ignore[reportPrivateUsage]
        patch.object(OllamaService, "is_model_installed", new_callable=AsyncMock, side_effect=connector_error),
        pytest.raises(HTTPException) as exc_info,
    ):
        await svc._install_from_modelfile(  # pyright: ignore[reportPrivateUsage]
            MagicMock(), model_id, "FROM llama3", model_id, instance_info, model
        )

    assert exc_info.value.status_code == 400
    assert "Cannot connect" in exc_info.value.detail


@pytest.mark.asyncio
async def test_install_from_modelfile_reraises_http_exception_with_original_status(svc: OllamaService, tmp_path: Path) -> None:
    model_id = "my-model"
    model = OllamaModel(id=model_id, size="1GB", type="llm", hash="abc", context=4096)
    instance_info = _make_installed_info(svc)

    with (
        patch.object(svc, "_get_working_dir", return_value=tmp_path),  # pyright: ignore[reportPrivateUsage]
        patch.object(OllamaService, "is_model_installed", new_callable=AsyncMock, return_value=True),
        patch.object(svc, "save_modelfile", new_callable=AsyncMock, side_effect=HTTPException(status_code=422, detail="custom")),
        pytest.raises(HTTPException) as exc_info,
    ):
        await svc._install_from_modelfile(  # pyright: ignore[reportPrivateUsage]
            MagicMock(), model_id, "FROM llama3", model_id, instance_info, model
        )

    assert exc_info.value.status_code == 422


@pytest.mark.asyncio
async def test_install_model_initializes_models_dict_for_instance(svc: OllamaService) -> None:
    installed = _make_installed_info(svc)
    svc.instances_info["default"].installed = installed
    del svc.models["default"]

    with pytest.raises(HTTPException):
        await svc._install_model("default", "any-model", InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]

    assert "default" in svc.models


@pytest.mark.asyncio
async def test_install_model_appends_context_to_existing_modelfile(svc: OllamaService) -> None:
    installed = _make_installed_info(svc)
    svc.instances_info["default"].installed = installed
    svc.models["default"]["test-llm"] = OllamaModel(
        id="test-llm",
        size="1GB",
        type="llm",
        hash="abc",
        context=4096,
        modelfile="FROM base-model\nPARAMETER temperature 0.7",
    )

    with patch.object(svc, "_install_from_modelfile", new_callable=AsyncMock, return_value=None) as mock_install:  # pyright: ignore[reportPrivateUsage]
        promise = await svc._install_model(  # pyright: ignore[reportPrivateUsage]
            "default", "test-llm", InstallModelIn(spec={"context_length": 8192})
        )
        await promise.wait()

    modelfile_recipe: str = mock_install.call_args.args[2]
    assert "num_ctx 8192" in modelfile_recipe
    assert "FROM base-model" in modelfile_recipe


@pytest.mark.asyncio
async def test_install_model_updates_context_from_modelfile_result(svc: OllamaService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info(svc)
    svc.instances_info["default"].installed = installed
    svc.models["default"]["test-llm"] = OllamaModel(
        id="test-llm",
        size="1GB",
        type="llm",
        hash="abc",
        context=4096,
        modelfile="FROM base-model",
    )

    with patch.object(svc, "_install_from_modelfile", new_callable=AsyncMock, return_value=16384):  # pyright: ignore[reportPrivateUsage]
        promise = await svc._install_model("default", "test-llm", InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    register_call = deps["endpoint_registry"].register_chat_completion_as_proxy.call_args
    assert register_call is not None


@pytest.mark.asyncio
async def test_install_model_downloads_when_no_modelfile_and_not_yet_installed(svc: OllamaService) -> None:
    installed = _make_installed_info(svc)
    svc.instances_info["default"].installed = installed
    svc.models["default"]["test-llm"] = OllamaModel(id="test-llm", size="1GB", type="llm", hash="abc", context=4096, modelfile=None)

    with (
        patch.object(OllamaService, "is_model_installed", new_callable=AsyncMock, return_value=False),
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock) as mock_dl,  # pyright: ignore[reportPrivateUsage]
    ):
        promise = await svc._install_model("default", "test-llm", InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert mock_dl.call_count == 1


@pytest.mark.asyncio
async def test_install_model_sends_keep_alive_when_alive_time_set(svc: OllamaService) -> None:
    installed = _make_installed_info(svc)
    svc.instances_info["default"].installed = installed
    svc.models["default"]["test-llm"] = OllamaModel(id="test-llm", size="1GB", type="llm", hash="abc", context=4096)
    keep_alive_calls = []

    async def mock_fetch(url: str, *args: object, **kwargs: object) -> FetchResult:  # type: ignore[misc]
        keep_alive_calls.append(url)
        return FetchResult(status_code=200, data="")

    with patch("server.services.ollama_service.fetch_from", side_effect=mock_fetch):
        promise = await svc._install_model(  # pyright: ignore[reportPrivateUsage]
            "default", "test-llm", InstallModelIn(spec={"alive_time": "30m"})
        )
        await promise.wait()

    assert any("generate" in url for url in keep_alive_calls)


@pytest.mark.asyncio
async def test_install_model_skips_alias_marking_when_hash_is_empty(svc: OllamaService) -> None:
    installed = _make_installed_info(svc)
    svc.instances_info["default"].installed = installed
    svc.models["default"]["test-llm"] = OllamaModel(id="test-llm", size="1GB", type="llm", hash="", context=4096)

    with patch("server.services.ollama_service.fetch_from", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = FetchResult(status_code=200, data="")
        promise = await svc._install_model("default", "test-llm", InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert "test-llm" in svc.models_downloaded


@pytest.mark.asyncio
async def test_uninstall_model_unregisters_embedding_endpoint(svc: OllamaService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info(svc)
    installed.models["test-emb"] = ModelInstalledInfo(
        id="test-emb",
        registered_name="test-emb",
        type="embedding",
        options=InstallModelIn(spec={}),
        registration_id="reg-emb",
        internal_name=None,
    )
    svc.instances_info["default"].installed = installed
    svc.models["default"]["test-emb"] = OllamaModel(id="test-emb", size="500MB", type="embedding", hash="def", context=None)

    await svc._uninstall_model("default", "test-emb", UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert deps["endpoint_registry"].unregister_embeddings.call_count == 1
    assert deps["endpoint_registry"].unregister_embeddings.call_args == call("test-emb", "reg-emb")


@pytest.mark.asyncio
async def test_uninstall_model_unregisters_txt2img_endpoint(svc: OllamaService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info(svc)
    installed.models["test-img"] = ModelInstalledInfo(
        id="test-img",
        registered_name="test-img",
        type="txt2img",
        options=InstallModelIn(spec={}),
        registration_id="reg-img",
        internal_name=None,
    )
    svc.instances_info["default"].installed = installed
    svc.models["default"]["test-img"] = OllamaModel(id="test-img", size="2GB", type="txt2img", hash="ghi", context=None)

    await svc._uninstall_model("default", "test-img", UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert deps["endpoint_registry"].unregister_image_generations.call_count == 1
    assert deps["endpoint_registry"].unregister_image_generations.call_args == call("test-img", "reg-img")


@pytest.mark.asyncio
async def test_uninstall_model_purge_returns_early_when_no_hash(svc: OllamaService) -> None:
    installed = _make_installed_info(svc)
    installed.models["test-llm"] = ModelInstalledInfo(
        id="test-llm",
        registered_name="test-llm",
        type="llm",
        options=InstallModelIn(spec={}),
        registration_id="",
        internal_name=None,
    )
    svc.instances_info["default"].installed = installed
    svc.models["default"]["test-llm"] = OllamaModel(id="test-llm", size="1GB", type="llm", hash="", context=4096)
    svc.models["default"]["other-model"] = OllamaModel(id="other-model", size="1GB", type="llm", hash="", context=4096)
    svc.models_downloaded["test-llm"] = DownloadedInfo()
    with (
        patch("server.services.ollama_service.fetch_from", new_callable=AsyncMock) as mock_fetch,
        patch.object(svc, "remove_modelfile", new_callable=AsyncMock),
    ):
        mock_fetch.return_value = FetchResult(status_code=200, data="")

        await svc._uninstall_model("default", "test-llm", UninstallModelIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert "test-llm" not in svc.models_downloaded
    assert mock_fetch.call_count == 1  # only the main model delete, no alias deletes


@pytest.mark.asyncio
async def test_uninstall_model_purge_also_removes_aliases_sharing_same_hash(svc: OllamaService) -> None:
    installed = _make_installed_info(svc)
    installed.models["test-llm"] = ModelInstalledInfo(
        id="test-llm",
        registered_name="test-llm",
        type="llm",
        options=InstallModelIn(spec={}),
        registration_id="",
        internal_name=None,
    )
    svc.instances_info["default"].installed = installed
    svc.models["default"]["test-llm"] = OllamaModel(id="test-llm", size="1GB", type="llm", hash="shared-hash", context=4096)
    svc.models["default"]["test-llm-alias"] = OllamaModel(id="test-llm-alias", size="1GB", type="llm", hash="shared-hash", context=4096)
    svc.models_downloaded["test-llm"] = DownloadedInfo()
    svc.models_downloaded["test-llm-alias"] = DownloadedInfo()
    with (
        patch("server.services.ollama_service.fetch_from", new_callable=AsyncMock) as mock_fetch,
        patch.object(svc, "remove_modelfile", new_callable=AsyncMock),
    ):
        mock_fetch.return_value = FetchResult(status_code=200, data="")

        await svc._uninstall_model("default", "test-llm", UninstallModelIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert "test-llm-alias" not in svc.models_downloaded
    assert mock_fetch.call_count >= 2  # main model + alias


@pytest.mark.asyncio
async def test_uninstall_model_purge_skips_alias_delete_from_downloaded_when_not_present(svc: OllamaService) -> None:
    installed = _make_installed_info(svc)
    installed.models["test-llm"] = ModelInstalledInfo(
        id="test-llm",
        registered_name="test-llm",
        type="llm",
        options=InstallModelIn(spec={}),
        registration_id="",
        internal_name=None,
    )
    svc.instances_info["default"].installed = installed
    svc.models["default"]["test-llm"] = OllamaModel(id="test-llm", size="1GB", type="llm", hash="shared-hash", context=4096)
    svc.models["default"]["test-llm-alias"] = OllamaModel(id="test-llm-alias", size="1GB", type="llm", hash="shared-hash", context=4096)
    svc.models_downloaded["test-llm"] = DownloadedInfo()
    # alias is intentionally NOT in models_downloaded
    with (
        patch("server.services.ollama_service.fetch_from", new_callable=AsyncMock) as mock_fetch,
        patch.object(svc, "remove_modelfile", new_callable=AsyncMock),
    ):
        mock_fetch.return_value = FetchResult(status_code=200, data="")

        await svc._uninstall_model("default", "test-llm", UninstallModelIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert "test-llm" not in svc.models_downloaded
    assert mock_fetch.call_count >= 2  # main model + alias


@pytest.mark.asyncio
async def test_download_with_ollama_skips_tracking_when_size_unparseable(svc: OllamaService) -> None:
    stream = MagicMock()
    records = json.dumps({"status": "pulling", "completed": 500000, "digest": "sha256:abc"})

    async def mock_stream(*args: object, **kwargs: object):  # type: ignore[misc]
        yield FetchResult(status_code=200, data=records)

    with patch("server.services.ollama_service.stream_fetch_from", side_effect=mock_stream):
        await svc._download_with_ollama(stream, "http://localhost:11434", "llama3", "unparseable-size")  # pyright: ignore[reportPrivateUsage]
    # start + end progress emits only, no per-record emit
    assert stream.emit.call_count == 2


@pytest.mark.asyncio
async def test_download_with_ollama_emits_progress_for_status_only_records(svc: OllamaService) -> None:
    stream = MagicMock()
    records = json.dumps({"status": "pulling manifest"})

    async def mock_stream(*args: object, **kwargs: object):  # type: ignore[misc]
        yield FetchResult(status_code=200, data=records)

    with patch("server.services.ollama_service.stream_fetch_from", side_effect=mock_stream):
        await svc._download_with_ollama(stream, "http://localhost:11434", "llama3", "1GB")  # pyright: ignore[reportPrivateUsage]

    assert stream.emit.call_count >= 2


@pytest.mark.asyncio
async def test_download_with_downloader_ignores_zero_byte_download_packet(svc: OllamaService, deps: dict[str, Any], tmp_path: Path) -> None:
    stream = MagicMock()

    async def gen_packets(*args: object, **kwargs: object):  # type: ignore[misc]
        yield DownloadedPacket(downloaded_bytes_size=0)

    deps["model_downloader"].download.side_effect = gen_packets

    with patch.object(svc, "_get_working_dir", return_value=tmp_path):  # pyright: ignore[reportPrivateUsage]
        await svc._download_with_downloader(  # pyright: ignore[reportPrivateUsage]
            stream, "http://localhost:11434", "https://example.com/model.gguf", "my-model", "1MB", 0, 1
        )

    assert stream.emit.call_count == 2  # only start and end


@pytest.mark.asyncio
async def test_download_with_downloader_ignores_pre_download_packet_with_zero_size(
    svc: OllamaService, deps: dict[str, Any], tmp_path: Path
) -> None:
    stream = MagicMock()

    async def gen_packets(*args: object, **kwargs: object):  # type: ignore[misc]
        yield PreDownloadPacket(file_bytes_size=0)

    deps["model_downloader"].download.side_effect = gen_packets

    with patch.object(svc, "_get_working_dir", return_value=tmp_path):  # pyright: ignore[reportPrivateUsage]
        await svc._download_with_downloader(  # pyright: ignore[reportPrivateUsage]
            stream, "http://localhost:11434", "https://example.com/model.gguf", "my-model", "1MB", 0, 1
        )

    assert stream.emit.call_count == 2  # only start and end


@pytest.mark.asyncio
async def test_install_from_modelfile_does_not_detect_context_for_relative_non_from_line(svc: OllamaService, tmp_path: Path) -> None:
    model_id = "my-model"
    model = OllamaModel(id=model_id, size="1GB", type="llm", hash="abc", context=4096)
    instance_info = _make_installed_info(svc)
    local_file = tmp_path / "main" / "adapter.bin"
    local_file.parent.mkdir(parents=True)
    local_file.write_bytes(b"")

    with (
        patch.object(svc, "_get_working_dir", return_value=tmp_path),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "save_modelfile", new_callable=AsyncMock, return_value="/root/.ollama/Modelfile"),
        patch.object(svc, "create_model_from_modelfile", new_callable=AsyncMock, return_value=""),
        patch("server.services.ollama_service.detect_context_window_from_path", new_callable=AsyncMock, return_value=8192) as mock_detect,
    ):
        result = await svc._install_from_modelfile(  # pyright: ignore[reportPrivateUsage]
            MagicMock(), model_id, "ADAPTER ./adapter.bin", model_id, instance_info, model
        )

    assert mock_detect.call_count == 0
    assert result is None


def test_docker_path_to_host_relative_path(svc: OllamaService, tmp_path: Path) -> None:
    svc._OLLAMA_DOCKER_ROOT = "/root/.ollama"  # pyright: ignore[reportPrivateUsage]

    with patch.object(svc, "_get_working_dir", return_value=tmp_path):  # pyright: ignore[reportPrivateUsage]
        result = svc._docker_path_to_host("./myfile")  # pyright: ignore[reportPrivateUsage]

    assert result == tmp_path / "main" / "myfile"


def test_docker_path_to_host_absolute_ollama_root_subpath(svc: OllamaService, tmp_path: Path) -> None:
    svc._OLLAMA_DOCKER_ROOT = "/root/.ollama"  # pyright: ignore[reportPrivateUsage]

    with patch.object(svc, "_get_working_dir", return_value=tmp_path):  # pyright: ignore[reportPrivateUsage]
        result = svc._docker_path_to_host("/root/.ollama/models/foo")  # pyright: ignore[reportPrivateUsage]

    assert result == tmp_path / "main" / "models/foo"


def test_docker_path_to_host_exact_root(svc: OllamaService, tmp_path: Path) -> None:
    svc._OLLAMA_DOCKER_ROOT = "/root/.ollama"  # pyright: ignore[reportPrivateUsage]

    with patch.object(svc, "_get_working_dir", return_value=tmp_path):  # pyright: ignore[reportPrivateUsage]
        result = svc._docker_path_to_host("/root/.ollama")  # pyright: ignore[reportPrivateUsage]

    assert result == tmp_path / "main"


def test_docker_path_to_host_other_absolute_path_returns_none(svc: OllamaService, tmp_path: Path) -> None:
    svc._OLLAMA_DOCKER_ROOT = "/root/.ollama"  # pyright: ignore[reportPrivateUsage]

    with patch.object(svc, "_get_working_dir", return_value=tmp_path):  # pyright: ignore[reportPrivateUsage]
        result = svc._docker_path_to_host("/other/path")  # pyright: ignore[reportPrivateUsage]

    assert result is None


def test_docker_path_to_host_no_prefix_returns_none(svc: OllamaService, tmp_path: Path) -> None:
    svc._OLLAMA_DOCKER_ROOT = "/root/.ollama"  # pyright: ignore[reportPrivateUsage]

    with patch.object(svc, "_get_working_dir", return_value=tmp_path):  # pyright: ignore[reportPrivateUsage]
        result = svc._docker_path_to_host("noprefix")  # pyright: ignore[reportPrivateUsage]

    assert result is None


@pytest.mark.asyncio
async def test_local_file_size_bytes_maps_to_real_file(svc: OllamaService, tmp_path: Path) -> None:
    target = tmp_path / "main" / "myfile.bin"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"hello world")

    with patch.object(svc, "_get_working_dir", return_value=tmp_path):  # pyright: ignore[reportPrivateUsage]
        result = await svc._local_file_size_bytes("./myfile.bin", "default")  # pyright: ignore[reportPrivateUsage]

    assert result == len(b"hello world")


@pytest.mark.asyncio
async def test_local_file_size_bytes_maps_but_file_not_exist(svc: OllamaService, tmp_path: Path) -> None:
    with patch.object(svc, "_get_working_dir", return_value=tmp_path):  # pyright: ignore[reportPrivateUsage]
        result = await svc._local_file_size_bytes("./nonexistent.bin", "default")  # pyright: ignore[reportPrivateUsage]

    assert result is None


@pytest.mark.asyncio
async def test_local_file_size_bytes_no_docker_path_and_not_installed(svc: OllamaService, tmp_path: Path) -> None:
    # Path is outside ollama root and instance not installed → None
    with patch.object(svc, "_get_working_dir", return_value=tmp_path):  # pyright: ignore[reportPrivateUsage]
        result = await svc._local_file_size_bytes("/other/path/model.bin", "default")  # pyright: ignore[reportPrivateUsage]

    assert result is None


@pytest.mark.asyncio
async def test_local_file_size_bytes_no_docker_path_but_installed_uses_docker_exec(
    svc: OllamaService, deps: dict[str, Any], tmp_path: Path
) -> None:
    installed = _make_installed_info(svc)
    svc.instances_info["default"].installed = installed
    deps["docker_service"].get_docker_compose_file_path.return_value = "/path/to/compose.yaml"
    deps["docker_service"].run_command_docker_compose = AsyncMock(return_value="12345\n")

    with patch.object(svc, "_get_working_dir", return_value=tmp_path):  # pyright: ignore[reportPrivateUsage]
        result = await svc._local_file_size_bytes("/other/path/model.bin", "default")  # pyright: ignore[reportPrivateUsage]

    assert result == 12345


@pytest.mark.asyncio
async def test_local_file_size_bytes_exception_in_docker_exec_returns_none(
    svc: OllamaService, deps: dict[str, Any], tmp_path: Path
) -> None:
    installed = _make_installed_info(svc)
    svc.instances_info["default"].installed = installed
    deps["docker_service"].get_docker_compose_file_path.return_value = "/path/to/compose.yaml"
    deps["docker_service"].run_command_docker_compose = AsyncMock(side_effect=RuntimeError("stat failed"))

    with patch.object(svc, "_get_working_dir", return_value=tmp_path):  # pyright: ignore[reportPrivateUsage]
        result = await svc._local_file_size_bytes("/other/path/model.bin", "default")  # pyright: ignore[reportPrivateUsage]

    assert result is None


@pytest.mark.asyncio
async def test_fetch_ref_bytes_absolute_path_calls_local(svc: OllamaService) -> None:
    with patch.object(svc, "_local_file_size_bytes", new=AsyncMock(return_value=999)) as mock_local:  # pyright: ignore[reportPrivateUsage]
        result = await svc._fetch_ref_bytes("/foo", "default")  # pyright: ignore[reportPrivateUsage]

    assert result == 999
    mock_local.assert_called_once_with("/foo", "default")


@pytest.mark.asyncio
async def test_fetch_ref_bytes_relative_path_calls_local(svc: OllamaService) -> None:
    with patch.object(svc, "_local_file_size_bytes", new=AsyncMock(return_value=777)) as mock_local:  # pyright: ignore[reportPrivateUsage]
        result = await svc._fetch_ref_bytes("./foo", "default")  # pyright: ignore[reportPrivateUsage]

    assert result == 777
    mock_local.assert_called_once_with("./foo", "default")


@pytest.mark.asyncio
async def test_fetch_ref_bytes_remote_ref_calls_fetch_ollama_ref_bytes(svc: OllamaService) -> None:
    with patch("server.services.ollama_service.fetch_ollama_ref_bytes", new=AsyncMock(return_value=5000)) as mock_remote:
        result = await svc._fetch_ref_bytes("llama3", "default")  # pyright: ignore[reportPrivateUsage]

    assert result == 5000
    mock_remote.assert_called_once_with("llama3")


@pytest.mark.asyncio
async def test_resolve_custom_model_size_modelfile_with_from(svc: OllamaService) -> None:
    with patch.object(svc, "_fetch_ref_bytes", new=AsyncMock(return_value=1000)):  # pyright: ignore[reportPrivateUsage]
        result = await svc._resolve_custom_model_size({"modelfile": "FROM llama3"})  # pyright: ignore[reportPrivateUsage]

    assert result == "1000.0 B"


@pytest.mark.asyncio
async def test_resolve_custom_model_size_modelfile_no_refs(svc: OllamaService) -> None:
    result = await svc._resolve_custom_model_size({"modelfile": "PARAMETER num_ctx 4096"})  # pyright: ignore[reportPrivateUsage]

    assert result is None


@pytest.mark.asyncio
async def test_resolve_custom_model_size_no_modelfile_uses_id(svc: OllamaService) -> None:
    with patch.object(svc, "_fetch_ref_bytes", new=AsyncMock(return_value=2048)):  # pyright: ignore[reportPrivateUsage]
        result = await svc._resolve_custom_model_size({"id": "llama3"})  # pyright: ignore[reportPrivateUsage]

    assert result == "2.0 KB"


@pytest.mark.asyncio
async def test_resolve_custom_model_size_modelfile_partial_on_exception(svc: OllamaService) -> None:
    mock = AsyncMock(side_effect=[Exception("fail"), 512])
    with patch.object(svc, "_fetch_ref_bytes", new=mock):  # pyright: ignore[reportPrivateUsage]
        result = await svc._resolve_custom_model_size({"modelfile": "FROM llama3\nADAPTER ./adapter.bin"})  # pyright: ignore[reportPrivateUsage]

    assert result == "512.0 B"


@pytest.mark.asyncio
async def test_resolve_custom_model_size_modelfile_all_refs_return_none(svc: OllamaService) -> None:
    with patch.object(svc, "_fetch_ref_bytes", new=AsyncMock(return_value=None)):  # pyright: ignore[reportPrivateUsage]
        result = await svc._resolve_custom_model_size({"modelfile": "FROM llama3"})  # pyright: ignore[reportPrivateUsage]

    assert result is None


@pytest.mark.asyncio
async def test_resolve_custom_model_size_exception_in_outer_try(svc: OllamaService) -> None:
    # Force an exception in the outer try by making _fetch_ref_bytes raise AttributeError
    with patch.object(svc, "_fetch_ref_bytes", side_effect=AttributeError("bad")):  # pyright: ignore[reportPrivateUsage]
        result = await svc._resolve_custom_model_size({"id": "llama3"})  # pyright: ignore[reportPrivateUsage]

    assert result is None


@pytest.mark.asyncio
async def test_get_memory_load_processes_blob_line(svc: OllamaService) -> None:
    installed = _make_installed_info(svc)
    svc.instances_info["default"].installed = installed
    # Line with a blob reference but no memory line (covers MODEL_BLOB_RE match + MEMORY_LINE_RE no-match branch)
    raw = "blobs/sha256-abc123def456\nsome other line without memory info\n"

    with (
        patch.object(svc, "_get_docker_logs", new_callable=AsyncMock, return_value=raw),
        patch.object(svc, "resolve_blob", new_callable=AsyncMock, return_value="llama3"),
    ):
        result = await svc.get_memory_load("default")

    assert isinstance(result, MemoryLoadOut)
    assert result.sessions == []


@pytest.mark.asyncio
async def test_get_vram_from_logs_with_preloaded_memory_load(svc: OllamaService) -> None:
    memory_load = MemoryLoadOut(sessions=[MemoryLoadSession(model="llama3", components=[], total="8.0 GiB")])

    result = await svc._get_vram_from_logs("default", "llama3", memory_load=memory_load)  # pyright: ignore[reportPrivateUsage]

    assert result == pytest.approx(8.0, abs=0.1)


@pytest.mark.asyncio
async def test_resolve_vram_info_cache_hit(svc: OllamaService) -> None:
    installed = _make_installed_info(svc)
    svc.instances_info["default"].installed = installed
    svc._vram_cache[("default", "llama3")] = 6.0  # pyright: ignore[reportPrivateUsage]
    loaded_info = {"llama3": 4096}

    is_loaded, vram = await svc._resolve_vram_info(  # pyright: ignore[reportPrivateUsage]
        "default", "llama3", None, loaded_info, {}, "http://localhost:11434", None
    )

    assert is_loaded is True
    assert vram == 6.0


@pytest.mark.asyncio
async def test_resolve_vram_info_vram_estimate_none_not_cached(svc: OllamaService) -> None:
    installed = _make_installed_info(svc)
    svc.instances_info["default"].installed = installed
    loaded_info = {"llama3": 4096}

    with (
        patch.object(svc, "_get_vram_from_logs", new_callable=AsyncMock, return_value=None),
        patch.object(svc, "_get_vram_estimate", new_callable=AsyncMock, return_value=None),
    ):
        is_loaded, vram = await svc._resolve_vram_info(  # pyright: ignore[reportPrivateUsage]
            "default", "llama3", None, loaded_info, {}, "http://localhost:11434", None
        )

    assert is_loaded is True
    assert vram is None
    assert ("default", "llama3") not in svc._vram_cache  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_resolve_vram_info_not_loaded_returns_formula_estimate(svc: OllamaService) -> None:
    installed = _make_installed_info(svc)
    svc.instances_info["default"].installed = installed
    svc._vram_cache[("default", "llama3")] = 6.0  # pyright: ignore[reportPrivateUsage]
    loaded_info: dict[str, int] = {}  # model not in VRAM

    with patch.object(svc, "_get_vram_estimate", new_callable=AsyncMock, return_value=4.5) as mock_estimate:
        is_loaded, vram = await svc._resolve_vram_info(  # pyright: ignore[reportPrivateUsage]
            "default", "llama3", 4096, loaded_info, {}, "http://localhost:11434", None
        )

    assert is_loaded is False
    assert vram == pytest.approx(4.5)
    assert ("default", "llama3") not in svc._vram_cache  # pyright: ignore[reportPrivateUsage]
    mock_estimate.assert_awaited_once()


@pytest.mark.asyncio
async def test_resolve_vram_info_not_loaded_estimate_none(svc: OllamaService) -> None:
    installed = _make_installed_info(svc)
    svc.instances_info["default"].installed = installed
    loaded_info: dict[str, int] = {}

    with patch.object(svc, "_get_vram_estimate", new_callable=AsyncMock, return_value=None):
        is_loaded, vram = await svc._resolve_vram_info(  # pyright: ignore[reportPrivateUsage]
            "default", "llama3", None, loaded_info, {}, "http://localhost:11434", None
        )

    assert is_loaded is False
    assert vram is None


@pytest.mark.asyncio
async def test_get_model_with_loaded_info_none(svc: OllamaService) -> None:
    installed = _make_installed_info(svc)
    svc.instances_info["default"].installed = installed
    model_id = next(iter(svc.models["default"]))

    with (
        patch.object(svc, "get_loaded_model_info", new_callable=AsyncMock, return_value=None),
        patch.object(svc, "_get_model_sizes", new_callable=AsyncMock, return_value={}),
    ):
        result = await svc.get_model("default", model_id)

    assert result.id == model_id


@pytest.mark.asyncio
async def test_download_with_ollama_interleaved_digests_progress_never_decreases(svc: OllamaService) -> None:
    # Regression test for issue #464 root cause 2:
    # Ollama streams progress records for multiple layers in parallel, interleaving digests.
    # The old single-(last_diggest, last_value) tracker treated every digest switch as a
    # fresh start (increment = full completed value), overcounting bytes and pushing percentage
    # above 1.0 prematurely — then a correction event caused a visible regression.
    stream = MagicMock()
    records = [
        {"status": "pulling", "digest": "sha256:aaa", "completed": 1_000_000_000},  # layer A: 1 GB
        {"status": "pulling", "digest": "sha256:bbb", "completed": 500_000_000},  # layer B: 0.5 GB
        {"status": "pulling", "digest": "sha256:aaa", "completed": 2_000_000_000},  # layer A delta: +1 GB
        {"status": "pulling", "digest": "sha256:bbb", "completed": 1_000_000_000},  # layer B delta: +0.5 GB
        {"status": "success"},
    ]
    data = "\n".join(json.dumps(r) for r in records)

    async def mock_stream(*args: object, **kwargs: object):  # type: ignore[misc]
        yield FetchResult(status_code=200, data=data)

    with patch("server.services.ollama_service.stream_fetch_from", side_effect=mock_stream):
        # Model size matches total bytes: 2 GB (layer A) + 1 GB (layer B) = 3 GB
        await svc._download_with_ollama(stream, "http://localhost:11434", "llama3", "3GB")  # pyright: ignore[reportPrivateUsage]

    progress_values = [
        call.args[0]["value"]
        for call in stream.emit.call_args_list
        if isinstance(call.args[0], dict) and call.args[0].get("type") == "progress"
    ]
    assert progress_values, "no progress events emitted"
    # Progress must be monotonically non-decreasing
    for prev, curr in itertools.pairwise(progress_values):
        assert curr >= prev, f"progress went backward: {prev} → {curr}"
    # Final progress before the explicit 1.0 must not have exceeded 1.0
    assert all(v <= 1.0 for v in progress_values)


@pytest.mark.asyncio
async def test_download_with_ollama_duplicate_completed_value_not_double_counted(svc: OllamaService) -> None:
    # Covers the increment == 0 branch: same completed value sent twice for a digest must not
    # add anything to progress (increment = max(0, value - last_values[digest]) == 0).
    stream = MagicMock()
    records = [
        {"status": "pulling", "digest": "sha256:aaa", "completed": 500_000_000},
        {"status": "pulling", "digest": "sha256:aaa", "completed": 500_000_000},  # duplicate — increment=0
        {"status": "success"},
    ]
    data = "\n".join(json.dumps(r) for r in records)

    async def mock_stream(*args: object, **kwargs: object):  # type: ignore[misc]
        yield FetchResult(status_code=200, data=data)

    with patch("server.services.ollama_service.stream_fetch_from", side_effect=mock_stream):
        await svc._download_with_ollama(stream, "http://localhost:11434", "llama3", "1GB")  # pyright: ignore[reportPrivateUsage]

    progress_values = [
        call.args[0]["value"]
        for call in stream.emit.call_args_list
        if isinstance(call.args[0], dict) and call.args[0].get("type") == "progress"
    ]
    assert progress_values, "no progress events emitted"
    # Both records for sha256:aaa report the same completed value — second must not advance progress.
    for prev, curr in itertools.pairwise(progress_values):
        assert curr >= prev, f"progress went backward: {prev} → {curr}"


@pytest.mark.asyncio
async def test_download_with_ollama_interleaved_digests_correct_total(svc: OllamaService) -> None:
    # Each digest's contribution should be counted exactly once — no overcounting.
    stream = MagicMock()
    records = [
        {"status": "pulling", "digest": "sha256:aaa", "completed": 1_000_000_000},
        {"status": "pulling", "digest": "sha256:bbb", "completed": 500_000_000},
        {"status": "pulling", "digest": "sha256:aaa", "completed": 2_000_000_000},
        {"status": "pulling", "digest": "sha256:bbb", "completed": 1_000_000_000},
    ]
    data = "\n".join(json.dumps(r) for r in records)

    async def mock_stream(*args: object, **kwargs: object):  # type: ignore[misc]
        yield FetchResult(status_code=200, data=data)

    with patch("server.services.ollama_service.stream_fetch_from", side_effect=mock_stream):
        await svc._download_with_ollama(stream, "http://localhost:11434", "llama3", "3GB")  # pyright: ignore[reportPrivateUsage]

    # The last progress event emitted before the final 1.0 should be ≤ 1.0.
    # With overcounting the value would exceed 1.0 (clamped to 1.0 by get_percentage),
    # but the intermediate events would have hit 1.0 too early.
    progress_values = [
        call.args[0]["value"]
        for call in stream.emit.call_args_list
        if isinstance(call.args[0], dict) and call.args[0].get("type") == "progress" and call.args[0]["value"] < 1.0
    ]
    # There should be at least one intermediate event below 1.0 (not immediately saturated)
    assert progress_values, "progress jumped straight to 1.0 — overcounting likely"
