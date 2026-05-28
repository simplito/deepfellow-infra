# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from server.models.api import CreateSpeechRequest
from server.models.models import InstallModelIn, ListModelsFilters, UninstallModelIn
from server.models.services import InstallServiceIn, UninstallServiceIn
from server.services.base2_service import Instance, InstanceConfig
from server.services.coqui_service import (
    CoquiCmdOptions,
    CoquiOptions,
    CoquiService,
    DownloadedInfo,
    InstalledInfo,
    ModelInstalledInfo,
    _const,  # pyright: ignore[reportPrivateUsage]
    _create_handler,  # pyright: ignore[reportPrivateUsage]
)
from server.utils.hardware import NvidiaGpuInfo


def _make_installed_info() -> InstalledInfo:
    return InstalledInfo(
        models={},
        options=InstallServiceIn(spec={}),
        parsed_options=CoquiOptions(),
    )


def _make_model_installed_info(model_id: str = "test-model", registration_id: str = "reg-1") -> ModelInstalledInfo:
    docker = MagicMock()
    docker.name = f"df-coqui-{model_id}"
    return ModelInstalledInfo(
        id=model_id,
        type="tts",
        registered_name=model_id,
        options=InstallModelIn(spec={}),
        docker=docker,
        container_host="localhost",
        container_port=5002,
        docker_exposed_port=5002,
        registration_id=registration_id,
    )


@pytest.fixture
def deps() -> dict[str, Any]:
    hw = MagicMock()
    hw.gpus = []
    hw.nvidia_gpus = []
    return {
        "config": MagicMock(),
        "endpoint_registry": MagicMock(),
        "service_provider": MagicMock(),
        "model_downloader": MagicMock(),
        "docker_service": MagicMock(),
        "hardware": hw,
    }


@pytest.fixture
def svc(deps: dict[str, Any]) -> CoquiService:
    return CoquiService(**deps)


def test_get_type(svc: CoquiService) -> None:
    assert svc.get_type() == "coqui"


def test_get_description_not_empty(svc: CoquiService) -> None:
    assert svc.get_description()


def test_service_has_docker(svc: CoquiService) -> None:
    assert svc.service_has_docker() is False


def test_is_not_cloud_service(svc: CoquiService) -> None:
    assert svc.is_cloud_service() is False


def test_get_spec_has_hardware_field(svc: CoquiService) -> None:
    spec = svc.get_spec()

    field_names = [f.name for f in spec.fields]
    assert "hardware" in field_names


def test_get_model_spec_has_alias_field(svc: CoquiService) -> None:
    spec = svc.get_model_spec()

    field_names = [f.name for f in spec.fields]
    assert "alias" in field_names


def test_get_model_spec_alias_is_optional(svc: CoquiService) -> None:
    spec = svc.get_model_spec()

    alias_field = next(f for f in spec.fields if f.name == "alias")
    assert alias_field.required is False


def test_get_custom_model_spec_is_none(svc: CoquiService) -> None:
    assert svc.get_custom_model_spec() is None


def test_coqui_options_hardware_defaults_none() -> None:
    opts = CoquiOptions()

    assert opts.hardware is None


def test_const_has_cpu_image() -> None:
    assert "cpu" in _const.images

    assert _const.images["cpu"].name


def test_const_has_gpu_image() -> None:
    assert "gpu" in _const.images

    assert _const.images["gpu"].name


def test_const_models_not_empty() -> None:
    assert len(_const.models) > 0


def test_const_model_has_required_fields() -> None:
    _, model = next(iter(_const.models.items()))

    assert model.docker_name
    assert model.default_speaker
    assert model.size


def test_default_models_loaded_on_init(svc: CoquiService) -> None:
    assert "default" in svc.models


def test_default_models_match_const(svc: CoquiService) -> None:
    for model_id in _const.models:
        assert model_id in svc.models["default"]


def test_get_image_gpu_true_returns_gpu_image(svc: CoquiService) -> None:
    image = svc._get_image(True)  # pyright: ignore[reportPrivateUsage]
    assert image == _const.images["gpu"]


def test_get_image_gpu_false_returns_cpu_image(svc: CoquiService) -> None:
    image = svc._get_image(False)  # pyright: ignore[reportPrivateUsage]
    assert image == _const.images["cpu"]


def test_build_coqui_command_uses_model_name(svc: CoquiService) -> None:
    cmd = svc._build_coqui_command(  # pyright: ignore[reportPrivateUsage]
        CoquiCmdOptions(model_name="tts_models/en/vctk/vits", model_path=None, cuda=False, language=None)
    )

    assert "tts_models/en/vctk/vits" in cmd[1]
    assert "--model_name" in cmd[1]


def test_build_coqui_command_uses_model_path_over_name(svc: CoquiService) -> None:
    cmd = svc._build_coqui_command(  # pyright: ignore[reportPrivateUsage]
        CoquiCmdOptions(model_name="some-name", model_path="/data/model.pth", cuda=False, language=None)
    )

    assert "--model_path" in cmd[1]
    assert "/data/model.pth" in cmd[1]
    assert "--model_name" not in cmd[1]


def test_build_coqui_command_includes_port_5002(svc: CoquiService) -> None:
    cmd = svc._build_coqui_command(  # pyright: ignore[reportPrivateUsage]
        CoquiCmdOptions(model_name="any-model", model_path=None, cuda=False, language=None)
    )

    assert "5002" in cmd[1]


def test_build_coqui_command_cuda_flag_when_true(svc: CoquiService) -> None:
    cmd = svc._build_coqui_command(  # pyright: ignore[reportPrivateUsage]
        CoquiCmdOptions(model_name="any-model", model_path=None, cuda=True, language=None)
    )

    assert "--use_cuda" in cmd[1]
    assert "true" in cmd[1]


def test_build_coqui_command_no_cuda_when_false(svc: CoquiService) -> None:
    cmd = svc._build_coqui_command(  # pyright: ignore[reportPrivateUsage]
        CoquiCmdOptions(model_name="any-model", model_path=None, cuda=False, language=None)
    )

    assert "--use_cuda" not in cmd[1]


def test_build_coqui_command_language_included(svc: CoquiService) -> None:
    cmd = svc._build_coqui_command(  # pyright: ignore[reportPrivateUsage]
        CoquiCmdOptions(model_name="any-model", model_path=None, cuda=False, language="en")
    )

    assert "--language" in cmd[1]
    assert "en" in cmd[1]


def test_build_coqui_command_no_language_omitted(svc: CoquiService) -> None:
    cmd = svc._build_coqui_command(  # pyright: ignore[reportPrivateUsage]
        CoquiCmdOptions(model_name="any-model", model_path=None, cuda=False, language=None)
    )

    assert "--language" not in cmd[1]


def test_build_coqui_command_raises_when_neither_name_nor_path(svc: CoquiService) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        svc._build_coqui_command(  # pyright: ignore[reportPrivateUsage]
            CoquiCmdOptions(model_name="", model_path=None, cuda=False, language=None)
        )


def test_build_coqui_command_starts_with_dash_c(svc: CoquiService) -> None:
    cmd = svc._build_coqui_command(  # pyright: ignore[reportPrivateUsage]
        CoquiCmdOptions(model_name="any-model", model_path=None, cuda=False, language=None)
    )

    assert cmd[0] == "-c"


def test_model_installed_info_get_info() -> None:
    model = _make_model_installed_info(registration_id="reg-42")

    info = model.get_info()

    assert info.registration_id == "reg-42"


def test_model_installed_info_base_url_set() -> None:
    model = _make_model_installed_info()

    assert "localhost" in model.base_url
    assert "5002" in model.base_url


def test_installed_info_init() -> None:
    models: dict[str, ModelInstalledInfo] = {}
    options = InstallServiceIn(spec={})
    parsed = CoquiOptions()

    info = InstalledInfo(models=models, options=options, parsed_options=parsed)

    assert info.models is models
    assert info.options is options
    assert info.parsed_options is parsed


def test_get_size_cpu_only(deps: dict[str, Any]) -> None:
    svc = CoquiService(**deps)

    sizes = svc.get_size()

    assert "cpu" in sizes
    assert "gpu" not in sizes


def test_get_size_with_gpu(deps: dict[str, Any]) -> None:
    deps["hardware"].gpus = [MagicMock(spec=NvidiaGpuInfo)]
    svc = CoquiService(**deps)

    sizes = svc.get_size()

    assert "gpu" in sizes


def test_get_installed_info_returns_spec_when_installed(svc: CoquiService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    result = svc.get_installed_info("default")

    assert result == installed.options.spec


def test_get_installed_info_calls_helper_when_not_installed(svc: CoquiService) -> None:
    with patch.object(svc, "_get_service_installed_info", return_value=False) as mock:  # pyright: ignore[reportPrivateUsage]
        result = svc.get_installed_info("default")

    assert mock.call_count == 1
    assert mock.call_args == call("default")
    assert result is False


def test_generate_instance_config_none_info(svc: CoquiService) -> None:
    config = svc._generate_instance_config(None, None)  # pyright: ignore[reportPrivateUsage]

    assert config.options is None
    assert config.models == []


def test_generate_instance_config_with_info(svc: CoquiService) -> None:
    info = _make_installed_info()
    info.models["model1"] = _make_model_installed_info("model1")

    config = svc._generate_instance_config(info, None)  # pyright: ignore[reportPrivateUsage]

    assert config.options == info.options
    assert len(config.models or []) == 1


def test_load_download_info_returns_dataclass(svc: CoquiService) -> None:
    result = svc._load_download_info({})  # pyright: ignore[reportPrivateUsage]

    assert isinstance(result, DownloadedInfo)


@pytest.mark.asyncio
async def test_install_instance_returns_installed_info(svc: CoquiService, deps: dict[str, Any]) -> None:
    deps["docker_service"].has_gpu_support = False
    options = InstallServiceIn(spec={})

    with (
        patch.object(svc, "_verify_docker_image", new_callable=AsyncMock),
        patch.object(svc, "_download_image_or_set_progress", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
    ):
        promise = await svc._install_instance("default", options)  # pyright: ignore[reportPrivateUsage]
        result = await promise.wait()

    assert isinstance(result, InstalledInfo)
    assert svc.service_downloaded is True


@pytest.mark.asyncio
async def test_install_instance_loads_default_models_when_missing(svc: CoquiService, deps: dict[str, Any]) -> None:
    svc.models.pop("default", None)
    deps["docker_service"].has_gpu_support = False
    options = InstallServiceIn(spec={})

    with (
        patch.object(svc, "_verify_docker_image", new_callable=AsyncMock),
        patch.object(svc, "_download_image_or_set_progress", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
    ):
        await svc._install_instance("default", options)  # pyright: ignore[reportPrivateUsage]

    assert "default" in svc.models


@pytest.mark.asyncio
async def test_uninstall_instance_clears_installed(svc: CoquiService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    options = UninstallServiceIn(purge=False)

    with patch.object(svc, "_uninstall_model", new_callable=AsyncMock):  # pyright: ignore[reportPrivateUsage]
        await svc._uninstall_instance("default", options)  # pyright: ignore[reportPrivateUsage]

    assert svc.instances_info["default"].installed is None


@pytest.mark.asyncio
async def test_uninstall_instance_purge_clears_download_state(svc: CoquiService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    svc.service_downloaded = True
    svc.models_downloaded = {"some-model": DownloadedInfo()}
    deps["docker_service"].remove_image = AsyncMock()
    options = UninstallServiceIn(purge=True)

    with (
        patch.object(svc, "_uninstall_model", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_clear_working_dir", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
    ):
        await svc._uninstall_instance("default", options)  # pyright: ignore[reportPrivateUsage]

    assert svc.service_downloaded is False
    assert svc.models_downloaded == {}


@pytest.mark.asyncio
async def test_uninstall_instance_non_default_purge_removes_instance(svc: CoquiService) -> None:
    svc.instances_info["extra"] = Instance(None, None, {}, InstanceConfig())
    installed = _make_installed_info()
    svc.instances_info["extra"].installed = installed
    options = UninstallServiceIn(purge=True)

    with (
        patch.object(svc, "_uninstall_model", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_clear_working_dir", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
    ):
        await svc._uninstall_instance("extra", options)  # pyright: ignore[reportPrivateUsage]

    assert "extra" not in svc.instances_info


def test_get_docker_compose_file_path_raises_400_no_model_id(svc: CoquiService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    with pytest.raises(HTTPException) as exc_info:
        svc.get_docker_compose_file_path("default", None)

    assert exc_info.value.status_code == 400


def test_get_docker_compose_file_path_raises_400_model_not_installed(svc: CoquiService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    with pytest.raises(HTTPException) as exc_info:
        svc.get_docker_compose_file_path("default", "nonexistent-model")

    assert exc_info.value.status_code == 400


def test_get_docker_compose_file_path_returns_path_for_installed_model(svc: CoquiService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    installed.models["test-model"] = _make_model_installed_info("test-model")
    svc.instances_info["default"].installed = installed
    expected = Path("/some/path/docker-compose.yml")
    deps["docker_service"].get_docker_compose_file_path.return_value = expected

    result = svc.get_docker_compose_file_path("default", "test-model")

    assert result == expected


@pytest.mark.asyncio
async def test_list_models_raises_404_for_unknown_instance(svc: CoquiService) -> None:
    with pytest.raises(HTTPException) as exc_info:
        await svc.list_models("nonexistent", ListModelsFilters())

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_list_models_returns_models_for_default_instance(svc: CoquiService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    result = await svc.list_models("default", ListModelsFilters())

    assert len(result.list) == len(svc.models["default"])


@pytest.mark.asyncio
async def test_list_models_filter_installed_only(svc: CoquiService) -> None:
    installed = _make_installed_info()
    model_id = next(iter(svc.models["default"]))
    installed.models[model_id] = _make_model_installed_info(model_id)
    svc.instances_info["default"].installed = installed

    result = await svc.list_models("default", ListModelsFilters(installed=True))

    assert len(result.list) >= 1
    assert all(bool(m.installed) for m in result.list)


@pytest.mark.asyncio
async def test_list_models_filter_not_installed(svc: CoquiService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    result = await svc.list_models("default", ListModelsFilters(installed=False))

    assert all(not m.installed for m in result.list)


@pytest.mark.asyncio
async def test_list_models_none_input_uses_all_instances(svc: CoquiService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    result = await svc.list_models(None, ListModelsFilters())

    assert len(result.list) > 0


@pytest.mark.asyncio
async def test_get_model_raises_400_for_unknown_model(svc: CoquiService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    with pytest.raises(HTTPException) as exc_info:
        await svc.get_model("default", "nonexistent-model")

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_get_model_returns_correct_model(svc: CoquiService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    model_id = next(iter(svc.models["default"]))

    result = await svc.get_model("default", model_id)

    assert result.id == model_id


@pytest.mark.asyncio
async def test_get_model_includes_installed_info_when_model_installed(svc: CoquiService) -> None:
    installed = _make_installed_info()
    model_id = next(iter(svc.models["default"]))
    installed.models[model_id] = _make_model_installed_info(model_id, "reg-1")
    svc.instances_info["default"].installed = installed

    result = await svc.get_model("default", model_id)

    assert bool(result.installed)


@pytest.mark.asyncio
async def test_install_model_returns_already_installed_when_present(svc: CoquiService) -> None:
    installed = _make_installed_info()
    model_id = next(iter(_const.models))
    installed.models[model_id] = _make_model_installed_info(model_id)
    svc.instances_info["default"].installed = installed

    promise = await svc._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
    result = await promise.wait()

    assert result.status == "OK"
    assert "Already" in (result.details or "")


@pytest.mark.asyncio
async def test_install_model_raises_400_for_unknown_model(svc: CoquiService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    with pytest.raises(HTTPException) as exc_info:
        await svc._install_model("default", "nonexistent-model", InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_install_model_registers_endpoint_and_returns_ok(svc: CoquiService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    installed.parsed_options = CoquiOptions(hardware=False)
    svc.instances_info["default"].installed = installed
    model_id = next(iter(_const.models))
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=5002)
    deps["docker_service"].get_docker_subnet.return_value = None
    deps["docker_service"].get_docker_container_name.return_value = "container"
    deps["docker_service"].get_container_host.return_value = "localhost"
    deps["docker_service"].get_container_port.return_value = 5002
    deps["endpoint_registry"].register_audio_speech.return_value = "reg-1"

    promise = await svc._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
    result = await promise.wait()

    assert result.status == "OK"
    assert model_id in installed.models
    assert deps["endpoint_registry"].register_audio_speech.call_count == 1


@pytest.mark.asyncio
async def test_install_model_uses_alias_as_registered_name(svc: CoquiService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    installed.parsed_options = CoquiOptions(hardware=False)
    svc.instances_info["default"].installed = installed
    model_id = next(iter(_const.models))
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=5002)
    deps["docker_service"].get_docker_subnet.return_value = None
    deps["docker_service"].get_docker_container_name.return_value = "container"
    deps["docker_service"].get_container_host.return_value = "localhost"
    deps["docker_service"].get_container_port.return_value = 5002
    deps["endpoint_registry"].register_audio_speech.return_value = "reg-1"

    await (await svc._install_model("default", model_id, InstallModelIn(spec={"alias": "my-voice"}))).wait()  # pyright: ignore[reportPrivateUsage]

    assert installed.models[model_id].registered_name == "my-voice"


@pytest.mark.asyncio
async def test_stop_instance_does_nothing_when_not_installed(svc: CoquiService) -> None:
    await svc.stop_instance("default")


@pytest.mark.asyncio
async def test_stop_instance_stops_all_model_dockers(svc: CoquiService) -> None:
    installed = _make_installed_info()
    installed.models["m1"] = _make_model_installed_info("m1")
    svc.instances_info["default"].installed = installed

    with patch.object(svc, "_stop_dockers_parallel", new_callable=AsyncMock) as mock_stop:  # pyright: ignore[reportPrivateUsage]
        await svc.stop_instance("default")

    assert mock_stop.call_count == 1


def test_get_working_output_dir_creates_and_returns_path(svc: CoquiService, tmp_path: Any) -> None:
    with patch.object(svc, "_get_working_dir", return_value=tmp_path):  # pyright: ignore[reportPrivateUsage]
        result = svc._get_working_output_dir()  # pyright: ignore[reportPrivateUsage]

    assert result == tmp_path / "output"
    assert result.exists()


@pytest.mark.asyncio
async def test_uninstall_model_removes_from_installed_and_unregisters(svc: CoquiService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    installed.models["m1"] = _make_model_installed_info("m1", "reg-1")
    svc.instances_info["default"].installed = installed
    deps["docker_service"].uninstall_docker = AsyncMock()

    await svc._uninstall_model("default", "m1", UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert "m1" not in installed.models
    assert deps["endpoint_registry"].unregister_audio_speech.call_count == 1


@pytest.mark.asyncio
async def test_uninstall_model_purge_removes_from_downloaded(svc: CoquiService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    installed.models["m1"] = _make_model_installed_info("m1", "reg-1")
    svc.instances_info["default"].installed = installed
    svc.models_downloaded["m1"] = DownloadedInfo()
    deps["docker_service"].uninstall_docker = AsyncMock()

    await svc._uninstall_model("default", "m1", UninstallModelIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert "m1" not in svc.models_downloaded


@pytest.mark.asyncio
async def test_uninstall_model_skips_when_not_in_installed(svc: CoquiService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    deps["docker_service"].uninstall_docker = AsyncMock()

    await svc._uninstall_model("default", "nonexistent", UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert deps["docker_service"].uninstall_docker.call_count == 0


@pytest.mark.asyncio
async def test_create_handler_uses_default_speaker_when_no_voice() -> None:
    handler = _create_handler("http://localhost:5002", "p225", "mp3")

    with patch("server.services.coqui_service.ffmpeg_audio_convert_async_gen", return_value=iter([])) as mock_ffmpeg:
        body = CreateSpeechRequest(model="tts-1", input="hello", voice=None)
        resp = await handler(body, None)

    assert isinstance(resp, StreamingResponse)
    assert mock_ffmpeg.call_count == 1
    assert mock_ffmpeg.call_args[0][1:] == ("wav", "mp3")


@pytest.mark.asyncio
async def test_create_handler_uses_body_voice_when_provided() -> None:
    handler = _create_handler("http://localhost:5002", "p225", "mp3")

    with patch("server.services.coqui_service.ffmpeg_audio_convert_async_gen", return_value=iter([])) as mock_ffmpeg:
        body = CreateSpeechRequest(model="tts-1", input="hello", voice="p300", format="wav")
        resp = await handler(body, None)

    assert isinstance(resp, StreamingResponse)
    args = mock_ffmpeg.call_args
    assert args[0][2] == "wav"


@pytest.mark.asyncio
async def test_create_handler_defaults_format_to_wav_when_none() -> None:
    handler = _create_handler("http://localhost:5002", "p225", None)

    with patch("server.services.coqui_service.ffmpeg_audio_convert_async_gen", return_value=iter([])) as mock_ffmpeg:
        body = CreateSpeechRequest(model="tts-1", input="hello", voice=None)
        resp = await handler(body, None)

    assert isinstance(resp, StreamingResponse)
    args = mock_ffmpeg.call_args
    assert args[0][2] == "wav"


@pytest.mark.asyncio
async def test_uninstall_instance_calls_uninstall_for_models_in_installed(svc: CoquiService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    installed.models["m1"] = _make_model_installed_info("m1")
    svc.instances_info["default"].installed = installed
    deps["docker_service"].uninstall_docker = AsyncMock()

    with patch.object(svc, "_uninstall_model", new_callable=AsyncMock) as mock_uninstall:  # pyright: ignore[reportPrivateUsage]
        await svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert mock_uninstall.call_count == 1


@pytest.mark.asyncio
async def test_list_models_skips_instances_not_in_filter(svc: CoquiService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    svc.instances_info["inst2"] = Instance(None, None, {}, InstanceConfig())
    svc.models["inst2"] = {}

    result = await svc.list_models("default", ListModelsFilters())

    instance_names_in_result = [m.service for m in result.list]
    assert not any("inst2" in s for s in instance_names_in_result)


@pytest.mark.asyncio
async def test_proxy_post_request_yields_nonempty_chunks() -> None:
    mock_chunk = b"audio-data"

    async def fake_iter():  # type: ignore[return]
        yield mock_chunk
        yield b""

    mock_resp = MagicMock()
    mock_resp.content.iter_any = fake_iter

    mock_get_ctx = MagicMock()
    mock_get_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_get_ctx.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    mock_session.get = MagicMock(return_value=mock_get_ctx)

    collected: list[bytes] = []

    def passthrough_gen(gen, *a, **kw):  # type: ignore[misc]
        async def _inner():  # type: ignore[return]
            async for chunk in gen:
                collected.append(chunk)
                yield chunk

        return _inner()

    with patch("server.services.coqui_service.ClientSession", return_value=mock_session):
        handler = _create_handler("http://localhost:5002", None, "wav")

        with patch("server.services.coqui_service.ffmpeg_audio_convert_async_gen", side_effect=passthrough_gen):
            body = CreateSpeechRequest(model="tts-1", input="hello", voice=None)
            resp = await handler(body, None)
            async for _ in resp.body_iterator:  # type: ignore[union-attr]
                pass

    assert mock_chunk in collected


@pytest.mark.asyncio
async def test_install_instance_hardware_already_in_spec_skips_default(svc: CoquiService, deps: dict[str, Any]) -> None:
    deps["docker_service"].has_gpu_support = False
    options = InstallServiceIn(spec={"hardware": False})

    with (
        patch.object(svc, "_verify_docker_image", new_callable=AsyncMock),
        patch.object(svc, "_download_image_or_set_progress", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
    ):
        promise = await svc._install_instance("default", options)  # pyright: ignore[reportPrivateUsage]
        result = await promise.wait()

    assert isinstance(result, InstalledInfo)
    assert result.parsed_options.hardware is False


@pytest.mark.asyncio
async def test_uninstall_instance_no_installed_still_clears(svc: CoquiService) -> None:
    assert svc.instances_info["default"].installed is None
    options = UninstallServiceIn(purge=False)

    await svc._uninstall_instance("default", options)  # pyright: ignore[reportPrivateUsage]

    assert svc.instances_info["default"].installed is None


@pytest.mark.asyncio
async def test_uninstall_instance_skips_model_installed_in_other_instance(svc: CoquiService) -> None:
    installed = _make_installed_info()
    installed.models["m1"] = _make_model_installed_info("m1")
    svc.instances_info["default"].installed = installed
    options = UninstallServiceIn(purge=False)

    with (
        patch.object(svc, "is_model_installed_in_other_instance", return_value=True),
        patch.object(svc, "_uninstall_model", new_callable=AsyncMock) as mock_uninstall,  # pyright: ignore[reportPrivateUsage]
    ):
        await svc._uninstall_instance("default", options)  # pyright: ignore[reportPrivateUsage]

    assert mock_uninstall.call_count == 0


@pytest.mark.asyncio
async def test_list_models_filter_installed_excludes_not_installed(svc: CoquiService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    result = await svc.list_models("default", ListModelsFilters(installed=True))

    assert len(result.list) == 0


@pytest.mark.asyncio
async def test_uninstall_model_non_tts_type_skips_unregister(svc: CoquiService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    model_info = _make_model_installed_info("m1", "reg-1")
    model_info.type = "other"
    installed.models["m1"] = model_info
    svc.instances_info["default"].installed = installed
    deps["docker_service"].uninstall_docker = AsyncMock()

    await svc._uninstall_model("default", "m1", UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert "m1" not in installed.models
    assert deps["endpoint_registry"].unregister_audio_speech.call_count == 0
    assert deps["docker_service"].uninstall_docker.call_count == 1
