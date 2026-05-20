# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
import io
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from fastapi import HTTPException
from PIL import Image
from pydantic import ValidationError

from server.models.api import ImagesRequest
from server.models.models import InstallModelIn, ListModelsFilters, UninstallModelIn
from server.models.services import InstallServiceIn, ServiceSize, UninstallServiceIn
from server.services.base2_service import CustomModel, Instance, InstanceConfig
from server.services.stable_diffusion_service import (
    DefaultSdNextConfig,
    DownloadedInfo,
    InstalledInfo,
    ModelInstalledInfo,
    QualityLevel,
    SDModelOptions,
    SDOptions,
    StableDiffusionModel,
    StableDiffusionService,
    _add_body_config,  # pyright: ignore[reportPrivateUsage]
    _const,  # pyright: ignore[reportPrivateUsage]
    _get_in_format,  # pyright: ignore[reportPrivateUsage]
    _stable_diffusion_handler,  # pyright: ignore[reportPrivateUsage]
    convert_b64png_to_b64jpg,
    convert_b64png_to_b64webp,
    get_image_size,
    remove_background_from_img,
    split_text_to_json_and_prompt,
)
from server.utils.core import (
    DownloadedPacket,
    PreDownloadPacket,
    Stream,
    StreamChunkProgress,
    SuccessDownloadPacket,
)
from server.utils.hardware import NvidiaGpuInfo


def _make_installed_info(instance: str = "default") -> InstalledInfo:
    docker = MagicMock()
    docker.name = f"stable-diffusion-{instance}"
    return InstalledInfo(
        docker=docker,
        models={},
        options=InstallServiceIn(spec={}),
        parsed_options=SDOptions(),
        container_host="localhost",
        container_port=7860,
        docker_exposed_port=7860,
        proxy_registration_id=None,
        base_url="http://localhost:7860",
    )


def _make_png_b64(mode: str = "RGB") -> str:
    color = (100, 150, 200, 200) if mode == "RGBA" else (100, 150, 200)
    img = Image.new(mode, (4, 4), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _make_sd_custom(custom_id: str = "uuid-1", model_id: str = "my-sd-model") -> CustomModel:
    return CustomModel(
        id=custom_id,
        data={
            "id": model_id,
            "filetype": "Stable-diffusion",
            "url": "https://civitai.com/api/download/models/123456789",
            "filename": "mymodel.safetensors",
            "size": "2GB",
        },
    )


# --- Fixtures ---


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
def svc(deps: dict[str, Any]) -> StableDiffusionService:
    return StableDiffusionService(**deps)


# --- Identity ---


def test_get_type(svc: StableDiffusionService) -> None:
    assert svc.get_type() == "stable-diffusion"


def test_get_description_not_empty(svc: StableDiffusionService) -> None:
    assert svc.get_description()


def test_service_has_docker(svc: StableDiffusionService) -> None:
    assert svc.service_has_docker() is True


def test_is_not_cloud_service(svc: StableDiffusionService) -> None:
    assert svc.is_cloud_service() is False


def test_const_has_cpu_and_gpu_images() -> None:
    assert "cpu" in _const.images
    assert "gpu" in _const.images


def test_const_models_not_empty() -> None:
    assert len(_const.models) > 0


def test_sd_options_hardware_defaults_none() -> None:
    opts = SDOptions()
    assert opts.hardware is None


def test_sd_options_expose_api_prefix_defaults_empty() -> None:
    opts = SDOptions()
    assert opts.expose_api_at_prefix == ""


@pytest.mark.parametrize("bad_prefix", ["bad prefix", "bad@prefix", "bad.prefix"])
def test_sd_options_invalid_prefix_raises(bad_prefix: str) -> None:
    with pytest.raises(ValidationError):
        SDOptions(expose_api_at_prefix=bad_prefix)


@pytest.mark.parametrize("good_prefix", ["my-sd", "SD_1", "prefix123"])
def test_sd_options_valid_prefix(good_prefix: str) -> None:
    opts = SDOptions(expose_api_at_prefix=good_prefix)

    assert opts.expose_api_at_prefix == good_prefix


def test_sd_model_options_alias_defaults_none() -> None:
    opts = SDModelOptions()

    assert opts.alias is None


def test_get_spec_has_hardware_field(svc: StableDiffusionService) -> None:
    spec = svc.get_spec()

    field_names = [f.name for f in spec.fields]
    assert "hardware" in field_names


def test_get_spec_has_expose_api_at_prefix_field(svc: StableDiffusionService) -> None:
    spec = svc.get_spec()

    field_names = [f.name for f in spec.fields]
    assert "expose_api_at_prefix" in field_names


def test_get_spec_expose_api_at_prefix_is_optional(svc: StableDiffusionService) -> None:
    spec = svc.get_spec()

    field = next(f for f in spec.fields if f.name == "expose_api_at_prefix")
    assert field.required is False


def test_get_model_spec_has_alias(svc: StableDiffusionService) -> None:
    spec = svc.get_model_spec()

    field_names = [f.name for f in spec.fields]
    assert "alias" in field_names


def test_get_custom_model_spec_not_none(svc: StableDiffusionService) -> None:
    assert svc.get_custom_model_spec() is not None


def test_get_custom_model_spec_has_required_fields(svc: StableDiffusionService) -> None:
    spec = svc.get_custom_model_spec()

    assert spec is not None
    field_names = {f.name for f in spec.fields}
    assert {"id", "filetype", "url", "filename", "size"} <= field_names


def test_get_image_gpu_true_returns_gpu_image(svc: StableDiffusionService) -> None:
    image = svc._get_image(True)  # pyright: ignore[reportPrivateUsage]

    assert image == _const.images["gpu"]


def test_get_image_gpu_false_returns_cpu_image(svc: StableDiffusionService) -> None:
    image = svc._get_image(False)  # pyright: ignore[reportPrivateUsage]

    assert image == _const.images["cpu"]


def test_add_custom_model_stores_model(svc: StableDiffusionService) -> None:
    svc._add_custom_model("default", _make_sd_custom())  # pyright: ignore[reportPrivateUsage]

    assert "my-sd-model" in svc.models["default"]


def test_add_custom_model_stores_custom_id(svc: StableDiffusionService) -> None:
    svc._add_custom_model("default", _make_sd_custom(custom_id="uuid-99"))  # pyright: ignore[reportPrivateUsage]

    assert svc.models["default"]["my-sd-model"].custom == "uuid-99"


def test_add_custom_model_stable_diffusion_type_is_txt2img(svc: StableDiffusionService) -> None:
    svc._add_custom_model("default", _make_sd_custom())  # pyright: ignore[reportPrivateUsage]

    assert svc.models["default"]["my-sd-model"].type == "txt2img"


def test_add_custom_model_lora_type_is_lora(svc: StableDiffusionService) -> None:
    custom = CustomModel(
        id="uuid-lora",
        data={
            "id": "my-lora",
            "filetype": "Lora",
            "url": "https://civitai.com/lora/123",
            "filename": "lora.safetensors",
            "size": "100MB",
        },
    )
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    assert svc.models["default"]["my-lora"].type == "lora"


def test_add_custom_model_duplicate_raises_http_400(svc: StableDiffusionService) -> None:
    svc._add_custom_model("default", _make_sd_custom())  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(HTTPException) as exc_info:
        svc._add_custom_model("default", _make_sd_custom())  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400


def test_default_instance_created(svc: StableDiffusionService) -> None:
    assert "default" in svc.instances_info


def test_default_models_loaded(svc: StableDiffusionService) -> None:
    assert "default" in svc.models
    assert len(svc.models["default"]) > 0


def test_model_installed_info_get_info() -> None:
    mi = ModelInstalledInfo(
        id="model-1",
        type="txt2img",
        registered_name="my-model",
        options=InstallModelIn(spec={}),
        model_path=Path("/some/path"),
        registration_id="reg-123",
    )

    info = mi.get_info()

    assert info.registration_id == "reg-123"
    assert info.spec == {}


def test_get_size_returns_cpu_size(svc: StableDiffusionService) -> None:
    sizes: ServiceSize = svc.get_size()

    assert "cpu" in sizes
    assert sizes["cpu"] == _const.images["cpu"].size  # pyright: ignore[reportArgumentType]


def test_get_size_returns_gpu_when_nvidia_gpu_present(deps: dict[str, Any]) -> None:
    gpu = NvidiaGpuInfo(id=0, name="NVIDIA RTX 3080", vram="10GB")
    deps["hardware"].gpus = [gpu]
    svc = StableDiffusionService(**deps)

    sizes = svc.get_size()

    assert "gpu" in sizes
    assert sizes["gpu"] == _const.images["gpu"].size  # pyright: ignore[reportArgumentType]


def test_get_size_no_gpu_key_without_gpu_support(svc: StableDiffusionService) -> None:
    sizes = svc.get_size()

    assert "gpu" not in sizes


def test_get_installed_info_returns_spec_when_installed(svc: StableDiffusionService) -> None:
    installed = _make_installed_info()
    installed.options.spec["hardware"] = True
    svc.instances_info["default"].installed = installed

    result = svc.get_installed_info("default")

    assert result == installed.options.spec


def test_get_installed_info_delegates_when_not_installed(svc: StableDiffusionService) -> None:
    svc.instances_info["default"].installed = None

    with patch.object(svc, "_get_service_installed_info", return_value=False) as mock:  # pyright: ignore[reportPrivateUsage]
        result = svc.get_installed_info("default")

    assert mock.call_count == 1
    assert mock.call_args == call("default")
    assert result is False


def test_generate_instance_config_no_info(svc: StableDiffusionService) -> None:
    config = svc._generate_instance_config(None, None)  # pyright: ignore[reportPrivateUsage]

    assert config.options is None
    assert config.models == []


def test_generate_instance_config_with_info_no_models(svc: StableDiffusionService) -> None:
    info = _make_installed_info()

    config = svc._generate_instance_config(info, None)  # pyright: ignore[reportPrivateUsage]
    assert config.options == info.options
    assert config.models == []


def test_generate_instance_config_with_models(svc: StableDiffusionService) -> None:
    info = _make_installed_info()
    info.models["m1"] = ModelInstalledInfo(
        id="m1",
        type="txt2img",
        registered_name="m1",
        options=InstallModelIn(spec={}),
        model_path=Path("/tmp/m1.safetensors"),
        registration_id="reg-1",
    )
    config = svc._generate_instance_config(info, None)  # pyright: ignore[reportPrivateUsage]

    assert len(config.models or []) == 1


def test_load_download_info_returns_downloaded_info(svc: StableDiffusionService) -> None:
    result = svc._load_download_info({"model_path": "/some/path"})  # pyright: ignore[reportPrivateUsage]

    assert isinstance(result, DownloadedInfo)
    assert result.model_path == "/some/path"


@pytest.mark.asyncio
async def test_update_config_creates_config_with_defaults(svc: StableDiffusionService, tmp_path: Path) -> None:
    with patch.object(svc, "_get_working_data_dir", return_value=tmp_path):  # pyright: ignore[reportPrivateUsage]
        await svc.update_config()

    data = json.loads((tmp_path / "config.json").read_text())
    defaults = DefaultSdNextConfig()._asdict()
    for key, value in defaults.items():
        assert data[key] == value


@pytest.mark.asyncio
async def test_update_config_preserves_existing_values(svc: StableDiffusionService, tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"custom_key": "kept"}))

    with patch.object(svc, "_get_working_data_dir", return_value=tmp_path):  # pyright: ignore[reportPrivateUsage]
        await svc.update_config()

    data = json.loads((tmp_path / "config.json").read_text())
    assert data["custom_key"] == "kept"


@pytest.mark.asyncio
async def test_update_config_handles_missing_file(svc: StableDiffusionService, tmp_path: Path) -> None:
    with patch.object(svc, "_get_working_data_dir", return_value=tmp_path):  # pyright: ignore[reportPrivateUsage]
        await svc.update_config()

    assert (tmp_path / "config.json").exists()


@pytest.mark.asyncio
async def test_update_config_handles_invalid_json(svc: StableDiffusionService, tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("not-json")

    with patch.object(svc, "_get_working_data_dir", return_value=tmp_path):  # pyright: ignore[reportPrivateUsage]
        await svc.update_config()

    data = json.loads((tmp_path / "config.json").read_text())
    assert "samples_format" in data


@pytest.mark.asyncio
async def test_stop_instance_does_nothing_when_not_installed(svc: StableDiffusionService) -> None:
    svc.instances_info["default"].installed = None

    with patch.object(svc, "_stop_docker", new_callable=AsyncMock) as mock_stop:  # pyright: ignore[reportPrivateUsage]
        await svc.stop_instance("default")

    assert mock_stop.call_count == 0


@pytest.mark.asyncio
async def test_stop_instance_calls_stop_docker_when_installed(svc: StableDiffusionService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    with patch.object(svc, "_stop_docker", new_callable=AsyncMock) as mock_stop:  # pyright: ignore[reportPrivateUsage]
        await svc.stop_instance("default")

    assert mock_stop.call_count == 1
    assert mock_stop.call_args == call(installed.docker)


def test_get_docker_compose_file_path_raises_400_with_model_id(svc: StableDiffusionService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    with pytest.raises(HTTPException) as exc_info:
        svc.get_docker_compose_file_path("default", "some-model")

    assert exc_info.value.status_code == 400


def test_get_docker_compose_file_path_returns_path_without_model_id(svc: StableDiffusionService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    expected = Path("/some/path/docker-compose.yml")
    deps["docker_service"].get_docker_compose_file_path.return_value = expected

    result = svc.get_docker_compose_file_path("default", None)

    assert result == expected


def test_add_custom_model_creates_models_dict_for_new_instance(svc: StableDiffusionService) -> None:
    svc.instances_info["extra"] = Instance(None, None, {}, InstanceConfig())
    custom = _make_sd_custom(model_id="new-model")

    svc._add_custom_model("extra", custom)  # pyright: ignore[reportPrivateUsage]

    assert "new-model" in svc.models["extra"]


def test_remove_custom_model_deletes_entry(svc: StableDiffusionService) -> None:
    svc._add_custom_model("default", _make_sd_custom())  # pyright: ignore[reportPrivateUsage]

    svc._remove_custom_model("default", _make_sd_custom())  # pyright: ignore[reportPrivateUsage]

    assert "my-sd-model" not in svc.models["default"]


def test_remove_custom_model_raises_400_when_in_use(svc: StableDiffusionService) -> None:
    svc._add_custom_model("default", _make_sd_custom())  # pyright: ignore[reportPrivateUsage]
    installed = _make_installed_info()
    installed.models["my-sd-model"] = ModelInstalledInfo(
        id="my-sd-model",
        type="txt2img",
        registered_name="my-sd-model",
        options=InstallModelIn(spec={}),
        model_path=Path("/tmp/mymodel.safetensors"),
        registration_id="reg-1",
    )
    svc.instances_info["default"].installed = installed

    with pytest.raises(HTTPException) as exc_info:
        svc._remove_custom_model("default", _make_sd_custom())  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_list_models_returns_all_models_for_valid_instance(svc: StableDiffusionService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    result = await svc.list_models("default", ListModelsFilters())

    assert len(result.list) == len(svc.models["default"])


@pytest.mark.asyncio
async def test_list_models_raises_404_for_unknown_instance(svc: StableDiffusionService) -> None:
    with pytest.raises(HTTPException) as exc_info:
        await svc.list_models("nonexistent", ListModelsFilters())

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_list_models_with_list_of_instances(svc: StableDiffusionService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    result = await svc.list_models(["default"], ListModelsFilters())

    assert len(result.list) == len(svc.models["default"])


@pytest.mark.asyncio
async def test_list_models_none_instance_uses_all(svc: StableDiffusionService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    result = await svc.list_models(None, ListModelsFilters())

    assert len(result.list) > 0


@pytest.mark.asyncio
async def test_list_models_filters_installed_only(svc: StableDiffusionService) -> None:
    installed = _make_installed_info()
    model_id = next(iter(svc.models["default"]))
    installed.models[model_id] = ModelInstalledInfo(
        id=model_id,
        type="txt2img",
        registered_name=model_id,
        options=InstallModelIn(spec={}),
        model_path=Path("/tmp/model.safetensors"),
        registration_id="reg-1",
    )
    svc.instances_info["default"].installed = installed

    result = await svc.list_models("default", ListModelsFilters(installed=True))

    assert len(result.list) >= 1
    assert all(bool(m.installed) for m in result.list)


@pytest.mark.asyncio
async def test_list_models_filters_not_installed(svc: StableDiffusionService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    result = await svc.list_models("default", ListModelsFilters(installed=False))

    assert all(not m.installed for m in result.list)


@pytest.mark.asyncio
async def test_get_model_returns_model_for_valid_id(svc: StableDiffusionService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    model_id = next(iter(svc.models["default"]))

    result = await svc.get_model("default", model_id)

    assert result.id == model_id


@pytest.mark.asyncio
async def test_get_model_raises_400_for_unknown_model(svc: StableDiffusionService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    with pytest.raises(HTTPException) as exc_info:
        await svc.get_model("default", "nonexistent-model")

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_get_model_returns_installed_info_when_model_in_info(svc: StableDiffusionService) -> None:
    installed = _make_installed_info()
    model_id = next(iter(svc.models["default"]))
    installed.models[model_id] = ModelInstalledInfo(
        id=model_id,
        type="txt2img",
        registered_name=model_id,
        options=InstallModelIn(spec={}),
        model_path=Path("/tmp/model.safetensors"),
        registration_id="reg-42",
    )
    svc.instances_info["default"].installed = installed

    result = await svc.get_model("default", model_id)

    assert result.installed is not None
    assert result.installed.registration_id == "reg-42"  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.asyncio
async def test_refresh_checkpoints_calls_post(svc: StableDiffusionService) -> None:
    client = AsyncMock()

    await svc.refresh_checkpoints(client, "http://localhost:7860")

    assert client.post.call_count == 1
    assert client.post.call_args == call("http://localhost:7860/sdapi/v1/refresh-checkpoints")


@pytest.mark.asyncio
async def test_refresh_vae_calls_post(svc: StableDiffusionService) -> None:
    client = AsyncMock()

    await svc.refresh_vae(client, "http://localhost:7860")

    assert client.post.call_count == 1
    assert client.post.call_args == call("http://localhost:7860/sdapi/v1/refresh-vae")


@pytest.mark.asyncio
async def test_refresh_loras_calls_post(svc: StableDiffusionService) -> None:
    client = AsyncMock()

    await svc.refresh_loras(client, "http://localhost:7860")

    assert client.post.call_count == 1
    assert client.post.call_args == call("http://localhost:7860/sdapi/v1/refresh-loras")


@pytest.mark.asyncio
async def test_refresh_models_calls_all_three_refresh_functions(svc: StableDiffusionService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    with (
        patch.object(svc, "refresh_checkpoints", new_callable=AsyncMock) as mock_cp,
        patch.object(svc, "refresh_vae", new_callable=AsyncMock) as mock_vae,
        patch.object(svc, "refresh_loras", new_callable=AsyncMock) as mock_loras,
        patch("server.services.stable_diffusion_service.ClientSession") as mock_session_cls,
    ):
        mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
        mock_session_cls.return_value.__aexit__ = AsyncMock(return_value=None)

        await svc.refresh_models("default")

    assert mock_cp.call_count == 1
    assert mock_vae.call_count == 1
    assert mock_loras.call_count == 1


@pytest.mark.asyncio
async def test_download_model_returns_path_and_filename(svc: StableDiffusionService, tmp_path: Path) -> None:
    model = StableDiffusionModel(
        filetype="Stable-diffusion",
        type="txt2img",
        url="https://example.com/model.safetensors",
        filename="model.safetensors",
        size="1GB",
    )
    expected_path = tmp_path / "model.safetensors"

    async def mock_download(*args: object) -> Any:
        yield PreDownloadPacket(file_bytes_size=1024)
        yield DownloadedPacket(downloaded_bytes_size=512)
        yield SuccessDownloadPacket(local_path=expected_path, filename="model.safetensors")

    svc.model_downloader.download.return_value = mock_download()  # pyright: ignore[reportAttributeAccessIssue]
    stream = MagicMock()

    with patch.object(svc, "_get_working_models_dir", return_value=tmp_path):  # pyright: ignore[reportPrivateUsage]
        local_path, filename = await svc._download_model(stream, model)  # pyright: ignore[reportPrivateUsage]

    assert local_path == expected_path
    assert filename == "model.safetensors"
    assert stream.emit.call_count >= 2


@pytest.mark.asyncio
async def test_download_model_emits_progress_with_zero_bytes(svc: StableDiffusionService, tmp_path: Path) -> None:
    model = StableDiffusionModel(
        filetype="Lora",
        type="lora",
        url="https://example.com/lora.safetensors",
        filename="lora.safetensors",
        size="100MB",
    )

    async def mock_download(*args: object) -> Any:
        yield DownloadedPacket(downloaded_bytes_size=0)
        yield SuccessDownloadPacket(local_path=tmp_path / "lora.safetensors", filename="lora.safetensors")

    svc.model_downloader.download.return_value = mock_download()  # pyright: ignore[reportAttributeAccessIssue]
    stream = MagicMock()

    with patch.object(svc, "_get_working_models_dir", return_value=tmp_path):  # pyright: ignore[reportPrivateUsage]
        local_path, _ = await svc._download_model(stream, model)  # pyright: ignore[reportPrivateUsage]

    assert local_path == tmp_path / "lora.safetensors"


@pytest.mark.asyncio
async def test_download_model_or_set_progress_starts_download(svc: StableDiffusionService) -> None:
    model = next(iter(svc.models["default"].values()))
    stream = MagicMock()
    path = (Path("/tmp/m.safetensors"), "m.safetensors")

    with patch.object(svc, "_download_model", new_callable=AsyncMock, return_value=path) as mock_dl:  # pyright: ignore[reportPrivateUsage]
        result_path, _ = await svc._download_model_or_set_progress(stream, model, "test-model")  # pyright: ignore[reportPrivateUsage]

    assert mock_dl.call_count == 1
    assert "test-model" not in svc.models_download_progress
    assert result_path == Path("/tmp/m.safetensors")


@pytest.mark.asyncio
async def test_download_model_or_set_progress_forwards_existing_stream(svc: StableDiffusionService) -> None:
    existing_stream: Stream[StreamChunkProgress] = Stream()  # type: ignore[type-arg]
    chunk = StreamChunkProgress(type="progress", stage="download", value=0.5, data={})
    existing_stream.emit(chunk)
    existing_stream.close()
    svc.models_download_progress["model-id"] = existing_stream  # type: ignore[assignment]
    output_stream = MagicMock()
    model = next(iter(svc.models["default"].values()))

    await svc._download_model_or_set_progress(output_stream, model, "model-id")  # pyright: ignore[reportPrivateUsage]

    assert output_stream.emit.call_count == 1
    assert output_stream.emit.call_args == call(chunk)


@pytest.mark.asyncio
async def test_download_model_or_set_progress_forwards_stream_with_path_data(svc: StableDiffusionService) -> None:
    existing_stream: Stream[StreamChunkProgress] = Stream()  # type: ignore[type-arg]
    chunk_with_data = StreamChunkProgress(
        type="progress", stage="download", value=1.0, data={"local_model_path": "/tmp/model.safetensors", "filename": "model.safetensors"}
    )
    existing_stream.emit(chunk_with_data)
    existing_stream.close()
    svc.models_download_progress["model-id"] = existing_stream  # type: ignore[assignment]
    output_stream = MagicMock()
    model = next(iter(svc.models["default"].values()))

    local_path, filename = await svc._download_model_or_set_progress(output_stream, model, "model-id")  # pyright: ignore[reportPrivateUsage]

    assert str(local_path) == "/tmp/model.safetensors"
    assert filename == "model.safetensors"


@pytest.mark.asyncio
async def test_install_model_returns_already_installed_when_model_in_info(svc: StableDiffusionService) -> None:
    installed = _make_installed_info()
    model_id = next(iter(svc.models["default"]))
    installed.models[model_id] = ModelInstalledInfo(
        id=model_id,
        type="txt2img",
        registered_name=model_id,
        options=InstallModelIn(spec={}),
        model_path=Path("/tmp/model.safetensors"),
        registration_id="",
    )
    svc.instances_info["default"].installed = installed

    promise = await svc._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]

    result = await promise.wait()
    assert result.status == "OK"
    assert "Already installed" in result.details


@pytest.mark.asyncio
async def test_install_model_raises_400_for_unknown_model(svc: StableDiffusionService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    with pytest.raises(HTTPException) as exc_info:
        await svc._install_model("default", "nonexistent-model", InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_install_model_registers_txt2img_endpoint(svc: StableDiffusionService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    model_id = "Plant Milk Walnut"
    path = (Path("/tmp/model.safetensors"), "model")

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=path),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "refresh_models", new_callable=AsyncMock),
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        result = await promise.wait()

    assert result.status == "OK"
    assert deps["endpoint_registry"].register_image_generations.call_count == 1


@pytest.mark.asyncio
async def test_install_model_lora_does_not_register_image_endpoint(svc: StableDiffusionService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    model_id = "Fantastic Landscapes"  # lora type

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=(Path("/tmp/lora.safetensors"), "lora")),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "refresh_models", new_callable=AsyncMock),
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        result = await promise.wait()

    assert result.status == "OK"
    assert deps["endpoint_registry"].register_image_generations.call_count == 0


@pytest.mark.asyncio
async def test_install_model_uses_alias_as_registered_name(svc: StableDiffusionService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    model_id = "Plant Milk Walnut"
    path = (Path("/tmp/model.safetensors"), "model")

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=path),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "refresh_models", new_callable=AsyncMock),
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn(spec={"alias": "my-alias"}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert installed.models[model_id].registered_name == "my-alias"


@pytest.mark.asyncio
async def test_install_model_marks_model_as_downloaded(svc: StableDiffusionService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    model_id = "Fantastic Landscapes"

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=(Path("/tmp/lora.safetensors"), "lora")),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "refresh_models", new_callable=AsyncMock),
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert model_id in svc.models_downloaded


@pytest.mark.asyncio
async def test_install_model_raises_400_when_download_returns_no_path(svc: StableDiffusionService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    model_id = "Plant Milk Walnut"

    with (  # noqa: PT012
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=(None, "")),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "refresh_models", new_callable=AsyncMock),
        pytest.raises(HTTPException) as exc_info,
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_uninstall_model_removes_txt2img_and_unregisters(svc: StableDiffusionService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    model_id = "Plant Milk Walnut"
    installed.models[model_id] = ModelInstalledInfo(
        id=model_id,
        type="txt2img",
        registered_name=model_id,
        options=InstallModelIn(spec={}),
        model_path=Path("/tmp/model.safetensors"),
        registration_id="reg-1",
    )
    svc.instances_info["default"].installed = installed

    with (
        patch.object(svc, "is_model_installed_in_other_instance", return_value=True),
        patch.object(svc, "refresh_models", new_callable=AsyncMock),
    ):
        await svc._uninstall_model("default", model_id, UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert model_id not in installed.models
    assert deps["endpoint_registry"].unregister_image_generations.call_count == 1
    assert deps["endpoint_registry"].unregister_image_generations.call_args == call(model_id, "reg-1")


@pytest.mark.asyncio
async def test_uninstall_model_purges_downloaded_file(svc: StableDiffusionService, tmp_path: Path) -> None:
    installed = _make_installed_info()
    model_id = "Plant Milk Walnut"
    model_file = tmp_path / "model.safetensors"
    model_file.write_text("data")
    installed.models[model_id] = ModelInstalledInfo(
        id=model_id,
        type="txt2img",
        registered_name=model_id,
        options=InstallModelIn(spec={}),
        model_path=model_file,
        registration_id="reg-1",
    )
    svc.instances_info["default"].installed = installed
    svc.models_downloaded[model_id] = DownloadedInfo(str(model_file))

    with (
        patch.object(svc, "is_model_installed_in_other_instance", return_value=True),
        patch.object(svc, "refresh_models", new_callable=AsyncMock),
    ):
        await svc._uninstall_model("default", model_id, UninstallModelIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert model_id not in svc.models_downloaded
    assert not model_file.exists()


@pytest.mark.asyncio
async def test_uninstall_model_does_nothing_for_unknown_model(svc: StableDiffusionService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    with patch.object(svc, "refresh_models", new_callable=AsyncMock):
        await svc._uninstall_model("default", "unknown-model", UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert deps["endpoint_registry"].unregister_image_generations.call_count == 0


def test_get_working_models_dir_creates_directory(svc: StableDiffusionService, tmp_path: Path) -> None:
    with patch.object(svc, "_get_working_dir", return_value=tmp_path):  # pyright: ignore[reportPrivateUsage]
        result = svc._get_working_models_dir()  # pyright: ignore[reportPrivateUsage]

    assert result == tmp_path / "models"
    assert result.is_dir()


def test_get_working_data_dir_creates_directory(svc: StableDiffusionService, tmp_path: Path) -> None:
    with patch.object(svc, "_get_working_dir", return_value=tmp_path):  # pyright: ignore[reportPrivateUsage]
        result = svc._get_working_data_dir()  # pyright: ignore[reportPrivateUsage]

    assert result == tmp_path / "data"
    assert result.is_dir()


def test_get_working_logs_creates_log_file(svc: StableDiffusionService, tmp_path: Path) -> None:
    with patch.object(svc, "_get_working_dir", return_value=tmp_path):  # pyright: ignore[reportPrivateUsage]
        result = svc._get_working_logs()  # pyright: ignore[reportPrivateUsage]

    assert result.name == "sdnext.log"
    assert result.exists()


@pytest.mark.asyncio
async def test_install_instance_raises_400_on_macos(svc: StableDiffusionService) -> None:
    options = InstallServiceIn(spec={"hardware": False})

    with (
        patch("server.services.stable_diffusion_service.platform.system", return_value="Darwin"),
        patch.object(svc, "_verify_docker_image", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        pytest.raises(HTTPException) as exc_info,
    ):
        await svc._install_instance("default", options)  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_install_instance_calls_docker_and_returns_installed_info(svc: StableDiffusionService, deps: dict[str, Any]) -> None:
    deps["docker_service"].get_docker_subnet.return_value = None
    deps["docker_service"].get_docker_container_name.return_value = "df-stable-diffusion"
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=7860)
    deps["docker_service"].get_container_host.return_value = "localhost"
    deps["docker_service"].get_container_port.return_value = 7860
    deps["docker_service"].get_user_for_docker = AsyncMock(return_value=None)
    options = InstallServiceIn(spec={"hardware": False})

    with (
        patch.object(svc, "_download_image_or_set_progress", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_verify_docker_image", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
        patch.object(svc, "update_config", new_callable=AsyncMock),
        patch("server.services.stable_diffusion_service.platform.system", return_value="Linux"),
    ):
        promise = await svc._install_instance("default", options)  # pyright: ignore[reportPrivateUsage]
        result = await promise.wait()

    assert isinstance(result, InstalledInfo)
    assert deps["docker_service"].install_and_run_docker.call_count == 1


@pytest.mark.asyncio
async def test_install_instance_registers_proxy_when_prefix_set(svc: StableDiffusionService, deps: dict[str, Any]) -> None:
    deps["docker_service"].get_docker_subnet.return_value = None
    deps["docker_service"].get_docker_container_name.return_value = "df-stable-diffusion"
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=7860)
    deps["docker_service"].get_container_host.return_value = "localhost"
    deps["docker_service"].get_container_port.return_value = 7860
    deps["docker_service"].get_user_for_docker = AsyncMock(return_value=None)
    options = InstallServiceIn(spec={"hardware": False, "expose_api_at_prefix": "my-sd"})

    with (
        patch.object(svc, "_download_image_or_set_progress", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_verify_docker_image", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
        patch.object(svc, "update_config", new_callable=AsyncMock),
        patch("server.services.stable_diffusion_service.platform.system", return_value="Linux"),
    ):
        promise = await svc._install_instance("default", options)  # pyright: ignore[reportPrivateUsage]
        result = await promise.wait()

    assert deps["endpoint_registry"].register_custom_endpoint_as_proxy.call_count == 1
    assert result.proxy_registration_id is not None


@pytest.mark.asyncio
async def test_uninstall_instance_does_nothing_when_not_installed(svc: StableDiffusionService, deps: dict[str, Any]) -> None:
    svc.instances_info["default"].installed = None
    deps["docker_service"].uninstall_docker = AsyncMock()

    await svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert deps["docker_service"].uninstall_docker.call_count == 0


@pytest.mark.asyncio
async def test_uninstall_instance_unregisters_txt2img_models(svc: StableDiffusionService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    model_id = "Plant Milk Walnut"
    installed.models[model_id] = ModelInstalledInfo(
        id=model_id,
        type="txt2img",
        registered_name=model_id,
        options=InstallModelIn(spec={}),
        model_path=Path("/tmp/model.safetensors"),
        registration_id="reg-txt2img",
    )
    svc.instances_info["default"].installed = installed
    deps["docker_service"].uninstall_docker = AsyncMock()

    await svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert deps["endpoint_registry"].unregister_image_generations.call_count == 1
    assert deps["endpoint_registry"].unregister_image_generations.call_args == call(model_id, "reg-txt2img")
    assert deps["docker_service"].uninstall_docker.call_count == 1


@pytest.mark.asyncio
async def test_uninstall_instance_unregisters_proxy_when_prefix_set(svc: StableDiffusionService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    installed.parsed_options = SDOptions(expose_api_at_prefix="my-sd")
    installed.proxy_registration_id = "proxy-reg-1"
    svc.instances_info["default"].installed = installed
    deps["docker_service"].uninstall_docker = AsyncMock()

    await svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert deps["endpoint_registry"].unregister_custom_endpoint.call_count == 1


@pytest.mark.asyncio
async def test_uninstall_instance_purge_clears_working_dir(svc: StableDiffusionService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    deps["docker_service"].uninstall_docker = AsyncMock()
    deps["docker_service"].remove_image = AsyncMock()

    with patch.object(svc, "_clear_working_dir", new_callable=AsyncMock) as mock_clear:  # pyright: ignore[reportPrivateUsage]
        await svc._uninstall_instance("default", UninstallServiceIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert svc.service_downloaded is False
    assert mock_clear.call_count == 1


@pytest.mark.asyncio
async def test_uninstall_instance_purge_removes_non_default_instance(svc: StableDiffusionService, deps: dict[str, Any]) -> None:
    svc.instances_info["extra"] = Instance(None, None, {}, InstanceConfig())
    installed = _make_installed_info(instance="extra")
    svc.instances_info["extra"].installed = installed
    svc.models["extra"] = {}
    deps["docker_service"].uninstall_docker = AsyncMock()
    deps["docker_service"].remove_image = AsyncMock()

    with patch.object(svc, "_clear_working_dir", new_callable=AsyncMock):  # pyright: ignore[reportPrivateUsage]
        await svc._uninstall_instance("extra", UninstallServiceIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert "extra" not in svc.instances_info


def test_split_text_no_sd_tags_returns_empty_dict_and_original() -> None:
    opts, text = split_text_to_json_and_prompt("hello world")

    assert opts == {}
    assert text == "hello world"


def test_split_text_with_valid_sd_json_extracts_settings() -> None:
    opts, text = split_text_to_json_and_prompt('<sd>{"steps": 30}</sd>draw a cat')

    assert opts["steps"] == 30  # pyright: ignore[reportTypedDictNotRequiredAccess]
    assert "draw a cat" in text
    assert "<sd>" not in text


def test_split_text_with_invalid_json_raises_422() -> None:
    with pytest.raises(HTTPException) as exc_info:
        split_text_to_json_and_prompt("<sd>not json</sd>draw a cat")

    assert exc_info.value.status_code == 422


def test_split_text_sd_json_with_multiline_content() -> None:
    opts, _ = split_text_to_json_and_prompt('<sd>{"steps": 15,\n"cfg_scale": 5}</sd>portrait')

    assert opts["steps"] == 15  # pyright: ignore[reportTypedDictNotRequiredAccess]
    assert opts["cfg_scale"] == 5  # pyright: ignore[reportTypedDictNotRequiredAccess]


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        ("512x768", (512, 768)),
        ("1024x1024", (1024, 1024)),
    ],
)
def test_get_image_size(size: str, expected: tuple[int, int]) -> None:
    w, h = get_image_size(size)
    assert (w, h) == expected


def test_get_image_size_invalid_format_raises_value_error() -> None:
    with pytest.raises(ValueError, match="dimension"):
        get_image_size("512")


def test_get_image_size_non_int_raises_value_error() -> None:
    with pytest.raises(ValueError, match="ints"):
        get_image_size("axb")


def test_get_image_size_three_parts_raises_value_error() -> None:
    with pytest.raises(ValueError, match="dimension"):
        get_image_size("512x768x3")


def test_convert_b64png_to_b64jpg_rgb_returns_valid_jpeg() -> None:
    result = convert_b64png_to_b64jpg(_make_png_b64("RGB"))

    img = Image.open(io.BytesIO(base64.b64decode(result)))
    assert img.format == "JPEG"


def test_convert_b64png_to_b64jpg_rgba_returns_valid_jpeg() -> None:
    result = convert_b64png_to_b64jpg(_make_png_b64("RGBA"))

    img = Image.open(io.BytesIO(base64.b64decode(result)))
    assert img.format == "JPEG"
    assert img.mode == "RGB"


def test_convert_b64png_to_b64jpg_custom_quality() -> None:
    high_q = convert_b64png_to_b64jpg(_make_png_b64(), quality=95)
    low_q = convert_b64png_to_b64jpg(_make_png_b64(), quality=10)

    assert len(high_q) >= len(low_q)


def test_convert_b64png_to_b64webp_returns_valid_webp() -> None:
    result = convert_b64png_to_b64webp(_make_png_b64())

    img = Image.open(io.BytesIO(base64.b64decode(result)))
    assert img.format == "WEBP"


def test_convert_b64png_to_b64webp_custom_quality() -> None:
    result = convert_b64png_to_b64webp(_make_png_b64(), quality=50)

    img = Image.open(io.BytesIO(base64.b64decode(result)))
    assert img.format == "WEBP"


def test_get_in_format_png_returns_same_string() -> None:
    assert _get_in_format("anystring", "png") == "anystring"


@pytest.mark.parametrize(("fmt", "expected"), [("jpg", "JPEG"), ("jpeg", "JPEG"), ("webp", "WEBP")])
def test_get_in_format_conversion(fmt: str, expected: str) -> None:
    b64_png = _make_png_b64()

    result = _get_in_format(b64_png, fmt)

    img = Image.open(io.BytesIO(base64.b64decode(result)))
    assert img.format == expected


def test_get_in_format_unknown_raises_value_error() -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        _get_in_format("abc", "gif")


# --- _add_body_config ---


def test_add_body_config_sets_prompt_from_remaining_text() -> None:
    body = ImagesRequest(prompt="draw a cat", model="sd")

    result = _add_body_config({}, body, "draw a cat")

    assert result["prompt"] == "draw a cat"  # pyright: ignore[reportTypedDictNotRequiredAccess]


def test_add_body_config_skips_prompt_when_already_in_settings() -> None:
    body = ImagesRequest(prompt="draw a cat", model="sd")

    result = _add_body_config({"prompt": "from settings"}, body, "draw a cat")

    assert result["prompt"] == "from settings"  # pyright: ignore[reportTypedDictNotRequiredAccess]


def test_add_body_config_sets_n_iter_from_body() -> None:
    body = ImagesRequest(prompt="test", model="sd", n=3)

    result = _add_body_config({}, body, "test")

    assert result["n_iter"] == 3  # pyright: ignore[reportTypedDictNotRequiredAccess]


def test_add_body_config_skips_n_iter_when_already_set() -> None:
    body = ImagesRequest(prompt="test", model="sd", n=3)

    result = _add_body_config({"n_iter": 1}, body, "test")

    assert result["n_iter"] == 1  # pyright: ignore[reportTypedDictNotRequiredAccess]


def test_add_body_config_sets_dimensions_from_size() -> None:
    body = ImagesRequest(prompt="test", model="sd", size="512x768")

    result = _add_body_config({}, body, "test")

    assert result["width"] == 512  # pyright: ignore[reportTypedDictNotRequiredAccess]
    assert result["height"] == 768  # pyright: ignore[reportTypedDictNotRequiredAccess]


def test_add_body_config_skips_size_when_auto() -> None:
    body = ImagesRequest(prompt="test", model="sd", size="auto")

    result = _add_body_config({}, body, "test")

    assert "width" not in result
    assert "height" not in result


def test_add_body_config_sets_steps_from_quality_high() -> None:
    body = ImagesRequest(prompt="test", model="sd", quality="high")

    result = _add_body_config({}, body, "test")

    assert result["steps"] == QualityLevel.high.value  # pyright: ignore[reportTypedDictNotRequiredAccess]


def test_add_body_config_sets_steps_from_quality_low() -> None:
    body = ImagesRequest(prompt="test", model="sd", quality="low")

    result = _add_body_config({}, body, "test")

    assert result["steps"] == QualityLevel.low.value  # pyright: ignore[reportTypedDictNotRequiredAccess]


def test_add_body_config_valid_b64_json_response_format_ok() -> None:
    body = ImagesRequest(prompt="test", model="sd", response_format="b64_json")

    result = _add_body_config({}, body, "test")

    assert isinstance(result, dict)


def test_add_body_config_invalid_response_format_raises_value_error() -> None:
    body = ImagesRequest(prompt="test", model="sd", response_format="url")

    with pytest.raises(ValueError, match="Response format"):
        _add_body_config({}, body, "test")


def test_add_body_config_partial_images_nonzero_raises_value_error() -> None:
    body = ImagesRequest(prompt="test", model="sd", partial_images=1)

    with pytest.raises(ValueError, match="Partial images"):
        _add_body_config({}, body, "test")


def test_add_body_config_stream_raises_value_error() -> None:
    body = ImagesRequest(prompt="test", model="sd", stream=True)

    with pytest.raises(ValueError, match="Stream"):
        _add_body_config({}, body, "test")


def test_add_body_config_does_not_modify_original_settings() -> None:
    original: dict[str, Any] = {}
    body = ImagesRequest(prompt="test", model="sd", n=2)

    _add_body_config(original, body, "test")  # pyright: ignore[reportArgumentType]

    assert "n_iter" not in original


@pytest.mark.asyncio
async def test_remove_background_returns_image_from_response() -> None:
    response_mock = AsyncMock()
    response_mock.json = AsyncMock(return_value={"image": "processed_b64"})

    client_mock = MagicMock()
    client_mock.post = AsyncMock(return_value=response_mock)

    session_mock = MagicMock()
    session_mock.__aenter__ = AsyncMock(return_value=client_mock)
    session_mock.__aexit__ = AsyncMock(return_value=None)

    with patch("server.services.stable_diffusion_service.ClientSession", return_value=session_mock):
        result = await remove_background_from_img("http://localhost:7860", "input_b64")

    assert result == "processed_b64"


@pytest.mark.asyncio
async def test_remove_background_returns_empty_string_when_no_image_key() -> None:
    response_mock = AsyncMock()
    response_mock.json = AsyncMock(return_value={})

    client_mock = MagicMock()
    client_mock.post = AsyncMock(return_value=response_mock)

    session_mock = MagicMock()
    session_mock.__aenter__ = AsyncMock(return_value=client_mock)
    session_mock.__aexit__ = AsyncMock(return_value=None)

    with patch("server.services.stable_diffusion_service.ClientSession", return_value=session_mock):
        result = await remove_background_from_img("http://localhost:7860", "input_b64")

    assert result == ""


@pytest.mark.asyncio
async def test_remove_background_returns_original_when_json_is_not_dict() -> None:
    response_mock = AsyncMock()
    # response.json() returns something without .get() -> AttributeError caught by except
    response_mock.json = AsyncMock(return_value=None)

    client_mock = MagicMock()
    client_mock.post = AsyncMock(return_value=response_mock)

    session_mock = MagicMock()
    session_mock.__aenter__ = AsyncMock(return_value=client_mock)
    session_mock.__aexit__ = AsyncMock(return_value=None)

    with patch("server.services.stable_diffusion_service.ClientSession", return_value=session_mock):
        result = await remove_background_from_img("http://localhost:7860", "original_b64")

    assert result == "original_b64"


def _make_session_mock(status: int = 200, response_data: dict[str, Any] | None = None) -> MagicMock:
    if response_data is None:
        response_data = {"images": ["fakepng"]}
    response_mock = AsyncMock()
    response_mock.status = status
    response_mock.json = AsyncMock(return_value=response_data)
    response_mock.content = b""

    client_mock = MagicMock()
    client_mock.post = AsyncMock(return_value=response_mock)

    session_mock = MagicMock()
    session_mock.__aenter__ = AsyncMock(return_value=client_mock)
    session_mock.__aexit__ = AsyncMock(return_value=None)
    return session_mock


@pytest.mark.asyncio
async def test_handler_returns_images_response_on_success() -> None:
    handler = _stable_diffusion_handler("http://localhost:7860", "my-model")
    body = ImagesRequest(prompt="draw a cat", model="sd")

    with patch("server.services.stable_diffusion_service.ClientSession", return_value=_make_session_mock()):
        result = await handler(body, None)

    assert len(result.data) == 1
    assert result.data[0].b64_json == "fakepng"


@pytest.mark.asyncio
async def test_handler_raises_500_when_sd_returns_non_200() -> None:
    handler = _stable_diffusion_handler("http://localhost:7860", "my-model")
    body = ImagesRequest(prompt="draw a cat", model="sd")

    with (
        patch("server.services.stable_diffusion_service.ClientSession", return_value=_make_session_mock(status=500)),
        pytest.raises(HTTPException) as exc_info,
    ):
        await handler(body, None)

    assert exc_info.value.status_code == 500


@pytest.mark.asyncio
async def test_handler_raises_422_for_invalid_body_config() -> None:
    handler = _stable_diffusion_handler("http://localhost:7860", "my-model")
    body = ImagesRequest(prompt="draw a cat", model="sd", response_format="url")

    with pytest.raises(HTTPException) as exc_info:
        await handler(body, None)

    assert exc_info.value.status_code == 422


@pytest.mark.asyncio
async def test_handler_raises_422_for_unsupported_output_format() -> None:
    handler = _stable_diffusion_handler("http://localhost:7860", "my-model")
    body = ImagesRequest(prompt="draw a cat", model="sd", output_format="jpeg")
    session_mock = _make_session_mock(response_data={"images": ["not-a-png"]})

    with (
        patch("server.services.stable_diffusion_service.ClientSession", return_value=session_mock),
        patch("server.services.stable_diffusion_service._get_in_format", side_effect=ValueError("Not Supported format")),  # pyright: ignore[reportPrivateUsage]
        pytest.raises(HTTPException) as exc_info,
    ):
        await handler(body, None)

    assert exc_info.value.status_code == 422


@pytest.mark.asyncio
async def test_handler_raises_422_for_invalid_sd_settings() -> None:
    handler = _stable_diffusion_handler("http://localhost:7860", "my-model")
    body = ImagesRequest(prompt="draw a cat", model="sd")

    with (
        patch("server.services.stable_diffusion_service._add_body_config", return_value={"sd_model_checkpoint": "x", "steps": 999}),  # pyright: ignore[reportPrivateUsage]
        pytest.raises(HTTPException) as exc_info,
    ):
        await handler(body, None)

    assert exc_info.value.status_code == 422


@pytest.mark.asyncio
async def test_handler_applies_background_removal_for_transparent() -> None:
    b64_png = _make_png_b64()
    handler = _stable_diffusion_handler("http://localhost:7860", "my-model")
    body = ImagesRequest(prompt="draw a cat", model="sd", background="transparent", output_format="png")
    session_mock = _make_session_mock(response_data={"images": [b64_png]})

    with (
        patch("server.services.stable_diffusion_service.ClientSession", return_value=session_mock),
        patch(
            "server.services.stable_diffusion_service.remove_background_from_img", new_callable=AsyncMock, return_value=b64_png
        ) as mock_rmbg,
    ):
        result = await handler(body, None)

    assert mock_rmbg.call_count == 1
    assert len(result.data) == 1


@pytest.mark.asyncio
async def test_handler_skips_background_removal_for_jpeg_output() -> None:
    b64_png = _make_png_b64()
    handler = _stable_diffusion_handler("http://localhost:7860", "my-model")
    body = ImagesRequest(prompt="draw a cat", model="sd", background="transparent", output_format="jpeg")
    session_mock = _make_session_mock(response_data={"images": [b64_png]})

    with (
        patch("server.services.stable_diffusion_service.ClientSession", return_value=session_mock),
        patch("server.services.stable_diffusion_service.remove_background_from_img", new_callable=AsyncMock) as mock_rmbg,
    ):
        await handler(body, None)

    assert mock_rmbg.call_count == 0


@pytest.mark.asyncio
async def test_handler_uses_prompt_as_revised_prompt() -> None:
    handler = _stable_diffusion_handler("http://localhost:7860", "my-model")
    body = ImagesRequest(prompt="a mountain landscape", model="sd")

    with patch("server.services.stable_diffusion_service.ClientSession", return_value=_make_session_mock()):
        result = await handler(body, None)

    assert result.data[0].revised_prompt == "a mountain landscape"


@pytest.mark.asyncio
async def test_handler_raises_500_on_unexpected_format_exception() -> None:
    handler = _stable_diffusion_handler("http://localhost:7860", "my-model")
    body = ImagesRequest(prompt="draw a cat", model="sd")

    with (
        patch("server.services.stable_diffusion_service.ClientSession", return_value=_make_session_mock()),
        patch("server.services.stable_diffusion_service._get_in_format", side_effect=RuntimeError("unexpected")),  # pyright: ignore[reportPrivateUsage]
        pytest.raises(HTTPException) as exc_info,
    ):
        await handler(body, None)

    assert exc_info.value.status_code == 500


@pytest.mark.asyncio
async def test_uninstall_instance_skips_endpoint_unregister_for_lora_model(svc: StableDiffusionService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    model_id = "Fantastic Landscapes"
    installed.models[model_id] = ModelInstalledInfo(
        id=model_id,
        type="lora",
        registered_name=model_id,
        options=InstallModelIn(spec={}),
        model_path=Path("/tmp/lora.safetensors"),
        registration_id="reg-lora",
    )
    svc.instances_info["default"].installed = installed
    deps["docker_service"].uninstall_docker = AsyncMock()

    await svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert deps["endpoint_registry"].unregister_image_generations.call_count == 0


@pytest.mark.asyncio
async def test_download_model_handles_pre_download_packet_with_no_size(svc: StableDiffusionService, tmp_path: Path) -> None:
    model = StableDiffusionModel(
        filetype="Stable-diffusion",
        type="txt2img",
        url="https://example.com/model.safetensors",
        filename="model.safetensors",
        size="1GB",
    )

    async def mock_download(*args: object) -> Any:
        yield PreDownloadPacket(file_bytes_size=None)  # pyright: ignore[reportArgumentType]

    svc.model_downloader.download.return_value = mock_download()  # pyright: ignore[reportAttributeAccessIssue]
    stream = MagicMock()

    with patch.object(svc, "_get_working_models_dir", return_value=tmp_path):  # pyright: ignore[reportPrivateUsage]
        local_path, filename = await svc._download_model(stream, model)  # pyright: ignore[reportPrivateUsage]

    assert local_path is None
    assert filename == ""


@pytest.mark.asyncio
async def test_uninstall_model_skips_endpoint_unregister_for_lora(svc: StableDiffusionService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    model_id = "Fantastic Landscapes"
    installed.models[model_id] = ModelInstalledInfo(
        id=model_id,
        type="lora",
        registered_name=model_id,
        options=InstallModelIn(spec={}),
        model_path=Path("/tmp/lora.safetensors"),
        registration_id="",
    )
    svc.instances_info["default"].installed = installed

    with (
        patch.object(svc, "is_model_installed_in_other_instance", return_value=True),
        patch.object(svc, "refresh_models", new_callable=AsyncMock),
    ):
        await svc._uninstall_model("default", model_id, UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert deps["endpoint_registry"].unregister_image_generations.call_count == 0
    assert model_id not in installed.models


@pytest.mark.asyncio
async def test_uninstall_model_purges_entry_when_model_path_is_empty(svc: StableDiffusionService) -> None:
    installed = _make_installed_info()
    model_id = "Fantastic Landscapes"
    svc.instances_info["default"].installed = installed
    svc.models_downloaded[model_id] = DownloadedInfo(model_path="")

    with patch.object(svc, "refresh_models", new_callable=AsyncMock):
        await svc._uninstall_model("default", model_id, UninstallModelIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert model_id not in svc.models_downloaded


# --- Edge cases for previously partial coverage ---


@pytest.mark.asyncio
async def test_install_instance_loads_default_models_for_new_instance(svc: StableDiffusionService, deps: dict[str, Any]) -> None:
    svc.instances_info["extra"] = Instance(None, None, {}, InstanceConfig())
    deps["docker_service"].get_docker_subnet.return_value = None
    deps["docker_service"].get_docker_container_name.return_value = "df-stable-diffusion-extra"
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=7860)
    deps["docker_service"].get_container_host.return_value = "localhost"
    deps["docker_service"].get_container_port.return_value = 7860
    deps["docker_service"].get_user_for_docker = AsyncMock(return_value=None)
    options = InstallServiceIn(spec={})  # no "hardware" key -> hits line 379 too

    with (
        patch.object(svc, "_download_image_or_set_progress", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_verify_docker_image", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
        patch.object(svc, "update_config", new_callable=AsyncMock),
        patch("server.services.stable_diffusion_service.platform.system", return_value="Linux"),
    ):
        promise = await svc._install_instance("extra", options)  # pyright: ignore[reportPrivateUsage]
        result = await promise.wait()

    assert "extra" in svc.models
    assert isinstance(result, InstalledInfo)


@pytest.mark.asyncio
async def test_list_models_skips_instances_not_in_requested_set(svc: StableDiffusionService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    svc.models["extra-instance"] = {"extra-model": next(iter(svc.models["default"].values()))}

    result = await svc.list_models("default", ListModelsFilters())

    model_ids = [m.id for m in result.list]
    assert "extra-model" not in model_ids


@pytest.mark.asyncio
async def test_get_model_initializes_empty_models_dict_for_instance(svc: StableDiffusionService) -> None:
    svc.instances_info["extra"] = Instance(None, None, {}, InstanceConfig())
    installed = _make_installed_info("extra")
    svc.instances_info["extra"].installed = installed
    svc.models["extra"] = {}

    with pytest.raises(HTTPException) as exc_info:
        await svc.get_model("extra", "nonexistent")

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_download_model_or_set_progress_breaks_on_non_download_chunk(svc: StableDiffusionService) -> None:
    existing_stream: Stream[Any] = Stream()
    existing_stream.emit({"type": "finish", "data": {}})
    existing_stream.close()
    svc.models_download_progress["model-id"] = existing_stream  # type: ignore[assignment]

    output_stream = MagicMock()
    model = next(iter(svc.models["default"].values()))

    local_path, _ = await svc._download_model_or_set_progress(output_stream, model, "model-id")  # pyright: ignore[reportPrivateUsage]

    assert output_stream.emit.call_count == 0
    assert local_path is None


@pytest.mark.asyncio
async def test_uninstall_model_makes_recursive_call_when_not_in_other_instance(svc: StableDiffusionService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    model_id = "Plant Milk Walnut"
    installed.models[model_id] = ModelInstalledInfo(
        id=model_id,
        type="txt2img",
        registered_name=model_id,
        options=InstallModelIn(spec={}),
        model_path=Path("/tmp/model.safetensors"),
        registration_id="reg-recursive",
    )
    svc.instances_info["default"].installed = installed

    with (
        patch.object(svc, "is_model_installed_in_other_instance", return_value=False),
        patch.object(svc, "refresh_models", new_callable=AsyncMock),
    ):
        await svc._uninstall_model("default", model_id, UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert model_id not in installed.models
    assert deps["endpoint_registry"].unregister_image_generations.call_count == 1
    assert deps["endpoint_registry"].unregister_image_generations.call_args == call(model_id, "reg-recursive")


@pytest.mark.asyncio
async def test_resolve_custom_model_size_returns_formatted_size(svc: StableDiffusionService, deps: dict[str, Any]) -> None:
    with patch("server.services.stable_diffusion_service.fetch_file_size_from_url", new=AsyncMock(return_value="3.0 GB")):
        result = await svc._resolve_custom_model_size({"url": "https://civitai.com/api/download/123"})  # pyright: ignore[reportPrivateUsage]

    assert result == "3.0 GB"


@pytest.mark.asyncio
async def test_resolve_custom_model_size_returns_none_on_exception(svc: StableDiffusionService, deps: dict[str, Any]) -> None:
    with patch("server.services.stable_diffusion_service.fetch_file_size_from_url", new=AsyncMock(side_effect=Exception("fail"))):
        result = await svc._resolve_custom_model_size({"url": "https://example.com"})  # pyright: ignore[reportPrivateUsage]

    assert result is None
