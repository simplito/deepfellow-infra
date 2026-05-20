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

from server.models.models import InstallModelIn, ListModelsFilters, UninstallModelIn
from server.models.services import InstallServiceIn, UninstallServiceIn
from server.services.base2_service import CustomModel, Instance, InstanceConfig
from server.services.speaches_ai_service import (
    DownloadedInfo,
    InstalledInfo,
    ModelInstalledInfo,
    SpeachesAIOptions,
    SpeachesAIService,
    SpeachesModel,
    SrvSpeachesCustomModel,
    _const,  # pyright: ignore[reportPrivateUsage]
)
from server.utils.core import DownloadedPacket, PreDownloadPacket, Stream, StreamChunkProgress
from server.utils.hardware import NvidiaGpuInfo


def _make_installed_info(instance: str = "default") -> InstalledInfo:
    docker = MagicMock()
    docker.name = f"speaches-{instance}"
    return InstalledInfo(
        docker=docker,
        models={},
        options=InstallServiceIn(spec={}),
        parsed_options=SpeachesAIOptions(),
        container_host="localhost",
        container_port=8000,
        docker_exposed_port=8000,
        base_url="http://localhost:8000",
    )


def _make_model_installed_info(model_id: str, model_type: str = "tts") -> ModelInstalledInfo:
    return ModelInstalledInfo(
        id=model_id,
        type=model_type,
        registered_name=model_id,
        options=InstallModelIn(spec={}),
        registration_id="",
        model_path=Path("/tmp"),
        default_model=None,
        langs_models=None,
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
def svc(deps: dict[str, Any]) -> SpeachesAIService:
    return SpeachesAIService(**deps)


def test_get_type(svc: SpeachesAIService) -> None:
    assert svc.get_type() == "speaches-ai"


def test_get_description_not_empty(svc: SpeachesAIService) -> None:
    assert svc.get_description()


def test_service_has_docker(svc: SpeachesAIService) -> None:
    assert svc.service_has_docker() is True


def test_is_not_cloud_service(svc: SpeachesAIService) -> None:
    assert svc.is_cloud_service() is False


def test_const_has_gpu_image() -> None:
    assert "gpu" in _const.images
    assert "cuda" in _const.images["gpu"].name.lower()


def test_const_has_cpu_image() -> None:
    assert "cpu" in _const.images
    assert _const.images["cpu"].name


def test_const_speech_models_not_empty() -> None:
    assert len(_const.audio_speech_models) > 0


def test_const_transcription_models_not_empty() -> None:
    assert len(_const.audio_transcriptions_models) > 0


def test_speaches_options_hardware_defaults_none() -> None:
    opts = SpeachesAIOptions()

    assert opts.hardware is None


def test_get_spec_has_hardware_field(svc: SpeachesAIService) -> None:
    spec = svc.get_spec()

    field_names = [f.name for f in spec.fields]
    assert "hardware" in field_names


def test_get_model_spec_has_alias(svc: SpeachesAIService) -> None:
    spec = svc.get_model_spec()

    field_names = [f.name for f in spec.fields]
    assert "alias" in field_names


def test_get_model_spec_alias_is_optional(svc: SpeachesAIService) -> None:
    spec = svc.get_model_spec()

    alias_field = next(f for f in spec.fields if f.name == "alias")
    assert alias_field.required is False


def test_get_custom_model_spec_not_none(svc: SpeachesAIService) -> None:
    assert svc.get_custom_model_spec() is not None


def test_get_custom_model_spec_has_required_fields(svc: SpeachesAIService) -> None:
    spec = svc.get_custom_model_spec()

    assert spec is not None
    field_names = {f.name for f in spec.fields}
    assert {"id", "default_model", "langs_models"} <= field_names


# --- Default model loading ---


def test_default_models_loaded_on_init(svc: SpeachesAIService) -> None:
    assert "default" in svc.models
    assert len(svc.models["default"]) > 0


def test_default_models_include_tts_models(svc: SpeachesAIService) -> None:
    models = svc.models["default"]
    tts_models = [m for m in models.values() if m.type == "tts"]
    assert len(tts_models) > 0


def test_default_models_include_stt_models(svc: SpeachesAIService) -> None:
    models = svc.models["default"]
    stt_models = [m for m in models.values() if m.type == "stt"]
    assert len(stt_models) > 0


# --- _get_image ---


def test_get_image_gpu_true_returns_gpu_image(svc: SpeachesAIService) -> None:
    image = svc._get_image(True)  # pyright: ignore[reportPrivateUsage]

    assert image == _const.images["gpu"]


def test_get_image_gpu_false_returns_cpu_image(svc: SpeachesAIService) -> None:
    image = svc._get_image(False)  # pyright: ignore[reportPrivateUsage]

    assert image == _const.images["cpu"]


def test_srv_speaches_custom_model_valid() -> None:
    m = SrvSpeachesCustomModel(id="my-custom", default_model="speaches-ai/piper-en_US-john-medium")

    assert m.id == "my-custom"
    assert m.default_model == "speaches-ai/piper-en_US-john-medium"
    assert m.langs_models is None


def test_srv_speaches_custom_model_with_langs_models() -> None:
    m = SrvSpeachesCustomModel(
        id="multi-lang",
        default_model="speaches-ai/piper-en_US-john-medium",
        langs_models={"pl": "speaches-ai/piper-pl_PL-darkman-medium"},
    )

    assert "pl" in m.langs_models  # type: ignore[operator]


def test_add_custom_model_stores_model(svc: SpeachesAIService) -> None:
    base_model_id = next(iter(svc.models["default"]))
    custom = CustomModel(
        id="uuid-1",
        data={"id": "my-tts", "default_model": base_model_id},
    )

    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    assert "my-tts" in svc.models["default"]


def test_add_custom_model_stores_custom_id(svc: SpeachesAIService) -> None:
    base_model_id = next(iter(svc.models["default"]))
    custom = CustomModel(id="uuid-99", data={"id": "my-tts2", "default_model": base_model_id})

    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    assert svc.models["default"]["my-tts2"].custom == "uuid-99"


def test_add_custom_model_type_is_tts(svc: SpeachesAIService) -> None:
    base_model_id = next(iter(svc.models["default"]))
    custom = CustomModel(id="uuid-1", data={"id": "my-tts3", "default_model": base_model_id})

    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    assert svc.models["default"]["my-tts3"].type == "tts"


def test_add_custom_model_duplicate_raises_http_400(svc: SpeachesAIService) -> None:
    base_model_id = next(iter(svc.models["default"]))
    custom = CustomModel(id="uuid-1", data={"id": "dup-tts", "default_model": base_model_id})

    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(HTTPException) as exc_info:
        svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400


def test_add_custom_model_unknown_default_model_raises_http_400(svc: SpeachesAIService) -> None:
    custom = CustomModel(id="uuid-1", data={"id": "bad-tts", "default_model": "nonexistent-model-xyz"})

    with pytest.raises(HTTPException) as exc_info:
        svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400


def test_model_installed_info_get_info() -> None:
    model = ModelInstalledInfo(
        id="my-tts",
        type="tts",
        registered_name="my-tts",
        options=InstallModelIn(spec={}),
        registration_id="reg-42",
        model_path=Path("/tmp"),
        default_model=None,
        langs_models=None,
    )

    info = model.get_info()

    assert info.registration_id == "reg-42"


def test_get_size_returns_cpu_size(svc: SpeachesAIService) -> None:
    sizes = svc.get_size()

    assert "cpu" in sizes
    assert sizes["cpu"] == _const.images["cpu"].size  # pyright: ignore[reportArgumentType]


def test_get_size_returns_gpu_when_nvidia_gpu_present(deps: dict[str, Any]) -> None:
    gpu = NvidiaGpuInfo(id=0, name="NVIDIA RTX 3080", vram="10GB")
    deps["hardware"].gpus = [gpu]
    svc = SpeachesAIService(**deps)

    sizes = svc.get_size()

    assert "gpu" in sizes
    assert sizes["gpu"] == _const.images["gpu"].size  # pyright: ignore[reportArgumentType]


def test_get_size_no_gpu_key_without_gpu_support(svc: SpeachesAIService) -> None:
    sizes = svc.get_size()

    assert "gpu" not in sizes


def test_get_installed_info_returns_spec_when_installed(svc: SpeachesAIService) -> None:
    installed = _make_installed_info()
    installed.options.spec["hardware"] = False
    svc.instances_info["default"].installed = installed

    result = svc.get_installed_info("default")

    assert result == installed.options.spec


def test_get_installed_info_delegates_when_not_installed(svc: SpeachesAIService) -> None:
    svc.instances_info["default"].installed = None

    with patch.object(svc, "_get_service_installed_info", return_value=False) as mock:  # pyright: ignore[reportPrivateUsage]
        result = svc.get_installed_info("default")

    assert mock.call_count == 1
    assert mock.call_args == call("default")
    assert result is False


def test_generate_instance_config_no_info(svc: SpeachesAIService) -> None:
    config = svc._generate_instance_config(None, None)  # pyright: ignore[reportPrivateUsage]

    assert config.options is None
    assert config.models == []


def test_generate_instance_config_with_info_and_models(svc: SpeachesAIService) -> None:
    info = _make_installed_info()
    info.models["my-tts"] = _make_model_installed_info("my-tts")

    config = svc._generate_instance_config(info, None)  # pyright: ignore[reportPrivateUsage]

    assert config.options == info.options
    assert len(config.models or []) == 1


def test_load_download_info_returns_downloaded_info(svc: SpeachesAIService) -> None:
    result = svc._load_download_info({"model_path": "/x"})  # pyright: ignore[reportPrivateUsage]

    assert isinstance(result, DownloadedInfo)
    assert result.model_path == "/x"


def test_get_docker_compose_file_path_raises_400_with_model_id(svc: SpeachesAIService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    with pytest.raises(HTTPException) as exc_info:
        svc.get_docker_compose_file_path("default", "some-model")

    assert exc_info.value.status_code == 400


def test_get_docker_compose_file_path_returns_path_without_model_id(svc: SpeachesAIService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    expected = Path("/some/path/docker-compose.yml")
    deps["docker_service"].get_docker_compose_file_path.return_value = expected

    result = svc.get_docker_compose_file_path("default", None)

    assert result == expected


def test_add_custom_model_creates_models_dict_for_new_instance(svc: SpeachesAIService) -> None:
    svc.instances_info["extra"] = Instance(None, None, {}, InstanceConfig())
    # empty string default_model is falsy → skips default_model validation
    custom = CustomModel(id="uuid-1", data={"id": "my-tts", "default_model": ""})

    svc._add_custom_model("extra", custom)  # pyright: ignore[reportPrivateUsage]

    assert "my-tts" in svc.models["extra"]


def test_add_custom_model_unknown_lang_model_raises_400(svc: SpeachesAIService) -> None:
    base_model_id = next(iter(svc.models["default"]))
    custom = CustomModel(
        id="uuid-1",
        data={
            "id": "my-tts",
            "default_model": base_model_id,
            "langs_models": {"pl": "nonexistent-model-xyz"},
        },
    )

    with pytest.raises(HTTPException) as exc_info:
        svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400


def test_add_custom_model_with_valid_langs_models(svc: SpeachesAIService) -> None:
    base_model_id = next(iter(svc.models["default"]))
    custom = CustomModel(
        id="uuid-1",
        data={
            "id": "multi-lang-tts",
            "default_model": base_model_id,
            "langs_models": {"pl": base_model_id},
        },
    )

    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    assert "multi-lang-tts" in svc.models["default"]


def test_remove_custom_model_deletes_entry(svc: SpeachesAIService) -> None:
    base_model_id = next(iter(svc.models["default"]))
    custom = CustomModel(id="uuid-1", data={"id": "to-remove", "default_model": base_model_id})
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    svc._remove_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    assert "to-remove" not in svc.models["default"]


def test_remove_custom_model_raises_400_when_in_use(svc: SpeachesAIService) -> None:
    base_model_id = next(iter(svc.models["default"]))
    custom = CustomModel(id="uuid-1", data={"id": "in-use", "default_model": base_model_id})
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]
    installed = _make_installed_info()
    installed.models["in-use"] = _make_model_installed_info("in-use")
    svc.instances_info["default"].installed = installed

    with pytest.raises(HTTPException) as exc_info:
        svc._remove_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_list_models_returns_all_models_for_valid_instance(svc: SpeachesAIService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    result = await svc.list_models("default", ListModelsFilters())

    assert len(result.list) == len(svc.models["default"])


@pytest.mark.asyncio
async def test_list_models_raises_404_for_unknown_instance(svc: SpeachesAIService) -> None:
    with pytest.raises(HTTPException) as exc_info:
        await svc.list_models("nonexistent", ListModelsFilters())

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_list_models_filters_to_installed_only(svc: SpeachesAIService) -> None:
    installed = _make_installed_info()
    model_id = next(iter(svc.models["default"]))
    installed.models[model_id] = _make_model_installed_info(model_id)
    svc.instances_info["default"].installed = installed

    result = await svc.list_models("default", ListModelsFilters(installed=True))

    assert len(result.list) >= 1
    assert all(bool(m.installed) for m in result.list)


@pytest.mark.asyncio
async def test_list_models_filters_to_not_installed(svc: SpeachesAIService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    result = await svc.list_models("default", ListModelsFilters(installed=False))

    assert all(not m.installed for m in result.list)


@pytest.mark.asyncio
async def test_list_models_with_list_of_instances(svc: SpeachesAIService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    result = await svc.list_models(["default"], ListModelsFilters())

    assert len(result.list) == len(svc.models["default"])


@pytest.mark.asyncio
async def test_list_models_none_instance_uses_all(svc: SpeachesAIService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    result = await svc.list_models(None, ListModelsFilters())

    assert len(result.list) > 0


@pytest.mark.asyncio
async def test_list_models_skips_instances_not_in_requested_set(svc: SpeachesAIService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    svc.models["extra-instance"] = {"extra-model": SpeachesModel(id="extra-model", type="tts")}

    result = await svc.list_models("default", ListModelsFilters())

    model_ids = [m.id for m in result.list]
    assert "extra-model" not in model_ids


# --- get_model ---


@pytest.mark.asyncio
async def test_get_model_returns_model_for_valid_id(svc: SpeachesAIService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    model_id = next(iter(svc.models["default"]))

    result = await svc.get_model("default", model_id)

    assert result.id == model_id


@pytest.mark.asyncio
async def test_get_model_raises_400_for_unknown_model(svc: SpeachesAIService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    with pytest.raises(HTTPException) as exc_info:
        await svc.get_model("default", "nonexistent-model")

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_get_model_returns_installed_info_when_in_info_models(svc: SpeachesAIService) -> None:
    installed = _make_installed_info()
    model_id = next(iter(svc.models["default"]))
    mi = _make_model_installed_info(model_id)
    mi.registration_id = "reg-42"
    installed.models[model_id] = mi
    svc.instances_info["default"].installed = installed

    result = await svc.get_model("default", model_id)

    assert result.installed is not None
    assert result.installed.registration_id == "reg-42"  # pyright: ignore[reportAttributeAccessIssue]


# --- choose_model_for_language ---


def test_choose_model_for_language_returns_matched_model() -> None:
    with patch("server.services.speaches_ai_service.detect", return_value="pl"):
        result = SpeachesAIService.choose_model_for_language(
            "Hello world",
            "default-model",
            {"pl": "pl-model"},
        )

    assert result == "pl-model"


def test_choose_model_for_language_returns_default_when_no_match() -> None:
    with patch("server.services.speaches_ai_service.detect", return_value="fr"):
        result = SpeachesAIService.choose_model_for_language(
            "Hello world",
            "default-model",
            {"pl": "pl-model"},
        )

    assert result == "default-model"


def test_choose_model_for_language_samples_long_text() -> None:
    long_text = "one two three four five six seven eight nine ten eleven twelve"
    captured: list[str] = []

    def capture_detect(text: str) -> str:
        captured.append(text)
        return "en"

    with patch("server.services.speaches_ai_service.detect", side_effect=capture_detect):
        SpeachesAIService.choose_model_for_language(long_text, "default-model", {})

    assert len(captured) == 1
    assert len(captured[0].split(" ")) <= 9


@pytest.mark.asyncio
async def test_stop_instance_does_nothing_when_not_installed(svc: SpeachesAIService) -> None:
    svc.instances_info["default"].installed = None

    with patch.object(svc, "_stop_docker", new_callable=AsyncMock) as mock_stop:  # pyright: ignore[reportPrivateUsage]
        await svc.stop_instance("default")

    assert mock_stop.call_count == 0


@pytest.mark.asyncio
async def test_stop_instance_calls_stop_docker_when_installed(svc: SpeachesAIService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    with patch.object(svc, "_stop_docker", new_callable=AsyncMock) as mock_stop:  # pyright: ignore[reportPrivateUsage]
        await svc.stop_instance("default")

    assert mock_stop.call_count == 1
    assert mock_stop.call_args == call(installed.docker)


@pytest.mark.asyncio
async def test_download_model_emits_start_and_end_progress(svc: SpeachesAIService) -> None:
    model = SpeachesModel(id="test-model", type="tts", size="100MB")
    stream = MagicMock()

    async def empty_gen(*args: object) -> object:  # type: ignore[misc]
        return
        yield

    svc.model_downloader.hugging_face_repo_with_blobs_downloader.download.return_value = empty_gen()  # pyright: ignore[reportAttributeAccessIssue]

    await svc._download_model(stream, model, "test-model", Path("/tmp"))  # pyright: ignore[reportPrivateUsage]

    assert stream.emit.call_count >= 2


@pytest.mark.asyncio
async def test_download_model_emits_incremental_progress_for_downloaded_packet(svc: SpeachesAIService) -> None:
    model = SpeachesModel(id="test-model", type="tts", size="100MB")
    stream = MagicMock()

    async def mock_gen(*args: object) -> object:  # type: ignore[misc]
        yield DownloadedPacket(downloaded_bytes_size=512)

    svc.model_downloader.hugging_face_repo_with_blobs_downloader.download.return_value = mock_gen()  # pyright: ignore[reportAttributeAccessIssue]

    await svc._download_model(stream, model, "test-model", Path("/tmp"))  # pyright: ignore[reportPrivateUsage]

    assert stream.emit.call_count > 2


@pytest.mark.asyncio
async def test_download_model_updates_max_from_pre_download_packet(svc: SpeachesAIService) -> None:
    model = SpeachesModel(id="test-model", type="tts", size="")
    stream = MagicMock()

    async def mock_gen(*args: object) -> object:  # type: ignore[misc]
        yield PreDownloadPacket(file_bytes_size=1024)

    svc.model_downloader.hugging_face_repo_with_blobs_downloader.download.return_value = mock_gen()  # pyright: ignore[reportAttributeAccessIssue]

    await svc._download_model(stream, model, "test-model", Path("/tmp"))  # pyright: ignore[reportPrivateUsage]

    assert stream.emit.call_count >= 2


# --- _download_model_or_set_progress ---


@pytest.mark.asyncio
async def test_download_model_or_set_progress_starts_fresh_download(svc: SpeachesAIService) -> None:
    model = SpeachesModel(id="test-model", type="tts", size="")
    stream = MagicMock()

    with patch.object(svc, "_download_model", new_callable=AsyncMock) as mock_dl:  # pyright: ignore[reportPrivateUsage]
        await svc._download_model_or_set_progress(stream, model, "test-model", Path("/tmp"))  # pyright: ignore[reportPrivateUsage]

    assert mock_dl.call_count == 1
    assert "test-model" not in svc.models_download_progress


@pytest.mark.asyncio
async def test_download_model_or_set_progress_forwards_existing_stream(svc: SpeachesAIService) -> None:
    existing_stream: Stream[StreamChunkProgress] = Stream()  # type: ignore[type-arg]
    chunk = StreamChunkProgress(type="progress", stage="download", value=0.5, data={})
    existing_stream.emit(chunk)
    existing_stream.close()
    svc.models_download_progress["test-model"] = existing_stream  # type: ignore[assignment]

    output_stream = MagicMock()
    model = SpeachesModel(id="test-model", type="tts", size="")

    await svc._download_model_or_set_progress(output_stream, model, "test-model", Path("/tmp"))  # pyright: ignore[reportPrivateUsage]

    assert output_stream.emit.call_count == 1
    assert output_stream.emit.call_args == call(chunk)


# --- _install_model ---


@pytest.mark.asyncio
async def test_install_model_returns_already_installed(svc: SpeachesAIService) -> None:
    installed = _make_installed_info()
    model_id = next(iter(svc.models["default"]))
    installed.models[model_id] = _make_model_installed_info(model_id)
    svc.instances_info["default"].installed = installed

    promise = await svc._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]

    result = await promise.wait()
    assert result.status == "OK"
    assert "Already installed" in result.details


@pytest.mark.asyncio
async def test_install_model_raises_400_for_unknown_model(svc: SpeachesAIService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    with pytest.raises(HTTPException) as exc_info:
        await svc._install_model("default", "nonexistent-model", InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_install_model_registers_tts_endpoint(svc: SpeachesAIService, deps: dict[str, Any], tmp_path: Path) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    tts_model_id = next(m.id for m in svc.models["default"].values() if m.type == "tts")

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_get_working_dir", return_value=tmp_path),  # pyright: ignore[reportPrivateUsage]
    ):
        promise = await svc._install_model("default", tts_model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert deps["endpoint_registry"].register_audio_speech.call_count == 1


@pytest.mark.asyncio
async def test_install_model_registers_stt_proxy(svc: SpeachesAIService, deps: dict[str, Any], tmp_path: Path) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    stt_model_id = next(m.id for m in svc.models["default"].values() if m.type == "stt")

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_get_working_dir", return_value=tmp_path),  # pyright: ignore[reportPrivateUsage]
    ):
        promise = await svc._install_model("default", stt_model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert deps["endpoint_registry"].register_audio_transcriptions_as_proxy.call_count == 1


@pytest.mark.asyncio
async def test_install_model_uses_alias_as_registered_name(svc: SpeachesAIService, tmp_path: Path) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    tts_model_id = next(m.id for m in svc.models["default"].values() if m.type == "tts")

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_get_working_dir", return_value=tmp_path),  # pyright: ignore[reportPrivateUsage]
    ):
        promise = await svc._install_model("default", tts_model_id, InstallModelIn(spec={"alias": "my-alias"}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert installed.models[tts_model_id].registered_name == "my-alias"


@pytest.mark.asyncio
async def test_install_model_marks_model_as_downloaded(svc: SpeachesAIService, tmp_path: Path) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    tts_model_id = next(m.id for m in svc.models["default"].values() if m.type == "tts")

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_get_working_dir", return_value=tmp_path),  # pyright: ignore[reportPrivateUsage]
    ):
        promise = await svc._install_model("default", tts_model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert tts_model_id in svc.models_downloaded


@pytest.mark.asyncio
async def test_install_model_skips_download_for_custom_with_langs_models(svc: SpeachesAIService, tmp_path: Path) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    base_model_id = next(iter(svc.models["default"]))
    custom_model_id = "my-custom-tts"
    svc.models["default"][custom_model_id] = SpeachesModel(
        id=custom_model_id,
        type="tts",
        custom="uuid-1",
        langs_models={"en": base_model_id},
    )

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock) as mock_dl,  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_get_working_dir", return_value=tmp_path),  # pyright: ignore[reportPrivateUsage]
    ):
        promise = await svc._install_model("default", custom_model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert mock_dl.call_count == 0


@pytest.mark.asyncio
async def test_uninstall_model_unregisters_tts_endpoint(svc: SpeachesAIService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    mi = _make_model_installed_info("my-tts", "tts")
    mi.registration_id = "reg-1"
    installed.models["my-tts"] = mi
    svc.instances_info["default"].installed = installed

    await svc._uninstall_model("default", "my-tts", UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert "my-tts" not in installed.models
    assert deps["endpoint_registry"].unregister_audio_speech.call_count == 1
    assert deps["endpoint_registry"].unregister_audio_speech.call_args == call("my-tts", "reg-1")


@pytest.mark.asyncio
async def test_uninstall_model_unregisters_stt_endpoint(svc: SpeachesAIService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    mi = _make_model_installed_info("my-stt", "stt")
    mi.registration_id = "reg-2"
    installed.models["my-stt"] = mi
    svc.instances_info["default"].installed = installed

    await svc._uninstall_model("default", "my-stt", UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert deps["endpoint_registry"].unregister_audio_transcriptions.call_count == 1
    assert deps["endpoint_registry"].unregister_audio_transcriptions.call_args == call("my-stt", "reg-2")


@pytest.mark.asyncio
async def test_uninstall_model_raises_409_when_used_in_custom_model_default(svc: SpeachesAIService) -> None:
    installed = _make_installed_info()
    installed.models["base-tts"] = _make_model_installed_info("base-tts", "tts")
    custom_mi = _make_model_installed_info("custom-tts", "tts")
    custom_mi.default_model = "base-tts"
    installed.models["custom-tts"] = custom_mi
    svc.instances_info["default"].installed = installed

    with pytest.raises(HTTPException) as exc_info:
        await svc._uninstall_model("default", "base-tts", UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 409


@pytest.mark.asyncio
async def test_uninstall_model_raises_409_when_used_in_custom_model_langs(svc: SpeachesAIService) -> None:
    installed = _make_installed_info()
    installed.models["base-tts"] = _make_model_installed_info("base-tts", "tts")
    custom_mi = _make_model_installed_info("custom-tts", "tts")
    custom_mi.langs_models = {"pl": "base-tts"}
    installed.models["custom-tts"] = custom_mi
    svc.instances_info["default"].installed = installed

    with pytest.raises(HTTPException) as exc_info:
        await svc._uninstall_model("default", "base-tts", UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 409


@pytest.mark.asyncio
async def test_uninstall_model_purges_model_directory(svc: SpeachesAIService, tmp_path: Path) -> None:
    installed = _make_installed_info()
    model_dir = tmp_path / "my-tts"
    model_dir.mkdir()
    installed.models["my-tts"] = _make_model_installed_info("my-tts", "tts")
    svc.instances_info["default"].installed = installed
    svc.models_downloaded["my-tts"] = DownloadedInfo(model_path=str(model_dir))

    await svc._uninstall_model("default", "my-tts", UninstallModelIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert not model_dir.exists()
    assert "my-tts" not in svc.models_downloaded


@pytest.mark.asyncio
async def test_uninstall_model_does_nothing_for_unknown_model(svc: SpeachesAIService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    await svc._uninstall_model("default", "nonexistent-model", UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert deps["endpoint_registry"].unregister_audio_speech.call_count == 0


@pytest.mark.asyncio
async def test_install_instance_raises_503_on_cuda_version_error(svc: SpeachesAIService, deps: dict[str, Any], tmp_path: Path) -> None:
    deps["docker_service"].get_docker_subnet.return_value = None
    deps["docker_service"].get_docker_container_name.return_value = "df-speaches"
    deps["docker_service"].install_and_run_docker = AsyncMock(
        side_effect=RuntimeError("docker failed", ["", "", "", "cuda>=12.9 required, please update your driver"])
    )
    options = InstallServiceIn(spec={"hardware": False})

    with (
        patch.object(svc, "_download_image_or_set_progress", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_verify_docker_image", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
        patch.object(svc, "_get_working_dir", return_value=tmp_path),  # pyright: ignore[reportPrivateUsage]
    ):
        promise = await svc._install_instance("default", options)  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(HTTPException) as exc_info:
            await promise.wait()

    assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_install_instance_reraises_non_cuda_runtime_error(svc: SpeachesAIService, deps: dict[str, Any], tmp_path: Path) -> None:
    deps["docker_service"].get_docker_subnet.return_value = None
    deps["docker_service"].get_docker_container_name.return_value = "df-speaches"
    deps["docker_service"].install_and_run_docker = AsyncMock(side_effect=RuntimeError("generic docker error"))
    options = InstallServiceIn(spec={"hardware": False})

    with (
        patch.object(svc, "_download_image_or_set_progress", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_verify_docker_image", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
        patch.object(svc, "_get_working_dir", return_value=tmp_path),  # pyright: ignore[reportPrivateUsage]
    ):
        promise = await svc._install_instance("default", options)  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(RuntimeError):
            await promise.wait()


@pytest.mark.asyncio
async def test_install_instance_returns_installed_info(svc: SpeachesAIService, deps: dict[str, Any], tmp_path: Path) -> None:
    deps["docker_service"].get_docker_subnet.return_value = None
    deps["docker_service"].get_docker_container_name.return_value = "df-speaches"
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=8000)
    deps["docker_service"].get_container_host.return_value = "localhost"
    deps["docker_service"].get_container_port.return_value = 8000
    options = InstallServiceIn(spec={"hardware": False})

    with (
        patch.object(svc, "_download_image_or_set_progress", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_verify_docker_image", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
        patch.object(svc, "_get_working_dir", return_value=tmp_path),  # pyright: ignore[reportPrivateUsage]
    ):
        promise = await svc._install_instance("default", options)  # pyright: ignore[reportPrivateUsage]
        result = await promise.wait()

    assert isinstance(result, InstalledInfo)


@pytest.mark.asyncio
async def test_install_instance_calls_docker_install(svc: SpeachesAIService, deps: dict[str, Any], tmp_path: Path) -> None:
    deps["docker_service"].get_docker_subnet.return_value = None
    deps["docker_service"].get_docker_container_name.return_value = "df-speaches"
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=8000)
    deps["docker_service"].get_container_host.return_value = "localhost"
    deps["docker_service"].get_container_port.return_value = 8000
    options = InstallServiceIn(spec={"hardware": False})

    with (
        patch.object(svc, "_download_image_or_set_progress", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_verify_docker_image", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
        patch.object(svc, "_get_working_dir", return_value=tmp_path),  # pyright: ignore[reportPrivateUsage]
    ):
        promise = await svc._install_instance("default", options)  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert deps["docker_service"].install_and_run_docker.call_count == 1


@pytest.mark.asyncio
async def test_install_instance_loads_default_models_for_new_instance(svc: SpeachesAIService, deps: dict[str, Any], tmp_path: Path) -> None:
    svc.instances_info["extra"] = Instance(None, None, {}, InstanceConfig())
    deps["docker_service"].get_docker_subnet.return_value = None
    deps["docker_service"].get_docker_container_name.return_value = "df-speaches-extra"
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=8000)
    deps["docker_service"].get_container_host.return_value = "localhost"
    deps["docker_service"].get_container_port.return_value = 8000
    options = InstallServiceIn(spec={})

    with (
        patch.object(svc, "_download_image_or_set_progress", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_verify_docker_image", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
        patch.object(svc, "_get_working_dir", return_value=tmp_path),  # pyright: ignore[reportPrivateUsage]
    ):
        promise = await svc._install_instance("extra", options)  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert "extra" in svc.models
    assert len(svc.models["extra"]) > 0


# --- _uninstall_instance ---


@pytest.mark.asyncio
async def test_uninstall_instance_does_nothing_when_not_installed(svc: SpeachesAIService, deps: dict[str, Any]) -> None:
    svc.instances_info["default"].installed = None
    deps["docker_service"].uninstall_docker = AsyncMock()

    await svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert deps["docker_service"].uninstall_docker.call_count == 0


@pytest.mark.asyncio
async def test_uninstall_instance_unregisters_tts_model(svc: SpeachesAIService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    mi = _make_model_installed_info("my-tts", "tts")
    mi.registration_id = "reg-tts"
    installed.models["my-tts"] = mi
    svc.instances_info["default"].installed = installed
    deps["docker_service"].uninstall_docker = AsyncMock()

    with patch.object(svc, "_uninstall_model", new_callable=AsyncMock):  # pyright: ignore[reportPrivateUsage]
        await svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert deps["endpoint_registry"].unregister_audio_speech.call_count == 1
    assert deps["endpoint_registry"].unregister_audio_speech.call_args == call("my-tts", "reg-tts")


@pytest.mark.asyncio
async def test_uninstall_instance_unregisters_stt_model(svc: SpeachesAIService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    mi = _make_model_installed_info("my-stt", "stt")
    mi.registration_id = "reg-stt"
    installed.models["my-stt"] = mi
    svc.instances_info["default"].installed = installed
    deps["docker_service"].uninstall_docker = AsyncMock()

    with patch.object(svc, "_uninstall_model", new_callable=AsyncMock):  # pyright: ignore[reportPrivateUsage]
        await svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert deps["endpoint_registry"].unregister_audio_transcriptions.call_count == 1
    assert deps["endpoint_registry"].unregister_audio_transcriptions.call_args == call("my-stt", "reg-stt")


@pytest.mark.asyncio
async def test_uninstall_instance_calls_docker_uninstall(svc: SpeachesAIService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    deps["docker_service"].uninstall_docker = AsyncMock()

    await svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert deps["docker_service"].uninstall_docker.call_count == 1


@pytest.mark.asyncio
async def test_uninstall_instance_purge_clears_working_dir(svc: SpeachesAIService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    deps["docker_service"].uninstall_docker = AsyncMock()
    deps["docker_service"].remove_image = AsyncMock()

    with patch.object(svc, "_clear_working_dir", new_callable=AsyncMock) as mock_clear:  # pyright: ignore[reportPrivateUsage]
        await svc._uninstall_instance("default", UninstallServiceIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert svc.service_downloaded is False
    assert mock_clear.call_count == 1


@pytest.mark.asyncio
async def test_uninstall_instance_purge_removes_non_default_instance(svc: SpeachesAIService, deps: dict[str, Any]) -> None:
    svc.instances_info["extra"] = Instance(None, None, {}, InstanceConfig())
    installed = _make_installed_info("extra")
    svc.instances_info["extra"].installed = installed
    svc.models["extra"] = {}
    deps["docker_service"].uninstall_docker = AsyncMock()

    await svc._uninstall_instance("extra", UninstallServiceIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert "extra" not in svc.instances_info


@pytest.mark.asyncio
async def test_uninstall_instance_purge_resets_default_instance(svc: SpeachesAIService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    deps["docker_service"].uninstall_docker = AsyncMock()
    deps["docker_service"].remove_image = AsyncMock()

    with patch.object(svc, "_clear_working_dir", new_callable=AsyncMock):  # pyright: ignore[reportPrivateUsage]
        await svc._uninstall_instance("default", UninstallServiceIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert "default" in svc.instances_info
    assert svc.instances_info["default"].installed is None


# --- Edge cases for additional branch coverage ---


@pytest.mark.asyncio
async def test_get_model_initializes_models_dict_for_instance_without_one(svc: SpeachesAIService) -> None:
    svc.instances_info["extra"] = Instance(None, None, {}, InstanceConfig())
    installed = _make_installed_info("extra")
    svc.instances_info["extra"].installed = installed
    # "extra" is not in svc.models → branch on line 779 creates empty dict

    with pytest.raises(HTTPException) as exc_info:
        await svc.get_model("extra", "nonexistent")

    assert exc_info.value.status_code == 400
    assert "extra" in svc.models


@pytest.mark.asyncio
async def test_download_model_or_set_progress_breaks_on_non_download_chunk(svc: SpeachesAIService) -> None:
    existing_stream: Stream[Any] = Stream()  # type: ignore[type-arg]
    existing_stream.emit({"type": "finish", "data": {}})
    existing_stream.close()
    svc.models_download_progress["test-model"] = existing_stream  # type: ignore[assignment]
    output_stream = MagicMock()
    model = SpeachesModel(id="test-model", type="tts", size="")

    await svc._download_model_or_set_progress(output_stream, model, "test-model", Path("/tmp"))  # pyright: ignore[reportPrivateUsage]

    assert output_stream.emit.call_count == 0


@pytest.mark.asyncio
async def test_install_model_initializes_models_dict_for_instance_without_one(svc: SpeachesAIService) -> None:
    svc.instances_info["extra"] = Instance(None, None, {}, InstanceConfig())
    installed = _make_installed_info("extra")
    svc.instances_info["extra"].installed = installed
    # "extra" not in svc.models → branch on line 843 creates empty dict

    with pytest.raises(HTTPException) as exc_info:
        await svc._install_model("extra", "nonexistent", InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400
    assert "extra" in svc.models


@pytest.mark.asyncio
async def test_tts_on_request_callback_calls_post_json(svc: SpeachesAIService, deps: dict[str, Any], tmp_path: Path) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    tts_model_id = next(m.id for m in svc.models["default"].values() if m.type == "tts")
    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_get_working_dir", return_value=tmp_path),  # pyright: ignore[reportPrivateUsage]
    ):
        promise = await svc._install_model("default", tts_model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    on_request = deps["endpoint_registry"].register_audio_speech.call_args.kwargs["endpoint"].on_request
    mock_body = MagicMock()
    mock_body.voice = "alloy"
    mock_body.input = "Hello world"

    with patch("server.services.speaches_ai_service.post_json", new_callable=AsyncMock) as mock_post_json:
        await on_request(mock_body, None)

    assert mock_post_json.call_count == 1


@pytest.mark.asyncio
async def test_tts_on_request_callback_selects_language_model(svc: SpeachesAIService, deps: dict[str, Any], tmp_path: Path) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    base_model_id = next(iter(svc.models["default"]))
    custom_model_id = "lang-tts"
    svc.models["default"][custom_model_id] = SpeachesModel(
        id=custom_model_id,
        type="tts",
        custom="uuid-1",
        default_model=base_model_id,
        langs_models={"pl": base_model_id},
    )
    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_get_working_dir", return_value=tmp_path),  # pyright: ignore[reportPrivateUsage]
    ):
        promise = await svc._install_model("default", custom_model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    on_request = deps["endpoint_registry"].register_audio_speech.call_args.kwargs["endpoint"].on_request
    mock_body = MagicMock()
    mock_body.voice = ""
    mock_body.input = "Witaj świecie jak się masz dzisiaj"
    with (
        patch("server.services.speaches_ai_service.post_json", new_callable=AsyncMock),
        patch("server.services.speaches_ai_service.detect", return_value="pl"),
    ):
        await on_request(mock_body, None)

    assert mock_body.model == base_model_id


@pytest.mark.asyncio
async def test_uninstall_instance_skips_uninstall_model_when_installed_in_other_instance(
    svc: SpeachesAIService, deps: dict[str, Any]
) -> None:
    installed = _make_installed_info()
    mi = _make_model_installed_info("my-tts", "tts")
    installed.models["my-tts"] = mi
    svc.instances_info["default"].installed = installed
    deps["docker_service"].uninstall_docker = AsyncMock()
    with (
        patch.object(svc, "is_model_installed_in_other_instance", return_value=True),
        patch.object(svc, "_uninstall_model", new_callable=AsyncMock) as mock_uninstall,  # pyright: ignore[reportPrivateUsage]
    ):
        await svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert mock_uninstall.call_count == 0


@pytest.mark.asyncio
async def test_download_model_ignores_zero_byte_downloaded_packet(svc: SpeachesAIService) -> None:
    model = SpeachesModel(id="test-model", type="tts", size="100MB")
    stream = MagicMock()

    async def mock_gen(*args: object) -> object:  # type: ignore[misc]
        yield DownloadedPacket(downloaded_bytes_size=0)

    svc.model_downloader.hugging_face_repo_with_blobs_downloader.download.return_value = mock_gen()  # pyright: ignore[reportAttributeAccessIssue]
    await svc._download_model(stream, model, "test-model", Path("/tmp"))  # pyright: ignore[reportPrivateUsage]
    assert stream.emit.call_count == 2


@pytest.mark.asyncio
async def test_download_model_skips_set_max_for_zero_file_bytes_size(svc: SpeachesAIService) -> None:
    model = SpeachesModel(id="test-model", type="tts", size="")
    stream = MagicMock()

    async def mock_gen(*args: object) -> object:  # type: ignore[misc]
        yield PreDownloadPacket(file_bytes_size=0)

    svc.model_downloader.hugging_face_repo_with_blobs_downloader.download.return_value = mock_gen()  # pyright: ignore[reportAttributeAccessIssue]
    await svc._download_model(stream, model, "test-model", Path("/tmp"))  # pyright: ignore[reportPrivateUsage]
    assert stream.emit.call_count == 2


@pytest.mark.asyncio
async def test_uninstall_model_purge_skips_rmtree_for_dot_path(svc: SpeachesAIService) -> None:
    installed = _make_installed_info()
    installed.models["my-tts"] = _make_model_installed_info("my-tts", "tts")
    svc.instances_info["default"].installed = installed
    svc.models_downloaded["my-tts"] = DownloadedInfo(model_path=".")

    with patch("server.services.speaches_ai_service.shutil.rmtree") as mock_rmtree:
        await svc._uninstall_model("default", "my-tts", UninstallModelIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert mock_rmtree.call_count == 0
    assert "my-tts" not in svc.models_downloaded
