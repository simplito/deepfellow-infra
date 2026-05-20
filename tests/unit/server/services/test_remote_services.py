# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from server.models.models import InstallModelIn, ListModelsFilters, UninstallModelIn
from server.models.services import InstallServiceIn, UninstallServiceIn
from server.services.base2_service import CustomModel, Instance, InstanceConfig
from server.services.googleai_service import GoogleAIService
from server.services.openai_service import OpenAIService
from server.services.remote_service import (
    BaseServiceOptions,
    DefaultRemoteServiceOptions,
    DownloadedInfo,
    InstalledInfo,
    ModelInstalledInfo,
    RemoteModel,
    get_model_props,
)


@pytest.fixture
def deps() -> dict[str, Any]:
    return {
        "config": MagicMock(),
        "endpoint_registry": MagicMock(),
        "service_provider": MagicMock(),
        "model_downloader": MagicMock(),
        "docker_service": MagicMock(),
        "hardware": MagicMock(gpus=[]),
    }


@pytest.fixture
def openai_svc(deps: dict[str, Any]) -> OpenAIService:
    return OpenAIService(**deps)


@pytest.fixture
def google_svc(deps: dict[str, Any]) -> GoogleAIService:
    return GoogleAIService(**deps)


# --- OpenAIService ---


def test_openai_get_type(openai_svc: OpenAIService) -> None:
    assert openai_svc.get_type() == "openai"


def test_openai_is_cloud(openai_svc: OpenAIService) -> None:
    assert openai_svc.is_cloud is True
    assert openai_svc.is_cloud_service() is True


def test_openai_get_description(openai_svc: OpenAIService) -> None:
    assert "OpenAI" in openai_svc.get_description()


def test_openai_get_default_url(openai_svc: OpenAIService) -> None:
    assert "openai.com" in openai_svc.get_default_url()


def test_openai_models_registry_contains_gpt4o(openai_svc: OpenAIService) -> None:
    assert "gpt-4o" in openai_svc.get_models_registry().models


def test_openai_models_registry_contains_embeddings(openai_svc: OpenAIService) -> None:
    registry = openai_svc.get_models_registry()
    embedding_models = [k for k, v in registry.models.items() if v.type == "embedding"]
    assert len(embedding_models) > 0


def test_openai_default_models_loaded(openai_svc: OpenAIService) -> None:
    assert "default" in openai_svc.models
    assert len(openai_svc.models["default"]) > 0


def test_openai_service_has_docker_false(openai_svc: OpenAIService) -> None:
    assert openai_svc.service_has_docker() is False


def test_openai_get_size_empty(openai_svc: OpenAIService) -> None:
    assert openai_svc.get_size() == ""


def test_openai_add_custom_model(openai_svc: OpenAIService) -> None:
    custom = CustomModel(
        id="c-1",
        data={"id": "my-gpt", "type": "llm"},
    )

    openai_svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    assert "my-gpt" in openai_svc.models["default"]


def test_openai_add_custom_model_duplicate_raises(openai_svc: OpenAIService) -> None:
    custom = CustomModel(id="c-2", data={"id": "dup-model", "type": "llm"})
    openai_svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    with pytest.raises(HTTPException) as exc_info:
        openai_svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400


def test_google_get_type(google_svc: GoogleAIService) -> None:
    assert google_svc.get_type() == "google"


def test_google_is_cloud(google_svc: GoogleAIService) -> None:
    assert google_svc.is_cloud is True
    assert google_svc.is_cloud_service() is True


def test_google_get_description(google_svc: GoogleAIService) -> None:
    assert "Google" in google_svc.get_description()


def test_google_get_default_url(google_svc: GoogleAIService) -> None:
    assert "google" in google_svc.get_default_url()


def test_google_api_version(google_svc: GoogleAIService) -> None:
    assert google_svc.api_version == "v1beta/openai/"


def test_google_models_registry_contains_gemini(google_svc: GoogleAIService) -> None:
    registry = google_svc.get_models_registry()

    gemini_models = [k for k in registry.models if k.startswith("gemini")]
    assert len(gemini_models) > 0


def test_google_default_models_loaded(google_svc: GoogleAIService) -> None:
    assert "default" in google_svc.models
    assert len(google_svc.models["default"]) > 0


def test_google_service_has_docker_false(google_svc: GoogleAIService) -> None:
    assert google_svc.service_has_docker() is False


def test_default_options_bearer_header() -> None:
    opts = DefaultRemoteServiceOptions(api_url="https://api.example.com", api_key="secret")

    assert opts.headers == {"Authorization": "Bearer secret"}


def test_default_options_empty_key_still_produces_header() -> None:
    opts = DefaultRemoteServiceOptions(api_url="https://api.example.com")

    assert "Authorization" in opts.headers


# --- get_model_props helper ---


def test_get_model_props_llm_includes_chat_completions() -> None:
    model = RemoteModel(type="llm", completions=True, legacy_completions=True, responses=True, messages=True)

    props = get_model_props(model)

    assert "/v1/chat/completions" in props.endpoints


def test_get_model_props_respects_disabled_endpoints() -> None:
    model = RemoteModel(type="llm", completions=False, legacy_completions=False, responses=False, messages=False)

    props = get_model_props(model)

    assert "/v1/chat/completions" not in props.endpoints
    assert "/v1/completions" not in props.endpoints


def test_base_service_options_headers_raises_not_implemented() -> None:
    opts = BaseServiceOptions(api_url="https://example.com")

    with pytest.raises(NotImplementedError):
        _ = opts.headers


def test_model_installed_info_get_info_returns_model_info() -> None:
    options = InstallModelIn(spec={"alias": "test"})
    info = ModelInstalledInfo(
        id="gpt-4o",
        registered_name="gpt-4o",
        type="llm",
        options=options,
        completions=True,
        legacy_completions=True,
        registration_id="reg-1",
    )

    result = info.get_info()

    assert result.spec == options.spec
    assert result.registration_id == "reg-1"


def test_openai_get_spec_has_api_url_and_api_key_fields(openai_svc: OpenAIService) -> None:
    spec = openai_svc.get_spec()

    field_names = [f.name for f in spec.fields]
    assert "api_url" in field_names
    assert "api_key" in field_names


def test_openai_get_model_spec_has_alias_field(openai_svc: OpenAIService) -> None:
    spec = openai_svc.get_model_spec()

    field_names = [f.name for f in spec.fields]
    assert "alias" in field_names


def test_openai_get_custom_model_spec_not_none_and_has_id_field(openai_svc: OpenAIService) -> None:
    spec = openai_svc.get_custom_model_spec()

    assert spec is not None
    field_names = [f.name for f in spec.fields]
    assert "id" in field_names
    assert "type" in field_names


def test_get_installed_info_when_not_installed_returns_false(openai_svc: OpenAIService) -> None:
    result = openai_svc.get_installed_info("default")

    assert result is False


def test_get_installed_info_when_installed_returns_spec(openai_svc: OpenAIService) -> None:
    spec = {"api_url": "https://api.openai.com/", "api_key": "k"}
    installed = InstalledInfo(
        models={},
        options=InstallServiceIn(spec=spec),
        parsed_options=DefaultRemoteServiceOptions(api_url="https://api.openai.com/", api_key="k"),
    )
    openai_svc.instances_info["default"].installed = installed
    result = openai_svc.get_installed_info("default")

    assert result == spec


def test_generate_instance_config_with_none_returns_empty_config(openai_svc: OpenAIService) -> None:
    config = openai_svc._generate_instance_config(None, None)  # pyright: ignore[reportPrivateUsage]

    assert config.options is None
    assert config.models == []
    assert config.custom is None


def test_generate_instance_config_with_info_includes_options_and_models(openai_svc: OpenAIService) -> None:
    options = InstallServiceIn(spec={"api_url": "https://api.openai.com/"})
    parsed = DefaultRemoteServiceOptions(api_url="https://api.openai.com/")
    model_info = ModelInstalledInfo(
        id="gpt-4o",
        registered_name="gpt-4o",
        type="llm",
        options=InstallModelIn(spec={}),
        completions=True,
        legacy_completions=True,
        registration_id="reg-1",
    )
    info = InstalledInfo(models={"gpt-4o": model_info}, options=options, parsed_options=parsed)
    config = openai_svc._generate_instance_config(info, None)  # pyright: ignore[reportPrivateUsage]

    assert config.options == options
    assert config.models is not None
    assert len(config.models) == 1
    assert config.models[0].model_id == "gpt-4o"


def test_load_download_info_returns_downloaded_info(openai_svc: OpenAIService) -> None:
    result = openai_svc._load_download_info({})  # pyright: ignore[reportPrivateUsage]

    assert isinstance(result, DownloadedInfo)


@pytest.mark.asyncio
async def test_install_instance_returns_installed_info(openai_svc: OpenAIService) -> None:
    options = InstallServiceIn(spec={"api_url": "https://api.openai.com/", "api_key": "k"})

    promise = await openai_svc._install_instance("default", options)  # pyright: ignore[reportPrivateUsage]
    result = await promise.wait()

    assert isinstance(result, InstalledInfo)
    assert openai_svc.service_downloaded is True


@pytest.mark.asyncio
async def test_install_instance_sets_default_url_if_missing(openai_svc: OpenAIService) -> None:
    options = InstallServiceIn(spec={"api_key": "k"})

    promise = await openai_svc._install_instance("default", options)  # pyright: ignore[reportPrivateUsage]
    result = await promise.wait()

    assert "api_url" in result.options.spec
    assert result.options.spec["api_url"] == openai_svc.get_default_url()


@pytest.mark.asyncio
async def test_install_instance_loads_default_models_for_new_instance(openai_svc: OpenAIService) -> None:
    openai_svc.models.pop("new-instance", None)
    options = InstallServiceIn(spec={"api_url": "https://api.openai.com/"})
    openai_svc.instances_info["new-instance"] = Instance(None, None, {}, InstanceConfig())

    promise = await openai_svc._install_instance("new-instance", options)  # pyright: ignore[reportPrivateUsage]
    await promise.wait()

    assert "new-instance" in openai_svc.models
    assert len(openai_svc.models["new-instance"]) > 0


def _make_installed(api_url: str = "https://api.openai.com/", api_key: str = "k") -> InstalledInfo:
    return InstalledInfo(
        models={},
        options=InstallServiceIn(spec={"api_url": api_url, "api_key": api_key}),
        parsed_options=DefaultRemoteServiceOptions(api_url=api_url, api_key=api_key),
    )


@pytest.mark.asyncio
async def test_uninstall_instance_without_purge_clears_installed(openai_svc: OpenAIService) -> None:
    openai_svc.instances_info["default"].installed = _make_installed()

    await openai_svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert openai_svc.instances_info["default"].installed is None


@pytest.mark.asyncio
async def test_uninstall_instance_with_purge_resets_service_downloaded(openai_svc: OpenAIService) -> None:
    openai_svc.instances_info["default"].installed = _make_installed()
    openai_svc.service_downloaded = True
    openai_svc._clear_working_dir = AsyncMock()  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage]

    await openai_svc._uninstall_instance("default", UninstallServiceIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert openai_svc.service_downloaded is False
    assert openai_svc._clear_working_dir.call_count == 1  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage]


@pytest.mark.asyncio
async def test_uninstall_instance_with_purge_deletes_non_default_instance(openai_svc: OpenAIService) -> None:
    openai_svc.instances_info["gpu-1"] = Instance(None, None, {}, InstanceConfig())
    openai_svc.instances_info["gpu-1"].installed = _make_installed()

    openai_svc._clear_working_dir = AsyncMock()  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage]

    await openai_svc._uninstall_instance("gpu-1", UninstallServiceIn(purge=True))  # pyright: ignore[reportPrivateUsage]
    assert "gpu-1" not in openai_svc.instances_info


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model_id", "model_type", "unregister_method"),
    [
        ("gpt-4o", "llm", "unregister_chat_completion"),
        ("tts-1", "tts", "unregister_audio_speech"),
        ("whisper-1", "stt", "unregister_audio_transcriptions"),
        ("dall-e-2", "txt2img", "unregister_image_generations"),
        ("text-embedding-ada-002", "embedding", "unregister_embeddings"),
    ],
)
async def test_uninstall_instance_unregisters_endpoint_per_type(
    openai_svc: OpenAIService,
    deps: dict[str, Any],
    model_id: str,
    model_type: str,
    unregister_method: str,
) -> None:
    model_info = ModelInstalledInfo(
        id=model_id,
        registered_name=model_id,
        type=model_type,
        options=InstallModelIn(spec={}),
        completions=True,
        legacy_completions=True,
        registration_id="reg-1",
    )
    installed = InstalledInfo(
        models={model_id: model_info},
        options=InstallServiceIn(spec={"api_url": "https://api.openai.com/"}),
        parsed_options=DefaultRemoteServiceOptions(api_url="https://api.openai.com/"),
    )
    openai_svc.instances_info["default"].installed = installed
    openai_svc._uninstall_model = AsyncMock()  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage]

    await openai_svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert getattr(deps["endpoint_registry"], unregister_method).call_count == 1


@pytest.mark.asyncio
async def test_uninstall_instance_when_not_installed_does_not_raise(openai_svc: OpenAIService) -> None:
    assert openai_svc.instances_info["default"].installed is None

    await openai_svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert openai_svc.instances_info["default"].installed is None


@pytest.mark.asyncio
async def test_uninstall_instance_skips_uninstall_model_when_shared_across_instances(openai_svc: OpenAIService) -> None:
    model_info = ModelInstalledInfo(
        id="gpt-4o",
        registered_name="gpt-4o",
        type="llm",
        options=InstallModelIn(spec={}),
        completions=True,
        legacy_completions=True,
        registration_id="reg-1",
    )
    installed = InstalledInfo(
        models={"gpt-4o": model_info},
        options=InstallServiceIn(spec={"api_url": "https://api.openai.com/"}),
        parsed_options=DefaultRemoteServiceOptions(api_url="https://api.openai.com/"),
    )
    openai_svc.instances_info["default"].installed = installed
    openai_svc.instances_info["gpu-1"] = Instance(None, None, {}, InstanceConfig())
    openai_svc.instances_info["gpu-1"].installed = InstalledInfo(
        models={"gpt-4o": model_info},
        options=InstallServiceIn(spec={"api_url": "https://api.openai.com/"}),
        parsed_options=DefaultRemoteServiceOptions(api_url="https://api.openai.com/"),
    )
    openai_svc._uninstall_model = AsyncMock()  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage]

    await openai_svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert openai_svc._uninstall_model.call_count == 0  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage]


def test_add_custom_model_creates_models_dict_for_new_instance(openai_svc: OpenAIService) -> None:
    openai_svc.models.pop("new-inst", None)
    openai_svc.instances_info["new-inst"] = Instance(None, None, {}, InstanceConfig())
    custom = CustomModel(id="cm-1", data={"id": "my-llm", "type": "llm"})

    openai_svc._add_custom_model("new-inst", custom)  # pyright: ignore[reportPrivateUsage]

    assert "new-inst" in openai_svc.models
    assert "my-llm" in openai_svc.models["new-inst"]


def test_remove_custom_model_removes_from_models(openai_svc: OpenAIService) -> None:
    custom = CustomModel(id="cm-2", data={"id": "my-llm", "type": "llm"})
    openai_svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]
    assert "my-llm" in openai_svc.models["default"]

    openai_svc._remove_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    assert "my-llm" not in openai_svc.models["default"]


def test_remove_custom_model_when_in_use_raises(openai_svc: OpenAIService) -> None:
    custom = CustomModel(id="cm-3", data={"id": "my-llm2", "type": "llm"})
    openai_svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]
    model_info = ModelInstalledInfo(
        id="my-llm2",
        registered_name="my-llm2",
        type="llm",
        options=InstallModelIn(spec={}),
        completions=True,
        legacy_completions=True,
        registration_id="reg-1",
    )
    openai_svc.instances_info["default"].installed = InstalledInfo(
        models={"my-llm2": model_info},
        options=InstallServiceIn(spec={"api_url": "https://api.openai.com/"}),
        parsed_options=DefaultRemoteServiceOptions(api_url="https://api.openai.com/"),
    )

    with pytest.raises(HTTPException) as exc:
        openai_svc._remove_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    assert exc.value.status_code == 400


# --- list_models ---


@pytest.mark.asyncio
async def test_list_models_unknown_instance_raises(openai_svc: OpenAIService) -> None:
    with pytest.raises(HTTPException) as exc:
        await openai_svc.list_models("nonexistent", ListModelsFilters())

    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_list_models_returns_models_for_default_instance(openai_svc: OpenAIService) -> None:
    openai_svc.instances_info["default"].installed = _make_installed()

    result = await openai_svc.list_models("default", ListModelsFilters())

    assert len(result.list) > 0


@pytest.mark.asyncio
async def test_list_models_none_input_uses_all_instances(openai_svc: OpenAIService) -> None:
    openai_svc.instances_info["default"].installed = _make_installed()

    result = await openai_svc.list_models(None, ListModelsFilters())

    assert len(result.list) > 0


@pytest.mark.asyncio
async def test_list_models_filters_installed_only(openai_svc: OpenAIService) -> None:
    installed = _make_installed()
    model_info = ModelInstalledInfo(
        id="gpt-4o",
        registered_name="gpt-4o",
        type="llm",
        options=InstallModelIn(spec={}),
        completions=True,
        legacy_completions=True,
        registration_id="reg-1",
    )
    installed.models["gpt-4o"] = model_info
    openai_svc.instances_info["default"].installed = installed

    result = await openai_svc.list_models("default", ListModelsFilters(installed=True))

    ids = [m.id for m in result.list]
    assert "gpt-4o" in ids
    for model in result.list:
        assert model.installed is not False


@pytest.mark.asyncio
async def test_list_models_with_list_of_instances(openai_svc: OpenAIService) -> None:
    openai_svc.instances_info["default"].installed = _make_installed()

    result = await openai_svc.list_models(["default"], ListModelsFilters())

    assert len(result.list) > 0


@pytest.mark.asyncio
async def test_get_model_not_found_raises(openai_svc: OpenAIService) -> None:
    openai_svc.instances_info["default"].installed = _make_installed()

    with pytest.raises(HTTPException) as exc:
        await openai_svc.get_model("default", "nonexistent-model")

    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_get_model_success(openai_svc: OpenAIService) -> None:
    openai_svc.instances_info["default"].installed = _make_installed()

    result = await openai_svc.get_model("default", "gpt-4o")

    assert result.id == "gpt-4o"
    assert result.type == "llm"


@pytest.mark.asyncio
async def test_get_model_when_installed_returns_install_info(openai_svc: OpenAIService) -> None:
    installed = _make_installed()
    model_info = ModelInstalledInfo(
        id="gpt-4o",
        registered_name="gpt-4o",
        type="llm",
        options=InstallModelIn(spec={}),
        completions=True,
        legacy_completions=True,
        registration_id="reg-1",
    )
    installed.models["gpt-4o"] = model_info
    openai_svc.instances_info["default"].installed = installed

    result = await openai_svc.get_model("default", "gpt-4o")

    assert result.installed is not False


@pytest.mark.asyncio
async def test_install_model_already_installed_returns_ok(openai_svc: OpenAIService) -> None:
    installed = _make_installed()
    installed.models["gpt-4o"] = ModelInstalledInfo(
        id="gpt-4o",
        registered_name="gpt-4o",
        type="llm",
        options=InstallModelIn(spec={}),
        completions=True,
        legacy_completions=True,
        registration_id="reg-1",
    )
    openai_svc.instances_info["default"].installed = installed

    promise = await openai_svc._install_model("default", "gpt-4o", InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
    result = await promise.wait()

    assert result.status == "OK"
    assert "Already" in result.details


@pytest.mark.asyncio
async def test_install_model_not_found_raises(openai_svc: OpenAIService) -> None:
    openai_svc.instances_info["default"].installed = _make_installed()

    with pytest.raises(HTTPException) as exc:
        await openai_svc._install_model("default", "nonexistent", InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]

    assert exc.value.status_code == 400


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model_id", "register_method"),
    [
        ("gpt-4o", "register_chat_completion_as_proxy"),
        ("tts-1", "register_audio_speech_as_proxy"),
        ("whisper-1", "register_audio_transcriptions_as_proxy"),
        ("dall-e-2", "register_image_generations_as_proxy"),
        ("text-embedding-ada-002", "register_embeddings_as_proxy"),
    ],
)
async def test_install_model_registers_correct_endpoint(
    openai_svc: OpenAIService,
    deps: dict[str, Any],
    model_id: str,
    register_method: str,
) -> None:
    openai_svc.instances_info["default"].installed = _make_installed()

    promise = await openai_svc._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
    result = await promise.wait()

    assert result.status == "OK"
    assert getattr(deps["endpoint_registry"], register_method).call_count == 1


@pytest.mark.asyncio
async def test_install_model_with_alias_uses_alias_as_registered_name(
    openai_svc: OpenAIService,
    deps: dict[str, Any],
) -> None:
    openai_svc.instances_info["default"].installed = _make_installed()

    promise = await openai_svc._install_model("default", "gpt-4o", InstallModelIn(spec={"alias": "my-alias"}))  # pyright: ignore[reportPrivateUsage]
    await promise.wait()

    installed_info = openai_svc.instances_info["default"].installed
    assert installed_info is not None
    assert installed_info.models["gpt-4o"].registered_name == "my-alias"


@pytest.mark.asyncio
async def test_install_model_adds_to_models_downloaded(openai_svc: OpenAIService) -> None:
    openai_svc.instances_info["default"].installed = _make_installed()

    promise = await openai_svc._install_model("default", "gpt-4o", InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
    await promise.wait()

    assert "gpt-4o" in openai_svc.models_downloaded


@pytest.mark.asyncio
async def test_uninstall_model_removes_from_info(openai_svc: OpenAIService) -> None:
    installed = _make_installed()
    installed.models["gpt-4o"] = ModelInstalledInfo(
        id="gpt-4o",
        registered_name="gpt-4o",
        type="llm",
        options=InstallModelIn(spec={}),
        completions=True,
        legacy_completions=True,
        registration_id="reg-1",
    )
    openai_svc.instances_info["default"].installed = installed

    await openai_svc._uninstall_model("default", "gpt-4o", UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert "gpt-4o" not in installed.models


@pytest.mark.asyncio
async def test_uninstall_model_with_purge_removes_from_downloaded(openai_svc: OpenAIService) -> None:
    installed = _make_installed()
    installed.models["gpt-4o"] = ModelInstalledInfo(
        id="gpt-4o",
        registered_name="gpt-4o",
        type="llm",
        options=InstallModelIn(spec={}),
        completions=True,
        legacy_completions=True,
        registration_id="reg-1",
    )
    openai_svc.instances_info["default"].installed = installed
    openai_svc.models_downloaded["gpt-4o"] = DownloadedInfo()
    await openai_svc._uninstall_model("default", "gpt-4o", UninstallModelIn(purge=True))  # pyright: ignore[reportPrivateUsage]
    assert "gpt-4o" not in openai_svc.models_downloaded


@pytest.mark.asyncio
async def test_uninstall_model_not_in_info_is_noop(openai_svc: OpenAIService) -> None:
    openai_svc.instances_info["default"].installed = _make_installed()

    await openai_svc._uninstall_model("default", "gpt-4o", UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert "gpt-4o" not in openai_svc.instances_info["default"].installed.models  # type: ignore[union-attr]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model_id", "model_type", "unregister_method"),
    [
        ("gpt-4o", "llm", "unregister_chat_completion"),
        ("tts-1", "tts", "unregister_audio_speech"),
        ("whisper-1", "stt", "unregister_audio_transcriptions"),
        ("dall-e-2", "txt2img", "unregister_image_generations"),
        ("text-embedding-ada-002", "embedding", "unregister_embeddings"),
    ],
)
async def test_uninstall_model_unregisters_correct_endpoint(
    openai_svc: OpenAIService,
    deps: dict[str, Any],
    model_id: str,
    model_type: str,
    unregister_method: str,
) -> None:
    installed = _make_installed()
    installed.models[model_id] = ModelInstalledInfo(
        id=model_id,
        registered_name=model_id,
        type=model_type,
        options=InstallModelIn(spec={}),
        completions=True,
        legacy_completions=True,
        registration_id="reg-1",
    )
    openai_svc.instances_info["default"].installed = installed

    await openai_svc._uninstall_model("default", model_id, UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert getattr(deps["endpoint_registry"], unregister_method).call_count == 1


@pytest.mark.asyncio
async def test_list_models_skips_instances_not_in_filter(openai_svc: OpenAIService) -> None:
    openai_svc.instances_info["default"].installed = _make_installed()
    openai_svc.instances_info["other"] = Instance(None, None, {}, InstanceConfig())
    openai_svc.instances_info["other"].installed = _make_installed()
    openai_svc.models["other"] = openai_svc.get_models_registry().models.copy()

    result = await openai_svc.list_models("default", ListModelsFilters())

    service_ids = {m.service for m in result.list}
    assert all("other" not in s for s in service_ids)


@pytest.mark.asyncio
async def test_get_model_initialises_empty_models_dict_for_instance(openai_svc: OpenAIService) -> None:
    installed = _make_installed()
    openai_svc.instances_info["default"].installed = installed
    openai_svc.models.pop("default", None)

    with pytest.raises(HTTPException) as exc:
        await openai_svc.get_model("default", "gpt-4o")

    assert exc.value.status_code == 400
    assert "default" in openai_svc.models


@pytest.mark.asyncio
async def test_install_model_initialises_empty_models_dict_for_instance(openai_svc: OpenAIService) -> None:
    installed = _make_installed()
    openai_svc.instances_info["default"].installed = installed
    openai_svc.models.pop("default", None)

    with pytest.raises(HTTPException) as exc:
        await openai_svc._install_model("default", "nonexistent", InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]

    assert exc.value.status_code == 400
    assert "default" in openai_svc.models
