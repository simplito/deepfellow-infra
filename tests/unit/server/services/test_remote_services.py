# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from server.endpointregistry import EndpointRegistry
from server.models.models import InstallModelIn, ListModelsFilters, UninstallModelIn
from server.models.services import InstallServiceIn, UninstallServiceIn
from server.services.base2_service import CustomModel, Instance, InstanceConfig
from server.services.deepseek_service import DeepSeekService
from server.services.googleai_service import GoogleAIService
from server.services.kimi_service import KimiService
from server.services.openai_service import OpenAIService
from server.services.remote_service import (
    BaseServiceOptions,
    DefaultRemoteServiceOptions,
    DownloadedInfo,
    InstalledInfo,
    LiveModelEntry,
    ModelInstalledInfo,
    RemoteCustomModel,
    RemoteModel,
    RemoteModelType,
    get_model_props,
)


def _make_model_installed_info(model_id: str = "gpt-4o", registration_id: str = "reg-1") -> ModelInstalledInfo:
    return ModelInstalledInfo(
        id=model_id,
        registered_name=model_id,
        type="llm",
        options=InstallModelIn(spec={}),
        completions=True,
        legacy_completions=True,
        registration_id=registration_id,
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
        "services_manager": MagicMock(),
    }


@pytest.fixture
def openai_svc(deps: dict[str, Any]) -> OpenAIService:
    return OpenAIService(**deps)


@pytest.fixture
def google_svc(deps: dict[str, Any]) -> GoogleAIService:
    return GoogleAIService(**deps)


@pytest.fixture
def deepseek_svc(deps: dict[str, Any]) -> DeepSeekService:
    return DeepSeekService(**deps)


@pytest.fixture
def kimi_svc(deps: dict[str, Any]) -> KimiService:
    return KimiService(**deps)


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


def test_deepseek_get_type(deepseek_svc: DeepSeekService) -> None:
    assert deepseek_svc.get_type() == "deepseek"


def test_deepseek_is_cloud(deepseek_svc: DeepSeekService) -> None:
    assert deepseek_svc.is_cloud is True
    assert deepseek_svc.is_cloud_service() is True


def test_deepseek_get_description(deepseek_svc: DeepSeekService) -> None:
    assert "DeepSeek" in deepseek_svc.get_description()


def test_deepseek_get_default_url(deepseek_svc: DeepSeekService) -> None:
    assert "deepseek.com" in deepseek_svc.get_default_url()


def test_deepseek_models_registry_contains_v4_models(deepseek_svc: DeepSeekService) -> None:
    registry = deepseek_svc.get_models_registry()
    assert "deepseek-v4-flash" in registry.models
    assert "deepseek-v4-pro" in registry.models


def test_deepseek_default_models_loaded(deepseek_svc: DeepSeekService) -> None:
    assert "default" in deepseek_svc.models
    assert len(deepseek_svc.models["default"]) > 0


def test_kimi_get_type(kimi_svc: KimiService) -> None:
    assert kimi_svc.get_type() == "kimi"


def test_kimi_is_cloud(kimi_svc: KimiService) -> None:
    assert kimi_svc.is_cloud is True
    assert kimi_svc.is_cloud_service() is True


def test_kimi_get_description(kimi_svc: KimiService) -> None:
    assert "Kimi" in kimi_svc.get_description()


def test_kimi_get_default_url(kimi_svc: KimiService) -> None:
    assert "moonshot.ai" in kimi_svc.get_default_url()


def test_kimi_models_registry_contains_k2_and_k3_models(kimi_svc: KimiService) -> None:
    registry = kimi_svc.get_models_registry()
    assert "kimi-k2.6" in registry.models
    assert "kimi-k3" in registry.models
    assert "kimi-k2.7-code" in registry.models
    assert "kimi-k2.7-code-highspeed" in registry.models


def test_kimi_default_models_loaded(kimi_svc: KimiService) -> None:
    assert "default" in kimi_svc.models
    assert len(kimi_svc.models["default"]) > 0


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


@pytest.mark.parametrize("value", [0, -100])
def test_get_model_props_drops_unusable_context_window(value: int) -> None:
    """An admin leaving the numeric field at 0 must not fail the install or publish a bogus ceiling."""
    model = RemoteModel(type="llm", context_length=value, max_context_length=value)

    props = get_model_props(model)

    assert props.context_window is None
    assert props.max_context_window is None


def test_get_model_props_keeps_declared_context_window() -> None:
    model = RemoteModel(type="llm", context_length=4096, max_context_length=8192)

    props = get_model_props(model)

    assert props.context_window == 4096
    assert props.max_context_window == 8192


def test_get_model_props_respects_disabled_endpoints() -> None:
    model = RemoteModel(type="llm", completions=False, legacy_completions=False, responses=False, messages=False)

    props = get_model_props(model)

    assert "/v1/chat/completions" not in props.endpoints
    assert "/v1/completions" not in props.endpoints


@pytest.mark.parametrize(
    ("model_type", "expected"),
    [
        ("embedding", ["/v1/embeddings"]),
        ("stt", ["/v1/audio/transcriptions"]),
        ("tts", ["/v1/audio/speech"]),
        ("txt2img", ["/v1/images/generations"]),
        ("decision", ["/v1/systemone"]),
    ],
)
def test_get_model_props_non_llm_advertises_only_its_own_endpoint(model_type: RemoteModelType, expected: list[str]) -> None:
    """LLM flags default to True, so a non-LLM model must not inherit the conversational endpoints from them."""
    model = RemoteModel(type=model_type)

    props = get_model_props(model)

    assert props.endpoints == expected


def test_get_model_props_custom_embedding_advertises_embeddings() -> None:
    model = RemoteCustomModel(id="my-embedder", type="embedding")

    props = get_model_props(model)

    assert props.endpoints == ["/v1/embeddings"]


def test_get_model_props_unresolved_type_advertises_no_endpoints() -> None:
    model = RemoteModel(type=None)

    props = get_model_props(model)

    assert props.endpoints == []


def test_base_service_options_headers_raises_not_implemented() -> None:
    opts = BaseServiceOptions(api_url="https://example.com")

    with pytest.raises(NotImplementedError):
        _ = opts.headers


def test_model_installed_info_get_info_returns_model_info() -> None:
    options = InstallModelIn(spec={"rename": "test"})
    info = _make_model_installed_info()
    info.options = options

    result = info.get_info()

    assert result.spec == options.spec
    assert result.registration_id == "reg-1"


def test_openai_get_spec_has_api_url_and_api_key_fields(openai_svc: OpenAIService) -> None:
    spec = openai_svc.get_spec()

    field_names = [f.name for f in spec.fields]
    assert "api_url" in field_names
    assert "api_key" in field_names


def test_openai_get_model_spec_has_rename_field(openai_svc: OpenAIService) -> None:
    spec = openai_svc.get_model_spec()

    field_names = [f.name for f in spec.fields]
    assert "rename" in field_names


def test_openai_get_custom_model_spec_not_none_and_has_id_field(openai_svc: OpenAIService) -> None:
    spec = openai_svc.get_custom_model_spec()

    assert spec is not None
    field_names = [f.name for f in spec.fields]
    assert "id" in field_names
    assert "type" in field_names


def test_get_custom_model_spec_offers_decision_type(openai_svc: OpenAIService) -> None:
    spec = openai_svc.get_custom_model_spec()

    assert spec is not None
    assert "decision" in next(f.values or [] for f in spec.fields if f.name == "type")


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
    config = openai_svc._generate_instance_config("default", None, None)  # pyright: ignore[reportPrivateUsage]

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
    config = openai_svc._generate_instance_config("default", info, None)  # pyright: ignore[reportPrivateUsage]

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
async def test_uninstall_instance_purge_with_other_instance_installed_does_not_clear_working_dir(
    openai_svc: OpenAIService,
) -> None:
    openai_svc.instances_info["default"].installed = _make_installed()
    openai_svc.instances_info["gpu-1"] = Instance(None, None, {}, InstanceConfig())
    openai_svc.instances_info["gpu-1"].installed = _make_installed()
    openai_svc._clear_working_dir = AsyncMock()  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage]

    await openai_svc._uninstall_instance("default", UninstallServiceIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert openai_svc._clear_working_dir.call_count == 0  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage]
    assert openai_svc.instances_info["default"].installed is None
    assert "gpu-1" in openai_svc.instances_info


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model_id", "model_type", "unregister_method"),
    [
        ("gpt-4o", "llm", "unregister_chat_completion"),
        ("tts-1", "tts", "unregister_audio_speech"),
        ("whisper-1", "stt", "unregister_audio_transcriptions"),
        ("dall-e-2", "txt2img", "unregister_image_generations"),
        ("text-embedding-ada-002", "embedding", "unregister_embeddings"),
        ("clef", "decision", "unregister_systemone"),
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
@pytest.mark.parametrize(
    "api_url",
    [
        "https://api.openai.com",
        "https://api.openai.com/",
        "https://api.openai.com/v1",
        "https://api.openai.com/v1/",
    ],
)
async def test_install_model_builds_correct_embeddings_url(
    openai_svc: OpenAIService,
    deps: dict[str, Any],
    api_url: str,
) -> None:
    """The proxy URL must always land on `/v1/embeddings`, whether or not the user's api_url already
    includes the `/v1` suffix or a trailing slash themselves.
    """
    openai_svc.instances_info["default"].installed = _make_installed(api_url=api_url)

    promise = await openai_svc._install_model("default", "text-embedding-ada-002", InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
    await promise.wait()

    options = deps["endpoint_registry"].register_embeddings_as_proxy.call_args.kwargs["options"]
    assert options.url == "https://api.openai.com/v1/embeddings"


@pytest.mark.asyncio
async def test_install_model_with_rename_uses_rename_as_registered_name(
    openai_svc: OpenAIService,
    deps: dict[str, Any],
) -> None:
    openai_svc.instances_info["default"].installed = _make_installed()

    promise = await openai_svc._install_model("default", "gpt-4o", InstallModelIn(spec={"rename": "my-alias"}))  # pyright: ignore[reportPrivateUsage]
    await promise.wait()

    installed_info = openai_svc.instances_info["default"].installed
    assert installed_info is not None
    assert installed_info.models["gpt-4o"].registered_name == "my-alias"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("spec", "expected_alias_of"),
    [
        ({"rename": "corp-gpt"}, {"corp-gpt": []}),
        ({"rename": "corp-gpt", "alias": "smart"}, {"corp-gpt": [], "smart": ["corp-gpt"]}),
    ],
)
async def test_install_model_with_rename_hides_original_id_from_models_list(
    deps: dict[str, Any],
    spec: dict[str, str],
    expected_alias_of: dict[str, list[str]],
) -> None:
    endpoint_registry = EndpointRegistry(
        config=MagicMock(), parent_infra=MagicMock(), model_tester=MagicMock(), metrics_registry=MagicMock()
    )
    svc = OpenAIService(**(deps | {"endpoint_registry": endpoint_registry}))
    svc.instances_info["default"].installed = _make_installed()

    promise = await svc._install_model("default", "gpt-4o", InstallModelIn(spec=spec))  # pyright: ignore[reportPrivateUsage]
    await promise.wait()

    models = endpoint_registry.get_models()
    assert {m.id: m.props.alias_of for m in models.data} == expected_alias_of
    assert "gpt-4o" not in models.model_dump_json()


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
        ("clef", "decision", "unregister_systemone"),
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


@pytest.mark.asyncio
async def test_install_model_registration_failure_rolls_back_model(openai_svc: OpenAIService, deps: dict[str, Any]) -> None:
    openai_svc.instances_info["default"].installed = _make_installed()
    deps["endpoint_registry"].register_chat_completion_as_proxy.side_effect = RuntimeError("registry down")

    promise = await openai_svc._install_model("default", "gpt-4o", InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(RuntimeError):
        await promise.wait()

    installed = openai_svc.instances_info["default"].installed
    assert installed is not None
    assert "gpt-4o" not in installed.models


@pytest.mark.asyncio
async def test_install_model_registration_failure_skips_rollback_when_already_removed(
    openai_svc: OpenAIService, deps: dict[str, Any]
) -> None:
    openai_svc.instances_info["default"].installed = _make_installed()

    def side_effect(*args: object, **kwargs: object) -> None:
        installed = openai_svc.instances_info["default"].installed
        assert installed is not None
        installed.models.pop("gpt-4o", None)
        raise RuntimeError("registry down")

    deps["endpoint_registry"].register_chat_completion_as_proxy.side_effect = side_effect

    promise = await openai_svc._install_model("default", "gpt-4o", InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(RuntimeError):
        await promise.wait()

    installed = openai_svc.instances_info["default"].installed
    assert installed is not None
    assert "gpt-4o" not in installed.models


# --- live model listing merge policy (cloud-model-autolist-policy) ---


def test_get_supported_model_types_derived_from_overlay(openai_svc: OpenAIService) -> None:
    types = openai_svc.get_supported_model_types()

    assert types is not None
    assert "llm" in types
    assert "embedding" in types


def test_merge_live_models_excludes_type_not_supported_by_instance(openai_svc: OpenAIService) -> None:
    openai_svc.get_supported_model_types = lambda: {"llm"}  # type: ignore[method-assign]

    merged = openai_svc._merge_live_models(  # pyright: ignore[reportPrivateUsage]
        "default", [LiveModelEntry(id="new-embedding-model", type="embedding")]
    )

    assert "new-embedding-model" not in merged


def test_merge_live_models_keeps_unresolved_type_entry(openai_svc: OpenAIService) -> None:
    merged = openai_svc._merge_live_models("default", [LiveModelEntry(id="totally-new-model")])  # pyright: ignore[reportPrivateUsage]

    assert "totally-new-model" in merged
    assert merged["totally-new-model"].capabilities_resolved is False
    assert merged["totally-new-model"].type is None


def test_merge_live_models_prefers_live_fields_over_overlay(openai_svc: OpenAIService) -> None:
    merged = openai_svc._merge_live_models(  # pyright: ignore[reportPrivateUsage]
        "default", [LiveModelEntry(id="gpt-4o", context_length=999)]
    )

    assert merged["gpt-4o"].context_length == 999
    assert merged["gpt-4o"].type == "llm"
    assert merged["gpt-4o"].capabilities_resolved is True


def test_merge_live_models_falls_back_to_overlay_when_live_omits_fields(openai_svc: OpenAIService) -> None:
    overlay_context_length = openai_svc.get_models_registry().models["gpt-4o"].context_length

    merged = openai_svc._merge_live_models("default", [LiveModelEntry(id="gpt-4o")])  # pyright: ignore[reportPrivateUsage]

    assert merged["gpt-4o"].type == "llm"
    assert merged["gpt-4o"].context_length == overlay_context_length


def test_merge_live_models_unresolved_when_no_overlay_match(openai_svc: OpenAIService) -> None:
    merged = openai_svc._merge_live_models("default", [LiveModelEntry(id="brand-new-model")])  # pyright: ignore[reportPrivateUsage]

    assert merged["brand-new-model"].capabilities_resolved is False


def test_merge_live_models_marks_installed_missing_model_as_stale(openai_svc: OpenAIService) -> None:
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

    merged = openai_svc._merge_live_models("default", [])  # pyright: ignore[reportPrivateUsage]

    assert merged["gpt-4o"].stale is True


def test_merge_live_models_drops_uninstalled_missing_model_silently(openai_svc: OpenAIService) -> None:
    openai_svc.instances_info["default"].installed = _make_installed()

    merged = openai_svc._merge_live_models("default", [])  # pyright: ignore[reportPrivateUsage]

    assert "gpt-4o" not in merged


def test_merge_live_models_skips_installed_model_absent_from_existing_catalog(openai_svc: OpenAIService) -> None:
    installed = _make_installed()
    installed.models["ghost-model"] = ModelInstalledInfo(
        id="ghost-model",
        registered_name="ghost-model",
        type="llm",
        options=InstallModelIn(spec={}),
        completions=True,
        legacy_completions=True,
        registration_id="reg-1",
    )
    openai_svc.instances_info["default"].installed = installed

    merged = openai_svc._merge_live_models("default", [])  # pyright: ignore[reportPrivateUsage]

    assert "ghost-model" not in merged


def test_merge_live_models_reappearing_clears_stale(openai_svc: OpenAIService) -> None:
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
    openai_svc.models["default"] = openai_svc._merge_live_models("default", [])  # pyright: ignore[reportPrivateUsage]
    assert openai_svc.models["default"]["gpt-4o"].stale is True

    merged_again = openai_svc._merge_live_models("default", [LiveModelEntry(id="gpt-4o")])  # pyright: ignore[reportPrivateUsage]

    assert merged_again["gpt-4o"].stale is False


def test_merge_live_models_preserves_custom_model_not_in_live_listing(openai_svc: OpenAIService) -> None:
    openai_svc.models["default"]["my-custom-model"] = RemoteModel(type="llm", custom="custom-id")

    merged = openai_svc._merge_live_models("default", [])  # pyright: ignore[reportPrivateUsage]

    assert "my-custom-model" in merged
    assert merged["my-custom-model"].custom == "custom-id"


def test_merge_live_models_custom_model_wins_over_colliding_live_entry(openai_svc: OpenAIService) -> None:
    openai_svc.models["default"]["gpt-4o"] = RemoteModel(type="embedding", custom="custom-id")

    merged = openai_svc._merge_live_models("default", [LiveModelEntry(id="gpt-4o")])  # pyright: ignore[reportPrivateUsage]

    assert merged["gpt-4o"].custom == "custom-id"
    assert merged["gpt-4o"].type == "embedding"


@pytest.mark.asyncio
async def test_install_model_refuses_when_capabilities_unresolved(openai_svc: OpenAIService) -> None:
    openai_svc.instances_info["default"].installed = _make_installed()
    openai_svc.models["default"]["mystery-model"] = RemoteModel(type=None)

    with pytest.raises(HTTPException) as exc:
        await openai_svc._install_model("default", "mystery-model", InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]

    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_refresh_catalog_raises_when_service_has_no_live_listing(deepseek_svc: DeepSeekService) -> None:
    with pytest.raises(HTTPException) as exc:
        await deepseek_svc.refresh_catalog()

    assert exc.value.status_code == 405


@pytest.mark.asyncio
async def test_refresh_catalog_merges_live_models_when_supported(openai_svc: OpenAIService) -> None:
    async def fake_fetch(instance: str) -> list[LiveModelEntry] | None:
        assert instance == "default"
        return [LiveModelEntry(id="gpt-4o-freshly-listed", type="llm")]

    openai_svc._fetch_live_models = fake_fetch  # type: ignore[method-assign] # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]

    result = await (await openai_svc.refresh_catalog()).wait()

    assert result.added == 1
    assert result.total == 1
    assert "gpt-4o-freshly-listed" in openai_svc.models["default"]


@pytest.mark.asyncio
async def test_refresh_catalog_raises_502_when_live_fetch_fails_for_supported_provider(openai_svc: OpenAIService) -> None:
    async def failing_fetch(instance: str) -> list[LiveModelEntry] | None:
        assert instance == "default"
        return None

    openai_svc._fetch_live_models = failing_fetch  # type: ignore[method-assign] # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]

    with pytest.raises(HTTPException) as exc:
        await openai_svc.refresh_catalog()

    assert exc.value.status_code == 502


@pytest.mark.asyncio
async def test_refresh_catalog_logs_partial_failure_without_raising(openai_svc: OpenAIService, caplog: pytest.LogCaptureFixture) -> None:
    """A refresh spanning several instances must not hide one instance's live-fetch failure behind an
    overall success just because another instance's fetch succeeded."""
    openai_svc.models["instance2"] = dict(openai_svc.models["default"])

    async def partial_fetch(instance: str) -> list[LiveModelEntry] | None:
        return [LiveModelEntry(id="gpt-4o-freshly-listed", type="llm")] if instance == "default" else None

    openai_svc._fetch_live_models = partial_fetch  # type: ignore[method-assign] # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]

    with caplog.at_level("WARNING"):
        result = await (await openai_svc.refresh_catalog()).wait()

    assert result.added == 1
    assert result.total == len(openai_svc.models["default"])
    assert any("instance2" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_after_install_swallows_merge_error_instead_of_raising(openai_svc: OpenAIService) -> None:
    """A bug in `_merge_live_models` must not surface as an install/update/startup failure -- by the
    time `_after_install` runs, the caller has already committed `installed`."""

    async def fake_fetch(instance: str) -> list[LiveModelEntry] | None:
        assert instance == "default"
        return [LiveModelEntry(id="freshly-installed-model")]

    def broken_merge(instance: str, live_entries: Any) -> dict[str, Any]:
        raise RuntimeError("boom")

    openai_svc._fetch_live_models = fake_fetch  # type: ignore[method-assign] # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
    openai_svc._merge_live_models = broken_merge  # type: ignore[method-assign] # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
    before = dict(openai_svc.models["default"])

    await openai_svc._after_install("default")  # pyright: ignore[reportPrivateUsage] # must not raise

    assert openai_svc.models["default"] == before


@pytest.mark.asyncio
async def test_after_install_merges_live_models_when_supported(openai_svc: OpenAIService) -> None:
    async def fake_fetch(instance: str) -> list[LiveModelEntry] | None:
        assert instance == "default"
        return [LiveModelEntry(id="freshly-installed-model")]

    openai_svc._fetch_live_models = fake_fetch  # type: ignore[method-assign] # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]

    await openai_svc._after_install("default")  # pyright: ignore[reportPrivateUsage]

    assert "freshly-installed-model" in openai_svc.models["default"]


@pytest.mark.asyncio
async def test_after_install_is_noop_when_live_listing_unsupported(deepseek_svc: DeepSeekService) -> None:
    before = dict(deepseek_svc.models["default"])

    await deepseek_svc._after_install("default")  # pyright: ignore[reportPrivateUsage]

    assert deepseek_svc.models["default"] == before
