# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from server.models.models import InstallModelIn, ListModelsFilters, UninstallModelIn
from server.models.services import InstallServiceIn, UninstallServiceIn
from server.services.base2_service import Instance, InstanceConfig
from server.services.sindri_service import (
    DownloadedInfo,
    InstalledInfo,
    ModelInstalledInfo,
    SindriAiModel,
    SindriModelOptions,
    SindriOptions,
    SindriService,
    _const,  # pyright: ignore[reportPrivateUsage]
)


@pytest.fixture
def deps() -> dict[str, Any]:
    return {
        "config": MagicMock(),
        "endpoint_registry": MagicMock(),
        "service_provider": MagicMock(),
        "model_downloader": MagicMock(),
        "docker_service": MagicMock(),
        "hardware": MagicMock(gpus=[], nvidia_gpus=[]),
    }


@pytest.fixture
def svc(deps: dict[str, Any]) -> SindriService:
    return SindriService(**deps)


def test_get_type(svc: SindriService) -> None:
    assert svc.get_type() == "sindri"


def test_get_description_not_empty(svc: SindriService) -> None:
    assert svc.get_description()


def test_service_has_docker(svc: SindriService) -> None:
    assert svc.service_has_docker() is True


def test_is_cloud_service(svc: SindriService) -> None:
    assert svc.is_cloud_service() is True


def test_const_image_has_name() -> None:
    assert _const.image.name


def test_const_models_not_empty() -> None:
    assert len(_const.models) > 0


def test_const_models_contain_gemma() -> None:
    assert any("gemma" in model_id.lower() for model_id in _const.models)


def test_const_models_have_context_length() -> None:
    for model in _const.models.values():
        assert model.context_length > 0
        assert model.max_context_length > 0


def test_sindri_options_api_key_required() -> None:
    with pytest.raises(ValidationError):
        SindriOptions()  # type: ignore[call-arg]


def test_sindri_options_api_key_stored() -> None:
    opts = SindriOptions(api_key="sk-abc123")

    assert opts.api_key == "sk-abc123"


def test_sindri_options_default_api_url() -> None:
    opts = SindriOptions(api_key="sk-abc")

    assert "sindri.app" in opts.api_url


def test_sindri_options_custom_api_url() -> None:
    opts = SindriOptions(api_key="sk-abc", api_url="https://custom.api/v1")

    assert opts.api_url == "https://custom.api/v1"


def test_sindri_model_options_alias_defaults_none() -> None:
    opts = SindriModelOptions()

    assert opts.alias is None


def test_get_spec_has_api_key_field(svc: SindriService) -> None:
    spec = svc.get_spec()

    field_names = {f.name for f in spec.fields}
    assert "api_key" in field_names


def test_get_spec_api_key_is_required(svc: SindriService) -> None:
    spec = svc.get_spec()

    api_key_field = next(f for f in spec.fields if f.name == "api_key")
    assert api_key_field.required is True


def test_get_spec_api_key_is_password_type(svc: SindriService) -> None:
    spec = svc.get_spec()

    api_key_field = next(f for f in spec.fields if f.name == "api_key")
    assert api_key_field.type == "password"


def test_get_spec_has_api_url_field(svc: SindriService) -> None:
    spec = svc.get_spec()

    field_names = {f.name for f in spec.fields}
    assert "api_url" in field_names


def test_get_spec_api_url_is_optional(svc: SindriService) -> None:
    spec = svc.get_spec()

    api_url_field = next(f for f in spec.fields if f.name == "api_url")
    assert api_url_field.required is False


def test_get_model_spec_has_alias(svc: SindriService) -> None:
    spec = svc.get_model_spec()

    field_names = {f.name for f in spec.fields}
    assert "alias" in field_names


def test_get_custom_model_spec_is_none(svc: SindriService) -> None:
    assert svc.get_custom_model_spec() is None


def test_get_image_returns_const_image(svc: SindriService) -> None:
    image = svc._get_image()  # pyright: ignore[reportPrivateUsage]

    assert image == _const.image


def test_default_models_loaded_on_init(svc: SindriService) -> None:
    assert "default" in svc.models
    assert len(svc.models["default"]) > 0


def test_default_models_match_const(svc: SindriService) -> None:
    for model_id in _const.models:
        assert model_id in svc.models["default"]


def _make_installed_info() -> InstalledInfo:
    docker = MagicMock()
    docker.name = "df-sindri-default"
    return InstalledInfo(
        docker=docker,
        models={},
        options=InstallServiceIn(spec={}),
        parsed_options=SindriOptions(api_key="sk-test"),
        container_host="localhost",
        container_port=8080,
        docker_exposed_port=8080,
        base_url="http://localhost:8080",
    )


def _make_model_installed_info(model_id: str = "gemma3:27b", registration_id: str = "reg-1") -> ModelInstalledInfo:
    return ModelInstalledInfo(
        id=model_id,
        type="llm",
        registered_name=model_id,
        options=InstallModelIn(spec={}),
        registration_id=registration_id,
    )


def test_get_size_returns_image_size(svc: SindriService) -> None:
    assert svc.get_size() == _const.image.size


def test_model_installed_info_get_info() -> None:
    info = _make_model_installed_info(registration_id="reg-99")

    result = info.get_info()

    assert result.registration_id == "reg-99"


def test_get_installed_info_returns_spec_when_installed(svc: SindriService) -> None:
    installed = _make_installed_info()
    installed.options.spec["api_key"] = "sk-x"
    svc.instances_info["default"].installed = installed

    result = svc.get_installed_info("default")

    assert result == installed.options.spec


def test_get_installed_info_calls_helper_when_not_installed(svc: SindriService) -> None:
    svc.instances_info["default"].installed = None

    with patch.object(svc, "_get_service_installed_info", return_value=False) as mock:  # pyright: ignore[reportPrivateUsage]
        result = svc.get_installed_info("default")

    assert mock.call_count == 1
    assert mock.call_args == call("default")
    assert result is False


def test_generate_instance_config_none_info(svc: SindriService) -> None:
    config = svc._generate_instance_config(None, None)  # pyright: ignore[reportPrivateUsage]

    assert config.options is None
    assert config.models == []


def test_generate_instance_config_with_info(svc: SindriService) -> None:
    installed = _make_installed_info()
    installed.models["gemma3:27b"] = _make_model_installed_info("gemma3:27b")

    config = svc._generate_instance_config(installed, None)  # pyright: ignore[reportPrivateUsage]

    assert config.options == installed.options
    assert len(config.models or []) == 1


def test_load_download_info_returns_dataclass(svc: SindriService) -> None:
    result = svc._load_download_info({})  # pyright: ignore[reportPrivateUsage]

    assert isinstance(result, DownloadedInfo)


def test_get_docker_compose_file_path_raises_400_when_model_id_given(svc: SindriService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    with pytest.raises(HTTPException) as exc_info:
        svc.get_docker_compose_file_path("default", "some-model")

    assert exc_info.value.status_code == 400


def test_get_docker_compose_file_path_delegates_to_docker_service(svc: SindriService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    deps["docker_service"].get_docker_compose_file_path.return_value = "/path/docker-compose.yml"

    result = svc.get_docker_compose_file_path("default", None)

    assert result == "/path/docker-compose.yml"
    assert deps["docker_service"].get_docker_compose_file_path.call_count == 1
    assert deps["docker_service"].get_docker_compose_file_path.call_args == call(installed.docker.name)


@pytest.mark.asyncio
async def test_stop_instance_does_nothing_when_not_installed(svc: SindriService) -> None:
    svc.instances_info["default"].installed = None

    with patch.object(svc, "_stop_docker", new_callable=AsyncMock) as mock:  # pyright: ignore[reportPrivateUsage]
        await svc.stop_instance("default")

    assert mock.call_count == 0


@pytest.mark.asyncio
async def test_stop_instance_stops_docker_when_installed(svc: SindriService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    with patch.object(svc, "_stop_docker", new_callable=AsyncMock) as mock:  # pyright: ignore[reportPrivateUsage]
        await svc.stop_instance("default")

    assert mock.call_count == 1
    assert mock.call_args == call(installed.docker)


@pytest.mark.asyncio
async def test_list_models_raises_404_for_unknown_instance(svc: SindriService) -> None:
    with pytest.raises(HTTPException) as exc_info:
        await svc.list_models("nonexistent", ListModelsFilters())

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_list_models_returns_all_models_no_filter(svc: SindriService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    result = await svc.list_models("default", ListModelsFilters())

    assert len(result.list) == len(svc.models["default"])


@pytest.mark.asyncio
async def test_list_models_filters_installed_only(svc: SindriService) -> None:
    installed = _make_installed_info()
    model_id = next(iter(svc.models["default"]))
    installed.models[model_id] = _make_model_installed_info(model_id)
    svc.instances_info["default"].installed = installed

    result = await svc.list_models("default", ListModelsFilters(installed=True))

    assert len(result.list) >= 1
    assert all(bool(m.installed) for m in result.list)


@pytest.mark.asyncio
async def test_list_models_filters_not_installed(svc: SindriService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    result = await svc.list_models("default", ListModelsFilters(installed=False))

    assert all(not m.installed for m in result.list)


@pytest.mark.asyncio
async def test_list_models_multiple_instances(svc: SindriService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    svc.instances_info["extra"] = Instance(None, None, {}, InstanceConfig())
    svc.load_default_models("extra")
    svc.instances_info["extra"].installed = _make_installed_info()

    result = await svc.list_models(None, ListModelsFilters())

    assert len(result.list) >= len(_const.models)


@pytest.mark.asyncio
async def test_get_model_raises_400_for_unknown_model(svc: SindriService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    with pytest.raises(HTTPException) as exc_info:
        await svc.get_model("default", "nonexistent-model")

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_get_model_returns_correct_model(svc: SindriService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    model_id = next(iter(_const.models))

    result = await svc.get_model("default", model_id)

    assert result.id == model_id


@pytest.mark.asyncio
async def test_get_model_shows_installed_when_in_models(svc: SindriService) -> None:
    installed = _make_installed_info()
    model_id = next(iter(_const.models))
    installed.models[model_id] = _make_model_installed_info(model_id, "reg-1")
    svc.instances_info["default"].installed = installed

    result = await svc.get_model("default", model_id)

    assert result.installed is not None


@pytest.mark.asyncio
async def test_install_model_returns_already_installed(svc: SindriService) -> None:
    installed = _make_installed_info()
    model_id = next(iter(_const.models))
    installed.models[model_id] = _make_model_installed_info(model_id)
    svc.instances_info["default"].installed = installed

    promise = await svc._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
    result = await promise.wait()

    assert result.status == "OK"
    assert "Already installed" in result.details


@pytest.mark.asyncio
async def test_install_model_raises_400_for_unknown_model(svc: SindriService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    with pytest.raises(HTTPException) as exc_info:
        await svc._install_model("default", "no-such-model", InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_install_model_registers_endpoint(svc: SindriService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    model_id = next(iter(_const.models))

    promise = await svc._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
    await promise.wait()

    assert deps["endpoint_registry"].register_chat_completion_as_proxy.call_count == 1


@pytest.mark.asyncio
async def test_install_model_uses_alias_as_registered_name(svc: SindriService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    model_id = next(iter(_const.models))

    promise = await svc._install_model("default", model_id, InstallModelIn(spec={"alias": "my-alias"}))  # pyright: ignore[reportPrivateUsage]
    await promise.wait()

    assert installed.models[model_id].registered_name == "my-alias"


@pytest.mark.asyncio
async def test_install_model_uses_model_id_when_no_alias(svc: SindriService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    model_id = next(iter(_const.models))

    promise = await svc._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
    await promise.wait()

    assert installed.models[model_id].registered_name == model_id


@pytest.mark.asyncio
async def test_install_model_marks_as_downloaded(svc: SindriService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    model_id = next(iter(_const.models))

    promise = await svc._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
    await promise.wait()

    assert model_id in svc.models_downloaded


@pytest.mark.asyncio
async def test_uninstall_model_removes_from_info_and_unregisters(svc: SindriService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    model_id = next(iter(_const.models))
    installed.models[model_id] = _make_model_installed_info(model_id, "reg-1")
    svc.instances_info["default"].installed = installed

    await svc._uninstall_model("default", model_id, UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert model_id not in installed.models
    assert deps["endpoint_registry"].unregister_chat_completion.call_count == 1
    assert deps["endpoint_registry"].unregister_chat_completion.call_args == call(model_id, "reg-1")


@pytest.mark.asyncio
async def test_uninstall_model_purge_removes_from_downloaded(svc: SindriService) -> None:
    installed = _make_installed_info()
    model_id = next(iter(_const.models))
    installed.models[model_id] = _make_model_installed_info(model_id)
    svc.instances_info["default"].installed = installed
    svc.models_downloaded[model_id] = DownloadedInfo()

    await svc._uninstall_model("default", model_id, UninstallModelIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert model_id not in svc.models_downloaded


@pytest.mark.asyncio
async def test_uninstall_model_no_purge_keeps_downloaded(svc: SindriService) -> None:
    installed = _make_installed_info()
    model_id = next(iter(_const.models))
    installed.models[model_id] = _make_model_installed_info(model_id)
    svc.instances_info["default"].installed = installed
    svc.models_downloaded[model_id] = DownloadedInfo()

    await svc._uninstall_model("default", model_id, UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert model_id in svc.models_downloaded


@pytest.mark.asyncio
async def test_uninstall_model_ignores_unknown_model(svc: SindriService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    await svc._uninstall_model("default", "no-such-model", UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert deps["endpoint_registry"].unregister_chat_completion.call_count == 0


@pytest.mark.asyncio
async def test_install_instance_returns_installed_info(svc: SindriService, deps: dict[str, Any], tmp_path: Any) -> None:
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=8080)
    deps["docker_service"].get_docker_subnet.return_value = None
    deps["docker_service"].get_docker_container_name.return_value = "container"
    deps["docker_service"].get_container_host.return_value = "localhost"
    deps["docker_service"].get_container_port.return_value = 8080
    options = InstallServiceIn(spec={"api_key": "sk-test"})

    with (
        patch.object(svc, "_download_image_or_set_progress", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_verify_docker_image", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_get_working_dir", return_value=tmp_path),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.sindri_service.get_base_url", return_value="http://localhost:8080"),
    ):
        promise = await svc._install_instance("default", options)  # pyright: ignore[reportPrivateUsage]
        result = await promise.wait()

    assert isinstance(result, InstalledInfo)
    assert result.models == {}
    assert deps["docker_service"].install_and_run_docker.call_count == 1


@pytest.mark.asyncio
async def test_install_instance_loads_default_models_when_missing(svc: SindriService, deps: dict[str, Any], tmp_path: Any) -> None:
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=8080)
    deps["docker_service"].get_docker_subnet.return_value = None
    deps["docker_service"].get_docker_container_name.return_value = "container"
    deps["docker_service"].get_container_host.return_value = "localhost"
    deps["docker_service"].get_container_port.return_value = 8080
    del svc.models["default"]
    options = InstallServiceIn(spec={"api_key": "sk-test"})

    with (
        patch.object(svc, "_download_image_or_set_progress", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_verify_docker_image", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_get_working_dir", return_value=tmp_path),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.sindri_service.get_base_url", return_value="http://localhost:8080"),
    ):
        promise = await svc._install_instance("default", options)  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert "default" in svc.models


@pytest.mark.asyncio
async def test_uninstall_instance_unregisters_all_models(svc: SindriService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    model_id = next(iter(_const.models))
    installed.models[model_id] = _make_model_installed_info(model_id, "reg-1")
    svc.instances_info["default"].installed = installed
    deps["docker_service"].uninstall_docker = AsyncMock()

    with patch.object(svc, "_uninstall_model", new_callable=AsyncMock) as mock_uninstall:  # pyright: ignore[reportPrivateUsage]
        await svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert mock_uninstall.call_count == 1
    assert deps["docker_service"].uninstall_docker.call_count == 1
    assert deps["docker_service"].uninstall_docker.call_args == call(installed.docker)
    assert svc.instances_info["default"].installed is None


@pytest.mark.asyncio
async def test_uninstall_instance_no_op_when_not_installed(svc: SindriService, deps: dict[str, Any]) -> None:
    svc.instances_info["default"].installed = None
    deps["docker_service"].uninstall_docker = AsyncMock()

    await svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert deps["docker_service"].uninstall_docker.call_count == 0


@pytest.mark.asyncio
async def test_uninstall_instance_purge_clears_service_state(svc: SindriService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
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
async def test_uninstall_instance_purge_non_default_deletes_instance(svc: SindriService, deps: dict[str, Any]) -> None:
    svc.instances_info["extra"] = Instance(None, None, {}, InstanceConfig())
    svc.instances_info["extra"].installed = _make_installed_info()
    deps["docker_service"].uninstall_docker = AsyncMock()
    deps["docker_service"].remove_image = AsyncMock()

    with (
        patch.object(svc, "_uninstall_model", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_clear_working_dir", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
    ):
        await svc._uninstall_instance("extra", UninstallServiceIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert "extra" not in svc.instances_info


@pytest.mark.asyncio
async def test_list_models_skips_instances_not_in_filter(svc: SindriService) -> None:
    svc.instances_info["extra"] = Instance(None, None, {}, InstanceConfig())
    svc.instances_info["extra"].installed = _make_installed_info()
    svc.load_default_models("extra")
    svc.instances_info["default"].installed = _make_installed_info()

    result = await svc.list_models("default", ListModelsFilters())

    service_ids = {m.service for m in result.list}
    assert all("extra" not in s for s in service_ids)


@pytest.mark.asyncio
async def test_get_model_initialises_models_dict_when_missing(svc: SindriService) -> None:
    svc.instances_info["default"].installed = _make_installed_info()
    svc.models.pop("default", None)

    with pytest.raises(HTTPException) as exc_info:
        await svc.get_model("default", "nonexistent-model")

    assert exc_info.value.status_code == 400
    assert "default" in svc.models


_NON_LLM_ID = "embed-model"
_NON_LLM_MODEL = SindriAiModel(type="embed", context_length=0, max_context_length=0, real_model_name="embed-model")


def _make_non_llm_model_installed_info(model_id: str = _NON_LLM_ID) -> ModelInstalledInfo:
    return ModelInstalledInfo(
        id=model_id,
        type="embed",
        registered_name=model_id,
        options=InstallModelIn(spec={}),
        registration_id="reg-embed",
    )


@pytest.mark.asyncio
async def test_uninstall_instance_non_llm_model_skips_endpoint_unregister(svc: SindriService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    installed.models[_NON_LLM_ID] = _make_non_llm_model_installed_info()
    svc.instances_info["default"].installed = installed
    deps["docker_service"].uninstall_docker = AsyncMock()

    with patch.object(svc, "_uninstall_model", new_callable=AsyncMock):  # pyright: ignore[reportPrivateUsage]
        await svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert deps["endpoint_registry"].unregister_chat_completion.call_count == 0


@pytest.mark.asyncio
async def test_uninstall_instance_skips_uninstall_when_model_in_other_instance(svc: SindriService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    model_id = next(iter(_const.models))
    installed.models[model_id] = _make_model_installed_info(model_id, "reg-1")
    svc.instances_info["default"].installed = installed
    deps["docker_service"].uninstall_docker = AsyncMock()

    with (
        patch.object(svc, "is_model_installed_in_other_instance", return_value=True),
        patch.object(svc, "_uninstall_model", new_callable=AsyncMock) as mock_uninstall,  # pyright: ignore[reportPrivateUsage]
    ):
        await svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert mock_uninstall.call_count == 0


@pytest.mark.asyncio
async def test_list_models_filter_false_excludes_installed_model(svc: SindriService) -> None:
    installed = _make_installed_info()
    model_id = next(iter(_const.models))
    installed.models[model_id] = _make_model_installed_info(model_id)
    svc.instances_info["default"].installed = installed

    result = await svc.list_models("default", ListModelsFilters(installed=False))

    assert not any(m.id == model_id for m in result.list)


@pytest.mark.asyncio
async def test_install_model_non_llm_skips_endpoint_registration(svc: SindriService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    svc.models["default"][_NON_LLM_ID] = _NON_LLM_MODEL

    with patch.dict(_const.models, {_NON_LLM_ID: _NON_LLM_MODEL}):
        promise = await svc._install_model("default", _NON_LLM_ID, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert deps["endpoint_registry"].register_chat_completion_as_proxy.call_count == 0
    assert _NON_LLM_ID in svc.models_downloaded


@pytest.mark.asyncio
async def test_uninstall_model_non_llm_skips_endpoint_unregister(svc: SindriService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    installed.models[_NON_LLM_ID] = _make_non_llm_model_installed_info()
    svc.instances_info["default"].installed = installed

    await svc._uninstall_model("default", _NON_LLM_ID, UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert _NON_LLM_ID not in installed.models
    assert deps["endpoint_registry"].unregister_chat_completion.call_count == 0


@pytest.mark.asyncio
async def test_install_model_initialises_models_dict_for_new_instance_already_installed(svc: SindriService) -> None:
    installed = _make_installed_info()
    svc.instances_info["new-inst"] = Instance(None, None, {}, InstanceConfig())
    svc.instances_info["new-inst"].installed = installed
    model_id = next(iter(_const.models))
    installed.models[model_id] = _make_model_installed_info(model_id)

    # new-inst has no models dict so line 356 triggers, then returns early (already installed)
    promise = await svc._install_model("new-inst", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]

    assert "new-inst" in svc.models
    result = await promise.wait()
    assert result.status == "OK"
