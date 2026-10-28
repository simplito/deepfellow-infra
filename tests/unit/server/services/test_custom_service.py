# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

import asyncio
import logging
from collections.abc import Callable
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from fastapi import HTTPException
from pydantic import SecretStr

from server.docker import DockerOptions
from server.models.api import ModelProps
from server.models.models import AddCustomModelIn, InstallModelIn, ListModelsFilters, ModelInfo, ModelSpecification, UninstallModelIn
from server.models.services import InstallServiceIn, UninstallServiceIn
from server.services.base2_service import CustomModel, Instance, InstanceConfig
from server.services.custom_service import (
    DOC_CHUNKER_DEFAULT_GATEWAY_URL,
    DOC_CHUNKER_PROXY_TIMEOUT_SECONDS,
    CustomService,
    DownloadedInfo,
    InstalledInfo,
    ModelInstalledInfo,
    SrvCustomModel,
    _const,  # pyright: ignore[reportPrivateUsage]
    _is_own_gateway_url,  # pyright: ignore[reportPrivateUsage]
    create_bge_m3_model,
    create_doc_chunker_model,
    create_gliner_model,
    create_lemmatizer_model,
    validate_doc_chunker_workers,
)
from server.utils.hardware import NvidiaGpuInfo
from server.utils.registry_client import RegistryUnavailableError

_CUSTOM_MODEL_DATA: dict[str, Any] = {
    "id": "my-custom",
    "default_prefix": "my-custom",
    "size": "1GB",
    "image": "test/image:latest",
    "image_port": 8000,
}


@pytest.fixture
def deps() -> dict[str, Any]:
    docker_svc = MagicMock()
    docker_svc.get_docker_subnet.return_value = "172.20.0.0/16"
    docker_svc.get_docker_container_name.side_effect = lambda name, **_: f"df-{name}"  # pyright: ignore[reportUnknownLambdaType]
    config = MagicMock()
    config.standard_proxy_timeout_seconds = 300
    config.infra_api_key = SecretStr("infra-key")
    config.infra_url = ""
    return {
        "config": config,
        "endpoint_registry": MagicMock(),
        "service_provider": MagicMock(),
        "model_downloader": MagicMock(),
        "docker_service": docker_svc,
        "hardware": MagicMock(gpus=[]),
        "services_manager": MagicMock(),
    }


@pytest.fixture
def svc(deps: dict[str, Any]) -> CustomService:
    return CustomService(**deps)


def test_get_type(svc: CustomService) -> None:
    assert svc.get_type() == "custom"


def test_get_description_not_empty(svc: CustomService) -> None:
    assert svc.get_description()


def test_get_size_empty(svc: CustomService) -> None:
    assert svc.get_size() == ""


def test_get_spec_returns_empty_fields(svc: CustomService) -> None:
    spec = svc.get_spec()
    assert spec.fields == []


def test_default_models_loaded_on_init(svc: CustomService) -> None:
    assert "default" in svc.models
    assert len(svc.models["default"]) > 0


def test_const_models_not_empty() -> None:
    assert len(_const.models) > 0


@pytest.mark.asyncio
async def test_load_service_strips_retired_finetune_model_entry(svc: CustomService) -> None:
    """A `services.json` written before deepfellow-finetune moved to its own service must not
    resurface it as a "failed to load" warning on every future startup."""
    raw_config = {
        "instances": {
            "default": {
                "options": {"spec": {}},
                "models": [
                    {"model_id": "deepfellow-finetune", "options": {"spec": {}}},
                    {"model_id": "gliner", "options": {"spec": {}}},
                ],
            }
        }
    }

    with (
        patch("server.services.base2_service.Base2Service.load_service", new=AsyncMock()) as mock_super_load,
        patch.object(svc, "_remove_retired_model_container", new=AsyncMock()),
        patch.object(svc, "_save", new=AsyncMock()),
    ):
        await svc.load_service(raw_config)

    loaded_config = mock_super_load.call_args.args[0]
    model_ids = [m["model_id"] for m in loaded_config["instances"]["default"]["models"]]
    assert model_ids == ["gliner"]


@pytest.mark.asyncio
async def test_load_service_removes_retired_model_container_and_saves_pruned_config(svc: CustomService) -> None:
    """The dropped entry's container is still running and holding its GPU/port; the pruned config is
    saved right away so the teardown happens only once instead of on every startup."""
    raw_config = {
        "instances": {
            "default": {"options": {"spec": {}}, "models": [{"model_id": "deepfellow-finetune", "options": {"spec": {}}}]},
            "other": {"options": {"spec": {}}, "models": [{"model_id": "deepfellow-finetune", "options": {"spec": {}}}]},
        }
    }

    with (
        patch("server.services.base2_service.Base2Service.load_service", new=AsyncMock()),
        patch.object(svc, "_remove_retired_model_container", new=AsyncMock()) as mock_remove,
        patch.object(svc, "_save", new=AsyncMock()) as mock_save,
    ):
        await svc.load_service(raw_config)

    assert mock_remove.call_args_list == [call("deepfellow-finetune")]
    mock_save.assert_awaited_once()


@pytest.mark.asyncio
async def test_load_service_without_retired_entries_neither_removes_nor_saves(svc: CustomService) -> None:
    raw_config = {"instances": {"default": {"options": {"spec": {}}, "models": [{"model_id": "gliner", "options": {"spec": {}}}]}}}

    with (
        patch("server.services.base2_service.Base2Service.load_service", new=AsyncMock()),
        patch.object(svc, "_remove_retired_model_container", new=AsyncMock()) as mock_remove,
        patch.object(svc, "_save", new=AsyncMock()) as mock_save,
    ):
        await svc.load_service(raw_config)

    mock_remove.assert_not_called()
    mock_save.assert_not_called()


@pytest.mark.asyncio
async def test_remove_retired_model_container_uninstalls_its_docker(svc: CustomService, deps: dict[str, Any]) -> None:
    deps["docker_service"].uninstall_docker = AsyncMock()

    await svc._remove_retired_model_container("deepfellow-finetune")  # pyright: ignore[reportPrivateUsage]

    docker_options = deps["docker_service"].uninstall_docker.call_args.args[0]
    assert docker_options.name == "deepfellow-finetune"
    assert docker_options.container_name == "df-deepfellow-finetune"


@pytest.mark.asyncio
async def test_remove_retired_model_container_logs_and_continues_on_failure(
    svc: CustomService, deps: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    deps["docker_service"].uninstall_docker = AsyncMock(side_effect=RuntimeError("docker down"))

    with caplog.at_level(logging.WARNING):
        await svc._remove_retired_model_container("deepfellow-finetune")  # pyright: ignore[reportPrivateUsage]

    assert "Failed to remove container 'df-deepfellow-finetune'" in caplog.text


@pytest.mark.asyncio
async def test_load_service_without_instances_key_still_calls_super(svc: CustomService) -> None:
    raw_config: dict[str, Any] = {}

    with patch("server.services.base2_service.Base2Service.load_service", new=AsyncMock()) as mock_super_load:
        await svc.load_service(raw_config)

    assert mock_super_load.call_count == 1
    assert mock_super_load.call_args == call(raw_config)


@pytest.mark.asyncio
async def test_load_service_skips_instance_entries_without_a_models_list(svc: CustomService) -> None:
    raw_config = {"instances": {"default": "not-a-dict", "other": {"options": {"spec": {}}}}}

    with patch("server.services.base2_service.Base2Service.load_service", new=AsyncMock()) as mock_super_load:
        await svc.load_service(raw_config)

    loaded_config = mock_super_load.call_args.args[0]
    assert loaded_config["instances"]["default"] == "not-a-dict"
    assert "models" not in loaded_config["instances"]["other"]


def test_service_has_docker(svc: CustomService) -> None:
    assert svc.service_has_docker() is False


def test_get_custom_model_spec_not_none(svc: CustomService) -> None:
    assert svc.get_custom_model_spec() is not None


def test_get_default_model_spec_contains_prefix_field(svc: CustomService) -> None:
    spec = svc.get_default_model_spec("my-prefix")
    field_names = [f.name for f in spec.fields]
    assert "prefix" in field_names


def test_get_default_model_spec_contains_proxy_timeout_field_exactly_once(svc: CustomService) -> None:
    """Covers bentoml, easyOCR and duplicated custom models, which all build their spec from this method."""
    spec = svc.get_default_model_spec("my-prefix")
    field_names = [f.name for f in spec.fields]
    assert field_names.count("proxy_timeout_seconds") == 1


def test_get_model_install_fields_returns_model_spec_fields(svc: CustomService) -> None:
    result = svc._get_model_install_fields("default", "doc_chunker")  # pyright: ignore[reportPrivateUsage]

    assert result == svc.models["default"]["doc_chunker"].model_spec.fields


def test_get_model_install_fields_returns_empty_list_for_unknown_model(svc: CustomService) -> None:
    result = svc._get_model_install_fields("default", "ghost-model")  # pyright: ignore[reportPrivateUsage]

    assert result == []


@pytest.mark.parametrize(
    "create_model",
    [create_bge_m3_model, create_lemmatizer_model, create_gliner_model, create_doc_chunker_model],
)
def test_create_model_spec_contains_proxy_timeout_field_exactly_once(
    svc: CustomService, create_model: Callable[[CustomService, str | None], SrvCustomModel]
) -> None:
    model = create_model(svc, "172.20.0.0/16")
    field_names = [f.name for f in model.model_spec.fields]
    assert field_names.count("proxy_timeout_seconds") == 1


@pytest.mark.parametrize(
    ("create_model", "name"),
    [
        (create_bge_m3_model, "deepfellow-bge-m3"),
        (create_lemmatizer_model, "df-lemmatizer"),
        (create_gliner_model, "df-gliner"),
        (create_doc_chunker_model, "doc_chunker"),
    ],
)
def test_create_model_derives_container_name_without_legacy_prefix(
    svc: CustomService, deps: dict[str, Any], create_model: Callable[[CustomService, str | None], SrvCustomModel], name: str
) -> None:
    """These always had a fixed name the legacy prefix never applied to, so it must stay unprefixed on upgrade."""
    model = create_model(svc, "172.20.0.0/16")

    docker_opts = model.options({"hardware": "CPU"})  # pyright: ignore[reportCallIssue]

    assert docker_opts.name == name
    assert docker_opts.container_name == f"df-{name}"
    deps["docker_service"].get_docker_container_name.assert_called_with(name, apply_legacy_prefix=False)


def test_model_installed_info_get_info() -> None:
    options = InstallModelIn(spec={"prefix": "test"})
    model_info = ModelInstalledInfo(
        id="test-model",
        options=options,
        docker_options=MagicMock(),
        container_host="127.0.0.1",
        container_port=8080,
        docker_exposed_port=8080,
        registration_id="reg-123",
        prefix="test",
        base_url="http://127.0.0.1:8080",
    )

    info = model_info.get_info()

    assert info.spec == options.spec
    assert info.registration_id == "reg-123"


@pytest.mark.asyncio
async def test_stop_instance_no_installed_is_noop(svc: CustomService, deps: dict[str, Any]) -> None:
    deps["docker_service"].stop_docker = AsyncMock()

    await svc.stop_instance("default")

    assert deps["docker_service"].stop_docker.call_count == 0


@pytest.mark.asyncio
async def test_stop_instance_with_installed_stops_containers(svc: CustomService, deps: dict[str, Any]) -> None:
    docker_opts = MagicMock()
    mock_model = MagicMock()
    mock_model.docker_options = docker_opts
    svc.instances_info["default"].installed = InstalledInfo(
        models={"lemmatizer": mock_model},
        options=InstallServiceIn(spec={}),
    )
    deps["docker_service"].stop_docker = AsyncMock()

    await svc.stop_instance("default")

    assert deps["docker_service"].stop_docker.call_count == 1
    assert deps["docker_service"].stop_docker.call_args == call(docker_opts)


def test_get_installed_info_when_not_installed_returns_false(svc: CustomService) -> None:
    result = svc.get_installed_info("default")
    assert result is False


def test_get_installed_info_when_installed_returns_spec(svc: CustomService) -> None:
    options = InstallServiceIn(spec={"key": "val"})
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=options)

    result = svc.get_installed_info("default")

    assert result == {"key": "val"}


def test_generate_instance_config_without_info(svc: CustomService) -> None:
    config = svc._generate_instance_config("default", None, None)  # pyright: ignore[reportPrivateUsage]

    assert config.options is None
    assert config.models == []


def test_generate_instance_config_with_info(svc: CustomService) -> None:
    mock_model_info = MagicMock()
    mock_model_info.id = "lemmatizer"
    mock_model_info.options = InstallModelIn(spec={})
    mock_model_info.docker_options.container_name = "df-lemmatizer"
    installed = InstalledInfo(models={"lemmatizer": mock_model_info}, options=InstallServiceIn(spec={}))

    config = svc._generate_instance_config("default", installed, None)  # pyright: ignore[reportPrivateUsage]

    assert config.options == installed.options
    assert config.models is not None
    assert len(config.models) == 1
    assert config.models[0].model_id == "lemmatizer"


def test_generate_instance_config_persists_resolved_container_name(svc: CustomService) -> None:
    mock_model_info = MagicMock()
    mock_model_info.id = "lemmatizer"
    mock_model_info.options = InstallModelIn(spec={})
    mock_model_info.docker_options.container_name = "df-lemmatizer_abc123"
    installed = InstalledInfo(models={"lemmatizer": mock_model_info}, options=InstallServiceIn(spec={}))

    config = svc._generate_instance_config("default", installed, None)  # pyright: ignore[reportPrivateUsage]

    assert config.models is not None
    assert config.models[0].container_name == "df-lemmatizer_abc123"


def test_load_download_info(svc: CustomService) -> None:
    info = svc._load_download_info({"image": "test/image:latest"})  # pyright: ignore[reportPrivateUsage]

    assert info.image == "test/image:latest"


@pytest.mark.asyncio
async def test_install_instance_returns_installed_info(svc: CustomService) -> None:
    options = InstallServiceIn(spec={})

    promise = await svc._install_instance("default", options)  # pyright: ignore[reportPrivateUsage]

    result = await promise.wait()
    assert isinstance(result, InstalledInfo)
    assert svc.service_downloaded is True


@pytest.mark.asyncio
async def test_uninstall_instance_without_purge_clears_installed(svc: CustomService) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))

    await svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert svc.instances_info["default"].installed is None


@pytest.mark.asyncio
async def test_uninstall_instance_with_purge_resets_service(svc: CustomService) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    svc._clear_working_dir = AsyncMock()  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage]

    await svc._uninstall_instance("default", UninstallServiceIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert svc.service_downloaded is False
    assert svc._clear_working_dir.call_count == 1  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage]


@pytest.mark.asyncio
async def test_uninstall_instance_with_models_uninstalls_each(svc: CustomService, deps: dict[str, Any]) -> None:
    mock_model = MagicMock()
    mock_model.id = "lemmatizer"
    mock_model.prefix = "lemmatizer"
    mock_model.registration_id = "reg-1"
    mock_model.docker_options = MagicMock()
    deps["docker_service"].uninstall_docker = AsyncMock()
    svc.instances_info["default"].installed = InstalledInfo(
        models={"lemmatizer": mock_model},
        options=InstallServiceIn(spec={}),
    )

    await svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert svc.instances_info["default"].installed is None
    assert deps["docker_service"].uninstall_docker.call_count == 1


def test_get_docker_compose_file_path_no_model_id_raises(svc: CustomService) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))

    with pytest.raises(HTTPException) as exc:
        svc.get_docker_compose_file_path("default", None)

    assert exc.value.status_code == 400


def test_get_docker_compose_file_path_model_not_found_raises(svc: CustomService) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))

    with pytest.raises(HTTPException) as exc:
        svc.get_docker_compose_file_path("default", "nonexistent")

    assert exc.value.status_code == 400


def test_get_docker_compose_file_path_success(svc: CustomService, deps: dict[str, Any]) -> None:
    expected = MagicMock()
    deps["docker_service"].get_docker_compose_file_path.return_value = expected
    svc.instances_info["default"].installed = InstalledInfo(
        models={"lemmatizer": MagicMock()},
        options=InstallServiceIn(spec={}),
    )

    result = svc.get_docker_compose_file_path("default", "lemmatizer")

    assert result == expected


def test_get_docker_options_returns_docker_options_for_installed_model(svc: CustomService) -> None:
    model_installed = MagicMock()
    svc.instances_info["default"].installed = InstalledInfo(
        models={"lemmatizer": model_installed},
        options=InstallServiceIn(spec={}),
    )

    result = svc.get_docker_options("default", "lemmatizer")

    assert result is model_installed.docker_options


def test_add_custom_model_adds_to_models(svc: CustomService) -> None:
    model = CustomModel(id="cm-1", data=_CUSTOM_MODEL_DATA)

    svc._add_custom_model("default", model)  # pyright: ignore[reportPrivateUsage]

    assert "my-custom" in svc.models["default"]


def test_get_custom_model_definition_returns_stored_spec(svc: CustomService) -> None:
    model = CustomModel(id="cm-1", data=_CUSTOM_MODEL_DATA)
    svc.instances_info["default"].config.custom = [model]

    assert svc.get_custom_model_definition("cm-1") == _CUSTOM_MODEL_DATA


def test_get_custom_model_definition_unknown_id_returns_none(svc: CustomService) -> None:
    svc.instances_info["default"].config.custom = [CustomModel(id="cm-1", data=_CUSTOM_MODEL_DATA)]

    assert svc.get_custom_model_definition("nonexistent-id") is None


def test_add_custom_model_duplicate_raises(svc: CustomService) -> None:
    svc._add_custom_model("default", CustomModel(id="cm-1", data=_CUSTOM_MODEL_DATA))  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(HTTPException) as exc:
        svc._add_custom_model("default", CustomModel(id="cm-2", data=_CUSTOM_MODEL_DATA))  # pyright: ignore[reportPrivateUsage]

    assert exc.value.status_code == 400


def test_add_custom_model_prefix_collision_raises_400(svc: CustomService) -> None:
    """Duplicating a custom model without changing `default_prefix` must be rejected on add - a
    different `id` alone previously wasn't enough to avoid two models sharing the same endpoint."""
    svc._add_custom_model("default", CustomModel(id="cm-1", data=_CUSTOM_MODEL_DATA))  # pyright: ignore[reportPrivateUsage]
    colliding_data = {**_CUSTOM_MODEL_DATA, "id": "other-custom"}

    with pytest.raises(HTTPException) as exc:
        svc._add_custom_model("default", CustomModel(id="cm-2", data=colliding_data))  # pyright: ignore[reportPrivateUsage]

    assert exc.value.status_code == 400


def test_remove_custom_model_when_in_use_raises(svc: CustomService) -> None:
    model = CustomModel(id="cm-1", data=_CUSTOM_MODEL_DATA)
    svc._add_custom_model("default", model)  # pyright: ignore[reportPrivateUsage]
    svc.instances_info["default"].installed = InstalledInfo(
        models={"my-custom": MagicMock()},
        options=InstallServiceIn(spec={}),
    )

    with pytest.raises(HTTPException) as exc:
        svc._remove_custom_model("default", model)  # pyright: ignore[reportPrivateUsage]

    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_update_custom_model_not_found_raises_404(svc: CustomService) -> None:
    with pytest.raises(HTTPException) as exc:
        await svc.update_custom_model("default", "nonexistent-id", AddCustomModelIn(spec={"name": "x"}))

    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_update_custom_model_persists_and_rebuilds_model(svc: CustomService) -> None:
    model = CustomModel(id="cm-1", data=dict(_CUSTOM_MODEL_DATA))
    svc._add_custom_model("default", model)  # pyright: ignore[reportPrivateUsage]
    svc.instances_info["default"].config.custom = [model]
    svc.service_provider.save_service_config = AsyncMock()

    new_data = {**_CUSTOM_MODEL_DATA, "image": "test/updated:latest"}
    await svc.update_custom_model("default", "cm-1", AddCustomModelIn(spec=new_data))

    assert svc.instances_info["default"].config.custom[0].data["image"] == "test/updated:latest"
    options = svc.models["default"]["my-custom"].options
    assert isinstance(options, DockerOptions)
    assert options.image == "test/updated:latest"


@pytest.mark.asyncio
async def test_update_custom_model_preserves_size_when_form_omits_it(svc: CustomService) -> None:
    model = CustomModel(id="cm-1", data=dict(_CUSTOM_MODEL_DATA))
    svc._add_custom_model("default", model)  # pyright: ignore[reportPrivateUsage]
    svc.instances_info["default"].config.custom = [model]
    svc.service_provider.save_service_config = AsyncMock()

    spec_without_size = {k: v for k, v in _CUSTOM_MODEL_DATA.items() if k != "size"}
    spec_without_size["image"] = "test/updated:latest"
    await svc.update_custom_model("default", "cm-1", AddCustomModelIn(spec=spec_without_size))

    assert svc.instances_info["default"].config.custom[0].data["size"] == "1GB"


@pytest.mark.asyncio
async def test_update_custom_model_rolls_back_when_add_fails(svc: CustomService) -> None:
    model_a = CustomModel(id="cm-1", data={**_CUSTOM_MODEL_DATA, "id": "model-a", "default_prefix": "model-a"})
    model_b = CustomModel(id="cm-2", data={**_CUSTOM_MODEL_DATA, "id": "model-b", "default_prefix": "model-b"})
    svc._add_custom_model("default", model_a)  # pyright: ignore[reportPrivateUsage]
    svc._add_custom_model("default", model_b)  # pyright: ignore[reportPrivateUsage]
    svc.instances_info["default"].config.custom = [model_a, model_b]
    svc.service_provider.save_service_config = AsyncMock()

    with pytest.raises(HTTPException) as exc:
        await svc.update_custom_model(
            "default", "cm-1", AddCustomModelIn(spec={**_CUSTOM_MODEL_DATA, "id": "model-b", "default_prefix": "model-b"})
        )

    assert exc.value.status_code == 400
    assert "model-a" in svc.models["default"]
    assert "model-b" in svc.models["default"]
    assert svc.instances_info["default"].config.custom[0].data["id"] == "model-a"


@pytest.mark.asyncio
async def test_update_custom_model_when_in_use_raises_400(svc: CustomService) -> None:
    model = CustomModel(id="cm-1", data=dict(_CUSTOM_MODEL_DATA))
    svc._add_custom_model("default", model)  # pyright: ignore[reportPrivateUsage]
    svc.instances_info["default"].config.custom = [model]
    svc.instances_info["default"].installed = InstalledInfo(
        models={"my-custom": MagicMock()},
        options=InstallServiceIn(spec={}),
    )

    with pytest.raises(HTTPException) as exc:
        await svc.update_custom_model("default", "cm-1", AddCustomModelIn(spec=_CUSTOM_MODEL_DATA))

    assert exc.value.status_code == 400


def test_remove_custom_model_success(svc: CustomService) -> None:
    model = CustomModel(id="cm-1", data=_CUSTOM_MODEL_DATA)
    svc._add_custom_model("default", model)  # pyright: ignore[reportPrivateUsage]

    svc._remove_custom_model("default", model)  # pyright: ignore[reportPrivateUsage]

    assert "my-custom" not in svc.models["default"]


@pytest.mark.asyncio
async def test_remove_custom_model_that_never_registered_still_prunes_config(svc: CustomService) -> None:
    """A model that failed to register at load time (e.g. a prefix collision caught by load_instance)
    is persisted in config.custom but never entered self.models - removing it must still succeed and
    prune the persisted entry, not KeyError on an unconditional del."""
    model = CustomModel(id="cm-1", data=_CUSTOM_MODEL_DATA)
    svc.instances_info["default"].config.custom = [model]
    svc.service_provider.save_service_config = AsyncMock()
    assert "my-custom" not in svc.models.get("default", {})

    await svc.remove_custom_model("default", "cm-1")

    assert svc.instances_info["default"].config.custom == []


@pytest.mark.asyncio
async def test_get_duplicate_spec_catalog_model_synthesizes_from_defaults(svc: CustomService) -> None:
    """DFINFRA-271: lemmatizer's `options` is a callable resolving docker options from install fields -
    with nothing installed, get_duplicate_spec must resolve it using the fields' own defaults."""
    spec = await svc.get_duplicate_spec("default", "lemmatizer")

    assert spec["id"] == "lemmatizer"
    assert spec["default_prefix"] == "lemmatizer"
    assert spec["size"] == "1.89GB"
    assert spec["image"] == "hub.simplito.com/deepfellow/deepfellow-lemmatizer:1.0.0-cpu"
    assert spec["image_port"] == 8090
    # generate_docker_options defaults these to int/bool literals, not strings - the duplicate spec
    # must stringify them, since SrvCustomCustomModel.envs is dict[str, str] (add_custom_model would
    # otherwise reject them with a validation error).
    assert spec["envs"] == {
        "DF_LEMMATIZER_NUM_WORKERS": "4",
        "DF_LEMMATIZER_QUEUE_MAX": "100000",
        "DF_LEMMATIZER_SHUTDOWN_TIMEOUT": "30",
    }
    assert all(isinstance(v, str) for v in spec["envs"].values())
    assert spec["healthcheck_cmd"] == "wget -q --spider http://localhost:8090/health"
    assert spec["healthcheck_start_period"] == "30s"
    assert spec["description"] == "Multilingual text lemmatization API."


@pytest.mark.asyncio
async def test_get_duplicate_spec_catalog_model_uses_installed_options(svc: CustomService) -> None:
    """When the source model is installed, duplicate resolves docker options from its actual install
    spec (e.g. a non-default num_workers), not the field defaults."""
    installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    installed.models["lemmatizer"] = ModelInstalledInfo(
        id="lemmatizer",
        options=InstallModelIn(spec={"num_workers": 8}),
        docker_options=MagicMock(),
        container_host="",
        container_port=0,
        docker_exposed_port=0,
        registration_id="reg-1",
        prefix="lemmatizer",
        base_url="",
    )
    svc.instances_info["default"].installed = installed

    spec = await svc.get_duplicate_spec("default", "lemmatizer")

    assert spec["envs"]["DF_LEMMATIZER_NUM_WORKERS"] == "8"


@pytest.mark.asyncio
async def test_get_duplicate_spec_preserves_selected_image_version(svc: CustomService) -> None:
    """Duplicating a model pinned to a non-default image_version must keep that version, not
    silently revert to the catalog default tag."""
    installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    installed.models["lemmatizer"] = ModelInstalledInfo(
        id="lemmatizer",
        options=InstallModelIn(spec={"image_version": "1.0.5-cpu"}),
        docker_options=MagicMock(),
        container_host="",
        container_port=0,
        docker_exposed_port=0,
        registration_id="reg-1",
        prefix="lemmatizer",
        base_url="",
    )
    svc.instances_info["default"].installed = installed

    spec = await svc.get_duplicate_spec("default", "lemmatizer")

    assert spec["image"] == "hub.simplito.com/deepfellow/deepfellow-lemmatizer:1.0.5-cpu"


@pytest.mark.asyncio
async def test_get_duplicate_spec_catalog_model_spec_is_addable(svc: CustomService) -> None:
    """End-to-end regression: the synthesized spec must actually pass `_add_custom_model`'s validation
    (SrvCustomCustomModel), not just look plausible - catches type mismatches like int-valued envs."""
    spec = await svc.get_duplicate_spec("default", "lemmatizer")
    spec["id"] = "lemmatizer-copy"
    spec["default_prefix"] = "lemmatizer-copy"

    svc._add_custom_model("default", CustomModel(id="new-uuid", data=spec))  # pyright: ignore[reportPrivateUsage]

    assert "lemmatizer-copy" in svc.models["default"]
    options = svc.models["default"]["lemmatizer-copy"].options
    assert isinstance(options, DockerOptions)
    # A single-colon, two-segment bind mount under the duplicate's own working directory - not the
    # three-segment, two-colon string produced by re-prefixing an already-expanded host:container path.
    assert options.volumes == [f"{svc.get_working_dir()}/lemmatizer_copy_default/volume_0:/root/.cache/stanza"]


@pytest.mark.asyncio
async def test_get_duplicate_spec_strips_host_side_off_an_expanded_volume(svc: CustomService) -> None:
    """`generate_docker_options` callables (e.g. lemmatizer) return an already-expanded host:container
    volume string - get_duplicate_spec must return just the container side, the raw form
    `_add_custom_model` expects, not hand it straight back to be re-prefixed a second time."""
    spec = await svc.get_duplicate_spec("default", "lemmatizer")

    assert spec["volumes"] == ["/root/.cache/stanza"]


@pytest.mark.asyncio
async def test_get_duplicate_spec_custom_backed_model_returns_stored_definition(svc: CustomService) -> None:
    model = CustomModel(id="cm-1", data=dict(_CUSTOM_MODEL_DATA))
    svc._add_custom_model("default", model)  # pyright: ignore[reportPrivateUsage]
    svc.instances_info["default"].config.custom = [model]

    spec = await svc.get_duplicate_spec("default", "my-custom")

    assert spec == _CUSTOM_MODEL_DATA


@pytest.mark.asyncio
async def test_get_duplicate_spec_custom_backed_missing_definition_raises_404(svc: CustomService) -> None:
    """Registry has `custom=<id>` but no matching CustomModel definition (data inconsistency)."""
    model = CustomModel(id="cm-1", data=dict(_CUSTOM_MODEL_DATA))
    svc._add_custom_model("default", model)  # pyright: ignore[reportPrivateUsage]
    # Deliberately not setting `svc.instances_info["default"].config.custom` here.

    with pytest.raises(HTTPException) as exc:
        await svc.get_duplicate_spec("default", "my-custom")

    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_get_duplicate_spec_unknown_model_raises_400(svc: CustomService) -> None:
    with pytest.raises(HTTPException) as exc:
        await svc.get_duplicate_spec("default", "ghost")

    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_list_models_unknown_instance_raises(svc: CustomService) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))

    with pytest.raises(HTTPException) as exc:
        await svc.list_models("ghost", ListModelsFilters())

    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_list_models_returns_all_when_no_filter(svc: CustomService) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))

    result = await svc.list_models("default", ListModelsFilters(installed=None))

    assert len(result.list) > 0


@pytest.mark.asyncio
async def test_list_models_filter_not_installed(svc: CustomService) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))

    result = await svc.list_models("default", ListModelsFilters(installed=False))

    assert len(result.list) > 0
    assert all(item.installed is False for item in result.list)


@pytest.mark.asyncio
async def test_list_models_filter_installed_returns_only_installed(svc: CustomService) -> None:
    mock_installed = MagicMock()
    mock_installed.get_info.return_value = ModelInfo(spec={"prefix": "lemmatizer"}, registration_id="r1")
    mock_installed.prefix = "lemmatizer"
    svc.instances_info["default"].installed = InstalledInfo(
        models={"lemmatizer": mock_installed},
        options=InstallServiceIn(spec={}),
    )

    result = await svc.list_models("default", ListModelsFilters(installed=True))

    assert len(result.list) == 1
    assert result.list[0].id == "lemmatizer"


@pytest.mark.asyncio
async def test_list_models_none_instance_iterates_all(svc: CustomService) -> None:
    svc.instances_info["gpu-1"] = Instance(None, None, {}, InstanceConfig())
    svc.load_default_models("gpu-1")
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    svc.instances_info["gpu-1"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))

    result = await svc.list_models(None, ListModelsFilters(installed=None))

    assert len(result.list) > 0


@pytest.mark.asyncio
async def test_get_model_not_found_raises(svc: CustomService) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))

    with pytest.raises(HTTPException) as exc:
        await svc.get_model("default", "ghost")

    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_get_model_not_installed_returns_false(svc: CustomService) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))

    result = await svc.get_model("default", "lemmatizer")

    assert result.id == "lemmatizer"
    assert result.installed is False


@pytest.mark.asyncio
async def test_get_model_installed_returns_model_info(svc: CustomService) -> None:
    model_info = ModelInfo(spec={"prefix": "lemmatizer"}, registration_id="r1")
    mock_installed = MagicMock()
    mock_installed.get_info.return_value = model_info
    mock_installed.prefix = "lemmatizer"
    svc.instances_info["default"].installed = InstalledInfo(
        models={"lemmatizer": mock_installed},
        options=InstallServiceIn(spec={}),
    )

    result = await svc.get_model("default", "lemmatizer")

    assert result.installed == model_info


@pytest.mark.asyncio
async def test_install_model_already_installed_returns_ok(svc: CustomService) -> None:
    svc.instances_info["default"].installed = InstalledInfo(
        models={"lemmatizer": MagicMock()},
        options=InstallServiceIn(spec={}),
    )

    promise = await svc._install_model("default", "lemmatizer", InstallModelIn())  # pyright: ignore[reportPrivateUsage]

    result = await promise.wait()
    assert result.status == "OK"
    assert "Already installed" in result.details


@pytest.mark.asyncio
async def test_install_model_not_found_raises(svc: CustomService) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))

    with pytest.raises(HTTPException) as exc:
        await svc._install_model("default", "ghost", InstallModelIn())  # pyright: ignore[reportPrivateUsage]

    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_install_model_success(svc: CustomService, deps: dict[str, Any]) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    deps["docker_service"].get_image_warnings = AsyncMock(return_value=[])
    deps["docker_service"].is_docker_image_pulled = AsyncMock(return_value=True)
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=(8090, True, False))
    deps["docker_service"].get_container_host.return_value = "172.20.0.1"
    deps["docker_service"].get_container_port.return_value = 8090
    deps["endpoint_registry"].register_custom_endpoint_as_proxy.return_value = "reg-1"

    promise = await svc._install_model(  # pyright: ignore[reportPrivateUsage]
        "default", "lemmatizer", InstallModelIn(spec={"prefix": "lemmatizer"})
    )
    result = await promise.wait()

    assert result.status == "OK"
    assert "lemmatizer" in svc.instances_info["default"].installed.models  # pyright: ignore[reportOptionalMemberAccess]
    registered_options = deps["endpoint_registry"].register_custom_endpoint_as_proxy.call_args.kwargs["options"]
    assert registered_options.read_timeout_seconds == 300


@pytest.mark.asyncio
async def test_install_model_uses_configured_standard_proxy_timeout(svc: CustomService, deps: dict[str, Any]) -> None:
    deps["config"].standard_proxy_timeout_seconds = 1800
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    deps["docker_service"].get_image_warnings = AsyncMock(return_value=[])
    deps["docker_service"].is_docker_image_pulled = AsyncMock(return_value=True)
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=(8090, True, False))
    deps["docker_service"].get_container_host.return_value = "172.20.0.1"
    deps["docker_service"].get_container_port.return_value = 8090
    deps["endpoint_registry"].register_custom_endpoint_as_proxy.return_value = "reg-1"

    promise = await svc._install_model(  # pyright: ignore[reportPrivateUsage]
        "default", "lemmatizer", InstallModelIn(spec={"prefix": "lemmatizer"})
    )
    await promise.wait()

    registered_options = deps["endpoint_registry"].register_custom_endpoint_as_proxy.call_args.kwargs["options"]
    assert registered_options.read_timeout_seconds == 1800


def _mock_successful_install(deps: dict[str, Any]) -> None:
    deps["docker_service"].get_image_warnings = AsyncMock(return_value=[])
    deps["docker_service"].is_docker_image_pulled = AsyncMock(return_value=True)
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=(8090, True, False))
    deps["docker_service"].get_container_host.return_value = "172.20.0.1"
    deps["docker_service"].get_container_port.return_value = 8090
    deps["endpoint_registry"].register_custom_endpoint_as_proxy.return_value = "reg-1"


@pytest.mark.asyncio
async def test_install_model_uses_the_models_default_proxy_timeout_over_the_standard_one(svc: CustomService, deps: dict[str, Any]) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    _mock_successful_install(deps)

    with patch.object(svc, "validate_docker_image_version_for_model", new_callable=AsyncMock):
        promise = await svc._install_model(  # pyright: ignore[reportPrivateUsage]
            "default",
            "doc_chunker",
            InstallModelIn(
                spec={"prefix": "doc_chunker", "hardware": "CPU", "max_concurrent": 2, "async_priority_workers": 1, "sync_only_workers": 0}
            ),
        )
    await promise.wait()

    registered_options = deps["endpoint_registry"].register_custom_endpoint_as_proxy.call_args.kwargs["options"]
    assert registered_options.read_timeout_seconds == DOC_CHUNKER_PROXY_TIMEOUT_SECONDS


@pytest.mark.asyncio
async def test_install_model_prefers_the_install_proxy_timeout_over_the_models_default(svc: CustomService, deps: dict[str, Any]) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    _mock_successful_install(deps)

    with patch.object(svc, "validate_docker_image_version_for_model", new_callable=AsyncMock):
        promise = await svc._install_model(  # pyright: ignore[reportPrivateUsage]
            "default",
            "doc_chunker",
            InstallModelIn(
                spec={
                    "prefix": "doc_chunker",
                    "hardware": "CPU",
                    "max_concurrent": 2,
                    "async_priority_workers": 1,
                    "sync_only_workers": 0,
                    "proxy_timeout_seconds": 42,
                }
            ),
        )
    await promise.wait()

    registered_options = deps["endpoint_registry"].register_custom_endpoint_as_proxy.call_args.kwargs["options"]
    assert registered_options.read_timeout_seconds == 42


def test_doc_chunker_defaults_to_an_hour_per_document_conversion(svc: CustomService) -> None:
    model = svc.models["default"]["doc_chunker"]

    docker_options = model.options({"hardware": "CPU"})  # pyright: ignore[reportCallIssue]

    assert docker_options.env_vars["DOCUMENT_TIMEOUT"] == "3600"  # pyright: ignore[reportOptionalSubscript]
    assert DOC_CHUNKER_PROXY_TIMEOUT_SECONDS > 3600


def test_doc_chunker_proxy_timeout_field_names_its_own_default(svc: CustomService) -> None:
    fields = svc.models["default"]["doc_chunker"].model_spec.fields

    proxy_field = next(field for field in fields if field.name == "proxy_timeout_seconds")

    assert f"{DOC_CHUNKER_PROXY_TIMEOUT_SECONDS}s" in proxy_field.description


def test_doc_chunker_mounts_data_dir_and_sets_async_job_defaults(svc: CustomService) -> None:
    model = svc.models["default"]["doc_chunker"]

    docker_options = model.options({"hardware": "CPU"})  # pyright: ignore[reportCallIssue]

    assert docker_options.volumes == [f"{svc.get_working_dir()}/doc_chunker/data:/var/lib/doc-chunker"]
    env_vars = docker_options.env_vars or {}
    assert env_vars["MAX_QUEUED_JOBS"] == "100"
    assert env_vars["JOB_RESULT_TTL"] == "86400"
    assert "TOKENIZER_DIR" not in env_vars


def test_doc_chunker_worker_fields_are_optional(svc: CustomService) -> None:
    fields = {field.name: field for field in svc.models["default"]["doc_chunker"].model_spec.fields}

    for name, default in (("max_concurrent", "2"), ("async_priority_workers", "0"), ("sync_only_workers", "0")):
        assert fields[name].required is False
        assert fields[name].default == default


def test_doc_chunker_passes_worker_split_to_container(svc: CustomService) -> None:
    model = svc.models["default"]["doc_chunker"]

    docker_options = model.options(  # pyright: ignore[reportCallIssue]
        {"hardware": "CPU", "max_concurrent": 4, "async_priority_workers": 2, "sync_only_workers": 1}
    )

    env_vars = docker_options.env_vars or {}
    assert env_vars["MAX_CONCURRENT"] == "4"
    assert env_vars["ASYNC_PRIORITY_WORKERS"] == "2"
    assert env_vars["SYNC_ONLY_WORKERS"] == "1"


def test_doc_chunker_options_fall_back_to_field_defaults_for_partial_spec(svc: CustomService) -> None:
    """Image resolution and model export call options with just the hardware - only install validates."""
    model = svc.models["default"]["doc_chunker"]

    docker_options = model.options({"hardware": "CPU"})  # pyright: ignore[reportCallIssue]

    env_vars = docker_options.env_vars or {}
    assert env_vars["MAX_CONCURRENT"] == "2"
    assert env_vars["ASYNC_PRIORITY_WORKERS"] == "0"
    assert env_vars["SYNC_ONLY_WORKERS"] == "0"


def test_doc_chunker_leaves_cleared_worker_fields_unset(svc: CustomService) -> None:
    """A number field cleared in the WebUI is sent as null - doc-chunker must fall back to its own default."""
    model = svc.models["default"]["doc_chunker"]

    docker_options = model.options(  # pyright: ignore[reportCallIssue]
        {"hardware": "CPU", "max_concurrent": None, "async_priority_workers": None, "sync_only_workers": None}
    )

    env_vars = docker_options.env_vars or {}
    assert "MAX_CONCURRENT" not in env_vars
    assert "ASYNC_PRIORITY_WORKERS" not in env_vars
    assert "SYNC_ONLY_WORKERS" not in env_vars


@pytest.mark.parametrize(
    "fields",
    [
        {"max_concurrent": 1, "async_priority_workers": 0, "sync_only_workers": 0},
        {"max_concurrent": 2, "async_priority_workers": 1, "sync_only_workers": 1},
        {"max_concurrent": "4", "async_priority_workers": "3", "sync_only_workers": "0"},
        {},
        {"max_concurrent": "", "async_priority_workers": "", "sync_only_workers": ""},
        # max_concurrent left to doc-chunker's auto-sizing - the split can't be checked against it
        {"max_concurrent": "", "async_priority_workers": 5, "sync_only_workers": 5},
        # unset async_priority_workers counts as doc-chunker's default of 0
        {"max_concurrent": 1, "async_priority_workers": "", "sync_only_workers": 0},
        {"max_concurrent": 3, "async_priority_workers": "", "sync_only_workers": 2},
    ],
)
def test_validate_doc_chunker_workers_accepts_valid_split(fields: dict[str, Any]) -> None:
    validate_doc_chunker_workers(fields)


@pytest.mark.parametrize(
    ("fields", "field", "message"),
    [
        (
            {"max_concurrent": 2, "async_priority_workers": "", "sync_only_workers": 2},
            "sync_only_workers",
            "must be lower than max_concurrent (2)",
        ),
        (
            {"max_concurrent": 3, "async_priority_workers": "", "sync_only_workers": 3},
            "sync_only_workers",
            "must be lower than max_concurrent (3)",
        ),
        (
            {"max_concurrent": 2, "sync_only_workers": 2},
            "sync_only_workers",
            "must be lower than max_concurrent (2)",
        ),
        (
            {"max_concurrent": 2, "async_priority_workers": 2, "sync_only_workers": 0},
            "async_priority_workers",
            "must be lower than max_concurrent (2)",
        ),
        (
            {"max_concurrent": 1, "async_priority_workers": 0, "sync_only_workers": 1},
            "sync_only_workers",
            "must be lower than max_concurrent (1)",
        ),
        (
            {"max_concurrent": 3, "async_priority_workers": 2, "sync_only_workers": 2},
            "sync_only_workers",
            "together with async_priority_workers (2) must not exceed max_concurrent (3)",
        ),
        ({"max_concurrent": 0, "async_priority_workers": 0, "sync_only_workers": 0}, "max_concurrent", "greater than or equal to 1"),
        ({"max_concurrent": 2, "async_priority_workers": 0, "sync_only_workers": -1}, "sync_only_workers", "greater than or equal to 0"),
        ({"max_concurrent": 2, "async_priority_workers": "many", "sync_only_workers": 0}, "async_priority_workers", "valid integer"),
        ({"max_concurrent": "1.5", "async_priority_workers": 0, "sync_only_workers": 0}, "max_concurrent", "valid integer"),
    ],
)
def test_validate_doc_chunker_workers_rejects_invalid_split(fields: dict[str, Any], field: str, message: str) -> None:
    with pytest.raises(HTTPException) as exc:
        validate_doc_chunker_workers(fields)

    assert exc.value.status_code == 400
    errors = cast("list[dict[str, str]]", exc.value.detail)
    assert any(error["field"] == field and message in error["message"] for error in errors)


@pytest.mark.asyncio
async def test_install_doc_chunker_rejects_invalid_worker_split_before_pulling_image(svc: CustomService, deps: dict[str, Any]) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    _mock_successful_install(deps)

    with patch.object(svc, "validate_docker_image_version_for_model", new_callable=AsyncMock), pytest.raises(HTTPException) as exc:
        await svc._install_model(  # pyright: ignore[reportPrivateUsage]
            "default",
            "doc_chunker",
            InstallModelIn(spec={"prefix": "doc_chunker", "hardware": "CPU", "max_concurrent": 2, "sync_only_workers": 2}),
        )

    assert exc.value.status_code == 400
    deps["docker_service"].install_and_run_docker.assert_not_called()
    assert ("default", "doc_chunker") not in svc._installing  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_install_model_rejects_unavailable_image_version(svc: CustomService, deps: dict[str, Any]) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    client = MagicMock()
    client.get_tags = AsyncMock(return_value=["1.0.0-cpu"])
    client.tag_exists = AsyncMock(return_value=False)
    with patch("server.services.custom_service.registry_for", return_value=client), pytest.raises(HTTPException) as exc:
        await svc._install_model(  # pyright: ignore[reportPrivateUsage]
            "default", "lemmatizer", InstallModelIn(spec={"prefix": "lemmatizer", "image_version": "9.9.9-bogus"})
        )

    assert exc.value.status_code == 400
    assert "lemmatizer" not in svc.instances_info["default"].installed.models  # pyright: ignore[reportOptionalMemberAccess]


@pytest.mark.asyncio
async def test_install_model_applies_selected_image_version(svc: CustomService, deps: dict[str, Any]) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    deps["docker_service"].get_image_warnings = AsyncMock(return_value=[])
    deps["docker_service"].is_docker_image_pulled = AsyncMock(return_value=True)
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=(8090, True, False))
    deps["docker_service"].get_container_host.return_value = "172.20.0.1"
    deps["docker_service"].get_container_port.return_value = 8090
    deps["endpoint_registry"].register_custom_endpoint_as_proxy.return_value = "reg-1"

    with patch.object(svc, "validate_docker_image_version_for_model", new_callable=AsyncMock):
        promise = await svc._install_model(  # pyright: ignore[reportPrivateUsage]
            "default", "lemmatizer", InstallModelIn(spec={"prefix": "lemmatizer", "image_version": "1.1.0-cpu"})
        )
    await promise.wait()

    installed_model = svc.instances_info["default"].installed.models["lemmatizer"]  # pyright: ignore[reportOptionalMemberAccess]
    assert installed_model.docker_options.image == "hub.simplito.com/deepfellow/deepfellow-lemmatizer:1.1.0-cpu"
    # The model's own default image logic is untouched for subsequent installs.
    assert svc.models["default"]["lemmatizer"].options({"hardware": "CPU"}).image == (  # pyright: ignore[reportCallIssue]
        "hub.simplito.com/deepfellow/deepfellow-lemmatizer:1.0.0-cpu"
    )


@pytest.mark.asyncio
async def test_install_model_cleans_up_installing_set_when_cancelled_during_setup(svc: CustomService) -> None:
    """A cancellation during the pre-promise docker-image verification must still discard the
    `_installing` entry - there was no cleanup at all for this window before this fix.
    """
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    model_id = "lemmatizer"
    reached = asyncio.Event()

    async def hanging_verify(*_args: Any, **_kwargs: Any) -> None:
        reached.set()
        await asyncio.Event().wait()  # never completes on its own - must be cancelled

    with (
        patch.object(svc, "validate_docker_image_version_for_model", new_callable=AsyncMock),
        patch.object(svc, "_verify_docker_image", new=hanging_verify),
    ):
        task = asyncio.create_task(
            svc._install_model("default", model_id, InstallModelIn(spec={"prefix": model_id}))  # pyright: ignore[reportPrivateUsage]
        )
        await reached.wait()

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert ("default", model_id) not in svc._installing  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_install_model_applies_selected_image_version_for_static_image_model(svc: CustomService, deps: dict[str, Any]) -> None:
    # "easyOCR" uses a single shared (non-callable) DockerOptions instance — the override must not
    # mutate it in place, or the override would leak into every future install of this model.
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    deps["docker_service"].get_image_warnings = AsyncMock(return_value=[])
    deps["docker_service"].is_docker_image_pulled = AsyncMock(return_value=True)
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=(8000, True, False))
    deps["docker_service"].get_container_host.return_value = "172.20.0.1"
    deps["docker_service"].get_container_port.return_value = 8000
    deps["endpoint_registry"].register_custom_endpoint_as_proxy.return_value = "reg-1"
    original_image = svc.models["default"]["easyOCR"].options.image  # pyright: ignore[reportFunctionMemberAccess]

    with patch.object(svc, "validate_docker_image_version_for_model", new_callable=AsyncMock):
        promise = await svc._install_model(  # pyright: ignore[reportPrivateUsage]
            "default", "easyOCR", InstallModelIn(spec={"prefix": "ocr", "image_version": "2.0.0"})
        )
    await promise.wait()

    installed_model = svc.instances_info["default"].installed.models["easyOCR"]  # pyright: ignore[reportOptionalMemberAccess]
    assert installed_model.docker_options.image == "hub.simplito.com/deepfellow/df-ocr:2.0.0"
    assert svc.models["default"]["easyOCR"].options.image == original_image  # pyright: ignore[reportFunctionMemberAccess]


@pytest.mark.asyncio
async def test_install_model_uses_container_name_override_when_given(svc: CustomService, deps: dict[str, Any]) -> None:
    # "easyOCR" uses a single shared (non-callable) DockerOptions instance — the override must not
    # mutate it in place, or the pinned name would leak into every future install of this model.
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    deps["docker_service"].get_image_warnings = AsyncMock(return_value=[])
    deps["docker_service"].is_docker_image_pulled = AsyncMock(return_value=True)
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=(8000, True, False))
    deps["docker_service"].get_container_host.return_value = "172.20.0.1"
    deps["docker_service"].get_container_port.return_value = 8000
    deps["endpoint_registry"].register_custom_endpoint_as_proxy.return_value = "reg-1"
    original_container_name = svc.models["default"]["easyOCR"].options.container_name  # pyright: ignore[reportFunctionMemberAccess]

    promise = await svc._install_model(  # pyright: ignore[reportPrivateUsage]
        "default", "easyOCR", InstallModelIn(spec={"prefix": "ocr"}), "pinned-name"
    )
    await promise.wait()

    docker_options = deps["docker_service"].install_and_run_docker.call_args.args[0]
    assert docker_options.container_name == "pinned-name"
    installed_model = svc.instances_info["default"].installed.models["easyOCR"]  # pyright: ignore[reportOptionalMemberAccess]
    assert installed_model.docker_options.container_name == "pinned-name"
    assert svc.models["default"]["easyOCR"].options.container_name == original_container_name  # pyright: ignore[reportFunctionMemberAccess]


@pytest.mark.asyncio
async def test_install_model_uses_container_name_override_for_callable_options(svc: CustomService, deps: dict[str, Any]) -> None:
    """Models whose options are built per install (e.g. bge-m3) must keep their pinned name on restart as well."""
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    deps["docker_service"].get_image_warnings = AsyncMock(return_value=[])
    deps["docker_service"].is_docker_image_pulled = AsyncMock(return_value=True)
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=(8070, True, False))
    deps["docker_service"].get_container_host.return_value = "172.20.0.1"
    deps["docker_service"].get_container_port.return_value = 8070
    deps["endpoint_registry"].register_custom_endpoint_as_proxy.return_value = "reg-1"

    promise = await svc._install_model(  # pyright: ignore[reportPrivateUsage]
        "default", "deepfellow-bge-m3", InstallModelIn(spec={"hardware": "CPU"}), "pinned-name"
    )
    await promise.wait()

    docker_options = deps["docker_service"].install_and_run_docker.call_args.args[0]
    assert docker_options.container_name == "pinned-name"
    installed_model = svc.instances_info["default"].installed.models["deepfellow-bge-m3"]  # pyright: ignore[reportOptionalMemberAccess]
    assert installed_model.docker_options.container_name == "pinned-name"


@pytest.mark.asyncio
async def test_install_model_falls_back_to_generated_container_name(svc: CustomService, deps: dict[str, Any]) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    deps["docker_service"].get_image_warnings = AsyncMock(return_value=[])
    deps["docker_service"].is_docker_image_pulled = AsyncMock(return_value=True)
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=(8000, True, False))
    deps["docker_service"].get_container_host.return_value = "172.20.0.1"
    deps["docker_service"].get_container_port.return_value = 8000
    deps["endpoint_registry"].register_custom_endpoint_as_proxy.return_value = "reg-1"

    promise = await svc._install_model("default", "easyOCR", InstallModelIn(spec={"prefix": "ocr"}))  # pyright: ignore[reportPrivateUsage]
    await promise.wait()

    docker_options = deps["docker_service"].install_and_run_docker.call_args.args[0]
    assert docker_options.container_name == "df-easyocr"


@pytest.mark.asyncio
async def test_install_model_applies_suffix_changed_after_model_list_was_built(svc: CustomService, deps: dict[str, Any]) -> None:
    """easyOCR's static options were built at load time; a later live suffix change must still reach new installs."""
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    deps["docker_service"].get_image_warnings = AsyncMock(return_value=[])
    deps["docker_service"].is_docker_image_pulled = AsyncMock(return_value=True)
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=(8000, True, False))
    deps["docker_service"].get_container_host.return_value = "172.20.0.1"
    deps["docker_service"].get_container_port.return_value = 8000
    deps["endpoint_registry"].register_custom_endpoint_as_proxy.return_value = "reg-1"
    deps["docker_service"].get_docker_container_name.side_effect = lambda name, **_: f"{name}_new"  # pyright: ignore[reportUnknownLambdaType]

    promise = await svc._install_model("default", "easyOCR", InstallModelIn(spec={"prefix": "ocr"}))  # pyright: ignore[reportPrivateUsage]
    await promise.wait()

    docker_options = deps["docker_service"].install_and_run_docker.call_args.args[0]
    assert docker_options.container_name == "easyocr_new"
    assert svc.models["default"]["easyOCR"].options.container_name == "df-easyocr"  # pyright: ignore[reportFunctionMemberAccess]


@pytest.mark.asyncio
@pytest.mark.parametrize(("override", "expected_host"), [("pinned-name", "pinned-name"), (None, "easyocr_new")])
async def test_install_model_derives_container_host_from_final_container_name(
    svc: CustomService, deps: dict[str, Any], override: str | None, expected_host: str
) -> None:
    """The registered host must follow the pinned or re-derived name, not the static options' load-time one."""
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    deps["docker_service"].get_image_warnings = AsyncMock(return_value=[])
    deps["docker_service"].is_docker_image_pulled = AsyncMock(return_value=True)
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=(8000, True, False))
    deps["docker_service"].get_container_host.side_effect = lambda _subnet, options: options.container_name  # pyright: ignore[reportUnknownLambdaType, reportUnknownMemberType]
    deps["docker_service"].get_container_port.return_value = 8000
    deps["docker_service"].get_docker_container_name.side_effect = lambda name, **_: f"{name}_new"  # pyright: ignore[reportUnknownLambdaType]
    deps["endpoint_registry"].register_custom_endpoint_as_proxy.return_value = "reg-1"

    promise = await svc._install_model("default", "easyOCR", InstallModelIn(spec={"prefix": "ocr"}), override)  # pyright: ignore[reportPrivateUsage]
    await promise.wait()

    installed_model = svc.instances_info["default"].installed.models["easyOCR"]  # pyright: ignore[reportOptionalMemberAccess]
    assert installed_model.container_host == expected_host


@pytest.mark.asyncio
async def test_install_model_uses_spec_proxy_timeout_seconds(svc: CustomService, deps: dict[str, Any]) -> None:
    deps["config"].standard_proxy_timeout_seconds = 1800
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    deps["docker_service"].get_image_warnings = AsyncMock(return_value=[])
    deps["docker_service"].is_docker_image_pulled = AsyncMock(return_value=True)
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=(8090, True, False))
    deps["docker_service"].get_container_host.return_value = "172.20.0.1"
    deps["docker_service"].get_container_port.return_value = 8090
    deps["endpoint_registry"].register_custom_endpoint_as_proxy.return_value = "reg-1"

    promise = await svc._install_model(  # pyright: ignore[reportPrivateUsage]
        "default", "lemmatizer", InstallModelIn(spec={"prefix": "lemmatizer", "proxy_timeout_seconds": 120})
    )
    await promise.wait()

    registered_options = deps["endpoint_registry"].register_custom_endpoint_as_proxy.call_args.kwargs["options"]
    assert registered_options.read_timeout_seconds == 120


@pytest.mark.asyncio
async def test_install_model_treats_null_proxy_timeout_seconds_as_unset(svc: CustomService, deps: dict[str, Any]) -> None:
    deps["config"].standard_proxy_timeout_seconds = 1800
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    deps["docker_service"].get_image_warnings = AsyncMock(return_value=[])
    deps["docker_service"].is_docker_image_pulled = AsyncMock(return_value=True)
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=(8090, True, False))
    deps["docker_service"].get_container_host.return_value = "172.20.0.1"
    deps["docker_service"].get_container_port.return_value = 8090
    deps["endpoint_registry"].register_custom_endpoint_as_proxy.return_value = "reg-1"

    promise = await svc._install_model(  # pyright: ignore[reportPrivateUsage]
        "default", "lemmatizer", InstallModelIn(spec={"prefix": "lemmatizer", "proxy_timeout_seconds": None})
    )
    await promise.wait()

    registered_options = deps["endpoint_registry"].register_custom_endpoint_as_proxy.call_args.kwargs["options"]
    assert registered_options.read_timeout_seconds == 1800


def test_proxy_timeout_field_has_no_baked_in_default(svc: CustomService) -> None:
    field = svc.get_proxy_timeout_field()
    assert field.default is None
    assert field.placeholder is None


@pytest.mark.asyncio
async def test_install_model_rejects_invalid_proxy_timeout_seconds_before_starting_docker(svc: CustomService, deps: dict[str, Any]) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))

    with pytest.raises(HTTPException) as exc:
        await svc._install_model(  # pyright: ignore[reportPrivateUsage]
            "default", "lemmatizer", InstallModelIn(spec={"prefix": "lemmatizer", "proxy_timeout_seconds": "abc"})
        )

    assert exc.value.status_code == 400
    deps["docker_service"].install_and_run_docker.assert_not_called()
    assert "lemmatizer" not in svc.instances_info["default"].installed.models  # pyright: ignore[reportOptionalMemberAccess]


@pytest.mark.asyncio
@pytest.mark.parametrize("proxy_timeout_seconds", [0, -1])
async def test_install_model_rejects_non_positive_proxy_timeout_seconds_before_starting_docker(
    svc: CustomService, deps: dict[str, Any], proxy_timeout_seconds: int
) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))

    with pytest.raises(HTTPException) as exc:
        await svc._install_model(  # pyright: ignore[reportPrivateUsage]
            "default", "lemmatizer", InstallModelIn(spec={"prefix": "lemmatizer", "proxy_timeout_seconds": proxy_timeout_seconds})
        )

    assert exc.value.status_code == 400
    deps["docker_service"].install_and_run_docker.assert_not_called()
    assert "lemmatizer" not in svc.instances_info["default"].installed.models  # pyright: ignore[reportOptionalMemberAccess]


@pytest.mark.asyncio
async def test_uninstall_model_removes_from_installed(svc: CustomService, deps: dict[str, Any]) -> None:
    mock_model = MagicMock()
    mock_model.prefix = "lemmatizer"
    mock_model.registration_id = "r1"
    deps["docker_service"].uninstall_docker = AsyncMock()
    svc.instances_info["default"].installed = InstalledInfo(
        models={"lemmatizer": mock_model},
        options=InstallServiceIn(spec={}),
    )

    await svc._uninstall_model("default", "lemmatizer", UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert "lemmatizer" not in svc.instances_info["default"].installed.models  # pyright: ignore[reportOptionalMemberAccess]
    assert deps["endpoint_registry"].unregister_custom_endpoint.call_count == 1


@pytest.mark.asyncio
async def test_uninstall_model_with_purge_removes_downloaded(svc: CustomService, deps: dict[str, Any]) -> None:
    mock_model = MagicMock()
    mock_model.prefix = "lemmatizer"
    mock_model.registration_id = "r1"
    deps["docker_service"].uninstall_docker = AsyncMock()
    deps["docker_service"].remove_image = AsyncMock()
    svc.models_downloaded["lemmatizer"] = DownloadedInfo(image="lemmatizer:latest")
    svc.instances_info["default"].installed = InstalledInfo(
        models={"lemmatizer": mock_model},
        options=InstallServiceIn(spec={}),
    )

    await svc._uninstall_model("default", "lemmatizer", UninstallModelIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert "lemmatizer" not in svc.models_downloaded
    assert deps["docker_service"].remove_image.call_count == 1
    assert deps["docker_service"].remove_image.call_args == call("lemmatizer:latest")


def test_create_lemmatizer_model_cpu_image(svc: CustomService) -> None:
    model = create_lemmatizer_model(svc, "172.20.0.0/16")

    assert callable(model.options)
    docker_opts = model.options({"hardware": "CPU"})  # pyright: ignore[reportCallIssue]
    assert "cpu" in docker_opts.image


def test_create_lemmatizer_model_gpu_image(deps: dict[str, Any]) -> None:
    gpu = NvidiaGpuInfo(name="RTX 4090", vram="24GB", id=0)
    deps["hardware"] = MagicMock(gpus=[gpu], cpu=MagicMock())
    svc_gpu = CustomService(**deps)

    model = create_lemmatizer_model(svc_gpu, "172.20.0.0/16")

    docker_opts = model.options({"hardware": "GPU"})  # pyright: ignore[reportCallIssue]
    assert "cuda" in docker_opts.image


def test_create_gliner_model_cpu_image(svc: CustomService) -> None:
    model = create_gliner_model(svc, "172.20.0.0/16")

    assert callable(model.options)
    docker_opts = model.options({"hardware": "CPU"})  # pyright: ignore[reportCallIssue]
    assert "cpu" in docker_opts.image


def test_create_gliner_model_gpu_image(deps: dict[str, Any]) -> None:
    gpu = NvidiaGpuInfo(name="RTX 4090", vram="24GB", id=0)
    deps["hardware"] = MagicMock(gpus=[gpu], cpu=MagicMock())
    svc_gpu = CustomService(**deps)

    model = create_gliner_model(svc_gpu, "172.20.0.0/16")

    docker_opts = model.options({"hardware": "GPU"})  # pyright: ignore[reportCallIssue]
    assert "gpu" in docker_opts.image


def test_create_bge_m3_model_cpu_image(svc: CustomService) -> None:
    model = create_bge_m3_model(svc, "172.20.0.0/16")

    assert callable(model.options)
    docker_opts = model.options({"hardware": "CPU"})  # pyright: ignore[reportCallIssue]
    assert "cpu" in docker_opts.image


def test_create_bge_m3_model_declares_context_window(svc: CustomService) -> None:
    """Consumers chunk against this ceiling; leaving it unset made them fall back to a smaller guess."""
    model = create_bge_m3_model(svc, "172.20.0.0/16")

    assert model.model_props.max_context_window == 8192
    assert model.model_props.context_window == 8192


def test_create_bge_m3_model_gpu_image(deps: dict[str, Any]) -> None:
    gpu = NvidiaGpuInfo(name="RTX 4090", vram="24GB", id=0)
    deps["hardware"] = MagicMock(gpus=[gpu], cpu=MagicMock())
    svc_gpu = CustomService(**deps)

    model = create_bge_m3_model(svc_gpu, "172.20.0.0/16")

    docker_opts = model.options({"hardware": "GPU"})  # pyright: ignore[reportCallIssue]
    assert "cuda" in docker_opts.image


def test_create_doc_chunker_model_cpu_image(svc: CustomService) -> None:
    model = create_doc_chunker_model(svc, "172.20.0.0/16")

    assert callable(model.options)
    docker_opts = model.options({"hardware": "CPU"})  # pyright: ignore[reportCallIssue]
    assert "cpu" in docker_opts.image


def test_create_doc_chunker_model_gpu_image(deps: dict[str, Any]) -> None:
    gpu = NvidiaGpuInfo(name="RTX 4090", vram="24GB", id=0)
    deps["hardware"] = MagicMock(gpus=[gpu], cpu=MagicMock())
    svc_gpu = CustomService(**deps)

    model = create_doc_chunker_model(svc_gpu, "172.20.0.0/16")

    docker_opts = model.options({"hardware": "GPU"})  # pyright: ignore[reportCallIssue]
    assert "gpu" in docker_opts.image


def test_create_doc_chunker_model_shares_one_gateway_address(svc: CustomService) -> None:
    """Image description and transcription talk to the same gateway, so the form asks for it once."""
    model = create_doc_chunker_model(svc, "172.20.0.0/16")

    docker_opts = model.options(  # pyright: ignore[reportCallIssue]
        {"hardware": "CPU", "infra_api_url": "http://infra:8086", "infra_api_key": "secret"}
    )

    assert docker_opts.env_vars is not None
    assert docker_opts.env_vars["INFRA_API_URL"] == "http://infra:8086"
    assert docker_opts.env_vars["INFRA_API_KEY"] == "secret"


def test_create_doc_chunker_model_falls_back_to_default_version(svc: CustomService) -> None:
    model = create_doc_chunker_model(svc, "172.20.0.0/16")

    docker_opts = model.options({"hardware": "CPU"})  # pyright: ignore[reportCallIssue]
    assert docker_opts.image == "hub.simplito.com/deepfellow/doc-chunker-cpu:v2.4.0"


@pytest.mark.parametrize(
    ("url", "infra_url", "expected"),
    [
        ("http://infra:8086", "", True),
        ("http://infra:8086/", "", True),
        ("HTTP://INFRA:8086", "", True),
        ("  http://infra:8086  ", "", True),
        ("http://infra:8087", "", False),
        ("http://infra", "", False),
        ("https://infra:8086", "", False),
        ("https://gateway.example.com", "", False),
        ("", "", False),
        ("http://infra:notaport", "", False),
        ("http://infra:8086", "https://infra.example.com", True),
        ("https://infra.example.com", "https://infra.example.com", True),
        ("HTTPS://Infra.Example.com/", "https://infra.example.com", True),
        ("https://example.com/infra/", "https://example.com/infra", True),
        ("https://example.com", "https://example.com/infra", False),
        ("https://gateway.example.com", "https://infra.example.com", False),
        ("https://infra.example.com", "https://infra.example.com:notaport", False),
    ],
)
def test_is_own_gateway_url_matches_only_this_infra(deps: dict[str, Any], url: str, infra_url: str, expected: bool):
    deps["config"].infra_url = infra_url

    result = _is_own_gateway_url(url, deps["config"])

    assert result is expected


@pytest.mark.parametrize(
    ("spec", "expected_url", "expected_key"),
    [
        ({}, DOC_CHUNKER_DEFAULT_GATEWAY_URL, "infra-key"),
        ({"infra_api_url": "", "infra_api_key": ""}, DOC_CHUNKER_DEFAULT_GATEWAY_URL, "infra-key"),
        ({"infra_api_url": None, "infra_api_key": None}, DOC_CHUNKER_DEFAULT_GATEWAY_URL, "infra-key"),
        ({"infra_api_url": "   "}, DOC_CHUNKER_DEFAULT_GATEWAY_URL, "infra-key"),
        ({"infra_api_url": "http://infra:8086/", "infra_api_key": ""}, "http://infra:8086/", "infra-key"),
        ({"infra_api_url": "http://infra:8086", "infra_api_key": "custom-key"}, "http://infra:8086", "custom-key"),
        ({"infra_api_url": "https://gateway.example.com"}, "https://gateway.example.com", ""),
        ({"infra_api_url": "https://gateway.example.com", "infra_api_key": ""}, "https://gateway.example.com", ""),
        ({"infra_api_url": "https://gateway.example.com", "infra_api_key": "gateway-key"}, "https://gateway.example.com", "gateway-key"),
    ],
)
def test_create_doc_chunker_model_resolves_gateway_address_and_key(
    svc: CustomService, spec: dict[str, Any], expected_url: str, expected_key: str
):
    model = create_doc_chunker_model(svc, "172.20.0.0/16")

    docker_opts = model.options({"hardware": "CPU", **spec})  # pyright: ignore[reportCallIssue]

    assert docker_opts.env_vars is not None
    assert docker_opts.env_vars["INFRA_API_URL"] == expected_url
    assert docker_opts.env_vars["INFRA_API_KEY"] == expected_key


def test_create_doc_chunker_model_uses_infra_key_for_public_infra_url(svc: CustomService, deps: dict[str, Any]):
    deps["config"].infra_url = "https://infra.example.com"
    model = create_doc_chunker_model(svc, "172.20.0.0/16")

    docker_opts = model.options({"hardware": "CPU", "infra_api_url": "HTTPS://Infra.Example.com/"})  # pyright: ignore[reportCallIssue]

    assert docker_opts.env_vars is not None
    assert docker_opts.env_vars["INFRA_API_KEY"] == "infra-key"


def test_create_doc_chunker_model_reads_rotated_infra_key(svc: CustomService, deps: dict[str, Any]):
    model = create_doc_chunker_model(svc, "172.20.0.0/16")
    model.options({"hardware": "CPU"})  # pyright: ignore[reportCallIssue]
    deps["config"].infra_api_key = SecretStr("rotated-key")

    docker_opts = model.options({"hardware": "CPU"})  # pyright: ignore[reportCallIssue]

    assert docker_opts.env_vars is not None
    assert docker_opts.env_vars["INFRA_API_KEY"] == "rotated-key"


def test_create_doc_chunker_model_leaves_install_fields_untouched(svc: CustomService):
    model = create_doc_chunker_model(svc, "172.20.0.0/16")
    model_fields: dict[str, Any] = {"hardware": "CPU"}

    model.options(model_fields)  # pyright: ignore[reportCallIssue]

    assert model_fields == {"hardware": "CPU"}


def test_doc_chunker_gateway_fields_default_to_this_infra_without_a_key(svc: CustomService):
    fields = {f.name: f for f in svc.models["default"]["doc_chunker"].model_spec.fields}

    url_field = fields["infra_api_url"]
    key_field = fields["infra_api_key"]

    assert url_field.default == DOC_CHUNKER_DEFAULT_GATEWAY_URL
    assert key_field.default == ""
    assert key_field.type == "password"
    assert "infra-key" not in svc.models["default"]["doc_chunker"].model_spec.model_dump_json()


def _set_installed_doc_chunker(svc: CustomService, spec: dict[str, Any]) -> None:
    installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    installed.models["doc_chunker"] = ModelInstalledInfo(
        id="doc_chunker",
        options=InstallModelIn(spec=spec),
        docker_options=MagicMock(),
        container_host="",
        container_port=0,
        docker_exposed_port=0,
        registration_id="reg-1",
        prefix="doc_chunker",
        base_url="",
    )
    svc.instances_info["default"].installed = installed


@pytest.mark.asyncio
async def test_get_duplicate_spec_blanks_infra_key_of_not_installed_doc_chunker(svc: CustomService):
    spec = await svc.get_duplicate_spec("default", "doc_chunker")

    assert spec["envs"]["INFRA_API_URL"] == DOC_CHUNKER_DEFAULT_GATEWAY_URL
    assert spec["envs"]["INFRA_API_KEY"] == ""
    assert "infra-key" not in spec["envs"].values()


@pytest.mark.asyncio
async def test_get_duplicate_spec_blanks_infra_key_of_installed_doc_chunker(svc: CustomService):
    _set_installed_doc_chunker(svc, {"hardware": "CPU", "infra_api_url": "http://infra:8086", "infra_api_key": ""})

    spec = await svc.get_duplicate_spec("default", "doc_chunker")

    assert spec["envs"]["INFRA_API_KEY"] == ""


@pytest.mark.asyncio
async def test_get_duplicate_spec_keeps_admin_provided_doc_chunker_key(svc: CustomService):
    _set_installed_doc_chunker(svc, {"hardware": "CPU", "infra_api_url": "http://infra:8086", "infra_api_key": "custom-key"})

    spec = await svc.get_duplicate_spec("default", "doc_chunker")

    assert spec["envs"]["INFRA_API_KEY"] == "custom-key"


@pytest.mark.asyncio
async def test_get_duplicate_spec_with_empty_infra_key_keeps_envs_as_is(svc: CustomService, deps: dict[str, Any]):
    deps["config"].infra_api_key = SecretStr("")

    spec = await svc.get_duplicate_spec("default", "doc_chunker")

    assert spec["envs"]["INFRA_API_KEY"] == ""
    assert spec["envs"]["MAX_CONCURRENT"] == "2"
    assert spec["envs"]["INFRA_API_URL"] == DOC_CHUNKER_DEFAULT_GATEWAY_URL


@pytest.mark.asyncio
async def test_install_model_doc_chunker_keeps_infra_key_out_of_persisted_config(svc: CustomService, deps: dict[str, Any]):
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    _mock_successful_install(deps)
    svc.service_provider.save_service_config = AsyncMock()
    svc.service_provider.dismiss_warnings_matching = AsyncMock()

    with patch.object(svc, "validate_docker_image_version_for_model", new_callable=AsyncMock):
        promise = await svc.install_model("default", "doc_chunker", InstallModelIn(spec={"hardware": "CPU"}))
    await promise.wait()
    instance_config = svc._generate_instance_config(  # pyright: ignore[reportPrivateUsage]
        "default", svc.instances_info["default"].installed, None
    )

    started_options: DockerOptions = deps["docker_service"].install_and_run_docker.call_args.args[0]
    assert started_options.env_vars is not None
    assert started_options.env_vars["INFRA_API_KEY"] == "infra-key"
    persisted_spec = svc.instances_info["default"].installed.models["doc_chunker"].options.spec  # pyright: ignore[reportOptionalMemberAccess]
    assert persisted_spec is not None
    assert persisted_spec["hardware"] == "CPU"
    assert persisted_spec["prefix"] == "doc_chunker"
    assert persisted_spec["infra_api_url"] == DOC_CHUNKER_DEFAULT_GATEWAY_URL
    assert persisted_spec["infra_api_key"] == ""
    assert "infra-key" not in instance_config.model_dump_json()


def test_loaded_doc_chunker_model_has_docker_tags_field(svc: CustomService) -> None:
    model = svc.models["default"]["doc_chunker"]

    field = next(f for f in model.model_spec.fields if f.name == "image_version")
    assert field.type == "docker-tags"
    assert field.depends_on == "hardware"
    assert field.docker_image == "hub.simplito.com/deepfellow/doc-chunker-cpu"


def test_get_docker_image_repo_for_model_doc_chunker_cpu(svc: CustomService) -> None:
    assert svc.get_docker_image_repo_for_model("doc_chunker", "CPU") == "hub.simplito.com/deepfellow/doc-chunker-cpu"


def test_get_docker_image_repo_for_model_doc_chunker_gpu(deps: dict[str, Any]) -> None:
    gpu = NvidiaGpuInfo(name="RTX 4090", vram="24GB", id=0)
    deps["hardware"] = MagicMock(gpus=[gpu], cpu=MagicMock())
    svc_gpu = CustomService(**deps)

    assert svc_gpu.get_docker_image_repo_for_model("doc_chunker", "GPU") == "hub.simplito.com/deepfellow/doc-chunker-gpu"


def test_get_docker_image_repo_for_model_unknown_model_returns_none(svc: CustomService) -> None:
    assert svc.get_docker_image_repo_for_model("unknown-model", "CPU") is None
    assert svc.get_docker_image_repo_for_model(None, "CPU") is None


def test_get_default_docker_tag_for_model_doc_chunker(svc: CustomService) -> None:
    assert svc.get_default_docker_tag_for_model("doc_chunker", "CPU") == "v2.4.0"


def test_get_default_docker_tag_for_model_unknown_model_returns_none(svc: CustomService) -> None:
    assert svc.get_default_docker_tag_for_model("unknown-model", "CPU") is None
    assert svc.get_default_docker_tag_for_model(None, "CPU") is None


@pytest.mark.asyncio
async def test_get_docker_tags_for_model_doc_chunker_fetches_from_registry(svc: CustomService) -> None:
    client = MagicMock()
    client.get_tags = AsyncMock(return_value=["v1.0.5", "v1.0.4", "v1.0.3"])
    with patch("server.services.custom_service.registry_for", return_value=client) as mock_registry_for:
        tags = await svc.get_docker_tags_for_model("doc_chunker", "CPU")

    mock_registry_for.assert_called_once_with("hub.simplito.com/deepfellow/doc-chunker-cpu")
    client.get_tags.assert_called_once_with("deepfellow/doc-chunker-cpu")
    assert tags == ["v1.0.5", "v1.0.4", "v1.0.3"]


@pytest.mark.asyncio
async def test_get_docker_tags_for_model_unknown_model_returns_empty(svc: CustomService) -> None:
    assert await svc.get_docker_tags_for_model("unknown-model", "CPU") == []


def _broken_model() -> SrvCustomModel:
    """A model whose options() always raises, to exercise the probing failure path."""

    def options(_spec: dict[str, Any]) -> DockerOptions:
        raise RuntimeError("boom")

    return SrvCustomModel(
        model_props=ModelProps(private=True, type="custom", endpoints=[]),
        model_spec=ModelSpecification(fields=[]),
        model_type="custom",
        default_prefix="broken",
        size="1GB",
        options=options,
    )


def _empty_image_model() -> SrvCustomModel:
    """A model whose options() resolves cleanly but with an empty image (no exception involved)."""

    def options(_spec: dict[str, Any]) -> DockerOptions:
        return DockerOptions(image_port=8000, name="empty", container_name="empty", image="", subnet=None)

    return SrvCustomModel(
        model_props=ModelProps(private=True, type="custom", endpoints=[]),
        model_spec=ModelSpecification(fields=[]),
        model_type="custom",
        default_prefix="empty",
        size="1GB",
        options=options,
    )


def test_get_default_docker_tag_for_model_returns_none_when_image_is_empty(svc: CustomService) -> None:
    svc.models["default"]["empty"] = _empty_image_model()

    assert svc.get_default_docker_tag_for_model("empty", "CPU") is None


@pytest.mark.asyncio
async def test_get_docker_tags_for_model_returns_empty_when_repo_is_empty(svc: CustomService) -> None:
    svc.models["default"]["empty"] = _empty_image_model()

    assert await svc.get_docker_tags_for_model("empty", "CPU") == []


def test_filter_tags_for_hardware_returns_unfiltered_when_an_image_is_empty(svc: CustomService) -> None:
    svc.models["default"]["empty"] = _empty_image_model()

    tags = svc._filter_tags_for_hardware(svc.models["default"]["empty"], ["v1", "v2"], "CPU")  # pyright: ignore[reportPrivateUsage]

    assert tags == ["v1", "v2"]


def test_get_docker_image_repo_for_model_raises_registry_unavailable_when_options_raises(svc: CustomService) -> None:
    svc.models["default"]["broken"] = _broken_model()

    with pytest.raises(RegistryUnavailableError):
        svc.get_docker_image_repo_for_model("broken", "CPU")


def test_get_docker_image_repo_for_model_logs_when_options_raises(svc: CustomService, caplog: pytest.LogCaptureFixture) -> None:
    svc.models["default"]["broken"] = _broken_model()

    with caplog.at_level(logging.ERROR, logger="uvicorn"), pytest.raises(RegistryUnavailableError):
        svc.get_docker_image_repo_for_model("broken", "CPU")

    assert any("Failed to resolve default image" in record.message and "broken" in record.message for record in caplog.records)


def test_get_default_docker_tag_for_model_raises_registry_unavailable_when_options_raises(svc: CustomService) -> None:
    svc.models["default"]["broken"] = _broken_model()

    with pytest.raises(RegistryUnavailableError):
        svc.get_default_docker_tag_for_model("broken", "CPU")


@pytest.mark.asyncio
async def test_get_docker_tags_for_model_raises_registry_unavailable_when_options_raises(svc: CustomService) -> None:
    svc.models["default"]["broken"] = _broken_model()

    with pytest.raises(RegistryUnavailableError):
        await svc.get_docker_tags_for_model("broken", "CPU")


@pytest.mark.asyncio
async def test_validate_docker_image_version_for_model_fails_open_when_options_raises(svc: CustomService) -> None:
    """A regression in the model's own DockerOptions builder must fail open (skip validation),
    not surface as HTTPException(400) claiming the tag itself is invalid."""
    svc.models["default"]["broken"] = _broken_model()

    await svc.validate_docker_image_version_for_model("broken", "v9.9", "CPU")


def test_attach_docker_tags_field_skips_when_default_image_unavailable(svc: CustomService) -> None:
    model = _broken_model()

    svc._attach_docker_tags_field(model)  # pyright: ignore[reportPrivateUsage]

    assert not any(f.type == "docker-tags" for f in model.model_spec.fields)


def test_attach_docker_tags_field_skips_when_image_is_empty(svc: CustomService) -> None:
    model = _empty_image_model()

    svc._attach_docker_tags_field(model)  # pyright: ignore[reportPrivateUsage]

    assert not any(f.type == "docker-tags" for f in model.model_spec.fields)


def test_attach_docker_tags_field_skips_when_image_has_no_tag(svc: CustomService) -> None:
    model = SrvCustomModel(
        model_props=ModelProps(private=True, type="custom", endpoints=[]),
        model_spec=ModelSpecification(fields=[]),
        model_type="custom",
        default_prefix="untagged",
        size="1GB",
        options=DockerOptions(
            image_port=8000, name="untagged", container_name="untagged", image="hub.simplito.com/deepfellow/untagged", subnet=None
        ),
    )

    svc._attach_docker_tags_field(model)  # pyright: ignore[reportPrivateUsage]

    assert not any(f.type == "docker-tags" for f in model.model_spec.fields)


def test_attach_docker_tags_field_is_idempotent(svc: CustomService) -> None:
    model = svc.models["default"]["doc_chunker"]
    fields_before = len(model.model_spec.fields)

    svc._attach_docker_tags_field(model)  # pyright: ignore[reportPrivateUsage]

    assert len(model.model_spec.fields) == fields_before


@pytest.mark.asyncio
async def test_get_docker_tags_for_model_not_filtered_when_gpu_probe_fails(svc: CustomService) -> None:
    def flaky_options(spec: dict[str, Any]) -> DockerOptions:
        if spec.get("hardware") == "GPU":
            raise RuntimeError("boom")
        return DockerOptions(
            image_port=1234,
            name="flaky",
            container_name="flaky",
            image="hub.simplito.com/deepfellow/flaky:1.0",
            subnet=None,
        )

    svc.models["default"]["flaky"] = SrvCustomModel(
        model_props=ModelProps(private=True, type="custom", endpoints=[]),
        model_spec=ModelSpecification(fields=[]),
        model_type="custom",
        default_prefix="flaky",
        size="1GB",
        options=flaky_options,
    )
    client = MagicMock()
    client.get_tags = AsyncMock(return_value=["1.0", "1.1"])
    with patch("server.services.custom_service.registry_for", return_value=client):
        tags = await svc.get_docker_tags_for_model("flaky", "CPU")

    assert tags == ["1.0", "1.1"]


@pytest.mark.asyncio
async def test_get_docker_tags_for_model_doc_chunker_not_filtered_when_repo_differs_by_hardware(deps: dict[str, Any]) -> None:
    gpu = NvidiaGpuInfo(name="RTX 4090", vram="24GB", id=0)
    deps["hardware"] = MagicMock(gpus=[gpu], cpu=MagicMock())
    svc_gpu = CustomService(**deps)
    client = MagicMock()
    client.get_tags = AsyncMock(return_value=["v1.0.3", "v1.0.4", "v1.0.5"])
    with patch("server.services.custom_service.registry_for", return_value=client):
        tags = await svc_gpu.get_docker_tags_for_model("doc_chunker", "CPU")

    assert tags == ["v1.0.3", "v1.0.4", "v1.0.5"]


def test_loaded_bge_m3_model_has_docker_tags_field(svc: CustomService) -> None:
    model = svc.models["default"]["deepfellow-bge-m3"]

    field = next(f for f in model.model_spec.fields if f.name == "image_version")
    assert field.type == "docker-tags"
    assert field.docker_image == "hub.simplito.com/deepfellow/deepfellow-bge-m3"


def test_loaded_lemmatizer_model_has_docker_tags_field(svc: CustomService) -> None:
    model = svc.models["default"]["lemmatizer"]

    field = next(f for f in model.model_spec.fields if f.name == "image_version")
    assert field.type == "docker-tags"
    assert field.docker_image == "hub.simplito.com/deepfellow/deepfellow-lemmatizer"


def test_get_docker_image_repo_for_model_bge_m3_same_repo_regardless_of_hardware(svc: CustomService) -> None:
    assert (
        svc.get_docker_image_repo_for_model("deepfellow-bge-m3", "CPU")
        == svc.get_docker_image_repo_for_model("deepfellow-bge-m3", "GPU")
        == "hub.simplito.com/deepfellow/deepfellow-bge-m3"
    )


def test_get_default_docker_tag_for_model_bge_m3_varies_by_hardware(deps: dict[str, Any]) -> None:
    gpu = NvidiaGpuInfo(name="RTX 4090", vram="24GB", id=0)
    deps["hardware"] = MagicMock(gpus=[gpu], cpu=MagicMock())
    svc_gpu = CustomService(**deps)

    assert svc_gpu.get_default_docker_tag_for_model("deepfellow-bge-m3", "CPU") == "1.0.0-cpu"
    assert svc_gpu.get_default_docker_tag_for_model("deepfellow-bge-m3", "GPU") == "1.0.0-cuda-12.8"


@pytest.mark.asyncio
async def test_get_docker_tags_for_model_lemmatizer_filters_by_hardware(deps: dict[str, Any]) -> None:
    # A GPU must actually be present for the model to know its image differs by hardware at all —
    # on a CPU-only host, requesting "GPU" resolves to the same (CPU) image, same as production.
    gpu = NvidiaGpuInfo(name="RTX 4090", vram="24GB", id=0)
    deps["hardware"] = MagicMock(gpus=[gpu], cpu=MagicMock())
    svc_gpu = CustomService(**deps)
    client = MagicMock()
    client.get_tags = AsyncMock(return_value=["1.0.0-cuda-12.8", "1.0.0-cpu"])
    with patch("server.services.custom_service.registry_for", return_value=client):
        cpu_tags = await svc_gpu.get_docker_tags_for_model("lemmatizer", "CPU")

    assert cpu_tags == ["1.0.0-cpu"]


@pytest.mark.asyncio
async def test_get_docker_tags_for_model_no_filtering_when_no_gpu_present(svc: CustomService) -> None:
    # Without a GPU present, the CPU-vs-GPU probe can't detect a hardware-dependent tag pattern
    # (identical to how the plain hardcoded image logic behaves without a real GPU) — tags pass
    # through unfiltered rather than being incorrectly narrowed.
    client = MagicMock()
    client.get_tags = AsyncMock(return_value=["1.0.0-cuda-12.8", "1.0.0-cpu"])
    with patch("server.services.custom_service.registry_for", return_value=client):
        tags = await svc.get_docker_tags_for_model("lemmatizer", "CPU")

    assert tags == ["1.0.0-cuda-12.8", "1.0.0-cpu"]


@pytest.mark.asyncio
async def test_get_docker_tags_for_model_bge_m3_filters_by_hardware_gpu(deps: dict[str, Any]) -> None:
    gpu = NvidiaGpuInfo(name="RTX 4090", vram="24GB", id=0)
    deps["hardware"] = MagicMock(gpus=[gpu], cpu=MagicMock())
    svc_gpu = CustomService(**deps)
    client = MagicMock()
    client.get_tags = AsyncMock(return_value=["1.0.0-cuda-12.8", "1.0.0-cpu"])
    with patch("server.services.custom_service.registry_for", return_value=client):
        gpu_tags = await svc_gpu.get_docker_tags_for_model("deepfellow-bge-m3", "GPU")

    assert gpu_tags == ["1.0.0-cuda-12.8"]


@pytest.mark.asyncio
async def test_get_docker_tags_for_model_bge_m3_filters_by_token_not_substring(deps: dict[str, Any]) -> None:
    # A CPU tag whose version numerically contains a GPU-only token (e.g. "12.8") must still match
    # on the "cpu" token, not get dropped by a substring check against the unwanted "12.8" token.
    gpu = NvidiaGpuInfo(name="RTX 4090", vram="24GB", id=0)
    deps["hardware"] = MagicMock(gpus=[gpu], cpu=MagicMock())
    svc_gpu = CustomService(**deps)
    client = MagicMock()
    client.get_tags = AsyncMock(return_value=["1.0.0-cuda-12.8", "1.12.8-cpu"])
    with patch("server.services.custom_service.registry_for", return_value=client):
        cpu_tags = await svc_gpu.get_docker_tags_for_model("deepfellow-bge-m3", "CPU")

    assert cpu_tags == ["1.12.8-cpu"]


@pytest.mark.asyncio
async def test_install_instance_loads_models_for_new_instance(svc: CustomService) -> None:
    # Remove models for "default" to trigger the load branch
    del svc.models["default"]
    options = InstallServiceIn(spec={})
    promise = await svc._install_instance("default", options)  # pyright: ignore[reportPrivateUsage]
    result = await promise.wait()
    assert isinstance(result, InstalledInfo)
    assert "default" in svc.models


@pytest.mark.asyncio
async def test_uninstall_instance_with_purge_deletes_non_default_instance(svc: CustomService) -> None:
    svc.instances_info["gpu-1"] = Instance(None, None, {}, InstanceConfig())
    svc.instances_info["gpu-1"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    svc._clear_working_dir = AsyncMock()  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage]

    await svc._uninstall_instance("gpu-1", UninstallServiceIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert "gpu-1" not in svc.instances_info


@pytest.mark.asyncio
async def test_uninstall_instance_purge_with_other_instance_installed_does_not_clear_working_dir(
    svc: CustomService,
) -> None:
    # Branch 267->272: another instance is still installed, so the shared working dir
    # cleanup must be skipped even though this instance is being purged.
    svc.instances_info["gpu-1"] = Instance(None, None, {}, InstanceConfig())
    svc.instances_info["gpu-1"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    svc.service_downloaded = True
    svc._clear_working_dir = AsyncMock()  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage]

    await svc._uninstall_instance("default", UninstallServiceIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert svc._clear_working_dir.call_count == 0  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage]
    assert svc.service_downloaded is True
    assert svc.instances_info["default"].installed is None
    assert "gpu-1" in svc.instances_info


def test_add_custom_model_creates_instance_dict_when_missing(svc: CustomService) -> None:
    del svc.models["default"]
    model = CustomModel(id="cm-1", data=_CUSTOM_MODEL_DATA)

    svc._add_custom_model("default", model)  # pyright: ignore[reportPrivateUsage]

    assert "my-custom" in svc.models["default"]


@pytest.mark.asyncio
async def test_get_model_initialises_empty_models_for_instance(svc: CustomService) -> None:
    del svc.models["default"]
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))

    with pytest.raises(HTTPException):
        await svc.get_model("default", "lemmatizer")

    assert "default" in svc.models


@pytest.mark.asyncio
async def test_list_models_skips_instance_not_in_requested(svc: CustomService) -> None:
    svc.instances_info["gpu-1"] = Instance(None, None, {}, InstanceConfig())
    svc.load_default_models("gpu-1")
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    svc.instances_info["gpu-1"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))

    result_default = await svc.list_models("default", ListModelsFilters(installed=None))
    result_gpu = await svc.list_models("gpu-1", ListModelsFilters(installed=None))

    # Each instance returns only its own models
    assert len(result_default.list) > 0
    assert len(result_gpu.list) > 0
    assert len(result_default.list) == len(result_gpu.list)


@pytest.mark.asyncio
async def test_install_model_uses_default_prefix_when_no_spec(svc: CustomService, deps: dict[str, Any]) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    deps["docker_service"].get_image_warnings = AsyncMock(return_value=[])
    deps["docker_service"].is_docker_image_pulled = AsyncMock(return_value=True)
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=(8090, True, False))
    deps["docker_service"].get_container_host.return_value = "172.20.0.1"
    deps["docker_service"].get_container_port.return_value = 8090
    deps["endpoint_registry"].register_custom_endpoint_as_proxy.return_value = "reg-1"

    # No spec at all → should use default_prefix "lemmatizer"
    promise = await svc._install_model("default", "lemmatizer", InstallModelIn())  # pyright: ignore[reportPrivateUsage]

    result = await promise.wait()
    assert result.status == "OK"


@pytest.mark.asyncio
async def test_install_model_uses_default_prefix_when_prefix_missing(svc: CustomService, deps: dict[str, Any]) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    deps["docker_service"].get_image_warnings = AsyncMock(return_value=[])
    deps["docker_service"].is_docker_image_pulled = AsyncMock(return_value=True)
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=(8090, True, False))
    deps["docker_service"].get_container_host.return_value = "172.20.0.1"
    deps["docker_service"].get_container_port.return_value = 8090
    deps["endpoint_registry"].register_custom_endpoint_as_proxy.return_value = "reg-1"

    # spec without "prefix" → should set default_prefix
    promise = await svc._install_model(  # pyright: ignore[reportPrivateUsage]
        "default", "lemmatizer", InstallModelIn(spec={"hardware": "CPU"})
    )

    result = await promise.wait()
    assert result.status == "OK"


@pytest.mark.asyncio
async def test_install_model_initialises_models_for_instance(svc: CustomService, deps: dict[str, Any]) -> None:
    del svc.models["default"]
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    deps["docker_service"].get_image_warnings = AsyncMock(return_value=[])

    with pytest.raises(HTTPException):
        await svc._install_model("default", "ghost", InstallModelIn())  # pyright: ignore[reportPrivateUsage]

    assert "default" in svc.models


@pytest.mark.asyncio
async def test_uninstall_instance_when_not_installed_is_noop(svc: CustomService) -> None:
    # Branch 239->244: installed is None, skip the if block entirely
    assert svc.instances_info["default"].installed is None

    await svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert svc.instances_info["default"].installed is None


@pytest.mark.asyncio
async def test_uninstall_instance_skips_model_shared_with_other_instance(svc: CustomService) -> None:
    # Branch 241->240: is_model_installed_in_other_instance returns True,
    # so the inner if condition is False and the for loop continues without calling _uninstall_model
    mock_model = MagicMock()
    mock_model.id = "lemmatizer"
    mock_model.prefix = "lemmatizer"
    mock_model.registration_id = "reg-1"
    mock_model.docker_options = MagicMock()

    shared_installed = InstalledInfo(models={"lemmatizer": mock_model}, options=InstallServiceIn(spec={}))
    svc.instances_info["default"].installed = shared_installed

    svc.instances_info["gpu-1"] = Instance(None, None, {}, InstanceConfig())
    svc.instances_info["gpu-1"].installed = InstalledInfo(
        models={"lemmatizer": mock_model},
        options=InstallServiceIn(spec={}),
    )

    svc._uninstall_model = AsyncMock()  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage]

    await svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert svc._uninstall_model.call_count == 0  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage]


@pytest.mark.asyncio
async def test_uninstall_model_when_not_in_installed_is_noop(svc: CustomService, deps: dict[str, Any]) -> None:
    # Branch 430->436: model_id not in info.models, skip the if block and go to purge check
    deps["docker_service"].uninstall_docker = AsyncMock()
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))

    await svc._uninstall_model("default", "lemmatizer", UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert deps["docker_service"].uninstall_docker.call_count == 0
    assert deps["endpoint_registry"].unregister_custom_endpoint.call_count == 0


@pytest.mark.asyncio
async def test_resolve_custom_model_size_returns_formatted_size(svc: CustomService, deps: dict[str, Any]) -> None:
    deps["docker_service"].get_docker_image_size = AsyncMock(return_value=1024**2)

    result = await svc._resolve_custom_model_size({"image": "myimage:latest"})  # pyright: ignore[reportPrivateUsage]

    assert result == "1.0 MB"


@pytest.mark.asyncio
async def test_resolve_custom_model_size_returns_none_when_image_size_is_none(svc: CustomService, deps: dict[str, Any]) -> None:
    deps["docker_service"].get_docker_image_size = AsyncMock(return_value=None)

    result = await svc._resolve_custom_model_size({"image": "myimage:latest"})  # pyright: ignore[reportPrivateUsage]

    assert result is None


@pytest.mark.asyncio
async def test_resolve_custom_model_size_returns_none_on_exception(svc: CustomService, deps: dict[str, Any]) -> None:
    deps["docker_service"].get_docker_image_size = AsyncMock(side_effect=Exception("fail"))

    result = await svc._resolve_custom_model_size({"image": "myimage:latest"})  # pyright: ignore[reportPrivateUsage]

    assert result is None


@pytest.mark.asyncio
async def test_install_model_registration_failure_rolls_back_model(svc: CustomService, deps: dict[str, Any]) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    deps["docker_service"].get_image_warnings = AsyncMock(return_value=[])
    deps["docker_service"].is_docker_image_pulled = AsyncMock(return_value=True)
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=(8090, True, False))
    deps["docker_service"].get_container_host.return_value = "172.20.0.1"
    deps["docker_service"].get_container_port.return_value = 8090
    deps["endpoint_registry"].register_custom_endpoint_as_proxy.side_effect = RuntimeError("registry down")

    promise = await svc._install_model(  # pyright: ignore[reportPrivateUsage]
        "default", "lemmatizer", InstallModelIn(spec={"prefix": "lemmatizer"})
    )
    with pytest.raises(RuntimeError):
        await promise.wait()

    installed = svc.instances_info["default"].installed
    assert installed is not None
    assert "lemmatizer" not in installed.models


@pytest.mark.asyncio
async def test_install_model_registration_failure_skips_rollback_when_already_removed(svc: CustomService, deps: dict[str, Any]) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    deps["docker_service"].get_image_warnings = AsyncMock(return_value=[])
    deps["docker_service"].is_docker_image_pulled = AsyncMock(return_value=True)
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=(8090, True, False))
    deps["docker_service"].get_container_host.return_value = "172.20.0.1"
    deps["docker_service"].get_container_port.return_value = 8090

    def side_effect(*args: object, **kwargs: object) -> None:
        installed = svc.instances_info["default"].installed
        assert installed is not None
        installed.models.pop("lemmatizer", None)
        raise RuntimeError("registry down")

    deps["endpoint_registry"].register_custom_endpoint_as_proxy.side_effect = side_effect

    promise = await svc._install_model(  # pyright: ignore[reportPrivateUsage]
        "default", "lemmatizer", InstallModelIn(spec={"prefix": "lemmatizer"})
    )
    with pytest.raises(RuntimeError):
        await promise.wait()

    installed = svc.instances_info["default"].installed
    assert installed is not None
    assert "lemmatizer" not in installed.models


def _model_with_password_field(svc: CustomService) -> str:
    return next(
        model_id for model_id, model in svc.models["default"].items() if any(field.type == "password" for field in model.model_spec.fields)
    )


def _installed_model(model_id: str, spec: dict[str, Any]) -> ModelInstalledInfo:
    return ModelInstalledInfo(
        id=model_id,
        options=InstallModelIn(spec=spec),
        docker_options=MagicMock(),
        container_host="127.0.0.1",
        container_port=8080,
        docker_exposed_port=8080,
        registration_id="reg-1",
        prefix="p",
        base_url="http://127.0.0.1:8080",
    )


def test_get_secret_values_includes_installed_model_password_option(svc: CustomService) -> None:
    model_id = _model_with_password_field(svc)
    model = svc.models["default"][model_id]
    password_field = next(field.name for field in model.model_spec.fields if field.type == "password")
    installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    installed.models[model_id] = _installed_model(model_id, {password_field: "gateway-key", "prefix": "p"})
    svc.instances_info["default"].installed = installed

    assert svc.get_secret_values() == ["gateway-key"]


def test_get_secret_values_skips_model_missing_from_catalog_and_options_without_spec(svc: CustomService) -> None:
    installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    installed.models["retired"] = _installed_model("retired", {"infra_api_key": "orphan-key"})
    model_id = _model_with_password_field(svc)
    no_spec = _installed_model(model_id, {})
    no_spec.options = InstallModelIn(spec=None)
    installed.models[model_id] = no_spec
    svc.instances_info["default"].installed = installed

    assert svc.get_secret_values() == []


def test_get_secret_values_empty_when_not_installed_custom(svc: CustomService) -> None:
    svc.instances_info["default"].installed = None

    assert svc.get_secret_values() == []
