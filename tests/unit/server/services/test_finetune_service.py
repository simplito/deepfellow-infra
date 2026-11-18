# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

import uuid
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call, patch

import aiohttp
import pytest
from fastapi import HTTPException
from pydantic import SecretStr

from server.models.models import InstallModelIn, ListModelsFilters, UninstallModelIn
from server.models.services import InstallServiceIn, UninstallServiceIn
from server.services.base2_service import Instance, InstanceConfig
from server.services.finetune_service import FinetuneOptions, FinetuneService, InstalledInfo


@pytest.fixture
def deps() -> dict[str, Any]:
    docker_svc = MagicMock()
    docker_svc.get_docker_subnet.return_value = "172.20.0.0/16"
    docker_svc.get_docker_container_name.side_effect = lambda name: f"df-{name}"  # pyright: ignore[reportUnknownLambdaType]
    docker_svc.get_user_for_docker = AsyncMock(return_value=None)
    config = MagicMock()
    config.standard_proxy_timeout_seconds = 300
    config.adapter_registry_secret = SecretStr("test-secret")
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
def svc(deps: dict[str, Any]) -> FinetuneService:
    return FinetuneService(**deps)


def _make_installed_info(base_url: str = "http://172.20.0.1:8333") -> InstalledInfo:
    docker = MagicMock()
    docker.container_name = "df-finetune"
    return InstalledInfo(
        docker=docker,
        options=InstallServiceIn(spec={}),
        parsed_options=FinetuneOptions(),
        container_host="172.20.0.1",
        container_port=8333,
        docker_exposed_port=8333,
        registration_id="reg-1",
        base_url=base_url,
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


@pytest.mark.asyncio
async def test_load_service_strips_any_persisted_model_entry(svc: FinetuneService) -> None:
    """A `services.json` written before the single-step install refactor may still list a
    per-model entry for deepfellow-finetune; it must not resurface as a "failed to load" warning."""
    raw_config = {
        "instances": {
            "default": {
                "options": {"spec": {"hardware": "GPU"}},
                "models": [{"model_id": "deepfellow-finetune", "options": {"spec": {}}}],
            }
        }
    }

    with patch("server.services.base2_service.Base2Service.load_service", new=AsyncMock()) as mock_super_load:
        await svc.load_service(raw_config)

    loaded_config = mock_super_load.call_args.args[0]
    assert loaded_config["instances"]["default"]["models"] == []


@pytest.mark.asyncio
async def test_load_service_without_instances_key_still_calls_super(svc: FinetuneService) -> None:
    raw_config: dict[str, Any] = {}

    with patch("server.services.base2_service.Base2Service.load_service", new=AsyncMock()) as mock_super_load:
        await svc.load_service(raw_config)

    assert mock_super_load.call_count == 1
    assert mock_super_load.call_args == call(raw_config)


@pytest.mark.asyncio
async def test_load_service_skips_non_dict_instance_entries(svc: FinetuneService) -> None:
    raw_config = {"instances": {"default": "not-a-dict"}}

    with patch("server.services.base2_service.Base2Service.load_service", new=AsyncMock()) as mock_super_load:
        await svc.load_service(raw_config)

    loaded_config = mock_super_load.call_args.args[0]
    assert loaded_config["instances"]["default"] == "not-a-dict"


def test_get_type(svc: FinetuneService) -> None:
    assert svc.get_type() == "finetune"


def test_get_description_not_empty(svc: FinetuneService) -> None:
    assert svc.get_description()


def test_get_size(svc: FinetuneService) -> None:
    assert svc.get_size() == "8.56GB"


def test_get_spec_contains_hardware_and_prefix_fields(svc: FinetuneService) -> None:
    field_names = [f.name for f in svc.get_spec().fields]
    assert "hardware" in field_names
    assert "prefix" in field_names
    assert "proxy_timeout_seconds" in field_names


def test_get_custom_model_spec_is_none(svc: FinetuneService) -> None:
    assert svc.get_custom_model_spec() is None


def test_get_installed_base_url_when_not_installed_returns_none(svc: FinetuneService) -> None:
    assert svc.get_installed_base_url() is None


def test_get_installed_base_url_when_installed_returns_url(svc: FinetuneService) -> None:
    svc.instances_info["default"].installed = _make_installed_info("http://172.20.0.1:8333")

    assert svc.get_installed_base_url() == "http://172.20.0.1:8333"


def test_get_installed_base_url_prefers_default_instance(svc: FinetuneService) -> None:
    svc.instances_info["second"] = Instance(_make_installed_info("http://172.20.0.2:8333"), None, {}, InstanceConfig())
    svc.instances_info["default"].installed = _make_installed_info("http://172.20.0.1:8333")

    assert svc.get_installed_base_url() == "http://172.20.0.1:8333"


def test_get_installed_base_url_falls_back_to_non_default_instance(svc: FinetuneService) -> None:
    """The adapters directory is shared, so a vLLM adapter install must work with only a non-default instance."""
    svc.instances_info["second"] = Instance(_make_installed_info("http://172.20.0.2:8333"), None, {}, InstanceConfig())

    assert svc.get_installed_base_url() == "http://172.20.0.2:8333"


def test_get_installed_base_url_for_explicit_instance_does_not_fall_back(svc: FinetuneService) -> None:
    svc.instances_info["second"] = Instance(_make_installed_info("http://172.20.0.2:8333"), None, {}, InstanceConfig())

    assert svc.get_installed_base_url("default") is None


@pytest.mark.asyncio
async def test_list_adapters_raises_409_when_not_installed(svc: FinetuneService) -> None:
    with pytest.raises(HTTPException) as exc:
        await svc.list_adapters()

    assert exc.value.status_code == 409


@pytest.mark.asyncio
async def test_list_adapters_returns_parsed_response(svc: FinetuneService) -> None:
    svc.instances_info["default"].installed = _make_installed_info("http://172.20.0.1:8333")
    payload = {
        "object": "list",
        "data": [
            {
                "id": "pirate-0c571d9a",
                "object": "model",
                "created": 1777548607,
                "owned_by": "deepfellow",
                "base_model": "meta-llama/llama-3.1-8b-instruct",
                "formats": ["safetensors", "gguf"],
                "lora": {"rank": 16, "alpha": 32},
                "gguf": {"format": "f16", "file": "adapter.gguf"},
            }
        ],
    }
    mock_session_cls = _mock_adapter_registry_session_cls(200, payload)

    with patch("server.services.finetune_service.aiohttp.ClientSession", mock_session_cls):
        result = await svc.list_adapters()

    assert len(result.data) == 1
    assert result.data[0].id == "pirate-0c571d9a"
    assert result.data[0].base_model == "meta-llama/llama-3.1-8b-instruct"
    mock_session_cls.return_value.get.assert_called_once_with("http://172.20.0.1:8333/v1/adapters")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [aiohttp.ClientConnectionError("connection refused"), TimeoutError()],
    ids=["unreachable", "timeout"],
)
async def test_list_adapters_raises_503_when_container_unavailable(svc: FinetuneService, error: Exception) -> None:
    svc.instances_info["default"].installed = _make_installed_info("http://172.20.0.1:8333")
    mock_session_cls = MagicMock()
    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.get = MagicMock(side_effect=error)
    mock_session_cls.return_value = mock_session

    with (
        patch("server.services.finetune_service.aiohttp.ClientSession", mock_session_cls),
        pytest.raises(HTTPException) as exc,
    ):
        await svc.list_adapters()

    assert exc.value.status_code == 503


@pytest.mark.asyncio
async def test_list_adapters_raises_502_when_container_returns_error_status(svc: FinetuneService) -> None:
    svc.instances_info["default"].installed = _make_installed_info("http://172.20.0.1:8333")
    mock_session_cls = _mock_adapter_registry_session_cls(401)
    mock_response = mock_session_cls.return_value.get.return_value
    mock_response.raise_for_status = MagicMock(
        side_effect=aiohttp.ClientResponseError(request_info=MagicMock(), history=(), status=401, message="Unauthorized")
    )

    with (
        patch("server.services.finetune_service.aiohttp.ClientSession", mock_session_cls),
        pytest.raises(HTTPException) as exc,
    ):
        await svc.list_adapters()

    assert exc.value.status_code == 502
    assert "HTTP 401" in exc.value.detail


def test_get_docker_options_returns_installed_container(svc: FinetuneService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    assert svc.get_docker_options("default", None) is installed.docker


def test_get_docker_options_rejects_model_id(svc: FinetuneService) -> None:
    svc.instances_info["default"].installed = _make_installed_info()

    with pytest.raises(HTTPException) as exc:
        svc.get_docker_options("default", "some-model")

    assert exc.value.status_code == 400


def test_service_has_docker_returns_true(svc: FinetuneService) -> None:
    assert svc.service_has_docker() is True


@pytest.mark.asyncio
async def test_stop_instance_no_installed_is_noop(svc: FinetuneService, deps: dict[str, Any]) -> None:
    deps["docker_service"].stop_docker = AsyncMock()

    await svc.stop_instance("default")

    assert deps["docker_service"].stop_docker.call_count == 0


@pytest.mark.asyncio
async def test_stop_instance_with_installed_stops_container(svc: FinetuneService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    deps["docker_service"].stop_docker = AsyncMock()

    await svc.stop_instance("default")

    assert deps["docker_service"].stop_docker.call_count == 1
    assert deps["docker_service"].stop_docker.call_args == call(installed.docker)


def test_get_installed_info_when_not_installed_returns_false(svc: FinetuneService) -> None:
    assert svc.get_installed_info("default") is False


def test_generate_instance_config_without_info(svc: FinetuneService) -> None:
    config = svc._generate_instance_config("default", None, None)  # pyright: ignore[reportPrivateUsage]

    assert config.options is None
    assert config.models == []
    assert config.registration_id is None


def test_generate_instance_config_with_info_has_no_models(svc: FinetuneService) -> None:
    installed = _make_installed_info()

    config = svc._generate_instance_config("default", installed, None)  # pyright: ignore[reportPrivateUsage]

    assert config.options == installed.options
    assert config.models == []
    assert config.container_name == "df-finetune"
    assert config.registration_id == "reg-1"


def test_load_download_info(svc: FinetuneService) -> None:
    info = svc._load_download_info({"image": "test/image:latest"})  # pyright: ignore[reportPrivateUsage]

    assert info.image == "test/image:latest"


def _mock_install_deps(deps: dict[str, Any]) -> None:
    deps["docker_service"].get_image_warnings = AsyncMock(return_value=[])
    deps["docker_service"].is_docker_image_pulled = AsyncMock(return_value=True)
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=(20000, True, False))
    deps["docker_service"].get_container_host.return_value = "172.20.0.1"
    deps["docker_service"].get_container_port.return_value = 8333
    deps["endpoint_registry"].register_custom_endpoint_as_proxy.return_value = "reg-1"


@pytest.mark.asyncio
@patch.object(FinetuneService, "_get_working_dir")
async def test_install_instance_creates_container_in_one_step(
    mock_get_working_dir: MagicMock, svc: FinetuneService, deps: dict[str, Any], tmp_path: Path
) -> None:
    """Installing the service must create and start the container directly - no separate per-model install step."""
    mock_get_working_dir.return_value = tmp_path
    _mock_install_deps(deps)

    promise = await svc._install_instance("default", InstallServiceIn(spec={"prefix": "deepfellow-finetune"}))  # pyright: ignore[reportPrivateUsage]
    result = await promise.wait()

    assert isinstance(result, InstalledInfo)
    assert svc.service_downloaded is True
    assert deps["docker_service"].install_and_run_docker.call_count == 1
    docker_options = deps["docker_service"].install_and_run_docker.call_args.args[0]
    assert docker_options.image_port == 8333
    assert docker_options.name == "finetune"
    assert docker_options.container_name == "df-finetune"
    assert docker_options.image == "hub.simplito.com/deepfellow/deepfellow-finetune:0.0.20"
    assert docker_options.volumes == [
        f"{tmp_path}/models:/workspace/models",
        f"{tmp_path}/adapters:/workspace/adapters",
        f"{tmp_path}/finetune/configs:/workspace/configs",
        f"{tmp_path}/finetune/data:/workspace/data",
        f"{tmp_path}/finetune/db:/data",
    ]
    assert docker_options.env_vars["DFFT_DB_PATH"] == "/data/worker.db"
    assert (tmp_path / "models").is_dir()
    assert (tmp_path / "adapters").is_dir()
    registered_options = deps["endpoint_registry"].register_custom_endpoint_as_proxy.call_args.kwargs["options"]
    assert registered_options.read_timeout_seconds == 300


@pytest.mark.asyncio
@patch.object(FinetuneService, "_get_working_dir")
async def test_install_instance_authenticates_worker_with_adapter_registry_secret(
    mock_get_working_dir: MagicMock, svc: FinetuneService, deps: dict[str, Any], tmp_path: Path
) -> None:
    """df-finetune >= 0.0.19 refuses to start without DFFT_TOKEN; the proxy drops the client's Authorization, so it must send the token."""
    mock_get_working_dir.return_value = tmp_path
    _mock_install_deps(deps)

    promise = await svc._install_instance("default", InstallServiceIn(spec={}))  # pyright: ignore[reportPrivateUsage]
    await promise.wait()

    docker_options = deps["docker_service"].install_and_run_docker.call_args.args[0]
    assert docker_options.env_vars["DFFT_TOKEN"] == "test-secret"
    assert "DFFT_ALLOW_NOAUTH" not in docker_options.env_vars
    registered_options = deps["endpoint_registry"].register_custom_endpoint_as_proxy.call_args.kwargs["options"]
    assert registered_options.headers == {"Authorization": "Bearer test-secret"}


@pytest.mark.asyncio
@patch.object(FinetuneService, "_get_working_dir")
async def test_install_instance_runs_worker_unauthenticated_without_adapter_registry_secret(
    mock_get_working_dir: MagicMock, svc: FinetuneService, deps: dict[str, Any], tmp_path: Path
) -> None:
    """Without a secret there's no token to give the worker, so it must be told to accept unauthenticated requests or it won't start."""
    mock_get_working_dir.return_value = tmp_path
    deps["config"].adapter_registry_secret = SecretStr("")
    _mock_install_deps(deps)

    promise = await svc._install_instance("default", InstallServiceIn(spec={}))  # pyright: ignore[reportPrivateUsage]
    await promise.wait()

    docker_options = deps["docker_service"].install_and_run_docker.call_args.args[0]
    assert docker_options.env_vars["DFFT_ALLOW_NOAUTH"] == "1"
    assert "DFFT_TOKEN" not in docker_options.env_vars
    registered_options = deps["endpoint_registry"].register_custom_endpoint_as_proxy.call_args.kwargs["options"]
    assert registered_options.headers == {}


@pytest.mark.asyncio
@patch.object(FinetuneService, "_get_working_dir")
async def test_install_instance_runs_as_infra_user_and_precreates_instance_dirs(
    mock_get_working_dir: MagicMock, svc: FinetuneService, deps: dict[str, Any], tmp_path: Path
) -> None:
    """The image runs as uid 1000; bind-mounted dirs created by the docker daemon would be root-owned and unwritable."""
    mock_get_working_dir.return_value = tmp_path
    _mock_install_deps(deps)
    deps["docker_service"].get_user_for_docker = AsyncMock(return_value="1234:1234")

    promise = await svc._install_instance("default", InstallServiceIn(spec={}))  # pyright: ignore[reportPrivateUsage]
    await promise.wait()

    docker_options = deps["docker_service"].install_and_run_docker.call_args.args[0]
    assert docker_options.user == "1234:1234"
    for subdir in ("configs", "data", "db"):
        assert (tmp_path / "finetune" / subdir).is_dir()


@pytest.mark.asyncio
@patch.object(FinetuneService, "_get_working_dir")
async def test_install_instance_uses_container_name_override_when_given(
    mock_get_working_dir: MagicMock, svc: FinetuneService, deps: dict[str, Any], tmp_path: Path
) -> None:
    mock_get_working_dir.return_value = tmp_path
    _mock_install_deps(deps)

    promise = await svc._install_instance("default", InstallServiceIn(spec={}), "pinned-name")  # pyright: ignore[reportPrivateUsage]
    await promise.wait()

    docker_options = deps["docker_service"].install_and_run_docker.call_args.args[0]
    assert docker_options.container_name == "pinned-name"


@pytest.mark.asyncio
@patch.object(FinetuneService, "_get_working_dir")
async def test_install_second_instance_gets_own_container_and_shares_models_and_adapters(
    mock_get_working_dir: MagicMock, svc: FinetuneService, deps: dict[str, Any], tmp_path: Path
) -> None:
    """A non-default instance must not collide with the default one on container name or per-instance state."""
    mock_get_working_dir.return_value = tmp_path
    _mock_install_deps(deps)

    promise = await svc._install_instance("second", InstallServiceIn(spec={"prefix": "finetune-second"}))  # pyright: ignore[reportPrivateUsage]
    await promise.wait()

    docker_options = deps["docker_service"].install_and_run_docker.call_args.args[0]
    assert docker_options.name == "finetune-second"
    assert docker_options.container_name == "df-finetune-second"
    assert docker_options.volumes == [
        f"{tmp_path}/models:/workspace/models",
        f"{tmp_path}/adapters:/workspace/adapters",
        f"{tmp_path}/finetune-second/configs:/workspace/configs",
        f"{tmp_path}/finetune-second/data:/workspace/data",
        f"{tmp_path}/finetune-second/db:/data",
    ]
    deps["docker_service"].get_container_host.assert_called_once_with("172.20.0.0/16", docker_options)


@pytest.mark.asyncio
async def test_install_instance_raises_400_when_prefix_used_by_another_instance(svc: FinetuneService, deps: dict[str, Any]) -> None:
    """Each instance has its own job database, so two instances must not share one proxy prefix."""
    svc.instances_info["default"].installed = _make_installed_info()
    _mock_install_deps(deps)

    with pytest.raises(HTTPException) as exc:
        await svc._install_instance("second", InstallServiceIn(spec={"prefix": "deepfellow-finetune"}))  # pyright: ignore[reportPrivateUsage]

    assert exc.value.status_code == 400
    assert deps["docker_service"].install_and_run_docker.call_count == 0


@pytest.mark.asyncio
@patch.object(FinetuneService, "_get_working_dir")
async def test_reinstall_instance_keeps_its_own_prefix(
    mock_get_working_dir: MagicMock, svc: FinetuneService, deps: dict[str, Any], tmp_path: Path
) -> None:
    mock_get_working_dir.return_value = tmp_path
    svc.instances_info["default"].installed = _make_installed_info()
    _mock_install_deps(deps)

    promise = await svc._install_instance("default", InstallServiceIn(spec={"prefix": "deepfellow-finetune"}))  # pyright: ignore[reportPrivateUsage]
    await promise.wait()

    assert deps["docker_service"].install_and_run_docker.call_count == 1


@pytest.mark.asyncio
async def test_uninstall_instance_tears_down_container(svc: FinetuneService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    deps["docker_service"].uninstall_docker = AsyncMock()

    await svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert svc.instances_info["default"].installed is None
    assert deps["docker_service"].uninstall_docker.call_count == 1
    assert deps["docker_service"].uninstall_docker.call_args == call(installed.docker)
    assert deps["endpoint_registry"].unregister_custom_endpoint.call_count == 1


@pytest.mark.asyncio
async def test_uninstall_instance_with_purge_removes_image(svc: FinetuneService, deps: dict[str, Any]) -> None:
    svc.instances_info["default"].installed = _make_installed_info()
    deps["docker_service"].uninstall_docker = AsyncMock()
    deps["docker_service"].remove_image = AsyncMock()
    svc._clear_working_dir = AsyncMock()  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage]

    await svc._uninstall_instance("default", UninstallServiceIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert svc.service_downloaded is False
    deps["docker_service"].remove_image.assert_called_once_with("hub.simplito.com/deepfellow/deepfellow-finetune:0.0.20")


@pytest.mark.asyncio
async def test_uninstall_instance_when_not_installed_skips_teardown(svc: FinetuneService, deps: dict[str, Any]) -> None:
    deps["docker_service"].uninstall_docker = AsyncMock()

    await svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert svc.instances_info["default"].installed is None
    assert deps["docker_service"].uninstall_docker.call_count == 0
    assert deps["endpoint_registry"].unregister_custom_endpoint.call_count == 0


@pytest.mark.asyncio
async def test_uninstall_instance_with_purge_keeps_image_when_other_instance_still_installed(
    svc: FinetuneService, deps: dict[str, Any]
) -> None:
    svc.instances_info["default"].installed = _make_installed_info()
    svc.instances_info["other"] = Instance(_make_installed_info(), None, {}, InstanceConfig())
    deps["docker_service"].uninstall_docker = AsyncMock()
    deps["docker_service"].remove_image = AsyncMock()
    svc._clear_working_dir = AsyncMock()  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage]

    await svc._uninstall_instance("default", UninstallServiceIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert deps["docker_service"].remove_image.call_count == 0
    assert svc._clear_working_dir.call_count == 0  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage]
    assert svc.instances_info["default"].installed is None


@pytest.mark.asyncio
async def test_uninstall_instance_with_purge_deletes_non_default_instance(svc: FinetuneService, deps: dict[str, Any]) -> None:
    svc.instances_info["other"] = Instance(_make_installed_info(), None, {}, InstanceConfig())
    deps["docker_service"].uninstall_docker = AsyncMock()
    deps["docker_service"].remove_image = AsyncMock()
    svc._clear_working_dir = AsyncMock()  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage]

    await svc._uninstall_instance("other", UninstallServiceIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert "other" not in svc.instances_info


@pytest.mark.asyncio
async def test_list_models_always_empty(svc: FinetuneService) -> None:
    result = await svc.list_models("default", ListModelsFilters())

    assert result.list == []


@pytest.mark.asyncio
async def test_get_model_raises(svc: FinetuneService) -> None:
    with pytest.raises(HTTPException) as exc:
        await svc.get_model("default", "deepfellow-finetune")

    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_install_model_raises(svc: FinetuneService) -> None:
    with pytest.raises(HTTPException) as exc:
        await svc._install_model("default", "deepfellow-finetune", InstallModelIn())  # pyright: ignore[reportPrivateUsage]

    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_uninstall_model_raises(svc: FinetuneService) -> None:
    with pytest.raises(HTTPException) as exc:
        await svc._uninstall_model("default", "deepfellow-finetune", UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert exc.value.status_code == 400


@pytest.fixture
def lifecycle(deps: dict[str, Any], tmp_path: Path) -> Any:
    """Mocks for running the real install/update/uninstall/load flows; registration returns the id it was given."""
    _mock_install_deps(deps)
    deps["docker_service"].uninstall_docker = AsyncMock()
    deps["endpoint_registry"].register_custom_endpoint_as_proxy.side_effect = lambda **kwargs: kwargs["registration_options"].id  # pyright: ignore[reportUnknownLambdaType, reportUnknownMemberType]
    provider = deps["service_provider"]
    provider.save_service_config = AsyncMock()
    provider.add_warning = AsyncMock()
    provider.dismiss_warnings_matching = AsyncMock()
    provider.dismiss_warnings_matching_any = AsyncMock()
    provider.dismiss_warnings_for_instance = AsyncMock()
    with patch.object(FinetuneService, "_get_working_dir", return_value=tmp_path):
        yield


def _registered_ids(deps: dict[str, Any]) -> list[str]:
    calls = deps["endpoint_registry"].register_custom_endpoint_as_proxy.call_args_list
    return [c.kwargs["registration_options"].id for c in calls]


def _persisted_instance(registration_id: str | None) -> dict[str, Any]:
    return {
        "options": {"stream": False, "ignore_warnings": False, "spec": {"prefix": "deepfellow-finetune"}},
        "models": [],
        "custom": None,
        "container_name": "df-finetune",
        "registration_id": registration_id,
    }


def _installed_registration_id(svc: FinetuneService, instance: str = "default") -> str:
    installed = svc.instances_info[instance].installed
    assert installed is not None
    return installed.registration_id


@pytest.mark.asyncio
async def test_install_instance_registers_under_pinned_id(svc: FinetuneService, deps: dict[str, Any], lifecycle: None) -> None:
    svc._pinned_registration_ids["default"] = "stable-id"  # pyright: ignore[reportPrivateUsage]

    promise = await svc._install_instance("default", InstallServiceIn(spec={}))  # pyright: ignore[reportPrivateUsage]
    result = await promise.wait()

    assert _registered_ids(deps) == ["stable-id"]
    assert result.registration_id == "stable-id"


@pytest.mark.asyncio
async def test_install_instance_without_pin_registers_under_new_id(svc: FinetuneService, deps: dict[str, Any], lifecycle: None) -> None:
    for _ in range(2):
        promise = await svc._install_instance("default", InstallServiceIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    first, second = _registered_ids(deps)
    assert uuid.UUID(first) != uuid.UUID(second)


@pytest.mark.asyncio
async def test_install_instance_with_persisted_id_reuses_it(svc: FinetuneService, deps: dict[str, Any], lifecycle: None) -> None:
    config = InstanceConfig.model_validate(_persisted_instance("stable-id"))
    assert config.options is not None

    promise = await svc.install_instance("default", config.options, config, save=False)
    await promise.wait()

    assert _registered_ids(deps) == ["stable-id"]
    assert _installed_registration_id(svc) == "stable-id"


@pytest.mark.asyncio
async def test_install_instance_without_persisted_id_gets_new_id(svc: FinetuneService, deps: dict[str, Any], lifecycle: None) -> None:
    config = InstanceConfig.model_validate(_persisted_instance(None))
    assert config.options is not None

    promise = await svc.install_instance("default", config.options, config, save=False)
    await promise.wait()

    assert uuid.UUID(_installed_registration_id(svc))


@pytest.mark.asyncio
async def test_user_install_drops_earlier_pin(svc: FinetuneService, deps: dict[str, Any], lifecycle: None) -> None:
    svc._pinned_registration_ids["default"] = "stale-id"  # pyright: ignore[reportPrivateUsage]

    promise = await svc.install_instance("default", InstallServiceIn(spec={}))
    await promise.wait()

    assert _installed_registration_id(svc) != "stale-id"
    assert "default" not in svc._pinned_registration_ids  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_install_instance_failure_drops_pin_from_config(svc: FinetuneService, deps: dict[str, Any], lifecycle: None) -> None:
    svc.instances_info["other"] = Instance(_make_installed_info(), None, {}, InstanceConfig())
    config = InstanceConfig.model_validate(_persisted_instance("stable-id"))
    assert config.options is not None

    with pytest.raises(HTTPException):
        await svc.install_instance("default", config.options, config)

    assert "default" not in svc._pinned_registration_ids  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_install_instance_on_installed_instance_keeps_pin(svc: FinetuneService, deps: dict[str, Any], lifecycle: None) -> None:
    svc.instances_info["default"].installed = _make_installed_info()
    svc._pinned_registration_ids["default"] = "stable-id"  # pyright: ignore[reportPrivateUsage]

    with pytest.raises(HTTPException) as exc:
        await svc.install_instance("default", InstallServiceIn(spec={}))

    assert exc.value.status_code == 400
    assert svc._pinned_registration_ids["default"] == "stable-id"  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_update_instance_keeps_registration_id_despite_concurrent_install(
    svc: FinetuneService, deps: dict[str, Any], lifecycle: None
) -> None:
    promise = await svc.install_instance("default", InstallServiceIn(spec={}))
    await promise.wait()
    original = _installed_registration_id(svc)
    rejections: list[int] = []

    async def _concurrent_install(*_: Any, **__: Any) -> None:
        try:
            await svc.install_instance("default", InstallServiceIn(spec={}))
        except HTTPException as e:
            rejections.append(e.status_code)

    deps["docker_service"].uninstall_docker.side_effect = _concurrent_install

    promise = await svc.update_instance("default", InstallServiceIn(spec={"hardware": False}))
    await promise.wait()

    assert rejections == [400]
    assert _registered_ids(deps) == [original, original]
    assert _installed_registration_id(svc) == original


@pytest.mark.asyncio
async def test_rollback_after_failed_update_keeps_registration_id_despite_concurrent_install(
    svc: FinetuneService, deps: dict[str, Any], lifecycle: None
) -> None:
    promise = await svc.install_instance("default", InstallServiceIn(spec={}))
    await promise.wait()
    original = _installed_registration_id(svc)
    rejections: list[int] = []

    async def _install_and_run_docker(*_: Any, **__: Any) -> tuple[int, bool, bool]:
        if rejections:
            return (20000, True, False)
        try:
            await svc.install_instance("default", InstallServiceIn(spec={}))
        except HTTPException as e:
            rejections.append(e.status_code)
        raise RuntimeError("container failed")

    deps["docker_service"].install_and_run_docker.side_effect = _install_and_run_docker

    promise = await svc.update_instance("default", InstallServiceIn(spec={"hardware": False}))
    with pytest.raises(RuntimeError):
        await promise.wait()

    assert rejections == [400]
    assert _registered_ids(deps) == [original, original]
    assert _installed_registration_id(svc) == original


@pytest.mark.asyncio
async def test_update_instance_keeps_registration_id(svc: FinetuneService, deps: dict[str, Any], lifecycle: None) -> None:
    promise = await svc.install_instance("default", InstallServiceIn(spec={}))
    await promise.wait()
    original = _installed_registration_id(svc)

    promise = await svc.update_instance("default", InstallServiceIn(spec={"hardware": False}))
    await promise.wait()

    assert _registered_ids(deps) == [original, original]
    assert _installed_registration_id(svc) == original


@pytest.mark.asyncio
async def test_rollback_after_failed_update_keeps_registration_id(svc: FinetuneService, deps: dict[str, Any], lifecycle: None) -> None:
    promise = await svc.install_instance("default", InstallServiceIn(spec={}))
    await promise.wait()
    original = _installed_registration_id(svc)
    deps["docker_service"].install_and_run_docker.side_effect = [RuntimeError("container failed"), (20000, True, False)]

    promise = await svc.update_instance("default", InstallServiceIn(spec={"hardware": False}))
    with pytest.raises(RuntimeError):
        await promise.wait()

    assert _installed_registration_id(svc) == original
    assert _registered_ids(deps) == [original, original]


@pytest.mark.asyncio
async def test_uninstall_then_install_gets_new_registration_id(svc: FinetuneService, deps: dict[str, Any], lifecycle: None) -> None:
    promise = await svc.install_instance("default", InstallServiceIn(spec={}))
    await promise.wait()
    original = _installed_registration_id(svc)

    await svc.uninstall_instance("default", UninstallServiceIn(purge=False))
    promise = await svc.install_instance("default", InstallServiceIn(spec={}))
    await promise.wait()

    assert _installed_registration_id(svc) != original


@pytest.mark.asyncio
async def test_load_service_reuses_persisted_registration_id(svc: FinetuneService, deps: dict[str, Any], lifecycle: None) -> None:
    await svc.load_service({"instances": {"default": _persisted_instance("stable-id")}})

    assert _registered_ids(deps) == ["stable-id"]
    assert _installed_registration_id(svc) == "stable-id"
    deps["service_provider"].save_service_config.assert_not_called()


@pytest.mark.asyncio
async def test_load_service_backfills_registration_id_for_legacy_config(
    svc: FinetuneService, deps: dict[str, Any], lifecycle: None
) -> None:
    await svc.load_service({"instances": {"default": _persisted_instance(None)}})

    registration_id = _installed_registration_id(svc)
    deps["service_provider"].save_service_config.assert_awaited_once()
    saved = deps["service_provider"].save_service_config.await_args.args[1]
    assert saved["instances"]["default"]["registration_id"] == registration_id


@pytest.mark.asyncio
async def test_load_service_warns_when_registration_id_backfill_fails(svc: FinetuneService, deps: dict[str, Any], lifecycle: None) -> None:
    deps["service_provider"].save_service_config.side_effect = OSError("disk full")

    await svc.load_service({"instances": {"default": _persisted_instance(None)}})

    assert svc.instances_info["default"].installed is not None
    warning = deps["service_provider"].add_warning.await_args
    assert warning.kwargs["instance"] == "default"
    assert "X-DF-Route" in warning.args[1]
    assert "disk full" in warning.args[1]
