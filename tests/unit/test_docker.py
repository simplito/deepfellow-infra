# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

import asyncio
import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
import yaml
from aiodocker import DockerError
from fastapi import HTTPException

import server.docker as docker_mod
from server.docker import (
    ContainerStatus,
    DockerDisabledError,
    DockerImageNameInfo,
    DockerNotInstalledError,
    DockerOptions,
    DockerPath,
    DockerService,
    _container_attached_to_network,  # type: ignore[reportPrivateUsage]
    _container_command_matches,  # type: ignore[reportPrivateUsage]
    _container_env_matches,  # type: ignore[reportPrivateUsage]
    _container_gpu_matches,  # type: ignore[reportPrivateUsage]
    _container_restart_policy_matches,  # type: ignore[reportPrivateUsage]
    _container_shm_size_matches,  # type: ignore[reportPrivateUsage]
    _container_user_matches,  # type: ignore[reportPrivateUsage]
    _container_volumes_match,  # type: ignore[reportPrivateUsage]
    _diagnose_gpu_error,  # type: ignore[reportPrivateUsage]
    _extract_error_excerpt,  # type: ignore[reportPrivateUsage]
    _extract_published_host_port,  # type: ignore[reportPrivateUsage]
    _is_container_name_conflict,  # type: ignore[reportPrivateUsage]
    _matches_for_adoption,  # type: ignore[reportPrivateUsage]
    _parse_bind_volume,  # type: ignore[reportPrivateUsage]
    _parse_byte_size,  # type: ignore[reportPrivateUsage]
    create_docker_service,
    get_docker_auths,
    normalize_docker_platform,
)
from server.utils.core import CommandResult, CommandResult2
from server.utils.exceptions import AppError, DockerComposeStartError, DockerImageAuthorizationError, DockerImageDoesNotExistError
from server.utils.hardware import IntelGpuInfo, NvidiaGpuInfo


def make_result(exit_code: int = 0, stdout: str = "", stderr: str = "") -> CommandResult:
    return CommandResult(exit_code=exit_code, stdout=stdout, stderr=stderr)


def make_result2(stdout: str = "", stderr: str = "") -> CommandResult2:
    return CommandResult2(stdout=stdout, stderr=stderr)


def _make_docker_mock(images_mock: MagicMock | None = None) -> tuple[MagicMock, MagicMock]:
    """Return (docker_cm, docker_instance) with async context manager set up."""
    instance = MagicMock()
    if images_mock is not None:
        instance.images = images_mock
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=instance)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm, instance


@pytest.fixture
def docker_service(tmp_path: Path) -> DockerService:
    config = MagicMock()
    config.compose_prefix = ""
    config.container_name_prefix = ""
    config.docker_subnet = ""
    config.get_storage_dir.return_value = tmp_path
    port_service = MagicMock()
    with patch("server.docker.get_docker_auths", return_value={}):
        return DockerService(
            config=config,
            port_service=port_service,
            docker_compose_cmd="docker compose",
            has_gpu_support=True,
            os="linux",
            architecture="amd64",
            is_rootless=False,
            host_platform="linux/amd64",
        )


DEFAULT_ADOPTION_IMAGE_ID = "sha256:" + "a" * 64


def _opts(
    name: str = "mymodel",
    image: str = "ubuntu:latest",
    image_port: int = 8080,
    container_name: str | None = None,
    **kwargs: Any,
) -> DockerOptions:
    return DockerOptions(name=name, container_name=container_name, image=image, image_port=image_port, **kwargs)


@pytest.mark.parametrize(
    ("platform_str", "expectation"),
    [
        ("linux/amd64", "linux/amd64"),
        ("linux/x86_64", "linux/amd64"),
        ("linux/arm64/v8", "linux/arm64/v8"),
        ("linux/arm64", "linux/arm64/v8"),
        ("linux/aarch64/v8", "linux/arm64/v8"),
        ("linux/aarch64", "linux/arm64/v8"),
        ("linux/armhf", "linux/arm/v7"),
        ("linux/armhf/v7", "linux/arm/v7"),
        ("linux/armv7l", "linux/arm/v7"),
        ("linux/armv7", "linux/arm/v7"),
        ("linux/arm", "linux/arm/v7"),
        ("linux/arm/v7", "linux/arm/v7"),
        ("linux/arm/v6", "linux/arm/v6"),
        ("linux/arm/v5", "linux/arm/v5"),
        ("linux/i386", "linux/386"),
        ("linux/386", "linux/386"),
        ("linux/arm64/v9", "linux/arm64/v9"),
        ("linux/fake", "linux/fake"),
        ("linux/fake/v1", "linux/fake/v1"),
    ],
)
def test_normalize_docker_platform(platform_str: str, expectation: str):
    result = normalize_docker_platform(platform_str)

    assert result == expectation


@pytest.mark.parametrize(
    ("full_image", "expected_registry", "expected_namespace", "expected_image_name"),
    [
        # Official Docker Hub image (one part)
        ("python", "docker.io", "library", "python"),
        # Docker Hub image with namespace (two parts)
        ("bitnami/redis", "docker.io", "bitnami", "redis"),
        # Third-party registry (three parts)
        ("ghcr.io/username/image", "ghcr.io", "username", "image"),
        # Registry with a port
        ("localhost:5000/my-app", "localhost:5000", "library", "my-app"),
        # Deeply nested namespace (e.g., AWS ECR or GitLab)
        ("123456789.dkr.ecr.us-east-1.amazonaws.com/org/team/app", "123456789.dkr.ecr.us-east-1.amazonaws.com", "org/team", "app"),
        # Registry with namespace and image
        ("my-reg.internal/dev-team/api-server", "my-reg.internal", "dev-team", "api-server"),
    ],
)
def test_docker_image_name_info_parse(full_image: str, expected_registry: str, expected_namespace: str, expected_image_name: str):
    """Test that various image strings are correctly parsed into components."""
    info = DockerImageNameInfo.parse(full_image)

    assert info.registry == expected_registry
    assert info.namespace == expected_namespace
    assert info.image_name == expected_image_name


@pytest.mark.parametrize(
    ("image", "digest", "expected"),
    [
        # plain name — digest appended with single @
        ("ubuntu", "sha256:abc", "ubuntu@sha256:abc"),
        # tagged image — digest appended after tag
        ("ubuntu:22.04", "sha256:abc", "ubuntu:22.04@sha256:abc"),
        # already-digested image — old digest replaced, no double @@
        (
            "ghcr.io/org/img@sha256:oldhash",
            "sha256:newhash",
            "ghcr.io/org/img@sha256:newhash",
        ),
        # no digest — image unchanged
        ("ubuntu", None, "ubuntu"),
    ],
)
def test_replace_image_digest(image: str, digest: str | None, expected: str):
    svc: DockerService = object.__new__(DockerService)
    result = svc.replace_image_digest(image, digest)
    assert result == expected
    if digest:
        assert "@@" not in result


def test_docker_image_name_info_is_frozen():
    """Verify that the dataclass is indeed frozen (immutable)."""
    info = DockerImageNameInfo.parse("alpine")

    with pytest.raises(Exception):  # dataclasses.FrozenInstanceError  # noqa: B017, PT011
        info.image_name = "ubuntu"  # type: ignore


def test_get_docker_auths_missing_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    assert get_docker_auths() == {}


def test_get_docker_auths_valid_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    docker_dir = tmp_path / ".docker"
    docker_dir.mkdir()
    config = {"auths": {"registry.example.com": {"auth": "token123"}}}
    (docker_dir / "config.json").write_text(json.dumps(config))

    result = get_docker_auths()

    assert result == {"registry.example.com": "token123"}


def test_get_docker_auths_filters_missing_auth_field(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    docker_dir = tmp_path / ".docker"
    docker_dir.mkdir()
    config = {"auths": {"registry.example.com": {}}}
    (docker_dir / "config.json").write_text(json.dumps(config))

    result = get_docker_auths()

    assert result == {}


def test_get_docker_auths_malformed_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    docker_dir = tmp_path / ".docker"
    docker_dir.mkdir()
    (docker_dir / "config.json").write_text("not json {{{")

    result = get_docker_auths()

    assert result == {}


def test_docker_service_init_stores_auths() -> None:
    config = MagicMock()
    port_service = MagicMock()
    fake_auths = {"reg.example.com": "mytoken"}

    with patch("server.docker.get_docker_auths", return_value=fake_auths):
        svc = DockerService(
            config=config,
            port_service=port_service,
            docker_compose_cmd="docker compose",
            has_gpu_support=False,
            os="linux",
            architecture="amd64",
            is_rootless=False,
            host_platform="linux/amd64",
        )

    assert svc.auths == fake_auths


@pytest.mark.parametrize(
    ("manifest", "expected"),
    [
        ({"layers": [{"size": 100}, {"size": 200}, {"size": 300}]}, 600),
        ({}, 0),
        ({"layers": [{"size": 100}, {}]}, 100),
    ],
    ids=["sums_layers", "no_layers_key", "missing_size_treated_as_zero"],
)
def test_calculate_total_layer_size(docker_service: DockerService, manifest: dict[str, Any], expected: int) -> None:
    assert docker_service.calculate_total_layer_size(manifest) == expected


@pytest.mark.parametrize(
    ("manifest", "expected"),
    [
        pytest.param({}, None, id="no_manifests"),
        pytest.param(
            {"manifests": [{"digest": "sha256:abc"}]},
            "sha256:abc",
            id="single_manifest",
        ),
        pytest.param(
            {
                "manifests": [
                    {"digest": "sha256:wrong", "platform": {"os": "windows", "architecture": "amd64"}},
                    {"digest": "sha256:right", "platform": {"os": "linux", "architecture": "amd64"}},
                ]
            },
            "sha256:right",
            id="exact_os_and_arch_match",
        ),
        pytest.param(
            {
                "manifests": [
                    {"digest": "sha256:arch-match", "platform": {"os": "windows", "architecture": "amd64"}},
                    {"digest": "sha256:no-match", "platform": {"os": "windows", "architecture": "arm64"}},
                ]
            },
            "sha256:arch-match",
            id="architecture_only_match",
        ),
        pytest.param(
            {
                "manifests": [
                    {"digest": "sha256:unknown", "platform": {"os": "unknown", "architecture": "unknown"}},
                    {"digest": "sha256:arm", "platform": {"os": "linux", "architecture": "arm64"}},
                ]
            },
            "sha256:unknown",
            id="unknown_unknown_fallback",
        ),
        pytest.param(
            {
                "manifests": [
                    {"digest": "sha256:first", "platform": {"os": "freebsd", "architecture": "riscv64"}},
                    {"digest": "sha256:second", "platform": {"os": "plan9", "architecture": "mips"}},
                ]
            },
            "sha256:first",
            id="first_manifest_last_fallback",
        ),
    ],
)
def test_get_platform_digest(docker_service: DockerService, manifest: dict[str, list[dict[str, str]]], expected: str | None) -> None:
    assert docker_service.get_platform_digest(manifest) == expected


def test_replace_image_digest_appends_digest(docker_service: DockerService) -> None:
    result = docker_service.replace_image_digest("ubuntu:latest", "sha256:abc")

    assert result == "ubuntu:latest@sha256:abc"


def test_replace_image_digest_strips_old_sha(docker_service: DockerService) -> None:
    result = docker_service.replace_image_digest("ubuntu:latest@sha256:old", "sha256:new")

    assert result.endswith("@sha256:new")
    assert "sha256:old" not in result


def test_replace_image_digest_none_returns_original(docker_service: DockerService) -> None:
    result = docker_service.replace_image_digest("ubuntu:latest", None)

    assert result == "ubuntu:latest"


@pytest.mark.asyncio
async def test_get_user_for_docker_rootless(tmp_path: Path) -> None:
    config = MagicMock()
    config.get_storage_dir.return_value = tmp_path

    with patch("server.docker.get_docker_auths", return_value={}):
        svc = DockerService(
            config=config,
            port_service=MagicMock(),
            docker_compose_cmd="docker compose",
            has_gpu_support=False,
            os="linux",
            architecture="amd64",
            is_rootless=True,
            host_platform="linux/amd64",
        )

    assert await svc.get_user_for_docker() == "0:0"


@pytest.mark.asyncio
async def test_get_user_for_docker_non_rootless(docker_service: DockerService) -> None:
    result = await docker_service.get_user_for_docker()

    assert result == f"{os.getuid()}:{os.getgid()}"


@pytest.mark.asyncio
async def test_start_docker_compose_command_contains_keywords(docker_service: DockerService, tmp_path: Path) -> None:
    compose_file = tmp_path / "compose.yaml"

    with patch("server.docker.Utils.run_command", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = make_result(exit_code=0)

        await docker_service.start_docker_compose(compose_file)

    cmd = mock_run.call_args[0][0]
    for kw in ["up", "-d", "--wait"]:
        assert kw in cmd


@pytest.mark.asyncio
async def test_stop_docker_compose_command_contains_keywords(docker_service: DockerService, tmp_path: Path) -> None:
    compose_file = tmp_path / "compose.yaml"

    with patch("server.docker.Utils.run_command_for_success", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = make_result2()
        await docker_service.stop_docker_compose(compose_file)

    cmd = mock_run.call_args[0][0]
    for kw in ["down", "--remove-orphans"]:
        assert kw in cmd


@pytest.mark.asyncio
async def test_restart_docker_compose_command_contains_keywords(docker_service: DockerService, tmp_path: Path) -> None:
    compose_file = tmp_path / "compose.yaml"
    compose_file.write_text("services: {}")
    options = _opts()

    with (
        patch.object(docker_service, "get_docker_compose_file_path", return_value=compose_file),
        patch("server.docker.Utils.run_command_for_success", new_callable=AsyncMock) as mock_run,
    ):
        mock_run.return_value = make_result2()
        await docker_service.restart_docker_compose(options)

    cmd = mock_run.call_args[0][0]
    assert "restart" in cmd


@pytest.mark.asyncio
async def test_restart_docker_compose_falls_back_to_container_when_file_missing(docker_service: DockerService, tmp_path: Path) -> None:
    compose_file = tmp_path / "missing.yaml"
    options = _opts(container_name="my-container")

    with (
        patch.object(docker_service, "get_docker_compose_file_path", return_value=compose_file),
        patch("server.docker.Utils.run_command_for_success", new_callable=AsyncMock) as mock_run,
        patch.object(docker_service, "_restart_container", new_callable=AsyncMock) as mock_restart_container,
    ):
        await docker_service.restart_docker_compose(options)

    assert mock_run.call_count == 0
    mock_restart_container.assert_called_once_with("my-container")


@pytest.mark.asyncio
async def test_restart_docker_compose_raises_when_no_file_and_no_container_name(docker_service: DockerService, tmp_path: Path) -> None:
    compose_file = tmp_path / "missing.yaml"
    options = _opts()  # container_name=None

    with (
        patch.object(docker_service, "get_docker_compose_file_path", return_value=compose_file),
        pytest.raises(HTTPException) as exc_info,
    ):
        await docker_service.restart_docker_compose(options)

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_stop_docker_calls_stop_compose_when_file_exists(docker_service: DockerService, tmp_path: Path) -> None:
    compose_file = tmp_path / "compose.yaml"
    compose_file.write_text("services: {}")
    options = _opts()

    with (
        patch.object(docker_service, "get_docker_compose_file_path", return_value=compose_file),
        patch.object(docker_service, "stop_docker_compose", new_callable=AsyncMock) as mock_stop,
    ):
        await docker_service.stop_docker(options)

    assert mock_stop.call_count == 1
    assert mock_stop.call_args == call(compose_file)


@pytest.mark.asyncio
async def test_stop_docker_does_not_call_stop_compose_when_file_missing(
    docker_service: DockerService, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    compose_file = tmp_path / "missing.yaml"
    options = _opts()  # container_name=None: no compose file and no way to identify the container

    with (
        patch.object(docker_service, "get_docker_compose_file_path", return_value=compose_file),
        patch.object(docker_service, "stop_docker_compose", new_callable=AsyncMock) as mock_stop,
        patch.object(docker_service, "_stop_and_remove_container", new_callable=AsyncMock) as mock_stop_and_remove,
        caplog.at_level("WARNING", logger="uvicorn.error"),
    ):
        await docker_service.stop_docker(options)

    assert mock_stop.call_count == 0
    mock_stop_and_remove.assert_not_called()
    assert any(record.levelname == "WARNING" and "mymodel" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_stop_docker_falls_back_to_container_when_file_missing(docker_service: DockerService, tmp_path: Path) -> None:
    compose_file = tmp_path / "missing.yaml"
    options = _opts(container_name="my-container")

    with (
        patch.object(docker_service, "get_docker_compose_file_path", return_value=compose_file),
        patch.object(docker_service, "stop_docker_compose", new_callable=AsyncMock) as mock_stop,
        patch.object(docker_service, "_stop_and_remove_container", new_callable=AsyncMock) as mock_stop_and_remove,
    ):
        await docker_service.stop_docker(options)

    assert mock_stop.call_count == 0
    mock_stop_and_remove.assert_called_once_with("my-container")


@pytest.mark.asyncio
async def test_get_docker_compose_logs_returns_stdout(docker_service: DockerService, tmp_path: Path) -> None:
    compose_file = tmp_path / "compose.yaml"
    with patch("server.docker.Utils.run_command_for_success", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = make_result2(stdout="log output")

        result = await docker_service.get_docker_compose_logs(compose_file)

    assert result == "log output"


@pytest.mark.asyncio
async def test_run_command_docker_compose_calls_exec_and_returns_stdout(docker_service: DockerService, tmp_path: Path) -> None:
    compose_file = tmp_path / "compose.yaml"
    with patch("server.docker.Utils.run_command_for_success", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = make_result2(stdout="exec output")

        result = await docker_service.run_command_docker_compose(compose_file, "myservice", "echo hello")

    assert result == "exec output"
    cmd = mock_run.call_args[0][0]
    assert "exec" in cmd
    assert "myservice" in cmd
    assert "echo" in cmd


@pytest.mark.asyncio
async def test_is_docker_compose_running_returns_false_when_file_missing(docker_service: DockerService, tmp_path: Path) -> None:
    compose_file = tmp_path / "missing.yaml"

    result = await docker_service.is_docker_compose_running(compose_file, "myservice")

    assert result is False


@pytest.mark.asyncio
async def test_is_docker_compose_running_returns_true_when_service_in_output(docker_service: DockerService, tmp_path: Path) -> None:
    compose_file = tmp_path / "compose.yaml"
    compose_file.write_text("services: {}")
    with patch("server.docker.Utils.run_command", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = make_result(exit_code=0, stdout="myservice\nother")

        result = await docker_service.is_docker_compose_running(compose_file, "myservice")

    assert result is True


@pytest.mark.asyncio
async def test_is_docker_compose_running_returns_false_when_service_not_in_output(docker_service: DockerService, tmp_path: Path) -> None:
    compose_file = tmp_path / "compose.yaml"
    compose_file.write_text("services: {}")
    with patch("server.docker.Utils.run_command", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = make_result(exit_code=0, stdout="other-service")

        result = await docker_service.is_docker_compose_running(compose_file, "myservice")

    assert result is False


@pytest.mark.asyncio
async def test_is_docker_compose_running_returns_false_on_exception(docker_service: DockerService, tmp_path: Path) -> None:
    compose_file = tmp_path / "compose.yaml"
    compose_file.write_text("services: {}")
    with patch("server.docker.Utils.run_command", new_callable=AsyncMock) as mock_run:
        mock_run.side_effect = RuntimeError("command failed")

        result = await docker_service.is_docker_compose_running(compose_file, "myservice")

    assert result is False


@pytest.mark.asyncio
async def test_is_docker_image_pulled_returns_true_when_found(docker_service: DockerService) -> None:
    images = MagicMock()
    images.get = AsyncMock(return_value=MagicMock())
    cm, _ = _make_docker_mock(images)

    with patch("server.docker.Docker", return_value=cm):
        result = await docker_service.is_docker_image_pulled("ubuntu:latest")

    assert result is True


@pytest.mark.asyncio
async def test_is_docker_image_pulled_returns_false_on_404(docker_service: DockerService) -> None:
    images = MagicMock()
    images.get = AsyncMock(side_effect=DockerError(status=404, message="not found"))
    cm, _ = _make_docker_mock(images)

    with patch("server.docker.Docker", return_value=cm):
        result = await docker_service.is_docker_image_pulled("ubuntu:latest")

    assert result is False


@pytest.mark.asyncio
async def test_remove_image_calls_delete_with_force(docker_service: DockerService) -> None:
    images = MagicMock()
    images.delete = AsyncMock()
    cm, _ = _make_docker_mock(images)

    with patch("server.docker.Docker", return_value=cm):
        await docker_service.remove_image("ubuntu:latest")

    assert images.delete.call_count == 1
    assert images.delete.call_args == call("ubuntu:latest", force=True)


@pytest.mark.asyncio
async def test_remove_image_ignores_404(docker_service: DockerService) -> None:
    images = MagicMock()
    images.delete = AsyncMock(side_effect=DockerError(status=404, message="not found"))
    cm, _ = _make_docker_mock(images)

    with patch("server.docker.Docker", return_value=cm):
        await docker_service.remove_image("ubuntu:latest")  # should not raise


def _mock_container_show(inspect_data: dict[str, Any], resolved_image_id: str = DEFAULT_ADOPTION_IMAGE_ID) -> tuple[MagicMock, MagicMock]:
    """Return (docker_cm, container_mock) with `.containers.container(name).show()` wired to return inspect_data.

    Also wires `.images.inspect(...)` to resolve to *resolved_image_id*, since adoption resolves the
    configured image's current local ID to compare against the live container's fixed image ID.
    """
    cm, instance = _make_docker_mock()
    container = MagicMock()
    container.show = AsyncMock(return_value=inspect_data)
    instance.containers = MagicMock()
    instance.containers.container = MagicMock(return_value=container)
    instance.images = MagicMock()
    instance.images.inspect = AsyncMock(return_value={"Id": resolved_image_id})
    return cm, instance.containers.container


@pytest.mark.asyncio
async def test_get_container_status_running_and_healthy(docker_service: DockerService) -> None:
    cm, container_fn = _mock_container_show({"State": {"Status": "running", "Health": {"Status": "healthy"}}, "RestartCount": 0})

    with patch("server.docker.Docker", return_value=cm):
        result = await docker_service.get_container_status("mymodel")

    assert result == ContainerStatus(exists=True, state="running", health="healthy", restart_count=0)
    container_fn.assert_called_once_with("mymodel")


@pytest.mark.asyncio
async def test_get_container_status_exited(docker_service: DockerService) -> None:
    cm, _ = _mock_container_show({"State": {"Status": "exited"}, "RestartCount": 2})

    with patch("server.docker.Docker", return_value=cm):
        result = await docker_service.get_container_status("mymodel")

    assert result == ContainerStatus(exists=True, state="exited", health="", restart_count=2)


@pytest.mark.asyncio
async def test_get_container_status_restarting(docker_service: DockerService) -> None:
    cm, _ = _mock_container_show({"State": {"Status": "restarting"}, "RestartCount": 5})

    with patch("server.docker.Docker", return_value=cm):
        result = await docker_service.get_container_status("mymodel")

    assert result == ContainerStatus(exists=True, state="restarting", health="", restart_count=5)


@pytest.mark.asyncio
async def test_get_container_status_running_but_unhealthy(docker_service: DockerService) -> None:
    cm, _ = _mock_container_show({"State": {"Status": "running", "Health": {"Status": "unhealthy"}}, "RestartCount": 0})

    with patch("server.docker.Docker", return_value=cm):
        result = await docker_service.get_container_status("mymodel")

    assert result == ContainerStatus(exists=True, state="running", health="unhealthy", restart_count=0)


@pytest.mark.asyncio
async def test_get_container_status_missing_container_returns_not_exists(docker_service: DockerService) -> None:
    instance = MagicMock()
    instance.containers = MagicMock()
    container = MagicMock()
    container.show = AsyncMock(side_effect=DockerError(status=404, message="no such container"))
    instance.containers.container = MagicMock(return_value=container)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=instance)
    cm.__aexit__ = AsyncMock(return_value=False)

    with patch("server.docker.Docker", return_value=cm):
        result = await docker_service.get_container_status("mymodel")

    assert result == ContainerStatus(exists=False, state="", health="", restart_count=0)


@pytest.mark.asyncio
async def test_get_container_status_reraises_non_404_docker_error(docker_service: DockerService) -> None:
    instance = MagicMock()
    instance.containers = MagicMock()
    container = MagicMock()
    container.show = AsyncMock(side_effect=DockerError(status=500, message="server error"))
    instance.containers.container = MagicMock(return_value=container)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=instance)
    cm.__aexit__ = AsyncMock(return_value=False)

    with patch("server.docker.Docker", return_value=cm), pytest.raises(DockerError):
        await docker_service.get_container_status("mymodel")


@pytest.mark.asyncio
async def test_remove_image_reraises_non_404(docker_service: DockerService) -> None:
    images = MagicMock()
    images.delete = AsyncMock(side_effect=DockerError(status=500, message="server error"))
    cm, _ = _make_docker_mock(images)

    with patch("server.docker.Docker", return_value=cm), pytest.raises(DockerError):
        await docker_service.remove_image("ubuntu:latest")


@pytest.mark.asyncio
async def test_stop_and_remove_container_stops_and_deletes(docker_service: DockerService) -> None:
    container = MagicMock()
    container.stop = AsyncMock()
    container.delete = AsyncMock()
    instance = MagicMock()
    instance.containers.container = MagicMock(return_value=container)
    cm, _ = _make_docker_mock()
    cm.__aenter__ = AsyncMock(return_value=instance)

    with patch("server.docker.Docker", return_value=cm):
        await docker_service._stop_and_remove_container("mymodel")  # type: ignore[reportPrivateUsage]

    instance.containers.container.assert_called_once_with("mymodel")
    container.stop.assert_called_once()
    container.delete.assert_called_once_with(force=True)


@pytest.mark.asyncio
async def test_stop_and_remove_container_tolerates_404_on_stop(docker_service: DockerService) -> None:
    container = MagicMock()
    container.stop = AsyncMock(side_effect=DockerError(status=404, message="no such container"))
    container.delete = AsyncMock()
    instance = MagicMock()
    instance.containers.container = MagicMock(return_value=container)
    cm, _ = _make_docker_mock()
    cm.__aenter__ = AsyncMock(return_value=instance)

    with patch("server.docker.Docker", return_value=cm):
        await docker_service._stop_and_remove_container("mymodel")  # type: ignore[reportPrivateUsage]  # should not raise

    container.delete.assert_not_called()


@pytest.mark.asyncio
async def test_stop_and_remove_container_tolerates_404_on_delete(docker_service: DockerService) -> None:
    container = MagicMock()
    container.stop = AsyncMock()
    container.delete = AsyncMock(side_effect=DockerError(status=404, message="no such container"))
    instance = MagicMock()
    instance.containers.container = MagicMock(return_value=container)
    cm, _ = _make_docker_mock()
    cm.__aenter__ = AsyncMock(return_value=instance)

    with patch("server.docker.Docker", return_value=cm):
        await docker_service._stop_and_remove_container("mymodel")  # type: ignore[reportPrivateUsage]  # should not raise


@pytest.mark.asyncio
async def test_stop_and_remove_container_reraises_non_404_on_stop(docker_service: DockerService) -> None:
    container = MagicMock()
    container.stop = AsyncMock(side_effect=DockerError(status=500, message="server error"))
    instance = MagicMock()
    instance.containers.container = MagicMock(return_value=container)
    cm, _ = _make_docker_mock()
    cm.__aenter__ = AsyncMock(return_value=instance)

    with patch("server.docker.Docker", return_value=cm), pytest.raises(DockerError):
        await docker_service._stop_and_remove_container("mymodel")  # type: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_stop_and_remove_container_reraises_non_404_on_delete(docker_service: DockerService) -> None:
    container = MagicMock()
    container.stop = AsyncMock()
    container.delete = AsyncMock(side_effect=DockerError(status=500, message="server error"))
    instance = MagicMock()
    instance.containers.container = MagicMock(return_value=container)
    cm, _ = _make_docker_mock()
    cm.__aenter__ = AsyncMock(return_value=instance)

    with patch("server.docker.Docker", return_value=cm), pytest.raises(DockerError):
        await docker_service._stop_and_remove_container("mymodel")  # type: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_restart_container_calls_restart(docker_service: DockerService) -> None:
    container = MagicMock()
    container.restart = AsyncMock()
    instance = MagicMock()
    instance.containers.container = MagicMock(return_value=container)
    cm, _ = _make_docker_mock()
    cm.__aenter__ = AsyncMock(return_value=instance)

    with patch("server.docker.Docker", return_value=cm):
        await docker_service._restart_container("mymodel")  # type: ignore[reportPrivateUsage]

    instance.containers.container.assert_called_once_with("mymodel")
    container.restart.assert_called_once()


@pytest.mark.asyncio
async def test_restart_container_raises_http_exception_on_404(docker_service: DockerService) -> None:
    container = MagicMock()
    container.restart = AsyncMock(side_effect=DockerError(status=404, message="no such container"))
    instance = MagicMock()
    instance.containers.container = MagicMock(return_value=container)
    cm, _ = _make_docker_mock()
    cm.__aenter__ = AsyncMock(return_value=instance)

    with patch("server.docker.Docker", return_value=cm), pytest.raises(HTTPException) as exc_info:
        await docker_service._restart_container("mymodel")  # type: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_restart_container_reraises_non_404_docker_error(docker_service: DockerService) -> None:
    container = MagicMock()
    container.restart = AsyncMock(side_effect=DockerError(status=500, message="server error"))
    instance = MagicMock()
    instance.containers.container = MagicMock(return_value=container)
    cm, _ = _make_docker_mock()
    cm.__aenter__ = AsyncMock(return_value=instance)

    with patch("server.docker.Docker", return_value=cm), pytest.raises(DockerError):
        await docker_service._restart_container("mymodel")  # type: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_is_docker_compose_healthy_false_when_file_missing(docker_service: DockerService, tmp_path: Path) -> None:
    compose_file = tmp_path / "missing.yaml"

    result = await docker_service.is_docker_compose_healthy(compose_file, "myservice")

    assert result is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("exit_code", "container_json", "expected"),
    [
        pytest.param(1, None, False, id="nonzero_exit"),
        pytest.param(0, {"Health": "healthy", "State": "running"}, True, id="healthy"),
        pytest.param(0, {"Health": "unhealthy", "State": "running"}, False, id="unhealthy"),
        pytest.param(0, {"Health": "", "State": "running"}, True, id="running_no_healthcheck"),
        pytest.param(0, {"Health": "", "State": "exited"}, False, id="state_stopped"),
    ],
)
async def test_is_docker_compose_healthy_cases(
    docker_service: DockerService,
    tmp_path: Path,
    exit_code: int,
    container_json: dict[str, str] | None,
    expected: bool,
) -> None:
    compose_file = tmp_path / "compose.yaml"
    compose_file.write_text("services: {}")
    stdout = json.dumps(container_json) if container_json is not None else ""
    with patch("server.docker.Utils.run_command", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = make_result(exit_code=exit_code, stdout=stdout)

        result = await docker_service.is_docker_compose_healthy(compose_file, "myservice")

    assert result is expected


@pytest.mark.asyncio
async def test_is_docker_compose_healthy_exception(
    docker_service: DockerService,
    tmp_path: Path,
) -> None:
    compose_file = tmp_path / "compose.yaml"
    compose_file.write_text("services: {}")
    with patch("server.docker.Utils.run_command", new_callable=AsyncMock) as mock_run:
        mock_run.side_effect = RuntimeError("unexpected")

        result = await docker_service.is_docker_compose_healthy(compose_file, "myservice")

    assert result is False


@pytest.mark.asyncio
async def test_generate_docker_compose_content_basic_port(docker_service: DockerService) -> None:
    options = _opts(image_port=8080)

    result = await docker_service.generate_docker_compose_content(options, 12345)

    service = result["services"]["mymodel"]
    assert service["ports"] == ["127.0.0.1:12345:8080"]  # pyright: ignore[reportTypedDictNotRequiredAccess]


@pytest.mark.asyncio
async def test_generate_docker_compose_content_subnet_mode(docker_service: DockerService) -> None:
    options = _opts(subnet="my-net")

    result = await docker_service.generate_docker_compose_content(options, None)

    service = result["services"]["mymodel"]
    assert "ports" not in service
    assert "my-net" in service["networks"]  # pyright: ignore[reportTypedDictNotRequiredAccess]
    assert result["networks"] == {"my-net": {"external": True}}  # pyright: ignore[reportTypedDictNotRequiredAccess]


@pytest.mark.asyncio
async def test_generate_docker_compose_content_nvidia_gpu(docker_service: DockerService) -> None:
    gpu = NvidiaGpuInfo(name="RTX 3090", vram="24 GB", id=0)
    options = _opts(hardware=[gpu])

    result = await docker_service.generate_docker_compose_content(options, 12345)

    deploy = result["services"]["mymodel"]["deploy"]  # pyright: ignore[reportTypedDictNotRequiredAccess]
    devices = deploy["resources"]["reservations"]["devices"]  # pyright: ignore[reportTypedDictNotRequiredAccess]
    assert devices[0]["driver"] == "nvidia"  # pyright: ignore[reportTypedDictNotRequiredAccess]
    assert "0" in devices[0]["device_ids"]  # pyright: ignore[reportTypedDictNotRequiredAccess]


@pytest.mark.asyncio
async def test_generate_docker_compose_content_nvidia_gpu_no_support_raises(tmp_path: Path) -> None:
    config = MagicMock()
    config.get_storage_dir.return_value = tmp_path
    config.compose_prefix = ""
    with patch("server.docker.get_docker_auths", return_value={}):
        svc = DockerService(
            config=config,
            port_service=MagicMock(),
            docker_compose_cmd="docker compose",
            has_gpu_support=False,
            os="linux",
            architecture="amd64",
            is_rootless=False,
            host_platform="linux/amd64",
        )
    gpu = NvidiaGpuInfo(name="RTX 3090", vram="24 GB", id=0)
    options = _opts(hardware=[gpu])

    with pytest.raises(AppError):
        await svc.generate_docker_compose_content(options, 12345)


@pytest.mark.asyncio
async def test_generate_docker_compose_content_intel_gpu(docker_service: DockerService) -> None:
    gpu = IntelGpuInfo(name="Intel Arc", vram=None, id=0)
    options = _opts(hardware=[gpu])
    with patch("server.docker.Path") as mock_path_cls:
        mock_dri = MagicMock()
        mock_dri.iterdir.return_value = iter([])
        mock_path_cls.side_effect = lambda p: mock_dri if p == "/dev/dri" else Path(p)  # pyright: ignore[reportUnknownLambdaType]

        result = await docker_service.generate_docker_compose_content(options, 12345)

    assert "/dev/dri:/dev/dri" in result["services"]["mymodel"]["devices"]  # pyright: ignore[reportTypedDictNotRequiredAccess]


@pytest.mark.asyncio
async def test_generate_docker_compose_content_no_port_no_subnet_raises(docker_service: DockerService) -> None:
    options = _opts()

    with pytest.raises(AppError):
        await docker_service.generate_docker_compose_content(options, None)


@pytest.mark.asyncio
async def test_generate_docker_compose_content_optional_fields(docker_service: DockerService) -> None:
    options = _opts(
        healthcheck={"test": "curl localhost"},
        command="serve",
        volumes=["/data:/data"],
        restart="always",
        shm_size="1g",
        entrypoint="/entrypoint.sh",
        user="1000:1000",
    )

    result = await docker_service.generate_docker_compose_content(options, 12345)

    service = result["services"]["mymodel"]
    assert service["healthcheck"] == {"test": "curl localhost"}  # pyright: ignore[reportTypedDictNotRequiredAccess]
    assert service["command"] == "serve"  # pyright: ignore[reportTypedDictNotRequiredAccess]
    assert service["volumes"] == ["/data:/data"]  # pyright: ignore[reportTypedDictNotRequiredAccess]
    assert service["restart"] == "always"  # pyright: ignore[reportTypedDictNotRequiredAccess]
    assert service["shm_size"] == "1g"  # pyright: ignore[reportTypedDictNotRequiredAccess]
    assert service["entrypoint"] == "/entrypoint.sh"  # pyright: ignore[reportTypedDictNotRequiredAccess]
    assert service["user"] == "1000:1000"  # pyright: ignore[reportTypedDictNotRequiredAccess]


@pytest.mark.asyncio
async def test_generate_docker_compose_content_container_name(docker_service: DockerService) -> None:
    options = DockerOptions(name="mymodel", container_name="my-container", image="ubuntu:latest", image_port=8080)

    result = await docker_service.generate_docker_compose_content(options, 12345)

    assert result["services"]["mymodel"]["container_name"] == "my-container"  # pyright: ignore[reportTypedDictNotRequiredAccess]


@pytest.mark.asyncio
async def test_has_docker_compose_difference_file_missing(docker_service: DockerService, tmp_path: Path) -> None:
    compose_file = tmp_path / "missing.yaml"
    options = _opts()

    has_diff, port = await docker_service.has_docker_compose_difference(compose_file, options)

    assert has_diff is True
    assert port is None


@pytest.mark.asyncio
async def test_has_docker_compose_difference_content_matches(docker_service: DockerService, tmp_path: Path) -> None:
    compose_file = tmp_path / "compose.yaml"
    options = _opts(image_port=8080)
    port = 12345

    content = await docker_service.generate_docker_compose_content(options, port)

    yaml_text = yaml.dump(content, default_flow_style=False, sort_keys=False)
    compose_file.write_text(yaml_text)

    has_diff, returned_port = await docker_service.has_docker_compose_difference(compose_file, options)

    assert has_diff is False
    assert returned_port == port


@pytest.mark.asyncio
async def test_has_docker_compose_difference_content_differs(docker_service: DockerService, tmp_path: Path) -> None:
    compose_file = tmp_path / "compose.yaml"
    options = _opts(image_port=8080)
    port = 12345

    content = await docker_service.generate_docker_compose_content(options, port)

    yaml_text = yaml.dump(content, default_flow_style=False, sort_keys=False)
    compose_file.write_text(yaml_text)

    changed_options = _opts(image="differentimage:latest", image_port=8080)
    has_diff, returned_port = await docker_service.has_docker_compose_difference(compose_file, changed_options)

    assert has_diff is True
    assert returned_port == port


@pytest.mark.asyncio
async def test_get_existing_or_free_port_docker_returns_existing_port(docker_service: DockerService, tmp_path: Path) -> None:
    compose_file = tmp_path / "compose.yaml"
    options = _opts(image_port=8080)
    # The method uses split(":")[0], so we need a 2-segment port format to parse successfully
    compose_yaml = yaml.dump({"services": {"mymodel": {"ports": ["11434:8080"]}}}, default_flow_style=False)
    compose_file.write_text(compose_yaml)
    docker_service.port_service.is_port_available.return_value = True  # pyright: ignore[reportAttributeAccessIssue]

    port = await docker_service.get_existing_or_free_port_docker(compose_file, options)

    assert port == 11434


@pytest.mark.asyncio
async def test_get_existing_or_free_port_docker_gets_free_port_when_taken(docker_service: DockerService, tmp_path: Path) -> None:
    compose_file = tmp_path / "compose.yaml"
    options = _opts(image_port=8080)
    compose_yaml = yaml.dump({"services": {"mymodel": {"ports": ["11434:8080"]}}}, default_flow_style=False)
    compose_file.write_text(compose_yaml)
    docker_service.port_service.is_port_available.return_value = False  # pyright: ignore[reportAttributeAccessIssue]
    docker_service.port_service.get_free_port.return_value = 22222  # pyright: ignore[reportAttributeAccessIssue]

    port = await docker_service.get_existing_or_free_port_docker(compose_file, options)

    assert port == 22222
    assert docker_service.port_service.get_free_port.call_count == 1  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.asyncio
async def test_get_existing_or_free_port_docker_gets_free_port_when_no_file(docker_service: DockerService, tmp_path: Path) -> None:
    compose_file = tmp_path / "missing.yaml"
    options = _opts()
    docker_service.port_service.get_free_port.return_value = 33333  # pyright: ignore[reportAttributeAccessIssue]

    port = await docker_service.get_existing_or_free_port_docker(compose_file, options)

    assert port == 33333
    assert docker_service.port_service.get_free_port.call_count == 1  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.asyncio
async def test_create_compose_file_writes_yaml_and_returns_port(docker_service: DockerService, tmp_path: Path) -> None:
    compose_file = tmp_path / "compose.yaml"
    options = _opts(image_port=8080)
    docker_service.port_service.get_free_port.return_value = 44444  # pyright: ignore[reportAttributeAccessIssue]

    port = await docker_service.create_compose_file(compose_file, options)

    assert port == 44444
    assert compose_file.exists()
    loaded = yaml.safe_load(compose_file.read_text())
    assert "services" in loaded


@pytest.mark.asyncio
async def test_install_and_run_docker_not_running_no_diff_port_available(docker_service: DockerService, tmp_path: Path) -> None:
    options = _opts()
    compose_file = tmp_path / "compose.yaml"
    docker_service.port_service.is_port_available.return_value = True  # pyright: ignore[reportAttributeAccessIssue]

    with (
        patch.object(docker_service, "get_docker_compose_file_path", return_value=compose_file),
        patch.object(docker_service, "is_docker_compose_running", new_callable=AsyncMock, return_value=False),
        patch.object(docker_service, "has_docker_compose_difference", new_callable=AsyncMock, return_value=(False, 11434)),
        patch.object(docker_service, "start_docker_compose", new_callable=AsyncMock, return_value="") as mock_start,
        patch.object(docker_service, "create_compose_file", new_callable=AsyncMock, return_value=11434) as mock_create,
        patch.object(docker_service, "is_docker_compose_healthy", new_callable=AsyncMock, return_value=True),
    ):
        port, restarted, adopted = await docker_service.install_and_run_docker(options)

    assert mock_start.call_count == 1
    assert mock_create.call_count == 0
    assert port == 11434
    assert restarted is True
    assert adopted is False


@pytest.mark.asyncio
async def test_install_and_run_docker_not_running_no_diff_port_taken(docker_service: DockerService, tmp_path: Path) -> None:
    options = _opts()
    compose_file = tmp_path / "compose.yaml"
    docker_service.port_service.is_port_available.return_value = False  # pyright: ignore[reportAttributeAccessIssue]

    with (
        patch.object(docker_service, "get_docker_compose_file_path", return_value=compose_file),
        patch.object(docker_service, "is_docker_compose_running", new_callable=AsyncMock, return_value=False),
        patch.object(docker_service, "has_docker_compose_difference", new_callable=AsyncMock, return_value=(False, 11434)),
        patch.object(docker_service, "create_compose_file", new_callable=AsyncMock, return_value=22222) as mock_create,
        patch.object(docker_service, "start_docker_compose", new_callable=AsyncMock, return_value="") as mock_start,
        patch.object(docker_service, "is_docker_compose_healthy", new_callable=AsyncMock, return_value=True),
    ):
        port, restarted, adopted = await docker_service.install_and_run_docker(options)

    assert mock_create.call_count == 1
    assert mock_start.call_count == 1
    assert port == 22222
    assert restarted is True
    assert adopted is False


@pytest.mark.asyncio
async def test_install_and_run_docker_not_running_has_difference(docker_service: DockerService, tmp_path: Path) -> None:
    options = _opts()
    compose_file = tmp_path / "compose.yaml"

    with (
        patch.object(docker_service, "get_docker_compose_file_path", return_value=compose_file),
        patch.object(docker_service, "is_docker_compose_running", new_callable=AsyncMock, return_value=False),
        patch.object(docker_service, "has_docker_compose_difference", new_callable=AsyncMock, return_value=(True, None)),
        patch.object(docker_service, "create_compose_file", new_callable=AsyncMock, return_value=55555) as mock_create,
        patch.object(docker_service, "start_docker_compose", new_callable=AsyncMock, return_value="") as mock_start,
        patch.object(docker_service, "is_docker_compose_healthy", new_callable=AsyncMock, return_value=True),
    ):
        port, restarted, adopted = await docker_service.install_and_run_docker(options)

    assert mock_create.call_count == 1
    assert mock_start.call_count == 1
    assert port == 55555
    assert restarted is True
    assert adopted is False


@pytest.mark.asyncio
async def test_install_and_run_docker_running_has_difference(docker_service: DockerService, tmp_path: Path) -> None:
    options = _opts()
    compose_file = tmp_path / "compose.yaml"

    with (
        patch.object(docker_service, "get_docker_compose_file_path", return_value=compose_file),
        patch.object(docker_service, "is_docker_compose_running", new_callable=AsyncMock, return_value=True),
        patch.object(docker_service, "has_docker_compose_difference", new_callable=AsyncMock, return_value=(True, 11434)),
        patch.object(docker_service, "stop_docker_compose", new_callable=AsyncMock) as mock_stop,
        patch.object(docker_service, "create_compose_file", new_callable=AsyncMock, return_value=66666) as mock_create,
        patch.object(docker_service, "start_docker_compose", new_callable=AsyncMock, return_value="") as mock_start,
        patch.object(docker_service, "is_docker_compose_healthy", new_callable=AsyncMock, return_value=True),
    ):
        port, restarted, adopted = await docker_service.install_and_run_docker(options)

    assert mock_stop.call_count == 1
    assert mock_create.call_count == 1
    assert mock_start.call_count == 1
    assert port == 66666
    assert restarted is True
    assert adopted is False


@pytest.mark.asyncio
async def test_install_and_run_docker_running_no_difference(docker_service: DockerService, tmp_path: Path) -> None:
    options = _opts()
    compose_file = tmp_path / "compose.yaml"

    with (
        patch.object(docker_service, "get_docker_compose_file_path", return_value=compose_file),
        patch.object(docker_service, "is_docker_compose_running", new_callable=AsyncMock, return_value=True),
        patch.object(docker_service, "has_docker_compose_difference", new_callable=AsyncMock, return_value=(False, 11434)),
        patch.object(docker_service, "stop_docker_compose", new_callable=AsyncMock) as mock_stop,
        patch.object(docker_service, "create_compose_file", new_callable=AsyncMock) as mock_create,
        patch.object(docker_service, "start_docker_compose", new_callable=AsyncMock) as mock_start,
        patch.object(docker_service, "is_docker_compose_healthy", new_callable=AsyncMock, return_value=True),
    ):
        port, restarted, adopted = await docker_service.install_and_run_docker(options)

    assert mock_stop.call_count == 0
    assert mock_create.call_count == 0
    assert mock_start.call_count == 0
    assert port == 11434
    assert restarted is False
    assert adopted is False


@pytest.mark.asyncio
async def test_install_and_run_docker_unhealthy_raises_app_error(docker_service: DockerService, tmp_path: Path) -> None:
    options = _opts()
    compose_file = tmp_path / "compose.yaml"

    with (
        patch.object(docker_service, "get_docker_compose_file_path", return_value=compose_file),
        patch.object(docker_service, "is_docker_compose_running", new_callable=AsyncMock, return_value=False),
        patch.object(docker_service, "has_docker_compose_difference", new_callable=AsyncMock, return_value=(True, None)),
        patch.object(docker_service, "create_compose_file", new_callable=AsyncMock, return_value=11434),
        patch.object(docker_service, "start_docker_compose", new_callable=AsyncMock, return_value=""),
        patch.object(docker_service, "is_docker_compose_healthy", new_callable=AsyncMock, return_value=False),
        pytest.raises(AppError),
    ):
        await docker_service.install_and_run_docker(options)


@pytest.mark.asyncio
async def test_install_and_run_docker_subnet_mode_port_is_minus_one(docker_service: DockerService, tmp_path: Path) -> None:
    options = _opts(subnet="my-net")
    compose_file = tmp_path / "compose.yaml"

    with (
        patch.object(docker_service, "get_docker_compose_file_path", return_value=compose_file),
        patch.object(docker_service, "is_docker_compose_running", new_callable=AsyncMock, return_value=True),
        patch.object(docker_service, "has_docker_compose_difference", new_callable=AsyncMock, return_value=(False, None)),
        patch.object(docker_service, "is_docker_compose_healthy", new_callable=AsyncMock, return_value=True),
    ):
        port, restarted, adopted = await docker_service.install_and_run_docker(options)

    assert port == -1
    assert restarted is False
    assert adopted is False


@pytest.mark.asyncio
async def test_install_and_run_docker_adoption_skips_health_check_and_reports_not_restarted(
    docker_service: DockerService, tmp_path: Path
) -> None:
    options = _opts()
    compose_file = tmp_path / "compose.yaml"

    with (
        patch.object(docker_service, "get_docker_compose_file_path", return_value=compose_file),
        patch.object(docker_service, "is_docker_compose_running", new_callable=AsyncMock, return_value=False),
        patch.object(docker_service, "has_docker_compose_difference", new_callable=AsyncMock, return_value=(True, None)),
        patch.object(docker_service, "_ensure_compose_running", new_callable=AsyncMock, return_value=(None, 44444, True)),
        patch.object(docker_service, "is_docker_compose_healthy", new_callable=AsyncMock) as mock_healthy,
    ):
        port, restarted, adopted = await docker_service.install_and_run_docker(options)

    mock_healthy.assert_not_called()
    assert port == 44444
    assert restarted is False
    assert adopted is True


@pytest.mark.asyncio
async def test_install_and_run_docker_adoption_in_subnet_mode_returns_minus_one(docker_service: DockerService, tmp_path: Path) -> None:
    options = _opts(subnet="my-net")
    compose_file = tmp_path / "compose.yaml"

    with (
        patch.object(docker_service, "get_docker_compose_file_path", return_value=compose_file),
        patch.object(docker_service, "is_docker_compose_running", new_callable=AsyncMock, return_value=False),
        patch.object(docker_service, "has_docker_compose_difference", new_callable=AsyncMock, return_value=(True, None)),
        patch.object(docker_service, "_ensure_compose_running", new_callable=AsyncMock, return_value=(None, -1, True)),
        patch.object(docker_service, "is_docker_compose_healthy", new_callable=AsyncMock) as mock_healthy,
    ):
        port, restarted, adopted = await docker_service.install_and_run_docker(options)

    mock_healthy.assert_not_called()
    assert port == -1
    assert restarted is False
    assert adopted is True


@pytest.mark.asyncio
async def test_install_and_run_docker_cancel_kills_up_subprocess_and_rollback_tears_down(
    docker_service: DockerService, tmp_path: Path
) -> None:
    """End-to-end kill-on-cancel proof against a real subprocess (no docker daemon required).

    Cancels `install_and_run_docker` while a fake `docker compose up -d --wait` is genuinely mid-run (proven by
    it having written its pidfile before sleeping), then asserts the real subprocess was killed - and that the
    existing rollback path's `docker compose down --remove-orphans` (stop_docker_compose) actually tears down
    a real subprocess too, rather than racing an orphaned still-running `up`.
    """
    compose_file = tmp_path / "compose.yaml"
    pid_file = tmp_path / "up.pid"
    down_log = tmp_path / "down.log"
    fake_compose = tmp_path / "fake-docker-compose.sh"
    fake_compose.write_text(
        f"""#!/bin/sh
case " $* " in
  *' up '*)
    echo $$ > {pid_file}
    sleep 30
    ;;
  *' down '*)
    echo down >> {down_log}
    ;;
esac
"""
    )
    fake_compose.chmod(0o755)
    docker_service.docker_compose_cmd = str(fake_compose)

    options = _opts()
    docker_service.port_service.is_port_available.return_value = True  # pyright: ignore[reportAttributeAccessIssue]

    real_create_subprocess_exec = asyncio.create_subprocess_exec
    captured_procs: list[asyncio.subprocess.Process] = []

    async def capture_create_subprocess_exec(*args: object, **kwargs: object) -> asyncio.subprocess.Process:
        proc = await real_create_subprocess_exec(*args, **kwargs)  # type: ignore[arg-type]
        captured_procs.append(proc)
        return proc

    with (
        patch.object(docker_service, "get_docker_compose_file_path", return_value=compose_file),
        patch.object(docker_service, "is_docker_compose_running", new_callable=AsyncMock, return_value=False),
        patch.object(docker_service, "has_docker_compose_difference", new_callable=AsyncMock, return_value=(False, 18080)),
        patch("asyncio.create_subprocess_exec", side_effect=capture_create_subprocess_exec),
    ):
        task = asyncio.create_task(docker_service.install_and_run_docker(options))
        for _ in range(100):
            if pid_file.exists():
                break
            await asyncio.sleep(0.05)
        else:
            pytest.fail("fake docker compose up never started")

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert len(captured_procs) == 1
    assert captured_procs[0].returncode is not None  # killed, not left running

    await docker_service.stop_docker_compose(compose_file)
    assert down_log.exists()
    assert down_log.read_text().strip() == "down"


@pytest.mark.asyncio
async def test_uninstall_docker_runs_down_and_removes_file(docker_service: DockerService, tmp_path: Path) -> None:
    options = _opts()
    compose_file = tmp_path / "mymodel.yaml"
    compose_file.write_text("services: {}")

    with (
        patch.object(docker_service, "get_docker_compose_file_path", return_value=compose_file),
        patch("server.docker.Utils.run_command", new_callable=AsyncMock) as mock_run,
    ):
        mock_run.return_value = make_result()

        await docker_service.uninstall_docker(options)

    assert mock_run.call_count == 1
    assert not compose_file.exists()


def test_get_docker_compose_file_path_no_prefix(docker_service: DockerService, tmp_path: Path) -> None:
    docker_service.config.compose_prefix = ""

    path = docker_service.get_docker_compose_file_path("mymodel")

    assert path.name == "mymodel.yaml"
    assert path.suffix == ".yaml"


def test_get_docker_compose_file_path_with_prefix(docker_service: DockerService, tmp_path: Path) -> None:
    docker_service.config.compose_prefix = "pf-"

    path = docker_service.get_docker_compose_file_path("mymodel")

    assert path.name == "compose.yaml"
    assert "pf-mymodel" in str(path)


@pytest.mark.parametrize(("prefix", "expected"), [("df-", "df-mymodel"), ("", "mymodel")])
def test_get_docker_container_name(docker_service: DockerService, prefix: str, expected: str) -> None:
    docker_service.config.container_name_prefix = prefix

    assert docker_service.get_docker_container_name("mymodel") == expected


@pytest.mark.parametrize(("subnet", "expected"), [("my-net", "my-net"), ("", None)])
def test_get_docker_subnet(docker_service: DockerService, subnet: str, expected: str | None) -> None:
    docker_service.config.docker_subnet = subnet

    assert docker_service.get_docker_subnet() == expected


@pytest.mark.parametrize(("subnet", "expected"), [("my-net", "my-container"), (None, "localhost")])
def test_get_container_host(docker_service: DockerService, subnet: str | None, expected: str) -> None:
    assert docker_service.get_container_host(subnet, "my-container") == expected


@pytest.mark.parametrize(("subnet", "expected"), [("my-net", 8080), (None, 11434)])
def test_get_container_port(docker_service: DockerService, subnet: str | None, expected: int) -> None:
    assert docker_service.get_container_port(subnet, exposed_port=11434, original_port=8080) == expected


@pytest.mark.parametrize(
    "platform_str",
    [
        "linux",
        "linux/arm64/v8/extra",
    ],
)
def test_normalize_docker_platform_invalid_format_raises(platform_str: str) -> None:
    with pytest.raises(ValueError, match="Invalid platform format"):
        normalize_docker_platform(platform_str)


def test_normalize_docker_platform_mismatched_variant_raises() -> None:
    # aarch64 maps to arm64/v8, but passing v9 conflicts
    with pytest.raises(ValueError, match="mismatched variant"):
        normalize_docker_platform("linux/aarch64/v9")


def test_normalize_docker_platform_arm_without_variant_raises() -> None:
    # ARCHES_WITH_REQUIRED_VARIANT contains 'arm', but 'arm' alone has DEFAULT_VARIANTS["arm"]="v7"
    # We need an arch in ARCHES_WITH_REQUIRED_VARIANT that has no alias and no default variant
    # 'arm64' is in ARCHES_WITH_REQUIRED_VARIANT but DEFAULT_VARIANTS has "arm64"="v8"
    # The error fires when arch_normalized is in ARCHES_WITH_REQUIRED_VARIANT and variant_normalized is None
    # That happens for a custom arch alias that maps to arm64 with variant=None and no default...
    # Actually the simplest path: patch DEFAULT_VARIANTS to remove arm64 default

    original = docker_mod.DEFAULT_VARIANTS.copy()
    docker_mod.DEFAULT_VARIANTS.pop("arm64", None)

    try:
        with pytest.raises(ValueError, match="requires a variant"):
            normalize_docker_platform("linux/arm64")
    finally:
        docker_mod.DEFAULT_VARIANTS.update(original)


def test_get_platform_digest_manifest_without_platform_key(docker_service: DockerService) -> None:
    manifest = {
        "manifests": [
            {"digest": "sha256:no-platform"},
            {"digest": "sha256:also-no-platform"},
        ]
    }

    # No platform_info on any manifest → falls through to first manifest
    result = docker_service.get_platform_digest(manifest)

    assert result == "sha256:no-platform"


@pytest.mark.asyncio
async def test_get_docker_manifest_returns_parsed_json(docker_service: DockerService) -> None:
    payload = {"manifests": [{"digest": "sha256:abc"}]}
    with patch("server.docker.Utils.run_command", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = make_result(exit_code=0, stdout=json.dumps(payload))

        result = await docker_service.get_docker_manifest("ubuntu:latest")

    assert result == payload


@pytest.mark.asyncio
async def test_get_docker_manifest_raises_does_not_exist(docker_service: DockerService) -> None:
    with patch("server.docker.Utils.run_command", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = make_result(exit_code=1, stderr="manifest for ubuntu not found")

        with pytest.raises(DockerImageDoesNotExistError):
            await docker_service.get_docker_manifest("ubuntu:latest")


@pytest.mark.asyncio
async def test_get_docker_manifest_raises_auth_error(docker_service: DockerService) -> None:
    with patch("server.docker.Utils.run_command", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = make_result(exit_code=1, stderr="authorization failed")

        with pytest.raises(DockerImageAuthorizationError):
            await docker_service.get_docker_manifest("private/image")


@pytest.mark.asyncio
async def test_get_docker_manifest_raises_runtime_error_on_bad_exit(docker_service: DockerService) -> None:
    with patch("server.docker.Utils.run_command", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = make_result(exit_code=1, stderr="some other error")

        with pytest.raises(RuntimeError):
            await docker_service.get_docker_manifest("ubuntu:latest")


@pytest.mark.asyncio
async def test_get_docker_manifest_raises_app_error_on_stderr(docker_service: DockerService) -> None:
    payload = {"manifests": []}
    with patch("server.docker.Utils.run_command", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = make_result(exit_code=0, stdout=json.dumps(payload), stderr="some warning")

        with pytest.raises(AppError):
            await docker_service.get_docker_manifest("ubuntu:latest")


@pytest.mark.asyncio
async def test_get_docker_manifest_raises_app_error_on_invalid_json(docker_service: DockerService) -> None:
    with patch("server.docker.Utils.run_command", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = make_result(exit_code=0, stdout="not json {{{")

        with pytest.raises(AppError):
            await docker_service.get_docker_manifest("ubuntu:latest")


@pytest.mark.asyncio
async def test_get_docker_image_size_returns_total(docker_service: DockerService) -> None:
    platform_manifest = {"manifests": [{"digest": "sha256:platform"}]}
    layers_manifest = {"layers": [{"size": 1000}, {"size": 2000}]}

    async def fake_get_manifest(image: str) -> dict[str, Any]:
        if "@sha256:platform" in image:
            return layers_manifest
        return platform_manifest

    with patch.object(docker_service, "get_docker_manifest", side_effect=fake_get_manifest):
        size = await docker_service.get_docker_image_size("ubuntu:latest")

    assert size == 3000


@pytest.mark.asyncio
async def test_get_docker_image_size_no_digest(docker_service: DockerService) -> None:
    platform_manifest = {}  # no manifests → get_platform_digest returns None
    layers_manifest = {"layers": [{"size": 500}]}

    call_count = 0

    async def fake_get_manifest(image: str) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        return layers_manifest if call_count > 1 else platform_manifest

    with patch.object(docker_service, "get_docker_manifest", side_effect=fake_get_manifest):
        size = await docker_service.get_docker_image_size("ubuntu:latest")

    assert size == 500


@pytest.mark.asyncio
async def test_get_image_platforms_from_local_inspect(docker_service: DockerService) -> None:
    image_data = [{"Os": "linux", "Architecture": "amd64"}]
    with patch("server.docker.Utils.run_command_for_success", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = make_result2(stdout=json.dumps(image_data))

        result = await docker_service.get_image_platforms("ubuntu:latest")

    assert result == ["linux/amd64"]


@pytest.mark.asyncio
async def test_get_image_platforms_empty_inspect_falls_back_to_manifest(docker_service: DockerService) -> None:
    manifest = {
        "manifests": [
            {"platform": {"os": "linux", "architecture": "amd64", "variant": ""}},
            {"platform": {"os": "unknown", "architecture": "unknown", "variant": ""}},
        ]
    }
    with (
        patch("server.docker.Utils.run_command_for_success", new_callable=AsyncMock) as mock_run,
        patch.object(docker_service, "get_docker_manifest", new_callable=AsyncMock, return_value=manifest),
    ):
        mock_run.side_effect = RuntimeError("not local")

        result = await docker_service.get_image_platforms("ubuntu:latest")

    assert "linux/amd64" in result


@pytest.mark.asyncio
async def test_get_image_platforms_no_arch_returns_empty(docker_service: DockerService) -> None:
    image_data = [{"Os": "linux", "Architecture": ""}]
    with patch("server.docker.Utils.run_command_for_success", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = make_result2(stdout=json.dumps(image_data))

        result = await docker_service.get_image_platforms("ubuntu:latest")

    assert result == []


@pytest.mark.asyncio
async def test_get_image_platforms_manifest_no_manifests_v2_with_config(docker_service: DockerService) -> None:
    config_digest = "sha256:configdigest"
    main_manifest = {
        "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
        "config": {"digest": config_digest},
    }
    image_manifest = {"os": "linux", "architecture": "amd64", "variant": ""}

    call_count = 0

    async def fake_manifest(image: str) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        return image_manifest if call_count > 1 else main_manifest

    with (
        patch("server.docker.Utils.run_command_for_success", new_callable=AsyncMock) as mock_run,
        patch.object(docker_service, "get_docker_manifest", side_effect=fake_manifest),
    ):
        mock_run.side_effect = RuntimeError("not local")

        result = await docker_service.get_image_platforms("ubuntu:latest")

    assert "linux/amd64" in result


@pytest.mark.asyncio
async def test_get_image_platforms_manifest_no_manifests_v2_manifest_error_returns_empty(docker_service: DockerService) -> None:
    config_digest = "sha256:configdigest"
    main_manifest = {
        "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
        "config": {"digest": config_digest},
    }

    call_count = 0

    async def fake_manifest(image: str) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        if call_count > 1:
            raise RuntimeError("failed")
        return main_manifest

    with (
        patch("server.docker.Utils.run_command_for_success", new_callable=AsyncMock) as mock_run,
        patch.object(docker_service, "get_docker_manifest", side_effect=fake_manifest),
    ):
        mock_run.side_effect = RuntimeError("not local")

        result = await docker_service.get_image_platforms("ubuntu:latest")

    assert result == []


@pytest.mark.asyncio
async def test_get_image_platforms_fallback_no_manifests_no_v2_returns_empty(docker_service: DockerService) -> None:
    main_manifest: dict[str, Any] = {}  # no manifests, no mediaType

    with (
        patch("server.docker.Utils.run_command_for_success", new_callable=AsyncMock) as mock_run,
        patch.object(docker_service, "get_docker_manifest", new_callable=AsyncMock, return_value=main_manifest),
    ):
        mock_run.side_effect = RuntimeError("not local")

        result = await docker_service.get_image_platforms("ubuntu:latest")

    assert result == []


@pytest.mark.asyncio
async def test_get_image_warnings_platform_mismatch(docker_service: DockerService) -> None:
    with patch.object(docker_service, "get_image_platforms", new_callable=AsyncMock, return_value=["linux/arm64"]):
        warnings = await docker_service.get_image_warnings("ubuntu:latest")

    assert any("platform mismatch" in w for w in warnings)


@pytest.mark.asyncio
async def test_get_image_warnings_no_warnings_when_platform_matches(docker_service: DockerService) -> None:
    with patch.object(docker_service, "get_image_platforms", new_callable=AsyncMock, return_value=["linux/amd64"]):
        warnings = await docker_service.get_image_warnings("ubuntu:latest")

    assert warnings == []


@pytest.mark.asyncio
async def test_get_image_warnings_image_does_not_exist(docker_service: DockerService) -> None:
    with patch.object(docker_service, "get_image_platforms", new_callable=AsyncMock, side_effect=DockerImageDoesNotExistError("ubuntu")):
        warnings = await docker_service.get_image_warnings("ubuntu:latest")

    assert any("does not exist" in w for w in warnings)


@pytest.mark.asyncio
async def test_get_image_warnings_authorization_error(docker_service: DockerService) -> None:
    with patch.object(
        docker_service,
        "get_image_platforms",
        new_callable=AsyncMock,
        side_effect=DockerImageAuthorizationError("private/img"),
    ):
        warnings = await docker_service.get_image_warnings("private/img:latest")

    assert any("authorization failed" in w for w in warnings)


@pytest.mark.asyncio
async def test_is_docker_image_pulled_returns_false_on_non_404_docker_error(docker_service: DockerService) -> None:
    images = MagicMock()
    images.get = AsyncMock(side_effect=DockerError(status=500, message="server error"))
    cm, _ = _make_docker_mock(images)

    with patch("server.docker.Docker", return_value=cm):
        result = await docker_service.is_docker_image_pulled("ubuntu:latest")

    assert result is False


@pytest.mark.asyncio
async def test_docker_pull_yields_progress(docker_service: DockerService) -> None:
    chunks = [
        {"id": "layer1", "status": "Downloading", "progressDetail": {"current": 500}},
        {"id": "layer1", "status": "Downloading", "progressDetail": {"current": 1000}},
        {"id": "layer2", "status": "Pull complete"},
    ]

    async def fake_pull(*args: Any, **kwargs: Any):
        for chunk in chunks:
            yield chunk

    images = MagicMock()
    images.pull = fake_pull
    cm, _ = _make_docker_mock(images)
    with patch("server.docker.Docker", return_value=cm):
        percentages = [p async for p in docker_service.docker_pull("ubuntu:latest", image_size=2000)]

    assert len(percentages) == 2
    assert percentages[-1] >= percentages[0]


@pytest.mark.asyncio
async def test_docker_pull_uses_auth_for_known_registry(docker_service: DockerService) -> None:
    docker_service.auths = {"ghcr.io": "mytoken"}
    captured_kwargs: dict[str, Any] = {}

    async def fake_pull(*args: Any, **kwargs: Any):
        captured_kwargs.update(kwargs)
        return
        yield  # make it an async generator

    images = MagicMock()
    images.pull = fake_pull
    cm, _ = _make_docker_mock(images)
    with patch("server.docker.Docker", return_value=cm):
        _ = [p async for p in docker_service.docker_pull("ghcr.io/user/image:latest", image_size=1000)]

    assert captured_kwargs.get("auth") == "mytoken"


@pytest.mark.asyncio
async def test_generate_docker_compose_content_intel_gpu_with_gids(docker_service: DockerService) -> None:
    gpu = IntelGpuInfo(name="Intel Arc", vram=None, id=0)
    options = _opts(hardware=[gpu])
    mock_dev = MagicMock()
    mock_dev.stat.return_value = MagicMock(st_gid=44)
    mock_dev.__str__ = lambda s: "/dev/dri/renderD128"  # pyright: ignore[reportAttributeAccessIssue, reportUnknownLambdaType]

    with patch("server.docker.Path") as mock_path_cls:
        mock_dri = MagicMock()
        mock_dri.iterdir.return_value = iter([mock_dev])

        def path_side_effect(p: Any) -> Any:
            if p == "/dev/dri":
                return mock_dri
            return Path(p)

        mock_path_cls.side_effect = path_side_effect
        mock_path_cls.stat = Path.stat

        with patch("server.docker.Path.stat", return_value=MagicMock(st_gid=44)):
            result = await docker_service.generate_docker_compose_content(options, 12345)

    assert "/dev/dri:/dev/dri" in result["services"]["mymodel"]["devices"]  # pyright: ignore[reportTypedDictNotRequiredAccess]


@pytest.mark.asyncio
async def test_has_docker_compose_difference_exception_returns_true_none(docker_service: DockerService, tmp_path: Path) -> None:
    compose_file = tmp_path / "compose.yaml"
    compose_file.write_text("not: valid: yaml: [[[")  # will still parse, need generate to fail
    options = _opts()

    with patch.object(docker_service, "generate_docker_compose_content", new_callable=AsyncMock, side_effect=RuntimeError("boom")):
        has_diff, port = await docker_service.has_docker_compose_difference(compose_file, options)

    assert has_diff is True
    assert port is None


@pytest.mark.asyncio
async def test_get_existing_or_free_port_docker_no_ports_in_file(docker_service: DockerService, tmp_path: Path) -> None:
    compose_file = tmp_path / "compose.yaml"
    # Service has no ports key
    compose_yaml = yaml.dump({"services": {"mymodel": {"image": "ubuntu:latest"}}}, default_flow_style=False)
    compose_file.write_text(compose_yaml)
    docker_service.port_service.get_free_port.return_value = 55555  # pyright: ignore[reportAttributeAccessIssue]

    port = await docker_service.get_existing_or_free_port_docker(compose_file, options=_opts())

    assert port == 55555
    assert docker_service.port_service.get_free_port.call_count == 1  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.asyncio
async def test_install_and_run_docker_port_none_raises_app_error(docker_service: DockerService, tmp_path: Path) -> None:
    options = _opts()
    compose_file = tmp_path / "compose.yaml"

    with (
        patch.object(docker_service, "get_docker_compose_file_path", return_value=compose_file),
        patch.object(docker_service, "is_docker_compose_running", new_callable=AsyncMock, return_value=True),
        # has_difference=False, port=None → running no diff path, port stays None
        patch.object(docker_service, "has_docker_compose_difference", new_callable=AsyncMock, return_value=(False, None)),
        patch.object(docker_service, "is_docker_compose_healthy", new_callable=AsyncMock, return_value=True),
        pytest.raises(AppError),
    ):
        await docker_service.install_and_run_docker(options)


@pytest.mark.asyncio
async def test_uninstall_docker_file_not_present_skips_unlink(
    docker_service: DockerService, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    options = _opts()  # container_name=None: no compose file and no way to identify the container
    compose_file = tmp_path / "nonexistent.yaml"  # does not exist
    with (
        patch.object(docker_service, "get_docker_compose_file_path", return_value=compose_file),
        patch("server.docker.Utils.run_command", new_callable=AsyncMock) as mock_run,
        patch.object(docker_service, "_stop_and_remove_container", new_callable=AsyncMock) as mock_stop_and_remove,
        caplog.at_level("WARNING", logger="uvicorn.error"),
    ):
        mock_run.return_value = make_result()

        await docker_service.uninstall_docker(options)

    assert mock_run.call_count == 0
    mock_stop_and_remove.assert_not_called()
    assert not compose_file.exists()
    assert any(record.levelname == "WARNING" and "mymodel" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_uninstall_docker_falls_back_to_container_when_file_missing(docker_service: DockerService, tmp_path: Path) -> None:
    options = _opts(container_name="my-container")
    compose_file = tmp_path / "nonexistent.yaml"  # does not exist
    with (
        patch.object(docker_service, "get_docker_compose_file_path", return_value=compose_file),
        patch("server.docker.Utils.run_command", new_callable=AsyncMock) as mock_run,
        patch.object(docker_service, "_stop_and_remove_container", new_callable=AsyncMock) as mock_stop_and_remove,
    ):
        await docker_service.uninstall_docker(options)

    assert mock_run.call_count == 0
    mock_stop_and_remove.assert_called_once_with("my-container")


@pytest.mark.asyncio
async def test_uninstall_docker_tolerates_docker_error_from_fallback(
    docker_service: DockerService, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    options = _opts(container_name="my-container")
    compose_file = tmp_path / "nonexistent.yaml"  # does not exist

    with (
        patch.object(docker_service, "get_docker_compose_file_path", return_value=compose_file),
        patch.object(
            docker_service,
            "_stop_and_remove_container",
            new_callable=AsyncMock,
            side_effect=DockerError(status=500, message="daemon unreachable"),
        ),
        caplog.at_level("WARNING", logger="uvicorn.error"),
    ):
        await docker_service.uninstall_docker(options)  # must not raise: uninstall is best-effort

    assert any(record.levelname == "WARNING" and "my-container" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_uninstall_docker_with_compose_prefix_calls_rmdir(docker_service: DockerService, tmp_path: Path) -> None:
    docker_service.config.compose_prefix = "pf-"
    options = _opts()
    service_dir = tmp_path / "pf-mymodel"
    service_dir.mkdir()
    compose_file = service_dir / "compose.yaml"
    compose_file.write_text("services: {}")
    with (
        patch.object(docker_service, "get_docker_compose_file_path", return_value=compose_file),
        patch("server.docker.Utils.run_command", new_callable=AsyncMock) as mock_run,
    ):
        mock_run.return_value = make_result()

        await docker_service.uninstall_docker(options)

    assert not compose_file.exists()
    # parent dir should have been removed (it was empty after unlink)
    assert not service_dir.exists()


# _diagnose_gpu_error


def test_diagnose_gpu_error_toolkit_pattern_returns_toolkit_message() -> None:
    result = _diagnose_gpu_error("could not select device driver for this platform")
    assert result is not None
    assert "toolkit" in result.lower()


def test_diagnose_gpu_error_driver_pattern_returns_driver_message() -> None:
    result = _diagnose_gpu_error("failed to initialize nvml: unknown error")
    assert result is not None
    assert "driver" in result.lower()


def test_diagnose_gpu_error_no_match_returns_none() -> None:
    assert _diagnose_gpu_error("container started successfully") is None


# _is_container_name_conflict


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('Conflict. The container name "/ollama" is already in use by container "abc".', True),
        ("the container name is already in use", True),
        ("THE CONTAINER NAME IS ALREADY IN USE", True),
        ("container started successfully", False),
        ("port is already allocated", False),
        ("failed to bind host port 0.0.0.0:8080/tcp: address already in use", False),
        ("Error starting userland proxy: listen tcp4 0.0.0.0:8080: bind: address already in use", False),
    ],
)
def test_is_container_name_conflict_matches_collision_messages(text: str, expected: bool):
    result = _is_container_name_conflict(text)

    assert result is expected


# _matches_for_adoption and its helpers


def _adoption_inspect(
    image: str = "ubuntu:latest",
    image_id: str = DEFAULT_ADOPTION_IMAGE_ID,
    status: str = "running",
    health: str | None = None,
    env: list[str] | None = None,
    mounts: list[dict[str, Any]] | None = None,
    ports: dict[str, Any] | None = None,
    cmd: list[str] | None = None,
    entrypoint: list[str] | None = None,
    user: str = "",
    networks: dict[str, Any] | None = None,
    host_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    state: dict[str, Any] = {"Status": status}
    if health is not None:
        state["Health"] = {"Status": health}
    return {
        "Image": image_id,
        "Config": {"Image": image, "Env": env or [], "Cmd": cmd, "Entrypoint": entrypoint, "User": user},
        "State": state,
        "Mounts": mounts or [],
        "NetworkSettings": {"Ports": ports or {}, "Networks": networks or {}},
        "HostConfig": host_config or {},
    }


def _nvidia_device_request(*device_ids: str) -> dict[str, Any]:
    return {"Driver": "nvidia", "Count": -1, "DeviceIDs": list(device_ids), "Capabilities": [["gpu"]]}


def _intel_dri_device() -> dict[str, Any]:
    return {"PathOnHost": "/dev/dri", "PathInContainer": "/dev/dri", "CgroupPermissions": "rwm"}


def _bind_mount(source: str, destination: str, rw: bool = True) -> dict[str, Any]:
    return {"Type": "bind", "Source": source, "Destination": destination, "RW": rw}


def test_matches_for_adoption_success_no_healthcheck() -> None:
    options = _opts(image="ubuntu:latest", image_port=8080, env_vars={"FOO": "bar"}, volumes=["/host/data:/data"])
    inspect = _adoption_inspect(
        image="ubuntu:latest",
        env=["FOO=bar", "PATH=/usr/bin"],
        mounts=[_bind_mount("/host/data", "/data")],
        ports={"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "44444"}]},
    )

    assert _matches_for_adoption(inspect, options, DEFAULT_ADOPTION_IMAGE_ID) == 44444


def test_matches_for_adoption_success_with_healthcheck() -> None:
    options = _opts(image_port=8080, healthcheck={"test": "CMD curl -f http://localhost/health"})
    inspect = _adoption_inspect(health="healthy", ports={"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "12345"}]})

    assert _matches_for_adoption(inspect, options, DEFAULT_ADOPTION_IMAGE_ID) == 12345


def test_matches_for_adoption_success_subnet_mode_skips_port_check() -> None:
    options = _opts(image_port=8080, subnet="my-net")
    inspect = _adoption_inspect(ports={}, networks={"my-net": {}})

    assert _matches_for_adoption(inspect, options, DEFAULT_ADOPTION_IMAGE_ID) == -1


def test_matches_for_adoption_subnet_not_attached_fails() -> None:
    options = _opts(image_port=8080, subnet="my-net")
    inspect = _adoption_inspect(ports={}, networks={"some-other-net": {}})

    assert _matches_for_adoption(inspect, options, DEFAULT_ADOPTION_IMAGE_ID) is None


def test_container_attached_to_network_true_when_present() -> None:
    assert _container_attached_to_network({"my-net": {}}, "my-net") is True


def test_container_attached_to_network_false_when_absent_or_missing() -> None:
    assert _container_attached_to_network({"other-net": {}}, "my-net") is False
    assert _container_attached_to_network(None, "my-net") is False
    assert _container_attached_to_network({}, "my-net") is False


def test_matches_for_adoption_same_tag_but_different_resolved_image_id_fails() -> None:
    """A mutable tag like :latest can be repointed at a newer pull - matching on the tag name alone

    would wrongly adopt a stale orphan still running the old image content. The live container's
    actual image ID must match what the tag currently resolves to.
    """
    options = _opts(image="ubuntu:latest", image_port=8080)
    stale_image_id = "sha256:" + "b" * 64
    inspect = _adoption_inspect(
        image="ubuntu:latest", image_id=stale_image_id, ports={"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "44444"}]}
    )

    assert _matches_for_adoption(inspect, options, DEFAULT_ADOPTION_IMAGE_ID) is None


def test_matches_for_adoption_different_tag_but_same_resolved_image_id_matches() -> None:
    """Image ID equality is what matters, not the name:tag string - e.g. the same image retagged."""
    options = _opts(image="ubuntu:latest", image_port=8080)
    inspect = _adoption_inspect(
        image="ubuntu:22.04", image_id=DEFAULT_ADOPTION_IMAGE_ID, ports={"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "44444"}]}
    )

    assert _matches_for_adoption(inspect, options, DEFAULT_ADOPTION_IMAGE_ID) == 44444


def test_matches_for_adoption_not_running_fails() -> None:
    options = _opts(image_port=8080)
    inspect = _adoption_inspect(status="exited", ports={"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "44444"}]})

    assert _matches_for_adoption(inspect, options, DEFAULT_ADOPTION_IMAGE_ID) is None


def test_matches_for_adoption_healthcheck_configured_but_unhealthy_fails() -> None:
    options = _opts(image_port=8080, healthcheck={"test": "CMD curl -f http://localhost/health"})
    inspect = _adoption_inspect(health="unhealthy", ports={"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "44444"}]})

    assert _matches_for_adoption(inspect, options, DEFAULT_ADOPTION_IMAGE_ID) is None


def test_matches_for_adoption_healthcheck_configured_but_health_missing_fails() -> None:
    options = _opts(image_port=8080, healthcheck={"test": "CMD curl -f http://localhost/health"})
    inspect = _adoption_inspect(ports={"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "44444"}]})

    assert _matches_for_adoption(inspect, options, DEFAULT_ADOPTION_IMAGE_ID) is None


def test_matches_for_adoption_port_not_published_fails() -> None:
    options = _opts(image_port=8080)
    inspect = _adoption_inspect(ports={})

    assert _matches_for_adoption(inspect, options, DEFAULT_ADOPTION_IMAGE_ID) is None


def test_matches_for_adoption_port_published_non_loopback_fails() -> None:
    """A container exposed to the whole network (0.0.0.0) isn't a match for a loopback-only install."""
    options = _opts(image_port=8080)
    inspect = _adoption_inspect(ports={"8080/tcp": [{"HostIp": "0.0.0.0", "HostPort": "44444"}]})

    assert _matches_for_adoption(inspect, options, DEFAULT_ADOPTION_IMAGE_ID) is None


def test_matches_for_adoption_env_var_missing_fails() -> None:
    options = _opts(image_port=8080, env_vars={"FOO": "bar"})
    inspect = _adoption_inspect(env=["OTHER=1"], ports={"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "44444"}]})

    assert _matches_for_adoption(inspect, options, DEFAULT_ADOPTION_IMAGE_ID) is None


def test_matches_for_adoption_env_var_value_mismatch_fails() -> None:
    options = _opts(image_port=8080, env_vars={"FOO": "bar"})
    inspect = _adoption_inspect(env=["FOO=other"], ports={"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "44444"}]})

    assert _matches_for_adoption(inspect, options, DEFAULT_ADOPTION_IMAGE_ID) is None


def test_matches_for_adoption_extra_container_env_is_ignored() -> None:
    options = _opts(image_port=8080, env_vars={"FOO": "bar"})
    inspect = _adoption_inspect(env=["FOO=bar", "IMAGE_DEFAULT=1"], ports={"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "44444"}]})

    assert _matches_for_adoption(inspect, options, DEFAULT_ADOPTION_IMAGE_ID) == 44444


def test_matches_for_adoption_volume_missing_fails() -> None:
    options = _opts(image_port=8080, volumes=["/host/data:/data"])
    inspect = _adoption_inspect(mounts=[], ports={"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "44444"}]})

    assert _matches_for_adoption(inspect, options, DEFAULT_ADOPTION_IMAGE_ID) is None


def test_matches_for_adoption_volume_source_mismatch_fails() -> None:
    """A container mounting a different host directory at the same container path must not match.

    Guards against a regression that matched only by container-path + mode (ignoring the host path),
    which would let a container backed by an entirely different host directory get adopted.
    """
    options = _opts(image_port=8080, volumes=["/host/data:/data"])
    inspect = _adoption_inspect(
        mounts=[_bind_mount("/some/other/host/dir", "/data")], ports={"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "44444"}]}
    )

    assert _matches_for_adoption(inspect, options, DEFAULT_ADOPTION_IMAGE_ID) is None


def test_matches_for_adoption_volume_mode_mismatch_fails() -> None:
    options = _opts(image_port=8080, volumes=["/host/data:/data:ro"])
    inspect = _adoption_inspect(
        mounts=[_bind_mount("/host/data", "/data", rw=True)], ports={"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "44444"}]}
    )

    assert _matches_for_adoption(inspect, options, DEFAULT_ADOPTION_IMAGE_ID) is None


def test_matches_for_adoption_string_command_matches() -> None:
    options = _opts(image_port=8080, command="--host 0.0.0.0 --port 8080")
    inspect = _adoption_inspect(
        cmd=["--host", "0.0.0.0", "--port", "8080"], ports={"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "44444"}]}
    )

    assert _matches_for_adoption(inspect, options, DEFAULT_ADOPTION_IMAGE_ID) == 44444


def test_matches_for_adoption_string_command_mismatch_fails() -> None:
    options = _opts(image_port=8080, command="--host 0.0.0.0 --port 8080 --max-model-len 4096")
    inspect = _adoption_inspect(
        cmd=["--host", "0.0.0.0", "--port", "8080"], ports={"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "44444"}]}
    )

    assert _matches_for_adoption(inspect, options, DEFAULT_ADOPTION_IMAGE_ID) is None


def test_matches_for_adoption_list_command_matches() -> None:
    options = _opts(image_port=8080, command=["-c", "exec server"])
    inspect = _adoption_inspect(cmd=["-c", "exec server"], ports={"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "44444"}]})

    assert _matches_for_adoption(inspect, options, DEFAULT_ADOPTION_IMAGE_ID) == 44444


def test_matches_for_adoption_list_command_mismatch_fails() -> None:
    options = _opts(image_port=8080, command=["-c", "exec server"])
    inspect = _adoption_inspect(cmd=["-c", "exec other"], ports={"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "44444"}]})

    assert _matches_for_adoption(inspect, options, DEFAULT_ADOPTION_IMAGE_ID) is None


def test_matches_for_adoption_no_command_configured_ignores_live_cmd() -> None:
    options = _opts(image_port=8080)
    inspect = _adoption_inspect(cmd=["anything", "at", "all"], ports={"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "44444"}]})

    assert _matches_for_adoption(inspect, options, DEFAULT_ADOPTION_IMAGE_ID) == 44444


def test_matches_for_adoption_entrypoint_mismatch_fails() -> None:
    options = _opts(image_port=8080, entrypoint="/entrypoint.sh")
    inspect = _adoption_inspect(entrypoint=["/other-entrypoint.sh"], ports={"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "44444"}]})

    assert _matches_for_adoption(inspect, options, DEFAULT_ADOPTION_IMAGE_ID) is None


def test_matches_for_adoption_entrypoint_matches() -> None:
    options = _opts(image_port=8080, entrypoint="/entrypoint.sh --flag")
    inspect = _adoption_inspect(entrypoint=["/entrypoint.sh", "--flag"], ports={"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "44444"}]})

    assert _matches_for_adoption(inspect, options, DEFAULT_ADOPTION_IMAGE_ID) == 44444


def test_matches_for_adoption_no_entrypoint_configured_ignores_live_entrypoint() -> None:
    options = _opts(image_port=8080)
    inspect = _adoption_inspect(entrypoint=["anything"], ports={"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "44444"}]})

    assert _matches_for_adoption(inspect, options, DEFAULT_ADOPTION_IMAGE_ID) == 44444


def test_matches_for_adoption_user_mismatch_fails() -> None:
    options = _opts(image_port=8080, user="app")
    inspect = _adoption_inspect(user="root", ports={"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "44444"}]})

    assert _matches_for_adoption(inspect, options, DEFAULT_ADOPTION_IMAGE_ID) is None


def test_matches_for_adoption_user_matches() -> None:
    options = _opts(image_port=8080, user="app")
    inspect = _adoption_inspect(user="app", ports={"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "44444"}]})

    assert _matches_for_adoption(inspect, options, DEFAULT_ADOPTION_IMAGE_ID) == 44444


def test_matches_for_adoption_no_user_configured_ignores_live_user() -> None:
    options = _opts(image_port=8080)
    inspect = _adoption_inspect(user="root", ports={"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "44444"}]})

    assert _matches_for_adoption(inspect, options, DEFAULT_ADOPTION_IMAGE_ID) == 44444


def test_matches_for_adoption_shm_size_mismatch_fails() -> None:
    options = _opts(image_port=8080, shm_size="16gb")
    inspect = _adoption_inspect(host_config={"ShmSize": 8 * 1024**3}, ports={"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "44444"}]})

    assert _matches_for_adoption(inspect, options, DEFAULT_ADOPTION_IMAGE_ID) is None


def test_matches_for_adoption_shm_size_matches() -> None:
    options = _opts(image_port=8080, shm_size="16gb")
    inspect = _adoption_inspect(host_config={"ShmSize": 16 * 1024**3}, ports={"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "44444"}]})

    assert _matches_for_adoption(inspect, options, DEFAULT_ADOPTION_IMAGE_ID) == 44444


def test_matches_for_adoption_no_shm_size_configured_ignores_live_shm_size() -> None:
    options = _opts(image_port=8080)
    inspect = _adoption_inspect(host_config={"ShmSize": 1}, ports={"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "44444"}]})

    assert _matches_for_adoption(inspect, options, DEFAULT_ADOPTION_IMAGE_ID) == 44444


def test_matches_for_adoption_restart_policy_mismatch_fails() -> None:
    options = _opts(image_port=8080, restart="unless-stopped")
    inspect = _adoption_inspect(
        host_config={"RestartPolicy": {"Name": "no"}}, ports={"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "44444"}]}
    )

    assert _matches_for_adoption(inspect, options, DEFAULT_ADOPTION_IMAGE_ID) is None


def test_matches_for_adoption_restart_policy_matches() -> None:
    options = _opts(image_port=8080, restart="unless-stopped")
    inspect = _adoption_inspect(
        host_config={"RestartPolicy": {"Name": "unless-stopped"}}, ports={"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "44444"}]}
    )

    assert _matches_for_adoption(inspect, options, DEFAULT_ADOPTION_IMAGE_ID) == 44444


def test_matches_for_adoption_no_restart_policy_configured_ignores_live_restart_policy() -> None:
    options = _opts(image_port=8080)
    inspect = _adoption_inspect(
        host_config={"RestartPolicy": {"Name": "no"}}, ports={"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "44444"}]}
    )

    assert _matches_for_adoption(inspect, options, DEFAULT_ADOPTION_IMAGE_ID) == 44444


def test_matches_for_adoption_gpu_count_based_request_fails() -> None:
    """A "--gpus all"-style reservation has no explicit DeviceIDs, so it can't be proven not to hold every GPU."""
    options = _opts(image_port=8080)
    all_gpus_request = {"Driver": "nvidia", "Count": -1, "DeviceIDs": None, "Capabilities": [["gpu"]]}
    inspect = _adoption_inspect(
        host_config={"DeviceRequests": [all_gpus_request]}, ports={"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "44444"}]}
    )

    assert _matches_for_adoption(inspect, options, DEFAULT_ADOPTION_IMAGE_ID) is None


def test_container_gpu_matches_count_based_request_without_device_ids_fails() -> None:
    all_gpus_request = {"Driver": "nvidia", "Count": -1, "DeviceIDs": None, "Capabilities": [["gpu"]]}
    assert _container_gpu_matches({"DeviceRequests": [all_gpus_request]}, None) is False
    assert _container_gpu_matches({"DeviceRequests": [all_gpus_request]}, [NvidiaGpuInfo("RTX", "24 GB", 0)]) is False


def test_container_user_matches_no_configured_user_always_matches() -> None:
    assert _container_user_matches({"User": "root"}, None) is True
    assert _container_user_matches({"User": ""}, None) is True


def test_container_user_matches_compares_exact_value() -> None:
    assert _container_user_matches({"User": "app"}, "app") is True
    assert _container_user_matches({"User": "root"}, "app") is False


def test_container_shm_size_matches_no_configured_value_always_matches() -> None:
    assert _container_shm_size_matches({"ShmSize": 1}, None) is True


def test_container_shm_size_matches_unparseable_configured_value_fails() -> None:
    assert _container_shm_size_matches({"ShmSize": 1024}, "not-a-size") is False


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1024", 1024),
        ("16gb", 16 * 1024**3),
        ("512m", 512 * 1024**2),
        ("2kb", 2 * 1024),
        ("1.5g", int(1.5 * 1024**3)),
        ("not-a-size", None),
        ("16xb", None),
        ("", None),
    ],
)
def test_parse_byte_size(value: str, expected: int | None) -> None:
    assert _parse_byte_size(value) == expected


def test_container_restart_policy_matches_no_configured_value_always_matches() -> None:
    assert _container_restart_policy_matches({"RestartPolicy": {"Name": "always"}}, None) is True


def test_container_restart_policy_matches_ignores_retry_count_suffix() -> None:
    assert _container_restart_policy_matches({"RestartPolicy": {"Name": "on-failure"}}, "on-failure:5") is True


def test_matches_for_adoption_gpu_device_missing_fails() -> None:
    """A container adopted without a matching Nvidia device reservation could already be holding VRAM elsewhere."""
    options = _opts(image_port=8080, hardware=[NvidiaGpuInfo("RTX", "24 GB", 0)])
    inspect = _adoption_inspect(host_config={}, ports={"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "44444"}]})

    assert _matches_for_adoption(inspect, options, DEFAULT_ADOPTION_IMAGE_ID) is None


def test_matches_for_adoption_gpu_device_id_mismatch_fails() -> None:
    options = _opts(image_port=8080, hardware=[NvidiaGpuInfo("RTX", "24 GB", 0)])
    inspect = _adoption_inspect(
        host_config={"DeviceRequests": [_nvidia_device_request("1")]}, ports={"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "44444"}]}
    )

    assert _matches_for_adoption(inspect, options, DEFAULT_ADOPTION_IMAGE_ID) is None


def test_matches_for_adoption_gpu_device_matches() -> None:
    options = _opts(image_port=8080, hardware=[NvidiaGpuInfo("RTX", "24 GB", 0)])
    inspect = _adoption_inspect(
        host_config={"DeviceRequests": [_nvidia_device_request("0")]}, ports={"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "44444"}]}
    )

    assert _matches_for_adoption(inspect, options, DEFAULT_ADOPTION_IMAGE_ID) == 44444


def test_matches_for_adoption_gpu_unexpectedly_attached_fails() -> None:
    """No GPU configured but the live container has one reserved - the reverse mismatch must also fail."""
    options = _opts(image_port=8080)
    inspect = _adoption_inspect(
        host_config={"DeviceRequests": [_nvidia_device_request("0")]}, ports={"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "44444"}]}
    )

    assert _matches_for_adoption(inspect, options, DEFAULT_ADOPTION_IMAGE_ID) is None


def test_matches_for_adoption_intel_gpu_missing_fails() -> None:
    options = _opts(image_port=8080, hardware=[IntelGpuInfo("Intel GPU", None, 0)])
    inspect = _adoption_inspect(host_config={}, ports={"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "44444"}]})

    assert _matches_for_adoption(inspect, options, DEFAULT_ADOPTION_IMAGE_ID) is None


def test_matches_for_adoption_intel_gpu_matches() -> None:
    options = _opts(image_port=8080, hardware=[IntelGpuInfo("Intel GPU", None, 0)])
    inspect = _adoption_inspect(
        host_config={"Devices": [_intel_dri_device()]}, ports={"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "44444"}]}
    )

    assert _matches_for_adoption(inspect, options, DEFAULT_ADOPTION_IMAGE_ID) == 44444


def test_container_gpu_matches_no_hardware_configured_requires_no_live_devices() -> None:
    assert _container_gpu_matches({}, None) is True
    assert _container_gpu_matches({"DeviceRequests": [_nvidia_device_request("0")]}, None) is False


def test_container_gpu_matches_nvidia_ids_compared_as_set_not_order() -> None:
    hardware = [NvidiaGpuInfo("A", None, 1), NvidiaGpuInfo("B", None, 0)]
    assert _container_gpu_matches({"DeviceRequests": [_nvidia_device_request("0", "1")]}, hardware) is True


def test_container_command_matches_no_configured_command_always_matches() -> None:
    assert _container_command_matches(["whatever"], None) is True
    assert _container_command_matches(None, None) is True


def test_container_command_matches_splits_string_command_with_shell_word_rules() -> None:
    assert _container_command_matches(["--host", "0.0.0.0"], "--host 0.0.0.0") is True
    assert _container_command_matches(["--host", "0.0.0.0"], "--host  0.0.0.0") is True


def test_container_command_matches_compares_list_command_as_is() -> None:
    assert _container_command_matches(["-c", "a && b"], ["-c", "a && b"]) is True
    assert _container_command_matches(["-c", "a && b"], ["-c", "a"]) is False


def test_container_command_matches_malformed_string_command_fails_safely() -> None:
    assert _container_command_matches(["anything"], "unterminated 'quote") is False


def test_extract_published_host_port_returns_none_when_unpublished() -> None:
    assert _extract_published_host_port({}, 8080) is None
    assert _extract_published_host_port(None, 8080) is None
    assert _extract_published_host_port({"8080/tcp": None}, 8080) is None


def test_extract_published_host_port_returns_host_port() -> None:
    assert _extract_published_host_port({"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "9999"}]}, 8080) == 9999


def test_extract_published_host_port_returns_none_for_non_numeric_host_port() -> None:
    assert _extract_published_host_port({"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "not-a-port"}]}, 8080) is None


def test_extract_published_host_port_ignores_non_loopback_binding() -> None:
    """A container published to 0.0.0.0 (reachable network-wide) isn't the same config as 127.0.0.1-only."""
    assert _extract_published_host_port({"8080/tcp": [{"HostIp": "0.0.0.0", "HostPort": "9999"}]}, 8080) is None


def test_extract_published_host_port_finds_loopback_binding_among_several() -> None:
    bindings = [{"HostIp": "0.0.0.0", "HostPort": "9999"}, {"HostIp": "127.0.0.1", "HostPort": "8888"}]
    assert _extract_published_host_port({"8080/tcp": bindings}, 8080) == 8888


def test_extract_published_host_port_keeps_looking_past_invalid_loopback_binding() -> None:
    """A malformed/empty loopback binding earlier in the list must not short-circuit the search."""
    bindings = [{"HostIp": "127.0.0.1", "HostPort": ""}, {"HostIp": "127.0.0.1", "HostPort": "8888"}]
    assert _extract_published_host_port({"8080/tcp": bindings}, 8080) == 8888

    bindings = [{"HostIp": "127.0.0.1", "HostPort": "not-a-port"}, {"HostIp": "127.0.0.1", "HostPort": "8888"}]
    assert _extract_published_host_port({"8080/tcp": bindings}, 8080) == 8888


def test_container_env_matches_ignores_extra_entries() -> None:
    assert _container_env_matches(["FOO=bar", "EXTRA=1"], {"FOO": "bar"}) is True


def test_container_env_matches_fails_on_missing_key() -> None:
    assert _container_env_matches(["OTHER=1"], {"FOO": "bar"}) is False


def test_container_env_matches_ignores_malformed_entries_without_equals_sign() -> None:
    assert _container_env_matches(["MALFORMED", "FOO=bar"], {"FOO": "bar"}) is True


def test_parse_bind_volume_parses_host_container_and_mode() -> None:
    assert _parse_bind_volume("/host:/container") == ("/host", "/container", True)
    assert _parse_bind_volume("/host:/container:ro") == ("/host", "/container", False)
    assert _parse_bind_volume("/host:/container:rw") == ("/host", "/container", True)


def test_parse_bind_volume_malformed_string_returns_none() -> None:
    assert _parse_bind_volume("/just/a/path") is None


def test_container_volumes_match_ignores_non_bind_mounts() -> None:
    """A named/anonymous volume mount must not satisfy a configured bind-mount requirement, even with matching paths."""
    mounts = [{"Type": "volume", "Source": "somevolume", "Destination": "/data"}]
    assert _container_volumes_match(mounts, ["somevolume:/data"]) is False


def test_container_volumes_match_matches_genuine_bind_mount() -> None:
    """Sibling of the above: the same Source/Destination pair on a real bind mount does satisfy the requirement."""
    mounts = [_bind_mount("somevolume", "/data")]
    assert _container_volumes_match(mounts, ["somevolume:/data"]) is True


def test_container_volumes_match_fails_on_malformed_configured_volume() -> None:
    assert _container_volumes_match([], ["/just/a/path"]) is False


# _extract_error_excerpt


def test_extract_error_excerpt_picks_error_lines_out_of_noisy_logs() -> None:
    logs = "\n".join(
        [
            "Starting container...",
            "Pulling layer 1/4",
            "Pulling layer 2/4",
            'Traceback (most recent call last): raise RuntimeError("model weights not found")',
            "Pulling layer 3/4",
        ]
    )

    result = _extract_error_excerpt(logs)

    assert "model weights not found" in result
    assert "Pulling layer" not in result


def test_extract_error_excerpt_falls_back_to_tail_snippet_when_no_error_line() -> None:
    logs = "\n".join(f"Pulling layer {i}/500" for i in range(500))

    result = _extract_error_excerpt(logs)

    assert result.endswith(logs[-500:])


def test_extract_error_excerpt_empty_logs_returns_empty() -> None:
    assert _extract_error_excerpt("") == ""


# start_docker_compose


@pytest.mark.asyncio
async def test_start_docker_compose_raises_docker_compose_start_error_on_nonzero_exit(
    docker_service: DockerService, tmp_path: Path
) -> None:
    compose_file = tmp_path / "compose.yaml"

    with patch("server.docker.Utils.run_command", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = make_result(exit_code=1, stdout="out", stderr="err")
        with pytest.raises(DockerComposeStartError):
            await docker_service.start_docker_compose(compose_file)


# _inspect_container_for_adoption


@pytest.mark.asyncio
async def test_inspect_container_for_adoption_missing_container_is_quiet(
    docker_service: DockerService, caplog: pytest.LogCaptureFixture
) -> None:
    """A 404 (container doesn't exist) is the expected case - no warning-level log."""
    instance = MagicMock()
    instance.containers = MagicMock()
    container = MagicMock()
    container.show = AsyncMock(side_effect=DockerError(status=404, message="no such container"))
    instance.containers.container = MagicMock(return_value=container)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=instance)
    cm.__aexit__ = AsyncMock(return_value=False)

    with patch("server.docker.Docker", return_value=cm), caplog.at_level("WARNING", logger="uvicorn.error"):
        result = await docker_service._inspect_container_for_adoption("mymodel")  # type: ignore[reportPrivateUsage]

    assert result is None
    assert not any(record.levelname == "WARNING" for record in caplog.records)


@pytest.mark.asyncio
async def test_inspect_container_for_adoption_non_404_docker_error_logs_warning(
    docker_service: DockerService, caplog: pytest.LogCaptureFixture
) -> None:
    instance = MagicMock()
    instance.containers = MagicMock()
    container = MagicMock()
    container.show = AsyncMock(side_effect=DockerError(status=500, message="internal server error"))
    instance.containers.container = MagicMock(return_value=container)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=instance)
    cm.__aexit__ = AsyncMock(return_value=False)

    with patch("server.docker.Docker", return_value=cm), caplog.at_level("WARNING", logger="uvicorn.error"):
        result = await docker_service._inspect_container_for_adoption("mymodel")  # type: ignore[reportPrivateUsage]

    assert result is None
    assert any(record.levelname == "WARNING" and "mymodel" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_inspect_container_for_adoption_unexpected_exception_logs_warning_and_returns_none(
    docker_service: DockerService, caplog: pytest.LogCaptureFixture
) -> None:
    """A non-DockerError failure (e.g. daemon unreachable) still falls back gracefully, but is logged."""
    instance = MagicMock()
    instance.containers = MagicMock()
    container = MagicMock()
    container.show = AsyncMock(side_effect=ConnectionError("connection refused"))
    instance.containers.container = MagicMock(return_value=container)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=instance)
    cm.__aexit__ = AsyncMock(return_value=False)

    with patch("server.docker.Docker", return_value=cm), caplog.at_level("WARNING", logger="uvicorn.error"):
        result = await docker_service._inspect_container_for_adoption("mymodel")  # type: ignore[reportPrivateUsage]

    assert result is None
    assert any(record.levelname == "WARNING" and "mymodel" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_inspect_container_for_adoption_non_dict_response_logs_warning_and_returns_none(
    docker_service: DockerService, caplog: pytest.LogCaptureFixture
) -> None:
    """aiodocker's return type hint isn't runtime-enforced - a malformed response must not crash adoption."""
    instance = MagicMock()
    instance.containers = MagicMock()
    container = MagicMock()
    container.show = AsyncMock(return_value=["not", "a", "dict"])
    instance.containers.container = MagicMock(return_value=container)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=instance)
    cm.__aexit__ = AsyncMock(return_value=False)

    with patch("server.docker.Docker", return_value=cm), caplog.at_level("WARNING", logger="uvicorn.error"):
        result = await docker_service._inspect_container_for_adoption("mymodel")  # type: ignore[reportPrivateUsage]

    assert result is None
    assert any(record.levelname == "WARNING" and "mymodel" in record.message for record in caplog.records)


# _resolve_image_id_for_adoption


@pytest.mark.asyncio
async def test_resolve_image_id_for_adoption_returns_id(docker_service: DockerService) -> None:
    cm, instance = _make_docker_mock()
    instance.images = MagicMock()
    instance.images.inspect = AsyncMock(return_value={"Id": DEFAULT_ADOPTION_IMAGE_ID})

    with patch("server.docker.Docker", return_value=cm):
        result = await docker_service._resolve_image_id_for_adoption("ubuntu:latest")  # type: ignore[reportPrivateUsage]

    assert result == DEFAULT_ADOPTION_IMAGE_ID


@pytest.mark.asyncio
async def test_resolve_image_id_for_adoption_missing_image_is_quiet(
    docker_service: DockerService, caplog: pytest.LogCaptureFixture
) -> None:
    """A 404 (image not pulled locally) is the expected case - no warning-level log."""
    cm, instance = _make_docker_mock()
    instance.images = MagicMock()
    instance.images.inspect = AsyncMock(side_effect=DockerError(status=404, message="no such image"))

    with patch("server.docker.Docker", return_value=cm), caplog.at_level("WARNING", logger="uvicorn.error"):
        result = await docker_service._resolve_image_id_for_adoption("ubuntu:latest")  # type: ignore[reportPrivateUsage]

    assert result is None
    assert not any(record.levelname == "WARNING" for record in caplog.records)


@pytest.mark.asyncio
async def test_resolve_image_id_for_adoption_non_404_docker_error_logs_warning(
    docker_service: DockerService, caplog: pytest.LogCaptureFixture
) -> None:
    cm, instance = _make_docker_mock()
    instance.images = MagicMock()
    instance.images.inspect = AsyncMock(side_effect=DockerError(status=500, message="internal server error"))

    with patch("server.docker.Docker", return_value=cm), caplog.at_level("WARNING", logger="uvicorn.error"):
        result = await docker_service._resolve_image_id_for_adoption("ubuntu:latest")  # type: ignore[reportPrivateUsage]

    assert result is None
    assert any(record.levelname == "WARNING" and "ubuntu:latest" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_resolve_image_id_for_adoption_unexpected_exception_logs_warning_and_returns_none(
    docker_service: DockerService, caplog: pytest.LogCaptureFixture
) -> None:
    cm, instance = _make_docker_mock()
    instance.images = MagicMock()
    instance.images.inspect = AsyncMock(side_effect=ConnectionError("connection refused"))

    with patch("server.docker.Docker", return_value=cm), caplog.at_level("WARNING", logger="uvicorn.error"):
        result = await docker_service._resolve_image_id_for_adoption("ubuntu:latest")  # type: ignore[reportPrivateUsage]

    assert result is None
    assert any(record.levelname == "WARNING" and "ubuntu:latest" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_resolve_image_id_for_adoption_non_dict_response_logs_warning_and_returns_none(
    docker_service: DockerService, caplog: pytest.LogCaptureFixture
) -> None:
    cm, instance = _make_docker_mock()
    instance.images = MagicMock()
    instance.images.inspect = AsyncMock(return_value=["not", "a", "dict"])

    with patch("server.docker.Docker", return_value=cm), caplog.at_level("WARNING", logger="uvicorn.error"):
        result = await docker_service._resolve_image_id_for_adoption("ubuntu:latest")  # type: ignore[reportPrivateUsage]

    assert result is None
    assert any(record.levelname == "WARNING" and "ubuntu:latest" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_resolve_image_id_for_adoption_missing_id_field_returns_none(docker_service: DockerService) -> None:
    cm, instance = _make_docker_mock()
    instance.images = MagicMock()
    instance.images.inspect = AsyncMock(return_value={"RepoTags": ["ubuntu:latest"]})

    with patch("server.docker.Docker", return_value=cm):
        result = await docker_service._resolve_image_id_for_adoption("ubuntu:latest")  # type: ignore[reportPrivateUsage]

    assert result is None


# _try_adopt_orphaned_container


@pytest.mark.asyncio
async def test_try_adopt_orphaned_container_logs_when_found_but_not_matched(
    docker_service: DockerService, caplog: pytest.LogCaptureFixture
) -> None:
    """A near-miss (e.g. wrong image) must not be totally silent - an operator needs a breadcrumb."""
    options = _opts(name="ollama", image="ubuntu:latest")
    options.container_name = "ollama"
    cm, _ = _mock_container_show({"Config": {"Image": "ubuntu:22.04", "Env": []}, "State": {"Status": "running"}, "Mounts": []})

    with patch("server.docker.Docker", return_value=cm), caplog.at_level("INFO", logger="uvicorn.error"):
        result = await docker_service._try_adopt_orphaned_container(options)  # type: ignore[reportPrivateUsage]

    assert result is None
    assert any("ollama" in record.message and "not adopting" in record.message.lower() for record in caplog.records)


@pytest.mark.asyncio
async def test_try_adopt_orphaned_container_logs_on_successful_adoption(
    docker_service: DockerService, caplog: pytest.LogCaptureFixture
) -> None:
    options = _opts(name="ollama", image="ubuntu:latest", image_port=8080)
    options.container_name = "ollama"
    cm, _ = _mock_container_show(
        {
            "Image": DEFAULT_ADOPTION_IMAGE_ID,
            "Config": {"Image": "ubuntu:latest", "Env": []},
            "State": {"Status": "running"},
            "Mounts": [],
            "NetworkSettings": {"Ports": {"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "44444"}]}},
        }
    )

    with patch("server.docker.Docker", return_value=cm), caplog.at_level("INFO", logger="uvicorn.error"):
        result = await docker_service._try_adopt_orphaned_container(options)  # type: ignore[reportPrivateUsage]

    assert result == 44444
    assert any("ollama" in record.message and "adopting" in record.message.lower() for record in caplog.records)


@pytest.mark.asyncio
async def test_try_adopt_orphaned_container_skips_lookup_when_container_name_not_set(docker_service: DockerService) -> None:
    """Without an explicit container_name there's no reliable way to know what name Compose assigned - never guess."""
    options = _opts(name="ollama", image="ubuntu:latest")
    assert options.container_name is None

    with patch("server.docker.Docker") as mock_docker:
        result = await docker_service._try_adopt_orphaned_container(options)  # type: ignore[reportPrivateUsage]

    assert result is None
    mock_docker.assert_not_called()


@pytest.mark.asyncio
async def test_try_adopt_orphaned_container_skips_adoption_when_image_id_unresolvable(docker_service: DockerService) -> None:
    """If the configured image's current ID can't be resolved (e.g. not pulled locally), adoption

    can't be proven safe, even if the container's Config.Image name/tag string matches - a name/tag
    match alone doesn't confirm it's running the same software as what would actually be pulled/run now.
    """
    options = _opts(name="ollama", image="ubuntu:latest", image_port=8080)
    options.container_name = "ollama"
    cm, _ = _mock_container_show(
        {
            "Image": DEFAULT_ADOPTION_IMAGE_ID,
            "Config": {"Image": "ubuntu:latest", "Env": []},
            "State": {"Status": "running"},
            "Mounts": [],
            "NetworkSettings": {"Ports": {"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "44444"}]}},
        }
    )
    cm.__aenter__.return_value.images.inspect = AsyncMock(side_effect=DockerError(status=404, message="no such image"))

    with patch("server.docker.Docker", return_value=cm):
        result = await docker_service._try_adopt_orphaned_container(options)  # type: ignore[reportPrivateUsage]

    assert result is None


# _ensure_compose_running


@pytest.mark.asyncio
async def test_ensure_compose_running_gpu_toolkit_error_raises_actionable_app_error(docker_service: DockerService, tmp_path: Path) -> None:
    gpu = NvidiaGpuInfo(name="RTX 3090", vram="24 GB", id=0)
    options = _opts(hardware=[gpu])
    compose_file = tmp_path / "compose.yaml"

    with (
        patch.object(docker_service, "create_compose_file", new_callable=AsyncMock, return_value=11434),
        patch.object(
            docker_service,
            "start_docker_compose",
            new_callable=AsyncMock,
            side_effect=DockerComposeStartError("", "could not select device driver"),
        ),
        pytest.raises(AppError, match="NVIDIA container toolkit"),
    ):
        await docker_service._ensure_compose_running(compose_file, options, is_running=False, has_difference=True, port=None)  # type: ignore[reportPrivateUsage]

    assert docker_service.has_gpu_support is True


@pytest.mark.asyncio
async def test_ensure_compose_running_gpu_unknown_error_raises_generic_app_error(docker_service: DockerService, tmp_path: Path) -> None:
    gpu = NvidiaGpuInfo(name="RTX 3090", vram="24 GB", id=0)
    options = _opts(hardware=[gpu])
    compose_file = tmp_path / "compose.yaml"

    with (
        patch.object(docker_service, "create_compose_file", new_callable=AsyncMock, return_value=11434),
        patch.object(
            docker_service,
            "start_docker_compose",
            new_callable=AsyncMock,
            side_effect=DockerComposeStartError("out", "unrelated error"),
        ),
        pytest.raises(AppError, match="Failed to start"),
    ):
        await docker_service._ensure_compose_running(compose_file, options, is_running=False, has_difference=True, port=None)  # type: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_ensure_compose_running_non_gpu_start_error_raises_app_error(docker_service: DockerService, tmp_path: Path) -> None:
    options = _opts()
    compose_file = tmp_path / "compose.yaml"

    with (
        patch.object(docker_service, "create_compose_file", new_callable=AsyncMock, return_value=11434),
        patch.object(
            docker_service,
            "start_docker_compose",
            new_callable=AsyncMock,
            side_effect=DockerComposeStartError("out", "err"),
        ),
        pytest.raises(AppError, match="Failed to start"),
    ):
        await docker_service._ensure_compose_running(compose_file, options, is_running=False, has_difference=True, port=None)  # type: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_ensure_compose_running_name_conflict_raises_http_409(docker_service: DockerService, tmp_path: Path) -> None:
    options = _opts(name="ollama")
    options.container_name = "ollama"
    compose_file = tmp_path / "compose.yaml"
    stderr = 'Conflict. The container name "/ollama" is already in use by container "abc123".'

    with (
        patch.object(docker_service, "create_compose_file", new_callable=AsyncMock, return_value=11434),
        patch.object(
            docker_service,
            "start_docker_compose",
            new_callable=AsyncMock,
            side_effect=DockerComposeStartError("", stderr),
        ),
        # The reason adoption doesn't succeed is covered by other tests (mismatched image,
        # container not found, etc.); this test only cares that a failed adoption surfaces as a 409
        # with the container name in the detail, so it stubs adoption directly instead of exercising
        # a real Docker connection.
        patch.object(docker_service, "_try_adopt_orphaned_container", new_callable=AsyncMock, return_value=None),
        pytest.raises(HTTPException) as exc_info,
    ):
        await docker_service._ensure_compose_running(compose_file, options, is_running=False, has_difference=True, port=None)  # type: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 409
    assert "ollama" in exc_info.value.detail


@pytest.mark.asyncio
async def test_ensure_compose_running_name_conflict_phrase_in_logs_only_is_not_treated_as_conflict(
    docker_service: DockerService, tmp_path: Path
) -> None:
    """A container log line like "port is already in use" must not be mistaken for a name collision."""
    options = _opts(name="ollama")
    compose_file = tmp_path / "compose.yaml"

    with (
        patch.object(docker_service, "create_compose_file", new_callable=AsyncMock, return_value=11434),
        patch.object(
            docker_service,
            "start_docker_compose",
            new_callable=AsyncMock,
            side_effect=DockerComposeStartError("out", "err"),
        ),
        patch.object(docker_service, "get_docker_compose_logs", new_callable=AsyncMock, return_value="port is already in use"),
        # If the routing logic ever mistook this for a name conflict it would try to adopt, which
        # means touching Docker — fail loudly on that instead of silently depending on there being
        # no real daemon reachable in the test environment.
        patch("server.docker.Docker", side_effect=AssertionError("adoption path must not be reached for a non-conflict error")),
        pytest.raises(AppError, match="Failed to start"),
    ):
        await docker_service._ensure_compose_running(compose_file, options, is_running=False, has_difference=True, port=None)  # type: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_ensure_compose_running_adopts_matching_orphan_and_removes_written_compose_file(
    docker_service: DockerService, tmp_path: Path
) -> None:
    options = _opts(name="ollama", image="ubuntu:latest", image_port=8080)
    options.container_name = "ollama"
    compose_file = tmp_path / "compose.yaml"
    stderr = 'Conflict. The container name "/ollama" is already in use by container "abc123".'

    async def _write_compose_file(path: Path, _opts: DockerOptions) -> int:
        path.write_text("stub")
        return 11434

    cm, _ = _mock_container_show(
        {
            "Image": DEFAULT_ADOPTION_IMAGE_ID,
            "Config": {"Image": "ubuntu:latest", "Env": []},
            "State": {"Status": "running"},
            "Mounts": [],
            "NetworkSettings": {"Ports": {"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "44444"}]}},
        }
    )

    with (
        patch.object(docker_service, "create_compose_file", new_callable=AsyncMock, side_effect=_write_compose_file),
        patch.object(
            docker_service,
            "start_docker_compose",
            new_callable=AsyncMock,
            side_effect=DockerComposeStartError("", stderr),
        ),
        patch("server.docker.Docker", return_value=cm),
    ):
        start_output, port, adopted = await docker_service._ensure_compose_running(  # type: ignore[reportPrivateUsage]
            compose_file, options, is_running=False, has_difference=True, port=None
        )

    assert start_output is None
    assert port == 44444
    assert adopted is True
    assert not compose_file.exists()
    docker_service.port_service.release_port.assert_called_once_with(11434)  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.asyncio
async def test_ensure_compose_running_adoption_same_port_does_not_release_it(docker_service: DockerService, tmp_path: Path) -> None:
    """The reserved port must only be released when adoption ends up using a *different* one."""
    options = _opts(name="ollama", image="ubuntu:latest", image_port=8080)
    options.container_name = "ollama"
    compose_file = tmp_path / "compose.yaml"
    stderr = 'Conflict. The container name "/ollama" is already in use by container "abc123".'

    async def _write_compose_file(path: Path, _opts: DockerOptions) -> int:
        path.write_text("stub")
        return 44444

    cm, _ = _mock_container_show(
        {
            "Image": DEFAULT_ADOPTION_IMAGE_ID,
            "Config": {"Image": "ubuntu:latest", "Env": []},
            "State": {"Status": "running"},
            "Mounts": [],
            "NetworkSettings": {"Ports": {"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "44444"}]}},
        }
    )

    with (
        patch.object(docker_service, "create_compose_file", new_callable=AsyncMock, side_effect=_write_compose_file),
        patch.object(
            docker_service,
            "start_docker_compose",
            new_callable=AsyncMock,
            side_effect=DockerComposeStartError("", stderr),
        ),
        patch("server.docker.Docker", return_value=cm),
    ):
        _, port, adopted = await docker_service._ensure_compose_running(  # type: ignore[reportPrivateUsage]
            compose_file, options, is_running=False, has_difference=True, port=None
        )

    assert port == 44444
    assert adopted is True
    docker_service.port_service.release_port.assert_not_called()  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.asyncio
async def test_ensure_compose_running_adoption_reverification_catches_crash_and_raises_app_error(
    docker_service: DockerService, tmp_path: Path
) -> None:
    """The match decision is based on one inspect snapshot; if the container crashes before adoption is
    committed, a fresh re-inspect right before commit must catch it instead of silently registering a
    dead container as adopted."""
    options = _opts(name="ollama", image="ubuntu:latest", image_port=8080)
    options.container_name = "ollama"
    compose_file = tmp_path / "compose.yaml"
    stderr = 'Conflict. The container name "/ollama" is already in use by container "abc123".'

    healthy_inspect = {
        "Image": DEFAULT_ADOPTION_IMAGE_ID,
        "Config": {"Image": "ubuntu:latest", "Env": []},
        "State": {"Status": "running"},
        "Mounts": [],
        "NetworkSettings": {"Ports": {"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "44444"}]}},
    }
    crashed_inspect = {**healthy_inspect, "State": {"Status": "exited"}}

    cm, instance = _make_docker_mock()
    container = MagicMock()
    container.show = AsyncMock(side_effect=[healthy_inspect, crashed_inspect])
    instance.containers = MagicMock()
    instance.containers.container = MagicMock(return_value=container)
    instance.images = MagicMock()
    instance.images.inspect = AsyncMock(return_value={"Id": DEFAULT_ADOPTION_IMAGE_ID})

    async def _write_compose_file(path: Path, _opts: DockerOptions) -> int:
        path.write_text("stub")
        return 11434

    with (
        patch.object(docker_service, "create_compose_file", new_callable=AsyncMock, side_effect=_write_compose_file),
        patch.object(
            docker_service,
            "start_docker_compose",
            new_callable=AsyncMock,
            side_effect=DockerComposeStartError("", stderr),
        ),
        patch("server.docker.Docker", return_value=cm),
        patch("server.docker.Utils.run_command", new_callable=AsyncMock, return_value=make_result(stdout="crash logs")),
        pytest.raises(AppError, match="no longer healthy"),
    ):
        await docker_service._ensure_compose_running(compose_file, options, is_running=False, has_difference=True, port=None)  # type: ignore[reportPrivateUsage]

    # A failed re-verification must not delete the compose file this attempt wrote - same as any other failed start.
    assert compose_file.exists()


@pytest.mark.asyncio
async def test_ensure_compose_running_adoption_reverification_inspect_failure_raises_app_error(
    docker_service: DockerService, tmp_path: Path
) -> None:
    """If the container disappears (or the daemon errors) between the match decision and the
    re-verification inspect, that must surface as a failed install too, not a silent adoption."""
    options = _opts(name="ollama", image="ubuntu:latest", image_port=8080)
    options.container_name = "ollama"
    compose_file = tmp_path / "compose.yaml"
    stderr = 'Conflict. The container name "/ollama" is already in use by container "abc123".'

    healthy_inspect = {
        "Image": DEFAULT_ADOPTION_IMAGE_ID,
        "Config": {"Image": "ubuntu:latest", "Env": []},
        "State": {"Status": "running"},
        "Mounts": [],
        "NetworkSettings": {"Ports": {"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "44444"}]}},
    }

    cm, instance = _make_docker_mock()
    container = MagicMock()
    container.show = AsyncMock(side_effect=[healthy_inspect, DockerError(status=404, message="no such container")])
    instance.containers = MagicMock()
    instance.containers.container = MagicMock(return_value=container)
    instance.images = MagicMock()
    instance.images.inspect = AsyncMock(return_value={"Id": DEFAULT_ADOPTION_IMAGE_ID})

    async def _write_compose_file(path: Path, _opts: DockerOptions) -> int:
        path.write_text("stub")
        return 11434

    with (
        patch.object(docker_service, "create_compose_file", new_callable=AsyncMock, side_effect=_write_compose_file),
        patch.object(
            docker_service,
            "start_docker_compose",
            new_callable=AsyncMock,
            side_effect=DockerComposeStartError("", stderr),
        ),
        patch("server.docker.Docker", return_value=cm),
        pytest.raises(AppError, match="could not be re-confirmed healthy"),
    ):
        await docker_service._ensure_compose_running(compose_file, options, is_running=False, has_difference=True, port=None)  # type: ignore[reportPrivateUsage]

    assert compose_file.exists()


@pytest.mark.asyncio
async def test_ensure_compose_running_adoption_survives_compose_file_unlink_failure(
    docker_service: DockerService, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A failure to remove the stale compose file (e.g. permission denied) must not crash a successful adoption."""
    options = _opts(name="ollama", image="ubuntu:latest", image_port=8080)
    options.container_name = "ollama"
    compose_file = tmp_path / "compose.yaml"
    compose_file.mkdir()  # unlink() on a directory raises IsADirectoryError, a plain OSError
    stderr = 'Conflict. The container name "/ollama" is already in use by container "abc123".'

    cm, _ = _mock_container_show(
        {
            "Image": DEFAULT_ADOPTION_IMAGE_ID,
            "Config": {"Image": "ubuntu:latest", "Env": []},
            "State": {"Status": "running"},
            "Mounts": [],
            "NetworkSettings": {"Ports": {"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "44444"}]}},
        }
    )

    with (
        patch.object(docker_service, "create_compose_file", new_callable=AsyncMock, return_value=11434),
        patch.object(
            docker_service,
            "start_docker_compose",
            new_callable=AsyncMock,
            side_effect=DockerComposeStartError("", stderr),
        ),
        patch("server.docker.Docker", return_value=cm),
        caplog.at_level("WARNING", logger="uvicorn.error"),
    ):
        start_output, port, adopted = await docker_service._ensure_compose_running(  # type: ignore[reportPrivateUsage]
            compose_file, options, is_running=False, has_difference=True, port=None
        )

    assert start_output is None
    assert port == 44444
    assert adopted is True
    assert compose_file.is_dir()  # unlink failed, so the (bogus) directory is still there
    assert any(record.levelname == "WARNING" and "ollama" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_ensure_compose_running_adoption_keeps_preexisting_compose_file_untouched(
    docker_service: DockerService, tmp_path: Path
) -> None:
    """A conflict on the no-create_compose_file branch must not delete a compose file it didn't write."""
    options = _opts(name="ollama", image="ubuntu:latest", image_port=8080)
    options.container_name = "ollama"
    compose_file = tmp_path / "compose.yaml"
    compose_file.write_text("preexisting")
    stderr = 'Conflict. The container name "/ollama" is already in use by container "abc123".'
    docker_service.port_service.is_port_available.return_value = True  # pyright: ignore[reportAttributeAccessIssue]

    cm, _ = _mock_container_show(
        {
            "Image": DEFAULT_ADOPTION_IMAGE_ID,
            "Config": {"Image": "ubuntu:latest", "Env": []},
            "State": {"Status": "running"},
            "Mounts": [],
            "NetworkSettings": {"Ports": {"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "44444"}]}},
        }
    )

    with (
        patch.object(
            docker_service,
            "start_docker_compose",
            new_callable=AsyncMock,
            side_effect=DockerComposeStartError("", stderr),
        ),
        patch("server.docker.Docker", return_value=cm),
    ):
        start_output, port, adopted = await docker_service._ensure_compose_running(  # type: ignore[reportPrivateUsage]
            compose_file, options, is_running=False, has_difference=False, port=11434
        )

    assert start_output is None
    assert port == 44444
    assert adopted is True
    assert compose_file.read_text() == "preexisting"


@pytest.mark.asyncio
async def test_ensure_compose_running_non_matching_orphan_still_raises_http_409(docker_service: DockerService, tmp_path: Path) -> None:
    options = _opts(name="ollama", image="ubuntu:latest", image_port=8080)
    options.container_name = "ollama"
    compose_file = tmp_path / "compose.yaml"
    stderr = 'Conflict. The container name "/ollama" is already in use by container "abc123".'

    cm, _ = _mock_container_show(
        {
            "Image": "sha256:" + "c" * 64,  # different resolved image id -> no match
            "Config": {"Image": "ubuntu:latest", "Env": []},
            "State": {"Status": "running"},
            "Mounts": [],
            "NetworkSettings": {"Ports": {"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "44444"}]}},
        }
    )

    with (
        patch.object(docker_service, "create_compose_file", new_callable=AsyncMock, return_value=11434),
        patch.object(
            docker_service,
            "start_docker_compose",
            new_callable=AsyncMock,
            side_effect=DockerComposeStartError("", stderr),
        ),
        patch("server.docker.Docker", return_value=cm),
        pytest.raises(HTTPException) as exc_info,
    ):
        await docker_service._ensure_compose_running(compose_file, options, is_running=False, has_difference=True, port=None)  # type: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 409


@pytest.mark.asyncio
async def test_ensure_compose_running_inspect_failure_falls_back_to_http_409(docker_service: DockerService, tmp_path: Path) -> None:
    options = _opts(name="ollama")
    options.container_name = "ollama"
    compose_file = tmp_path / "compose.yaml"
    stderr = 'Conflict. The container name "/ollama" is already in use by container "abc123".'

    instance = MagicMock()
    instance.containers = MagicMock()
    container = MagicMock()
    container.show = AsyncMock(side_effect=DockerError(status=404, message="no such container"))
    instance.containers.container = MagicMock(return_value=container)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=instance)
    cm.__aexit__ = AsyncMock(return_value=False)

    with (
        patch.object(docker_service, "create_compose_file", new_callable=AsyncMock, return_value=11434),
        patch.object(
            docker_service,
            "start_docker_compose",
            new_callable=AsyncMock,
            side_effect=DockerComposeStartError("", stderr),
        ),
        patch("server.docker.Docker", return_value=cm),
        pytest.raises(HTTPException) as exc_info,
    ):
        await docker_service._ensure_compose_running(compose_file, options, is_running=False, has_difference=True, port=None)  # type: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 409


@pytest.mark.asyncio
async def test_ensure_compose_running_short_message_but_logs_full_output(
    docker_service: DockerService, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    options = _opts(name="ollama")
    compose_file = tmp_path / "compose.yaml"
    long_logs = "x" * 10_000

    with (
        patch.object(docker_service, "create_compose_file", new_callable=AsyncMock, return_value=11434),
        patch.object(
            docker_service,
            "start_docker_compose",
            new_callable=AsyncMock,
            side_effect=DockerComposeStartError("out", "err"),
        ),
        patch.object(docker_service, "get_docker_compose_logs", new_callable=AsyncMock, return_value=long_logs),
        caplog.at_level("ERROR", logger="uvicorn.error"),
        pytest.raises(AppError, match="Failed to start") as exc_info,
    ):
        await docker_service._ensure_compose_running(compose_file, options, is_running=False, has_difference=True, port=None)  # type: ignore[reportPrivateUsage]

    assert len(str(exc_info.value)) < len(long_logs)
    assert long_logs in caplog.text


@pytest.mark.asyncio
async def test_ensure_compose_running_bounds_pathologically_large_logs_in_server_log(
    docker_service: DockerService, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A runaway container (e.g. a stuck download progress stream) must not flood the server log."""
    options = _opts(name="ollama")
    compose_file = tmp_path / "compose.yaml"
    huge_logs = "x" * 100_000

    with (
        patch.object(docker_service, "create_compose_file", new_callable=AsyncMock, return_value=11434),
        patch.object(
            docker_service,
            "start_docker_compose",
            new_callable=AsyncMock,
            side_effect=DockerComposeStartError("out", "err"),
        ),
        patch.object(docker_service, "get_docker_compose_logs", new_callable=AsyncMock, return_value=huge_logs),
        caplog.at_level("ERROR", logger="uvicorn.error"),
        pytest.raises(AppError, match="Failed to start"),
    ):
        await docker_service._ensure_compose_running(compose_file, options, is_running=False, has_difference=True, port=None)  # type: ignore[reportPrivateUsage]

    assert huge_logs not in caplog.text
    assert len(caplog.text) < len(huge_logs)


# _assert_compose_healthy


@pytest.mark.asyncio
async def test_assert_compose_healthy_gpu_error_in_logs_raises_actionable_app_error(docker_service: DockerService, tmp_path: Path) -> None:
    gpu = NvidiaGpuInfo(name="RTX 3090", vram="24 GB", id=0)
    options = _opts(hardware=[gpu])
    compose_file = tmp_path / "compose.yaml"

    with (
        patch.object(docker_service, "is_docker_compose_healthy", new_callable=AsyncMock, return_value=False),
        patch.object(docker_service, "get_docker_compose_logs", new_callable=AsyncMock, return_value="could not select device driver"),
        pytest.raises(AppError, match="NVIDIA container toolkit"),
    ):
        await docker_service._assert_compose_healthy(compose_file, options, start_output=None)  # type: ignore[reportPrivateUsage]

    assert docker_service.has_gpu_support is True


@pytest.mark.asyncio
async def test_assert_compose_healthy_gpu_no_known_error_raises_generic_app_error(docker_service: DockerService, tmp_path: Path) -> None:
    gpu = NvidiaGpuInfo(name="RTX 3090", vram="24 GB", id=0)
    options = _opts(hardware=[gpu])
    compose_file = tmp_path / "compose.yaml"

    with (
        patch.object(docker_service, "is_docker_compose_healthy", new_callable=AsyncMock, return_value=False),
        patch.object(docker_service, "get_docker_compose_logs", new_callable=AsyncMock, return_value="unrelated output"),
        pytest.raises(AppError, match="failed to become healthy"),
    ):
        await docker_service._assert_compose_healthy(compose_file, options, start_output=None)  # type: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_assert_compose_healthy_short_message_but_logs_full_output(
    docker_service: DockerService, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    options = _opts(name="ollama")
    compose_file = tmp_path / "compose.yaml"
    long_logs = "x" * 10_000

    with (
        patch.object(docker_service, "is_docker_compose_healthy", new_callable=AsyncMock, return_value=False),
        patch.object(docker_service, "get_docker_compose_logs", new_callable=AsyncMock, return_value=long_logs),
        caplog.at_level("ERROR", logger="uvicorn.error"),
        pytest.raises(AppError, match="failed to become healthy") as exc_info,
    ):
        await docker_service._assert_compose_healthy(compose_file, options, start_output=None)  # type: ignore[reportPrivateUsage]

    assert len(str(exc_info.value)) < len(long_logs)
    assert long_logs in caplog.text


def test_get_docker_compose_dir_creates_dir_when_missing(tmp_path: Path) -> None:
    config = MagicMock()
    config.compose_prefix = ""
    config.container_name_prefix = ""
    config.docker_subnet = ""
    # Point to a storage dir that does NOT yet have a "config" subdir
    storage = tmp_path / "storage"
    storage.mkdir()
    config.get_storage_dir.return_value = storage

    with patch("server.docker.get_docker_auths", return_value={}):
        svc = DockerService(
            config=config,
            port_service=MagicMock(),
            docker_compose_cmd="docker compose",
            has_gpu_support=False,
            os="linux",
            architecture="amd64",
            is_rootless=False,
            host_platform="linux/amd64",
        )

    result = svc.get_docker_compose_dir()

    assert result.is_dir()
    assert result == storage / "config"


def test_get_docker_compose_file_path_with_prefix_creates_dir(tmp_path: Path) -> None:
    config = MagicMock()
    config.compose_prefix = "df-"
    config.container_name_prefix = ""
    config.docker_subnet = ""
    config.get_storage_dir.return_value = tmp_path
    with patch("server.docker.get_docker_auths", return_value={}):
        svc = DockerService(
            config=config,
            port_service=MagicMock(),
            docker_compose_cmd="docker compose",
            has_gpu_support=False,
            os="linux",
            architecture="amd64",
            is_rootless=False,
            host_platform="linux/amd64",
        )

    path = svc.get_docker_compose_file_path("mymodel")

    assert path.name == "compose.yaml"
    assert path.parent.is_dir()
    assert "df-mymodel" in str(path)


@pytest.mark.asyncio
async def test_create_docker_service_docker_compose_plugin(tmp_path: Path) -> None:
    config = MagicMock()
    config.external_only = False
    config.get_storage_dir.return_value = tmp_path
    port_service = MagicMock()

    with (
        patch("server.docker.shutil.which", return_value="/usr/bin/docker"),
        patch("server.docker.Utils.run_command", new_callable=AsyncMock) as mock_cmd,
        patch("server.docker.get_docker_auths", return_value={}),
        patch("server.docker.get_os", return_value="linux"),
        patch("server.docker.get_cpu_architecture", return_value="amd64"),
        patch("server.docker.platform.machine", return_value="x86_64"),
    ):
        # docker compose version → success; gpu support → fail; docker info → rootless
        mock_cmd.side_effect = [
            make_result(exit_code=0, stdout="Docker Compose version v2"),  # compose version
            make_result(exit_code=1),  # gpu check
            make_result(exit_code=0, stdout="rootless"),  # docker info
        ]

        svc = await create_docker_service(port_service, config)

    assert svc.docker_compose_cmd == "docker compose"
    assert svc.is_rootless is True
    assert svc.has_gpu_support is False


@pytest.mark.asyncio
async def test_create_docker_service_falls_back_to_docker_compose_binary(tmp_path: Path) -> None:
    config = MagicMock()
    config.external_only = False
    config.get_storage_dir.return_value = tmp_path
    port_service = MagicMock()

    def which_side_effect(cmd: str) -> str | None:
        return "/usr/bin/docker-compose" if cmd == "docker-compose" else ("/usr/bin/docker" if cmd == "docker" else None)

    with (
        patch("server.docker.shutil.which", side_effect=which_side_effect),
        patch("server.docker.Utils.run_command", new_callable=AsyncMock) as mock_cmd,
        patch("server.docker.get_docker_auths", return_value={}),
        patch("server.docker.get_os", return_value="linux"),
        patch("server.docker.get_cpu_architecture", return_value="amd64"),
        patch("server.docker.platform.machine", return_value="x86_64"),
    ):
        mock_cmd.side_effect = [
            make_result(exit_code=1),  # docker compose plugin not available
            make_result(exit_code=1),  # gpu check
            make_result(exit_code=0, stdout=""),  # docker info (not rootless)
        ]

        svc = await create_docker_service(port_service, config)

    assert svc.docker_compose_cmd == "docker-compose"


@pytest.mark.asyncio
async def test_create_docker_service_external_only_makes_no_docker_calls() -> None:
    config = MagicMock()
    config.external_only = True
    port_service = MagicMock()

    def _fail_which(_cmd: str) -> str | None:
        raise AssertionError("shutil.which must not be called in external-only mode")

    def _fail_run(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("no docker command must run in external-only mode")

    with (
        patch("server.docker.shutil.which", side_effect=_fail_which),
        patch("server.docker.Utils.run_command", new_callable=AsyncMock, side_effect=_fail_run),
        patch("server.docker.get_docker_auths", return_value={}),
        patch("server.docker.get_os", return_value="linux"),
        patch("server.docker.get_cpu_architecture", return_value="amd64"),
        patch("server.docker.platform.machine", return_value="x86_64"),
    ):
        svc = await create_docker_service(port_service, config)

    assert svc.docker_compose_cmd == ""
    assert svc.has_gpu_support is False
    assert svc.is_rootless is False


def test_docker_path_add_combines_paths() -> None:
    dp = DockerPath(local_path=Path("/local/base"), docker_path=Path("/docker/base"))

    result = dp.add(Path("sub/dir"))
    assert result.local_path == Path("/local/base/sub/dir")
    assert result.docker_path == Path("/docker/base/sub/dir")


@pytest.mark.asyncio
async def test_get_image_platforms_empty_image_data_returns_empty(docker_service: DockerService) -> None:
    with patch("server.docker.Utils.run_command_for_success", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = make_result2(stdout="[]")  # valid JSON but empty list

        result = await docker_service.get_image_platforms("ubuntu:latest")

    assert result == []


@pytest.mark.asyncio
async def test_get_existing_or_free_port_docker_exception_in_parse_gets_free_port(docker_service: DockerService, tmp_path: Path) -> None:
    compose_file = tmp_path / "compose.yaml"
    compose_file.write_text("{bad yaml [[[")
    docker_service.port_service.get_free_port.return_value = 77777  # pyright: ignore[reportAttributeAccessIssue]

    with patch("server.docker.yaml.safe_load", side_effect=Exception("parse error")):
        port = await docker_service.get_existing_or_free_port_docker(compose_file, _opts())

    assert port == 77777


@pytest.mark.asyncio
async def test_create_docker_service_raises_when_docker_not_installed(tmp_path: Path) -> None:
    config = MagicMock()
    config.external_only = False
    port_service = MagicMock()

    with (
        patch("server.docker.shutil.which", return_value=None),
        patch("server.docker.Utils.run_command", new_callable=AsyncMock),
        pytest.raises(DockerNotInstalledError),
    ):
        await create_docker_service(port_service, config)


def test_calculate_total_layer_size_skips_non_int_size(docker_service: DockerService) -> None:
    manifest = {"layers": [{"size": 100}, {"size": "not-an-int"}, {"size": 200}]}

    assert docker_service.calculate_total_layer_size(manifest) == 300


@pytest.mark.asyncio
async def test_get_image_platforms_manifest_entry_without_platform_key_skipped(docker_service: DockerService) -> None:
    manifest = {
        "manifests": [
            {"digest": "sha256:no-platform"},  # no "platform" key
            {"platform": {"os": "linux", "architecture": "amd64", "variant": ""}},
        ]
    }
    with (
        patch("server.docker.Utils.run_command_for_success", new_callable=AsyncMock) as mock_run,
        patch.object(docker_service, "get_docker_manifest", new_callable=AsyncMock, return_value=manifest),
    ):
        mock_run.side_effect = RuntimeError("not local")

        result = await docker_service.get_image_platforms("ubuntu:latest")

    assert "linux/amd64" in result
    assert len(result) == 1


def test_get_docker_compose_dir_existing_dir_skips_mkdir(docker_service: DockerService, tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    result = docker_service.get_docker_compose_dir()

    assert result == config_dir
    assert result.is_dir()


def test_get_docker_compose_file_path_with_prefix_existing_dir_skips_mkdir(docker_service: DockerService, tmp_path: Path) -> None:
    docker_service.config.compose_prefix = "pf-"
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    existing_dir = config_dir / "pf-mymodel"
    existing_dir.mkdir()

    path = docker_service.get_docker_compose_file_path("mymodel")

    assert path.name == "compose.yaml"
    assert path.parent == existing_dir


@pytest.mark.asyncio
async def test_create_docker_service_raises_when_compose_not_available(tmp_path: Path) -> None:
    config = MagicMock()
    config.external_only = False
    port_service = MagicMock()

    def which_side_effect(cmd: str) -> str | None:
        return "/usr/bin/docker" if cmd == "docker" else None

    with (
        patch("server.docker.shutil.which", side_effect=which_side_effect),
        patch("server.docker.Utils.run_command", new_callable=AsyncMock) as mock_cmd,
    ):
        mock_cmd.return_value = make_result(exit_code=1)  # docker compose version fails

        with pytest.raises(DockerNotInstalledError):
            await create_docker_service(port_service, config)


class _AsyncLineIterator:
    """Async iterator over a list of byte lines for mocking asyncio StreamReader."""

    def __init__(self, lines: list[bytes]) -> None:
        self._iter = iter(lines)

    def __aiter__(self) -> "_AsyncLineIterator":
        return self

    async def __anext__(self) -> bytes:
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration from None


@pytest.mark.asyncio
async def test_build_image_calls_docker_build_with_correct_args(docker_service: DockerService, tmp_path: Path) -> None:
    """build_image shells out to 'docker build -t <tag> <dir>'."""
    mock_proc = MagicMock()
    mock_proc.stdout = _AsyncLineIterator([b"Step 1/3 : FROM node:lts-slim\n"])
    mock_proc.wait = AsyncMock()
    mock_proc.returncode = 0

    with patch("server.docker.asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=mock_proc) as mock_exec:
        stream = MagicMock()
        stream.emit = MagicMock()
        await docker_service.build_image(tmp_path, "my-tag:latest", stream)  # type: ignore[arg-type]

    mock_exec.assert_called_once_with(
        "docker",
        "build",
        "-t",
        "my-tag:latest",
        str(tmp_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )


@pytest.mark.asyncio
async def test_build_image_emits_log_lines_as_progress(docker_service: DockerService, tmp_path: Path) -> None:
    """build_image emits each build output line as a progress chunk."""
    mock_proc = MagicMock()
    mock_proc.stdout = _AsyncLineIterator([b"Step 1/2\n", b"Step 2/2\n"])
    mock_proc.wait = AsyncMock()
    mock_proc.returncode = 0

    with patch("server.docker.asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=mock_proc):
        stream = MagicMock()
        stream.emit = MagicMock()
        await docker_service.build_image(tmp_path, "tag:latest", stream)  # type: ignore[arg-type]

    assert stream.emit.call_count == 2
    first_call_arg = stream.emit.call_args_list[0][0][0]
    assert first_call_arg["type"] == "progress"
    assert first_call_arg["stage"] == "install"
    assert "log" in first_call_arg["data"]


@pytest.mark.asyncio
async def test_build_image_raises_on_nonzero_exit(docker_service: DockerService, tmp_path: Path) -> None:
    """build_image raises RuntimeError when docker build exits non-zero."""
    mock_proc = MagicMock()
    mock_proc.stdout = _AsyncLineIterator([b"error: build failed\n"])
    mock_proc.wait = AsyncMock()
    mock_proc.returncode = 1

    with patch("server.docker.asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=mock_proc):
        stream = MagicMock()
        stream.emit = MagicMock()
        with pytest.raises(RuntimeError, match="docker build failed"):
            await docker_service.build_image(tmp_path, "tag:latest", stream)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_build_image_timeout_raises_runtime_error(docker_service: DockerService, tmp_path: Path) -> None:
    """build_image kills the process and raises RuntimeError when docker build times out."""
    mock_proc = MagicMock()
    mock_proc.stdout = _AsyncLineIterator([])
    mock_proc.wait = AsyncMock(side_effect=[TimeoutError, None])
    mock_proc.kill = MagicMock()

    with patch("server.docker.asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=mock_proc):
        stream = MagicMock()
        stream.emit = MagicMock()
        with pytest.raises(RuntimeError, match="timed out"):
            await docker_service.build_image(tmp_path, "tag:latest", stream)  # type: ignore[arg-type]

    mock_proc.kill.assert_called_once()


@pytest.mark.asyncio
async def test_build_image_raises_runtime_error_when_stdout_is_none(docker_service: DockerService, tmp_path: Path) -> None:
    """build_image raises RuntimeError immediately when the subprocess has no stdout pipe."""
    mock_proc = MagicMock()
    mock_proc.stdout = None

    with patch("server.docker.asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=mock_proc):
        stream = MagicMock()
        with pytest.raises(RuntimeError, match="no stdout"):
            await docker_service.build_image(tmp_path, "tag:latest", stream)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_build_image_kills_process_when_stdout_iteration_raises(docker_service: DockerService, tmp_path: Path) -> None:
    """build_image kills the process and re-raises when an error occurs while reading stdout."""

    async def _broken_stream():
        raise RuntimeError("broken pipe")
        yield  # makes it an async generator

    mock_proc = MagicMock()
    mock_proc.stdout = _broken_stream()
    mock_proc.kill = MagicMock()
    mock_proc.wait = AsyncMock()

    with patch("server.docker.asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=mock_proc):
        stream = MagicMock()
        with pytest.raises(RuntimeError, match="broken pipe"):
            await docker_service.build_image(tmp_path, "tag:latest", stream)  # type: ignore[arg-type]

    mock_proc.kill.assert_called_once()
    mock_proc.wait.assert_called_once()


@pytest.mark.asyncio
async def test_get_local_docker_image_size_returns_size(docker_service: DockerService) -> None:
    mock_result = MagicMock()
    mock_result.stdout = "123456789\n"

    with patch("server.docker.Utils.run_command_for_success", new=AsyncMock(return_value=mock_result)):
        result = await docker_service.get_local_docker_image_size("ubuntu:latest")

    assert result == 123456789


@pytest.mark.asyncio
async def test_get_local_docker_image_size_returns_none_on_exception(docker_service: DockerService) -> None:
    with patch("server.docker.Utils.run_command_for_success", new=AsyncMock(side_effect=RuntimeError("not found"))):
        result = await docker_service.get_local_docker_image_size("nonexistent:image")

    assert result is None


@pytest.mark.asyncio
async def test_external_only_service_does_not_read_docker_auth_config() -> None:
    """Reading ~/.docker/config.json is a Docker interaction; external-only promises none."""
    config = MagicMock()
    config.external_only = True

    def _fail_auths() -> dict[str, str]:
        raise AssertionError("docker auth config must not be read in external-only mode")

    with (
        patch("server.docker.get_docker_auths", side_effect=_fail_auths),
        patch("server.docker.get_os", return_value="linux"),
        patch("server.docker.get_cpu_architecture", return_value="amd64"),
        patch("server.docker.platform.machine", return_value="x86_64"),
    ):
        service = await create_docker_service(MagicMock(), config)

    assert service.auths == {}
    assert service.external_only is True


@pytest.mark.asyncio
async def test_external_only_docker_operations_fail_deliberately() -> None:
    """A Docker call in external-only mode is a bug, and must say so rather than IndexError."""
    config = MagicMock()
    config.external_only = True

    with (
        patch("server.docker.get_docker_auths", return_value={}),
        patch("server.docker.get_os", return_value="linux"),
        patch("server.docker.get_cpu_architecture", return_value="amd64"),
        patch("server.docker.platform.machine", return_value="x86_64"),
    ):
        service = await create_docker_service(MagicMock(), config)

    with pytest.raises(DockerDisabledError, match="DF_EXTERNAL_ONLY"):
        _ = service._docker_bin  # type: ignore[reportPrivateUsage]


def test_normal_mode_still_resolves_the_docker_binary() -> None:
    service = DockerService(
        MagicMock(),
        MagicMock(),
        "docker compose",
        has_gpu_support=False,
        os="linux",
        architecture="amd64",
        is_rootless=False,
        host_platform="linux/amd64",
    )
    assert service._docker_bin == "docker"  # type: ignore[reportPrivateUsage]
