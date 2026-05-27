# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
import yaml
from aiodocker import DockerError

import server.docker as docker_mod
from server.docker import (
    DockerImageNameInfo,
    DockerNotInstalledError,
    DockerOptions,
    DockerPath,
    DockerService,
    create_docker_service,
    get_docker_auths,
    normalize_docker_platform,
)
from server.utils.core import CommandResult, CommandResult2
from server.utils.exceptions import AppError, DockerImageAuthorizationError, DockerImageDoesNotExistError
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


def _opts(
    name: str = "mymodel",
    image: str = "ubuntu:latest",
    image_port: int = 8080,
    **kwargs: Any,
) -> DockerOptions:
    return DockerOptions(name=name, container_name=None, image=image, image_port=image_port, **kwargs)


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

    with patch("server.docker.Utils.run_command_for_success", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = make_result2()

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

    with patch("server.docker.Utils.run_command_for_success", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = make_result2()
        await docker_service.restart_docker_compose(compose_file)

    cmd = mock_run.call_args[0][0]
    assert "restart" in cmd


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
async def test_stop_docker_does_not_call_stop_compose_when_file_missing(docker_service: DockerService, tmp_path: Path) -> None:
    compose_file = tmp_path / "missing.yaml"
    options = _opts()

    with (
        patch.object(docker_service, "get_docker_compose_file_path", return_value=compose_file),
        patch.object(docker_service, "stop_docker_compose", new_callable=AsyncMock) as mock_stop,
    ):
        await docker_service.stop_docker(options)

    assert mock_stop.call_count == 0


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


@pytest.mark.asyncio
async def test_remove_image_reraises_non_404(docker_service: DockerService) -> None:
    images = MagicMock()
    images.delete = AsyncMock(side_effect=DockerError(status=500, message="server error"))
    cm, _ = _make_docker_mock(images)

    with patch("server.docker.Docker", return_value=cm), pytest.raises(DockerError):
        await docker_service.remove_image("ubuntu:latest")


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
        port = await docker_service.install_and_run_docker(options)

    assert mock_start.call_count == 1
    assert mock_create.call_count == 0
    assert port == 11434


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
        port = await docker_service.install_and_run_docker(options)

    assert mock_create.call_count == 1
    assert mock_start.call_count == 1
    assert port == 22222


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
        port = await docker_service.install_and_run_docker(options)

    assert mock_create.call_count == 1
    assert mock_start.call_count == 1
    assert port == 55555


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
        port = await docker_service.install_and_run_docker(options)

    assert mock_stop.call_count == 1
    assert mock_create.call_count == 1
    assert mock_start.call_count == 1
    assert port == 66666


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
        port = await docker_service.install_and_run_docker(options)

    assert mock_stop.call_count == 0
    assert mock_create.call_count == 0
    assert mock_start.call_count == 0
    assert port == 11434


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
        port = await docker_service.install_and_run_docker(options)

    assert port == -1


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
async def test_uninstall_docker_file_not_present_skips_unlink(docker_service: DockerService, tmp_path: Path) -> None:
    options = _opts()
    compose_file = tmp_path / "nonexistent.yaml"  # does not exist
    with (
        patch.object(docker_service, "get_docker_compose_file_path", return_value=compose_file),
        patch("server.docker.Utils.run_command", new_callable=AsyncMock) as mock_run,
    ):
        mock_run.return_value = make_result()

        await docker_service.uninstall_docker(options)

    assert mock_run.call_count == 1
    assert not compose_file.exists()


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
