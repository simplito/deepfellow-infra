"""Docker backend."""

import asyncio
import json
import shutil
from pathlib import Path
from typing import NotRequired, TypedDict

import yaml
from pydantic import BaseModel

from server.applicationcontext import ApplicationContext
from server.utils.core import CommandResult2, Utils
from server.utils.exceptions import AppError


class DockerOptions:
    def __init__(
        self,
        name: str,
        image: str,
        image_port: int,
        command: str | None = None,
        use_gpu: bool = False,
        volumes: list[str] | None = None,
        restart: str | None = None,
        env_vars: dict[str, str] | None = None,
        api_endpoint: str | None = None,
        ulimits: dict[str, str] | None = None,
        shm_size: str | None = None,
        entrypoint: str | None = None,
        healthcheck: str | None = None,
        reset_uid: bool = False,
        subnet: str | None = None,
    ):
        self.name = name
        self.image = image
        self.command = command
        self.image_port = image_port
        self.env_vars = env_vars or {}
        self.api_endpoint = api_endpoint
        self.service_name = Utils.sanitize_service_name(name)
        self.restart = restart
        self.volumes = volumes
        self.use_gpu = use_gpu
        self.uliumits = ulimits
        self.shm_size = shm_size
        self.entrypoint = entrypoint
        self.healthcheck = healthcheck
        self.reset_uid = reset_uid
        self.subnet = subnet


class DockerNotInstalledError(Exception):
    pass


class DockerComposeStatus(BaseModel):
    success: bool
    info: str


class DockerComposeDevice(TypedDict):
    driver: str
    count: NotRequired[int]
    device_ids: NotRequired[list[int]]
    capabilities: list[str]


class DockerComposeReservations(TypedDict):
    devices: list[DockerComposeDevice]


class DockerComposeResource(TypedDict):
    reservations: DockerComposeReservations


class DockerComposeDeploy(TypedDict):
    resources: DockerComposeResource


class DockerComposeService(TypedDict):
    image: str
    ports: NotRequired[list[str]]
    environment: NotRequired[dict[str, str]]
    healthcheck: NotRequired[str]
    command: NotRequired[str]
    volumes: NotRequired[list[str]]
    restart: NotRequired[str]
    shm_size: NotRequired[str]
    entrypoint: NotRequired[str]
    user: NotRequired[str]
    deploy: NotRequired[DockerComposeDeploy]
    networks: NotRequired[list[str]]


class DockerComposeNetwork(TypedDict):
    external: bool


class DockerComposeContent(TypedDict):
    services: dict[str, DockerComposeService]
    networks: NotRequired[dict[str, DockerComposeNetwork]]


class DockerImage(BaseModel):
    name: str
    size: str


_docker_compose_cmd: str | None = None
_has_gpu_support: bool | None = None


def get_docker_compoes_cmd() -> str:
    """Return docker compose command."""
    global _docker_compose_cmd
    if _docker_compose_cmd is None:
        if shutil.which("docker-compose"):
            _docker_compose_cmd = "docker-compose"
        elif shutil.which("docker"):
            _docker_compose_cmd = "docker compose"
        else:
            raise DockerNotInstalledError("Docker is not installed.")
    return _docker_compose_cmd


async def has_gpu_support() -> bool:
    """Return whether there is GPU support."""
    global _has_gpu_support
    if _has_gpu_support is None:
        result = await Utils.run_command("docker run --gpus all --rm busybox echo")
        _has_gpu_support = result.exit_code == 0
    return _has_gpu_support


async def start_docker_compose(docker_compose_file_path: Path) -> CommandResult2:
    """Start given docker compose."""
    docker_compose_cmd = get_docker_compoes_cmd()
    cmd_parts = [*docker_compose_cmd.split(), "-f", str(docker_compose_file_path), "up", "-d"]
    command = " ".join(Utils.shell_escape(part) for part in cmd_parts)
    return await Utils.run_command_for_success(command)


async def stop_docker_compose(docker_compose_file_path: Path) -> None:
    """Stop given docker compose."""
    docker_compose_cmd = get_docker_compoes_cmd()
    cmd_parts = [*docker_compose_cmd.split(), "-f", str(docker_compose_file_path), "down", "--remove-orphans"]
    command = " ".join(Utils.shell_escape(part) for part in cmd_parts)
    await Utils.run_command_for_success(command)


async def docker_compose_status(docker_compose_file_path: Path) -> DockerComposeStatus:
    """Get status for given docker compose."""
    docker_compose_cmd = get_docker_compoes_cmd()
    res = await Utils.run_command(f"{docker_compose_cmd} -f {docker_compose_file_path} logs")
    if res.exit_code == 1 and res.stderr.strip() == f"Error: file '{docker_compose_file_path}' not found":
        return DockerComposeStatus(success=True, info="not found")
    return DockerComposeStatus(success=True, info=res.stdout)


async def is_docker_compose_running(docker_compose_file_path: Path, service_name: str) -> bool:
    """Check whether the service from given docker compose is running."""
    docker_compose_cmd = get_docker_compoes_cmd()
    if not docker_compose_file_path.exists():
        return False
    try:
        cmd = f"{docker_compose_cmd} -f {docker_compose_file_path} ps --services --filter status=running"
        result = await Utils.run_command(cmd)
        return service_name in result.stdout  # noqa: TRY300
    except Exception:
        return False


async def docker_pull(image: str) -> bool:
    """Pull given docker image."""
    try:
        cmd = f"docker pull {image}"
        result = await Utils.run_command(cmd)
        return result.exit_code == 0  # noqa: TRY300
    except Exception:
        return False


async def is_docker_compose_healthy(docker_compose_file_path: Path, service_name: str) -> bool:
    """Check whether the service from given docker compose is healthy."""
    docker_compose_cmd = get_docker_compoes_cmd()
    if not docker_compose_file_path.exists():
        print(f"{docker_compose_file_path} not found")
        return False

    try:
        cmd = f"{docker_compose_cmd} -f {docker_compose_file_path} ps {service_name} --format json"
        result = await Utils.run_command(cmd)

        if result.exit_code != 0:
            print(f"{docker_compose_file_path} {cmd} exit code is not 0. Exit code is {result.exit_code}")
            return False

        container = json.loads(result.stdout)

        health = container.get("Health", "").lower()
        if "unhealthy" in health:
            print(f"Docker container {service_name} is unhealthy")
            return False
        if "healthy" in health:
            return True

        state = container.get("State", "").lower()
        if state == "running":
            return True

        print(f"Docker container {service_name} is in state {state}")
        return False  # noqa: TRY300
    except Exception as exc:
        print(f"Error while checking health of docker container {service_name}. Error: {exc}")
        return False


async def generate_docker_compose_content(options: DockerOptions, port: int) -> DockerComposeContent:  # noqa: C901
    """Generate docker compose content."""
    service: DockerComposeService = {
        "image": options.image,
        "environment": options.env_vars,
    }
    if not options.subnet:
        service["ports"] = [f"{port}:{options.image_port}"]
    if options.healthcheck:
        service["healthcheck"] = options.healthcheck
    if options.command:
        service["command"] = options.command
    if options.volumes:
        service["volumes"] = options.volumes
    if options.restart:
        service["restart"] = options.restart
    if options.shm_size:
        service["shm_size"] = options.shm_size
    if options.entrypoint:
        service["entrypoint"] = options.entrypoint
    if options.reset_uid:
        service["user"] = "0:0"
    if options.use_gpu:
        if not await has_gpu_support():
            raise AppError("Docker doesn't support GPU on this machine.")
        service["deploy"] = {"resources": {"reservations": {"devices": [{"driver": "nvidia", "count": 1, "capabilities": ["gpu"]}]}}}
    if options.subnet:
        service["networks"] = [options.subnet]
    docker_compose_content: DockerComposeContent = {"services": {options.service_name: service}}
    if options.subnet:
        docker_compose_content["networks"] = {options.subnet: {"external": True}}
    return docker_compose_content


async def has_docker_compose_difference(docker_compose_file_path: Path, options: DockerOptions) -> tuple[bool, int | None]:
    """Check whether there is any differences between given file and the one generated from options.

    Returns flag and port gathered from given file.
    """
    if not docker_compose_file_path.exists():
        return True, None

    try:
        # Read current docker compose file
        current_content = docker_compose_file_path.read_text()
        current_config = yaml.safe_load(current_content)

        # # Generate desired configuration
        current_service = current_config.get("services", {}).get(options.service_name, {})
        current_ports = current_service.get("ports", [])
        current_port = int(current_ports[0].split(":")[0])

        desired_config = await generate_docker_compose_content(options, current_port)
        desired_content = yaml.dump(desired_config, default_flow_style=False, sort_keys=False)
        # Check image, command, environment
        return (current_content != desired_content, current_port)  # noqa: TRY300
    except Exception:
        return True, None


async def render_docker_compose(docker_compose_file_path: Path, options: DockerOptions, ctx: ApplicationContext) -> int:
    """Generate docker compose content and save it under given path, it also retrieve free port and returns it."""
    port = None
    # Check if old port is occupied and get a new one if needed
    if docker_compose_file_path.exists():
        try:
            current_content = docker_compose_file_path.read_text()
            current_config = yaml.safe_load(current_content)
            current_service = current_config.get("services", {}).get(options.service_name, {})
            current_ports = current_service.get("ports", [])

            if current_ports:
                current_port = int(current_ports[0].split(":")[0])
                # Check if port is still available
                if ctx.is_port_available(current_port):
                    port = current_port
        except Exception:
            pass
    # Get new port
    if port is None:
        port = ctx.get_free_port()
    docker_compose_file_content = await generate_docker_compose_content(options, port)

    # Save rendered docker compose as file
    docker_compose_yaml = yaml.dump(docker_compose_file_content, default_flow_style=False, sort_keys=False)
    Utils.save_file(docker_compose_file_path.parent / (options.name + ".yaml"), docker_compose_yaml)

    return port


async def install_and_run_docker(ctx: ApplicationContext, options: DockerOptions) -> int:
    """Run docker compose and return port under it works."""
    port = None
    docker_compose_dir = ctx.get_docker_compose_dir()
    docker_compose_file_path = docker_compose_dir / (options.name + ".yaml")
    service_name = options.service_name

    # Check if docker compose is working
    is_running = await is_docker_compose_running(docker_compose_file_path, service_name)

    # Check if there would be a difference in docker compose
    has_difference, port = await has_docker_compose_difference(docker_compose_file_path, options)

    # print(f"{service_name}\n{is_running=}\n{has_difference=}\n")
    start_output = ""

    # Handle different scenarios based on running state, health, and differences
    if not is_running and not has_difference:
        # Not running, no difference -> start
        start_output = await start_docker_compose(docker_compose_file_path)
    elif not is_running and has_difference:
        # Not running, has difference -> render then start
        port = await render_docker_compose(docker_compose_file_path, options, ctx)
        start_output = await start_docker_compose(docker_compose_file_path)
    elif is_running and has_difference:
        # Running but has difference -> stop -> render -> start
        # print(f"{service_name} config was changed. Restarting...")
        await stop_docker_compose(docker_compose_file_path)
        port = await render_docker_compose(docker_compose_file_path, options, ctx)
        start_output = await start_docker_compose(docker_compose_file_path)

    # Check if container is healthy after starting
    is_healthy = await is_docker_compose_healthy(docker_compose_file_path, service_name)
    retry_count = 0
    max_retries = 3
    while not is_healthy and retry_count < max_retries:
        print(
            f"Container {options.name} wasn't healthy. Restarting and allocating a new port."
            f"(Attempt {retry_count + 1}/{max_retries})\n Error: {start_output}"
        )
        await stop_docker_compose(docker_compose_file_path)
        port = await render_docker_compose(docker_compose_file_path, options, ctx)
        start_output = await start_docker_compose(docker_compose_file_path)

        # Wait before checking health again
        await asyncio.sleep(2)

        is_healthy = await is_docker_compose_healthy(docker_compose_file_path, service_name)
        retry_count += 1

    if not is_healthy:
        msg = f"Container {options.name} failed to become healthy after {max_retries} attempts"
        raise AppError(msg)

    if port is None:
        raise AppError("Cannot register service.")

    return port


async def uninstall_docker(ctx: ApplicationContext, options: DockerOptions) -> None:
    """Stop docker compose and remove the file."""
    docker_compose_cmd = get_docker_compoes_cmd()
    docker_compose_file = Path(ctx.get_docker_compose_dir() / (options.name + ".yaml"))
    await Utils.run_command(f"{docker_compose_cmd} -f {docker_compose_file} down")
    if docker_compose_file.is_file():
        docker_compose_file.unlink()
