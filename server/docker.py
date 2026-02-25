# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Docker backend."""

import grp
import json
import logging
import os
import platform
import re
import shutil
from collections.abc import AsyncGenerator, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NotRequired, TypedDict

import yaml
from aiodocker import Docker, DockerError
from pydantic import BaseModel

from server.config import AppSettings
from server.portservice import PortService
from server.utils.core import CommandResult2, Utils, get_cpu_architecture, get_os
from server.utils.exceptions import AppError, DockerImageAuthorizationError, DockerImageDoesNotExistError
from server.utils.hardware import HardwarePartInfo, IntelGpuInfo, NvidiaGpuInfo
from server.utils.loading import Progress
from server.utils.logger import uvicorn_logger

logger = logging.getLogger("uvicorn.error")

ARCH_ALIASES = {
    "x86_64": ("amd64", None),
    "aarch64": ("arm64", "v8"),
    "armhf": ("arm", "v7"),
    "armv7l": ("arm", "v7"),
    "armv7": ("arm", "v7"),
    "i386": ("386", None),
}

ARCHES_WITH_REQUIRED_VARIANT = ["arm", "arm64"]

DEFAULT_VARIANTS = {
    "arm": "v7",
    "arm64": "v8",
}


def normalize_docker_platform(platform_str: str) -> str:
    """Normalize the docker platform to format os/arch[/variant] with aliases."""
    parts = platform_str.split("/")
    if len(parts) == 2:
        os_part, arch_part = parts
        variant_part = None
    elif len(parts) == 3:
        os_part, arch_part, variant_part = parts
    else:
        msg = f"Invalid platform format: {platform_str}"
        raise ValueError(msg)

    (arch_normalized, variant_x) = ARCH_ALIASES.get(arch_part, (arch_part, variant_part))
    if variant_part and variant_x is not None and variant_x != variant_part:
        msg = f"Platform '{platform_str}' has mismatched variant '{arch_normalized}' '{variant_part}' != '{variant_x}"
        raise ValueError(msg)
    variant_part = variant_part or variant_x
    variant_normalized = variant_part if variant_part else DEFAULT_VARIANTS.get(arch_normalized)
    if arch_normalized in ARCHES_WITH_REQUIRED_VARIANT and not variant_normalized:
        msg = f"Platform '{platform_str}' requires a variant for architecture '{arch_normalized}'"
        raise ValueError(msg)
    return f"{os_part}/{arch_normalized}/{variant_normalized}" if variant_normalized else f"{os_part}/{arch_normalized}"


def get_docker_auths() -> dict[str, str]:
    """Get docker auth."""
    config_path = Path.home() / ".docker" / "config.json"
    if not config_path.exists():
        return {}
    try:
        # .read_text() handles opening and closing the file automatically
        config: dict[str, Any] = json.loads(config_path.read_text())
        auths_raw: dict[str, dict[str, str]] = config.get("auths", {})
        return {host: host_data["auth"] for host, host_data in auths_raw.items() if host_data.get("auth")}
    except (OSError, json.JSONDecodeError):
        return {}


@dataclass(frozen=True)
class DockerImageNameInfo:
    registry: str
    namespace: str
    image_name: str

    @classmethod
    def parse(cls, full_image: str) -> "DockerImageNameInfo":
        """Parse docker image name to registry, namespace and image_name."""
        parts = full_image.split("/")

        registry = "docker.io"  # Default
        namespace = "library"  # Default for official Docker Hub images
        image_name = ""

        # Check if the first part is a registry host
        # Registries usually have a '.' (ghcr.io) or a ':' (localhost:5000)
        if len(parts) > 1 and ("." in parts[0] or ":" in parts[0]):
            registry = parts[0]
            remaining = parts[1:]
        else:
            remaining = parts

        # Handle namespace and image name
        if len(remaining) == 2:
            namespace = remaining[0]
            image_name = remaining[1]
        elif len(remaining) == 1:
            image_name = remaining[0]
        else:
            # For complex paths like ECR or deeply nested registries
            namespace = "/".join(remaining[:-1])
            image_name = remaining[-1]

        return cls(registry, namespace, image_name)


class DockerOptions:
    def __init__(
        self,
        name: str,
        container_name: str | None,
        image: str,
        image_port: int,
        command: str | None = None,
        hardware: Sequence[HardwarePartInfo] | None = None,
        volumes: list[str] | None = None,
        restart: str | None = None,
        env_vars: dict[str, str] | None = None,
        api_endpoint: str | None = None,
        ulimits: dict[str, str] | None = None,
        shm_size: str | None = None,
        entrypoint: str | None = None,
        healthcheck: dict[str, str] | None = None,
        user: str | None = None,
        subnet: str | None = None,
    ):
        self.name = name
        self.image = image
        self.command = command
        self.image_port = image_port
        self.env_vars = env_vars or {}
        self.api_endpoint = api_endpoint
        self.service_name = Utils.sanitize_service_name(name)
        self.container_name = container_name
        self.restart = restart
        self.volumes = volumes
        self.hardware = hardware
        self.uliumits = ulimits
        self.shm_size = shm_size
        self.entrypoint = entrypoint
        self.healthcheck = healthcheck
        self.user = user
        self.subnet = subnet


class DockerNotInstalledError(Exception):
    pass


class DockerComposeStatus(BaseModel):
    success: bool
    info: str


class DockerComposeDevice(TypedDict):
    driver: str
    count: NotRequired[int]
    device_ids: NotRequired[list[str]]
    capabilities: list[str]


class DockerComposeReservations(TypedDict):
    devices: list[DockerComposeDevice]


class DockerComposeResource(TypedDict):
    reservations: DockerComposeReservations


class DockerComposeDeploy(TypedDict):
    resources: DockerComposeResource


class DockerComposeService(TypedDict):
    image: str
    container_name: NotRequired[str]
    ports: NotRequired[list[str]]
    environment: NotRequired[dict[str, str]]
    healthcheck: NotRequired[dict[str, str]]
    command: NotRequired[str]
    volumes: NotRequired[list[str]]
    restart: NotRequired[str]
    shm_size: NotRequired[str]
    entrypoint: NotRequired[str]
    user: NotRequired[str]
    deploy: NotRequired[DockerComposeDeploy]
    devices: NotRequired[list[str]]
    group_add: NotRequired[list[str]]
    networks: NotRequired[list[str]]


class DockerComposeNetwork(TypedDict):
    external: bool


class DockerComposeContent(TypedDict):
    services: dict[str, DockerComposeService]
    networks: NotRequired[dict[str, DockerComposeNetwork]]


class DockerImage(BaseModel):
    name: str
    size: str


class DockerService:
    auths: dict[str, str]

    def __init__(
        self,
        config: AppSettings,
        port_service: PortService,
        docker_compose_cmd: str,
        has_gpu_support: bool,
        os: str,
        architecture: str,
        is_rootless: bool,
        host_platform: str,
    ):
        self.config = config
        self.port_service = port_service
        self.docker_compose_cmd = docker_compose_cmd
        self.has_gpu_support = has_gpu_support
        self.is_rootless = is_rootless
        self.os = os
        self.architecture = architecture
        self.host_platform = host_platform
        self.auths = get_docker_auths()

    def calculate_total_layer_size(self, manifest: dict[str, Any]) -> int:
        """Parse an docker image manifest and calculates the total size.

        Sum up the 'size' of all entries in the 'layers' array.

        Args:
            manifest: data in format:

        Returns:
            int: total size in bytes
        """
        layers = manifest.get("layers")

        if not layers:
            logger.debug("Image Manifest JSON structure missing 'layers' array.")
            return 0

        total_size = 0
        for layer in layers:
            layer_size = layer.get("size", 0)
            # We only count layer sizes that are integers (i.e., not missing)
            if isinstance(layer_size, int):
                total_size += layer_size

        return total_size

    def get_platform_digest(self, manifest: dict[str, Any]) -> str | None:
        """Parse the docker platform index and finds the digest matching the target platform.

        Cases:
            1. There is no manifests, there are layers -> return None.
            2. There is only one manifest and we return sha from it.
            3. There is manifest with same os and same architecture.
            4. There is manifest with same architecture but different os and there is no manifest with same os and architecture.
            5. We use first manifest with unknown os and unknown architecture when above are not met
            6. We use first manifest when when above are not met

        Args:
            manifest: return from docker builx imagetools inspect --raw {image}

        Returns:
            full docker sha with sha256:

        """
        manifests = manifest.get("manifests", [])

        if not manifests:
            return None

        # If there is only one digest then we the take it.
        if len(manifests) == 1:
            return manifests[0].get("digest")

        first_unknown_digest = None
        matching_architecture_digest = None

        for manifest in manifests:
            # Check for a precise platform match
            platform_info = manifest.get("platform")
            if platform_info:
                manifest_architecture = platform_info.get("architecture")
                manifest_os = platform_info.get("os")
                digest = manifest.get("digest")

                if manifest_architecture == self.architecture:
                    if manifest_os == self.os:
                        # If os and architecture agree then we take it.
                        return digest

                    # if os doesn't agree we might use it later
                    matching_architecture_digest = digest

                # Capture the first 'unknown/unknown' for the fallback logic
                if manifest_os == "unknown" and manifest_architecture == "unknown" and first_unknown_digest is None:
                    first_unknown_digest = digest

        # Fallback logic based on user request:
        # 1. Use digest with right architecture
        if matching_architecture_digest:
            return matching_architecture_digest

        # 1. Use the first 'unknown/unknown' digest found.
        if first_unknown_digest:
            return first_unknown_digest

        # 2. Use the first digest in the whole list if no exact match and no unknown/unknown match.
        #    We re-parse the data or check the first entry if the list isn't empty.
        return manifests[0].get("digest")

    async def get_docker_manifest(self, image: str) -> dict[str, Any]:
        """Get docker indexes list.

        Args:
            image: docker image ex. ubuntu

        Returns:
            output from docker builx imagetools inspect --raw {image} in python dict format.
        """
        cmd_parts = ["docker", "buildx", "imagetools", "inspect", image, "--raw"]
        cmd = " ".join(Utils.shell_escape(part) for part in cmd_parts)
        output = await Utils.run_command(cmd)
        if output.exit_code != 0:
            if "not found" in output.stderr:
                raise DockerImageDoesNotExistError(image)
            if "authorization failed" in output.stderr or "failed to authorize" in output.stderr:
                raise DockerImageAuthorizationError(image)
            raise RuntimeError("Invalid exit code for command", (output.exit_code, cmd, output.stdout, output.stderr))
        if output.stderr:
            logger.exception(output.stderr)
            raise AppError("Something went wrong. Check logs.")

        try:
            output_json = dict(json.loads(output.stdout))
        except json.JSONDecodeError as err:
            logger.exception("Error when con went to json")
            raise AppError("Something went wrong. Check logs.") from err

        return output_json

    def replace_image_digest(self, image: str, digest: str | None) -> str:
        """Replace docker image sha or add it on the end if it is."""
        if digest:
            if "sha256:" in image:
                image = re.sub("sha256:.*", "", image)
            return f"{image}@{digest}"
        return image

    async def get_docker_image_size(self, image: str) -> int:
        """Get docker image size in bytes."""
        platforms_manifest = await self.get_docker_manifest(image)
        platform_digest = self.get_platform_digest(platforms_manifest)
        platform_image = image
        if platform_digest:
            platform_image = self.replace_image_digest(image, platform_digest)
        layers_manifest = await self.get_docker_manifest(platform_image)
        return self.calculate_total_layer_size(layers_manifest)

    async def get_image_platforms(self, image: str) -> list[str]:
        """Get platform of a Docker image in format 'os/architecture'.

        Args:
            image: Docker image name (e.g., 'ubuntu:latest', 'vllm/vllm-openai:latest')

        Returns:
            Platform string like 'linux/arm64' or 'linux/amd64', or None if image not found.
        """
        try:
            # Inspect the image to get its metadata
            result = await Utils.run_command_for_success(f"docker image inspect {image}")

            # Parse JSON output
            image_data = json.loads(result.stdout)

            if not image_data or len(image_data) == 0:
                return []
        except Exception:
            main_manifest = await self.get_docker_manifest(image)
            manifests = main_manifest.get("manifests", [])
            if not manifests:
                if (
                    main_manifest.get("mediaType") == "application/vnd.docker.distribution.manifest.v2+json"
                    and (config := main_manifest.get("config"))
                    and isinstance(config, dict)
                    and (digest := config.get("digest"))  # type: ignore
                    and isinstance(digest, str)
                ):
                    new_image = image.split("@")[0] + "@" + digest
                    try:
                        image_manifest = await self.get_docker_manifest(new_image)
                        os = image_manifest.get("os", "")
                        architecture = image_manifest.get("architecture", "")
                        variant = image_manifest.get("variant", "")
                    except RuntimeError:
                        return []
                    return [normalize_docker_platform(f"{os}/{architecture}{'/' + variant if variant else ''}")]

                return []
            platforms = set[str]()
            for manifest in manifests:
                platform_info = manifest.get("platform")
                if platform_info:
                    variant = platform_info.get("variant", "")
                    os = platform_info.get("os", "")
                    architecture = platform_info.get("architecture", "")
                    if os != "unknown" and architecture != "unknown":
                        image = normalize_docker_platform(f"{os}/{architecture}{'/' + variant if variant else ''}")
                        platforms.add(image)
            return list(platforms)
        else:
            # Get OS and Architecture from the first image
            os_type = image_data[0].get("Os", "linux").lower()
            arch = image_data[0].get("Architecture", "").lower()
            return [normalize_docker_platform(f"{os_type}/{arch}")] if arch else []

    async def get_image_warnings(self, image: str) -> list[str]:
        """Get warnings about run given image on current platform."""
        warnings: list[str] = []
        try:
            platforms = await self.get_image_platforms(image)
            if self.host_platform not in platforms and platforms:
                warnings.append(
                    f"The docker image {image} is not compatible with the current system, "
                    f"platform mismatch, {self.host_platform} not in [{','.join(platforms)}]."
                )
        except DockerImageDoesNotExistError:
            warnings.append(f"Docker image does not exist {image}.")
        except DockerImageAuthorizationError:
            warnings.append(f"Cannot access docker image, authorization failed {image}.")
        return warnings

    async def get_user_for_docker(self) -> str:
        """Get user for docker."""
        return "0:0" if self.is_rootless else f"{os.getuid()}:{os.getgid()}"

    async def start_docker_compose(self, docker_compose_file_path: Path) -> CommandResult2:
        """Start given docker compose."""
        docker_compose_cmd = self.docker_compose_cmd
        cmd_parts = [*docker_compose_cmd.split(), "-f", str(docker_compose_file_path), "up", "-d", "--wait"]
        command = " ".join(Utils.shell_escape(part) for part in cmd_parts)
        return await Utils.run_command_for_success(command)

    async def stop_docker(self, options: DockerOptions) -> None:
        """Stop docker."""
        docker_compose_file_path = self.get_docker_compose_file_path(options.name)
        if docker_compose_file_path.exists():
            await self.stop_docker_compose(docker_compose_file_path)

    async def stop_docker_compose(self, docker_compose_file_path: Path) -> None:
        """Stop given docker compose."""
        docker_compose_cmd = self.docker_compose_cmd
        cmd_parts = [*docker_compose_cmd.split(), "-f", str(docker_compose_file_path), "down", "--remove-orphans"]
        command = " ".join(Utils.shell_escape(part) for part in cmd_parts)
        await Utils.run_command_for_success(command)

    async def restart_docker_compose(self, docker_compose_file_path: Path) -> None:
        """Restart given docker compose."""
        docker_compose_cmd = self.docker_compose_cmd
        cmd_parts = [*docker_compose_cmd.split(), "-f", str(docker_compose_file_path), "restart"]
        command = " ".join(Utils.shell_escape(part) for part in cmd_parts)
        await Utils.run_command_for_success(command)

    async def get_docker_compose_logs(self, docker_compose_file_path: Path) -> str:
        """Get docker compose logs."""
        docker_compose_cmd = self.docker_compose_cmd
        cmd_parts = [*docker_compose_cmd.split(), "-f", str(docker_compose_file_path), "logs"]
        command = " ".join(Utils.shell_escape(part) for part in cmd_parts)
        result = await Utils.run_command_for_success(command)
        return result.stdout

    async def run_command_docker_compose(self, filepath: Path, service_name: str, command: str) -> str:
        """Run command in docker compose service."""
        docker_compose_cmd = self.docker_compose_cmd
        cmd_parts = [*docker_compose_cmd.split(), "-f", str(filepath), "exec", service_name, *command.split(" ")]
        command = " ".join(Utils.shell_escape(part) for part in cmd_parts)
        result = await Utils.run_command_for_success(command)
        return result.stdout

    async def docker_compose_status(self, docker_compose_file_path: Path) -> DockerComposeStatus:
        """Get status for given docker compose."""
        docker_compose_cmd = self.docker_compose_cmd
        res = await Utils.run_command(f"{docker_compose_cmd} -f {docker_compose_file_path} logs")
        if res.exit_code == 1 and res.stderr.strip() == f"Error: file '{docker_compose_file_path}' not found":
            return DockerComposeStatus(success=True, info="not found")
        return DockerComposeStatus(success=True, info=res.stdout)

    async def is_docker_compose_running(self, docker_compose_file_path: Path, service_name: str) -> bool:
        """Check whether the service from given docker compose is running."""
        docker_compose_cmd = self.docker_compose_cmd
        if not docker_compose_file_path.exists():
            return False
        try:
            cmd = f"{docker_compose_cmd} -f {docker_compose_file_path} ps --services --filter status=running"
            result = await Utils.run_command(cmd)
            return service_name in result.stdout  # noqa: TRY300
        except Exception:
            return False

    async def is_docker_image_pulled(self, full_image_name: str) -> bool:
        """Check whether given image is pulled."""
        try:
            async with Docker() as docker:
                await docker.images.get(full_image_name)
                return True
        except DockerError as e:
            # DockerError is raised for various API issues. A 404 status code
            # specifically means the resource (image) was not found.
            if e.status == 404:
                uvicorn_logger.info(f"Image '{full_image_name}' not found locally.")

        return False

    async def docker_pull(self, full_image_name: str, image_size: float) -> AsyncGenerator[float]:
        """Pull given docker image."""
        progress = Progress(image_size)
        bytes_per_id: dict[str, float] = {}
        additional_params = {}

        image_name_info = DockerImageNameInfo.parse(full_image_name)
        if image_name_info.registry in self.auths:
            additional_params = {"auth": self.auths[image_name_info.registry]}

        async with Docker() as docker:
            async for chunk in docker.images.pull(full_image_name, stream=True, timeout=24 * 60 * 60, **additional_params):
                if (
                    (id := chunk.get("id"))
                    and (chunk.get("status") == "Downloading")
                    and (value := chunk.get("progressDetail", {}).get("current"))
                ):
                    bytes_per_id[id] = value
                    image_bytes = sum(bytes_per_id.values())
                    progress.set_actual_value(image_bytes)
                    yield progress.get_percentage()

    async def remove_image(self, image_name: str) -> None:
        """Remove docker image."""
        try:
            async with Docker() as docker:
                await docker.images.delete(image_name, force=True)
        except DockerError as err:
            if err.status != 404:
                raise

    async def is_docker_compose_healthy(self, docker_compose_file_path: Path, service_name: str) -> bool:
        """Check whether the service from given docker compose is healthy."""
        docker_compose_cmd = self.docker_compose_cmd
        if not docker_compose_file_path.exists():
            msg = f"{docker_compose_file_path} not found"
            logger.debug(msg)
            return False

        try:
            cmd = f"{docker_compose_cmd} -f {docker_compose_file_path} ps {service_name} --format json"
            result = await Utils.run_command(cmd)

            if result.exit_code != 0:
                msg = f"{docker_compose_file_path} {cmd} exit code is not 0. Exit code is {result.exit_code}"
                logger.debug(msg)
                return False

            container = json.loads(result.stdout)

            health = container.get("Health", "").lower()
            if "unhealthy" in health:
                uvicorn_logger.warning(f"Docker container {service_name} is unhealthy")
                return False
            if "healthy" in health:
                return True

            state = container.get("State", "").lower()
            if state == "running":
                return True

            uvicorn_logger.info(f"Docker container {service_name} is in state {state}")
            return False  # noqa: TRY300
        except Exception as exc:
            uvicorn_logger.warning(f"Error while checking health of docker container {service_name}. Error: {exc}")
            return False

    async def generate_docker_compose_content(self, options: DockerOptions, port: int | None) -> DockerComposeContent:  # noqa: C901
        """Generate docker compose content."""
        service: DockerComposeService = {
            "image": options.image,
            "environment": options.env_vars,
        }
        if options.container_name:
            service["container_name"] = options.container_name
        if not options.subnet:
            if not port:
                raise AppError("Port is required when not in subnet mode")
            service["ports"] = [f"127.0.0.1:{port}:{options.image_port}"]
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
        if options.user:
            service["user"] = options.user
        if options.hardware:
            nvidia_gpus = [gpu for gpu in options.hardware if isinstance(gpu, NvidiaGpuInfo)]
            if nvidia_gpus:
                if not self.has_gpu_support:
                    raise AppError("Docker doesn't support GPU on this machine.")
                service["deploy"] = {
                    "resources": {
                        "reservations": {
                            "devices": [{"driver": "nvidia", "device_ids": [f"{gpu.id!s}" for gpu in nvidia_gpus], "capabilities": ["gpu"]}]
                        }
                    }
                }
            intel_gpus = [gpu for gpu in options.hardware if isinstance(gpu, IntelGpuInfo)]
            if intel_gpus:
                service["devices"] = ["/dev/dri:/dev/dri"]
                gids: list[str] = []
                for name in ("render", "video"):
                    with suppress(KeyError):
                        gids.append(str(grp.getgrnam(name).gr_gid))
                if gids:
                    service["group_add"] = gids
        if options.subnet:
            service["networks"] = [options.subnet]
        docker_compose_content: DockerComposeContent = {"services": {options.service_name: service}}
        if options.subnet:
            docker_compose_content["networks"] = {options.subnet: {"external": True}}
        return docker_compose_content

    async def has_docker_compose_difference(self, docker_compose_file_path: Path, options: DockerOptions) -> tuple[bool, int | None]:
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
            current_port = (
                int(
                    current_port_segments[0] if len(current_port_segments := current_ports[0].split(":")) == 2 else current_port_segments[1]
                )
                if len(current_ports) > 0
                else None
            )

            desired_config = await self.generate_docker_compose_content(options, current_port)
            desired_content = yaml.dump(desired_config, default_flow_style=False, sort_keys=False)
            # Check image, command, environment
            return (current_content != desired_content, current_port)  # noqa: TRY300
        except Exception:
            logger.exception("Error during checking docker compose differences")
            return True, None

    async def get_existing_or_free_port_docker(
        self,
        docker_compose_file_path: Path,
        options: DockerOptions,
    ) -> int:
        """Return existing or free port."""
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
                    if self.port_service.is_port_available(current_port):
                        port = current_port
            except Exception:
                pass
        # Get new port
        if port is None:
            port = self.port_service.get_free_port()

        return port

    async def create_compose_file(self, docker_compose_file_path: Path, options: DockerOptions) -> int:
        """Generate docker compose content and save it under given path, it also retrieve free port and returns it."""
        port = await self.get_existing_or_free_port_docker(docker_compose_file_path, options)
        docker_compose_file_content = await self.generate_docker_compose_content(options, port)
        docker_compose_yaml = yaml.dump(docker_compose_file_content, default_flow_style=False, sort_keys=False)
        Utils.save_file(docker_compose_file_path, docker_compose_yaml)

        return port

    async def install_and_run_docker(self, options: DockerOptions) -> int:
        """Run docker compose and return port under it works."""
        port = None
        docker_compose_file_path = self.get_docker_compose_file_path(options.name)
        service_name = options.service_name

        # Check if docker compose is working
        is_running = await self.is_docker_compose_running(docker_compose_file_path, service_name)

        # Check if there would be a difference in docker compose
        has_difference, port = await self.has_docker_compose_difference(docker_compose_file_path, options)

        # print(f"{service_name}\n{is_running=}\n{has_difference=}\n")
        start_output = ""

        # Handle different scenarios based on running state, health, and differences
        if not is_running and not has_difference:
            if port is not None and self.port_service.is_port_available(port):
                # Not running, no difference -> start
                start_output = await self.start_docker_compose(docker_compose_file_path)
            else:
                # Old port is taken -> render then start
                port = await self.create_compose_file(docker_compose_file_path, options)
                start_output = await self.start_docker_compose(docker_compose_file_path)
        elif not is_running and has_difference:
            # Not running, has difference -> render then start
            port = await self.create_compose_file(docker_compose_file_path, options)
            start_output = await self.start_docker_compose(docker_compose_file_path)
        elif is_running and has_difference:
            # Running but has difference -> stop -> render -> start
            # print(f"{service_name} config was changed. Restarting...")
            await self.stop_docker_compose(docker_compose_file_path)
            port = await self.create_compose_file(docker_compose_file_path, options)
            start_output = await self.start_docker_compose(docker_compose_file_path)

        # Check if container is healthy after starting
        is_healthy = await self.is_docker_compose_healthy(docker_compose_file_path, service_name)

        if not is_healthy:
            msg = f"Container {options.name} failed to become healthy, output: {start_output}"
            raise AppError(msg)

        if not port and options.subnet:
            # in subnet mode the port is not used so it could be anything, for example -1
            port = -1
        if port is None:
            raise AppError("Cannot register service.")

        return port

    async def uninstall_docker(self, options: DockerOptions) -> None:
        """Stop docker compose and remove the file."""
        docker_compose_cmd = self.docker_compose_cmd
        docker_compose_file = self.get_docker_compose_file_path(options.name)
        await Utils.run_command(f"{docker_compose_cmd} -f {docker_compose_file} down")
        if docker_compose_file.is_file():
            docker_compose_file.unlink()
        if self.config.compose_prefix:
            with suppress(Exception):
                docker_compose_file.parent.rmdir()

    def get_docker_compose_dir(self) -> Path:
        """Get docker compose dir."""
        dir = self.config.get_storage_dir() / "./config"
        if not dir.is_dir():
            dir.mkdir(parents=True)
        return dir

    def get_docker_compose_file_path(self, name: str) -> Path:
        """Get docker compose dir."""
        dir = self.get_docker_compose_dir()
        if not self.config.compose_prefix:
            return dir / (name + ".yaml")
        dir = dir / (self.config.compose_prefix + name)
        if not dir.is_dir():
            dir.mkdir(parents=True)
        return dir / "compose.yaml"

    def get_docker_container_name(self, name: str) -> str:
        """Return docker container name."""
        return self.config.container_name_prefix + name if self.config.container_name_prefix else name

    def get_docker_subnet(self) -> str | None:
        """Return docker subnet name or None if it is not set."""
        return self.config.docker_subnet if self.config.docker_subnet else None

    def get_container_host(self, subnet: str | None, container_name: str) -> str:
        """Return container_name if there is docker_subnet in config otherwise return localhost."""
        return container_name if subnet else "localhost"

    def get_container_port(self, subnet: str | None, exposed_port: int, original_port: int) -> int:
        """Return container_name if there is docker_subnet in config otherwise return localhost."""
        return original_port if subnet else exposed_port


async def create_docker_service(port_service: PortService, config: AppSettings) -> DockerService:
    """Create docker service."""

    async def get_docker_compose_cmd() -> str:
        """Return docker compose command."""
        if shutil.which("docker"):
            result = await Utils.run_command("docker compose version")
            if result.exit_code == 0:
                return "docker compose"
            if shutil.which("docker-compose"):
                return "docker-compose"

        raise DockerNotInstalledError("Docker is not installed.")

    async def has_gpu_support() -> bool:
        """Return whether there is GPU support."""
        result = await Utils.run_command("docker run --gpus all --rm busybox echo")
        return result.exit_code == 0

    async def is_rootless() -> bool:
        """Return whether docker is running in rootless mode."""
        result = await Utils.run_command("docker info")
        return result.exit_code == 0 and "rootless" in result.stdout

    def get_host_platform() -> str:
        """Get the current host platform normalized for docker."""
        arch = platform.machine().lower()
        return normalize_docker_platform(f"linux/{arch}")

    return DockerService(
        config,
        port_service,
        await get_docker_compose_cmd(),
        await has_gpu_support(),
        get_os(),
        get_cpu_architecture(),
        await is_rootless(),
        get_host_platform(),
    )
