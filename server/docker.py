# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Docker backend."""

import asyncio
import json
import logging
import os
import platform
import re
import shlex
import shutil
from collections.abc import AsyncGenerator, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NotRequired, TypedDict, cast

import yaml
from aiodocker import Docker, DockerError
from fastapi import HTTPException
from pydantic import BaseModel

from server.config import AppSettings
from server.portservice import PortService
from server.utils.core import CommandResult2, Stream, StreamChunk, StreamChunkProgress, Utils, get_cpu_architecture, get_os
from server.utils.exceptions import AppError, DockerComposeStartError, DockerImageAuthorizationError, DockerImageDoesNotExistError
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

# Cap on how many trailing characters of docker compose logs are surfaced in the short
# exception message shown in the WebUI when no line matches _ERROR_LINE_MARKERS.
_SHORT_LOG_CHARS = 500

# Cap on how many trailing characters of docker compose logs are written to logger.error.
# Much larger than _SHORT_LOG_CHARS since this is for server-side debugging, not the WebUI,
# but a failed model download can still emit megabytes of progress lines — stay bounded.
_MAX_LOGGED_CHARS = 20_000

# Substrings that mark a log line as actually describing the failure, as opposed to routine
# startup chatter — used to pick a short, relevant excerpt instead of an arbitrary tail cut.
_ERROR_LINE_MARKERS = ("error", "fatal", "exception", "traceback", "panic", "critical")

# How many matched error lines to keep, most recent first.
_MAX_ERROR_LINES = 5

# Substrings that indicate the NVIDIA container toolkit is not installed.
_GPU_TOOLKIT_PATTERNS = (
    "could not select device driver",
    "nvidia-container-cli",
    "nvidia container cli",
)

# Substrings that indicate the GPU driver itself has failed.
_GPU_DRIVER_PATTERNS = (
    "failed to initialize nvml",
    "no devices were found",
    "nvml error",
    "driver/library version mismatch",
)


def _diagnose_gpu_error(text: str) -> str | None:
    """Return a user-facing message if *text* contains a known GPU failure, else None."""
    lower = text.lower()
    if any(p in lower for p in _GPU_TOOLKIT_PATTERNS):
        return (
            "GPU is unavailable: the NVIDIA container toolkit is not installed on this host. "
            "Install it with: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
        )
    if any(p in lower for p in _GPU_DRIVER_PATTERNS):
        return (
            "GPU driver failure detected. The NVIDIA drivers may have crashed or been updated "
            "without a reboot. Run `nvidia-smi` on the host to verify driver status."
        )
    return None


# Matches Docker's actual name-conflict wording, e.g.:
#   Conflict. The container name "/ollama" is already in use by container "abc123"...
# Anchored on "container name ... is already in use" rather than the bare "is already in use"
# substring, so unrelated errors that happen to share that tail (e.g. port-bind failures) aren't
# misclassified as a name conflict and routed into orphan-adoption.
_CONTAINER_NAME_CONFLICT_RE = re.compile(r"container name .*is already in use", re.IGNORECASE)


def _short_log_snippet(logs: str) -> str:
    """Return the last _SHORT_LOG_CHARS characters of *logs*, for a bounded user-facing message."""
    if len(logs) <= _SHORT_LOG_CHARS:
        return logs
    return f"...(see server logs for full output)...\n{logs[-_SHORT_LOG_CHARS:]}"


def _bounded_for_logging(text: str) -> str:
    """Return the last _MAX_LOGGED_CHARS characters of *text*, so a runaway container can't flood the server log."""
    if len(text) <= _MAX_LOGGED_CHARS:
        return text
    return f"...(truncated, {len(text) - _MAX_LOGGED_CHARS} chars omitted)...\n{text[-_MAX_LOGGED_CHARS:]}"


def _extract_error_excerpt(logs: str) -> str:
    """Return the most relevant lines of *logs* for a short user-facing message.

    Picks lines that look like they actually describe the failure (see _ERROR_LINE_MARKERS)
    instead of an arbitrary tail cut, so unrelated startup chatter doesn't crowd out the cause.
    Falls back to a plain tail snippet when nothing matches.
    """
    matches = [line for line in logs.splitlines() if any(marker in line.lower() for marker in _ERROR_LINE_MARKERS)]
    if not matches:
        return _short_log_snippet(logs)
    return "\n".join(matches[-_MAX_ERROR_LINES:])


def _is_container_name_conflict(text: str) -> bool:
    """Return True if *text* indicates a docker container name collision."""
    return _CONTAINER_NAME_CONFLICT_RE.search(text) is not None


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
class ContainerStatus:
    exists: bool
    state: str
    health: str
    restart_count: int


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
        command: str | list[str] | None = None,
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
        self.ulimits = ulimits
        self.shm_size = shm_size
        self.entrypoint = entrypoint
        self.healthcheck = healthcheck
        self.user = user
        self.subnet = subnet


class DockerNotInstalledError(Exception):
    pass


class DockerDisabledError(Exception):
    """Raised when a Docker operation is attempted while DF_EXTERNAL_ONLY is set.

    External-only mode has no Docker daemon by design, so reaching Docker here is a
    programming error rather than an environment problem: it means a code path that
    should be unreachable in this mode was taken.
    """

    def __init__(self) -> None:
        super().__init__("Docker is disabled by DF_EXTERNAL_ONLY")


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
    command: NotRequired[str | list[str]]
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


class DockerPath(BaseModel):
    local_path: Path
    docker_path: Path

    def add(self, path: Path) -> "DockerPath":
        """Add path part."""
        return DockerPath(local_path=self.local_path / path, docker_path=self.docker_path / path)


def _extract_published_host_port(ports: dict[str, Any] | None, image_port: int) -> int | None:
    """Return the loopback-bound host port for f"{image_port}/tcp" in a container's NetworkSettings.Ports, else None.

    Only a binding whose HostIp is 127.0.0.1 counts - generate_docker_compose_content always publishes
    as "127.0.0.1:{port}:{image_port}", so a container exposed more broadly (e.g. to 0.0.0.0, reachable
    from the whole network) isn't the same configuration, even though its port is technically also
    reachable from the host.
    """
    if not ports:
        return None
    bindings = ports.get(f"{image_port}/tcp")
    if not bindings:
        return None
    for binding in bindings:
        if binding.get("HostIp") != "127.0.0.1":
            continue
        host_port = binding.get("HostPort")
        if not host_port:
            continue
        try:
            return int(host_port)
        except (TypeError, ValueError):
            continue
    return None


def _container_env_matches(container_env: list[str], expected: dict[str, str]) -> bool:
    """Return True if every key/value in *expected* is present with an identical value in *container_env*.

    *container_env* is a list of "KEY=VALUE" strings, as in a container's Config.Env. Extra
    entries in *container_env* beyond what's in *expected* (e.g. image-default env vars) are ignored.
    """
    actual: dict[str, str] = {}
    for entry in container_env:
        key, sep, value = entry.partition("=")
        if sep:
            actual[key] = value
    return all(actual.get(key) == value for key, value in expected.items())


def _parse_bind_volume(volume: str) -> tuple[str, str, bool] | None:
    """Parse a compose-style "host:container[:mode]" bind-mount string into (host_path, container_path, expect_rw)."""
    parts = volume.split(":")
    if len(parts) == 2:
        host, container = parts
        mode = "rw"
    elif len(parts) == 3:
        host, container, mode = parts
    else:
        return None
    return host, container, "ro" not in mode.split(",")


def _container_volumes_match(mounts: list[dict[str, Any]], expected_volumes: list[str]) -> bool:
    """Return True if every configured bind-mount volume string has a matching bind Mount on the live container."""
    binds = [mount for mount in mounts if mount.get("Type") == "bind"]
    for volume in expected_volumes:
        parsed = _parse_bind_volume(volume)
        if parsed is None:
            return False
        host, container, expect_rw = parsed
        if not any(
            mount.get("Source") == host and mount.get("Destination") == container and bool(mount.get("RW")) == expect_rw for mount in binds
        ):
            return False
    return True


def _container_command_matches(actual_cmd: list[str] | None, configured_command: str | list[str] | None) -> bool:
    """Return True if *configured_command* matches a container's live Config.Cmd.

    No command configured always matches - command wasn't part of what was requested, so there's
    nothing to compare. A configured list (e.g. a `bash -c "..."` wrapper) is compared as-is; a
    configured string is split with shell-word rules first, since that's how docker compose turns a
    string `command:` into the argv Docker actually records. A string that fails to split (unbalanced
    quotes - possible for the free-text "Docker command" field on custom/MCP services) never matches,
    same as any other mismatch.
    """
    if not configured_command:
        return True
    if isinstance(configured_command, list):
        expected_cmd = configured_command
    else:
        try:
            expected_cmd = shlex.split(configured_command)
        except ValueError:
            return False
    return actual_cmd == expected_cmd


_BYTE_SIZE_UNITS = {"b": 1, "k": 1024, "kb": 1024, "m": 1024**2, "mb": 1024**2, "g": 1024**3, "gb": 1024**3, "t": 1024**4, "tb": 1024**4}


def _parse_byte_size(value: str) -> int | None:
    """Parse a compose-style byte-size string (e.g. "16gb", "512m", "1024") into a byte count, or None if unparseable."""
    match = re.fullmatch(r"\s*([0-9]*\.?[0-9]+)\s*([a-zA-Z]*)\s*", value)
    if not match:
        return None
    number, unit = match.groups()
    multiplier = 1 if not unit else _BYTE_SIZE_UNITS.get(unit.lower())
    if multiplier is None:
        return None
    return int(float(number) * multiplier)


def _container_user_matches(config: dict[str, Any], configured_user: str | None) -> bool:
    """Return True if *configured_user* matches a container's live Config.User.

    No user configured always matches - user wasn't part of what was requested, so there's nothing
    to compare (mirrors _container_command_matches for an unconfigured command).
    """
    if not configured_user:
        return True
    return config.get("User") == configured_user


def _container_shm_size_matches(host_config: dict[str, Any], configured_shm_size: str | None) -> bool:
    """Return True if *configured_shm_size* matches the live container's HostConfig.ShmSize.

    *configured_shm_size* is a compose byte-size string, e.g. "16gb". No shm_size configured always
    matches - Docker reports its own default ShmSize on every
    container regardless, so there's nothing meaningful to compare it against.
    """
    if not configured_shm_size:
        return True
    expected_bytes = _parse_byte_size(configured_shm_size)
    return expected_bytes is not None and host_config.get("ShmSize") == expected_bytes


def _container_restart_policy_matches(host_config: dict[str, Any], configured_restart: str | None) -> bool:
    """Return True if *configured_restart* matches the live container's HostConfig.RestartPolicy.Name.

    *configured_restart* is a compose restart string, e.g. "unless-stopped" or "on-failure:5". No
    restart policy configured always matches - nothing was requested to compare against.
    """
    if not configured_restart:
        return True
    name, _sep, _max_retries = configured_restart.partition(":")
    return (host_config.get("RestartPolicy") or {}).get("Name") == name


def _container_attached_to_network(networks: dict[str, Any] | None, subnet: str) -> bool:
    """Return True if a container's live NetworkSettings.Networks includes *subnet*.

    In subnet mode the app addresses services by container name over this specific docker network
    (see get_container_host), so attachment to it is what "reachable" actually means - the analogue
    of the published-port check used outside subnet mode.
    """
    return bool(networks) and subnet in networks


def _container_gpu_matches(host_config: dict[str, Any], hardware: Sequence[HardwarePartInfo] | None) -> bool:
    """Return True if a container's live HostConfig GPU reservations match *hardware*.

    Mirrors what generate_docker_compose_content writes: Nvidia GPUs become a HostConfig.DeviceRequests
    entry with driver "nvidia" and an explicit DeviceIDs list, compared here as an exact set; Intel
    GPUs become a /dev/dri HostConfig.Devices mapping, compared here by presence only (the compose
    side doesn't encode which Intel card, just that one is wired in). A mismatch means the live
    container could be holding VRAM on a different GPU than *hardware* now requests (or none at all),
    which is unsafe to adopt silently.

    A count-based Nvidia request (e.g. "--gpus all", recorded as DeviceIDs: null with a non-zero
    Count) can reserve GPUs without naming any device ID, so it can't be verified against
    *expected_nvidia_ids* at all — it's rejected outright rather than being read as "no GPU reserved".
    """
    nvidia_requests = [request for request in (host_config.get("DeviceRequests") or []) if request.get("Driver") == "nvidia"]
    if any(not request.get("DeviceIDs") and request.get("Count", 0) != 0 for request in nvidia_requests):
        return False

    expected_nvidia_ids = sorted(str(gpu.id) for gpu in (hardware or []) if isinstance(gpu, NvidiaGpuInfo))
    actual_nvidia_ids = sorted(device_id for request in nvidia_requests for device_id in request.get("DeviceIDs") or [])
    if expected_nvidia_ids != actual_nvidia_ids:
        return False

    expects_intel = any(isinstance(gpu, IntelGpuInfo) for gpu in (hardware or []))
    has_intel_device = any(device.get("PathOnHost") == "/dev/dri" for device in host_config.get("Devices") or [])
    return expects_intel == has_intel_device


def _matches_for_adoption(inspect_data: dict[str, Any], options: DockerOptions, expected_image_id: str) -> int | None:  # noqa: C901
    """Return the port to reuse if *inspect_data* is a confident match for *options*, else None.

    A confident match requires: the live container's resolved image ID (top-level Image, a sha256
    digest fixed at container-creation time) equals *expected_image_id* — the current image ID that
    options.image resolves to locally. Comparing Config.Image (the name:tag string, e.g. "x:latest")
    instead would be too loose: a mutable tag can be repointed at a newer pull later, so an orphaned
    container created from a stale pull of "x:latest" would still show Config.Image == "x:latest"
    while actually running different software. The container is running; if a healthcheck is
    configured, live health is "healthy" (not just running); every configured env var present with
    an identical value; every configured bind-mount volume present; the configured command and
    entrypoint (if any) match the live command/entrypoint — this matters most for vLLM/SGLang, which
    encode the model path, revision, and context length in `command` rather than image or env vars;
    the configured user, shm_size, and restart policy (if any) match their live Config/HostConfig
    counterparts — otherwise a container built under a stale config could get adopted while still
    running as the wrong user or with the wrong shared-memory size; the live GPU device reservations
    match options.hardware (see _container_gpu_matches) — otherwise a container already holding VRAM
    on the wrong GPU could get silently adopted; and either the container is attached to options.subnet
    (subnet mode) or options.image_port is published to the host (otherwise). In subnet mode the port
    isn't used, so -1 is returned as a placeholder instead of None (which signals "no match").

    options.ulimits is intentionally not compared: generate_docker_compose_content never actually
    writes it into the compose file, so there is nothing on the live container to compare it against.
    """
    config = inspect_data.get("Config", {}) or {}
    state = inspect_data.get("State", {}) or {}
    network_settings = inspect_data.get("NetworkSettings", {}) or {}
    host_config = inspect_data.get("HostConfig", {}) or {}

    if inspect_data.get("Image") != expected_image_id:
        return None
    if state.get("Status") != "running":
        return None
    if options.healthcheck and (state.get("Health", {}) or {}).get("Status") != "healthy":
        return None
    if not _container_env_matches(config.get("Env") or [], options.env_vars):
        return None
    if not _container_volumes_match(inspect_data.get("Mounts") or [], options.volumes or []):
        return None
    if not _container_command_matches(config.get("Cmd"), options.command):
        return None
    if not _container_command_matches(config.get("Entrypoint"), options.entrypoint):
        return None
    if not _container_user_matches(config, options.user):
        return None
    if not _container_shm_size_matches(host_config, options.shm_size):
        return None
    if not _container_restart_policy_matches(host_config, options.restart):
        return None
    if not _container_gpu_matches(host_config, options.hardware):
        return None

    if options.subnet:
        if not _container_attached_to_network(network_settings.get("Networks"), options.subnet):
            return None
        return -1
    return _extract_published_host_port(network_settings.get("Ports"), options.image_port)


class DockerService:
    auths: dict[str, str]
    build_timeout = 30 * 60  # 30 minutes

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
        external_only: bool = False,
    ):
        self.config = config
        self.port_service = port_service
        self.external_only = external_only
        self.docker_compose_cmd = docker_compose_cmd
        self.has_gpu_support = has_gpu_support
        self.is_rootless = is_rootless
        self.os = os
        self.architecture = architecture
        self.host_platform = host_platform
        # Reading ~/.docker/config.json is itself a Docker interaction; external-only mode
        # promises none at construction.
        self.auths = {} if external_only else get_docker_auths()

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
        output = await Utils.run_command(cmd_parts)
        if output.exit_code != 0:
            if "not found" in output.stderr:
                raise DockerImageDoesNotExistError(image)
            if "authorization failed" in output.stderr or "failed to authorize" in output.stderr:
                raise DockerImageAuthorizationError(image)
            raise RuntimeError("Invalid exit code for command", (output.exit_code, cmd_parts, output.stdout, output.stderr))
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
                image = re.sub(r"@sha256:[^\s]*", "", image)
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

    async def get_local_docker_image_size(self, image: str) -> int | None:
        """Get size in bytes of a locally available image."""
        try:
            result = await Utils.run_command_for_success(["docker", "image", "inspect", image, "--format", "{{.Size}}"])
            return int(result.stdout.strip())
        except Exception:
            return None

    async def get_image_platforms(self, image: str) -> list[str]:
        """Get platform of a Docker image in format 'os/architecture'.

        Args:
            image: Docker image name (e.g., 'ubuntu:latest', 'vllm/vllm-openai:latest')

        Returns:
            Platform string like 'linux/arm64' or 'linux/amd64', or None if image not found.
        """
        try:
            # Inspect the image to get its metadata
            result = await Utils.run_command_for_success(["docker", "image", "inspect", image])

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
        result = await Utils.run_command(cmd_parts)
        if result.exit_code != 0:
            raise DockerComposeStartError(result.stdout, result.stderr)
        return CommandResult2(stdout=result.stdout, stderr=result.stderr)

    async def stop_docker(self, options: DockerOptions) -> None:
        """Stop the service's compose stack, or stop/remove the container directly if it has no compose file.

        Falls back to stopping/removing the container directly (see DFINFRA-281 for why a service may have no
        compose file, e.g. an adopted container). Logs a warning and does nothing if neither a compose file nor
        a container name is available.
        """
        docker_compose_file_path = self.get_docker_compose_file_path(options.name)
        if docker_compose_file_path.exists():
            await self.stop_docker_compose(docker_compose_file_path)
        elif options.container_name:
            await self._stop_and_remove_container(options.container_name)
        else:
            logger.warning("Cannot stop %r: no compose file and no container name to fall back to", options.name)

    async def stop_docker_compose(self, docker_compose_file_path: Path) -> None:
        """Stop given docker compose."""
        docker_compose_cmd = self.docker_compose_cmd
        cmd_parts = [*docker_compose_cmd.split(), "-f", str(docker_compose_file_path), "down", "--remove-orphans"]
        await Utils.run_command_for_success(cmd_parts)

    async def restart_docker_compose(self, options: DockerOptions) -> None:
        """Restart the compose-managed service for options, or the container directly if no compose file exists for it.

        Raises HTTPException(400) if neither a compose file nor a container name is available to restart —
        unlike stop_docker/uninstall_docker, there's no idempotent "nothing to do" reading for a restart.
        """
        docker_compose_file_path = self.get_docker_compose_file_path(options.name)
        if docker_compose_file_path.exists():
            docker_compose_cmd = self.docker_compose_cmd
            cmd_parts = [*docker_compose_cmd.split(), "-f", str(docker_compose_file_path), "restart"]
            await Utils.run_command_for_success(cmd_parts)
        elif options.container_name:
            await self._restart_container(options.container_name)
        else:
            msg = f"Cannot restart {options.name}: no compose file and no container name"
            raise HTTPException(400, msg)

    async def get_docker_compose_logs(self, docker_compose_file_path: Path) -> str:
        """Get docker compose logs."""
        docker_compose_cmd = self.docker_compose_cmd
        cmd_parts = [*docker_compose_cmd.split(), "-f", str(docker_compose_file_path), "logs"]
        result = await Utils.run_command_for_success(cmd_parts)
        return result.stdout

    async def run_command_docker_compose(self, filepath: Path, service_name: str, command: str) -> str:
        """Run command in docker compose service."""
        docker_compose_cmd = self.docker_compose_cmd
        cmd_parts = [*docker_compose_cmd.split(), "-f", str(filepath), "exec", service_name, *command.split(" ")]
        result = await Utils.run_command_for_success(cmd_parts)
        return result.stdout

    async def is_docker_compose_running(self, docker_compose_file_path: Path, service_name: str) -> bool:
        """Check whether the service from given docker compose is running."""
        docker_compose_cmd = self.docker_compose_cmd
        if not docker_compose_file_path.exists():
            return False
        try:
            cmd_parts = [*docker_compose_cmd.split(), "-f", str(docker_compose_file_path), "ps", "--services", "--filter", "status=running"]
            result = await Utils.run_command(cmd_parts)
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

    @property
    def _docker_bin(self) -> str:
        """Return the docker binary derived from docker_compose_cmd.

        "docker compose" → "docker"; "docker-compose" → "docker"
        (create_docker_service guarantees the docker binary is in PATH when docker-compose is used).
        """
        if self.external_only:
            # Without this the empty compose command would surface as an IndexError from
            # somewhere unrelated, which is a much worse way to learn the mode is wrong.
            raise DockerDisabledError
        first = self.docker_compose_cmd.split()[0]
        return "docker" if first == "docker-compose" else first

    async def build_image(self, build_context_dir: Path, tag: str, stream: Stream[StreamChunk]) -> None:
        """Build a Docker image from a Dockerfile in build_context_dir, streaming build log lines as progress."""
        proc = await asyncio.create_subprocess_exec(
            self._docker_bin,
            "build",
            "-t",
            tag,
            str(build_context_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        if proc.stdout is None:
            raise RuntimeError("docker build process has no stdout")
        try:
            async for raw_line in proc.stdout:
                line = raw_line.decode(errors="replace").rstrip()
                stream.emit(StreamChunkProgress(type="progress", stage="install", value=0, data={"log": line}))
        except BaseException:
            with suppress(ProcessLookupError):
                proc.kill()
            await proc.wait()
            raise
        try:
            await asyncio.wait_for(proc.wait(), timeout=self.build_timeout)
        except TimeoutError:
            with suppress(ProcessLookupError):
                proc.kill()
            await proc.wait()
            msg = f"docker build timed out after {self.build_timeout}s"
            raise RuntimeError(msg) from None
        if proc.returncode != 0:
            msg = f"docker build failed with exit code {proc.returncode}"
            raise RuntimeError(msg)

    async def is_docker_compose_healthy(self, docker_compose_file_path: Path, service_name: str) -> bool:
        """Check whether the service from given docker compose is healthy."""
        docker_compose_cmd = self.docker_compose_cmd
        if not docker_compose_file_path.exists():
            msg = f"{docker_compose_file_path} not found"
            logger.debug(msg)
            return False

        try:
            cmd_parts = [*docker_compose_cmd.split(), "-f", str(docker_compose_file_path), "ps", service_name, "--format", "json"]
            result = await Utils.run_command(cmd_parts)

            if result.exit_code != 0:
                msg = f"{docker_compose_file_path} {cmd_parts} exit code is not 0. Exit code is {result.exit_code}"
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

    async def _inspect_container(self, container_name: str) -> dict[str, Any]:
        """Return the raw docker inspect data for container_name.

        Raises DockerError (404 if the container doesn't exist) or TypeError on a malformed
        (non-dict) response. Callers decide how to interpret failures.
        """
        async with Docker() as docker:
            data = await docker.containers.container(container_name).show()
        if not isinstance(data, dict):  # pyright: ignore[reportUnnecessaryIsInstance]  # aiodocker's type hint isn't runtime-enforced
            msg = f"Unexpected inspect response type {type(data)} for container {container_name!r}"
            raise TypeError(msg)
        return data

    async def _stop_and_remove_container(self, container_name: str) -> None:
        """Stop and remove container_name directly via the Docker API, tolerating it already being gone.

        Fallback for stop_docker/uninstall_docker when no compose file exists for the service (e.g. an
        adopted container, see DFINFRA-281).
        """
        logger.info("No compose file for this service; stopping and removing container %r directly", container_name)
        async with Docker() as docker:
            container = docker.containers.container(container_name)
            try:
                await container.stop()
            except DockerError as e:
                if e.status != 404:
                    raise
                logger.info("Container %r already gone; nothing to stop/remove", container_name)
                return
            try:
                await container.delete(force=True)
            except DockerError as e:
                if e.status != 404:
                    raise
                logger.info("Container %r already removed", container_name)

    async def _restart_container(self, container_name: str) -> None:
        """Restart container_name directly via the Docker API.

        Fallback for restart_docker_compose when no compose file exists for the service (e.g. an
        adopted container, see DFINFRA-281). Unlike _stop_and_remove_container, does not tolerate the
        container being missing — raises HTTPException(400) on a 404.
        """
        logger.info("No compose file for this service; restarting container %r directly", container_name)
        try:
            async with Docker() as docker:
                await docker.containers.container(container_name).restart()
        except DockerError as e:
            if e.status == 404:
                msg = f"Cannot restart {container_name!r}: container not found"
                raise HTTPException(400, msg) from e
            raise

    async def get_container_status(self, container_name: str) -> ContainerStatus:
        """Return a container's real state (process status, healthcheck status, restart count) for reconciliation checks after install."""
        try:
            info = await self._inspect_container(container_name)
        except DockerError as e:
            if e.status == 404:
                return ContainerStatus(exists=False, state="", health="", restart_count=0)
            raise

        state = info.get("State", {})
        health = state.get("Health", {})
        return ContainerStatus(
            exists=True,
            state=state.get("Status", ""),
            health=health.get("Status", ""),
            restart_count=info.get("RestartCount", 0),
        )

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
                    raise AppError("GPU is not available on this machine")
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
                gids: set[str] = set()
                for dev in Path("/dev/dri").iterdir():
                    with suppress(OSError):
                        gids.add(str(Path.stat(dev).st_gid))
                gids.discard("0")
                if gids:
                    service["group_add"] = sorted(gids)
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

    async def _inspect_container_for_adoption(self, container_name: str) -> dict[str, Any] | None:
        """Return the raw docker inspect data for container_name, or None if it can't be inspected for any reason.

        Any failure means adoption can't proceed, so this never raises — the caller falls back to the
        normal conflict behavior. A missing container (404) is the expected, common case and stays
        quiet; anything else (daemon unreachable, permission error, an unexpected DockerError status,
        or a malformed non-dict response) is unusual enough to warrant an operator's attention, so
        it's logged at warning instead.
        """
        try:
            return await self._inspect_container(container_name)
        except DockerError as e:
            if e.status == 404:
                return None
            logger.warning(
                "Could not inspect container %r for adoption (docker API error, status=%s)", container_name, e.status, exc_info=True
            )
            return None
        except Exception:
            logger.warning("Could not inspect container %r for adoption", container_name, exc_info=True)
            return None

    async def _resolve_image_id_for_adoption(self, image: str) -> str | None:
        """Return the local image ID (sha256 digest) that *image* currently resolves to, or None if unresolvable.

        A container's own Config.Image is just the name:tag string it was created with, which stays
        stable even if a mutable tag like ":latest" is later repointed at a newer pull — so it can't be
        used to prove an orphaned container is still running the same software. Resolving *image*'s
        current ID here, to compare against the live container's fixed image ID, is what actually
        detects that drift. Any failure (image not pulled locally, daemon unreachable, malformed
        response) means it can't be proven safe to adopt, so the caller treats None as "no match".
        """
        try:
            async with Docker() as docker:
                data = await docker.images.inspect(image)
        except DockerError as e:
            if e.status != 404:
                logger.warning("Could not resolve image %r for adoption (docker API error, status=%s)", image, e.status, exc_info=True)
            return None
        except Exception:
            logger.warning("Could not resolve image %r for adoption", image, exc_info=True)
            return None
        if not isinstance(data, dict):  # pyright: ignore[reportUnnecessaryIsInstance]  # aiodocker's type hint isn't runtime-enforced
            logger.warning("Could not resolve image %r for adoption (unexpected response type %s)", image, type(data))
            return None
        image_id = data.get("Id")
        return image_id if isinstance(image_id, str) else None

    async def _try_adopt_orphaned_container(self, options: DockerOptions) -> int | None:
        """Return the port to adopt at if the pre-existing container for *options* is a confident match, else None.

        Without an explicit container_name there's no reliable way to know what name Docker Compose
        actually assigned the conflicting container — guessing options.name (the service name) risks
        inspecting an unrelated container that happens to share it, so adoption is skipped entirely.
        """
        if options.container_name is None:
            return None
        inspect_data = await self._inspect_container_for_adoption(options.container_name)
        if inspect_data is None:
            return None
        expected_image_id = await self._resolve_image_id_for_adoption(options.image)
        if expected_image_id is None:
            return None
        port = _matches_for_adoption(inspect_data, options, expected_image_id)
        if port is not None:
            logger.info("Adopting pre-existing container %r for service %r (image=%r)", options.container_name, options.name, options.image)
        else:
            logger.info(
                "Container %r exists but isn't a confident match for service %r — not adopting it",
                options.container_name,
                options.name,
            )
        return port

    async def _assert_adopted_container_still_healthy(self, container_name: str, options: DockerOptions) -> None:
        """Raise AppError unless a fresh inspect of *container_name* still looks adoptable.

        `_try_adopt_orphaned_container`'s match decision is based on a single inspect snapshot taken
        before this call; re-inspecting here, right before adoption is committed to and the caller
        registers the endpoint as live, narrows (but can't eliminate) the window in which the container
        could have crashed or flipped unhealthy — catching that here means it surfaces as a normal
        failed-install error with logs, like any other failed start, instead of a live endpoint quietly
        pointing at a dead container.
        """
        try:
            info = await self._inspect_container(container_name)
        except Exception as exc:
            msg = f"Adopted container {container_name!r} for {options.name} could not be re-confirmed healthy: {exc}"
            raise AppError(msg) from exc
        state = info.get("State", {}) or {}
        is_running = state.get("Status") == "running"
        is_healthy = not options.healthcheck or (state.get("Health", {}) or {}).get("Status") == "healthy"
        if is_running and is_healthy:
            return
        logs = ""
        with suppress(Exception):
            result = await Utils.run_command(["docker", "logs", "--tail", "200", container_name])
            logs = result.stdout
        short = _extract_error_excerpt(logs) if logs else ""
        msg = (
            f"Adopted container {container_name!r} for {options.name} is no longer healthy "
            f"(status={state.get('Status')!r}): {short or 'unknown error, see server logs for details'}"
        )
        raise AppError(msg)

    async def _adopt_on_name_conflict_or_raise(
        self, compose_path: Path, options: DockerOptions, wrote_compose_file: bool, reserved_port: int | None
    ) -> int:
        """Adopt the conflicting container if it's a confident match, else raise the existing 409.

        On adoption, deletes *compose_path* if this install attempt wrote it (see _ensure_compose_running).
        Adoption is already a success at that point (the container itself is fine), so a failure to
        remove the stale file is logged rather than allowed to turn a successful install into a crash.

        *reserved_port* is the port this install attempt had reserved (or was about to reuse) before the
        conflict; when adoption ends up using a different port, the reserved one is released back to the
        pool instead of leaking for the lifetime of the process.
        """
        adopted_port = await self._try_adopt_orphaned_container(options)
        if adopted_port is None:
            name = options.container_name or options.name
            raise HTTPException(409, f"A container named '{name}' already exists — remove it or choose a different name.") from None
        # adopted_port is only ever non-None when options.container_name was set (see _try_adopt_orphaned_container).
        await self._assert_adopted_container_still_healthy(cast("str", options.container_name), options)
        if wrote_compose_file:
            try:
                compose_path.unlink(missing_ok=True)
            except OSError:
                logger.warning(
                    "Adopted container for %r but failed to remove stale compose file %s", options.name, compose_path, exc_info=True
                )
        if reserved_port is not None and reserved_port != adopted_port:
            self.port_service.release_port(reserved_port)
        return adopted_port

    async def _ensure_compose_running(
        self,
        compose_path: Path,
        options: DockerOptions,
        is_running: bool,
        has_difference: bool,
        port: int | None,
    ) -> tuple[CommandResult2 | None, int | None, bool]:
        """Start, stop, or recreate the compose stack as needed. Returns (start_output, port, adopted).

        When adopted is True, start_output is always None. Any compose file written during this call —
        newly created, or an existing one overwritten before the restart — is deleted on adoption; a
        compose file left untouched by this call (not written at all) stays that way — see
        _adopt_on_name_conflict_or_raise. If adoption returns a different port than the one this call
        was about to start with, that unused port is released back to the pool.
        """
        uses_gpu = bool(options.hardware)
        start_output: CommandResult2 | None = None
        wrote_compose_file = False
        try:
            if not is_running and not has_difference:
                if port is not None and self.port_service.is_port_available(port):
                    start_output = await self.start_docker_compose(compose_path)
                else:
                    port = await self.create_compose_file(compose_path, options)
                    wrote_compose_file = True
                    start_output = await self.start_docker_compose(compose_path)
            elif not is_running and has_difference:
                port = await self.create_compose_file(compose_path, options)
                wrote_compose_file = True
                start_output = await self.start_docker_compose(compose_path)
            elif is_running and has_difference:
                logger.debug("%s config changed, restarting", options.service_name)
                await self.stop_docker_compose(compose_path)
                port = await self.create_compose_file(compose_path, options)
                wrote_compose_file = True
                start_output = await self.start_docker_compose(compose_path)
        except DockerComposeStartError as exc:
            exc_output = "\n".join(filter(None, [exc.stdout, exc.stderr]))
            if _is_container_name_conflict(exc_output):
                adopted_port = await self._adopt_on_name_conflict_or_raise(compose_path, options, wrote_compose_file, port)
                return None, adopted_port, True
            logs = ""
            with suppress(Exception):
                logs = await self.get_docker_compose_logs(compose_path)
            combined = "\n".join(filter(None, [exc_output, logs]))
            if uses_gpu:
                gpu_msg = _diagnose_gpu_error(combined)
                if gpu_msg:
                    raise AppError(gpu_msg) from None
            if logs:
                logger.exception("Failed to start %s — full compose logs:\n%s", options.name, _bounded_for_logging(combined))
            short = "\n".join(filter(None, [exc_output, _extract_error_excerpt(logs)]))
            msg = f"Failed to start {options.name}: {short or 'unknown error, see server logs for details'}"
            raise AppError(msg) from None
        return start_output, port, False

    async def _assert_compose_healthy(self, compose_path: Path, options: DockerOptions, start_output: CommandResult2 | None) -> None:
        """Raise AppError if the container is not healthy after starting."""
        is_healthy = await self.is_docker_compose_healthy(compose_path, options.service_name)
        if is_healthy:
            return
        logs = ""
        with suppress(Exception):
            logs = await self.get_docker_compose_logs(compose_path)
        combined = "\n".join(
            filter(
                None,
                [
                    start_output.stdout if start_output else "",
                    start_output.stderr if start_output else "",
                    logs,
                ],
            )
        )
        if bool(options.hardware):
            gpu_msg = _diagnose_gpu_error(combined)
            if gpu_msg:
                raise AppError(gpu_msg)
        if logs:
            logger.error("Container %s failed to become healthy — full compose logs:\n%s", options.name, _bounded_for_logging(combined))
        short = "\n".join(
            filter(
                None,
                [
                    start_output.stdout if start_output else "",
                    start_output.stderr if start_output else "",
                    _extract_error_excerpt(logs),
                ],
            )
        )
        msg = f"Container {options.name} failed to become healthy"
        if short:
            msg = f"{msg}:\n{short}"
        raise AppError(msg)

    async def install_and_run_docker(self, options: DockerOptions) -> tuple[int, bool, bool]:
        """Run docker compose and return (port, whether the container was (re)started, whether it was adopted).

        An adopted container (see `_try_adopt_orphaned_container`) pre-dates this call — it wasn't created by
        it — which callers doing rollback-on-later-failure need to know so they don't tear down a container
        this install attempt never created (see DFINFRA-297 follow-up).
        """
        compose_path = self.get_docker_compose_file_path(options.name)
        is_running = await self.is_docker_compose_running(compose_path, options.service_name)
        has_difference, port = await self.has_docker_compose_difference(compose_path, options)

        logger.debug("docker_compose_start: %s is_running=%s has_difference=%s", options.service_name, is_running, has_difference)

        start_output, port, adopted = await self._ensure_compose_running(compose_path, options, is_running, has_difference, port)
        if not adopted:
            await self._assert_compose_healthy(compose_path, options, start_output)

        if not port and options.subnet:
            # in subnet mode the port is not used so it could be anything, for example -1
            port = -1
        if port is None:
            raise AppError("Engine not available: cannot allocate service port")
        return port, start_output is not None, adopted

    async def uninstall_docker(self, options: DockerOptions) -> None:
        """Stop the service and remove its compose file, or remove the container directly if it has no compose file.

        Falls back to stopping/removing the container directly (and removes nothing on disk) when no compose file
        exists for the service — e.g. an adopted container, see DFINFRA-281.
        """
        docker_compose_cmd = self.docker_compose_cmd
        docker_compose_file = self.get_docker_compose_file_path(options.name)
        if docker_compose_file.is_file():
            await Utils.run_command([*docker_compose_cmd.split(), "-f", str(docker_compose_file), "down"])
        elif options.container_name:
            # Uninstall is best-effort by design (the compose branch above ignores `down`'s exit code too): a
            # Docker daemon hiccup here must not abort uninstall and leave the service stuck as still-installed.
            try:
                await self._stop_and_remove_container(options.container_name)
            except DockerError:
                logger.warning(
                    "Failed to remove container %r while uninstalling %r; continuing uninstall anyway",
                    options.container_name,
                    options.name,
                    exc_info=True,
                )
        else:
            logger.warning("Cannot uninstall %r: no compose file and no container name to fall back to", options.name)
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
    """Create docker service (external-only mode returns an unprobed service, no Docker calls)."""
    if config.external_only:

        def get_host_platform_external_only() -> str:
            arch = platform.machine().lower()
            return normalize_docker_platform(f"linux/{arch}")

        return DockerService(
            config,
            port_service,
            "",
            False,
            get_os(),
            get_cpu_architecture(),
            False,
            get_host_platform_external_only(),
            external_only=True,
        )

    async def get_docker_compose_cmd() -> str:
        """Return docker compose command."""
        if shutil.which("docker"):
            result = await Utils.run_command(["docker", "compose", "version"])
            if result.exit_code == 0:
                return "docker compose"
            if shutil.which("docker-compose"):
                return "docker-compose"

        raise DockerNotInstalledError("Docker is not installed.")

    async def has_gpu_support() -> bool:
        """Return whether there is GPU support."""
        result = await Utils.run_command(["docker", "run", "--gpus", "all", "--rm", "busybox", "echo"])
        return result.exit_code == 0

    async def is_rootless() -> bool:
        """Return whether docker is running in rootless mode."""
        result = await Utils.run_command(["docker", "info"])
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
