# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Mcp service."""

import asyncio
import json
import logging
import shlex
import shutil
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import HTTPException
from pydantic import BaseModel, Field, field_validator

from server.applicationcontext import get_base_url
from server.docker import DockerImage, DockerOptions
from server.endpointregistry import ProxyOptions, RegistrationId
from server.models.api import ModelProps
from server.models.models import (
    CustomModelField,
    CustomModelId,
    CustomModelSpecification,
    InstallModelIn,
    InstallModelOut,
    ListModelsFilters,
    ListModelsOut,
    ModelField,
    ModelInfo,
    ModelSpecification,
    RetrieveModelOut,
    UninstallModelIn,
)
from server.models.services import (
    InstallServiceIn,
    InstallServiceProgress,
    ServiceOptions,
    ServiceSize,
    ServiceSpecification,
    UninstallServiceIn,
)
from server.services.base2_service import Base2Service, CustomModel, Instance, InstanceConfig, ModelConfig
from server.utils.core import (
    PromiseWithProgress,
    Stream,
    StreamChunk,
    StreamChunkProgress,
    normalize_name,
    try_parse_pydantic,
)
from server.utils.size_fetcher import fmt_size


class McpUserVariant(StrEnum):
    node_headless = "node-headless"
    node_headed = "node-headed"
    python_headless = "python-headless"
    python_headed = "python-headed"


McpModelKind = Literal["custom", "user", "proxy"]

PythonVersion = Literal["3.10", "3.11", "3.12", "3.13", "3.14", "latest"]
NodeVersion = Literal["20", "22", "24", "latest"]

_DEFAULT_PYTHON_VERSION: PythonVersion = "3.13"
_DEFAULT_NODE_VERSION: NodeVersion = "22"


def _version_to_base_image(variant: McpUserVariant, python_version: PythonVersion | None, node_version: NodeVersion | None) -> str:
    if variant in (McpUserVariant.python_headless, McpUserVariant.python_headed):
        ver = python_version or _DEFAULT_PYTHON_VERSION
        return "python:slim" if ver == "latest" else f"python:{ver}-slim"
    ver = node_version or _DEFAULT_NODE_VERSION
    return "node:slim" if ver == "latest" else f"node:{ver}-slim"


# Per-variant setup layers: RUN + EXPOSE + ENTRYPOINT (no FROM, no CMD).
_DOCKERFILE_SETUP: dict[McpUserVariant, str] = {
    McpUserVariant.node_headless: """\
RUN npm install -g supergateway
EXPOSE 8000
ENTRYPOINT ["supergateway", "--port", "8000", "--outputTransport", "streamableHttp", "--stateful", "--stdio"]
""",
    McpUserVariant.node_headed: """\
RUN apt-get update && apt-get install -y chromium --no-install-recommends && rm -rf /var/lib/apt/lists/*
RUN npm install -g supergateway
EXPOSE 8000
ENTRYPOINT ["supergateway", "--port", "8000", "--outputTransport", "streamableHttp", "--stateful", "--stdio"]
""",
    McpUserVariant.python_headless: """\
RUN pip install --no-cache-dir uv mcp-proxy
EXPOSE 8000
ENTRYPOINT ["mcp-proxy", "--host", "0.0.0.0", "--port", "8000", "--"]
""",
    McpUserVariant.python_headed: """\
RUN apt-get update && apt-get install -y chromium --no-install-recommends && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv mcp-proxy
EXPOSE 8000
ENTRYPOINT ["mcp-proxy", "--host", "0.0.0.0", "--port", "8000", "--"]
""",
}


def _build_dockerfile(variant: McpUserVariant, cmd_json: str, base_image: str) -> str:
    return f"FROM {base_image}\n{_DOCKERFILE_SETUP[variant]}CMD {cmd_json}\n"


type SrvPcpModelX = Callable[["McpService", str | None], SrvMcpModel]


@dataclass
class SrvMcpModel:
    model_props: ModelProps
    model_spec: ModelSpecification
    model_type: str
    default_prefix: str
    size: str
    options: DockerOptions | None
    required_envs: list[str] | None = None
    required_headers: list[str] | None = None
    envs: dict[str, str] | None = None
    headers: dict[str, str] | None = None
    custom: CustomModelId | None = None
    kind: McpModelKind = field(default="custom")
    variant: str | None = field(default=None)
    command: str | None = field(default=None)
    base_image: str | None = field(default=None)
    python_version: str | None = field(default=None)
    node_version: str | None = field(default=None)
    proxy_url: str | None = field(default=None)
    proxy_transport: str = field(default="streamable_http")


class SrvMcpCustomModel(BaseModel):
    id: str
    private: bool = True
    default_prefix: Annotated[str, Field(pattern=r"^[a-zA-Z0-9_-]+$")]
    size: str
    image: str
    image_port: int
    command: str | None = None
    hardware: str | bool | None = None
    volumes: list[str] | None = None
    envs: dict[str, str] | None = None
    headers: dict[str, str] | None = None
    healthcheck_cmd: str | None = None
    healthcheck_start_period: Annotated[str, Field(pattern=r"^\d+[smh]$")] | None = None
    required_envs: dict[str, str] | None = None
    required_headers: dict[str, str] | None = None
    proxy_transport: Literal["streamable_http", "sse"] = "streamable_http"


class SrvMcpUserModel(BaseModel):
    kind: Literal["user"] = "user"
    id: str
    name: str
    variant: McpUserVariant
    command: str
    python_version: PythonVersion | None = None
    node_version: NodeVersion | None = None
    base_image: str | None = None
    envs: dict[str, str] | None = None
    required_envs: dict[str, str] | None = None
    private: bool = True
    default_prefix: Annotated[str, Field(pattern=r"^[a-zA-Z0-9_-]+$")] | None = None
    size: str = ""


class SrvMcpProxyModel(BaseModel):
    """Remote MCP server registered as a persistent proxy endpoint."""

    kind: Literal["proxy"] = "proxy"
    id: str
    name: str
    server_url: str
    transport: Literal["streamable_http", "sse"] = "streamable_http"
    private: bool = True
    default_prefix: Annotated[str, Field(pattern=r"^[a-zA-Z0-9_-]+$")] | None = None
    headers: dict[str, str] | None = None
    required_headers: dict[str, str] | None = None


@dataclass
class McpConst:
    models: dict[str, SrvPcpModelX]


class McpModelOptions(BaseModel):
    prefix: Annotated[str, Field(pattern=r"^[a-zA-Z0-9_-]+$")]
    envs: dict[str, str] = {}
    headers: dict[str, str] = {}

    @field_validator("headers", "envs", mode="before")
    @classmethod
    def empty_string_to_dict(cls, v: Literal[""] | dict[str, str]) -> dict[str, str]:
        """If the input is an empty string, return an empty dictionary."""
        if v == "":
            return {}
        return v


@dataclass
class ModelInstalledInfo:
    id: str
    options: InstallModelIn
    docker_options: DockerOptions | None
    container_host: str
    container_port: int
    docker_exposed_port: int
    registration_id: RegistrationId
    prefix: str
    base_url: str
    headers: dict[str, str]
    envs: dict[str, str]

    def get_info(self) -> ModelInfo:
        """Get info."""
        return ModelInfo(spec=self.options.spec, registration_id=self.registration_id)


@dataclass
class InstalledInfo:
    models: dict[str, ModelInstalledInfo]
    options: InstallServiceIn


@dataclass
class DownloadedInfo:
    image: str


logger = logging.getLogger("uvicorn.error")


class McpService(Base2Service[InstalledInfo, DownloadedInfo]):
    models: dict[str, dict[str, SrvMcpModel]]

    def _after_init(self) -> None:
        self.models = {}
        self.load_default_models("default")

    def load_default_models(self, instance: str) -> None:
        """Load default models to instance."""
        self.models[instance] = {}
        subnet = self.docker_service.get_docker_subnet()
        for model in _const.models.copy():
            self.models[instance][model] = _const.models[model](self, subnet)

    def get_type(self) -> str:
        """Return the service id."""
        return "mcp"

    def get_description(self) -> str:
        """Return the service description."""
        return "Your option to add MCP server."

    def get_size(self) -> ServiceSize:
        """Return the service size."""
        return ""

    def get_spec(self) -> ServiceSpecification:
        """Return the service specification."""
        return ServiceSpecification(fields=[])

    def get_default_model_spec(
        self,
        default_prefix: str,
        required_envs: list[str] | dict[str, str] | None = None,
        required_headers: list[str] | dict[str, str] | None = None,
    ) -> ModelSpecification:
        """Return the model specification."""
        return ModelSpecification(
            fields=[
                ModelField(
                    type="text",
                    name="prefix",
                    description="Endpoint prefix",
                    required=True,
                    placeholder="my-prefix",
                    default=default_prefix,
                ),
                ModelField(
                    type="map",
                    name="envs",
                    description=(
                        "\n".join(
                            ["Custom enviromental variables.", f"Required variables: {', '.join(required_envs)}" if required_envs else ""]
                        )
                    ),
                    required=False,
                    placeholder="envs",
                    default=(
                        json.dumps(dict.fromkeys(required_envs, "") if isinstance(required_envs, list) else required_envs)
                        if required_envs
                        else ""
                    ),
                ),
                ModelField(
                    type="map",
                    name="headers",
                    description=(
                        "\n".join(["Custom headers.", f"Required headers: {', '.join(required_headers)}" if required_headers else ""])
                    ),
                    required=False,
                    placeholder="headers",
                    default=(
                        json.dumps(dict.fromkeys(required_headers, "") if isinstance(required_headers, list) else required_headers)
                        if required_headers
                        else ""
                    ),
                ),
            ]
        )

    async def stop_instance(self, instance: str) -> None:
        """Stop all MCP service Docker containers."""
        installed = self.get_instance_info(instance).installed
        if not installed:
            return
        await self._stop_dockers_parallel([m.docker_options for m in installed.models.values() if m.docker_options is not None])

    def get_custom_model_spec(self) -> CustomModelSpecification | None:
        """Return the custom model specification or None if custom model is not supported."""
        return CustomModelSpecification(
            fields=[
                CustomModelField(type="text", name="id", description="Model ID", placeholder="my-custom-model"),
                CustomModelField(type="bool", name="private", description="Model is private", default="true"),
                CustomModelField(
                    type="text",
                    name="default_prefix",
                    description="Default model endpoint prefix [a-zA-Z0-9_-]",
                    placeholder="custom-model",
                ),
                CustomModelField(type="text", name="image", description="Docker image", placeholder="company/image"),
                CustomModelField(type="text", name="image_port", description="Docker image port", placeholder="8000"),
                CustomModelField(type="text", name="command", description="Docker command", placeholder="/bin/myapp", required=False),
                CustomModelField(
                    type="text",
                    name="healthcheck_cmd",
                    description="Healthcheck command",
                    placeholder="curl --fail 127.0.0.1:8000 | exit 1",
                    required=False,
                ),
                CustomModelField(
                    type="text",
                    name="healthcheck_start_period",
                    description="Healthcheck start period",
                    placeholder="10s",
                    required=False,
                ),
                CustomModelField(type="list", name="volumes", description="Bind mounts", placeholder="/work/storage", required=False),
                CustomModelField(type="map", name="envs", description="Docker environment variables", required=False),
                CustomModelField(type="map", name="headers", description="Headers for mcp connection", required=False),
                CustomModelField(
                    type="map", name="required_envs", description="Required envs in installation. Key values can be empty.", required=False
                ),
                CustomModelField(
                    type="map",
                    name="required_headers",
                    description="Required headers in installation. Key values can be empty.",
                    required=False,
                ),
                CustomModelField(type="text", name="size", description="Model size", placeholder="1 GB", required=False),
                CustomModelField(
                    type="oneof",
                    name="proxy_transport",
                    description="MCP transport protocol",
                    default="streamable_http",
                    required=False,
                    values=["streamable_http", "sse"],
                ),
            ]
        )

    async def _resolve_custom_model_size(self, spec: dict[str, Any], instance: str = "") -> str | None:  # noqa: ARG002
        try:
            size_bytes = await self.docker_service.get_docker_image_size(spec["image"])
            return fmt_size(size_bytes) if size_bytes else None
        except Exception:
            return None

    async def _persist_custom_model_size(self, instance: str, model: SrvMcpModel) -> None:
        if not model.custom:
            return
        config = self.get_instance_info(instance).config
        for custom_model in config.custom or []:
            if custom_model.id == model.custom:
                custom_model.data["size"] = model.size
                await self._save()
                break

    def get_installed_info(self, instance: str) -> bool | InstallServiceProgress | ServiceOptions:
        """Get service installed info."""
        installed = self.get_instance_info(instance).installed
        return self._get_service_installed_info(instance) if installed is None else installed.options.spec

    def _generate_instance_config(self, info: InstalledInfo | None, custom: list[CustomModel] | None) -> InstanceConfig:
        return InstanceConfig(
            options=info.options if info else None,
            models=[ModelConfig(model_id=x.id, options=x.options) for x in info.models.values()] if info else [],
            custom=custom,
        )

    def _load_download_info(self, data: dict[str, Any]) -> DownloadedInfo:
        return DownloadedInfo(**data)

    async def _install_instance(self, instance: str, options: InstallServiceIn) -> PromiseWithProgress[InstalledInfo, StreamChunk]:
        if not self.models.get(instance):
            self.load_default_models(instance)

        async def func(stream: Stream[StreamChunk]) -> InstalledInfo:  # noqa: ARG001
            self.service_downloaded = True
            return InstalledInfo(models={}, options=options)

        return PromiseWithProgress(func=func)

    async def _uninstall_instance(self, instance: str, options: UninstallServiceIn) -> None:
        installed = self.get_instance_info(instance).installed
        if installed:
            results = await asyncio.gather(
                *[
                    self._uninstall_model(instance, model.id, UninstallModelIn(purge=options.purge))
                    for model in installed.models.copy().values()
                    if not self.is_model_installed_in_other_instance(instance, model.id)
                ],
                return_exceptions=True,
            )
            for result in results:
                if isinstance(result, BaseException):
                    logger.exception("Error uninstalling model during service teardown", exc_info=result)

        self.instances_info[instance].installed = None

        if options.purge:
            if len(self.instances_info) < 2:
                self.service_downloaded = False
                await self._clear_working_dir()
                self.models_downloaded = {}

            if instance == "default":
                self.instances_info["default"] = Instance(None, None, {}, InstanceConfig())
            else:
                del self.instances_info[instance]

    def get_docker_compose_file_path(self, instance: str, model_id: str | None) -> Path:
        """Get docker compose file path."""
        info = self.get_instance_installed_info(instance)
        if not model_id:
            raise HTTPException(400, "Docker is not bound with this object")

        model_installed = info.models.get(model_id, None)
        if not model_installed:
            raise HTTPException(status_code=400, detail="Model not installed")

        if model_installed.docker_options is None:
            raise HTTPException(400, "Docker is not bound with this model")
        return self.docker_service.get_docker_compose_file_path(model_installed.docker_options.name)

    def _get_dockerfile_dir(self, instance: str, name: str) -> Path:
        return self.get_working_dir() / "models" / instance / name

    def _write_dockerfile(self, instance: str, name: str, variant: McpUserVariant, command: str, base_image: str) -> None:
        dockerfile_dir = self._get_dockerfile_dir(instance, name)
        dockerfile_dir.mkdir(parents=True, exist_ok=True)
        # supergateway (node) expects the full command as a single string passed to --stdio;
        # mcp-proxy (python) expects the command split into individual args after --.
        if variant in (McpUserVariant.node_headless, McpUserVariant.node_headed):
            cmd_json = json.dumps([command])
        else:
            try:
                cmd_json = json.dumps(shlex.split(command))
            except ValueError as e:
                raise HTTPException(400, f"Invalid command syntax: {e}") from e
        content = _build_dockerfile(variant, cmd_json, base_image)
        (dockerfile_dir / "Dockerfile").write_text(content, encoding="utf-8")

    def _delete_dockerfile_dir(self, instance: str, name: str) -> None:
        dockerfile_dir = self._get_dockerfile_dir(instance, name)
        if dockerfile_dir.exists():
            shutil.rmtree(dockerfile_dir)

    def _build_user_model(self, instance: str, parsed: SrvMcpUserModel, custom_id: CustomModelId | None = None) -> SrvMcpModel:
        effective_base_image = parsed.base_image or _version_to_base_image(parsed.variant, parsed.python_version, parsed.node_version)
        self._write_dockerfile(instance, parsed.id, parsed.variant, parsed.command, effective_base_image)
        name = normalize_name(f"{parsed.id}_{instance}")
        image_tag = f"deepfellow-mcp-{name}:latest"
        prefix = parsed.default_prefix or normalize_name(parsed.id)
        subnet = self.docker_service.get_docker_subnet()
        required_envs = list(parsed.required_envs.keys() if parsed.required_envs else [])
        return SrvMcpModel(
            model_props=ModelProps(private=parsed.private, type="mcp", endpoints=[f"/mcp/{prefix}/mcp"]),
            model_spec=self.get_default_model_spec(prefix, parsed.required_envs),
            model_type="mcp",
            default_prefix=prefix,
            size=parsed.size,
            options=DockerOptions(
                image_port=8000,
                name=name,
                container_name=self.docker_service.get_docker_container_name(name),
                image=image_tag,
                env_vars=parsed.envs,
                restart="unless-stopped",
                subnet=subnet,
            ),
            custom=custom_id,
            required_envs=required_envs,
            kind="user",
            variant=parsed.variant.value,
            command=parsed.command,
            base_image=parsed.base_image,
            python_version=parsed.python_version,
            node_version=parsed.node_version,
            envs=parsed.envs,
        )

    def _add_custom_model(self, instance: str, model: CustomModel) -> None:
        kind = model.data.get("kind")
        if kind == "user":
            self._add_user_model(instance, model)
        elif kind == "proxy":
            self._add_proxy_model(instance, model)
        else:
            self._add_image_model(instance, model)

    def _add_image_model(self, instance: str, model: CustomModel) -> None:
        parsed = try_parse_pydantic(SrvMcpCustomModel, model.data)

        if not self.models.get(instance):
            self.models[instance] = {}

        if parsed.id in self.models[instance]:
            raise HTTPException(400, f"Model with {parsed.id} id already exists.")
        name = normalize_name(f"{parsed.id}-{instance}")
        subnet = self.docker_service.get_docker_subnet()
        required_envs = list(parsed.required_envs.keys() if parsed.required_envs else [])
        required_headers = list(parsed.required_headers.keys() if parsed.required_headers else [])
        self.models[instance][parsed.id] = SrvMcpModel(
            model_props=ModelProps(private=parsed.private, type="mcp", endpoints=[f"/mcp/{parsed.default_prefix}/mcp"]),
            model_spec=self.get_default_model_spec(parsed.default_prefix, parsed.required_envs, parsed.required_headers),
            model_type="mcp",
            default_prefix=parsed.default_prefix,
            size=parsed.size,
            options=DockerOptions(
                image_port=parsed.image_port,
                name=name,
                container_name=self.docker_service.get_docker_container_name(name),
                image=parsed.image,
                command=parsed.command,
                hardware=self.get_specified_hardware_parts(parsed.hardware),
                env_vars=parsed.envs,
                restart="unless-stopped",
                volumes=[f"{self.get_working_dir()}/{name}/volume_{i}:{volume}" for i, volume in enumerate(parsed.volumes or [])],
                subnet=subnet,
                healthcheck={
                    "test": parsed.healthcheck_cmd,
                    "interval": "30s",
                    "timeout": "10s",
                    "retries": "3",
                    "start_period": parsed.healthcheck_start_period or "10s",
                }
                if parsed.healthcheck_cmd
                else None,
            ),
            custom=model.id,
            headers=parsed.headers,
            required_envs=required_envs,
            required_headers=required_headers,
            proxy_transport=parsed.proxy_transport,
        )

    def _add_user_model(self, instance: str, model: CustomModel) -> None:
        parsed = try_parse_pydantic(SrvMcpUserModel, model.data)

        if not self.models.get(instance):
            self.models[instance] = {}

        if parsed.id in self.models[instance]:
            raise HTTPException(400, f"Model with {parsed.id} id already exists.")

        prefix = parsed.default_prefix or normalize_name(parsed.id)
        self._check_prefix_collision(instance, prefix, exclude_model_id=None)

        srv_model = self._build_user_model(instance, parsed, custom_id=model.id)
        self.models[instance][parsed.id] = srv_model

    def _add_proxy_model(self, instance: str, model: CustomModel) -> None:
        parsed = try_parse_pydantic(SrvMcpProxyModel, model.data)

        if not self.models.get(instance):
            self.models[instance] = {}

        if parsed.id in self.models[instance]:
            raise HTTPException(400, f"Model with {parsed.id} id already exists.")

        prefix = parsed.default_prefix or normalize_name(parsed.id)
        self._check_prefix_collision(instance, prefix, exclude_model_id=None)

        required_headers = list(parsed.required_headers.keys() if parsed.required_headers else [])
        self.models[instance][parsed.id] = SrvMcpModel(
            model_props=ModelProps(private=parsed.private, type="mcp", endpoints=[f"/mcp/{prefix}/mcp"]),
            model_spec=self.get_default_model_spec(prefix, None, parsed.required_headers),
            model_type="mcp",
            default_prefix=prefix,
            size="",
            options=None,
            custom=model.id,
            kind="proxy",
            headers=parsed.headers,
            required_headers=required_headers,
            proxy_url=parsed.server_url,
            proxy_transport=parsed.transport,
        )

    def _check_prefix_collision(self, instance: str, prefix: str, exclude_model_id: str | None) -> None:
        for mid, m in self.models.get(instance, {}).items():
            if mid != exclude_model_id and m.default_prefix == prefix:
                raise HTTPException(400, f"Prefix '{prefix}' is already in use by model '{mid}'.")

    async def _update_custom_model(self, instance: str, model: CustomModel, new_data: dict[str, Any]) -> None:
        kind = model.data.get("kind")
        if kind == "proxy":
            parsed_old = try_parse_pydantic(SrvMcpProxyModel, model.data)
            parsed_new = try_parse_pydantic(SrvMcpProxyModel, new_data)
            if parsed_new.id != parsed_old.id:
                raise HTTPException(400, "Cannot change the server ID.")
            installed = self.get_instance_info(instance).installed
            if installed and parsed_old.id in installed.models:
                raise HTTPException(400, "Cannot update an installed server. Uninstall it first.")
            new_prefix = parsed_new.default_prefix or normalize_name(parsed_new.id)
            self._check_prefix_collision(instance, new_prefix, exclude_model_id=parsed_old.id)
            if instance in self.models and parsed_old.id in self.models[instance]:
                del self.models[instance][parsed_old.id]
            self._add_proxy_model(instance, CustomModel(id=model.id, data=new_data))
            return
        if kind != "user":
            raise HTTPException(400, "Only user-defined MCP servers support editing.")

        parsed_old = try_parse_pydantic(SrvMcpUserModel, model.data)
        parsed_new = try_parse_pydantic(SrvMcpUserModel, new_data)

        if parsed_new.id != parsed_old.id:
            raise HTTPException(400, "Cannot change the server ID.")

        installed = self.get_instance_info(instance).installed
        if installed and parsed_old.id in installed.models:
            raise HTTPException(400, "Cannot update an installed server. Uninstall it first.")

        new_prefix = parsed_new.default_prefix or normalize_name(parsed_new.id)
        self._check_prefix_collision(instance, new_prefix, exclude_model_id=parsed_old.id)

        old_name = normalize_name(f"{parsed_old.id}_{instance}")
        old_tag = f"deepfellow-mcp-{old_name}:latest"

        # Build the new model first (writes Dockerfile). If this fails the old model is preserved.
        new_srv_model = self._build_user_model(instance, parsed_new, custom_id=model.id)

        await self.docker_service.remove_image(old_tag)
        if instance in self.models and parsed_old.id in self.models[instance]:
            del self.models[instance][parsed_old.id]
        if instance not in self.models:
            self.models[instance] = {}
        self.models[instance][parsed_new.id] = new_srv_model

    def _remove_custom_model(self, instance: str, model: CustomModel) -> None:
        installed = self.get_instance_info(instance).installed
        kind = model.data.get("kind")
        if kind == "user":
            parsed = try_parse_pydantic(SrvMcpUserModel, model.data)
            if installed and parsed.id in installed.models:
                raise HTTPException(400, "Cannot remove custom model, it is in use, uninstall it first.")
            self._delete_dockerfile_dir(instance, parsed.id)
            if instance in self.models and parsed.id in self.models[instance]:
                del self.models[instance][parsed.id]
        elif kind == "proxy":
            parsed_proxy = try_parse_pydantic(SrvMcpProxyModel, model.data)
            if installed and parsed_proxy.id in installed.models:
                raise HTTPException(400, "Cannot remove custom model, it is in use, uninstall it first.")
            if instance in self.models and parsed_proxy.id in self.models[instance]:
                del self.models[instance][parsed_proxy.id]
        else:
            parsed_custom = try_parse_pydantic(SrvMcpCustomModel, model.data)
            if installed and parsed_custom.id in installed.models:
                raise HTTPException(400, "Cannot remove custom model, it is in use, uninstall it first.")
            if instance in self.models and parsed_custom.id in self.models[instance]:
                del self.models[instance][parsed_custom.id]

    def _get_custom_spec(self, model_id: str, model: SrvMcpModel) -> dict[str, Any] | None:
        if not model.custom:
            return None
        if model.kind == "proxy":
            spec: dict[str, Any] = {
                "kind": "proxy",
                "id": model_id,
                "name": model_id,
                "server_url": model.proxy_url,
                "transport": model.proxy_transport,
                "default_prefix": model.default_prefix,
            }
            if model.headers:
                spec["headers"] = model.headers
            return spec
        if model.kind == "user":
            spec = {
                "kind": "user",
                "id": model_id,
                "name": model_id,
                "command": model.command or "",
                "variant": model.variant,
                "default_prefix": model.default_prefix,
            }
            if model.envs:
                spec["envs"] = model.envs
            if model.base_image:
                spec["base_image"] = model.base_image
            if model.python_version:
                spec["python_version"] = model.python_version
            if model.node_version:
                spec["node_version"] = model.node_version
            return spec
        return None

    async def list_models(self, input_instance: str | list[str] | None, filters: ListModelsFilters) -> ListModelsOut:
        """List models."""
        instances = [input_instance] if isinstance(input_instance, str) else input_instance if input_instance else self.instances_info

        for instance in instances:
            if instance not in self.instances_info:
                raise HTTPException(404, f"Instance {instance} doesn't exist.")

        out_list: list[RetrieveModelOut] = []
        for instance_name, instance_models in self.models.items():
            if instance_name not in instances:
                continue

            info = self.get_instance_installed_info(instance_name)
            for model_id, model in instance_models.items():
                if model_id in info.models:
                    installed = info.models[model_id].get_info()
                else:
                    installed = self._get_model_installed_info(instance_name, model_id)

                if filters.installed is None or filters.installed == bool(installed):
                    out_list.append(
                        RetrieveModelOut(
                            id=model_id,
                            service=self.get_id(instance_name),
                            type=model.model_type,
                            installed=installed,
                            downloaded=model_id in self.models_downloaded,
                            size=model.size,
                            custom=model.custom,
                            spec=model.model_spec,
                            has_docker=model.kind != "proxy",
                            variant=model.variant,
                            command=model.command,
                            base_image=model.base_image,
                            custom_spec=self._get_custom_spec(model_id, model),
                        )
                    )

        return ListModelsOut(list=out_list)

    async def get_model(self, instance: str, model_id: str) -> RetrieveModelOut:
        """Get the model."""
        info = self.get_instance_installed_info(instance)
        if not self.models.get(instance):
            self.models[instance] = {}
        if model_id not in self.models[instance]:
            raise HTTPException(status_code=400, detail="Model not found")

        model = self.models[instance][model_id]
        installed = info.models[model_id].get_info() if model_id in info.models else self._get_model_installed_info(instance, model_id)
        return RetrieveModelOut(
            id=model_id,
            service=self.get_id(instance),
            type=model.model_type,
            installed=installed,
            downloaded=model_id in self.models_downloaded,
            size=model.size,
            custom=model.custom,
            spec=model.model_spec,
            has_docker=model.kind != "proxy",
            variant=model.variant,
            command=model.command,
            base_image=model.base_image,
            custom_spec=self._get_custom_spec(model_id, model),
        )

    def check_envs(self, required_envs: list[str] | None, envs: dict[str, str]) -> None:
        """Check enviromental variables."""
        if required_envs and envs:
            missing_keys = [key for key in required_envs if key not in envs]
            if missing_keys:
                raise HTTPException(
                    status_code=422, detail=f"The following required environment variables are missing: {', '.join(missing_keys)}"
                )

            empty_keys = [key for key in required_envs if not envs.get(key)]
            if empty_keys:
                raise HTTPException(
                    status_code=422, detail=f"The following environment variables are present but have no value: {', '.join(empty_keys)}"
                )

    def check_headers(self, required_headers: list[str] | None, headers: dict[str, str]) -> None:
        """Check headers."""
        if required_headers and headers:
            missing_keys = [key for key in required_headers if key not in headers]
            if missing_keys:
                raise HTTPException(status_code=422, detail=f"The following required headers are missing: {', '.join(missing_keys)}")

            empty_keys = [key for key in required_headers if not headers.get(key)]
            if empty_keys:
                raise HTTPException(status_code=422, detail=f"The following headers are present but have no value: {', '.join(empty_keys)}")

    def _register_proxy_model(self, model: SrvMcpModel, parsed_options: McpModelOptions) -> RegistrationId:
        """Register a proxy model endpoint, choosing SSE or Streamable HTTP transport."""
        if model.proxy_url is None:
            raise HTTPException(400, "proxy_url is required for proxy models")
        merged_headers = {**(model.headers or {}), **parsed_options.headers}
        proxy_options = ProxyOptions(
            url=model.proxy_url,
            headers=merged_headers if merged_headers else None,
            allowed_request_headers=["accept", "mcp-session-id"],
            allowed_response_headers=["accept", "mcp-session-id"],
        )
        if model.proxy_transport == "sse":
            return self.endpoint_registry.register_mcp_sse_endpoint_as_proxy(
                url=parsed_options.prefix, props=model.model_props, options=proxy_options, registration_options=None
            )
        return self.endpoint_registry.register_mcp_endpoint_as_proxy(
            url=parsed_options.prefix, props=model.model_props, options=proxy_options, registration_options=None
        )

    async def _install_model(  # noqa: C901
        self, instance: str, model_id: str, options: InstallModelIn
    ) -> PromiseWithProgress[InstallModelOut, StreamChunk]:
        info = self.get_instance_installed_info(instance)
        self.models.setdefault(instance, {})
        if model_id in info.models:
            return PromiseWithProgress(value=InstallModelOut(status="OK", details="Already installed"))
        if model_id not in self.models[instance]:
            raise HTTPException(400, "Model not found")
        model = self.models[instance][model_id]
        if not options.spec:
            options.spec = {}
        if "prefix" not in options.spec:
            options.spec["prefix"] = model.default_prefix
        model.model_props.prefix = options.spec["prefix"]
        parsed_model_options = try_parse_pydantic(McpModelOptions, options.spec)

        self.check_envs(model.required_envs, parsed_model_options.envs)
        self.check_headers(model.required_headers, parsed_model_options.headers)

        if model.kind == "proxy":
            registration_id = self._register_proxy_model(model, parsed_model_options)
            assert model.proxy_url is not None
            info.models[model_id] = ModelInstalledInfo(
                id=model_id,
                options=options,
                docker_options=None,
                container_host="",
                container_port=0,
                docker_exposed_port=0,
                registration_id=registration_id,
                prefix=parsed_model_options.prefix,
                base_url=model.proxy_url,
                headers=parsed_model_options.headers,
                envs={},
            )
            return PromiseWithProgress(value=InstallModelOut(status="OK", details="Installed"))

        if model.options is None:
            raise HTTPException(400, "options are required for this model kind")
        if model.kind != "user":
            await self._verify_docker_image(model.options.image, options.ignore_warnings)

        async def func(stream: Stream[StreamChunk]) -> InstallModelOut:
            model_dir = self._get_working_dir() / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            subnet = self.docker_service.get_docker_subnet()
            docker_options = model.options
            assert docker_options is not None
            if model.kind == "user":
                dockerfile_dir = self._get_dockerfile_dir(instance, model_id)
                await self.docker_service.build_image(dockerfile_dir, docker_options.image, stream)
                size_bytes = await self.docker_service.get_local_docker_image_size(docker_options.image)
                if size_bytes:
                    model.size = fmt_size(size_bytes)
                    await self._persist_custom_model_size(instance, model)
            else:
                image = DockerImage(name=docker_options.image, size=model.size)
                await self._download_image_or_set_progress(stream, image)
            stream.emit(StreamChunkProgress(type="progress", stage="install", value=0, data={}))
            docker_options_edited = deepcopy(docker_options)
            docker_options_edited.env_vars = docker_options_edited.env_vars | parsed_model_options.envs
            docker_exposed_port = await self.docker_service.install_and_run_docker(docker_options_edited)
            container_host = self.docker_service.get_container_host(subnet, docker_options_edited.name)
            container_port = self.docker_service.get_container_port(subnet, docker_exposed_port, docker_options_edited.image_port)
            info.models[model_id] = model_info = ModelInstalledInfo(
                id=model_id,
                options=options,
                docker_options=docker_options_edited,
                container_host=container_host,
                container_port=container_port,
                docker_exposed_port=docker_exposed_port,
                registration_id="",
                prefix=parsed_model_options.prefix,
                base_url=get_base_url(container_host, container_port),
                headers=parsed_model_options.headers,
                envs=parsed_model_options.envs,
            )
            if model.proxy_transport == "sse":
                model_info.registration_id = self.endpoint_registry.register_mcp_sse_endpoint_as_proxy(
                    url=model_info.prefix,
                    props=model.model_props,
                    options=ProxyOptions(
                        url=model_info.base_url + "/sse",
                        allowed_request_headers=["accept", "mcp-session-id"],
                        allowed_response_headers=["accept", "mcp-session-id"],
                        headers=model.headers | parsed_model_options.headers if model.headers else parsed_model_options.headers,
                    ),
                    registration_options=None,
                )
            else:
                model_info.registration_id = self.endpoint_registry.register_mcp_endpoint_as_proxy(
                    url=model_info.prefix,
                    props=model.model_props,
                    options=ProxyOptions(
                        url=model_info.base_url + "/mcp",
                        allowed_request_headers=["accept", "mcp-session-id"],
                        allowed_response_headers=["accept", "mcp-session-id"],
                        headers=model.headers | parsed_model_options.headers if model.headers else parsed_model_options.headers,
                    ),
                    registration_options=None,
                )
            self.models_downloaded[model_id] = DownloadedInfo(docker_options.image)
            stream.emit(StreamChunkProgress(type="progress", stage="install", value=1, data={}))
            return InstallModelOut(status="OK", details="Installed")

        return PromiseWithProgress(func=func)

    async def _uninstall_model(self, instance: str, model_id: str, options: UninstallModelIn) -> None:
        info = self.get_instance_installed_info(instance)
        srv_model = self.models.get(instance, {}).get(model_id)
        if model_id in info.models:
            model_info = info.models[model_id]
            self.endpoint_registry.unregister_mcp_endpoint(model_info.prefix, model_info.registration_id)
            if srv_model and srv_model.kind != "proxy" and model_info.docker_options:
                await self.docker_service.uninstall_docker(model_info.docker_options)
            del info.models[model_id]

        if options.purge and model_id in self.models_downloaded:
            await self.docker_service.remove_image(self.models_downloaded[model_id].image)
            del self.models_downloaded[model_id]

        if options.purge and srv_model and srv_model.kind == "user":
            self._delete_dockerfile_dir(instance, model_id)

    def get_working_dir(self) -> Path:
        """Get working dir."""
        return self._get_working_dir()


_const = McpConst(
    models={
        "open-websearch": lambda mcp_service, subnet: SrvMcpModel(
            model_props=ModelProps(private=True, type="mcp", endpoints=["/mcp/open-websearch/mcp"]),
            model_spec=mcp_service.get_default_model_spec("open-websearch"),
            model_type="mcp",
            default_prefix="open-websearch",
            size="427MB",
            options=DockerOptions(
                image_port=3000,
                name="open-websearch",
                container_name=mcp_service.docker_service.get_docker_container_name("open-websearch"),
                image="hub.simplito.com/deepfellow/open-websearch:v1.2.0",
                env_vars={},
                subnet=subnet,
            ),
        ),
        "brave-search": lambda mcp_service, subnet: SrvMcpModel(
            model_props=ModelProps(private=True, type="mcp", endpoints=["/mcp/brave-search/mcp"]),
            model_spec=mcp_service.get_default_model_spec(default_prefix="brave-search", required_envs=["BRAVE_API_KEY"]),
            model_type="mcp",
            default_prefix="brave-search",
            size="316MB",
            options=DockerOptions(
                image_port=8080,
                name="brave-search",
                container_name=mcp_service.docker_service.get_docker_container_name("brave-search"),
                image="hub.simplito.com/deepfellow/brave-search-mcp-server:v2.0.72",
                env_vars={},
                subnet=subnet,
            ),
            required_envs=["BRAVE_API_KEY"],
        ),
        "web-search": lambda mcp_service, subnet: SrvMcpModel(
            model_props=ModelProps(private=True, type="mcp", endpoints=["/mcp/web-search/mcp"]),
            model_spec=mcp_service.get_default_model_spec("web-search"),
            model_type="mcp",
            default_prefix="web-search",
            size="3.84GB",
            options=DockerOptions(
                image_port=8080,
                name="web-search",
                container_name=mcp_service.docker_service.get_docker_container_name("web-search"),
                image="hub.simplito.com/deepfellow/web-search-mcp:v0.3.2",
                env_vars={},
                subnet=subnet,
            ),
        ),
        "serpapi": lambda mcp_service, subnet: SrvMcpModel(
            model_props=ModelProps(private=True, type="mcp", endpoints=["/mcp/serpapi/mcp"]),
            model_spec=mcp_service.get_default_model_spec(default_prefix="serpapi", required_headers={"Authorization": "Bearer "}),
            model_type="mcp",
            default_prefix="serpapi",
            size="454MB",
            options=DockerOptions(
                image_port=8000,
                name="serpapi",
                container_name=mcp_service.docker_service.get_docker_container_name("serpapi"),
                image="hub.simplito.com/deepfellow/serpapi-mcp:61998a0",
                env_vars={},
                subnet=subnet,
            ),
            required_headers=["Authorization"],
        ),
        "ollama-websearch": lambda mcp_service, subnet: SrvMcpModel(
            model_props=ModelProps(private=True, type="mcp", endpoints=["/mcp/ollama-websearch/mcp"]),
            model_spec=mcp_service.get_default_model_spec(default_prefix="ollama-websearch", required_envs=["OLLAMA_API_KEY"]),
            model_type="mcp",
            default_prefix="ollama-websearch",
            size="625MB",
            options=DockerOptions(
                image_port=8000,
                name="ollama-websearch",
                container_name=mcp_service.docker_service.get_docker_container_name("ollama-websearch"),
                image="hub.simplito.com/deepfellow/ollama-websearch-mcp-server:v1.0.2",
                env_vars={},
                subnet=subnet,
            ),
            required_envs=["OLLAMA_API_KEY"],
        ),
        "scrapling": lambda mcp_service, subnet: SrvMcpModel(
            model_props=ModelProps(private=True, type="mcp", endpoints=["/mcp/scrapling/mcp"]),
            model_spec=mcp_service.get_default_model_spec(default_prefix="scrapling"),
            model_type="mcp",
            default_prefix="scrapling",
            size="2GB",
            options=DockerOptions(
                image_port=8000,
                name="scrapling",
                container_name=mcp_service.docker_service.get_docker_container_name("scrapling"),
                image="ghcr.io/d4vinci/scrapling@sha256:77af4d59a6d00e40b918358943503ee6cafc44ad21fb60d5a545e17d0d40cd7a",
                subnet=subnet,
                command="mcp --http",
            ),
        ),
        "firecrawl": lambda mcp_service, subnet: SrvMcpModel(
            model_props=ModelProps(private=True, type="mcp", endpoints=["/mcp/firecrawl/mcp"]),
            model_spec=mcp_service.get_default_model_spec(default_prefix="firecrawl", required_envs=["FIRECRAWL_API_KEY"]),
            model_type="mcp",
            default_prefix="firecrawl",
            size="506MB",
            options=DockerOptions(
                image_port=3000,
                name="firecrawl",
                container_name=mcp_service.docker_service.get_docker_container_name("firecrawl"),
                image="mcp/firecrawl@sha256:a3b74109dced0a16aea59e3c38903fa9a0788498e8652ddfbecf0155172f7af6",
                subnet=subnet,
                command="tail -f /dev/null",
                env_vars={"HTTP_STREAMABLE_SERVER": "true", "HOST": "0.0.0.0", "PORT": "3000"},
            ),
            required_envs=["FIRECRAWL_API_KEY"],
        ),
        "duckduckgo": lambda mcp_service, subnet: SrvMcpModel(
            model_props=ModelProps(private=True, type="mcp", endpoints=["/mcp/duckduckgo/mcp"]),
            model_spec=mcp_service.get_default_model_spec(default_prefix="duckduckgo"),
            model_type="mcp",
            default_prefix="duckduckgo",
            size="276MB",
            options=DockerOptions(
                image_port=8000,
                name="duckduckgo",
                container_name=mcp_service.docker_service.get_docker_container_name("duckduckgo"),
                image="ghcr.io/nickclyde/duckduckgo-mcp-server:0.4.0",
                subnet=subnet,
                command="python -m duckduckgo_mcp_server.server --transport streamable-http --host 0.0.0.0 --port 8000",
            ),
        ),
    },
)
