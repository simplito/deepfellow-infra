# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Mcp service."""

import asyncio
import html
import json
import secrets
import shlex
import shutil
import time
from collections.abc import Awaitable, Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urljoin, urlparse

import aiohttp
from fastapi import Depends, HTTPException
from fastapi import Path as ApiPath
from pydantic import BaseModel, Field, field_validator

from server.applicationcontext import get_base_url
from server.core.dependencies import get_services_manager
from server.docker import DockerImage, DockerOptions
from server.endpointregistry import ProxyOptions, RegistrationId, RegistrationOptions
from server.models.api import ModelProps
from server.models.models import (
    CustomModelField,
    CustomModelId,
    CustomModelSpecification,
    InstallModelIn,
    InstallModelOut,
    ListModelsFilters,
    ListModelsOut,
    McpHealthCheckResult,
    McpToolInfo,
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
from server.services_manager import ServicesManager
from server.utils.core import (
    PromiseWithProgress,
    Stream,
    StreamChunk,
    StreamChunkProgress,
    Utils,
    normalize_name,
    try_parse_pydantic,
)
from server.utils.docker_image_version import apply_image_version_override, docker_tags_model_field, split_image_repo_tag
from server.utils.logger import logger
from server.utils.mcp_oauth import (
    McpOAuthConfig,
    McpOAuthError,
    McpOAuthStateStore,
    PendingOAuthFlow,
    build_authorize_url,
    discover_authorization_server_for_resource,
    exchange_code_for_token,
    generate_pkce_pair,
    has_valid_access_token,
    refresh_access_token,
    register_dynamic_client,
)
from server.utils.registry_client import image_without_registry_prefix, registry_for
from server.utils.size_fetcher import fmt_size
from server.utils.tracing import tracer


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

_HEALTH_MONITOR_INTERVAL_SECONDS = 30.0
# Consecutive failed probes before a healthy MCP is hidden; a single timeout must not flap it (and the mesh parent).
_HEALTH_MONITOR_FAILURE_THRESHOLD = 3


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
    description: str = field(default="")
    repository_url: str | None = field(default=None)
    oauth: McpOAuthConfig | None = field(default=None)


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
    description: str = ""
    repository_url: str | None = None


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
    description: str = ""
    repository_url: str | None = None


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
    description: str = ""
    repository_url: str | None = None
    oauth: McpOAuthConfig | None = None


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


class McpOAuthStatusOut(BaseModel):
    """Non-secret OAuth status for a proxy model, returned to the WebUI."""

    enabled: bool
    status: Literal["disabled", "not_started", "pending", "authorized", "expired", "error"]
    has_client_id: bool = False
    has_client_secret: bool = False
    expires_at: float | None = None
    last_error: str | None = None


def get_mcp_service_instance(
    service_id: Annotated[str, ApiPath(description="The ID of the MCP service instance to use.")],
    services_manager: Annotated[ServicesManager, Depends(get_services_manager)],
) -> tuple["McpService", str]:
    """Resolve `service_id` to the MCP service and its instance name."""
    service_type, instance = services_manager.split_service_type_and_instance(service_id)
    service = services_manager.services.get(service_type)
    if not isinstance(service, McpService):
        raise HTTPException(404, f"Service {service_id} is not an MCP service")
    return service, instance


def render_oauth_callback_page(message: str) -> str:
    """Render the HTML page shown to the admin's browser after an OAuth redirect."""
    return f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>DeepFellow — MCP Authorization</title></head>
<body style="font-family: system-ui, sans-serif; display: flex; align-items: center; justify-content: center; height: 100vh; margin: 0;">
  <p style="max-width: 32rem; text-align: center;">{html.escape(message)}</p>
</body>
</html>"""


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


_MCP_INIT_PAYLOAD: dict[str, Any] = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "deepfellow-healthcheck", "version": "1.0"},
    },
}
_MCP_INITIALIZED_NOTIF: dict[str, Any] = {"jsonrpc": "2.0", "method": "notifications/initialized"}
_MCP_TOOLS_LIST_PAYLOAD: dict[str, Any] = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
_MCP_SESSION_TERMINATE_TIMEOUT = aiohttp.ClientTimeout(total=5)


def _parse_mcp_tools(raw: list[Any]) -> list[McpToolInfo]:
    tools = []
    for t in raw:
        if isinstance(t, dict):
            tools.append(
                McpToolInfo(
                    name=t.get("name", ""),
                    description=t.get("description", ""),
                    input_schema=t.get("inputSchema") or t.get("input_schema") or {},
                )
            )
    return tools


async def _read_first_sse_json(resp: aiohttp.ClientResponse) -> dict[str, Any] | None:
    """Read an SSE stream and return the data of the first non-empty data-only event."""
    data_buf = ""
    async for raw_line in resp.content:
        try:
            line = raw_line.decode("utf-8").rstrip("\r\n")
        except UnicodeDecodeError:
            continue
        if line.startswith("data:"):
            data_buf += line[5:].strip()
        elif line == "" and data_buf:
            try:
                return json.loads(data_buf)
            except json.JSONDecodeError:
                pass
            data_buf = ""
    return None


async def _terminate_mcp_session(client: aiohttp.ClientSession, mcp_url: str, headers: dict[str, str]) -> None:
    """Best-effort DELETE of a streamable-HTTP MCP session; failures are logged and ignored."""
    try:
        async with client.delete(mcp_url, headers=headers, timeout=_MCP_SESSION_TERMINATE_TIMEOUT) as resp:
            logger.debug("MCP session termination at %s returned HTTP %s", mcp_url, resp.status)
    except Exception as exc:
        logger.debug("MCP session termination at %s failed: %s: %s", mcp_url, type(exc).__name__, exc)


async def _fetch_tools_from_mcp_endpoint(mcp_url: str, extra_headers: dict[str, str]) -> McpHealthCheckResult:  # noqa: C901
    """Probe a streamable-HTTP MCP server, run the full init handshake, and return tools."""
    try:
        conn_timeout = aiohttp.ClientTimeout(total=15, connect=5)
        session_id: str | None = None
        unauthorized = False

        async with aiohttp.ClientSession() as client:

            def _req_headers() -> dict[str, str]:
                h: dict[str, str] = {
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                    **extra_headers,
                }
                if session_id:
                    h["Mcp-Session-Id"] = session_id
                return h

            async def _post(payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
                nonlocal session_id, unauthorized
                async with client.post(mcp_url, json=payload, headers=_req_headers(), timeout=conn_timeout) as resp:
                    if sid := resp.headers.get("Mcp-Session-Id"):
                        session_id = sid
                    if resp.status == 401:
                        unauthorized = True
                        return None, "HTTP 401"
                    if resp.status == 202:
                        return None, None
                    if resp.status != 200:
                        return None, f"HTTP {resp.status}"
                    ct = resp.headers.get("Content-Type", "")
                    data = await _read_first_sse_json(resp) if "text/event-stream" in ct else await resp.json()
                    if data is None:
                        return None, "Empty response"
                    if "error" in data:
                        return None, str(data["error"].get("message", data["error"]))
                    return data.get("result"), None

            async def _close_session() -> None:
                if session_id:
                    await _terminate_mcp_session(client, mcp_url, _req_headers())

            try:
                _, init_err = await _post(_MCP_INIT_PAYLOAD)
                if unauthorized:
                    return McpHealthCheckResult(healthy=False, requires_oauth=True, error="initialize: HTTP 401 Unauthorized")
                if init_err:
                    return McpHealthCheckResult(healthy=False, error=f"initialize: {init_err}")

                await _post(_MCP_INITIALIZED_NOTIF)

                tools_result, tools_err = await _post(_MCP_TOOLS_LIST_PAYLOAD)
                if tools_err:
                    return McpHealthCheckResult(healthy=True, transport="streamable_http", error=f"tools/list: {tools_err}")

                raw_tools = (tools_result or {}).get("tools", [])
                return McpHealthCheckResult(healthy=True, transport="streamable_http", tools=_parse_mcp_tools(raw_tools))
            finally:
                await _close_session()

    except aiohttp.ClientConnectorError as exc:
        return McpHealthCheckResult(healthy=False, error=f"Connection refused: {exc}")
    except Exception as exc:
        return McpHealthCheckResult(healthy=False, error=f"{type(exc).__name__}: {exc}")


@dataclass
class _SseState:
    endpoint_ready: asyncio.Event
    session_url_holder: list[str]
    response_futures: dict[int, asyncio.Future[dict[str, Any]]]


def _dispatch_sse_event(event_name: str, data_buf: str, state: _SseState) -> None:
    if event_name == "endpoint" and data_buf:
        state.session_url_holder.append(data_buf)
        state.endpoint_ready.set()
        return
    if not data_buf:
        return
    try:
        msg: dict[str, Any] = json.loads(data_buf)
        msg_id: int | None = msg.get("id")
        if msg_id is not None and msg_id in state.response_futures:
            fut = state.response_futures[msg_id]
            if not fut.done():
                fut.set_result(msg)
    except json.JSONDecodeError:
        pass


async def _run_sse_reader(sse_resp: aiohttp.ClientResponse, state: _SseState) -> None:
    event_name = ""
    data_buf = ""
    async for raw_line in sse_resp.content:
        try:
            line = raw_line.decode("utf-8").rstrip("\r\n")
        except UnicodeDecodeError:
            continue
        if line.startswith("event:"):
            event_name = line[6:].strip()
        elif line.startswith("data:"):
            data_buf += line[5:].strip()
        elif line == "":
            _dispatch_sse_event(event_name, data_buf, state)
            event_name = ""
            data_buf = ""


async def _sse_rpc(
    client: aiohttp.ClientSession,
    session_url: str,
    payload: dict[str, Any],
    state: _SseState,
    *,
    response_timeout: float = 10.0,
) -> dict[str, Any] | None:
    loop = asyncio.get_event_loop()
    msg_id: int | None = payload.get("id")
    fut: asyncio.Future[dict[str, Any]] | None = None
    if msg_id is not None:
        fut = loop.create_future()
        state.response_futures[msg_id] = fut
    try:
        async with client.post(
            session_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=aiohttp.ClientTimeout(total=5),
        ) as _:
            pass
    except Exception:
        if fut and not fut.done():
            fut.cancel()
        raise
    else:
        if fut is not None:
            return await asyncio.wait_for(asyncio.shield(fut), timeout=response_timeout)
        return None


async def _fetch_tools_from_sse_endpoint(sse_url: str, extra_headers: dict[str, str]) -> McpHealthCheckResult:  # noqa: C901
    """Probe an SSE-transport MCP server, run the full init handshake, and return tools."""
    try:
        timeout = aiohttp.ClientTimeout(total=25, connect=5)
        sse_headers = {"Accept": "text/event-stream", **extra_headers}
        state = _SseState(asyncio.Event(), [], {})

        async with aiohttp.ClientSession() as client, client.get(sse_url, headers=sse_headers, timeout=timeout) as sse_resp:
            if sse_resp.status == 401:
                return McpHealthCheckResult(healthy=False, requires_oauth=True, error="SSE GET returned HTTP 401 Unauthorized")
            if sse_resp.status != 200:
                return McpHealthCheckResult(healthy=False, error=f"SSE GET returned HTTP {sse_resp.status}")
            ct = sse_resp.headers.get("Content-Type", "")
            if "text/event-stream" not in ct:
                return McpHealthCheckResult(healthy=False, error=f"Not an SSE endpoint (Content-Type: {ct})")

            reader_task = asyncio.create_task(_run_sse_reader(sse_resp, state))
            endpoint_wait = state.endpoint_ready.wait()
            try:
                await asyncio.wait_for(endpoint_wait, timeout=8.0)
            except TimeoutError:
                endpoint_wait.close()
                return McpHealthCheckResult(healthy=True, transport="sse", error="Timeout waiting for SSE endpoint event")
            finally:
                if not state.endpoint_ready.is_set():
                    reader_task.cancel()

            raw_session_url = state.session_url_holder[0]
            if raw_session_url.startswith(("http://", "https://")):
                session_url = raw_session_url
            else:
                parsed = urlparse(sse_url)
                session_url = urljoin(f"{parsed.scheme}://{parsed.netloc}", raw_session_url)

            try:
                await _sse_rpc(client, session_url, _MCP_INIT_PAYLOAD, state)
                await _sse_rpc(client, session_url, _MCP_INITIALIZED_NOTIF, state)
                tools_data = await _sse_rpc(client, session_url, _MCP_TOOLS_LIST_PAYLOAD, state)
            except TimeoutError:
                return McpHealthCheckResult(healthy=True, transport="sse", error="Timeout waiting for MCP response")

            if tools_data is None:
                return McpHealthCheckResult(healthy=True, transport="sse", error="No response to tools/list")
            if "error" in tools_data:
                err = tools_data["error"].get("message", tools_data["error"])
                return McpHealthCheckResult(healthy=True, transport="sse", error=f"tools/list: {err}")

            raw_tools = tools_data.get("result", {}).get("tools", [])
            return McpHealthCheckResult(healthy=True, transport="sse", tools=_parse_mcp_tools(raw_tools))

    except aiohttp.ClientConnectorError as exc:
        return McpHealthCheckResult(healthy=False, error=f"Connection refused: {exc}")
    except Exception as exc:
        return McpHealthCheckResult(healthy=False, error=f"{type(exc).__name__}: {exc}")


class McpService(Base2Service[InstalledInfo, DownloadedInfo]):
    models: dict[str, dict[str, SrvMcpModel]]
    _installing: set[tuple[str, str]]
    _persists_model_definitions = False

    def _after_init(self) -> None:
        self._background_tasks: set[asyncio.Task[None]] = set()
        self.load_default_models("default")
        self._installing = set()
        self._oauth_state_store = McpOAuthStateStore()
        self._oauth_refresh_locks: dict[tuple[str, str], asyncio.Lock] = {}
        self._oauth_refresh_tasks: dict[tuple[str, str], asyncio.Task[None]] = {}
        self._health_monitor_tasks: dict[tuple[str, str], asyncio.Task[None]] = {}

    def load_default_models(self, instance: str) -> None:
        """Load default models to instance."""
        self.models[instance] = {}
        subnet = self.docker_service.get_docker_subnet()
        for model_id in _const.models.copy():
            model = _const.models[model_id](self, subnet)
            self._attach_docker_tags_field(model)
            self.models[instance][model_id] = model

    def _attach_docker_tags_field(self, model: "SrvMcpModel") -> None:
        """Append a "docker-tags" version-selection field, inferred from the model's own default image.

        Skipped for models without a pinned registry image (proxy servers, user-built images) or
        whose image is digest-pinned rather than tagged (no floating version to select).
        """
        if model.options is None or any(f.type == "docker-tags" for f in model.model_spec.fields):
            return
        repo, tag = split_image_repo_tag(model.options.image)
        if not tag:
            return
        model.model_spec.fields.append(docker_tags_model_field(repo))

    def _get_builtin_model(self, model_id: str | None) -> "SrvMcpModel | None":
        if not model_id:
            return None
        return self.models.get("default", {}).get(model_id)

    def get_docker_image_repo_for_model(self, model_id: str | None, hardware: str | None) -> str | None:  # noqa: ARG002
        """Return the Docker repo for a built-in MCP model, derived from its own default image."""
        model = self._get_builtin_model(model_id)
        if not model or not model.options:
            return None
        return split_image_repo_tag(model.options.image)[0]

    def get_default_docker_tag_for_model(self, model_id: str | None, hardware: str | None) -> str | None:  # noqa: ARG002
        """Return a built-in MCP model's default image tag, derived from its own default image."""
        model = self._get_builtin_model(model_id)
        if not model or not model.options:
            return None
        return split_image_repo_tag(model.options.image)[1] or None

    async def get_docker_tags_for_model(self, model_id: str | None, hardware: str | None) -> list[str]:
        """Fetch available Docker image tags for a built-in MCP model."""
        repo = self.get_docker_image_repo_for_model(model_id, hardware)
        if not repo:
            return []
        client = registry_for(repo)
        return await client.get_tags(image_without_registry_prefix(repo))

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
                    required_keys=list(required_envs) if required_envs else None,
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
                    required_keys=list(required_headers) if required_headers else None,
                ),
            ]
        )

    async def stop_instance(self, instance: str) -> None:
        """Stop all MCP service Docker containers."""
        for key in [key for key in self._health_monitor_tasks if key[0] == instance]:
            self._stop_health_monitor(*key)
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

    async def _persist_proxy_oauth(self, instance: str, model_id: str, oauth: McpOAuthConfig) -> None:
        """Persist updated OAuth config/tokens for a proxy model back to services.json."""
        model = (self.models.get(instance) or {}).get(model_id)
        if model is None or not model.custom:
            return
        model.oauth = oauth
        config = self.get_instance_info(instance).config
        for custom_model in config.custom or []:
            if custom_model.id == model.custom:
                custom_model.data["oauth"] = oauth.model_dump(mode="json", exclude_none=True)
                await self._save()
                break
        else:
            logger.warning(
                "No custom model entry matching %s found for %s/%s; OAuth token update was not persisted to disk.",
                model.custom,
                instance,
                model_id,
            )

    def get_installed_info(self, instance: str) -> bool | InstallServiceProgress | ServiceOptions:
        """Get service installed info."""
        installed = self.get_instance_info(instance).installed
        return self._get_service_installed_info(instance) if installed is None else installed.options.spec

    def _generate_instance_config(self, instance: str, info: InstalledInfo | None, custom: list[CustomModel] | None) -> InstanceConfig:  # noqa: ARG002
        return InstanceConfig(
            options=info.options if info else None,
            models=[
                ModelConfig(model_id=x.id, options=x.options, container_name=x.docker_options.container_name if x.docker_options else None)
                for x in info.models.values()
            ]
            if info
            else [],
            custom=custom,
        )

    def _load_download_info(self, data: dict[str, Any]) -> DownloadedInfo:
        return DownloadedInfo(**data)

    async def _install_instance(
        self,
        instance: str,
        options: InstallServiceIn,
        container_name_override: str | None = None,  # noqa: ARG002
    ) -> PromiseWithProgress[InstalledInfo, StreamChunk]:
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
            if not any(i.installed for i in self.instances_info.values()):
                self.service_downloaded = False
                await self._clear_working_dir()
                self.models_downloaded = {}

            if instance == "default":
                self.instances_info["default"] = Instance(None, None, {}, InstanceConfig())
            else:
                del self.instances_info[instance]

    def get_docker_options(self, instance: str, model_id: str | None) -> DockerOptions:
        """Return the resolved DockerOptions for this instance/model."""
        info = self.get_instance_installed_info(instance)
        if not model_id:
            raise HTTPException(400, "Docker is not bound with this object")

        model_installed = info.models.get(model_id, None)
        if not model_installed:
            raise HTTPException(status_code=400, detail="Model not installed")

        if model_installed.docker_options is None:
            raise HTTPException(400, "Docker is not bound with this model")
        return model_installed.docker_options

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
            model_props=ModelProps(private=parsed.private, type="mcp", endpoints=[f"/mcp/{prefix}/mcp"], transport="streamable_http"),
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
            description=parsed.description,
            repository_url=parsed.repository_url,
        )

    def _add_custom_model(self, instance: str, model: CustomModel) -> None:
        kind = model.data.get("kind")
        if kind == "user":
            self._add_user_model(instance, model)
        elif kind == "proxy":
            self._add_proxy_model(instance, model)
        else:
            self._add_image_model(instance, model)

    def _build_image_model(self, instance: str, parsed: SrvMcpCustomModel, custom_id: CustomModelId | None = None) -> SrvMcpModel:
        name = normalize_name(f"{parsed.id}-{instance}")
        subnet = self.docker_service.get_docker_subnet()
        required_envs = list(parsed.required_envs.keys() if parsed.required_envs else [])
        required_headers = list(parsed.required_headers.keys() if parsed.required_headers else [])
        return SrvMcpModel(
            model_props=ModelProps(
                private=parsed.private, type="mcp", endpoints=[f"/mcp/{parsed.default_prefix}/mcp"], transport=parsed.proxy_transport
            ),
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
            custom=custom_id,
            headers=parsed.headers,
            required_envs=required_envs,
            required_headers=required_headers,
            proxy_transport=parsed.proxy_transport,
            description=parsed.description,
            repository_url=parsed.repository_url,
        )

    def _add_image_model(self, instance: str, model: CustomModel) -> None:
        parsed = try_parse_pydantic(SrvMcpCustomModel, model.data)

        if not self.models.get(instance):
            self.models[instance] = {}

        if parsed.id in self.models[instance]:
            raise HTTPException(400, f"Model with {parsed.id} id already exists.")
        self._check_prefix_collision(instance, parsed.default_prefix, exclude_model_id=None)
        self.models[instance][parsed.id] = self._build_image_model(instance, parsed, custom_id=model.id)

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

    def _build_proxy_model(self, parsed: SrvMcpProxyModel, prefix: str, custom_id: CustomModelId | None = None) -> SrvMcpModel:
        """Construct a proxy `SrvMcpModel` with no side effects - no registry guard, no registration.

        Kept separate from `_add_proxy_model` so an edit can validate and build the replacement before
        touching `self.models`, the same way the "user"/image branches already do via their own
        `_build_user_model`/`_build_image_model` helpers.
        """
        required_headers = list(parsed.required_headers.keys() if parsed.required_headers else [])
        return SrvMcpModel(
            model_props=ModelProps(private=parsed.private, type="mcp", endpoints=[f"/mcp/{prefix}/mcp"], transport=parsed.transport),
            model_spec=self.get_default_model_spec(prefix, None, parsed.required_headers),
            model_type="mcp",
            default_prefix=prefix,
            size="",
            options=None,
            custom=custom_id,
            kind="proxy",
            headers=parsed.headers,
            required_headers=required_headers,
            proxy_url=parsed.server_url,
            proxy_transport=parsed.transport,
            description=parsed.description,
            repository_url=parsed.repository_url,
            oauth=parsed.oauth,
        )

    def _add_proxy_model(self, instance: str, model: CustomModel) -> None:
        parsed = try_parse_pydantic(SrvMcpProxyModel, model.data)

        if not self.models.get(instance):
            self.models[instance] = {}

        if parsed.id in self.models[instance]:
            raise HTTPException(400, f"Model with {parsed.id} id already exists.")

        prefix = parsed.default_prefix or normalize_name(parsed.id)
        self._check_prefix_collision(instance, prefix, exclude_model_id=None)

        self.models[instance][parsed.id] = self._build_proxy_model(parsed, prefix, custom_id=model.id)

    async def _update_custom_model(self, instance: str, model: CustomModel, new_data: dict[str, Any]) -> None:  # noqa: C901
        kind = model.data.get("kind")
        if kind == "proxy":
            parsed_old = try_parse_pydantic(SrvMcpProxyModel, model.data)
            parsed_new = try_parse_pydantic(SrvMcpProxyModel, new_data)
            if parsed_new.id != parsed_old.id:
                raise HTTPException(400, "Cannot change the server ID.")
            if self._oauth_state_store.find_for_model(instance, parsed_old.id):
                raise HTTPException(400, "Cannot edit this MCP server while an OAuth authorization is pending.")
            installed = self.get_instance_info(instance).installed
            if installed and parsed_old.id in installed.models:
                raise HTTPException(400, "Cannot update an installed server. Uninstall it first.")
            new_prefix = parsed_new.default_prefix or normalize_name(parsed_new.id)
            self._check_prefix_collision(instance, new_prefix, exclude_model_id=parsed_old.id)
            if parsed_old.oauth and parsed_new.oauth:
                # The WebUI never round-trips secrets/tokens (they're scrubbed from custom_spec, see
                # `_get_custom_spec`), so an edit must preserve them rather than silently wiping them out.
                merged_oauth = parsed_new.oauth.model_copy(
                    update={
                        "client_secret": parsed_new.oauth.client_secret or parsed_old.oauth.client_secret,
                        "access_token": parsed_old.oauth.access_token,
                        "refresh_token": parsed_old.oauth.refresh_token,
                        "token_expires_at": parsed_old.oauth.token_expires_at,
                        "authorization_endpoint": parsed_new.oauth.authorization_endpoint or parsed_old.oauth.authorization_endpoint,
                        "token_endpoint": parsed_new.oauth.token_endpoint or parsed_old.oauth.token_endpoint,
                        "registration_endpoint": parsed_new.oauth.registration_endpoint or parsed_old.oauth.registration_endpoint,
                        "resource": parsed_new.oauth.resource or parsed_old.oauth.resource,
                        "last_error": parsed_old.oauth.last_error,
                    }
                )
                new_data = {**new_data, "oauth": merged_oauth.model_dump(mode="json", exclude_none=True)}
            elif parsed_old.oauth and not parsed_new.oauth:
                logger.warning(
                    "Edit request for MCP server %s/%s omitted the oauth field while an OAuth config existed; "
                    "preserving the existing OAuth config instead of erasing it.",
                    instance,
                    parsed_old.id,
                )
                new_data = {**new_data, "oauth": parsed_old.oauth.model_dump(mode="json", exclude_none=True)}
            # Build (and validate) the new model first, from the possibly oauth-merged new_data - not
            # the earlier parsed_new - before touching the registry. _add_proxy_model's own guard
            # ("already exists") would always reject the id here since it's unchanged, and previously
            # the old entry was deleted first just to dodge that guard - so a failure re-parsing/
            # validating new_data left the server deleted from the registry (and so from the endpoint
            # list) until a process restart. Same build-then-swap ordering the "user"/image branches
            # below already use.
            new_srv_model = self._build_proxy_model(try_parse_pydantic(SrvMcpProxyModel, new_data), new_prefix, custom_id=model.id)
            if instance in self.models and parsed_old.id in self.models[instance]:
                del self.models[instance][parsed_old.id]
            if instance not in self.models:
                self.models[instance] = {}
            self.models[instance][parsed_new.id] = new_srv_model
            return
        if kind == "user":
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
            # Id is immutable here, so the new Dockerfile lands at the same path as the old one -
            # back up its content so a failed remove_image below can be undone at the file level too,
            # not just in the in-memory registry. Without this, a later rollback-rebuild (edit_model's
            # except branch, which reinstalls the OLD model_id from whatever's on disk) would silently
            # build the NEW, partially-applied command/variant/base_image under the OLD model's identity.
            dockerfile_path = self._get_dockerfile_dir(instance, parsed_old.id) / "Dockerfile"
            old_dockerfile_content = dockerfile_path.read_text(encoding="utf-8") if dockerfile_path.exists() else None

            # Build the new model first (writes Dockerfile). If this fails the old model is preserved.
            new_srv_model = self._build_user_model(instance, parsed_new, custom_id=model.id)

            try:
                await self.docker_service.remove_image(old_tag)
            except Exception:
                if old_dockerfile_content is not None:
                    dockerfile_path.write_text(old_dockerfile_content, encoding="utf-8")
                raise
            if instance in self.models and parsed_old.id in self.models[instance]:
                del self.models[instance][parsed_old.id]
            if instance not in self.models:
                self.models[instance] = {}
            self.models[instance][parsed_new.id] = new_srv_model
            return

        # Plain docker-image MCP server (the default/`None` kind). Unlike `user`, there's no locally
        # built image to rebuild - `image` is an external reference - so this is a re-register, same
        # shape as `_add_image_model`, not a Dockerfile rebuild.
        parsed_old_image = try_parse_pydantic(SrvMcpCustomModel, model.data)
        parsed_new_image = try_parse_pydantic(SrvMcpCustomModel, new_data)

        if parsed_new_image.id != parsed_old_image.id:
            raise HTTPException(400, "Cannot change the server ID.")

        installed = self.get_instance_info(instance).installed
        if installed and parsed_old_image.id in installed.models:
            raise HTTPException(400, "Cannot update an installed server. Uninstall it first.")

        new_prefix = parsed_new_image.default_prefix or normalize_name(parsed_new_image.id)
        self._check_prefix_collision(instance, new_prefix, exclude_model_id=parsed_old_image.id)

        # Build the new registration first (validates the image/etc.) - if this fails, the old model
        # is preserved, same ordering as the "user" kind branch above. No rollback needed since the
        # old registration is only ever replaced once the new one is already known-good.
        new_srv_model = self._build_image_model(instance, parsed_new_image, custom_id=model.id)
        if instance in self.models and parsed_old_image.id in self.models[instance]:
            del self.models[instance][parsed_old_image.id]
        if instance not in self.models:
            self.models[instance] = {}
        self.models[instance][parsed_new_image.id] = new_srv_model

    def _validate_edit(self, instance: str, model_id: str) -> None:
        model = self.models.get(instance, {}).get(model_id)
        if model and model.kind == "proxy" and self._oauth_state_store.find_for_model(instance, model_id):
            raise HTTPException(400, "Cannot edit this MCP server while an OAuth authorization is pending.")

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
            if self._oauth_state_store.find_for_model(instance, parsed_proxy.id):
                raise HTTPException(400, "Cannot remove this MCP server while an OAuth authorization is pending.")
            if instance in self.models and parsed_proxy.id in self.models[instance]:
                del self.models[instance][parsed_proxy.id]
            self._clear_oauth_refresh_state(instance, parsed_proxy.id)
        else:
            parsed_custom = try_parse_pydantic(SrvMcpCustomModel, model.data)
            if installed and parsed_custom.id in installed.models:
                raise HTTPException(400, "Cannot remove custom model, it is in use, uninstall it first.")
            if instance in self.models and parsed_custom.id in self.models[instance]:
                del self.models[instance][parsed_custom.id]

    def _get_custom_spec(self, model_id: str, model: SrvMcpModel) -> dict[str, Any] | None:  # noqa: C901
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
            if model.oauth:
                # Client secret / access / refresh tokens are never sent to the WebUI — only enough
                # to prefill an edit form and show a status badge. See `McpOAuthConfig`.
                spec["oauth"] = {
                    "enabled": model.oauth.enabled,
                    "client_id": model.oauth.client_id,
                    "scope": model.oauth.scope,
                }
            if model.description:
                spec["description"] = model.description
            if model.repository_url:
                spec["repository_url"] = model.repository_url
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
            if model.description:
                spec["description"] = model.description
            if model.repository_url:
                spec["repository_url"] = model.repository_url
            return spec
        return None

    async def get_duplicate_spec(self, instance: str, model_id: str) -> dict[str, Any]:
        """Return a full add-model spec for duplicating `model_id`, custom-backed or catalog."""
        model = self.models.get(instance, {}).get(model_id)
        if model is None:
            raise HTTPException(400, "Model not found")
        if model.custom:
            definition = self.get_custom_model_definition(model.custom)
            if definition is None:
                raise HTTPException(404, f"Custom model definition for {model_id} not found.")
            definition = dict(definition)
            oauth = definition.get("oauth")
            if isinstance(oauth, dict):
                # The WebUI never round-trips secrets/tokens (see `_get_custom_spec`) - a duplicate
                # is an independent model with its own OAuth flow, so it must not inherit the
                # original's live client_secret/access_token/refresh_token either.
                definition["oauth"] = {k: oauth[k] for k in ("enabled", "client_id", "scope") if k in oauth}
            return definition
        # Catalog model: no stored definition exists (it's hardcoded in `_const.models`), so
        # synthesize one from the live registered model - the same shape `_add_image_model` expects.
        if model.options is None:
            raise HTTPException(400, "This model cannot be duplicated.")
        command = model.options.command if isinstance(model.options.command, str) else None
        healthcheck = model.options.healthcheck or {}
        # `ModelInstalledInfo.envs`/`.headers` hold only what the admin explicitly supplied at
        # install time (e.g. an API key) - they don't include the catalog's own static defaults
        # (those are merged into the actual container's env vars, but never written back here). So
        # merge both when installed, catalog defaults as the base, install-time values overriding -
        # using just the installed values alone would silently drop the catalog's own defaults.
        installed = self.get_instance_info(instance).installed
        installed_model = installed.models.get(model_id) if installed else None
        envs = {**(model.options.env_vars or {}), **(installed_model.envs or {})} if installed_model else model.options.env_vars
        headers = {**(model.headers or {}), **(installed_model.headers or {})} if installed_model else model.headers
        image_version = (installed_model.options.spec or {}).get("image_version") if installed_model else None
        docker_options = apply_image_version_override(model.options, image_version)
        return {
            "id": model_id,
            "private": True,
            "default_prefix": model.default_prefix,
            "size": model.size,
            "image": docker_options.image,
            "image_port": model.options.image_port,
            "command": command,
            # See the equivalent CustomService.get_duplicate_spec for why the host side is stripped:
            # model.options.volumes is already a fully-expanded host:container bind mount, but
            # _add_image_model re-prefixes whatever it receives assuming a bare container path.
            "volumes": [v.split(":", 1)[1] if ":" in v else v for v in (model.options.volumes or [])],
            "envs": dict(envs) if envs else None,
            "headers": dict(headers) if headers else None,
            "healthcheck_cmd": healthcheck.get("test"),
            "healthcheck_start_period": healthcheck.get("start_period"),
            "required_envs": dict.fromkeys(model.required_envs, "") if model.required_envs else None,
            "required_headers": dict.fromkeys(model.required_headers, "") if model.required_headers else None,
            "description": model.description,
            "repository_url": model.repository_url,
        }

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
                            default_prefix=model.default_prefix,
                            effective_prefix=(info.models[model_id].prefix if model_id in info.models else model.default_prefix),
                            spec=model.model_spec,
                            has_docker=model.kind != "proxy",
                            variant=model.variant,
                            command=model.command,
                            base_image=model.base_image,
                            custom_spec=self._get_custom_spec(model_id, model),
                            description=model.description or None,
                            repository_url=model.repository_url,
                        )
                    )

        return ListModelsOut(list=out_list)

    def _get_model_install_fields(self, instance: str, model_id: str) -> Sequence[ModelField]:
        """Return the install spec fields of a model, or none if the model is unknown."""
        model = self.models.get(instance, {}).get(model_id)
        return model.model_spec.fields if model else []

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
            default_prefix=model.default_prefix,
            effective_prefix=(info.models[model_id].prefix if model_id in info.models else model.default_prefix),
            spec=model.model_spec,
            has_docker=model.kind != "proxy",
            variant=model.variant,
            command=model.command,
            base_image=model.base_image,
            custom_spec=self._get_custom_spec(model_id, model),
            description=model.description or None,
            repository_url=model.repository_url,
        )

    async def _fetch_tools_background(self, instance: str, model_id: str) -> None:
        """Background task: probe MCP server with retries until tools are fetched successfully."""
        for delay in [3.0, 8.0, 20.0, 40.0, 60.0]:
            await asyncio.sleep(delay)
            try:
                result = await self.healthcheck_model(instance, model_id)
                # Once OAuth is required, further retries can't succeed until the admin
                # authorizes — that completion path (`complete_oauth_callback`) restarts this.
                if result.healthy or result.requires_oauth:
                    return
            except Exception:
                logger.exception(f"{self.get_id(instance)} background tools fetch for {model_id!r} failed")

    def _apply_healthcheck_result(self, model: SrvMcpModel, prefix: str, result: McpHealthCheckResult) -> None:
        """Persist tool list, transport, and healthy flag from a successful health check.

        Notifies the parent infra when anything a mesh parent mirrors (tools, transport, health) changed.
        """
        changed = False
        # Registrations may share `model.model_props`, so compare against them before it is overwritten below.
        for reg_model in self.endpoint_registry.mcp_endpoints.models.get(prefix, {}).values():
            changed = changed or not reg_model.healthy or reg_model.props.tools != result.tools
            reg_model.healthy = True
            reg_model.props.tools = result.tools
            if result.transport:
                changed = changed or reg_model.props.transport != result.transport
                reg_model.props.transport = result.transport
        model.model_props.tools = result.tools
        if result.transport:
            model.model_props.transport = result.transport
        if changed:
            self.endpoint_registry.parent_infra.send_models_list()

    def _mark_unhealthy(self, prefix: str) -> None:
        """Hide an MCP whose server stopped answering, keeping its last known tools, and tell the mesh parent."""
        changed = False
        for reg_model in self.endpoint_registry.mcp_endpoints.models.get(prefix, {}).values():
            if reg_model.healthy:
                reg_model.healthy = False
                changed = True
        if changed:
            self.endpoint_registry.parent_infra.send_models_list()

    async def _health_monitor_background(self, instance: str, model_id: str) -> None:
        """Re-probe an installed MCP periodically so a server that dies stops being reported as healthy.

        Fail-safe: only a probe that completes and reports the server unreachable counts as a failure. A crash of
        the check itself (or a probe needing OAuth) leaves the current state untouched, and the loop keeps running.
        """
        consecutive_failures = 0
        while True:
            await asyncio.sleep(_HEALTH_MONITOR_INTERVAL_SECONDS)
            instance_info = self.instances_info.get(instance)
            info = instance_info.installed if instance_info else None
            if info is None or model_id not in info.models:
                return
            prefix = info.models[model_id].prefix
            try:
                result = await self.healthcheck_model(instance, model_id)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("MCP health monitor probe crashed for %s/%s; keeping current state", instance, model_id)
                continue
            if result.healthy or result.requires_oauth:
                consecutive_failures = 0
                continue
            consecutive_failures += 1
            if consecutive_failures >= _HEALTH_MONITOR_FAILURE_THRESHOLD:
                logger.warning(
                    "MCP %s/%s failed %d consecutive health checks (%s); marking unhealthy",
                    instance,
                    model_id,
                    consecutive_failures,
                    result.error,
                )
                try:
                    self._mark_unhealthy(prefix)
                except Exception:
                    logger.exception("Failed to mark MCP %s/%s unhealthy", instance, model_id)

    def _start_health_monitor(self, instance: str, model_id: str) -> None:
        key = (instance, model_id)
        existing = self._health_monitor_tasks.get(key)
        if existing is not None and not existing.done():
            return
        task = asyncio.create_task(self._health_monitor_background(instance, model_id))
        self._health_monitor_tasks[key] = task
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        task.add_done_callback(lambda t: self._health_monitor_tasks.pop(key, None) if self._health_monitor_tasks.get(key) is t else None)

    def _stop_health_monitor(self, instance: str, model_id: str) -> None:
        task = self._health_monitor_tasks.pop((instance, model_id), None)
        if task is not None:
            task.cancel()

    async def healthcheck_model(self, instance: str, model_id: str) -> McpHealthCheckResult:  # noqa: C901
        """Probe an installed MCP server: check liveness, detect transport, and fetch tool list."""
        start_time = time.monotonic()
        base_attrs = {"mcp.instance": instance, "mcp.model_id": model_id}
        outcome = "error"
        with tracer.span("mcp.healthcheck", **base_attrs) as span:
            try:
                try:
                    info = self.get_instance_installed_info(instance)
                except HTTPException:
                    # Instance missing/not installed (404/400) is bad input, not an operational
                    # failure — same reasoning as the invalid_request cases below: keep it out of
                    # "error" so it doesn't skew error-rate/latency alerting for genuine MCP server
                    # problems (e.g. an admin uninstalling the model mid-retry-loop is a benign race).
                    outcome = "invalid_request"
                    raise
                if model_id not in info.models:
                    # Bad input (unknown model), not an operational failure — keep it out of "error"
                    # so it doesn't pollute error-rate/latency alerting for genuine MCP server problems.
                    outcome = "invalid_request"
                    raise HTTPException(status_code=400, detail="Model is not installed")
                model_info = info.models[model_id]
                model = (self.models.get(instance) or {}).get(model_id)
                if model is None:
                    outcome = "invalid_request"
                    raise HTTPException(status_code=400, detail="Model not found in registry")

                base_url = model_info.base_url
                headers = model_info.headers or {}
                transport = model.proxy_transport  # "streamable_http" | "sse"

                if model.kind == "proxy":
                    if model.oauth and model.oauth.enabled and has_valid_access_token(model.oauth):
                        headers = {**headers, "Authorization": f"Bearer {model.oauth.access_token}"}
                    # proxy_url is already the full endpoint URL (may end with /mcp or /sse)
                    if transport == "sse":
                        result = await _fetch_tools_from_sse_endpoint(base_url, headers)
                    else:
                        result = await _fetch_tools_from_mcp_endpoint(base_url, headers)
                    if result.requires_oauth and not (model.oauth and model.oauth.enabled):
                        await self._auto_detect_oauth(instance, model_id)
                else:
                    # Docker / user model — base_url is http://host:port, transport path appended separately
                    if transport == "sse":
                        result = await _fetch_tools_from_sse_endpoint(f"{base_url}/sse", headers)
                        if not result.healthy:
                            fallback = await _fetch_tools_from_mcp_endpoint(f"{base_url}/mcp", headers)
                            result = fallback if fallback.healthy else result
                    else:
                        result = await _fetch_tools_from_mcp_endpoint(f"{base_url}/mcp", headers)
                        if not result.healthy:
                            fallback = await _fetch_tools_from_sse_endpoint(f"{base_url}/sse", headers)
                            result = fallback if fallback.healthy else result

                    if result.healthy and not result.error and result.transport and result.transport != transport:
                        # The live proxy registered at install time still targets `transport`, but only
                        # the other one actually answers — re-register it so real traffic (not just this
                        # probe) stops hitting the broken endpoint. `healthy=True` alone isn't enough to
                        # act on: it only means the transport connected and `initialize` completed, while
                        # a later step (waiting for the SSE endpoint event, the MCP response, or
                        # tools/list itself) can still time out or error — reported as `healthy=True`
                        # with `error` set and no tools. Re-registering on that would permanently point
                        # the live proxy at a transport that never actually returned tools either, so
                        # `error` must be unset too for this to count as a real, complete success.
                        merged_headers = model.headers | headers if model.headers else headers
                        try:
                            new_registration_id = self._register_docker_mcp_proxy(model, model_info, result.transport, merged_headers)
                        except Exception:
                            logger.exception(
                                "Failed to register MCP proxy for prefix %s on transport %s; keeping the existing registration %s",
                                model_info.prefix,
                                result.transport,
                                model_info.registration_id,
                            )
                            raise
                        old_registration_id = model_info.registration_id
                        # Update bookkeeping before unregistering the old entry: `unregister_mcp_endpoint`
                        # can still raise after removing it (e.g. the parent-infra notification fails), and
                        # we'd rather keep pointing at the new, working registration than leave `model_info`
                        # referencing an id that's already gone from the registry.
                        model_info.registration_id = new_registration_id
                        model.proxy_transport = result.transport
                        try:
                            self.endpoint_registry.unregister_mcp_endpoint(model_info.prefix, old_registration_id)
                        except Exception:
                            logger.exception(
                                "Failed to unregister stale MCP proxy registration %s for prefix %s"
                                " after re-registering it on transport %s",
                                old_registration_id,
                                model_info.prefix,
                                result.transport,
                            )

                if result.healthy:
                    self._apply_healthcheck_result(model, model_info.prefix, result)

                outcome = "healthy" if result.healthy else ("oauth_required" if result.requires_oauth else "unhealthy")
                if span is not None:
                    span.set_attribute("mcp.transport", result.transport or "<unknown>")
                    span.set_attribute("mcp.tool_count", len(result.tools))

                return result
            finally:
                # In a `finally`, not just the success path — an attempt that raised (e.g. the
                # `HTTPException`s above) is still an attempt and should show up in the metric,
                # not silently vanish from it. `outcome` stays "error" for any of those cases.
                if span is not None:
                    span.set_attribute("mcp.outcome", outcome)
                attrs = {**base_attrs, "mcp.outcome": outcome}
                if instruments := tracer.mcp_instruments():
                    instruments.healthcheck_count.add(1, attrs)
                    # `invalid_request` is bad input rejected before any probe ran — its ~0ms
                    # duration would skew p50/p95 and mask genuinely slow MCP servers.
                    if outcome != "invalid_request":
                        instruments.healthcheck_duration_ms.record((time.monotonic() - start_time) * 1000, attrs)

    def check_envs(self, required_envs: list[str] | None, envs: dict[str, str]) -> None:
        """Check enviromental variables."""
        if not required_envs:
            return

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
        if not required_headers:
            return

        missing_keys = [key for key in required_headers if key not in headers]
        if missing_keys:
            raise HTTPException(status_code=422, detail=f"The following required headers are missing: {', '.join(missing_keys)}")

        empty_keys = [key for key in required_headers if not headers.get(key)]
        if empty_keys:
            raise HTTPException(status_code=422, detail=f"The following headers are present but have no value: {', '.join(empty_keys)}")

    def _register_proxy_model(self, model: SrvMcpModel, parsed_options: McpModelOptions, instance: str, model_id: str) -> RegistrationId:
        """Register a proxy model endpoint, choosing SSE or Streamable HTTP transport."""
        if model.proxy_url is None:
            raise HTTPException(400, "proxy_url is required for proxy models")
        merged_headers = {**(model.headers or {}), **parsed_options.headers}
        dynamic_headers = None
        on_reauth = None
        if model.oauth and model.oauth.enabled:
            dynamic_headers = self._make_oauth_header_provider(instance, model_id)
            on_reauth = self._make_oauth_refresh_callback(instance, model_id)
        proxy_options = ProxyOptions(
            url=model.proxy_url,
            headers=merged_headers if merged_headers else None,
            allowed_request_headers=["accept", "mcp-session-id"],
            allowed_response_headers=["accept", "mcp-session-id", "www-authenticate"],
            dynamic_headers=dynamic_headers,
            on_reauth=on_reauth,
        )
        registration_options = RegistrationOptions(origin="local", owned_by=self.get_type())
        if model.proxy_transport == "sse":
            return self.endpoint_registry.register_mcp_sse_endpoint_as_proxy(
                url=parsed_options.prefix, props=model.model_props, options=proxy_options, registration_options=registration_options
            )
        return self.endpoint_registry.register_mcp_endpoint_as_proxy(
            url=parsed_options.prefix, props=model.model_props, options=proxy_options, registration_options=registration_options
        )

    def _register_docker_mcp_proxy(
        self, model: SrvMcpModel, model_info: ModelInstalledInfo, transport: str, headers: dict[str, str]
    ) -> RegistrationId:
        """Register (or re-register) the live MCP proxy for a docker-installed model at the given transport."""
        registration_options = RegistrationOptions(origin="local", owned_by=self.get_type())
        if transport == "sse":
            return self.endpoint_registry.register_mcp_sse_endpoint_as_proxy(
                url=model_info.prefix,
                props=model.model_props,
                options=ProxyOptions(
                    url=model_info.base_url + "/sse",
                    allowed_request_headers=["accept", "mcp-session-id"],
                    allowed_response_headers=["accept", "mcp-session-id"],
                    headers=headers,
                ),
                registration_options=registration_options,
            )
        return self.endpoint_registry.register_mcp_endpoint_as_proxy(
            url=model_info.prefix,
            props=model.model_props,
            options=ProxyOptions(
                url=model_info.base_url + "/mcp",
                allowed_request_headers=["accept", "mcp-session-id"],
                allowed_response_headers=["accept", "mcp-session-id"],
                headers=headers,
            ),
            registration_options=registration_options,
        )

    def _get_oauth_lock(self, instance: str, model_id: str) -> asyncio.Lock:
        key = (instance, model_id)
        if key not in self._oauth_refresh_locks:
            self._oauth_refresh_locks[key] = asyncio.Lock()
        return self._oauth_refresh_locks[key]

    async def _refresh_oauth_token(self, instance: str, model_id: str) -> McpOAuthConfig | None:  # noqa: C901
        """Refresh the access token for a proxy model's OAuth config, persisting the result."""
        async with self._get_oauth_lock(instance, model_id):
            model = (self.models.get(instance) or {}).get(model_id)
            if model is None or model.oauth is None or not model.oauth.refresh_token or not model.oauth.token_endpoint:
                return model.oauth if model else None
            if has_valid_access_token(model.oauth):
                return model.oauth  # a concurrent caller already refreshed it while we waited for the lock
            if not model.oauth.client_id:
                return model.oauth
            attrs = {"mcp.instance": instance, "mcp.model_id": model_id}
            try:
                with tracer.span("mcp.oauth.refresh", **attrs) as span:
                    try:
                        token = await refresh_access_token(
                            model.oauth.token_endpoint,
                            model.oauth.refresh_token,
                            model.oauth.client_id,
                            model.oauth.client_secret,
                            model.oauth.resource or model.proxy_url or "",
                        )
                    except McpOAuthError:
                        if span is not None:
                            span.set_attribute("mcp.outcome", "failure")
                        raise
                    if span is not None:
                        span.set_attribute("mcp.outcome", "success")
            except McpOAuthError as exc:
                if instruments := tracer.mcp_instruments():
                    instruments.oauth_refresh_count.add(1, {**attrs, "mcp.outcome": "failure"})
                model.oauth.last_error = str(exc)
                await self._persist_proxy_oauth(instance, model_id, model.oauth)
                return None
            if instruments := tracer.mcp_instruments():
                instruments.oauth_refresh_count.add(1, {**attrs, "mcp.outcome": "success"})
            model.oauth.access_token = token.access_token
            if token.refresh_token:
                model.oauth.refresh_token = token.refresh_token
            model.oauth.token_expires_at = time.time() + token.expires_in if token.expires_in else None
            model.oauth.last_error = None
            await self._persist_proxy_oauth(instance, model_id, model.oauth)
            return model.oauth

    def _make_oauth_header_provider(self, instance: str, model_id: str) -> Callable[[], Awaitable[dict[str, str]]]:
        """Build a `ProxyOptions.dynamic_headers` provider reading live token state off `self.models`."""

        async def provider() -> dict[str, str]:
            model = (self.models.get(instance) or {}).get(model_id)
            oauth = model.oauth if model else None
            if oauth is None or not oauth.access_token:
                return {}
            if not has_valid_access_token(oauth) and oauth.refresh_token:
                oauth = await self._refresh_oauth_token(instance, model_id) or oauth
            return {"Authorization": f"Bearer {oauth.access_token}"} if oauth.access_token else {}

        return provider

    def _make_oauth_refresh_callback(self, instance: str, model_id: str) -> Callable[[], Awaitable[dict[str, str] | None]]:
        """Build a `ProxyOptions.on_reauth` callback for the reactive 401-retry path."""

        async def on_reauth() -> dict[str, str] | None:
            oauth = await self._refresh_oauth_token(instance, model_id)
            if oauth and oauth.access_token:
                return {"Authorization": f"Bearer {oauth.access_token}"}
            return None

        return on_reauth

    _OAUTH_REFRESH_LEAD_SECONDS = 60
    _OAUTH_REFRESH_MIN_BACKOFF_SECONDS = 5.0
    _OAUTH_REFRESH_MAX_BACKOFF_SECONDS = 3600.0
    _OAUTH_REFRESH_MAX_CONSECUTIVE_FAILURES = 8

    async def _oauth_refresh_background(self, instance: str, model_id: str) -> None:
        """Background task: proactively refresh an OAuth access token shortly before it expires.

        Supplements (does not replace) the reactive 401-retry path wired through `on_reauth` — a 401
        can still happen regardless (revocation, clock skew), which that path handles independently.

        A refresh failure (e.g. a revoked refresh token) backs off exponentially instead of retrying
        every few seconds forever, and this task gives up after too many consecutive failures — the
        reactive path and `last_error`/status badge remain available for the admin to notice and fix.
        """
        consecutive_failures = 0
        while True:
            model = (self.models.get(instance) or {}).get(model_id)
            oauth = model.oauth if model else None
            if oauth is None or not oauth.enabled or not oauth.refresh_token or oauth.token_expires_at is None:
                return
            if consecutive_failures:
                sleep_for = min(
                    self._OAUTH_REFRESH_MIN_BACKOFF_SECONDS * (2**consecutive_failures),
                    self._OAUTH_REFRESH_MAX_BACKOFF_SECONDS,
                )
            else:
                sleep_for = max(
                    oauth.token_expires_at - time.time() - self._OAUTH_REFRESH_LEAD_SECONDS,
                    self._OAUTH_REFRESH_MIN_BACKOFF_SECONDS,
                )
            await asyncio.sleep(sleep_for)
            still_live = (self.models.get(instance) or {}).get(model_id)
            if still_live is None or still_live.oauth is None or not still_live.oauth.refresh_token:
                return
            try:
                refreshed = await self._refresh_oauth_token(instance, model_id)
            except Exception:
                logger.exception("Proactive OAuth refresh failed for %s/%s", instance, model_id)
                refreshed = None
            if refreshed is not None and has_valid_access_token(refreshed):
                consecutive_failures = 0
                continue
            consecutive_failures += 1
            if consecutive_failures >= self._OAUTH_REFRESH_MAX_CONSECUTIVE_FAILURES:
                logger.warning(
                    "Giving up on proactive OAuth refresh for %s/%s after %d consecutive failures",
                    instance,
                    model_id,
                    consecutive_failures,
                )
                return

    def _start_oauth_refresh_background(self, instance: str, model_id: str) -> None:
        key = (instance, model_id)
        existing = self._oauth_refresh_tasks.get(key)
        if existing is not None and not existing.done():
            return
        task = asyncio.create_task(self._oauth_refresh_background(instance, model_id))
        self._oauth_refresh_tasks[key] = task
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        task.add_done_callback(lambda t: self._oauth_refresh_tasks.pop(key, None) if self._oauth_refresh_tasks.get(key) is t else None)

    def _clear_oauth_refresh_state(self, instance: str, model_id: str) -> None:
        """Drop the refresh lock and cancel any running proactive-refresh task for a removed proxy model."""
        self._oauth_refresh_locks.pop((instance, model_id), None)
        refresh_task = self._oauth_refresh_tasks.pop((instance, model_id), None)
        if refresh_task is not None:
            refresh_task.cancel()

    def _get_oauth_model(self, instance: str, model_id: str) -> SrvMcpModel:
        model = (self.models.get(instance) or {}).get(model_id)
        if model is None or model.kind != "proxy":
            raise HTTPException(404, "Proxy MCP model not found")
        return model

    async def _discover_oauth_endpoints(self, instance: str, model_id: str, model: SrvMcpModel) -> None:
        """Discover and cache the AS endpoints protecting a proxy model's remote MCP server."""
        assert model.oauth is not None
        assert model.proxy_url is not None
        metadata = await discover_authorization_server_for_resource(model.proxy_url)
        if metadata is None:
            return
        model.oauth.authorization_endpoint = metadata.authorization_endpoint
        model.oauth.token_endpoint = metadata.token_endpoint
        model.oauth.registration_endpoint = metadata.registration_endpoint
        await self._persist_proxy_oauth(instance, model_id, model.oauth)

    async def _auto_detect_oauth(self, instance: str, model_id: str) -> None:
        """Auto-enable OAuth for a proxy model after a health check surfaced a 401.

        The admin never has to pre-declare that a remote MCP server needs OAuth: this runs the
        same RFC 9728/8414 discovery as `_discover_oauth_endpoints`, but also flips `oauth.enabled`
        on (creating the config if none existed) and re-registers the live endpoint so it's ready
        to attach a token as soon as the admin authorizes. If discovery itself fails, OAuth is still
        marked required so the WebUI can prompt for a manually-configured client_id.
        """
        model = (self.models.get(instance) or {}).get(model_id)
        if model is None or model.kind != "proxy" or model.proxy_url is None or (model.oauth and model.oauth.enabled):
            return
        async with self._get_oauth_lock(instance, model_id):
            # Re-check after acquiring the lock: a concurrent healthcheck may have already run this.
            model = (self.models.get(instance) or {}).get(model_id)
            if model is None or model.kind != "proxy" or model.proxy_url is None or (model.oauth and model.oauth.enabled):
                return
            metadata = await discover_authorization_server_for_resource(model.proxy_url)
            oauth = model.oauth or McpOAuthConfig()
            oauth.enabled = True
            oauth.resource = oauth.resource or model.proxy_url
            if metadata:
                oauth.authorization_endpoint = metadata.authorization_endpoint
                oauth.token_endpoint = metadata.token_endpoint
                oauth.registration_endpoint = metadata.registration_endpoint
                oauth.last_error = None
            else:
                oauth.last_error = (
                    "This server responded with 401 Unauthorized, but its OAuth authorization server could not be "
                    "auto-discovered. Edit this server to provide a client_id manually."
                )
            model.oauth = oauth
            logger.info("Auto-detected OAuth requirement for MCP proxy model %s/%s", instance, model_id)
            await self._persist_proxy_oauth(instance, model_id, oauth)
            await self._reregister_installed_proxy(instance, model_id)

    async def _reregister_installed_proxy(self, instance: str, model_id: str) -> None:
        """Re-register a proxy model's endpoint so a live install picks up freshly authorized tokens."""
        installed = self.instances_info.get(instance)
        installed_info = installed.installed if installed else None
        if installed_info is None:
            return
        model_info = installed_info.models.get(model_id)
        model = (self.models.get(instance) or {}).get(model_id)
        if model_info is None or model is None:
            return
        self.endpoint_registry.unregister_mcp_endpoint(model_info.prefix, model_info.registration_id)
        parsed_options = McpModelOptions(prefix=model_info.prefix, headers=model_info.headers)
        model_info.registration_id = self._register_proxy_model(model, parsed_options, instance, model_id)

    def _build_oauth_redirect_uri(self) -> str:
        """Build the OAuth callback redirect_uri from `infra_url`, preserving any reverse-proxy path prefix."""
        parsed = urlparse(self.config.infra_url)
        if not parsed.scheme or not parsed.netloc:
            raise HTTPException(
                400,
                "DF_INFRA_URL is not configured. Set the infra URL (Settings) to this server's externally-reachable "
                "address before starting an OAuth flow — it's needed to build the OAuth callback redirect_uri.",
            )
        return Utils.join_url(self.config.infra_url, "mcp-oauth/callback")

    async def start_oauth_flow(self, instance: str, model_id: str) -> str:
        """Begin an OAuth authorization-code flow for a proxy model; returns the authorize URL to open."""
        model = self._get_oauth_model(instance, model_id)
        if model.oauth is None or not model.oauth.enabled:
            raise HTTPException(400, "OAuth is not enabled for this model")
        assert model.proxy_url is not None

        if not model.oauth.authorization_endpoint or not model.oauth.token_endpoint:
            await self._discover_oauth_endpoints(instance, model_id, model)
            model = self._get_oauth_model(instance, model_id)

        assert model.oauth is not None
        if not model.oauth.authorization_endpoint or not model.oauth.token_endpoint:
            raise HTTPException(400, "Could not discover an OAuth authorization server for this MCP server.")

        redirect_uri = self._build_oauth_redirect_uri()

        if not model.oauth.client_id:
            if not model.oauth.registration_endpoint:
                raise HTTPException(
                    400,
                    "No client_id configured and this server does not support dynamic client registration. Provide a client_id manually.",
                )
            dcr = await register_dynamic_client(model.oauth.registration_endpoint, redirect_uri, "DeepFellow Infra")
            if dcr is None:
                raise HTTPException(400, "No client_id configured and dynamic client registration failed. Provide a client_id manually.")
            model.oauth.client_id = dcr.client_id
            model.oauth.client_secret = dcr.client_secret
            await self._persist_proxy_oauth(instance, model_id, model.oauth)
            model = self._get_oauth_model(instance, model_id)

        oauth = model.oauth
        assert oauth is not None
        assert oauth.client_id is not None
        assert oauth.authorization_endpoint is not None
        assert model.proxy_url is not None
        code_verifier, code_challenge = generate_pkce_pair()
        state = secrets.token_urlsafe(32)
        resource = oauth.resource or model.proxy_url
        self._oauth_state_store.add(
            state,
            PendingOAuthFlow(
                instance=instance, model_id=model_id, code_verifier=code_verifier, redirect_uri=redirect_uri, resource=resource
            ),
        )
        return build_authorize_url(
            oauth.authorization_endpoint,
            oauth.client_id,
            redirect_uri,
            state,
            code_challenge,
            resource,
            oauth.scope,
        )

    async def complete_oauth_callback(self, state: str, code: str | None, error: str | None) -> str:  # noqa: C901
        """Handle the OAuth redirect callback. Returns a short human-readable status message."""
        flow = self._oauth_state_store.pop(state)
        if flow is None:
            return "This authorization link is invalid or has expired. Please try again from DeepFellow."

        model = (self.models.get(flow.instance) or {}).get(flow.model_id)
        if model is None or model.oauth is None or not model.oauth.enabled:
            return "This MCP server no longer exists or OAuth was disabled. You can close this tab."

        if error:
            model.oauth.last_error = error
            await self._persist_proxy_oauth(flow.instance, flow.model_id, model.oauth)
            return f"Authorization failed: {error}. You can close this tab and try again."

        if code is None:
            return "No authorization code was returned. You can close this tab and try again."

        if not model.oauth.authorization_endpoint or not model.oauth.token_endpoint:
            await self._discover_oauth_endpoints(flow.instance, flow.model_id, model)
            model = (self.models.get(flow.instance) or {}).get(flow.model_id)
            if model is None or model.oauth is None or not model.oauth.token_endpoint:
                return "Could not discover this server's OAuth token endpoint. You can close this tab and try again."

        if not model.oauth.client_id:
            if model.oauth.registration_endpoint:
                dcr = await register_dynamic_client(model.oauth.registration_endpoint, flow.redirect_uri, "DeepFellow Infra")
                if dcr:
                    model.oauth.client_id = dcr.client_id
                    model.oauth.client_secret = dcr.client_secret
                    await self._persist_proxy_oauth(flow.instance, flow.model_id, model.oauth)
            if not model.oauth.client_id:
                return "This MCP server's OAuth client is no longer configured. You can close this tab and try again."

        try:
            token = await exchange_code_for_token(
                model.oauth.token_endpoint,
                code,
                flow.redirect_uri,
                model.oauth.client_id,
                model.oauth.client_secret,
                flow.code_verifier,
                flow.resource,
            )
        except McpOAuthError as exc:
            if has_valid_access_token(model.oauth):
                # `state` is single-use (popped above), so a resubmission of this exact callback URL
                # would already have hit the "invalid or expired" branch, not this one. Getting here
                # with a valid token already in place means a *different* flow for this model (e.g. two
                # "Authorize" attempts started back-to-back) finished first — report success instead of
                # surfacing a confusing error for something that already worked.
                return "Already authorized. You can close this tab."
            model.oauth.last_error = str(exc)
            await self._persist_proxy_oauth(flow.instance, flow.model_id, model.oauth)
            return f"Authorization failed: {exc}. You can close this tab and try again."

        model.oauth.access_token = token.access_token
        if token.refresh_token:
            model.oauth.refresh_token = token.refresh_token
        model.oauth.token_expires_at = time.time() + token.expires_in if token.expires_in else None
        model.oauth.last_error = None
        await self._persist_proxy_oauth(flow.instance, flow.model_id, model.oauth)

        await self._reregister_installed_proxy(flow.instance, flow.model_id)
        if model.oauth.refresh_token:
            self._start_oauth_refresh_background(flow.instance, flow.model_id)

        installed = self.instances_info.get(flow.instance)
        if installed and installed.installed and flow.model_id in installed.installed.models:
            # Refresh the tool list now that the model can authenticate, instead of waiting on
            # the admin to click "Test" — this mirrors the retry task kicked off at install time.
            task = asyncio.create_task(self._fetch_tools_background(flow.instance, flow.model_id))
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

        return "Authorization complete. You can close this tab."

    def get_oauth_status(self, instance: str, model_id: str) -> McpOAuthStatusOut:
        """Return non-secret OAuth status for the WebUI: never a client_secret, access_token, or refresh_token."""
        model = self._get_oauth_model(instance, model_id)
        oauth = model.oauth
        if oauth is None or not oauth.enabled:
            return McpOAuthStatusOut(enabled=False, status="disabled")

        if self._oauth_state_store.find_for_model(instance, model_id):
            status: Literal["pending", "authorized", "expired", "error", "not_started"] = "pending"
        elif has_valid_access_token(oauth):
            status = "authorized"
        elif oauth.access_token:
            status = "expired"
        elif oauth.last_error:
            status = "error"
        else:
            status = "not_started"

        return McpOAuthStatusOut(
            enabled=True,
            status=status,
            has_client_id=bool(oauth.client_id),
            has_client_secret=bool(oauth.client_secret),
            expires_at=oauth.token_expires_at,
            last_error=oauth.last_error,
        )

    async def _install_model(  # noqa: C901
        self,
        instance: str,
        model_id: str,
        options: InstallModelIn,
        container_name_override: str | None = None,
    ) -> PromiseWithProgress[InstallModelOut, StreamChunk]:
        info = self.get_instance_installed_info(instance)
        self.models.setdefault(instance, {})
        key = (instance, model_id)
        if model_id in info.models or key in self._installing:
            return PromiseWithProgress(value=InstallModelOut(status="OK", details="Already installed or being installed right now."))
        if model_id not in self.models[instance]:
            raise HTTPException(400, "Model not found")
        self._installing.add(key)
        model = self.models[instance][model_id]
        if not options.spec:
            options.spec = {}
        if "prefix" not in options.spec:
            options.spec["prefix"] = model.default_prefix
        model.model_props.prefix = options.spec["prefix"]
        try:
            parsed_model_options = try_parse_pydantic(McpModelOptions, options.spec)

            self.check_envs(model.required_envs, parsed_model_options.envs)
            self.check_headers(model.required_headers, parsed_model_options.headers)

            if model.kind == "proxy":
                registration_id = self._register_proxy_model(model, parsed_model_options, instance, model_id)
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
                # Probe synchronously so an OAuth requirement (or any other startup error) is
                # reflected in this response, rather than only surfacing later via the background
                # retry loop — the admin shouldn't have to wait around and notice a badge appear.
                try:
                    probe = await self.healthcheck_model(instance, model_id)
                except Exception:
                    probe = None
                if probe is None or not (probe.healthy or probe.requires_oauth):
                    task = asyncio.create_task(self._fetch_tools_background(instance, model_id))
                    self._background_tasks.add(task)
                    task.add_done_callback(self._background_tasks.discard)
                if model.oauth and model.oauth.enabled and model.oauth.refresh_token:
                    self._start_oauth_refresh_background(instance, model_id)
                self._start_health_monitor(instance, model_id)
                self._installing.discard(key)
                requires_oauth = bool(probe and probe.requires_oauth)
                details = "Installed. OAuth authorization required — use the Authorize action." if requires_oauth else "Installed"
                return PromiseWithProgress(value=InstallModelOut(status="OK", details=details, requires_oauth=requires_oauth))

            if model.options is None:
                raise HTTPException(400, "options are required for this model kind.")  # noqa: TRY301
            docker_options = model.options
            if model.kind != "user":
                await self.validate_docker_image_version_for_model(
                    model_id, options.spec.get("image_version"), options.spec.get("hardware")
                )
                docker_options = apply_image_version_override(docker_options, options.spec.get("image_version"))
                await self._verify_docker_image(docker_options.image, options.ignore_warnings)
        except BaseException:
            self._installing.discard(key)
            raise

        async def func(stream: Stream[StreamChunk]) -> InstallModelOut:
            model_info: ModelInstalledInfo | None = None
            docker_options_edited: DockerOptions | None = None
            adopted = False
            try:
                model_dir = self._get_working_dir() / "models"
                model_dir.mkdir(parents=True, exist_ok=True)
                subnet = self.docker_service.get_docker_subnet()
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
                # Re-derived rather than taken from model.options, built with the suffix current when the model
                # list was loaded - a live container_name_suffix change must apply to new installs too.
                docker_options_edited.container_name = container_name_override or self.docker_service.get_docker_container_name(
                    docker_options_edited.name
                )
                container_host = self.docker_service.get_container_host(subnet, docker_options_edited)
                # {container_host} lets an image allow-list the Host header DF Infra sends it (e.g. the MCP SDK's
                # DNS-rebinding protection), which depends on the resolved - possibly suffixed - container name.
                docker_options_edited.env_vars = {
                    key: value.replace("{container_host}", container_host) for key, value in docker_options_edited.env_vars.items()
                } | parsed_model_options.envs
                docker_exposed_port, _, adopted = await self.docker_service.install_and_run_docker(docker_options_edited)
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
                merged_headers = model.headers | parsed_model_options.headers if model.headers else parsed_model_options.headers
                model_info.registration_id = self._register_docker_mcp_proxy(model, model_info, model.proxy_transport, merged_headers)
                self.models_downloaded[model_id] = DownloadedInfo(docker_options.image)
                task = asyncio.create_task(self._fetch_tools_background(instance, model_id))
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
                self._start_health_monitor(instance, model_id)
                stream.emit(StreamChunkProgress(type="progress", stage="install", value=1, data={}))
                return InstallModelOut(status="OK", details="Installed")
            except BaseException:
                # Each cleanup step is independently guarded so one raising (e.g. the endpoint unregister) can't
                # skip the rest - in particular, it must never skip the docker rollback below.
                if model_info is not None and model_info.registration_id:
                    try:
                        self.endpoint_registry.unregister_mcp_endpoint(model_info.prefix, model_info.registration_id)
                    except Exception:
                        logger.exception("Error unregistering MCP endpoint for %r during install rollback", model_id)
                if model_info is not None and info.models.get(model_id) is model_info:
                    info.models.pop(model_id, None)
                if docker_options_edited is not None:
                    await self._rollback_failed_install_docker(docker_options_edited, adopted=adopted)
                raise
            finally:
                self._installing.discard(key)

        return PromiseWithProgress(func=func)

    async def _uninstall_model(self, instance: str, model_id: str, options: UninstallModelIn) -> None:
        info = self.get_instance_installed_info(instance)
        srv_model = self.models.get(instance, {}).get(model_id)
        self._stop_health_monitor(instance, model_id)
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
            model_props=ModelProps(private=True, type="mcp", endpoints=["/mcp/open-websearch/mcp"], transport="streamable_http"),
            model_spec=mcp_service.get_default_model_spec("open-websearch"),
            model_type="mcp",
            default_prefix="open-websearch",
            size="427MB",
            options=DockerOptions(
                image_port=3000,
                name="open-websearch",
                container_name=mcp_service.docker_service.get_docker_container_name("open-websearch"),
                image="hub.simplito.com/deepfellow/open-websearch:v2.1.9",
                env_vars={},
                subnet=subnet,
            ),
            description="Multi-engine customizable web search with no API key required.",
            repository_url="https://github.com/aas-ee/open-websearch",
        ),
        "brave-search": lambda mcp_service, subnet: SrvMcpModel(
            model_props=ModelProps(private=True, type="mcp", endpoints=["/mcp/brave-search/mcp"], transport="streamable_http"),
            model_spec=mcp_service.get_default_model_spec(default_prefix="brave-search", required_envs=["BRAVE_API_KEY"]),
            model_type="mcp",
            default_prefix="brave-search",
            size="316MB",
            options=DockerOptions(
                image_port=8080,
                name="brave-search",
                container_name=mcp_service.docker_service.get_docker_container_name("brave-search"),
                image="hub.simplito.com/deepfellow/brave-search-mcp-server:v2.0.72",
                env_vars={"BRAVE_MCP_TRANSPORT": "http"},
                subnet=subnet,
            ),
            required_envs=["BRAVE_API_KEY"],
            description="Web search powered by the Brave Search API. Requires a Brave API key.",
            repository_url="https://github.com/brave/brave-search-mcp-server",
        ),
        "web-search": lambda mcp_service, subnet: SrvMcpModel(
            model_props=ModelProps(private=True, type="mcp", endpoints=["/mcp/web-search/mcp"], transport="streamable_http"),
            model_spec=mcp_service.get_default_model_spec("web-search"),
            model_type="mcp",
            default_prefix="web-search",
            size="3.84GB",
            options=DockerOptions(
                image_port=8080,
                name="web-search",
                container_name=mcp_service.docker_service.get_docker_container_name("web-search"),
                image="hub.simplito.com/deepfellow/web-search-mcp:v0.3.4",
                env_vars={},
                subnet=subnet,
            ),
            description="Concurrent multi-engine web search with full-page content extraction — no API key required.",
            repository_url="https://github.com/mrkrsl/web-search-mcp",
        ),
        "serpapi": lambda mcp_service, subnet: SrvMcpModel(
            model_props=ModelProps(private=True, type="mcp", endpoints=["/mcp/serpapi/mcp"], transport="streamable_http"),
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
            description="Google websearch solution based on Serp api. Requires a SerpApi key in Authorization header.",
            repository_url="https://github.com/serpapi/serpapi-mcp",
        ),
        "ollama-websearch": lambda mcp_service, subnet: SrvMcpModel(
            model_props=ModelProps(private=True, type="mcp", endpoints=["/mcp/ollama-websearch/mcp"], transport="streamable_http"),
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
            description="Ollama easy to use web search solution. Require Ollama Api Key",
            repository_url="https://docs.ollama.com/capabilities/web-search",
        ),
        "scrapling": lambda mcp_service, subnet: SrvMcpModel(
            model_props=ModelProps(private=True, type="mcp", endpoints=["/mcp/scrapling/mcp"], transport="streamable_http"),
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
            description="Fast, expanded, anti-bot-bypass web scraping and content extraction.",
            repository_url="https://github.com/D4Vinci/Scrapling",
        ),
        "firecrawl": lambda mcp_service, subnet: SrvMcpModel(
            model_props=ModelProps(private=True, type="mcp", endpoints=["/mcp/firecrawl/mcp"], transport="streamable_http"),
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
            description="Web crawling and markdown extraction via the Firecrawl API. Requires a Firecrawl API key.",
            repository_url="https://github.com/firecrawl/firecrawl-mcp-server",
        ),
        "duckduckgo": lambda mcp_service, subnet: SrvMcpModel(
            model_props=ModelProps(private=True, type="mcp", endpoints=["/mcp/duckduckgo/mcp"], transport="streamable_http"),
            model_spec=mcp_service.get_default_model_spec(default_prefix="duckduckgo"),
            model_type="mcp",
            default_prefix="duckduckgo",
            size="276MB",
            options=DockerOptions(
                image_port=8000,
                name="duckduckgo",
                container_name=mcp_service.docker_service.get_docker_container_name("duckduckgo"),
                image="ghcr.io/nickclyde/duckduckgo-mcp-server:0.7.0",
                subnet=subnet,
                command="python -m duckduckgo_mcp_server.server --transport streamable-http --host 0.0.0.0 --port 8000",
                # Any port: without docker_subnet DF Infra connects via localhost:<exposed port>, not :8000.
                env_vars={"DDG_ALLOWED_HOSTS": "{container_host}:*"},
            ),
            description="DuckDuckGo web search solution — no API key required.",
            repository_url="https://github.com/nickclyde/duckduckgo-mcp-server",
        ),
    },
)
