# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Docker Model Runner service."""

import html
import json
import platform
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel

from server.config import get_main_dir
from server.endpointregistry import ProxyOptions, RegistrationId, RegistrationOptions
from server.models.api import EMBEDDINGS_ENDPOINTS, ModelProps
from server.models.models import (
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
    OneOfOption,
    ServiceField,
    ServiceOptions,
    ServiceSpecification,
    UninstallServiceIn,
)
from server.services.base2_service import Base2Service, CustomModel, Instance, InstanceConfig, ModelConfig
from server.utils.core import (
    PromiseWithProgress,
    Stream,
    StreamChunk,
    StreamChunkProgress,
    Utils,
    convert_size_to_bytes,
    fetch_from,
    stream_fetch_from,
    try_parse_pydantic,
)
from server.utils.loading import Progress
from server.utils.logger import logger
from server.utils.size_fetcher import fmt_size


class DmrModel(BaseModel):
    id: str
    size: str
    type: str


class DmrRegistryEntry(BaseModel):
    name: str
    size: str


class DmrRegistry(BaseModel):
    llms: list[DmrRegistryEntry]
    embeddings: list[DmrRegistryEntry]


def _read_models() -> dict[str, DmrModel]:
    path = get_main_dir() / "./static/docker-model-runner-min.json"
    with path.open(encoding="utf-8") as f:
        data = json.loads(f.read())
        registry = DmrRegistry(llms=data.get("llms", []), embeddings=data.get("embeddings", []))
        result: dict[str, DmrModel] = {}
        for entry in registry.llms:
            result[entry.name] = DmrModel(id=entry.name, size=entry.size, type="llm")
        for entry in registry.embeddings:
            result[entry.name] = DmrModel(id=entry.name, size=entry.size, type="embedding")
        return result


_const_models = _read_models()


def _normalize_dmr_model_id(raw_id: str) -> str:
    """Normalize a model id reported by DMR's list endpoint (e.g. docker.io/ai/smollm2:latest) to the canonical short id (ai/smollm2)."""
    return raw_id.removeprefix("docker.io/").removesuffix(":latest")


@dataclass
class ModelInstalledInfo:
    id: str
    registered_name: str
    type: str
    options: InstallModelIn
    registration_id: RegistrationId

    def get_info(self) -> ModelInfo:
        """Get info."""
        return ModelInfo(spec=self.options.spec, registration_id=self.registration_id)


class DockerModelRunnerOptions(BaseModel):
    url: str = "http://localhost:12434"
    backend: str = "none"


class DockerModelRunnerModelOptions(BaseModel):
    rename: str | None = None
    alias: str | None = None


@dataclass
class InstalledInfo:
    models: dict[str, ModelInstalledInfo]
    options: InstallServiceIn
    parsed_options: DockerModelRunnerOptions
    base_url: str
    backend: str


@dataclass
class DownloadedInfo:
    pass


class DockerModelRunnerService(Base2Service[InstalledInfo, DownloadedInfo]):
    models: dict[str, dict[str, DmrModel]]

    def _after_init(self) -> None:
        self.models = {}
        self._load_default_models("default")

    def _load_default_models(self, instance: str) -> None:
        self.models[instance] = _const_models.copy()

    def get_type(self) -> str:
        """Return the service type."""
        return "docker-model-runner"

    def get_description(self) -> str:
        """Return the service description."""
        return "Docker Model Runner — llama.cpp, vLLM, and vLLM-Metal (Apple Silicon)."

    def get_size(self) -> str:
        """Return the service size."""
        return ""

    def get_spec(self) -> ServiceSpecification:
        """Return the service specification."""
        system = platform.system().lower()
        machine = platform.machine().lower()
        if system == "darwin" and machine in ("arm64", "aarch64"):
            default_backend = "vllm-metal"
        elif system == "linux":
            default_backend = "vllm"
        else:
            default_backend = "none"

        return ServiceSpecification(
            fields=[
                ServiceField(
                    type="text",
                    name="url",
                    description="Docker Model Runner base URL",
                    default="http://localhost:12434",
                ),
                ServiceField(
                    type="oneof",
                    name="backend",
                    description="Inference backend to install (llama.cpp is built-in; vLLM and vLLM-Metal require installation)",
                    required=False,
                    default=default_backend,
                    values=[
                        OneOfOption(value="none", label="None (llama.cpp built-in)"),
                        OneOfOption(value="vllm", label="vLLM (Linux / NVIDIA CUDA)"),
                        OneOfOption(value="vllm-metal", label="vLLM-Metal (macOS / Apple Silicon)"),
                    ],
                ),
            ]
        )

    def get_model_spec(self) -> ModelSpecification:
        """Return the model specification."""
        return ModelSpecification(
            fields=[
                ModelField(type="text", name="rename", description="Rename the model, hiding its original name", required=False),
                ModelField(
                    type="text", name="alias", description="Additional name for the model, the original name keeps working", required=False
                ),
            ]
        )

    def get_custom_model_spec(self) -> CustomModelSpecification | None:
        """Return None — custom models not supported."""
        return None

    def get_installed_info(self, instance: str) -> bool | InstallServiceProgress | ServiceOptions:
        """Get service installed info."""
        installed = self.get_instance_info(instance).installed
        return self._get_service_installed_info(instance) if installed is None else installed.options.spec

    def _generate_instance_config(self, instance: str, info: InstalledInfo | None, custom: list[CustomModel] | None) -> InstanceConfig:  # noqa: ARG002
        return InstanceConfig(
            options=info.options if info else None,
            models=[ModelConfig(model_id=x.id, options=x.options) for x in info.models.values()] if info else [],
            custom=custom,
        )

    async def load_instance(self, instance: str, instance_data: InstanceConfig) -> None:
        """Load instance from persisted config, skipping models no longer present in DMR."""
        if not instance_data.options:
            return

        self._load_default_models(instance)

        promise = await self.install_instance(instance, instance_data.options, instance_data, save=False)
        await promise.wait()
        await self._save()

        instance_info = self.instances_info[instance]
        installed = instance_info.installed
        if installed:
            instance_info.config.models = [ModelConfig(model_id=mid, options=installed.models[mid].options) for mid in installed.models]

    def service_has_docker(self) -> bool:
        """Return False — DMR is managed by Docker Desktop, not by us."""
        return False

    async def stop_instance(self, instance: str) -> None:
        """Stop instance — no-op for DMR."""

    def _load_download_info(self, data: dict[str, Any]) -> DownloadedInfo:  # noqa: ARG002
        return DownloadedInfo()

    def _determine_model_type(self, model_id: str, instance: str) -> str:
        if instance in self.models and model_id in self.models[instance]:
            return self.models[instance][model_id].type
        if model_id in _const_models:
            return _const_models[model_id].type
        return "llm"

    def _register_model(
        self, instance: str, installed_info: InstalledInfo, model_id: str, size_bytes: int, options: InstallModelIn | None = None
    ) -> None:
        model_type = self._determine_model_type(model_id, instance)
        size_str = fmt_size(size_bytes) if size_bytes else ""

        self.models[instance][model_id] = DmrModel(id=model_id, size=size_str, type=model_type)
        self.models_downloaded[model_id] = DownloadedInfo()

        parsed_options = (
            try_parse_pydantic(DockerModelRunnerModelOptions, options.spec) if options and options.spec else DockerModelRunnerModelOptions()
        )
        registered_name = parsed_options.rename if parsed_options.rename else model_id
        registered_alias = parsed_options.alias

        model_info = ModelInstalledInfo(
            id=model_id,
            type=model_type,
            registered_name=registered_name,
            options=options or InstallModelIn(),
            registration_id="",
        )
        if model_type == "llm":
            model_info.registration_id = self.endpoint_registry.register_chat_completion_as_proxy(
                model=registered_name,
                props=ModelProps(private=True, type="llm"),
                chat_completions=ProxyOptions(
                    url=f"{installed_info.base_url}/engines/v1/chat/completions",
                    rewrite_model_to=model_id,
                    forward_extra_fields=True,
                ),
                completions=ProxyOptions(
                    url=f"{installed_info.base_url}/engines/v1/completions",
                    rewrite_model_to=model_id,
                    forward_extra_fields=True,
                ),
                responses=None,
                messages=None,
                ollama_chat=None,
                registration_options=RegistrationOptions(origin="local", alias=registered_alias),
            )
        if model_type == "embedding":
            model_info.registration_id = self.endpoint_registry.register_embeddings_as_proxy(
                model=registered_name,
                props=ModelProps(private=True, type="embedding", endpoints=EMBEDDINGS_ENDPOINTS),
                options=ProxyOptions(
                    url=f"{installed_info.base_url}/engines/v1/embeddings",
                    rewrite_model_to=model_id,
                ),
                registration_options=RegistrationOptions(origin="local", alias=registered_alias),
            )
        installed_info.models[model_id] = model_info

    def _remove_stale_models(self, installed_info: InstalledInfo, stale_ids: set[str]) -> None:
        for model_id in stale_ids:
            model = installed_info.models.pop(model_id, None)
            if model is not None:
                if model.type == "llm":
                    self.endpoint_registry.unregister_chat_completion(model.registered_name, model.registration_id)
                if model.type == "embedding":
                    self.endpoint_registry.unregister_embeddings(model.registered_name, model.registration_id)
            self.models_downloaded.pop(model_id, None)

    async def _sync_models(self, instance: str, installed_info: InstalledInfo, *, is_initial: bool = False) -> None:
        res = await fetch_from(f"{installed_info.base_url}/engines/v1/models", "GET", None)
        if res.status_code != 200:
            logger.warning(f"DMR model sync for instance {instance} failed: HTTP {res.status_code}")
            return

        data = json.loads(res.data)
        saved_models = {m.model_id: m.options for m in (self.get_instance_info(instance).config.models or [])}
        current_model_ids: set[str] = set()

        for entry in data.get("data", []):
            model_id: str = _normalize_dmr_model_id(entry["id"])
            current_model_ids.add(model_id)

            if model_id in installed_info.models:
                continue
            if is_initial:
                # Previously downloaded but no longer in the persisted config means it was
                # explicitly uninstalled without purge — its weights are still in DMR, but it
                # must not be silently resurrected as installed on restart.
                if model_id in self.models_downloaded and model_id not in saved_models:
                    continue
            elif model_id in self.models_downloaded:
                continue

            size_bytes: int = entry.get("size", 0) or 0
            self._register_model(instance, installed_info, model_id, size_bytes, saved_models.get(model_id))

        self._remove_stale_models(installed_info, (set(installed_info.models) | set(saved_models)) - current_model_ids)

    async def _install_instance(
        self,
        instance: str,
        options: InstallServiceIn,
        container_name_override: str | None = None,  # noqa: ARG002
    ) -> PromiseWithProgress[InstalledInfo, StreamChunk]:
        if not self.models.get(instance):
            self._load_default_models(instance)

        parsed_options = try_parse_pydantic(DockerModelRunnerOptions, options.spec)

        def _raise_connection_error(url: str) -> None:
            msg = f"Cannot connect to Docker Model Runner at {url}"
            raise HTTPException(status_code=400, detail=msg)

        async def func(stream: Stream[StreamChunk]) -> InstalledInfo:
            stream.emit(StreamChunkProgress(type="progress", stage="install", value=0, data={}))
            try:
                res = await fetch_from(f"{parsed_options.url.rstrip('/')}/engines/v1/models", "GET", None)
                if res.status_code != 200:
                    _raise_connection_error(parsed_options.url)
            except HTTPException:
                raise
            except Exception as e:
                msg = f"Cannot connect to Docker Model Runner at {parsed_options.url}: {e!s}"
                raise HTTPException(status_code=400, detail=msg) from e

            stream.emit(StreamChunkProgress(type="progress", stage="install", value=0.2, data={}))

            if parsed_options.backend != "none":
                stream.emit(StreamChunkProgress(type="progress", stage="install", value=0.3, data={}))
                result = await Utils.run_command(["docker", "model", "install-runner", "--backend", parsed_options.backend])
                if result.exit_code != 0:
                    msg = f"Failed to install DMR backend '{parsed_options.backend}': {result.stderr}"
                    raise HTTPException(status_code=500, detail=msg)
                stream.emit(StreamChunkProgress(type="progress", stage="install", value=0.6, data={}))

            installed_info = InstalledInfo(
                models={},
                options=options,
                parsed_options=parsed_options,
                base_url=parsed_options.url.rstrip("/"),
                backend=parsed_options.backend,
            )
            stream.emit(StreamChunkProgress(type="progress", stage="install", value=0.8, data={}))
            await self._sync_models(instance, installed_info, is_initial=True)
            stream.emit(StreamChunkProgress(type="progress", stage="install", value=1, data={}))
            return installed_info

        return PromiseWithProgress(func=func)

    async def _uninstall_instance(self, instance: str, options: UninstallServiceIn) -> None:
        installed = self.get_instance_info(instance).installed
        if installed:
            for model in list(installed.models.values()):
                if model.type == "llm":
                    self.endpoint_registry.unregister_chat_completion(model.registered_name, model.registration_id)
                if model.type == "embedding":
                    self.endpoint_registry.unregister_embeddings(model.registered_name, model.registration_id)

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

    async def list_models(self, input_instance: str | list[str] | None, filters: ListModelsFilters) -> ListModelsOut:
        """List models."""
        instances = [input_instance] if isinstance(input_instance, str) else input_instance if input_instance else list(self.instances_info)

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
                            type=model.type,
                            installed=installed,
                            downloaded=model_id in self.models_downloaded,
                            size=model.size,
                            custom=None,
                            spec=self.get_model_spec(),
                            has_docker=False,
                        )
                    )
        return ListModelsOut(list=out_list)

    async def get_model(self, instance: str, model_id: str) -> RetrieveModelOut:
        """Get model."""
        info = self.get_instance_installed_info(instance)

        if model_id not in self.models.get(instance, {}):
            raise HTTPException(status_code=400, detail="Model not found")

        model = self.models[instance][model_id]
        installed = info.models[model_id].get_info() if model_id in info.models else self._get_model_installed_info(instance, model_id)
        return RetrieveModelOut(
            id=model_id,
            service=self.get_id(instance),
            type=model.type,
            installed=installed,
            downloaded=model_id in self.models_downloaded,
            size=model.size,
            custom=None,
            spec=self.get_model_spec(),
            has_docker=False,
        )

    def _handle_pull_record(
        self, stream: Stream[StreamChunk], record: dict[str, Any], progress: Progress, last_layer_values: dict[str, int]
    ) -> None:
        if "error" in record or record.get("type") == "error":
            raise HTTPException(status_code=500, detail=str(record.get("error") or record.get("message") or "Unknown DMR pull error"))

        handled = False
        layer = record.get("layer")
        if isinstance(layer, dict) and isinstance(layer.get("current"), (int, float)) and progress.max > 0:
            layer_id = str(layer.get("id") or "")
            current = layer["current"]
            increment = max(0, current - last_layer_values.get(layer_id, 0))
            if increment > 0:
                progress.add_to_actual_value(increment)
            last_layer_values[layer_id] = current
            stream.emit(StreamChunkProgress(type="progress", stage="download", value=min(progress.get_percentage(), 0.99), data={}))
            handled = True
        else:
            completed = record.get("completed") or record.get("downloaded") or record.get("loaded")
            total = record.get("total") or record.get("size") or progress.max
            if isinstance(completed, (int, float)) and isinstance(total, (int, float)) and total > 0:
                pct = min(completed / total, 0.99)
                stream.emit(StreamChunkProgress(type="progress", stage="download", value=pct, data={}))
                handled = True

        if not handled and (
            record.get("type") in ("success", "complete", "done") or record.get("status") in ("success", "done", "complete", "finished")
        ):
            stream.emit(StreamChunkProgress(type="progress", stage="download", value=0.99, data={}))

    def _parse_pull_line(self, line: str) -> dict[str, Any] | None:
        """Parse a single DMR pull stream line, tolerating HTML-escaped payloads."""
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            pass
        # Some DMR builds emit the pull progress stream HTML-escaped (&#34; instead of "),
        # so retry the unescaped payload before giving up on the line.
        unescaped = html.unescape(line)
        if unescaped == line:
            return None
        try:
            return json.loads(unescaped)
        except json.JSONDecodeError:
            return None

    def _handle_pull_line(
        self, stream: Stream[StreamChunk], line: str, progress: Progress, last_layer_values: dict[str, int], skipped_lines: list[str]
    ) -> None:
        line = line.strip()
        if not line:
            return
        record = self._parse_pull_line(line)
        if record is None:
            # Warn once per pull — an unparseable stream would otherwise flood the log with one entry per line.
            if not skipped_lines:
                logger.warning(f"DMR pull stream sent an unparseable line: {line!r}")
            skipped_lines.append(line)
            return
        self._handle_pull_record(stream, record, progress, last_layer_values)

    async def _pull_model(self, stream: Stream[StreamChunk], model: DmrModel, model_id: str, base_url: str) -> None:
        total_bytes = convert_size_to_bytes(model.size) or 0
        progress = Progress(total_bytes)
        last_layer_values: dict[str, int] = {}
        skipped_lines: list[str] = []

        stream.emit(StreamChunkProgress(type="progress", stage="download", value=0, data={}))
        # Network chunks aren't newline-aligned — a single JSON progress line can span multiple
        # chunks, so buffer across iterations and only parse complete lines.
        buffer = ""
        async for chunk in stream_fetch_from(f"{base_url}/models/create", "POST", {"from": model_id}, timeout=24 * 60 * 60):
            if chunk.status_code not in (200, 201):
                msg = f"Failed to pull model {model_id}: HTTP {chunk.status_code}"
                raise HTTPException(status_code=500, detail=msg)

            buffer += chunk.data
            *complete_lines, buffer = buffer.split("\n")
            for line in complete_lines:
                self._handle_pull_line(stream, line, progress, last_layer_values, skipped_lines)

        self._handle_pull_line(stream, buffer, progress, last_layer_values, skipped_lines)
        if skipped_lines:
            logger.warning(f"DMR pull stream sent {len(skipped_lines)} unparseable line(s) while pulling {model_id}")
        stream.emit(StreamChunkProgress(type="progress", stage="download", value=1, data={}))

    async def _install_model(
        self,
        instance: str,
        model_id: str,
        options: InstallModelIn,
        container_name_override: str | None = None,  # noqa: ARG002
    ) -> PromiseWithProgress[InstallModelOut, StreamChunk]:
        parsed_options = (
            try_parse_pydantic(DockerModelRunnerModelOptions, options.spec) if options.spec else DockerModelRunnerModelOptions()
        )
        info = self.get_instance_installed_info(instance)

        if model_id in info.models:
            return PromiseWithProgress(value=InstallModelOut(status="OK", details="Already installed"))

        if model_id not in self.models.get(instance, {}):
            raise HTTPException(400, "Model not found")

        model = self.models[instance][model_id]

        if model_id.startswith("mlx-community/") and not (
            platform.system().lower() == "darwin" and platform.machine().lower() in ("arm64", "aarch64")
        ):
            raise HTTPException(400, "MLX models are only supported on macOS with Apple Silicon")

        async def func(stream: Stream[StreamChunk]) -> InstallModelOut:
            await self._pull_model(stream, model, model_id, info.base_url)

            stream.emit(StreamChunkProgress(type="progress", stage="install", value=0, data={}))
            registered_name = parsed_options.rename if parsed_options.rename else model_id
            registered_alias = parsed_options.alias

            model_info = ModelInstalledInfo(
                id=model_id,
                type=model.type,
                registered_name=registered_name,
                options=options,
                registration_id="",
            )
            if model.type == "llm":
                model_info.registration_id = self.endpoint_registry.register_chat_completion_as_proxy(
                    model=registered_name,
                    props=ModelProps(private=True, type="llm"),
                    chat_completions=ProxyOptions(
                        url=f"{info.base_url}/engines/v1/chat/completions",
                        rewrite_model_to=model_id,
                        forward_extra_fields=True,
                    ),
                    completions=ProxyOptions(
                        url=f"{info.base_url}/engines/v1/completions",
                        rewrite_model_to=model_id,
                        forward_extra_fields=True,
                    ),
                    responses=None,
                    messages=None,
                    ollama_chat=None,
                    registration_options=RegistrationOptions(origin="local", alias=registered_alias),
                )
            if model.type == "embedding":
                model_info.registration_id = self.endpoint_registry.register_embeddings_as_proxy(
                    model=registered_name,
                    props=ModelProps(private=True, type="embedding", endpoints=EMBEDDINGS_ENDPOINTS),
                    options=ProxyOptions(
                        url=f"{info.base_url}/engines/v1/embeddings",
                        rewrite_model_to=model_id,
                    ),
                    registration_options=RegistrationOptions(origin="local", alias=registered_alias),
                )

            info.models[model_id] = model_info
            self.models_downloaded[model_id] = DownloadedInfo()
            stream.emit(StreamChunkProgress(type="progress", stage="install", value=1, data={}))
            await self._save()
            return InstallModelOut(status="OK", details="Installed")

        return PromiseWithProgress(func=func)

    async def _uninstall_model(self, instance: str, model_id: str, options: UninstallModelIn) -> None:
        info = self.get_instance_installed_info(instance)
        if model_id in info.models:
            model = info.models[model_id]
            del info.models[model_id]
            if model.type == "llm":
                self.endpoint_registry.unregister_chat_completion(model.registered_name, model.registration_id)
            if model.type == "embedding":
                self.endpoint_registry.unregister_embeddings(model.registered_name, model.registration_id)

        if options.purge and model_id in self.models_downloaded:
            await fetch_from(f"{info.base_url}/models/{Utils.str_encode(model_id)}", "DELETE", None)
            del self.models_downloaded[model_id]

        await self._save()
