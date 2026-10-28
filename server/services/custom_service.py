# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Custom service."""

import asyncio
import copy
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Final
from urllib.parse import urlsplit

from fastapi import HTTPException
from pydantic import BaseModel, BeforeValidator, Field, ValidationInfo, field_validator

from server.applicationcontext import get_base_url
from server.config import AppSettings
from server.docker import DockerImage, DockerOptions
from server.endpointregistry import ProxyOptions, RegistrationId, RegistrationOptions
from server.models.api import ModelProps
from server.models.models import (
    CustomModelField,
    CustomModelId,
    CustomModelSpecification,
    InstallModelIn,
    InstallModelOptions,
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
from server.serviceprovider import ServiceRawConfig
from server.services.base2_service import Base2Service, CustomModel, Instance, InstanceConfig, ModelConfig
from server.utils.core import (
    PromiseWithProgress,
    Stream,
    StreamChunk,
    StreamChunkProgress,
    normalize_name,
    try_parse_pydantic,
)
from server.utils.docker_image_version import apply_image_version_override, docker_tags_model_field, split_image_repo_tag
from server.utils.hardware import HardwarePartInfo, NvidiaGpuInfo
from server.utils.logger import logger
from server.utils.registry_client import RegistryUnavailableError, image_without_registry_prefix, registry_for
from server.utils.size_fetcher import fmt_size

# Token window accepted by BAAI/bge-m3, which the deepfellow-bge-m3 image serves unmodified.
BGE_M3_CONTEXT_WINDOW = 8192
DOC_CHUNKER_PROXY_TIMEOUT_SECONDS: Final = 7200
# This Infra's compose service name, as its containers on the docker subnet reach it. Not unique when several Infra instances
# share one docker network: it resolves to any of them, so this Infra's key can reach the wrong one.
DOC_CHUNKER_DEFAULT_GATEWAY_URL: Final = "http://infra:8086"

type SrvCustomModelX = Callable[["CustomService", str | None], SrvCustomModel]
type DockerOptionsOrCallable = DockerOptions | Callable[[InstallModelOptions], DockerOptions]


def _is_gpu_hardware(hardware_parts: Sequence[HardwarePartInfo]) -> bool:
    return any(isinstance(h, NvidiaGpuInfo) for h in hardware_parts)


def _normalize_url(url: str) -> tuple[str, str | None, int | None, str] | None:
    """Return the comparable parts of `url`, or None when it can't be parsed (e.g. a non-numeric port)."""
    try:
        parts = urlsplit(url.strip())
        return parts.scheme.lower(), parts.hostname, parts.port, parts.path.rstrip("/")
    except ValueError:
        return None


def _is_own_gateway_url(url: str, config: AppSettings) -> bool:
    """Whether `url` is this Infra's subnet address or its public `infra_url` (literal match, no DNS)."""
    normalized = _normalize_url(url)
    if normalized is None:
        return False
    own_urls = [DOC_CHUNKER_DEFAULT_GATEWAY_URL, config.infra_url] if config.infra_url else [DOC_CHUNKER_DEFAULT_GATEWAY_URL]
    return any(normalized == _normalize_url(own_url) for own_url in own_urls)


def _model_default_image(model: "SrvCustomModel", hardware: str | None) -> str | None:
    """Return the image a model's own DockerOptions would use for the given hardware.

    Used to derive image-version selection generically from each model's existing hardware-aware
    (or static) image logic, instead of maintaining a separate per-model image registry.

    Raises RegistryUnavailableError (logged at `exception` level, naming the model) if the
    model's DockerOptions builder itself fails — a regression there is a server bug, not a "tag
    not available" situation, so callers should route it through the same fail-open handling as a
    genuinely unreachable registry rather than reporting it as an invalid tag.
    """
    try:
        docker_options = model.options({"hardware": hardware}) if isinstance(model.options, Callable) else model.options
    except Exception as exc:
        logger.exception("Failed to resolve default image for custom model %r (hardware=%r)", model.default_prefix, hardware)
        msg = f"Failed to resolve default image for custom model {model.default_prefix!r}"
        raise RegistryUnavailableError(msg) from exc
    return docker_options.image or None


def _tag_diff_tokens(tag_a: str, tag_b: str) -> tuple[set[str], set[str]]:
    """Return the "-"-separated tokens unique to each tag, e.g. ("1.0.0-cpu", "1.0.0-cuda-12.8") -> ({"cpu"}, {"cuda", "12.8"})."""
    tokens_a = {t for t in tag_a.lower().split("-") if t}
    tokens_b = {t for t in tag_b.lower().split("-") if t}
    return tokens_a - tokens_b, tokens_b - tokens_a


@dataclass
class SrvCustomModel:
    model_props: ModelProps
    model_spec: ModelSpecification
    model_type: str
    default_prefix: str
    size: str
    options: DockerOptionsOrCallable
    custom: CustomModelId | None = None
    description: str = ""
    repository_url: str | None = None
    # Proxy read timeout used when the install leaves `proxy_timeout_seconds` empty; None falls back to
    # the global `standard_proxy_timeout_seconds`. For services that answer only once a long job is done.
    default_proxy_timeout_seconds: int | None = None
    validate_install_options: Callable[[InstallModelOptions], None] | None = None


class SrvCustomCustomModel(BaseModel):
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
    healthcheck_cmd: str | None = None
    healthcheck_start_period: Annotated[str, Field(pattern=r"^\d+[smh]$")] | None = None
    description: str = ""
    repository_url: str | None = None


@dataclass
class CustomConst:
    models: dict[str, SrvCustomModelX]


class CustomModelOptions(BaseModel):
    prefix: Annotated[str, Field(pattern=r"^[a-zA-Z0-9_-]+$")]
    proxy_timeout_seconds: Annotated[int, Field(gt=0)] | None = None


@dataclass
class ModelInstalledInfo:
    id: str
    options: InstallModelIn
    docker_options: DockerOptions
    container_host: str
    container_port: int
    docker_exposed_port: int
    registration_id: RegistrationId
    prefix: str
    base_url: str

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


#: Built-in model ids that used to live in this service's catalog and have since moved to their own
#: dedicated service, mapped to that service's id. A `services.json` written before the move can still
#: list one as installed; without filtering it out here, `load_model` would treat it as a real failure
#: (it's no longer in `_const.models`) and raise a permanent "failed to load" warning on every future
#: startup instead of just quietly reflecting that it's gone from this catalog.
_RETIRED_MODEL_IDS: dict[str, str] = {"deepfellow-finetune": "finetune"}


class CustomService(Base2Service[InstalledInfo, DownloadedInfo]):
    models: dict[str, dict[str, "SrvCustomModel"]]
    _installing: set[tuple[str, str]]
    _persists_model_definitions = False

    def _after_init(self) -> None:
        self.load_default_models("default")
        self._installing = set()

    def get_secret_values(self) -> list[str]:
        """Return the service's and installed models' `password` option values."""
        values = super().get_secret_values()
        for instance_name, instance in self.instances_info.items():
            if instance.installed is None:
                continue
            for model_id, model_info in instance.installed.models.items():
                model = self.models.get(instance_name, {}).get(model_id)
                if model is None:
                    continue
                spec = model_info.options.spec or {}
                values += [str(spec[field.name]) for field in model.model_spec.fields if field.type == "password" and spec.get(field.name)]
        return values

    async def load_service(self, config: ServiceRawConfig) -> None:
        """Load service using the config, dropping any persisted entry for a retired model id.

        A dropped entry's container keeps running (`restart: unless-stopped`), holding its GPU and host
        port with nothing tracking it anymore, so it's torn down too, and the pruned config is saved
        right away so that happens only once.
        """
        retired: set[str] = set()
        instances = config.get("instances")
        if isinstance(instances, dict):
            for instance in instances.values():
                if isinstance(instance, dict) and isinstance(instance.get("models"), list):
                    kept: list[Any] = []
                    for m in instance["models"]:
                        if isinstance(m, dict) and m.get("model_id") in _RETIRED_MODEL_IDS:
                            retired.add(m["model_id"])
                        else:
                            kept.append(m)
                    instance["models"] = kept
        await super().load_service(config)
        if retired:
            for model_id in sorted(retired):
                await self._remove_retired_model_container(model_id)
            await self._save()

    async def _remove_retired_model_container(self, model_id: str) -> None:
        """Best-effort teardown of a retired model's container; a failure must not abort startup.

        The replacement service names its containers after its own service id (e.g. `finetune`, not
        `deepfellow-finetune`), so this container is never one of its own. Data volumes are always
        left in place.
        """
        container_name = self.docker_service.get_docker_container_name(model_id)
        try:
            await self.docker_service.uninstall_docker(DockerOptions(name=model_id, container_name=container_name, image="", image_port=0))
        except Exception:
            logger.warning("Failed to remove container %r of retired custom model %r", container_name, model_id, exc_info=True)
            return
        logger.info(
            "Removed container %r of retired custom model %r; install the %r service instead",
            container_name,
            model_id,
            _RETIRED_MODEL_IDS[model_id],
        )

    def load_default_models(self, instance: str) -> None:
        """Load default models to instance."""
        self.models[instance] = {}
        subnet = self.docker_service.get_docker_subnet()
        for model_id in _const.models.copy():
            model = _const.models[model_id](self, subnet)
            self._attach_docker_tags_field(model)
            self.models[instance][model_id] = model

    def _attach_docker_tags_field(self, model: "SrvCustomModel") -> None:
        """Append a "docker-tags" version-selection field, inferred from the model's own default image.

        Every built-in model already knows how to build its own DockerOptions (hardware-aware or
        not); this reuses that logic to derive the repo instead of requiring each model to also
        register itself in a separate image lookup.
        """
        if any(f.type == "docker-tags" for f in model.model_spec.fields):
            return
        try:
            default_image = _model_default_image(model, None)
        except RegistryUnavailableError:
            return
        if not default_image:
            return
        repo, tag = split_image_repo_tag(default_image)
        if not tag:
            return
        model.model_spec.fields.append(docker_tags_model_field(repo, depends_on="hardware"))

    def get_type(self) -> str:
        """Return the service id."""
        return "custom"

    def get_description(self) -> str:
        """Return the service description."""
        return "Your option to add custom service."

    def get_size(self) -> ServiceSize:
        """Return the service size."""
        return ""

    def get_spec(self) -> ServiceSpecification:
        """Return the service specification."""
        return ServiceSpecification(fields=[])

    def get_default_model_spec(self, default_prefix: str) -> ModelSpecification:
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
                self.get_proxy_timeout_field(),
            ]
        )

    def get_proxy_timeout_field(self, default: int | None = None) -> ModelField:
        """Field for the per-install proxy read timeout.

        Left unset, it falls back to `default` (the model's `default_proxy_timeout_seconds`) or, without
        one, to the global admin default as of install time — a later change to the global default does
        not affect already-installed services.
        """
        fallback = f"{default}s" if default is not None else "the global default"
        return ModelField(
            type="number",
            name="proxy_timeout_seconds",
            description=f"Seconds the proxy waits for a response from this service before timing out. Leave empty to use {fallback}.",
            required=False,
        )

    async def stop_instance(self, instance: str) -> None:
        """Stop all custom service Docker containers."""
        installed = self.get_instance_info(instance).installed
        if not installed:
            return
        await self._stop_dockers_parallel([model.docker_options for model in installed.models.values()])

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
                CustomModelField(type="list", name="volumes", description="Docker volumes", placeholder="/work/storage", required=False),
                CustomModelField(type="map", name="envs", description="Docker environment variables", required=False),
                CustomModelField(type="text", name="description", description="Short description of the service", required=False),
                CustomModelField(type="text", name="repository_url", description="Repository URL", required=False),
            ]
        )

    def _get_builtin_model(self, model_id: str | None) -> "SrvCustomModel | None":
        if not model_id:
            return None
        return self.models.get("default", {}).get(model_id)

    def get_docker_image_repo_for_model(self, model_id: str | None, hardware: str | None) -> str | None:
        """Return the Docker repo for a built-in model, derived from its own default image."""
        model = self._get_builtin_model(model_id)
        if not model:
            return None
        image = _model_default_image(model, hardware)
        return split_image_repo_tag(image)[0] if image else None

    def get_default_docker_tag_for_model(self, model_id: str | None, hardware: str | None) -> str | None:
        """Return a built-in model's default image tag, derived from its own default image."""
        model = self._get_builtin_model(model_id)
        if not model:
            return None
        image = _model_default_image(model, hardware)
        if not image:
            return None
        return split_image_repo_tag(image)[1] or None

    async def get_docker_tags_for_model(self, model_id: str | None, hardware: str | None) -> list[str]:
        """Fetch available Docker image tags for a built-in model, hardware-filtered when the repo is shared."""
        model = self._get_builtin_model(model_id)
        if not model:
            return []
        repo = self.get_docker_image_repo_for_model(model_id, hardware)
        if not repo:
            return []
        client = registry_for(repo)
        tags = await client.get_tags(image_without_registry_prefix(repo))
        return self._filter_tags_for_hardware(model, tags, hardware)

    def _filter_tags_for_hardware(self, model: "SrvCustomModel", tags: list[str], hardware: str | None) -> list[str]:
        """Narrow tags to the selected hardware when one repo's tags encode both variants (e.g. "1.0.0-cpu").

        Compares the model's own default image at CPU vs GPU to discover whether hardware affects
        the tag at all, and if so which tokens distinguish the variants — no per-model config needed.
        """
        try:
            cpu_image = _model_default_image(model, "CPU")
            gpu_image = _model_default_image(model, "GPU")
        except RegistryUnavailableError:
            return tags
        if not cpu_image or not gpu_image:
            return tags
        cpu_repo, cpu_tag = split_image_repo_tag(cpu_image)
        gpu_repo, gpu_tag = split_image_repo_tag(gpu_image)
        if cpu_repo != gpu_repo:
            # Hardware already selects the repo itself, so every tag in it is valid.
            return tags
        cpu_only, gpu_only = _tag_diff_tokens(cpu_tag, gpu_tag)
        if not cpu_only and not gpu_only:
            return tags
        is_gpu = _is_gpu_hardware(self.get_specified_hardware_parts(hardware))
        wanted, unwanted = (gpu_only, cpu_only) if is_gpu else (cpu_only, gpu_only)
        result = []
        for t in tags:
            tokens = {x for x in t.lower().split("-") if x}
            if wanted & tokens and not unwanted & tokens:
                result.append(t)
        return result

    async def _resolve_custom_model_size(self, spec: dict[str, Any], instance: str = "") -> str | None:  # noqa: ARG002
        try:
            size_bytes = await self.docker_service.get_docker_image_size(spec["image"])
            return fmt_size(size_bytes) if size_bytes else None
        except Exception:
            return None

    def get_installed_info(self, instance: str) -> bool | InstallServiceProgress | ServiceOptions:
        """Get service installed info."""
        installed = self.get_instance_info(instance).installed
        return self._get_service_installed_info(instance) if installed is None else installed.options.spec

    def _generate_instance_config(self, instance: str, info: InstalledInfo | None, custom: list[CustomModel] | None) -> InstanceConfig:  # noqa: ARG002
        return InstanceConfig(
            options=info.options if info else None,
            models=[
                ModelConfig(model_id=x.id, options=x.options, container_name=x.docker_options.container_name) for x in info.models.values()
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
            await asyncio.gather(
                *[
                    self._uninstall_model(instance, model.id, UninstallModelIn(purge=options.purge))
                    for model in installed.models.copy().values()
                    if not self.is_model_installed_in_other_instance(instance, model.id)
                ]
            )

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

        return model_installed.docker_options

    def _add_custom_model(self, instance: str, model: CustomModel) -> None:
        parsed = try_parse_pydantic(SrvCustomCustomModel, model.data)

        if not self.models.get(instance):
            self.models[instance] = {}

        if parsed.id in self.models[instance]:
            raise HTTPException(400, "Model with given id already exists.")
        self._check_prefix_collision(instance, parsed.default_prefix, exclude_model_id=None)
        name = normalize_name(f"{parsed.id}-{instance}")
        subnet = self.docker_service.get_docker_subnet()
        self.models[instance][parsed.id] = SrvCustomModel(
            model_props=ModelProps(private=parsed.private, type="custom", endpoints=[f"/custom/{parsed.default_prefix}"]),
            model_spec=self.get_default_model_spec(parsed.default_prefix),
            model_type="custom",
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
            description=parsed.description,
            repository_url=parsed.repository_url,
        )

    def _remove_custom_model(self, instance: str, model: CustomModel) -> None:
        installed = self.get_instance_info(instance).installed
        parsed = try_parse_pydantic(SrvCustomCustomModel, model.data)
        if installed and parsed.id in installed.models:
            raise HTTPException(400, "Cannot remove custom model, it is in use, uninstall it first.")
        # A model that failed to register (e.g. a prefix collision caught by load_instance) is
        # persisted in config.custom but never made it into self.models - removing it must still
        # succeed, not KeyError, so its persisted entry can actually be pruned.
        if instance in self.models and parsed.id in self.models[instance]:
            del self.models[instance][parsed.id]

    async def get_duplicate_spec(self, instance: str, model_id: str) -> dict[str, Any]:
        """Return a full add-model spec for duplicating `model_id`, custom-backed or catalog."""
        model = self.models.get(instance, {}).get(model_id)
        if model is None:
            raise HTTPException(400, "Model not found")
        if model.custom:
            definition = self.get_custom_model_definition(model.custom)
            if definition is None:
                raise HTTPException(404, f"Custom model definition for {model_id} not found.")
            return dict(definition)
        # Catalog model: `options` may be a callable resolving hardware/env-dependent docker options
        # from install-time fields - use the current install options if installed, else field defaults.
        installed = self.get_instance_info(instance).installed
        installed_model = installed.models.get(model_id) if installed else None
        install_spec: dict[str, Any] = (installed_model.options.spec or {}) if installed_model else {}
        docker_options = model.options(install_spec) if isinstance(model.options, Callable) else model.options
        docker_options = apply_image_version_override(docker_options, install_spec.get("image_version"))
        command = docker_options.command if isinstance(docker_options.command, str) else None
        healthcheck = docker_options.healthcheck or {}
        hardware_spec = self.canonicalize_hardware_spec(install_spec["hardware"]) if "hardware" in install_spec else None
        # A duplicate is persisted in services.json and can't follow key rotation, so it must not carry this Infra's key.
        infra_api_key = self.config.infra_api_key.get_secret_value()
        envs = (
            {key: "" if infra_api_key and value == infra_api_key else value for key, value in docker_options.env_vars.items()}
            if docker_options.env_vars
            else None
        )
        return {
            "id": model_id,
            "private": True,
            "default_prefix": model.default_prefix,
            "size": model.size,
            "image": docker_options.image,
            "image_port": docker_options.image_port,
            "command": command,
            "hardware": hardware_spec,
            # `docker_options.volumes` is already a fully-expanded `host:container` bind mount (see
            # e.g. lemmatizer's `generate_docker_options`), but `_add_custom_model` always re-prefixes
            # whatever it receives with the duplicate's own working directory, assuming a bare
            # container path - strip the host side back off so it gets that raw form, not a
            # three-segment, two-colon string Compose can't parse.
            "volumes": [v.split(":", 1)[1] if ":" in v else v for v in (docker_options.volumes or [])],
            "envs": envs,
            "healthcheck_cmd": healthcheck.get("test"),
            "healthcheck_start_period": healthcheck.get("start_period"),
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
                            has_docker=True,
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
            has_docker=True,
            description=model.description or None,
            repository_url=model.repository_url,
        )

    async def _install_model(  # noqa: C901
        self,
        instance: str,
        model_id: str,
        options: InstallModelIn,
        container_name_override: str | None = None,
    ) -> PromiseWithProgress[InstallModelOut, StreamChunk]:
        info = self.get_instance_installed_info(instance)
        if not self.models.get(instance):
            self.models[instance] = {}
        key = (instance, model_id)
        if model_id in info.models or key in self._installing:
            return PromiseWithProgress(value=InstallModelOut(status="OK", details="Already installed or being installed right now."))
        if model_id not in self.models[instance]:
            raise HTTPException(400, "Model not found")
        model = self.models[instance][model_id]
        if not options.spec:
            options.spec = {}
        if "prefix" not in options.spec:
            options.spec["prefix"] = model.default_prefix
        model.model_props.prefix = options.spec["prefix"]
        parsed_model_options = try_parse_pydantic(CustomModelOptions, options.spec)

        self._installing.add(key)

        try:
            if model.validate_install_options:
                model.validate_install_options(options.spec)
            await self.validate_docker_image_version_for_model(model_id, options.spec.get("image_version"), options.spec.get("hardware"))
            docker_options = model.options(options.spec) if isinstance(model.options, Callable) else model.options
            docker_options = apply_image_version_override(docker_options, options.spec.get("image_version"))
            if container_name_override or not isinstance(model.options, Callable):
                # Copy first - a static, non-callable model.options is shared across installs. Its container_name
                # was derived when the model list was built, so re-derive it: a live suffix change must apply too.
                docker_options = copy.copy(docker_options)
                docker_options.container_name = container_name_override or self.docker_service.get_docker_container_name(
                    docker_options.name
                )
            await self._verify_docker_image(docker_options.image, options.ignore_warnings)
        except BaseException:
            self._installing.discard(key)
            raise

        async def func(stream: Stream[StreamChunk]) -> InstallModelOut:
            try:
                model_dir = self._get_working_dir() / "models"
                model_dir.mkdir(parents=True, exist_ok=True)
                subnet = self.docker_service.get_docker_subnet()
                image = DockerImage(name=docker_options.image, size=model.size)
                await self._download_image_or_set_progress(stream, image)
                stream.emit(StreamChunkProgress(type="progress", stage="install", value=0, data={}))
                docker_exposed_port, _, _ = await self.docker_service.install_and_run_docker(docker_options)
                container_host = self.docker_service.get_container_host(subnet, docker_options)
                container_port = self.docker_service.get_container_port(subnet, docker_exposed_port, docker_options.image_port)
                info.models[model_id] = model_info = ModelInstalledInfo(
                    id=model_id,
                    options=options,
                    docker_options=docker_options,
                    container_host=container_host,
                    container_port=container_port,
                    docker_exposed_port=docker_exposed_port,
                    registration_id="",
                    prefix=parsed_model_options.prefix,
                    base_url=get_base_url(container_host, container_port),
                )
                proxy_timeout_seconds = next(
                    timeout
                    for timeout in (
                        parsed_model_options.proxy_timeout_seconds,
                        model.default_proxy_timeout_seconds,
                        self.config.standard_proxy_timeout_seconds,
                    )
                    if timeout is not None
                )
                try:
                    model_info.registration_id = self.endpoint_registry.register_custom_endpoint_as_proxy(
                        url=model_info.prefix,
                        props=model.model_props,
                        options=ProxyOptions(url=model_info.base_url, read_timeout_seconds=proxy_timeout_seconds),
                        registration_options=RegistrationOptions(origin="local", owned_by=self.get_type()),
                    )
                    self.models_downloaded[model_id] = DownloadedInfo(image.name)
                    stream.emit(StreamChunkProgress(type="progress", stage="install", value=1, data={}))
                    return InstallModelOut(status="OK", details="Installed")
                except Exception:
                    if info.models.get(model_id) is model_info:
                        info.models.pop(model_id, None)
                    raise
            finally:
                self._installing.discard(key)

        return PromiseWithProgress(func=func)

    async def _uninstall_model(self, instance: str, model_id: str, options: UninstallModelIn) -> None:
        info = self.get_instance_installed_info(instance)
        if model_id in info.models:
            model = info.models[model_id]
            self.endpoint_registry.unregister_custom_endpoint(model.prefix, model.registration_id)
            await self.docker_service.uninstall_docker(model.docker_options)
            del info.models[model_id]

        if options.purge and model_id in self.models_downloaded:
            await self.docker_service.remove_image(self.models_downloaded[model_id].image)
            del self.models_downloaded[model_id]

    def get_working_dir(self) -> Path:
        """Get working dir."""
        return self._get_working_dir()


def create_bge_m3_model(custom_service: CustomService, subnet: str | None) -> SrvCustomModel:
    """Create bge 3 model."""
    fields: list[ModelField] = custom_service.add_hardware_field_to_model_spec()
    fields.extend(
        [
            ModelField(
                type="text",
                name="prefix",
                description="Endpoint prefix",
                required=True,
                placeholder="deepfellow-bge-m3",
                default="deepfellow-bge-m3",
            ),
            ModelField(
                type="number",
                name="num_workers",
                description="Number of worker threads",
                required=False,
                placeholder="4",
                default="4",
            ),
            ModelField(
                type="number",
                name="queue_max",
                description="Maximum size of the job queue",
                required=False,
                placeholder="100000",
                default="100000",
            ),
            ModelField(
                type="number",
                name="shutdown_timeout",
                description="Grace period for finishing model tasks on shutdown.",
                required=False,
                placeholder="30",
                default="30",
            ),
            custom_service.get_proxy_timeout_field(),
        ]
    )

    def generate_docker_options(model_fields: InstallModelOptions) -> DockerOptions:
        hardware_parts = custom_service.get_specified_hardware_parts(model_fields.get("hardware"))
        image = (
            "hub.simplito.com/deepfellow/deepfellow-bge-m3:1.0.0-cuda-12.8"
            if _is_gpu_hardware(hardware_parts)
            else "hub.simplito.com/deepfellow/deepfellow-bge-m3:1.0.0-cpu"
        )
        return DockerOptions(
            image_port=8070,
            name="deepfellow-bge-m3",
            container_name=custom_service.docker_service.get_docker_container_name("deepfellow-bge-m3", apply_legacy_prefix=False),
            image=image,
            restart="unless-stopped",
            hardware=hardware_parts,
            subnet=subnet,
            env_vars={
                "DF_BGE_M3_NUM_WORKERS": model_fields.get("num_workers", 4),
                "DF_BGE_M3_QUEUE_MAX": model_fields.get("queue_max", 100000),
                "DF_BGE_M3_SHUTDOWN_TIMEOUT": model_fields.get("shutdown_timeout", 30),
            },
            healthcheck={
                "test": "wget -q --spider http://localhost:8070/health",
                "interval": "30s",
                "timeout": "10s",
                "retries": "3",
                "start_period": "30s",
            },
        )

    return SrvCustomModel(
        model_props=ModelProps(
            private=True,
            type="custom",
            endpoints=["/custom/deepfellow-bge-m3/v1/embeddings"],
            context_window=BGE_M3_CONTEXT_WINDOW,
            max_context_window=BGE_M3_CONTEXT_WINDOW,
        ),
        model_spec=ModelSpecification(fields=fields),
        model_type="custom",
        default_prefix="deepfellow-bge-m3",
        size="5.00GB",
        options=generate_docker_options,
        description="High-performance multilingual text embeddings model for sparse vectors.",
        repository_url="https://gitlab2.simplito.com/df/deepfellow-bge-m3",
    )


def create_lemmatizer_model(custom_service: CustomService, subnet: str | None) -> SrvCustomModel:
    """Create lemmatizer model."""
    fields: list[ModelField] = custom_service.add_hardware_field_to_model_spec()
    fields.extend(
        [
            ModelField(
                type="text",
                name="prefix",
                description="Endpoint prefix",
                required=True,
                placeholder="lemmatizer",
                default="lemmatizer",
            ),
            ModelField(
                type="number",
                name="num_workers",
                description="Number of worker threads",
                required=False,
                placeholder="4",
                default="4",
            ),
            ModelField(
                type="number",
                name="queue_max",
                description="Maximum size of the job queue",
                required=False,
                placeholder="100000",
                default="100000",
            ),
            ModelField(
                type="number",
                name="shutdown_timeout",
                description="Grace period for finishing lemmatizer tasks on shutdown.",
                required=False,
                placeholder="30",
                default="30",
            ),
            custom_service.get_proxy_timeout_field(),
        ]
    )

    def generate_docker_options(model_fields: InstallModelOptions) -> DockerOptions:
        hardware_parts = custom_service.get_specified_hardware_parts(model_fields.get("hardware"))
        image = (
            "hub.simplito.com/deepfellow/deepfellow-lemmatizer:1.0.0-cuda-12.8"
            if _is_gpu_hardware(hardware_parts)
            else "hub.simplito.com/deepfellow/deepfellow-lemmatizer:1.0.0-cpu"
        )
        return DockerOptions(
            image_port=8090,
            name="df-lemmatizer",
            container_name=custom_service.docker_service.get_docker_container_name("df-lemmatizer", apply_legacy_prefix=False),
            image=image,
            restart="unless-stopped",
            volumes=[f"{custom_service.get_working_dir()}/lemmatizer/cache/stanza:/root/.cache/stanza"],
            hardware=hardware_parts,
            subnet=subnet,
            env_vars={
                "DF_LEMMATIZER_NUM_WORKERS": model_fields.get("num_workers", 4),
                "DF_LEMMATIZER_QUEUE_MAX": model_fields.get("queue_max", 100000),
                "DF_LEMMATIZER_SHUTDOWN_TIMEOUT": model_fields.get("shutdown_timeout", 30),
            },
            healthcheck={
                "test": "wget -q --spider http://localhost:8090/health",
                "interval": "30s",
                "timeout": "10s",
                "retries": "3",
                "start_period": "30s",
            },
        )

    return SrvCustomModel(
        model_props=ModelProps(private=True, type="custom", endpoints=["/custom/v1/lemmatize"]),
        model_spec=ModelSpecification(fields=fields),
        model_type="custom",
        default_prefix="lemmatizer",
        size="1.89GB",
        options=generate_docker_options,
        description="Multilingual text lemmatization API.",
        repository_url="https://gitlab2.simplito.com/df/deepfellow-lemmatizer",
    )


def create_gliner_model(custom_service: CustomService, subnet: str | None) -> SrvCustomModel:
    """Create GLiNER named-entity-recognition model."""
    fields: list[ModelField] = custom_service.add_hardware_field_to_model_spec()
    fields.append(
        ModelField(
            type="text",
            name="prefix",
            description="Endpoint prefix",
            required=True,
            placeholder="gliner",
            default="gliner",
        )
    )
    fields.append(custom_service.get_proxy_timeout_field())

    def generate_docker_options(model_fields: InstallModelOptions) -> DockerOptions:
        hardware_parts = custom_service.get_specified_hardware_parts(model_fields.get("hardware"))
        image = (
            "hub.simplito.com/deepfellow/gliner-gpu:main"
            if any(isinstance(h, NvidiaGpuInfo) for h in hardware_parts)
            else "hub.simplito.com/deepfellow/gliner-cpu:main"
        )
        return DockerOptions(
            image_port=8000,
            name="df-gliner",
            container_name=custom_service.docker_service.get_docker_container_name("df-gliner", apply_legacy_prefix=False),
            image=image,
            restart="unless-stopped",
            hardware=hardware_parts,
            subnet=subnet,
            healthcheck={
                "test": "wget -q --spider http://localhost:8000/health",
                "interval": "30s",
                "timeout": "10s",
                "retries": "3",
                "start_period": "30s",
            },
        )

    return SrvCustomModel(
        model_props=ModelProps(private=True, type="custom", endpoints=["/custom/gliner/extract_entities"]),
        model_spec=ModelSpecification(fields=fields),
        model_type="custom",
        default_prefix="gliner",
        size="9.68GB",
        options=generate_docker_options,
        description="GLiNER named-entity-recognition API for knowledge-graph extraction.",
        repository_url="https://gitlab2.simplito.com/df/df-docker-images",
    )


def _empty_to_none(value: object) -> object:
    return None if value == "" else value


# An emptied optional form field arrives as "", which doc-chunker reads as unset (its own default).
type OptionalPositiveInt = Annotated[Annotated[int, Field(ge=1)] | None, BeforeValidator(_empty_to_none)]
type OptionalNonNegativeInt = Annotated[Annotated[int, Field(ge=0)] | None, BeforeValidator(_empty_to_none)]


class DocChunkerWorkerOptions(BaseModel):
    """Worker split doc-chunker accepts; it refuses to start with any other.

    Each field may be left unset, so doc-chunker falls back to its own default: max_concurrent is
    then auto-sized from the container's CPU/RAM limits (unknown here, so the split can't be
    checked against it), async_priority_workers 0, sync_only_workers 0.

    Cross-field rules are field validators rather than a model validator so each error carries the
    offending field - `format_pydantic_errors` reads the error location.
    """

    max_concurrent: OptionalPositiveInt = None
    async_priority_workers: OptionalNonNegativeInt = None
    sync_only_workers: OptionalNonNegativeInt = None

    @field_validator("async_priority_workers")
    @classmethod
    def _async_priority_below_max_concurrent(cls, value: int | None, info: ValidationInfo) -> int | None:
        max_concurrent = info.data.get("max_concurrent")
        if max_concurrent is not None and value is not None and value >= max_concurrent:
            msg = f"must be lower than max_concurrent ({max_concurrent})"
            raise ValueError(msg)
        return value

    @field_validator("sync_only_workers")
    @classmethod
    def _sync_only_fits_max_concurrent(cls, value: int | None, info: ValidationInfo) -> int | None:
        max_concurrent = info.data.get("max_concurrent")
        if max_concurrent is None or value is None:
            return value
        if value >= max_concurrent:
            msg = f"must be lower than max_concurrent ({max_concurrent})"
            raise ValueError(msg)
        if "async_priority_workers" not in info.data:
            return value  # async_priority_workers itself failed validation and already reports its own error
        async_priority = info.data["async_priority_workers"] or 0  # unset uses doc-chunker's default of 0
        if async_priority + value > max_concurrent:
            msg = f"together with async_priority_workers ({async_priority}) must not exceed max_concurrent ({max_concurrent})"
            raise ValueError(msg)
        return value


def validate_doc_chunker_workers(model_fields: InstallModelOptions) -> None:
    """Reject a worker split doc-chunker would refuse to start with."""
    try_parse_pydantic(DocChunkerWorkerOptions, model_fields)


def create_doc_chunker_model(custom_service: CustomService, subnet: str | None) -> SrvCustomModel:
    """Create doc-chunker model."""
    fields: list[ModelField] = custom_service.add_hardware_field_to_model_spec()
    fields.extend(
        [
            ModelField(
                type="text",
                name="prefix",
                description="Endpoint prefix",
                required=True,
                placeholder="doc_chunker",
                default="doc_chunker",
            ),
            ModelField(
                type="number",
                name="document_timeout",
                description="Timeout in seconds for a single document conversion",
                required=False,
                placeholder="3600",
                default="3600",
            ),
            ModelField(
                type="number",
                name="libreoffice_timeout",
                description="Timeout in seconds for a single LibreOffice conversion (legacy Office and OpenDocument files)",
                required=False,
                placeholder="120",
                default="120",
            ),
            ModelField(
                type="number",
                name="max_concurrent",
                description=(
                    "Workers | total parallel worker count. Workers not assigned as async priority or sync only "
                    "take sync jobs first and async jobs when none is waiting"
                ),
                required=False,
                placeholder="2",
                default="2",
            ),
            ModelField(
                type="number",
                name="async_priority_workers",
                description=(
                    "Workers | workers that take queued async jobs first and sync jobs when no async job is waiting. "
                    "Must be lower than max_concurrent; together with sync_only_workers must not exceed it"
                ),
                required=False,
                placeholder="0",
                default="0",
            ),
            ModelField(
                type="number",
                name="sync_only_workers",
                description=(
                    "Workers | workers reserved for sync jobs, never used by async jobs. "
                    "Must be lower than max_concurrent; together with async_priority_workers must not exceed it"
                ),
                required=False,
                placeholder="0",
                default="0",
            ),
            ModelField(
                type="number",
                name="max_queued_jobs",
                description="Async jobs | jobs that can wait in queue for a slot; above this limit new jobs are rejected",
                required=False,
                placeholder="100",
                default="100",
            ),
            ModelField(
                type="number",
                name="job_result_ttl",
                description="Async jobs | seconds a finished job's status and result stay available",
                required=False,
                placeholder="86400",
                default="86400",
            ),
            # Image description and speech-to-text both go through the same OpenAI-compatible
            # gateway, so they share one address and key. doc-chunker appends the endpoint paths
            # itself, and the model is chosen per request by the caller — neither belongs here.
            ModelField(
                type="text",
                name="infra_api_url",
                description="Inference gateway | base URL of the OpenAI-compatible API used to describe images and transcribe audio. ",
                required=True,
                placeholder=DOC_CHUNKER_DEFAULT_GATEWAY_URL,
                default=DOC_CHUNKER_DEFAULT_GATEWAY_URL,
            ),
            # Empty on purpose: this Infra's key is filled in only in generate_docker_options, never in the spec.
            ModelField(
                type="password",
                name="infra_api_key",
                description=(
                    "Inference gateway | API key, if the gateway requires one. Empty uses this Infra's key when the URL "
                    "points at this Infra; after rotating it, restart Infra or reinstall doc-chunker."
                ),
                required=False,
                placeholder="Empty: this Infra's API key",
                default="",
            ),
            ModelField(
                type="text",
                name="picture_description_prompt",
                description="Image description | prompt sent to the vision model",
                required=False,
                placeholder="Describe this image in detail.",
                default="Describe this image in detail.",
            ),
            ModelField(
                type="number",
                name="picture_description_api_max_time",
                description="Image description | maximum wait time in seconds for a single gateway response",
                required=False,
                placeholder="300",
                default="300",
            ),
            ModelField(
                type="number",
                name="picture_description_concurrency",
                description="Image description | pictures described in parallel within one document",
                required=False,
                placeholder="4",
                default="4",
            ),
            ModelField(
                type="text",
                name="audio_stt_max_time",
                description="Transcription | maximum wait time in seconds for a gateway response",
                required=False,
                placeholder="600",
                default="600",
            ),
            ModelField(
                type="text",
                name="audio_silence_threshold",
                description="Transcription | minimum duration (seconds) of silence used to split audio into segments",
                required=False,
                placeholder="2",
                default="2",
            ),
            custom_service.get_proxy_timeout_field(DOC_CHUNKER_PROXY_TIMEOUT_SECONDS),
        ]
    )

    def generate_docker_options(model_fields: InstallModelOptions) -> DockerOptions:
        hardware_parts = custom_service.get_specified_hardware_parts(model_fields.get("hardware"))
        # A blank value, or a model installed before install_model filled in field defaults.
        gateway_url = str(model_fields.get("infra_api_url") or "").strip() or DOC_CHUNKER_DEFAULT_GATEWAY_URL
        gateway_api_key = str(model_fields.get("infra_api_key") or "")
        if not gateway_api_key and _is_own_gateway_url(gateway_url, custom_service.config):
            # Read on every call, so a rotated key lands on the next container rebuild.
            gateway_api_key = custom_service.config.infra_api_key.get_secret_value()
        image = (
            "hub.simplito.com/deepfellow/doc-chunker-gpu:v2.4.0"
            if _is_gpu_hardware(hardware_parts)
            else "hub.simplito.com/deepfellow/doc-chunker-cpu:v2.4.0"
        )
        return DockerOptions(
            image_port=8000,
            name="doc_chunker",
            container_name=custom_service.docker_service.get_docker_container_name("doc_chunker", apply_legacy_prefix=False),
            image=image,
            restart="unless-stopped",
            # DATA_DIR (default /var/lib/doc-chunker): registered tokenizers and async job uploads/results.
            volumes=[f"{custom_service.get_working_dir()}/doc_chunker/data:/var/lib/doc-chunker"],
            hardware=hardware_parts,
            subnet=subnet,
            env_vars={
                "MAX_CONCURRENT": model_fields.get("max_concurrent", 2),
                "ASYNC_PRIORITY_WORKERS": model_fields.get("async_priority_workers", 0),
                "SYNC_ONLY_WORKERS": model_fields.get("sync_only_workers", 0),
                "MAX_QUEUED_JOBS": model_fields.get("max_queued_jobs", 100),
                "JOB_RESULT_TTL": model_fields.get("job_result_ttl", 86400),
                "DOCUMENT_TIMEOUT": model_fields.get("document_timeout", 3600),
                "LIBREOFFICE_TIMEOUT": model_fields.get("libreoffice_timeout", 120),
                "INFRA_API_URL": gateway_url,
                "INFRA_API_KEY": gateway_api_key,
                "PICTURE_DESCRIPTION_PROMPT": model_fields.get("picture_description_prompt", "Describe this image in detail."),
                "PICTURE_DESCRIPTION_API_MAX_TIME": model_fields.get("picture_description_api_max_time", "300"),
                "PICTURE_DESCRIPTION_CONCURRENCY": model_fields.get("picture_description_concurrency", 4),
                "AUDIO_STT_MAX_TIME": model_fields.get("audio_stt_max_time", "600"),
                "AUDIO_SILENCE_THRESHOLD": model_fields.get("audio_silence_threshold", "2"),
            },
            healthcheck={
                "test": "wget -q --spider http://localhost:8000/health",
                "interval": "30s",
                "timeout": "10s",
                "retries": "3",
                "start_period": "30s",
            },
        )

    return SrvCustomModel(
        model_props=ModelProps(private=True, type="custom", endpoints=["/custom/doc_chunker/"]),
        model_spec=ModelSpecification(fields=fields),
        model_type="custom",
        default_prefix="doc_chunker",
        size="15GB",
        options=generate_docker_options,
        description="Converts many data types (pdf, xlxs, mp3, etc.) into text chunks suitable for vector search.",
        default_proxy_timeout_seconds=DOC_CHUNKER_PROXY_TIMEOUT_SECONDS,
        validate_install_options=validate_doc_chunker_workers,
    )


_const = CustomConst(
    models={
        "bentoml/example-summarization": lambda custom_service, subnet: SrvCustomModel(
            model_props=ModelProps(private=True, type="custom", endpoints=["/custom/bentoml-summarize"]),
            model_spec=custom_service.get_default_model_spec("bentoml-summarize"),
            model_type="custom",
            default_prefix="bentoml-summarize",
            size="12GB",
            options=DockerOptions(
                image_port=3000,
                name="bentoml",
                container_name=custom_service.docker_service.get_docker_container_name("bentoml"),
                image="hub.simplito.com/deepfellow/bentomlexample:1.0.1",
                command="serve",
                env_vars={},
                subnet=subnet,
            ),
            description="BentoML example to create text summarization.",
        ),
        "easyOCR": lambda custom_service, subnet: SrvCustomModel(
            model_props=ModelProps(private=True, type="custom", endpoints=["/custom/ocr"]),
            model_spec=custom_service.get_default_model_spec("ocr"),
            model_type="custom",
            default_prefix="ocr",
            size="10GB",
            options=DockerOptions(
                image_port=8000,
                name="easyocr",
                container_name=custom_service.docker_service.get_docker_container_name("easyocr"),
                image="hub.simplito.com/deepfellow/df-ocr:1.0.1",
                env_vars={},
                restart="unless-stopped",
                volumes=[f"{custom_service.get_working_dir()}/easyocr/model:/root/.EasyOCR/model"],
                subnet=subnet,
            ),
            description="Optical character recognition service, to change binary data to text.",
        ),
        "lemmatizer": create_lemmatizer_model,
        "gliner": create_gliner_model,
        "doc_chunker": create_doc_chunker_model,
        "deepfellow-bge-m3": create_bge_m3_model,
    }
)
