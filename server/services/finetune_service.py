# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Finetune service.

Owns one `deepfellow-finetune` Docker container per instance (an LLM LoRA fine-tuning framework
with an OpenAI-compatible fine-tuning job queue and adapter registry). Unlike most `Base2Service`
subclasses, there is nothing to separately "install" per model - the container itself *is* the
whole service, so it's created directly by `_install_instance` (one step), not by a second,
per-model install call. This service has no installable models at all.

The HuggingFace base-model cache and the adapters directory are shared by all instances, so every
instance lists (and can delete) adapters trained by any other one. Uploaded training files,
configs and the job database are per instance.
"""

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import aiohttp
from fastapi import Depends, HTTPException
from fastapi import Path as ApiPath
from pydantic import BaseModel, Field

from server.applicationcontext import get_base_url
from server.core.dependencies import get_services_manager
from server.docker import DockerOptions
from server.endpointregistry import ProxyOptions, RegistrationId, RegistrationOptions
from server.models.api import ModelProps
from server.models.models import (
    CustomModelSpecification,
    InstallModelIn,
    InstallModelOut,
    ListModelsFilters,
    ListModelsOut,
    RetrieveModelOut,
    UninstallModelIn,
)
from server.models.services import (
    InstallServiceIn,
    InstallServiceOut,
    InstallServiceProgress,
    ServiceField,
    ServiceOptions,
    ServiceSize,
    ServiceSpecification,
    UninstallServiceIn,
)
from server.serviceprovider import ServiceRawConfig
from server.services.base2_service import Base2Service, CustomModel, Instance, InstanceConfig
from server.services_manager import ServicesManager
from server.utils.core import PromiseWithProgress, Stream, StreamChunk, StreamChunkProgress, Utils, try_parse_pydantic

_FINETUNE_MODEL_ID = "deepfellow-finetune"
_FINETUNE_IMAGE = f"hub.simplito.com/deepfellow/{_FINETUNE_MODEL_ID}:0.0.20"


class FinetuneOptions(BaseModel):
    hardware: str | bool | None = None
    prefix: Annotated[str, Field(pattern=r"^[a-zA-Z0-9_-]+$")] = _FINETUNE_MODEL_ID
    proxy_timeout_seconds: Annotated[int, Field(gt=0)] | None = None


@dataclass
class InstalledInfo:
    docker: DockerOptions
    options: InstallServiceIn
    parsed_options: FinetuneOptions
    container_host: str
    container_port: int
    docker_exposed_port: int
    registration_id: RegistrationId
    base_url: str


@dataclass
class DownloadedInfo:
    image: str


class FinetuneAdapterOut(BaseModel):
    """A single fine-tuned LoRA adapter, as reported by df-finetune's OpenAI-compatible model listing."""

    id: str
    object: str = "model"
    created: int
    owned_by: str = "deepfellow"
    base_model: str
    formats: list[str]
    lora: dict[str, Any] | None = None
    gguf: dict[str, Any] | None = None


class FinetuneAdapterListOut(BaseModel):
    object: str = "list"
    data: list[FinetuneAdapterOut]


class FinetuneService(Base2Service[InstalledInfo, DownloadedInfo]):
    # Registration id the next `_install_instance` re-registers the instance under. Set from the persisted
    # config on load, and by `_uninstall_instance` so update and its rollback (which bypass `install_instance`)
    # keep the id. A user install on a free instance drops it, so a fresh install gets a new id; one on a busy instance leaves it.
    _pinned_registration_ids: dict[str, RegistrationId]
    _pins_registration_ids = True

    def _after_init(self) -> None:
        self._pinned_registration_ids = {}

    async def install_instance(
        self,
        instance: str,
        options: InstallServiceIn,
        instance_config: InstanceConfig | None = None,
        save: bool = True,
        run_after_install: bool = True,
    ) -> PromiseWithProgress[InstallServiceOut, StreamChunk]:
        """Install the instance, re-registering it under the registration id persisted in `instance_config`.

        Without one (a user install), drops any pin left by an earlier uninstall, so the instance gets a new id.
        A busy (installed or installing) instance is rejected by the base without touching the pin.
        """
        # Let the base reject a busy instance before touching the pin - an in-flight update relies on it
        info = self.instances_info.get(instance)
        if info and (info.installed or info.installing):
            return await super().install_instance(instance, options, instance_config, save=save, run_after_install=run_after_install)
        pinned = instance_config.registration_id if instance_config else None
        if pinned:
            self._pinned_registration_ids[instance] = pinned
        else:
            self._pinned_registration_ids.pop(instance, None)
        try:
            return await super().install_instance(instance, options, instance_config, save=save, run_after_install=run_after_install)
        except BaseException:
            self._pinned_registration_ids.pop(instance, None)
            raise

    async def load_service(self, config: ServiceRawConfig) -> None:
        """Load service using the config, ignoring any persisted per-model entries.

        This service has no installable models - the container is created directly by
        `_install_instance`. A `services.json` written before that (when installing still went
        through a separate per-model step) may still list one; without dropping it here,
        `load_model` would treat it as a real failure and raise a permanent "failed to load"
        warning on every future startup instead of just quietly being obsolete.
        """
        instances = config.get("instances")
        if isinstance(instances, dict):
            for instance in instances.values():
                if isinstance(instance, dict):
                    instance["models"] = []
        await super().load_service(config)

    def get_type(self) -> str:
        """Return the service id."""
        return "finetune"

    def get_description(self) -> str:
        """Return the service description."""
        return "LLM LoRA fine-tuning framework with an OpenAI-compatible fine-tuning job queue."

    def get_size(self) -> ServiceSize:
        """Return the service size."""
        return "8.56GB"

    def get_spec(self) -> ServiceSpecification:
        """Return the service specification."""
        fields = self.add_hardware_field_to_spec()
        fields.extend(
            [
                ServiceField(
                    type="text",
                    name="prefix",
                    description="Endpoint prefix",
                    required=True,
                    placeholder=_FINETUNE_MODEL_ID,
                    default=_FINETUNE_MODEL_ID,
                ),
                ServiceField(
                    type="number",
                    name="proxy_timeout_seconds",
                    description="Seconds the proxy waits for a response from this service before timing out. "
                    "Leave empty to use the global default.",
                    required=False,
                ),
            ]
        )
        return ServiceSpecification(fields=fields)

    def get_custom_model_spec(self) -> CustomModelSpecification | None:
        """Return the custom model specification or None if custom model is not supported."""
        return None

    def get_installed_info(self, instance: str) -> bool | InstallServiceProgress | ServiceOptions:
        """Get service installed info."""
        installed = self.get_instance_info(instance).installed
        return self._get_service_installed_info(instance) if installed is None else installed.options.spec

    def get_installed_base_url(self, instance: str | None = None) -> str | None:
        """Live base URL of a running deepfellow-finetune container, or None if not installed.

        Without `instance`, returns the default instance's URL, falling back to any other installed
        instance - the adapters directory is shared, so every instance's registry serves every adapter.
        """
        if instance is not None:
            installed = self.get_instance_info(instance).installed
            return installed.base_url if installed else None
        candidates = sorted(self.instances_info.items(), key=lambda item: item[0] != "default")
        return next((info.installed.base_url for _, info in candidates if info.installed), None)

    async def list_adapters(self, instance: str = "default") -> FinetuneAdapterListOut:
        """List fine-tuned LoRA adapters from the running container's own `/v1/adapters`.

        Raises 409 when the service isn't installed (a state the caller must fix), 502 when the
        container answers with an error status, and 503 when it can't be reached or times out - so a
        client can tell a configuration problem from a container outage, and retry only the latter.
        """
        base_url = self.get_installed_base_url(instance)
        if not base_url:
            raise HTTPException(409, "deepfellow-finetune is not installed.")
        headers = Utils.create_bearer_header(self.config.adapter_registry_secret.get_secret_value())
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20), headers=headers) as session:
            try:
                async with session.get(f"{base_url.rstrip('/')}/v1/adapters") as resp:
                    resp.raise_for_status()
                    data = await resp.json()
            except aiohttp.ClientResponseError as exc:
                raise HTTPException(502, f"deepfellow-finetune at {base_url} returned HTTP {exc.status}: {exc.message}") from exc
            except (aiohttp.ClientError, TimeoutError) as exc:
                raise HTTPException(503, f"Could not reach deepfellow-finetune at {base_url}: {exc!r}") from exc
        return FinetuneAdapterListOut.model_validate(data)

    def get_docker_options(self, instance: str, model_id: str | None) -> DockerOptions:
        """Return the resolved DockerOptions for this instance."""
        if model_id:
            raise HTTPException(400, "Docker is not bound with this object")
        return self.get_instance_installed_info(instance).docker

    def service_has_docker(self) -> bool:
        """Return true when docker is started when service is installed."""
        return True

    async def stop_instance(self, instance: str) -> None:
        """Stop the deepfellow-finetune Docker container."""
        installed = self.get_instance_info(instance).installed
        if not installed:
            return
        await self._stop_docker(installed.docker)

    def _generate_instance_config(self, instance: str, info: InstalledInfo | None, custom: list[CustomModel] | None) -> InstanceConfig:  # noqa: ARG002
        return InstanceConfig(
            options=info.options if info else None,
            models=[],
            custom=custom,
            container_name=info.docker.container_name if info else None,
            registration_id=info.registration_id if info else None,
        )

    def _load_download_info(self, data: dict[str, Any]) -> DownloadedInfo:
        return DownloadedInfo(**data)

    def _get_shared_dir(self, name: str) -> Path:
        """Get a working subdirectory mounted by every instance's container."""
        path = self._get_working_dir() / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    async def _install_instance(
        self, instance: str, options: InstallServiceIn, container_name_override: str | None = None
    ) -> PromiseWithProgress[InstalledInfo, StreamChunk]:
        parsed_options = try_parse_pydantic(FinetuneOptions, options.spec)
        # Each instance keeps its own job database, so a shared prefix would route a job's
        # requests to whichever instance registered last
        for other, info in self.instances_info.items():
            if other != instance and info.installed and info.installed.parsed_options.prefix == parsed_options.prefix:
                raise HTTPException(400, f"Prefix {parsed_options.prefix!r} is already used by Finetune instance {other!r}.")
        hardware_parts = self.get_specified_hardware_parts(parsed_options.hardware)
        # Not popped: a rollback after a failed update calls _install_instance() again (bypassing install_instance(),
        # so nothing re-pins) and must re-register under the same id
        pinned_registration_id = self._pinned_registration_ids.get(instance) or str(uuid.uuid4())
        image = _FINETUNE_IMAGE
        await self._verify_docker_image(image, options.ignore_warnings)

        async def func(stream: Stream[StreamChunk]) -> InstalledInfo:
            self.service_downloaded = True
            stream.emit(StreamChunkProgress(type="progress", stage="install", value=0, data={}))
            subnet = self.docker_service.get_docker_subnet()
            name = self.get_service_id(instance)
            instance_dir = self.get_working_dir() / name
            # Created here rather than by the docker daemon on bind-mount, so they're owned by the same
            # uid the container runs as (see `user` below) and the worker can write to them
            for subdir in ("configs", "data", "db"):
                (instance_dir / subdir).mkdir(parents=True, exist_ok=True)
            # df-finetune (>= 0.0.19) requires a bearer token on every endpoint except /health. Infra
            # authenticates to it with the adapter registry secret: directly (`list_adapters`, the vLLM
            # adapter download) and via the proxy below. Without a secret, run it unauthenticated as before
            # (it's only reachable on the docker subnet and through infra's authenticated proxy).
            worker_token = self.config.adapter_registry_secret.get_secret_value()
            auth_env = {"DFFT_TOKEN": worker_token} if worker_token else {"DFFT_ALLOW_NOAUTH": "1"}
            docker_options = DockerOptions(
                image_port=8333,
                name=name,
                container_name=container_name_override or self.docker_service.get_docker_container_name(name),
                image=image,
                restart="unless-stopped",
                volumes=[
                    f"{self._get_shared_dir('models')}:/workspace/models",
                    f"{self._get_shared_dir('adapters')}:/workspace/adapters",
                    f"{instance_dir}/configs:/workspace/configs",
                    f"{instance_dir}/data:/workspace/data",
                    f"{instance_dir}/db:/data",
                ],
                hardware=hardware_parts,
                subnet=subnet,
                # The image runs as uid 1000 by default; run as infra's own user instead so the container
                # can write the bind-mounted dirs infra creates (and infra can delete what it writes)
                user=await self.docker_service.get_user_for_docker(),
                env_vars={
                    "HF_TOKEN": self.config.hugging_face_token.get_secret_value(),
                    "DFFT_DB_PATH": "/data/worker.db",
                    **auth_env,
                },
                healthcheck={
                    "test": "wget -q --spider http://localhost:8333/health",
                    "interval": "30s",
                    "timeout": "10s",
                    "retries": "3",
                    "start_period": "30s",
                },
            )
            docker_exposed_port, _, _ = await self.docker_service.install_and_run_docker(docker_options)
            container_host = self.docker_service.get_container_host(subnet, docker_options)
            container_port = self.docker_service.get_container_port(subnet, docker_exposed_port, docker_options.image_port)
            base_url = get_base_url(container_host, container_port)
            proxy_timeout_seconds = (
                parsed_options.proxy_timeout_seconds
                if parsed_options.proxy_timeout_seconds is not None
                else self.config.standard_proxy_timeout_seconds
            )
            registration_id = self.endpoint_registry.register_custom_endpoint_as_proxy(
                url=parsed_options.prefix,
                props=ModelProps(private=True, type="custom", endpoints=[f"/custom/{parsed_options.prefix}"]),
                # The proxy drops the client's Authorization (infra's own API key); authenticate to the worker instead
                options=ProxyOptions(
                    url=base_url,
                    headers=Utils.create_bearer_header(worker_token),
                    read_timeout_seconds=proxy_timeout_seconds,
                ),
                registration_options=RegistrationOptions(origin="local", id=pinned_registration_id, owned_by=self.get_type()),
            )
            stream.emit(StreamChunkProgress(type="progress", stage="install", value=1, data={}))
            return InstalledInfo(
                docker=docker_options,
                options=options,
                parsed_options=parsed_options,
                container_host=container_host,
                container_port=container_port,
                docker_exposed_port=docker_exposed_port,
                registration_id=registration_id,
                base_url=base_url,
            )

        return PromiseWithProgress(func=func)

    async def _uninstall_instance(self, instance: str, options: UninstallServiceIn) -> None:
        installed = self.get_instance_info(instance).installed
        if installed:
            # Intentional: update_instance() calls _uninstall_instance()/_install_instance() directly, bypassing install_instance(),
            # so this pin is what keeps the registration id across an update; install_instance() re-sets it, so it never goes stale
            self._pinned_registration_ids[instance] = installed.registration_id
            self.endpoint_registry.unregister_custom_endpoint(installed.parsed_options.prefix, installed.registration_id)
            await self.docker_service.uninstall_docker(installed.docker)

        self.instances_info[instance].installed = None

        if options.purge:
            if not any(i.installed for i in self.instances_info.values()):
                self.service_downloaded = False
                await self.docker_service.remove_image(_FINETUNE_IMAGE)
                await self._clear_working_dir()
                self.models_downloaded = {}

            if instance == "default":
                self.instances_info["default"] = Instance(None, None, {}, InstanceConfig())
            else:
                del self.instances_info[instance]

    def get_working_dir(self) -> Path:
        """Get working dir."""
        return self._get_working_dir()

    async def list_models(self, input_instance: str | list[str] | None, filters: ListModelsFilters) -> ListModelsOut:  # noqa: ARG002
        """List models. This service has no installable models - the container itself is the whole service."""
        return ListModelsOut(list=[])

    async def get_model(self, instance: str, model_id: str) -> RetrieveModelOut:  # noqa: ARG002
        """Raise - this service has no installable models."""
        raise HTTPException(400, "This service has no installable models.")

    async def _install_model(
        self,
        instance: str,  # noqa: ARG002
        model_id: str,  # noqa: ARG002
        options: InstallModelIn,  # noqa: ARG002
        container_name_override: str | None = None,  # noqa: ARG002
    ) -> PromiseWithProgress[InstallModelOut, StreamChunk]:
        raise HTTPException(400, "This service has no installable models.")

    async def _uninstall_model(self, instance: str, model_id: str, options: UninstallModelIn) -> None:  # noqa: ARG002
        raise HTTPException(400, "This service has no installable models.")


def get_finetune_service_instance(
    service_id: Annotated[str, ApiPath(description="The ID of the Finetune service instance to use.")],
    services_manager: Annotated[ServicesManager, Depends(get_services_manager)],
) -> tuple["FinetuneService", str]:
    """Resolve `service_id` to the Finetune service and its instance name."""
    service_type, instance = services_manager.split_service_type_and_instance(service_id)
    service = services_manager.services.get(service_type)
    if not isinstance(service, FinetuneService):
        raise HTTPException(404, f"Service {service_id} is not a Finetune service")
    return service, instance
