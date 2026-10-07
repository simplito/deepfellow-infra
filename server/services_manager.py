# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Services manager."""

import asyncio
import logging
import re
from typing import Any

from fastapi import HTTPException

from server.models.models import (
    AddCustomModelIn,
    CustomModelId,
    InstallModelIn,
    InstallModelOut,
    ListModelsFilters,
    ListModelsOut,
    RetrieveModelOut,
    UninstallModelIn,
)
from server.models.services import (
    CatalogRefreshOut,
    InstallServiceIn,
    InstallServiceOut,
    ListAllModelsFilters,
    ListAllModelsOut,
    ListServicesFilters,
    ListServicesOut,
    RetrieveServiceOut,
    UninstallServiceIn,
)
from server.serviceprovider import ServiceRawConfig
from server.services.base2_service import Base2Service
from server.services.base_service import BaseService
from server.utils.core import PromiseWithProgress, StreamChunk

logger = logging.getLogger("uvicorn.error")


class ServicesManager:
    EXTERNAL_ONLY_ALLOWED_SERVICE_TYPES: frozenset[str] = frozenset({"openai"})

    def __init__(self, external_only: bool = False):
        self.services: dict[str, BaseService] = {}
        self.external_only = external_only
        self.docker_tags_cache: dict[tuple[str, str | None, str | None, str | None], tuple[list[str], str, float]] = {}

    def split_service_type_and_instance(self, service_id: str) -> tuple[str, str]:
        """Convert service id to service type and instance with validation."""
        service_id_parts = service_id.split("|")
        len_service_parts = len(service_id_parts)

        if len_service_parts == 2:
            true_service_id, instance = service_id_parts
        elif len_service_parts == 1:
            true_service_id, instance = service_id, "default"
        else:
            raise HTTPException(404, "Incorrect service_id format")

        # Validation: Alphanumeric, underscores, and dashes only
        # ^[a-zA-Z0-9_-]+$ ensures the entire string matches these criteria
        for value, label in [(true_service_id, "service_id"), (instance, "instance")]:
            # Check Character Limit
            length = 64
            if len(value) > length:
                raise HTTPException(400, f"{label} exceeds maximum length of {length} characters")
            # Check Characters (Regex)
            pattern = r"^[a-zA-Z0-9_-]+$"
            if not re.match(pattern, value):
                raise HTTPException(400, f"{label} contains invalid characters (only alphanumeric, _, and - allowed)")

        return true_service_id, instance

    def register_service(self, service: BaseService) -> None:
        """Register service."""
        service_id = service.get_type()
        if service_id in self.services:
            raise RuntimeError("Service already registered", service_id)
        self.services[service_id] = service

    async def drain_warning_tasks(self) -> None:
        """Await every service's pending fire-and-forget warning writes so shutdown doesn't drop them."""
        drains = [service.drain_warning_tasks() for service in self.services.values() if isinstance(service, Base2Service)]
        if drains:
            await asyncio.gather(*drains)

    async def load_service(self, service_id: str, service_cfg: ServiceRawConfig) -> None:
        """Load service."""
        if service_id in self.services:
            await self.services[service_id].load_service(service_cfg)

    def _get_service(self, service_id: str) -> BaseService:
        """Retrieve service."""
        if service_id in self.services:
            return self.services[service_id]
        if self.external_only and service_id not in self.EXTERNAL_ONLY_ALLOWED_SERVICE_TYPES:
            raise HTTPException(status_code=409, detail=f"Service '{service_id}' is disabled by DF_EXTERNAL_ONLY")
        raise HTTPException(status_code=404, detail=f"Service does not exist {service_id}")

    async def list_services(self, filters: ListServicesFilters) -> ListServicesOut:
        """List services."""
        list = []
        for _, v in self.services.items():
            for instance in v.instances_info:
                if filters.installed is None or filters.installed == v.is_installed(instance):
                    list.append(v.get_info(instance))

        return ListServicesOut(list=list)

    async def get_service(self, service_id: str) -> RetrieveServiceOut:
        """Get the service."""
        service_type, instance = self.split_service_type_and_instance(service_id)
        return self._get_service(service_type).get_info(instance)

    async def install_service(self, service_id: str, options: InstallServiceIn) -> PromiseWithProgress[InstallServiceOut, StreamChunk]:
        """Install the service."""
        service_type, instance = self.split_service_type_and_instance(service_id)
        return await self._get_service(service_type).install_instance(instance, options)

    async def update_service(self, service_id: str, options: InstallServiceIn) -> PromiseWithProgress[InstallServiceOut, StreamChunk]:
        """Update an installed service's options."""
        service_type, instance = self.split_service_type_and_instance(service_id)
        return await self._get_service(service_type).update_instance(instance, options)

    async def uninstall_service(self, service_id: str, options: UninstallServiceIn) -> None:
        """Uninstall the service."""
        service_type, instance = self.split_service_type_and_instance(service_id)
        await self._get_service(service_type).uninstall_instance(instance, options)

    async def list_models_from_all_services(self, filters: ListAllModelsFilters) -> ListAllModelsOut:
        """List models from all services."""
        filter = ListModelsFilters(installed=filters.installed)
        res: list[ListModelsOut] = []
        for service in self.services.values():
            for instance in service.instances_info:
                if service.is_installed(instance) and (filters.service_id is None or service.get_type() == filters.service_id):
                    out = await service.list_models(instance, filter)
                    for model in out.list:
                        self._enrich_custom_spec(service, model)
                    res.append(out)
        return ListAllModelsOut(list=[x for sublist in res for x in sublist.list])

    async def list_models_from_service(self, service_id: str, filters: ListModelsFilters) -> ListModelsOut:
        """List models."""
        service_type, instance = self.split_service_type_and_instance(service_id)
        service = self._get_service(service_type)
        out = await service.list_models(instance, filters)
        for model in out.list:
            self._enrich_custom_spec(service, model)
        return out

    async def get_model_from_service(self, service_id: str, model_id: str) -> RetrieveModelOut:
        """Get the model from service."""
        service_type, instance = self.split_service_type_and_instance(service_id)
        service = self._get_service(service_type)
        model = await service.get_model(instance, model_id)
        self._enrich_custom_spec(service, model)
        return model

    @staticmethod
    def _enrich_custom_spec(service: BaseService, model: RetrieveModelOut) -> None:
        """Fill custom_spec for a custom model from its stored definition, unless the service already set it."""
        if model.custom and model.custom_spec is None:
            model.custom_spec = service.get_custom_model_definition(model.custom)

    async def get_model_install_progress(self, service_id: str, model_id: str) -> PromiseWithProgress[InstallModelOut, StreamChunk]:
        """Get model install progress."""
        service_type, instance = self.split_service_type_and_instance(service_id)
        return await self._get_service(service_type).get_model_install_progress(instance, model_id)

    async def cancel_model_install(self, service_id: str, model_id: str) -> None:
        """Cancel an in-progress model install."""
        service_type, instance = self.split_service_type_and_instance(service_id)
        await self._get_service(service_type).cancel_model_install(instance, model_id)

    async def get_service_install_progress(self, service_id: str) -> PromiseWithProgress[InstallServiceOut, StreamChunk]:
        """Get modelinstall progress."""
        service_type, instance = self.split_service_type_and_instance(service_id)
        return await self._get_service(service_type).get_instance_install_progress(instance)

    async def cancel_service_install(self, service_id: str) -> None:
        """Cancel an in-progress service instance install or update."""
        service_type, instance = self.split_service_type_and_instance(service_id)
        await self._get_service(service_type).cancel_instance_install(instance)

    async def install_model_in_service(
        self,
        service_id: str,
        model_id: str,
        options: InstallModelIn,
    ) -> PromiseWithProgress[InstallModelOut, StreamChunk]:
        """Install the model in service."""
        service_type, instance = self.split_service_type_and_instance(service_id)
        return await self._get_service(service_type).install_model(instance, model_id, options)

    async def uninstall_model_from_service(self, service_id: str, model_id: str, options: UninstallModelIn) -> None:
        """Uninstall the model from service."""
        service_type, instance = self.split_service_type_and_instance(service_id)
        await self._get_service(service_type).uninstall_model(instance, model_id, options)

    async def add_custom_model(self, service_id: str, options: AddCustomModelIn) -> CustomModelId:
        """Add custom model."""
        service_type, instance = self.split_service_type_and_instance(service_id)
        return await self._get_service(service_type).add_custom_model(instance, options)

    async def remove_custom_model(self, service_id: str, custom_model_id: CustomModelId) -> None:
        """Remove custom model."""
        service_type, instance = self.split_service_type_and_instance(service_id)
        return await self._get_service(service_type).remove_custom_model(instance, custom_model_id)

    async def update_custom_model(self, service_id: str, custom_model_id: CustomModelId, options: AddCustomModelIn) -> None:
        """Update custom model."""
        service_type, instance = self.split_service_type_and_instance(service_id)
        return await self._get_service(service_type).update_custom_model(instance, custom_model_id, options)

    async def edit_model(
        self, service_id: str, custom_model_id: CustomModelId, new_definition: AddCustomModelIn
    ) -> PromiseWithProgress[InstallModelOut, StreamChunk] | None:
        """Edit a custom-backed model's definition, uninstalling and reinstalling it if installed."""
        service_type, instance = self.split_service_type_and_instance(service_id)
        return await self._get_service(service_type).edit_model(instance, custom_model_id, new_definition)

    async def edit_model_install_options(
        self, service_id: str, model_id: str, new_options: InstallModelIn
    ) -> tuple[bool, PromiseWithProgress[InstallModelOut, StreamChunk]]:
        """Edit a model's install-time options only, reinstalling it if installed."""
        service_type, instance = self.split_service_type_and_instance(service_id)
        return await self._get_service(service_type).edit_model_install_options(instance, model_id, new_options)

    async def get_duplicate_spec(self, service_id: str, model_id: str) -> dict[str, Any]:
        """Return a full add-model spec synthesized from an existing model, for duplication."""
        service_type, instance = self.split_service_type_and_instance(service_id)
        return await self._get_service(service_type).get_duplicate_spec(instance, model_id)

    async def sync_models_in_service(self, service_id: str) -> None:
        """Trigger immediate model sync for the service instance."""
        service_type, instance = self.split_service_type_and_instance(service_id)
        await self._get_service(service_type).sync_models(instance)

    async def refresh_catalog(self, service_id: str) -> PromiseWithProgress[CatalogRefreshOut, StreamChunk]:
        """Refresh the model catalog from the external library API.

        Raises HTTPException 405 if the service does not support catalog refresh.
        """
        service_type, _instance = self.split_service_type_and_instance(service_id)
        return await self._get_service(service_type).refresh_catalog()

    async def get_docker_tags_for_service(self, service_id: str, hardware: str | None, model_id: str | None = None) -> list[str]:
        """Fetch available Docker image tags for the service, hardware-filtered and optionally model-scoped."""
        if self.external_only:
            raise HTTPException(status_code=409, detail="Docker image tags are disabled by DF_EXTERNAL_ONLY")
        service_type, _instance = self.split_service_type_and_instance(service_id)
        return await self._get_service(service_type).get_docker_tags_for_model(model_id, hardware)

    def get_default_docker_tag_for_service(self, service_id: str, hardware: str | None, model_id: str | None = None) -> str | None:
        """Return the pinned default Docker image tag for the service, hardware-aware and optionally model-scoped."""
        service_type, _instance = self.split_service_type_and_instance(service_id)
        return self._get_service(service_type).get_default_docker_tag_for_model(model_id, hardware)

    def get_docker_image_repo_for_service(self, service_id: str, hardware: str | None, model_id: str | None = None) -> str | None:
        """Return the Docker repo tags are fetched from for the service, hardware-aware and optionally model-scoped.

        None if the service's tags come from the same repo regardless of hardware.
        """
        service_type, _instance = self.split_service_type_and_instance(service_id)
        return self._get_service(service_type).get_docker_image_repo_for_model(model_id, hardware)

    async def get_docker_logs(self, service_id: str, model_id: str | None) -> str:
        """Get docker logs."""
        service_type, instance = self.split_service_type_and_instance(service_id)
        return await self._get_service(service_type).get_docker_logs(instance, model_id)

    async def get_docker_compose_file(self, service_id: str, model_id: str | None) -> str:
        """Get docker compose file."""
        service_type, instance = self.split_service_type_and_instance(service_id)
        return await self._get_service(service_type).get_docker_compose_file(instance, model_id)

    async def restart_docker(self, service_id: str, model_id: str | None) -> None:
        """Get docker compose file."""
        service_type, instance = self.split_service_type_and_instance(service_id)
        return await self._get_service(service_type).restart_docker(instance, model_id)

    async def stop_all_services(self) -> None:
        """Stop all installed services gracefully."""

        async def stop_service(service_id: str, service: BaseService) -> None:
            if service.instances_info:
                for instance in service.instances_info:
                    if service.is_installed(instance):
                        try:
                            logger.info("Stopping service: %s", service_id)
                            await service.stop_instance(instance)
                            logger.info("Successfully stopped service: %s", service_id)
                        except Exception:
                            logger.exception("Error stopping service %s", service_id)

        logger.info("Shutting down: stopping services...")
        tasks = [asyncio.create_task(stop_service(service_id, service)) for service_id, service in self.services.items()]
        await asyncio.gather(*tasks)
        logger.info("All services stopped")
