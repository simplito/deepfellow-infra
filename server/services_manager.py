# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Services manager."""

import asyncio
import logging
import re

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
from server.services.base_service import BaseService
from server.utils.core import PromiseWithProgress, StreamChunk

logger = logging.getLogger("uvicorn.error")


class ServicesManager:
    def __init__(self):
        self.services: dict[str, BaseService] = {}

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

    async def load_service(self, service_id: str, service_cfg: ServiceRawConfig) -> None:
        """Load service."""
        if service_id in self.services:
            await self.services[service_id].load_service(service_cfg)

    def _get_service(self, service_id: str) -> BaseService:
        """Retrieve service."""
        if service_id in self.services:
            return self.services[service_id]
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
                    res.append(await service.list_models(instance, filter))
        return ListAllModelsOut(list=[x for sublist in res for x in sublist.list])

    async def list_models_from_service(self, service_id: str, filters: ListModelsFilters) -> ListModelsOut:
        """List models."""
        service_type, instance = self.split_service_type_and_instance(service_id)
        return await self._get_service(service_type).list_models(instance, filters)

    async def get_model_from_service(self, service_id: str, model_id: str) -> RetrieveModelOut:
        """Get the model from service."""
        service_type, instance = self.split_service_type_and_instance(service_id)
        return await self._get_service(service_type).get_model(instance, model_id)

    async def get_model_install_progress(self, service_id: str, model_id: str) -> PromiseWithProgress[InstallModelOut, StreamChunk]:
        """Get model install progress."""
        service_type, instance = self.split_service_type_and_instance(service_id)
        return self._get_service(service_type).get_model_install_progress(instance, model_id)

    async def get_service_install_progress(self, service_id: str) -> PromiseWithProgress[InstallServiceOut, StreamChunk]:
        """Get modelinstall progress."""
        service_type, instance = self.split_service_type_and_instance(service_id)
        return self._get_service(service_type).get_instance_install_progress(instance)

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
