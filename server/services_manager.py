# DeepFellow Software Framework.
# Copyright © 2025 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Services manager."""

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


class ServicesManager:
    def __init__(
        self,
    ):
        self.services: dict[str, BaseService] = {}

    def register_service(self, service: BaseService) -> None:
        """Register service."""
        service_id = service.get_id()
        if service_id in self.services:
            raise RuntimeError("Service already registered", service_id)
        self.services[service_id] = service

    async def load_service(self, service_id: str, service_cfg: ServiceRawConfig) -> None:
        """Load service."""
        if service_id in self.services:
            await self.services[service_id].load(service_cfg)

    def _get_service(self, service_id: str) -> BaseService:
        """Retrieve service."""
        if service_id in self.services:
            return self.services[service_id]
        raise HTTPException(status_code=404, detail=f"Service does not exist {service_id}")

    async def list_services(self, filters: ListServicesFilters) -> ListServicesOut:
        """List services."""
        list = [v.get_info() for _, v in self.services.items() if filters.installed is None or filters.installed == v.is_installed()]
        return ListServicesOut(list=list)

    async def get_service(self, service_id: str) -> RetrieveServiceOut:
        """Get the service."""
        return self._get_service(service_id).get_info()

    async def install_service(self, service_id: str, options: InstallServiceIn) -> PromiseWithProgress[InstallServiceOut, StreamChunk]:
        """Install the service."""
        return await self._get_service(service_id).install(options)

    async def uninstall_service(self, service_id: str, options: UninstallServiceIn) -> None:
        """Uninstall the service."""
        await self._get_service(service_id).uninstall(options)

    async def list_models_from_all_services(self, filters: ListAllModelsFilters) -> ListAllModelsOut:
        """List models from all services."""
        print("list_models_from_all_services", filters)
        filter = ListModelsFilters(installed=filters.installed)
        res: list[ListModelsOut] = []
        for service in self.services.values():
            if service.is_installed() and (filters.service_id is None or service.get_id() == filters.service_id):
                res.append(await service.list_models(filter))
        return ListAllModelsOut(list=[x for sublist in res for x in sublist.list])

    async def list_models_from_service(self, service_id: str, filters: ListModelsFilters) -> ListModelsOut:
        """List models."""
        return await self._get_service(service_id).list_models(filters)

    async def get_model_from_service(self, service_id: str, model_id: str) -> RetrieveModelOut:
        """Get the model from service."""
        return await self._get_service(service_id).get_model(model_id)

    async def install_model_in_service(
        self,
        service_id: str,
        model_id: str,
        options: InstallModelIn,
    ) -> PromiseWithProgress[InstallModelOut, StreamChunk]:
        """Install the model in service."""
        return await self._get_service(service_id).install_model(model_id, options)

    async def uninstall_model_from_service(self, service_id: str, model_id: str, options: UninstallModelIn) -> None:
        """Uninstall the model from service."""
        await self._get_service(service_id).uninstall_model(model_id, options)

    async def add_custom_model(self, service_id: str, options: AddCustomModelIn) -> CustomModelId:
        """Add custom model."""
        return await self._get_service(service_id).add_custom_model(options)

    async def remove_custom_model(self, service_id: str, custom_model_id: CustomModelId) -> None:
        """Remove custom model."""
        return await self._get_service(service_id).remove_custom_model(custom_model_id)

    async def get_docker_logs(self, service_id: str, model_id: str | None) -> str:
        """Get docker logs."""
        return await self._get_service(service_id).get_docker_logs(model_id)

    async def get_docker_compose_file(self, service_id: str, model_id: str | None) -> str:
        """Get docker compose file."""
        return await self._get_service(service_id).get_docker_compose_file(model_id)

    async def restart_docker(self, service_id: str, model_id: str | None) -> None:
        """Get docker compose file."""
        return await self._get_service(service_id).restart_docker(model_id)
