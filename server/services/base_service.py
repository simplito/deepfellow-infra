# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Base service."""

from abc import ABC, abstractmethod
from typing import Any

from server.models.models import (
    AddCustomModelIn,
    CustomModelId,
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
    RetrieveServiceOut,
    ServiceOptions,
    ServiceSize,
    ServiceSpecification,
    UninstallServiceIn,
)
from server.serviceprovider import ServiceRawConfig
from server.utils.core import PromiseWithProgress, StreamChunk


class BaseService(ABC):
    instances_info: dict[str, Any]

    def get_id(self, instance: str) -> str:
        """Return the service id."""
        if instance == "default":
            return self.get_type()
        return f"{self.get_type()}|{instance}"

    def get_service_id(self, instance: str) -> str:
        """Return the service id in form usable in a filesystem path name."""
        if instance == "default":
            return self.get_type()
        return f"{self.get_type()}-{instance}"

    @abstractmethod
    def get_type(self) -> str:
        """Return the service type."""

    @abstractmethod
    def get_description(self) -> str:
        """Return the service description."""

    @abstractmethod
    def get_size(self) -> ServiceSize:
        """Return the service size."""

    @abstractmethod
    def get_spec(self) -> ServiceSpecification:
        """Return the service specification."""

    @abstractmethod
    def get_custom_model_spec(self) -> CustomModelSpecification | None:
        """Return the custom model specification or None if custom model is not supported."""

    def get_info(self, instance: str) -> RetrieveServiceOut:
        """Return the service info."""
        return RetrieveServiceOut(
            id=self.get_id(instance),
            type=self.get_type(),
            instance=instance,
            description=self.get_description(),
            installed=self.get_installed_info(instance),
            downloaded=self.get_downloaded(),
            spec=self.get_spec(),
            size=self.get_size(),
            custom_model_spec=self.get_custom_model_spec(),
            has_docker=self.service_has_docker(),
        )

    @abstractmethod
    def get_instance_install_progress(self, instance: str) -> PromiseWithProgress[InstallServiceOut, StreamChunk]:
        """Return actually installing instance."""

    @abstractmethod
    def get_model_install_progress(self, instance: str, model: str) -> PromiseWithProgress[InstallModelOut, StreamChunk]:
        """Return actually installing models."""

    @abstractmethod
    def is_installed(self, instance: str) -> bool:
        """Check whether instance is installed."""

    @abstractmethod
    def get_installed_info(self, instance: str) -> bool | InstallServiceProgress | ServiceOptions:
        """Get service installed info."""

    @abstractmethod
    def get_downloaded(self) -> bool:
        """Get service downloaded info."""

    def service_has_docker(self) -> bool:
        """Return true when docker is started when service is installed."""
        return False

    @abstractmethod
    async def load_service(self, config: ServiceRawConfig) -> None:
        """Load service using the config."""

    @abstractmethod
    async def install_instance(self, instance: str, options: InstallServiceIn) -> PromiseWithProgress[InstallServiceOut, StreamChunk]:
        """Install the service."""

    @abstractmethod
    async def uninstall_instance(self, instance: str, options: UninstallServiceIn) -> None:
        """Uninstall the instance."""

    @abstractmethod
    async def list_models(self, input_instance: str | list[str] | None, filters: ListModelsFilters) -> ListModelsOut:
        """List models."""

    @abstractmethod
    async def get_model(self, instance: str, model_id: str) -> RetrieveModelOut:
        """Get the model."""

    @abstractmethod
    async def install_model(
        self, instance: str, model_id: str, options: InstallModelIn
    ) -> PromiseWithProgress[InstallModelOut, StreamChunk]:
        """Install the model."""

    @abstractmethod
    async def uninstall_model(self, instance: str, model_id: str, options: UninstallModelIn) -> None:
        """Uninstall the model."""

    @abstractmethod
    async def add_custom_model(self, instance: str, options: AddCustomModelIn) -> CustomModelId:
        """Add custom model."""

    @abstractmethod
    async def remove_custom_model(self, instance: str, custom_model_id: CustomModelId) -> None:
        """Remove custom model."""

    @abstractmethod
    async def get_docker_logs(self, instance: str, model_id: str | None) -> str:
        """Get docker logs."""

    @abstractmethod
    async def get_docker_compose_file(self, instance: str, model_id: str | None) -> str:
        """Get docker compose file."""

    @abstractmethod
    async def restart_docker(self, instance: str, model_id: str | None) -> None:
        """Get docker compose file."""

    @abstractmethod
    async def stop_instance(self, instance: str) -> None:
        """Stop the service gracefully.

        This method is called during application shutdown when DF_STOP_CONTAINERS_ON_SHUTDOWN is enabled.
        Services with Docker containers should stop their containers here.
        Services without Docker (proxies, external services) can leave this as a no-op.
        """
