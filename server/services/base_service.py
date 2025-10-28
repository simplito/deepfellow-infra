"""Base service."""

from abc import ABC, abstractmethod

from server.models.models import (
    AddCustomModelIn,
    CustomModelId,
    CustomModelSpecification,
    InstallModelIn,
    ListModelsFilters,
    ListModelsOut,
    RetrieveModelOut,
    UninstallModelIn,
)
from server.models.services import (
    InstallServiceIn,
    RetrieveServiceOut,
    ServiceOptions,
    ServiceSize,
    ServiceSpecification,
    UninstallServiceIn,
)
from server.serviceprovider import ServiceRawConfig


class BaseService(ABC):
    @abstractmethod
    def get_id(self) -> str:
        """Return the service id."""

    @abstractmethod
    def get_size(self) -> ServiceSize:
        """Return the service size."""

    @abstractmethod
    def get_spec(self) -> ServiceSpecification:
        """Return the service specification."""

    @abstractmethod
    def get_custom_model_spec(self) -> CustomModelSpecification | None:
        """Return the custom model specification or None if custom model is not supported."""

    def get_info(self) -> RetrieveServiceOut:
        """Return the service info."""
        return RetrieveServiceOut(
            id=self.get_id(),
            installed=self.get_installed_info(),
            spec=self.get_spec(),
            size=self.get_size(),
            custom_model_spec=self.get_custom_model_spec(),
            has_docker=self.service_has_docker(),
        )

    @abstractmethod
    def is_installed(self) -> bool:
        """Check whether service is installed."""

    @abstractmethod
    def get_installed_info(self) -> bool | ServiceOptions:
        """Get service installed info."""

    def service_has_docker(self) -> bool:
        """Return true when docker is started when service is installed."""
        return False

    @abstractmethod
    async def load(self, config: ServiceRawConfig) -> None:
        """Load service using the config."""

    @abstractmethod
    async def install(self, options: InstallServiceIn) -> None:
        """Install the service."""

    @abstractmethod
    async def uninstall(self, options: UninstallServiceIn) -> None:
        """Uninstall the service."""

    @abstractmethod
    async def list_models(self, filters: ListModelsFilters) -> ListModelsOut:
        """List models."""

    @abstractmethod
    async def get_model(self, model_id: str) -> RetrieveModelOut:
        """Get the model."""

    @abstractmethod
    async def install_model(self, model_id: str, options: InstallModelIn) -> None:
        """Install the model."""

    @abstractmethod
    async def uninstall_model(self, model_id: str, options: UninstallModelIn) -> None:
        """Uninstall the model."""

    @abstractmethod
    async def add_custom_model(self, options: AddCustomModelIn) -> CustomModelId:
        """Add custom model."""

    @abstractmethod
    async def remove_custom_model(self, custom_model_id: CustomModelId) -> None:
        """Remove custom model."""

    @abstractmethod
    async def get_docker_logs(self, model_id: str | None) -> str:
        """Get docker logs."""

    @abstractmethod
    async def get_docker_compose_file(self, model_id: str | None) -> str:
        """Get docker compose file."""

    @abstractmethod
    async def restart_docker(self, model_id: str | None) -> None:
        """Get docker compose file."""
