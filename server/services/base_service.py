"""Base service."""

from abc import ABC, abstractmethod

from server.models.models import InstallModelIn, ListModelsFilters, ListModelsOut, RetrieveModelOut, UninstallModelIn
from server.models.services import InstallServiceIn, RetrieveServiceOut, UninstallServiceIn
from server.serviceprovider import ServiceRawConfig


class BaseService(ABC):
    @abstractmethod
    def get_id(self) -> str:
        """Return the service id."""

    def get_info(self) -> RetrieveServiceOut:
        """Return the service info."""
        return RetrieveServiceOut(id=self.get_id(), installed=self.is_installed())

    @abstractmethod
    def is_installed(self) -> bool:
        """Check whether service is installed."""

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
