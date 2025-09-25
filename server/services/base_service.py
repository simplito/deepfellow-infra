"""Base service."""

from abc import ABC, abstractmethod

from server.models.models import InstallModelIn, ListModelsFilters, ListModelsOut, RetrieveModelOut, UninstallModelIn
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

    def get_info(self) -> RetrieveServiceOut:
        """Return the service info."""
        return RetrieveServiceOut(id=self.get_id(), installed=self.get_installed_info(), spec=self.get_spec(), size=self.get_size())

    @abstractmethod
    def is_installed(self) -> bool:
        """Check whether service is installed."""

    @abstractmethod
    def get_installed_info(self) -> bool | ServiceOptions:
        """Get service installed info."""

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
