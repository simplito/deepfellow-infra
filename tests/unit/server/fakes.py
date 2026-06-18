# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any
from unittest.mock import MagicMock

from server.models.models import (
    AddCustomModelIn,
    CustomModelId,
    InstallModelIn,
    InstallModelOut,
    ListModelsFilters,
    ListModelsOut,
    UninstallModelIn,
)
from server.models.services import (
    InstallServiceIn,
    InstallServiceOut,
    ServiceSpecification,
    UninstallServiceIn,
)
from server.serviceprovider import ServiceRawConfig
from server.services.base_service import BaseService
from server.utils.core import PromiseWithProgress, StreamChunk


class FakeService(BaseService):
    """Minimal concrete BaseService for use in tests.

    Implements all abstract methods with safe no-op defaults. Override individual
    methods with AsyncMock/MagicMock in each test for behaviour-specific assertions.
    """

    instances_info: dict[str, Any]

    def __init__(self, service_type: str, instances: list[str] | None = None, installed: bool = True) -> None:
        self._type = service_type
        self.instances_info = {inst: {} for inst in (instances or ["default"])}
        self._installed = installed

    def get_type(self) -> str:
        return self._type

    def get_description(self) -> str:
        return "fake"

    def get_size(self) -> Any:
        return "small"

    def get_spec(self) -> ServiceSpecification:
        return ServiceSpecification(fields=[])

    def get_custom_model_spec(self) -> None:
        return None

    def get_instance_install_progress(self, instance: str) -> PromiseWithProgress[InstallServiceOut, StreamChunk]:
        return MagicMock()

    def get_model_install_progress(self, instance: str, model: str) -> PromiseWithProgress[InstallModelOut, StreamChunk]:
        return MagicMock()

    def is_installed(self, instance: str) -> bool:
        return self._installed

    def get_installed_info(self, instance: str) -> bool:
        return self._installed

    def get_downloaded(self) -> bool:
        return True

    async def load_service(self, config: ServiceRawConfig) -> None:
        pass

    async def install_instance(self, instance: str, options: InstallServiceIn) -> PromiseWithProgress[InstallServiceOut, StreamChunk]:
        return MagicMock()

    async def update_instance(self, instance: str, options: InstallServiceIn) -> PromiseWithProgress[InstallServiceOut, StreamChunk]:
        return MagicMock()

    async def uninstall_instance(self, instance: str, options: UninstallServiceIn) -> None:
        pass

    async def list_models(self, input_instance: Any, filters: ListModelsFilters) -> ListModelsOut:
        return ListModelsOut(list=[])

    async def get_model(self, instance: str, model_id: str) -> Any:
        return MagicMock()

    async def install_model(
        self, instance: str, model_id: str, options: InstallModelIn
    ) -> PromiseWithProgress[InstallModelOut, StreamChunk]:
        return MagicMock()

    async def uninstall_model(self, instance: str, model_id: str, options: UninstallModelIn) -> None:
        pass

    async def add_custom_model(self, instance: str, options: AddCustomModelIn) -> CustomModelId:
        return "custom-id"

    async def remove_custom_model(self, instance: str, custom_model_id: CustomModelId) -> None:
        pass

    async def get_docker_logs(self, instance: str, model_id: str | None) -> str:
        return "logs"

    async def get_docker_compose_file(self, instance: str, model_id: str | None) -> str:
        return "compose"

    async def restart_docker(self, instance: str, model_id: str | None) -> None:
        pass

    async def stop_instance(self, instance: str) -> None:
        pass
