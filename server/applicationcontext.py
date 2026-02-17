# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Application Content module."""

import asyncio
import logging
import time

from server.config import AppSettings
from server.services_manager import ServicesManager

from .endpointregistry import EndpointRegistry
from .serviceprovider import ServiceProvider, ServiceRawConfig

logger = logging.getLogger("uvicorn.error")


class ApplicationContext:
    def __init__(
        self,
        endpoint_registry: EndpointRegistry,
        config: AppSettings,
        service_provider: ServiceProvider,
        services_manager: ServicesManager,
    ):
        self.endpoint_registry = endpoint_registry
        self.config = config
        self.service_provider = service_provider
        self.services_manager = services_manager
        self.allocated_ports = set[int]()

    async def _load_service(self, service_id: str, service_cfg: ServiceRawConfig) -> None:
        """Load single service."""
        start = time.time()
        logger.info(f"{service_id} loading...")  # noqa: G004
        try:
            await self.services_manager.load_service(service_id, service_cfg)
            logger.info(f"{service_id} fully loaded in {round(time.time() - start, 1)}s")  # noqa: G004
        except Exception:
            logger.exception(f"{service_id} error occurs during loading {round(time.time() - start, 1)}s")  # noqa: G004

    async def load_services(self) -> None:
        """Load all service from bootstrap."""
        info = await self.service_provider.load()
        tasks = [asyncio.create_task(self._load_service(service_id, service_cfg)) for service_id, service_cfg in info["services"].items()]
        await asyncio.gather(*tasks)


def get_base_url(host: str, port: int) -> str:
    """Get base url."""
    return f"http://{host}:{port}"
