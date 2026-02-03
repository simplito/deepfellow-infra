# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prometheus metrics collector."""

from prometheus_client import generate_latest

from server.config import AppSettings
from server.endpointregistry import EndpointRegistry
from server.metrics_registry import MetricsRegistry
from server.services_manager import ServicesManager
from server.utils.hardware import Hardware
from server.websockets.infra_websocket_server import InfraWebsocketServer


class MetricsService:
    def __init__(
        self,
        endpoint_registry: EndpointRegistry,
        config: AppSettings,
        hardware: Hardware,
        services_manager: ServicesManager,
        metrics_registry: MetricsRegistry,
        infra_websocket_server: InfraWebsocketServer,
    ):
        self.endpoint_registry = endpoint_registry
        self.config = config
        self.hardware = hardware
        self.services_manager = services_manager
        self.metrics_registry = metrics_registry
        self.infra_websocket_server = infra_websocket_server

    def get_current_metrics(self) -> bytes:
        """Collect all metrics and return Prometheus text format output."""
        self._collect()
        return self._get_metrics()

    def _collect(
        self,
    ) -> None:
        """Collect all metrics from current state."""
        installed_count = sum(1 for s in self.services_manager.services.values() if s.is_installed())
        self.metrics_registry.services_installed.set(installed_count)
        self.metrics_registry.models_installed.set(len(self.endpoint_registry.list_models()))
        self.metrics_registry.info.info({"name": self.config.name, "infra_url": self.config.infra_url})
        self.metrics_registry.gpu_count.set(len(self.hardware.gpus))
        self.metrics_registry.subinfra_count.set(len(self.infra_websocket_server.connections))

        self.metrics_registry.model_usage.clear()
        for model in self.endpoint_registry.list_models():
            self.metrics_registry.model_usage.labels(model_name=model.name, model_type=model.type).set(model.usage)

    def _get_metrics(self) -> bytes:
        """Generate Prometheus text format output."""
        return generate_latest(self.metrics_registry.registry)
