# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prometheus metrics definitions."""

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, Info


class MetricsRegistry:
    def __init__(self, registry: CollectorRegistry | None = None):
        self.registry = registry or CollectorRegistry(auto_describe=True)  # REGISTRY

        self.info = Info(
            "deepfellow",
            "DeepFellow Infra information",
            registry=self.registry,
        )

        self.models_installed = Gauge(
            "deepfellow_models_installed",
            "Number of currently installed models over all services",
            registry=self.registry,
        )

        self.services_installed = Gauge(
            "deepfellow_services_installed",
            "Number of installed services",
            registry=self.registry,
        )

        self.gpu_count = Gauge(
            "deepfellow_infra_gpu_count",
            "Number of available GPUs",
            registry=self.registry,
        )

        self.model_usage = Gauge(
            "deepfellow_model_usage",
            "Current concurrent usage per model",
            ["model_name", "model_type"],
            registry=self.registry,
        )

        self.request_total = Counter(
            "deepfellow_requests_total",
            "Total number of requests",
            ["model_name", "model_type", "status"],
            registry=self.registry,
        )

        self.request_duration = Histogram(
            "deepfellow_request_duration_seconds",
            "Request duration in seconds",
            ["model_name", "model_type"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0],
            registry=self.registry,
        )

        self.request_errors = Counter(
            "deepfellow_request_errors_total",
            "Total request errors by type",
            ["model_name", "error_type"],
            registry=self.registry,
        )

        self.requests_in_flight = Gauge(
            "deepfellow_requests_in_flight",
            "Number of requests currently being processed",
            ["model_type"],
            registry=self.registry,
        )
