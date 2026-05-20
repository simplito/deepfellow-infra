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

import pytest
from prometheus_client import CollectorRegistry

from server.metrics import MetricsService
from server.metrics_registry import MetricsRegistry


def _make_instance(installed: object) -> MagicMock:
    inst = MagicMock()
    inst.installed = installed
    return inst


def _make_service(instances: dict[str, object]) -> MagicMock:
    svc = MagicMock()
    svc.instances_info = instances
    return svc


def _make_model(name: str = "llama3", type: str = "llm", usage: int = 0) -> MagicMock:
    m = MagicMock()
    m.name = name
    m.type = type
    m.usage = usage
    return m


def _make_metrics_service(
    services: dict[str, Any] | None = None,
    models: list[Any] | None = None,
    gpus: list[Any] | None = None,
    connections: list[Any] | None = None,
    name: str = "test-infra",
    infra_url: str = "http://localhost",
) -> MetricsService:
    registry = MetricsRegistry(registry=CollectorRegistry(auto_describe=True))

    config = MagicMock()
    config.name = name
    config.infra_url = infra_url

    hardware = MagicMock()
    hardware.gpus = gpus if gpus is not None else []

    services_manager = MagicMock()
    services_manager.services = services if services is not None else {}

    endpoint_registry = MagicMock()
    endpoint_registry.list_models.return_value = models if models is not None else []

    infra_ws = MagicMock()
    infra_ws.connections = connections if connections is not None else []

    return MetricsService(
        endpoint_registry=endpoint_registry,
        config=config,
        hardware=hardware,
        services_manager=services_manager,
        metrics_registry=registry,
        infra_websocket_server=infra_ws,
    )


def test_stores_all_dependencies() -> None:
    svc = _make_metrics_service()
    assert svc.endpoint_registry is not None
    assert svc.config is not None
    assert svc.hardware is not None
    assert svc.services_manager is not None
    assert svc.metrics_registry is not None
    assert svc.infra_websocket_server is not None


@pytest.mark.parametrize(
    ("services", "expected"),
    [
        pytest.param({}, 0, id="no_services"),
        pytest.param(
            {
                "svc1": _make_service(
                    {
                        "a": _make_instance(installed=True),
                        "b": _make_instance(installed=None),
                    }
                ),
                "svc2": _make_service({"c": _make_instance(installed="some_info")}),
            },
            2,
            id="counts_only_installed",
        ),
        pytest.param(
            {
                "svc1": _make_service(
                    {
                        "a": _make_instance(installed=None),
                        "b": _make_instance(installed=None),
                    }
                ),
            },
            0,
            id="all_not_installed",
        ),
        pytest.param(
            {
                "svc1": _make_service({"a": _make_instance(True), "b": _make_instance(True)}),
                "svc2": _make_service({"c": _make_instance(True)}),
            },
            3,
            id="all_installed_across_multiple_services",
        ),
    ],
)
def test_get_installed_instances_quantity(services: dict[str, Any], expected: int) -> None:
    svc = _make_metrics_service(services=services)
    assert svc.get_installed_instances_quantity() == expected


def test_collect_sets_services_installed_count() -> None:
    services = {"s": _make_service({"i": _make_instance(True)})}
    svc = _make_metrics_service(services=services)
    svc._collect()  # pyright: ignore[reportPrivateUsage]

    result = svc.metrics_registry.services_installed._value.get()  # pyright: ignore[reportPrivateUsage]

    assert result == 1.0


def test_collect_sets_models_installed_count() -> None:
    models = [_make_model("m1"), _make_model("m2")]
    svc = _make_metrics_service(models=models)
    svc._collect()  # pyright: ignore[reportPrivateUsage]

    result = svc.metrics_registry.models_installed._value.get()  # pyright: ignore[reportPrivateUsage]

    assert result == 2.0


def test_collect_sets_gpu_count() -> None:
    svc = _make_metrics_service(gpus=["gpu0", "gpu1"])
    svc._collect()  # pyright: ignore[reportPrivateUsage]

    result = svc.metrics_registry.gpu_count._value.get()  # pyright: ignore[reportPrivateUsage]

    assert result == 2.0


def test_collect_sets_subinfra_count() -> None:
    svc = _make_metrics_service(connections=["ws1", "ws2", "ws3"])
    svc._collect()  # pyright: ignore[reportPrivateUsage]

    result = svc.metrics_registry.subinfra_count._value.get()  # pyright: ignore[reportPrivateUsage]

    assert result == 3.0


def test_collect_sets_model_usage_labels() -> None:
    models = [_make_model("llama3", "llm", 5)]
    svc = _make_metrics_service(models=models)
    svc._collect()  # pyright: ignore[reportPrivateUsage]

    value = svc.metrics_registry.model_usage.labels(model_name="llama3", model_type="llm")._value.get()  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage]

    assert value == 5.0


def test_collect_clears_model_usage_before_repopulating() -> None:
    models = [_make_model("llama3", "llm", 10)]
    svc = _make_metrics_service(models=models)
    svc._collect()  # pyright: ignore[reportPrivateUsage]
    svc.endpoint_registry.list_models.return_value = [_make_model("llama3", "llm", 3)]  # pyright: ignore[reportAttributeAccessIssue]
    svc._collect()  # pyright: ignore[reportPrivateUsage]

    value = svc.metrics_registry.model_usage.labels(model_name="llama3", model_type="llm")._value.get()  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage]

    assert value == 3.0


def test_get_metrics_returns_bytes() -> None:
    svc = _make_metrics_service()

    result = svc._get_metrics()  # pyright: ignore[reportPrivateUsage]

    assert isinstance(result, bytes)


def test_get_metrics_output_contains_registry_metrics() -> None:
    svc = _make_metrics_service()
    svc.metrics_registry.models_installed.set(7)

    result = svc._get_metrics()  # pyright: ignore[reportPrivateUsage]

    assert b"deepfellow_models_installed 7.0" in result


def test_get_current_metrics_returns_bytes() -> None:
    svc = _make_metrics_service()

    result = svc.get_current_metrics()

    assert isinstance(result, bytes)


def test_get_current_metrics_reflects_current_state() -> None:
    models = [_make_model("whisper", "stt", 2)]
    svc = _make_metrics_service(models=models)

    result = svc.get_current_metrics()

    assert b"deepfellow_models_installed 1.0" in result
    assert b'deepfellow_model_usage{model_name="whisper",model_type="stt"} 2.0' in result
