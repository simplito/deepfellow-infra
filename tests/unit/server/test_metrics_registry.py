# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

from prometheus_client import CollectorRegistry

from server.metrics_registry import MetricsRegistry


def make_registry() -> MetricsRegistry:
    return MetricsRegistry(registry=CollectorRegistry(auto_describe=True))


def test_creates_own_registry_when_none_provided() -> None:
    mr = MetricsRegistry()

    assert mr.registry is not None


def test_uses_provided_registry() -> None:
    custom = CollectorRegistry(auto_describe=True)

    mr = MetricsRegistry(registry=custom)

    assert mr.registry is custom


def test_all_metrics_attributes_exist() -> None:
    mr = make_registry()

    assert hasattr(mr, "info")
    assert hasattr(mr, "models_installed")
    assert hasattr(mr, "services_installed")
    assert hasattr(mr, "subinfra_count")
    assert hasattr(mr, "gpu_count")
    assert hasattr(mr, "model_usage")
    assert hasattr(mr, "request_total")
    assert hasattr(mr, "request_duration")
    assert hasattr(mr, "request_errors")
    assert hasattr(mr, "requests_in_flight")


def test_info_metric_can_be_set() -> None:
    mr = make_registry()

    mr.info.info({"version": "1.0.0", "env": "test"})


def test_models_installed_gauge() -> None:
    mr = make_registry()

    mr.models_installed.set(5)
    assert mr.models_installed._value.get() == 5.0  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage]


def test_services_installed_gauge() -> None:
    mr = make_registry()

    mr.services_installed.set(3)
    assert mr.services_installed._value.get() == 3.0  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage]


def test_subinfra_count_gauge() -> None:
    mr = make_registry()

    mr.subinfra_count.set(2)
    assert mr.subinfra_count._value.get() == 2.0  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage]


def test_gpu_count_gauge() -> None:
    mr = make_registry()

    mr.gpu_count.set(4)
    assert mr.gpu_count._value.get() == 4.0  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage]


def test_model_usage_gauge_with_labels() -> None:
    mr = make_registry()

    mr.model_usage.labels(model_name="llama3", model_type="llm").set(7)

    value = mr.model_usage.labels(model_name="llama3", model_type="llm")._value.get()  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage]
    assert value == 7.0


def test_request_total_counter_with_labels() -> None:
    mr = make_registry()

    mr.request_total.labels(model_name="llama3", model_type="llm", status="200").inc()

    value = mr.request_total.labels(model_name="llama3", model_type="llm", status="200")._value.get()  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage]
    assert value == 1.0


def test_request_duration_histogram_with_labels() -> None:
    mr = make_registry()

    child = mr.request_duration.labels(model_name="llama3", model_type="llm")

    child.observe(1.5)
    assert child._sum.get() == 1.5  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage]


def test_request_errors_counter_with_labels() -> None:
    mr = make_registry()
    mr.request_errors.labels(model_name="llama3", error_type="timeout").inc(3)

    value = mr.request_errors.labels(model_name="llama3", error_type="timeout")._value.get()  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage]

    assert value == 3.0


def test_requests_in_flight_gauge_with_labels() -> None:
    mr = make_registry()
    mr.requests_in_flight.labels(model_type="llm").set(2)

    value = mr.requests_in_flight.labels(model_type="llm")._value.get()  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage]

    assert value == 2.0


def test_multiple_instances_use_isolated_registries() -> None:
    mr1 = make_registry()
    mr2 = make_registry()

    mr1.models_installed.set(10)
    mr2.models_installed.set(20)

    assert mr1.models_installed._value.get() == 10.0  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage]
    assert mr2.models_installed._value.get() == 20.0  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage]
