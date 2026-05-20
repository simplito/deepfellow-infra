# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for server/api/metrics.py endpoints."""

from unittest.mock import MagicMock

from fastapi import FastAPI
from starlette.testclient import TestClient

from server.api.metrics import router
from server.core.dependencies import auth_metrics, get_metrics_service


def _make_metrics_service(content: str = "# metrics\n") -> MagicMock:
    svc = MagicMock()
    svc.get_current_metrics.return_value = content
    return svc


def _make_app(metrics_service: MagicMock | None = None) -> FastAPI:
    if metrics_service is None:
        metrics_service = _make_metrics_service()
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_metrics_service] = lambda: metrics_service
    app.dependency_overrides[auth_metrics] = lambda: None
    return app


def test_metrics_200() -> None:
    with TestClient(_make_app()) as client:
        resp = client.get("/metrics")

    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]


def test_metrics_content() -> None:
    svc = _make_metrics_service("my_metric 1.0\n")

    with TestClient(_make_app(svc)) as client:
        resp = client.get("/metrics")

    assert "my_metric" in resp.text
    assert svc.get_current_metrics.call_count == 1
