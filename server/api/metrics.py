# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Metrics API."""

from typing import Annotated

from fastapi import APIRouter, Depends, Response

from server.core.dependencies import get_metrics_service
from server.metrics import MetricsService

router = APIRouter(tags=["Metrics"])


@router.get("/metrics")
async def metrics_endpoint(
    metrics_service: Annotated[MetricsService, Depends(get_metrics_service)],
) -> Response:
    """Get Prometheus metrics."""
    return Response(
        content=metrics_service.get_current_metrics(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
