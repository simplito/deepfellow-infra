# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for server/api/utils.py endpoints."""

from fastapi import FastAPI
from starlette.testclient import TestClient

from server.api.utils import router


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


def test_health() -> None:
    with TestClient(_make_app()) as client:
        resp = client.get("/health")

    assert resp.status_code == 200
    assert resp.json() == "OK"
