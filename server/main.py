# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Main app module."""

from fastapi import FastAPI

from server.api import mesh, metrics, models, openai, services, utils
from server.api.fallback import StaticFilesHandler
from server.lifecycle import lifespan
from server.websockets import api as websocket

app = FastAPI(lifespan=lifespan)
app.include_router(services.router)
app.include_router(models.router)
app.include_router(websocket.router)
app.include_router(openai.router)
app.include_router(mesh.router)
app.include_router(metrics.router)
app.include_router(utils.router)

app.mount("/", StaticFilesHandler(directory="static", html=True), name="static")
