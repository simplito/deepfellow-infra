# DeepFellow Software Framework.
# Copyright © 2025 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Custom static files handler for SPA routing support."""

from pathlib import Path

from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException
from starlette.responses import FileResponse, Response
from starlette.types import Scope


class StaticFilesHandler(StaticFiles):
    """Custom static files handler that serves index.html for 404s (SPA routing)."""

    async def get_response(self, path: str, scope: Scope) -> Response:
        """Return an HTTP response, given the incoming path, method and request headers."""
        try:
            return await super().get_response(path, scope)
        except HTTPException as e:
            if e.status_code == 404:
                # SPA routing: serve index.html for all 404s
                index_path = Path("static") / "index.html"
                return FileResponse(index_path)
            raise
