# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Others API."""

from fastapi import APIRouter

router = APIRouter(tags=["Utils"])


@router.get("/health", summary="Checks if the server is running.")
async def health() -> str:
    """Check if the server is running."""
    return "OK"
