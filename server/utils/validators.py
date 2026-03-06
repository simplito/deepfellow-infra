# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Methods for data validation."""

from typing import Annotated

from pydantic import AfterValidator


def validate_no_traversal(v: str) -> str:
    """Raise if path contains '..'."""
    if v.startswith("../") or "/../" in v:
        raise ValueError("Security risk: Path traversal sequence '/../' is not allowed")
    return v


type SafePath = Annotated[str, AfterValidator(validate_no_traversal)]
