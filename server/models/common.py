# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Common types."""

from typing import Any

# from fastapi.responses import JSONResponse, StreamingResponse
# from pydantic import BaseModel

type FormFields = dict[str, Any]
type JsonSerializable = dict[str, Any]
type StarletteResponse = Any  # JsonSerializable | JSONResponse | BaseModel | StreamingResponse
