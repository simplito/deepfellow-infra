# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Infra client."""

from typing import Literal

from pydantic import BaseModel, Field

from server.models.api import Model, RegistrationId


class AncestorInfo(BaseModel):
    url: str
    name: str
    models: list[Model] = []


class InitResponse(BaseModel):
    ancestors: list[AncestorInfo]


class TopologyUpdateRequest(BaseModel):
    action: Literal["join", "leave"]
    url: str
    name: str = ""
    models: list[Model] = []
    children: dict[str, "TopologyUpdateRequest"] = {}


class InitRequest(BaseModel):
    auth: str
    name: str
    url: str
    api_key: str
    models: list[Model]
    children: dict[str, TopologyUpdateRequest] = {}
    check_key: str = Field(min_length=1)


class UsageChangeRequest(BaseModel):
    id: RegistrationId
    usage: int


class UpdateModelsRequest(BaseModel):
    models: list[Model]
