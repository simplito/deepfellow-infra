# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Models for models api."""

from typing import Any, Literal

from pydantic import BaseModel

from server.models.api import RegistrationId

type InstallModelOptions = dict[str, Any]
type CustomModelDefiniton = dict[str, Any]
type CustomModelId = str


class ModelIdQuery(BaseModel):
    model_id: str


class OneOfOption(BaseModel):
    value: str
    label: str


class ModelField(BaseModel):
    type: str
    name: str
    description: str
    default: str | None = None
    placeholder: str | None = None
    required: bool = True
    values: list[OneOfOption | str] | None = None


class ModelSpecification(BaseModel):
    fields: list[ModelField]


class CustomModelField(BaseModel):
    type: str
    name: str
    description: str
    default: str | None = None
    placeholder: str | None = None
    required: bool = True
    values: list[str] | None = None
    display: str | None = None


class CustomModelSpecification(BaseModel):
    fields: list[CustomModelField]


class InstallModelIn(BaseModel):
    stream: bool = False
    ignore_warnings: bool = False
    spec: InstallModelOptions | None = None


class ModelInfo(BaseModel):
    spec: InstallModelOptions | None
    registration_id: RegistrationId


class InstallModelOut(BaseModel):
    status: Literal["OK"]
    details: str


class UninstallModelIn(BaseModel):
    purge: bool = True


class UninstallModelOut(BaseModel):
    status: Literal["OK"]


class InstallModelProgress(BaseModel):
    stage: str
    value: float


class RetrieveModelOut(BaseModel):
    id: str
    service: str
    type: str
    installed: bool | InstallModelProgress | ModelInfo
    downloaded: bool
    custom: CustomModelId | None = None
    size: str
    spec: ModelSpecification
    has_docker: bool


class ListModelsFilters(BaseModel):
    installed: bool | None = None


class ListModelsOut(BaseModel):
    list: list[RetrieveModelOut]


class AddCustomModelIn(BaseModel):
    spec: CustomModelDefiniton


class AddCustomModelOut(BaseModel):
    custom_model_id: CustomModelId


class RemoveCustomModelOut(BaseModel):
    status: Literal["OK"]
