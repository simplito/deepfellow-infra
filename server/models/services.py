# DeepFellow Software Framework.
# Copyright © 2025 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Models for services api."""

from typing import Any, Literal

from pydantic import BaseModel

from server.models.models import CustomModelSpecification, RetrieveModelOut

type ServiceOptions = dict[str, Any]
type ServiceSize = dict[str, str] | str


class InstallServiceIn(BaseModel):
    spec: ServiceOptions


class InstallServiceOut(BaseModel):
    status: Literal["OK"]


class UninstallServiceIn(BaseModel):
    purge: bool


class UninstallServiceOut(BaseModel):
    status: Literal["OK"]


class ServiceField(BaseModel):
    type: str
    name: str
    description: str
    default: str | None = None
    placeholder: str | None = None
    required: bool = True


class ServiceSpecification(BaseModel):
    fields: list[ServiceField]


class RetrieveServiceOut(BaseModel):
    id: str
    description: str
    installed: bool | ServiceOptions
    spec: ServiceSpecification
    size: ServiceSize
    custom_model_spec: CustomModelSpecification | None
    has_docker: bool


class ListServicesFilters(BaseModel):
    installed: bool | None = None


class ListServicesOut(BaseModel):
    list: list[RetrieveServiceOut]


class ListAllModelsFilters(BaseModel):
    installed: bool | None = None
    service_id: str | None = None


class ListAllModelsOut(BaseModel):
    list: list[RetrieveModelOut]


class OptionalModelIdQuery(BaseModel):
    model_id: str | None = None


class RetrieveDockerLogsOut(BaseModel):
    logs: str


class RetrieveDockerComposeFileOut(BaseModel):
    compose_file: str


class RestartDockerContainerOut(BaseModel):
    status: Literal["OK"]
