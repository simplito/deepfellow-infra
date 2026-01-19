# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Infra client."""

from pydantic import BaseModel

from server.models.api import Model, RegistrationId


class InitRequest(BaseModel):
    auth: str
    name: str
    url: str
    api_key: str
    models: list[Model]
    check_key: str | None = None  # TODO: params.check_key should be switched to requried parameter after old infra migration


class UsageChangeRequest(BaseModel):
    id: RegistrationId
    usage: int


class AddModelRequest(BaseModel):
    model: Model


class RemoveModelRequest(BaseModel):
    id: RegistrationId


class UpdateModelsRequest(BaseModel):
    models: list[Model]
