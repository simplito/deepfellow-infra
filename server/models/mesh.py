# DeepFellow Software Framework.
# Copyright © 2025 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Models for mesh api."""

from pydantic import BaseModel


class MeshInfoModel(BaseModel):
    name: str
    type: str


class MeshInfoInfra(BaseModel):
    name: str
    url: str
    models: list[MeshInfoModel]


class MeshInfo(BaseModel):
    connections: list[MeshInfoInfra]


class ShowMeshInfoOut(BaseModel):
    info: MeshInfo
