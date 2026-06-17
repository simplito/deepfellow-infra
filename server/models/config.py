# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Models for config api."""

from pydantic import BaseModel


class ConfigEntry(BaseModel):
    key: str
    value: str
    is_secret: bool


class ConfigOut(BaseModel):
    entries: list[ConfigEntry]


class ConfigRevealOut(BaseModel):
    key: str
    value: str
