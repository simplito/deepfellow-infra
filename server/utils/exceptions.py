# DeepFellow Software Framework.
# Copyright © 2025 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Custom exceptions."""

from typing import Any


class AppError(Exception):
    """Raised when error in application occurs."""


class AppStartError(Exception):
    """Raise when an error occurs during application startup."""


class ApiError(Exception):
    """Hold code and message that can be send to client."""

    def __init__(self, message: str, code: int, data: Any = None):  # noqa: ANN401
        super().__init__(message)
        self.message = message
        self.code = code
        self.data = data


class DockerImageDoesNotExistError(Exception):
    """Raise when docker image does not exist."""

    def __init__(self, image: str):
        super().__init__(f"Docker image does not exist {image}")
        self.image = image


class DockerImageAuthorizationError(Exception):
    """Raise when authorization error occurs during accessing to docker image."""

    def __init__(self, image: str):
        super().__init__(f"Cannot access docker image, authorization failed {image}")
        self.image = image
