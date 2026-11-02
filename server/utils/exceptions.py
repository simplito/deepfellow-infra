# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

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


class DockerPlatformMismatchError(AppError):
    """Raised when a docker image can't run on this host because it is not built for the host platform."""


class DockerComposeStartError(AppError):
    """Raised when docker compose up exits with a non-zero code."""

    def __init__(self, stdout: str, stderr: str) -> None:
        super().__init__(stderr or stdout or "docker compose failed")
        self.stdout = stdout
        self.stderr = stderr
