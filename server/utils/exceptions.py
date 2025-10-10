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
