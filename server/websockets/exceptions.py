"""Websockets exceptions."""

from abc import ABC

from fastapi import WebSocketException


class WebsocketExceptionBase(WebSocketException, ABC):
    status_code: int
    detail: str

    def __init__(self) -> None:
        super().__init__(self.status_code, self.detail)


class AuthError(WebsocketExceptionBase):
    status_code: int = 32002
    detail: str = "Incorrect auth credentials."
