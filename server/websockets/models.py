"""Infra client."""

from pydantic import BaseModel

from server.models.api import Model, RegistrationId


class InitRequest(BaseModel):
    auth: str
    name: str
    url: str
    api_key: str
    models: list[Model]


class UsageChangeRequest(BaseModel):
    id: RegistrationId
    usage: int


class AddModelRequest(BaseModel):
    model: Model


class RemoveModelRequest(BaseModel):
    id: RegistrationId


class UpdateModelsRequest(BaseModel):
    models: list[Model]
