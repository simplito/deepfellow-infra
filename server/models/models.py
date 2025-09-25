"""Models for models api."""

from typing import Literal

from pydantic import BaseModel


class ModelIdQuery(BaseModel):
    model_id: str


class InstallModelIn(BaseModel):
    alias: str | None = None


class InstallModelOut(BaseModel):
    status: Literal["OK"]


class UninstallModelIn(BaseModel):
    purge: bool = True


class UninstallModelOut(BaseModel):
    status: Literal["OK"]


class RetrieveModelOut(BaseModel):
    id: str
    service: str
    type: str
    installed: bool
    size: str


class ListModelsFilters(BaseModel):
    installed: bool | None = None


class ListModelsOut(BaseModel):
    list: list[RetrieveModelOut]
