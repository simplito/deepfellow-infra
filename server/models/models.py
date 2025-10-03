"""Models for models api."""

from typing import Any, Literal

from pydantic import BaseModel

type InstallModelOptions = dict[str, Any]


class ModelIdQuery(BaseModel):
    model_id: str


class ModelField(BaseModel):
    type: str
    name: str
    description: str
    default: str | None = None
    placeholder: str | None = None
    required: bool = True


class ModelSpecification(BaseModel):
    fields: list[ModelField]


class InstallModelIn(BaseModel):
    spec: InstallModelOptions | None = None


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
    installed: bool | InstallModelIn
    size: str
    spec: ModelSpecification


class ListModelsFilters(BaseModel):
    installed: bool | None = None


class ListModelsOut(BaseModel):
    list: list[RetrieveModelOut]
