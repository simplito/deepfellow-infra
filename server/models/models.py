"""Models for models api."""

from typing import Any, Literal

from pydantic import BaseModel

type InstallModelOptions = dict[str, Any]
type CustomModelDefiniton = dict[str, Any]
type CustomModelId = str


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


class CustomModelField(BaseModel):
    type: str
    name: str
    description: str
    default: str | None = None
    placeholder: str | None = None
    required: bool = True
    values: list[str] | None = None
    display: str | None = None


class CustomModelSpecification(BaseModel):
    fields: list[CustomModelField]


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
    custom: CustomModelId | None = None
    size: str
    spec: ModelSpecification
    has_docker: bool


class ListModelsFilters(BaseModel):
    installed: bool | None = None


class ListModelsOut(BaseModel):
    list: list[RetrieveModelOut]


class AddCustomModelIn(BaseModel):
    spec: CustomModelDefiniton


class AddCustomModelOut(BaseModel):
    custom_model_id: CustomModelId


class RemoveCustomModelOut(BaseModel):
    status: Literal["OK"]
