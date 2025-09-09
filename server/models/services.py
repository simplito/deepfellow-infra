"""Models for services api."""

from typing import Any, Literal

from pydantic import BaseModel

from server.models.models import RetrieveModelOut

type ServiceOptions = dict[str, Any]


class InstallServiceIn(BaseModel):
    spec: ServiceOptions


class InstallServiceOut(BaseModel):
    status: Literal["OK"]


class UninstallServiceIn(BaseModel):
    purge: bool


class UninstallServiceOut(BaseModel):
    status: Literal["OK"]


class ServiceField(BaseModel):
    type: str
    name: str
    description: str
    default: str | None = None


class ServiceSpecification(BaseModel):
    fields: list[ServiceField]


class RetrieveServiceOut(BaseModel):
    id: str
    installed: bool | ServiceOptions
    spec: ServiceSpecification


class ListServicesFilters(BaseModel):
    installed: bool | None = None


class ListServicesOut(BaseModel):
    list: list[RetrieveServiceOut]


class ListAllModelsFilters(BaseModel):
    installed: bool = False
    service_id: str | None = None


class ListAllModelsOut(BaseModel):
    list: list[RetrieveModelOut]
