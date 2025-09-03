"""Models for services api."""

from typing import Literal

from pydantic import BaseModel

from server.models.models import RetrieveModelOut


class InstallServiceIn(BaseModel):
    gpu: bool


class InstallServiceOut(BaseModel):
    status: Literal["OK"]


class UninstallServiceIn(BaseModel):
    purge: bool


class UninstallServiceOut(BaseModel):
    status: Literal["OK"]


class RetrieveServiceOut(BaseModel):
    id: str
    installed: bool


class ListServicesFilters(BaseModel):
    installed: bool | None = None


class ListServicesOut(BaseModel):
    list: list[RetrieveServiceOut]


class ListAllModelsFilters(BaseModel):
    installed: bool = False
    service_id: str | None = None


class ListAllModelsOut(BaseModel):
    list: list[RetrieveModelOut]
