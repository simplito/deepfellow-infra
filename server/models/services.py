# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Models for services api."""

from typing import Any, Literal

from pydantic import BaseModel

from server.models.models import CustomModelSpecification, OneOfOption, RetrieveModelOut

type ServiceOptions = dict[str, Any]
type ServiceSize = dict[str, str] | str


class InstallServiceIn(BaseModel):
    stream: bool = False
    ignore_warnings: bool = False
    spec: ServiceOptions


class InstallServiceOut(BaseModel):
    status: Literal["OK"]


class UninstallServiceIn(BaseModel):
    purge: bool


class UninstallServiceOut(BaseModel):
    status: Literal["OK"]


class CancelServiceInstallOut(BaseModel):
    status: Literal["OK"]


class ServiceField(BaseModel):
    type: str
    name: str
    description: str
    default: str | None = None
    placeholder: str | None = None
    required: bool = True
    values: list[OneOfOption | str] | None = None
    docker_image: str | None = None
    depends_on: str | None = None


class ServiceSpecification(BaseModel):
    fields: list[ServiceField]


class InstallServiceProgress(BaseModel):
    stage: str
    value: float
    image: str | None = None  # docker image being pulled, if the current stage is pulling one


class RetrieveServiceOut(BaseModel):
    id: str
    type: str
    instance: str
    description: str
    installed: bool | InstallServiceProgress | ServiceOptions
    downloaded: bool
    spec: ServiceSpecification
    size: ServiceSize
    custom_model_spec: CustomModelSpecification | None
    has_docker: bool
    is_cloud: bool = False
    disabled_reason: str | None = None
    catalog_refresh_unavailable_reason: str | None = None


class ListServicesFilters(BaseModel):
    installed: bool | None = None


class ListServicesOut(BaseModel):
    list: list[RetrieveServiceOut]
    load_error: str | None = None


class ListAllModelsFilters(BaseModel):
    installed: bool | None = None
    service_id: str | None = None


class ListAllModelsOut(BaseModel):
    list: list[RetrieveModelOut]


class OptionalModelIdQuery(BaseModel):
    model_id: str | None = None


class RetrieveDockerLogsOut(BaseModel):
    logs: str


class RetrieveDockerComposeFileOut(BaseModel):
    compose_file: str


class RestartDockerContainerOut(BaseModel):
    status: Literal["OK"]


class InfraSettingsOut(BaseModel):
    cloud_enabled: bool


class UpdateInfraSettingsIn(BaseModel):
    cloud_enabled: bool


class GpuCardStats(BaseModel):
    name: str
    total_vram_gb: float
    used_vram_gb: float


class GpuStats(BaseModel):
    total_vram_gb: float
    used_vram_gb: float
    gpus: list[GpuCardStats] | None = None


class SystemStats(BaseModel):
    cpu_percent: float
    cpu_model: str
    ram_total_gb: float
    ram_used_gb: float


class MemoryLoadComponent(BaseModel):
    name: str
    device: str | None
    size: str


class MemoryLoadSession(BaseModel):
    model: str | None = None
    components: list[MemoryLoadComponent]
    total: str


class MemoryLoadOut(BaseModel):
    sessions: list[MemoryLoadSession]


class CatalogRefreshOut(BaseModel):
    added: int
    total: int


class NewerNeedsCuda(BaseModel):
    """A newer image tag that is held back because the host's NVIDIA driver supports too old a CUDA version."""

    tag: str
    required_cuda: str
    host_cuda: str


class DockerTagsOut(BaseModel):
    image: str
    tags: list[str]
    default: str | None = None
    # Newest tag of the same variant as `current` (or the default), None when it cannot be determined.
    newest: str | None = None
    # Set when an even newer tag exists but needs a newer CUDA than the host's driver supports.
    newest_needs_cuda: NewerNeedsCuda | None = None
    # True if `tags` is empty because the registry couldn't be reached, not because the image
    # genuinely has no tags - lets the frontend show an error+retry state instead of "No tags found."
    registry_unavailable: bool = False
