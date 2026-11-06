# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Services API."""

import asyncio
import contextlib
import time
from typing import Annotated

from fastapi import APIRouter, Body, Depends, HTTPException, Path, Query, Request
from fastapi.responses import JSONResponse, Response

from server.core.dependencies import (
    auth_admin,
    get_endpoint_registry,
    get_service_provider,
    get_services_manager,
    require_services_config_loaded,
)
from server.endpointregistry import EndpointRegistry
from server.models.api import RegistrationId
from server.models.services import (
    CancelServiceInstallOut,
    CatalogRefreshOut,
    DockerTagsOut,
    InstallServiceIn,
    ListAllModelsFilters,
    ListAllModelsOut,
    ListServicesFilters,
    ListServicesOut,
    NewerNeedsCuda,
    OptionalModelIdQuery,
    RestartDockerContainerOut,
    RetrieveDockerComposeFileOut,
    RetrieveDockerLogsOut,
    RetrieveServiceOut,
    UninstallServiceIn,
    UninstallServiceOut,
)
from server.serviceprovider import ServiceProvider
from server.services_manager import ServicesManager
from server.utils.core import convert_promise_with_progress_to_fastapi_response
from server.utils.docker_image_version import has_cuda_variant, newer_tag_needing_cuda, newest_docker_tag
from server.utils.hardware import get_nvidia_max_cuda_version
from server.utils.logger import logger
from server.utils.registry_client import RegistryUnavailableError
from server.utils.tracing import tracer

router = APIRouter(prefix="/admin/services", tags=["Services"])

_CATALOG_REFRESH_TTL = 6 * 3600  # 6 hours
catalog_refresh_cache: dict[str, tuple[int, int, float]] = {}  # service_id -> (added, total, timestamp)
_DOCKER_TAGS_TTL = 7200  # seconds


@router.post(
    "/{service_id}",
    summary="Install the service.",
    dependencies=[Depends(require_services_config_loaded)],
)
@tracer.trace_request()
async def install_service(
    request: Request,  # noqa: ARG001 needed for tracer
    model: Annotated[InstallServiceIn, Body()],
    service_id: Annotated[str, Path(description="The ID of the service to install")],
    services_manager: Annotated[ServicesManager, Depends(get_services_manager)],
    _: Annotated[str, Depends(auth_admin)],
) -> Response:
    """Install the service."""
    msg = f"{service_id} service installing."
    logger.debug(msg)
    promise = await services_manager.install_service(service_id, model)
    if model.stream:
        return await convert_promise_with_progress_to_fastapi_response(promise)
    result = await promise.wait()
    result_json = JSONResponse(result.model_dump())
    msg = f"{service_id} service installed."
    logger.info(msg)
    return result_json


@router.post(
    "/{service_id}/cancel",
    summary="Cancel an in-progress service install or update.",
)
@tracer.trace_request()
async def cancel_service_install(
    request: Request,  # noqa: ARG001 needed for tracer
    service_id: Annotated[str, Path(description="The ID of the service to use.")],
    services_manager: Annotated[ServicesManager, Depends(get_services_manager)],
    _: Annotated[str, Depends(auth_admin)],
) -> CancelServiceInstallOut:
    """Cancel an in-progress service install or update, stopping the underlying Docker work."""
    msg = f"{service_id} service install cancelling."
    logger.debug(msg)
    await services_manager.cancel_service_install(service_id)
    msg = f"{service_id} service install cancelled."
    logger.info(msg)
    return CancelServiceInstallOut(status="OK")


@router.put(
    "/{service_id}",
    summary="Update the service configuration.",
    dependencies=[Depends(require_services_config_loaded)],
)
@tracer.trace_request()
async def update_service(
    request: Request,  # noqa: ARG001 needed for tracer
    model: Annotated[InstallServiceIn, Body()],
    service_id: Annotated[str, Path(description="The ID of the service to update")],
    services_manager: Annotated[ServicesManager, Depends(get_services_manager)],
    _: Annotated[str, Depends(auth_admin)],
) -> Response:
    """Update an installed service's configuration, recreating the underlying container as needed."""
    msg = f"{service_id} service updating."
    logger.debug(msg)
    promise = await services_manager.update_service(service_id, model)
    if model.stream:
        return await convert_promise_with_progress_to_fastapi_response(promise)
    result = await promise.wait()
    result_json = JSONResponse(result.model_dump())
    msg = f"{service_id} service updated."
    logger.info(msg)
    return result_json


@router.delete(
    "/{service_id}",
    summary="Uninstall the service.",
    dependencies=[Depends(require_services_config_loaded)],
)
@tracer.trace_request()
async def uninstall_service(
    request: Request,  # noqa: ARG001 needed for tracer
    model: Annotated[UninstallServiceIn, Body()],
    service_id: Annotated[str, Path(description="The ID of the service to uninstall")],
    services_manager: Annotated[ServicesManager, Depends(get_services_manager)],
    _: Annotated[str, Depends(auth_admin)],
) -> UninstallServiceOut:
    """Uninstall the service."""
    msg = f"{service_id} service uninstalling."
    logger.debug(msg)
    await services_manager.uninstall_service(service_id, model)
    msg = f"{service_id} service uninstalled."
    logger.info(msg)
    return UninstallServiceOut(status="OK")


@router.get(
    "/models",
    summary="List models among all services.",
)
async def list_models(
    filters: Annotated[ListAllModelsFilters, Query()],
    services_manager: Annotated[ServicesManager, Depends(get_services_manager)],
    _: Annotated[str, Depends(auth_admin)],
) -> ListAllModelsOut:
    """List models among all services."""
    return await services_manager.list_models_from_all_services(filters)


@router.get("/model/test/{registration_id}", summary="Test a model")
async def test_model(
    registration_id: Annotated[RegistrationId, Path(description="The ID of the installed model")],
    endpoint_registry: Annotated[EndpointRegistry, Depends(get_endpoint_registry)],
    _: Annotated[str, Depends(auth_admin)],  # noqa: PT019
) -> JSONResponse:
    """Test a model by making a simple request to it."""
    result = await endpoint_registry.test_model(registration_id)
    return JSONResponse(result)


@router.post(
    "/{service_id}/catalog/refresh",
    summary="Refresh the model catalog from the external library API.",
)
async def refresh_catalog(
    service_id: Annotated[str, Path(description="The ID of the service")],
    services_manager: Annotated[ServicesManager, Depends(get_services_manager)],
    _: Annotated[str, Depends(auth_admin)],
    force: Annotated[bool, Query(description="Bypass the 6-hour TTL and force a fresh fetch")] = False,
) -> Response:
    """Fetch trending models from the external catalog API and merge them into the service's model list.

    Results are cached for 6 hours. Pass ?force=true to bypass the cache. The fetch may run as a
    background job with streamed progress (e.g. for vLLM/llama.cpp/SGLang, which hit HuggingFace
    dozens-to-hundreds of times) instead of completing before the response is returned.
    """
    if not force:
        cached = catalog_refresh_cache.get(service_id)
        if cached and time.monotonic() - cached[2] < _CATALOG_REFRESH_TTL:
            cached_added, cached_total, _ts = cached
            return JSONResponse(CatalogRefreshOut(added=cached_added, total=cached_total).model_dump())

    promise = await services_manager.refresh_catalog(service_id)

    if not promise.has_stream:
        # Already resolved (e.g. Ollama's fast, non-streaming refresh) - update the cache
        # synchronously so a following request within the TTL reliably observes it.
        result = await promise.wait()
        catalog_refresh_cache[service_id] = (result.added, result.total, time.monotonic())
        return JSONResponse(result.model_dump())

    async def _cache_when_done() -> None:
        with contextlib.suppress(Exception):
            result = await promise.wait()
            catalog_refresh_cache[service_id] = (result.added, result.total, time.monotonic())

    asyncio.create_task(_cache_when_done())  # noqa: RUF006 - fire-and-forget cache population, independent of the streamed response
    return await convert_promise_with_progress_to_fastapi_response(promise)


async def _newest_tags(tags: list[str], default_tag: str | None, current: str | None) -> tuple[str | None, NewerNeedsCuda | None]:
    """Newest tag of the running tag's variant, and a newer one held back by the host's CUDA version, if any.

    A CUDA build may go up to the CUDA version the host driver supports.
    """
    reference = current or default_tag
    max_cuda = await get_nvidia_max_cuda_version() if has_cuda_variant(reference) else None
    candidates = [*tags, *([default_tag] if default_tag else [])]
    blocked = newer_tag_needing_cuda(candidates, reference, max_cuda)
    needs_cuda = (
        NewerNeedsCuda(tag=blocked[0], required_cuda=f"{blocked[1][0]}.{blocked[1][1]}", host_cuda=f"{max_cuda[0]}.{max_cuda[1]}")
        if blocked and max_cuda
        else None
    )
    return newest_docker_tag(candidates, reference, max_cuda), needs_cuda


@router.get(
    "/{service_id}/docker-tags",
    summary="Fetch available Docker image tags for the service.",
)
async def get_docker_tags(
    service_id: Annotated[str, Path(description="The ID of the service")],
    services_manager: Annotated[ServicesManager, Depends(get_services_manager)],
    _: Annotated[str, Depends(auth_admin)],
    hardware: Annotated[str | None, Query(description="Hardware variant for tag filtering (e.g. gpu, cpu)")] = None,
    field_name: Annotated[str | None, Query(description="Name of the docker-tags field requesting tags")] = None,
    model_id: Annotated[str | None, Query(description="ID of the model requesting tags, for services installing several models")] = None,
    current: Annotated[
        str | None, Query(description="Currently installed tag; `newest` keeps its variant (defaults to the default tag)")
    ] = None,
) -> DockerTagsOut:
    """Fetch available Docker image tags for the service from its registry, with hardware- and model-aware filtering."""
    try:
        default_tag = services_manager.get_default_docker_tag_for_service(service_id, hardware, model_id)
    except RegistryUnavailableError:
        default_tag = None

    cache_key = (service_id, hardware, field_name, model_id)
    cached = services_manager.docker_tags_cache.get(cache_key)
    if cached and time.monotonic() - cached[2] < _DOCKER_TAGS_TTL:
        cached_tags, cached_image, _ts = cached
        newest, needs_cuda = await _newest_tags(cached_tags, default_tag, current)
        return DockerTagsOut(image=cached_image, tags=cached_tags, default=default_tag, newest=newest, newest_needs_cuda=needs_cuda)

    model_fields = []
    if model_id:
        try:
            model = await services_manager.get_model_from_service(service_id, model_id)
            model_fields = model.spec.fields
        except HTTPException:
            model_fields = []

    service = await services_manager.get_service(service_id)
    docker_tags_fields = [f for f in [*service.spec.fields, *model_fields] if f.type == "docker-tags" and f.docker_image]
    matched_field = (next((f for f in docker_tags_fields if f.name == field_name), None) if field_name else None) or (
        docker_tags_fields[0] if docker_tags_fields else None
    )
    try:
        docker_image_repo = services_manager.get_docker_image_repo_for_service(service_id, hardware, model_id)
    except RegistryUnavailableError:
        docker_image_repo = None
    docker_image = docker_image_repo or (matched_field.docker_image if matched_field else None) or service_id
    registry_unavailable = False
    try:
        tags = await services_manager.get_docker_tags_for_service(service_id, hardware, model_id)
    except RegistryUnavailableError:
        logger.warning("Docker tags unavailable for service %r model %r; registry unreachable", service_id, model_id)
        tags = []
        registry_unavailable = True
    if tags:
        services_manager.docker_tags_cache[cache_key] = (tags, docker_image, time.monotonic())
    newest, needs_cuda = await _newest_tags(tags, default_tag, current)
    return DockerTagsOut(
        image=docker_image,
        tags=tags,
        default=default_tag,
        newest=newest,
        newest_needs_cuda=needs_cuda,
        registry_unavailable=registry_unavailable,
    )


@router.get(
    "/{service_id}",
    summary="Retrieve the service.",
)
async def retrieve_service(
    service_id: Annotated[str, Path(description="The ID of the service to retrieve")],
    services_manager: Annotated[ServicesManager, Depends(get_services_manager)],
    _: Annotated[str, Depends(auth_admin)],
) -> RetrieveServiceOut:
    """Retrieve the service."""
    return await services_manager.get_service(service_id)


@router.get(
    "/{service_id}/progress",
    summary="Get progress of installing service.",
)
async def get_install_progress_service(
    service_id: Annotated[str, Path(description="The ID of the service to use.")],
    services_manager: Annotated[ServicesManager, Depends(get_services_manager)],
    _: Annotated[str, Depends(auth_admin)],
) -> Response:
    """Get progress of installing service."""
    promise = await services_manager.get_service_install_progress(service_id)
    return await convert_promise_with_progress_to_fastapi_response(promise)


@router.get(
    "",
    summary="List services.",
)
async def list_service(
    filters: Annotated[ListServicesFilters, Query()],
    services_manager: Annotated[ServicesManager, Depends(get_services_manager)],
    service_provider: Annotated[ServiceProvider, Depends(get_service_provider)],
    _: Annotated[str, Depends(auth_admin)],
) -> ListServicesOut:
    """List services."""
    result = await services_manager.list_services(filters)
    result.load_error = service_provider.load_error
    return result


@router.get(
    "/{service_id}/docker/logs",
    summary="Retrieve docker logs.",
)
async def retrieve_docker_logs(
    service_id: Annotated[str, Path(description="The ID of the service")],
    query: Annotated[OptionalModelIdQuery, Query()],
    services_manager: Annotated[ServicesManager, Depends(get_services_manager)],
    _: Annotated[str, Depends(auth_admin)],
) -> RetrieveDockerLogsOut:
    """Retrieve docker logs."""
    logs = await services_manager.get_docker_logs(service_id, query.model_id)
    return RetrieveDockerLogsOut(logs=logs)


@router.get(
    "/{service_id}/docker/compose",
    summary="Retrieve docker compose file.",
)
async def retrieve_docker_compose_file(
    service_id: Annotated[str, Path(description="The ID of the service")],
    query: Annotated[OptionalModelIdQuery, Query()],
    services_manager: Annotated[ServicesManager, Depends(get_services_manager)],
    _: Annotated[str, Depends(auth_admin)],
) -> RetrieveDockerComposeFileOut:
    """Retrieve docker logs."""
    compose_file = await services_manager.get_docker_compose_file(service_id, query.model_id)
    return RetrieveDockerComposeFileOut(compose_file=compose_file)


@router.post(
    "/{service_id}/docker/restart",
    summary="Restart docker container.",
    dependencies=[Depends(require_services_config_loaded)],
)
@tracer.trace_request()
async def restart_docker_container(
    request: Request,  # noqa: ARG001 needed for tracer
    service_id: Annotated[str, Path(description="The ID of the service")],
    query: Annotated[OptionalModelIdQuery, Query()],
    services_manager: Annotated[ServicesManager, Depends(get_services_manager)],
    _: Annotated[str, Depends(auth_admin)],
) -> RestartDockerContainerOut:
    """Retrieve docker logs."""
    await services_manager.restart_docker(service_id, query.model_id)
    return RestartDockerContainerOut(status="OK")
