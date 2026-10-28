# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Base2 service."""

import asyncio
import shutil
import time
import uuid
from abc import abstractmethod
from collections.abc import Awaitable, Callable, Iterator, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar, cast

from fastapi import HTTPException
from pydantic import BaseModel

from server.config import AppSettings
from server.docker import (
    ContainerStatus,
    DockerImage,
    DockerOptions,
    DockerService,
)
from server.endpointregistry import EndpointRegistry
from server.models.models import (
    AddCustomModelIn,
    CustomModelDefiniton,
    CustomModelId,
    InstallModelIn,
    InstallModelOut,
    InstallModelProgress,
    ModelField,
    ModelSpecification,
    UninstallModelIn,
)
from server.models.services import (
    InstallServiceIn,
    InstallServiceOut,
    InstallServiceProgress,
    OneOfOption,
    ServiceField,
    UninstallServiceIn,
)
from server.serviceprovider import ServiceProvider, ServiceRawConfig
from server.services.base_service import BaseService
from server.utils.core import (
    PromiseWithProgress,
    Stream,
    StreamChunk,
    StreamChunkProgress,
    Utils,
    convert_size_to_bytes,
    is_own_cancellation,
)
from server.utils.exceptions import DockerImageAuthorizationError, DockerImageDoesNotExistError
from server.utils.hardware import GpuInfo, Hardware, HardwarePartInfo, NvidiaGpuInfo
from server.utils.logger import logger
from server.utils.model_downloader import ModelDownloader
from server.utils.sentry import UpstreamError, capture_exception
from server.utils.spec_defaults import fill_spec_defaults

if TYPE_CHECKING:
    from server.services_manager import ServicesManager


class ModelConfig(BaseModel):
    model_id: str
    options: InstallModelIn
    definition: dict[str, Any] | None = None
    capacity: int | None = None
    capacity_known: bool = False
    gpu_memory_utilization: float | None = None
    # Resolved container name, pinned so a live container_name_suffix change doesn't rename it on restart.
    container_name: str | None = None


class CustomModel(BaseModel):
    id: CustomModelId
    data: CustomModelDefiniton


_CUSTOM_MODEL_NON_SERVING_FIELDS = {"id", "size"}


def _custom_model_serving_fields_changed(old: dict[str, Any], new: dict[str, Any]) -> bool:
    """Return True if any field affecting whether the repo can be served differs between specs.

    `id` (display name) and `size` are cosmetic/derived and don't affect serviceability, so an
    update that only touches those shouldn't force re-validation of a model that already passed
    it - which would otherwise be able to permanently strand a model added before validation
    existed, or during a transient HuggingFace outage.

    Values are compared with falsy values normalized to `None` first, so a field missing from an
    older spec (e.g. `revision`, or `skip_validation` on a model added before that field existed)
    doesn't register as "changed" just because the current caller submits it explicitly as `""` or
    `False` - which would otherwise defeat the point of this check for exactly the legacy models it
    exists to protect.
    """
    keys = (set(old) | set(new)) - _CUSTOM_MODEL_NON_SERVING_FIELDS
    return any((old.get(k) or None) != (new.get(k) or None) for k in keys)


class InstanceConfig(BaseModel):
    options: InstallServiceIn | None = None
    models: list[ModelConfig] | None = None
    custom: list[CustomModel] | None = None
    # Same purpose as ModelConfig.container_name, for single-container-per-instance services (e.g. Ollama).
    container_name: str | None = None


class ServiceConfig(BaseModel):
    instances: dict[str, InstanceConfig] | None = None
    downloaded: dict[str, Any] | None = None
    service_downloaded: bool | None = True


InstalledInfoType = TypeVar("InstalledInfoType")
DownloadInfoType = TypeVar("DownloadInfoType")


class _HasModels(Protocol):
    """Structural view of every service's ``InstalledInfo``: it always carries a ``models`` mapping."""

    models: dict[str, Any]


class _HasOptions(Protocol):
    """Structural view of every service's ``InstalledInfo`` options."""

    options: InstallServiceIn


class InstallingModel:
    """Tracks one model install.

    Reserved as soon as `install_model()` decides to start one, resolved once the service's
    `_install_model()` actually produces the real promise - so a concurrent caller can find the
    reservation and wait for the real promise instead of racing `_install_model()` a second time.
    """

    last_chunk: StreamChunk | None = None

    def __init__(self) -> None:
        self.promise: PromiseWithProgress[InstallModelOut, StreamChunk] | None = None
        self.chained_promise: PromiseWithProgress[InstallModelOut, StreamChunk] | None = None
        self.task: asyncio.Task[None] | None = None
        self.pending_task: asyncio.Task[PromiseWithProgress[InstallModelOut, StreamChunk]] | None = None
        self._ready = asyncio.Event()
        self._error: BaseException | None = None

    def resolve(
        self,
        promise: PromiseWithProgress[InstallModelOut, StreamChunk],
        on_success: Callable[[InstallModelOut], Awaitable[InstallModelOut]],
        on_error: Callable[[Exception], None],
    ) -> PromiseWithProgress[InstallModelOut, StreamChunk]:
        """Attach the real install promise once it exists, unblocking any concurrent waiters.

        Chains the one-time post-install bookkeeping (`on_success`/`on_error`) onto the real promise
        exactly once here, rather than letting each caller of `install_model()` chain its own copy -
        every caller (the original one and any that joined this reservation) shares the single
        resulting promise, so bookkeeping like persisting config or recording a warning runs once per
        real install, not once per concurrent caller.
        """
        self.promise = promise
        self.chained_promise = promise.next(on_success, on_error)

        async def the_func() -> None:
            async for chunk in promise.progress.as_generator():
                self.last_chunk = chunk

        self.task = asyncio.create_task(the_func())
        self._ready.set()
        return self.chained_promise

    def reject(self, error: BaseException) -> None:
        """Unblock any concurrent waiters with a failure - no promise will ever exist for this reservation.

        A raw `CancelledError` is converted to a plain `HTTPException` before being stored: it's
        only meaningful as a real cancellation to whichever task actually owned it (that task sees
        the original error via its own `raise`, untouched by this) - an unrelated waiter (a
        progress-watcher, a concurrent joiner) reattaching later just needs a clean, ordinary error
        to turn into a response, not a `BaseException` most frameworks won't handle gracefully.
        """
        if isinstance(error, asyncio.CancelledError):
            error = HTTPException(409, "Model install was cancelled")
        self._error = error
        self._ready.set()

    async def wait_ready(self) -> PromiseWithProgress[InstallModelOut, StreamChunk]:
        """Wait until the real install promise is available, then return it.

        Re-raises the original failure if the reservation was rejected instead of resolved, so a
        concurrent caller learns the real outcome instead of hanging forever.
        """
        await self._ready.wait()
        if self._error is not None:
            raise self._error
        assert self.promise is not None
        return self.promise

    async def wait_chained(self) -> PromiseWithProgress[InstallModelOut, StreamChunk]:
        """Wait until the reservation resolves, then return the single shared post-bookkeeping promise."""
        await self._ready.wait()
        if self._error is not None:
            raise self._error
        assert self.chained_promise is not None
        return self.chained_promise


class InstallingInstance(Generic[InstalledInfoType]):  # noqa: UP046
    """Tracks one service-instance install/update.

    Reserved synchronously as soon as `install_instance()`/`update_instance()` decide to proceed,
    resolved once the service's `_install_instance()` actually produces the real promise - so the
    "already installing" guard, and any concurrent reader of the install progress, see "installing"
    for the whole duration instead of a false "not installing" while `_install_instance()` is still
    doing its (potentially slow) pre-promise setup work.

    `timeout_seconds` bounds how long a progress reader waits on `wait_ready()`/`wait_chained()` before
    giving up - it's set by whichever caller (`install_instance()`/`update_instance()`) created the
    reservation, since they don't do the same amount of pre-promise work (see the two
    `_INSTANCE_*_STATUS_WAIT_TIMEOUT_SECONDS` constants below).
    """

    last_chunk: StreamChunk | None = None

    def __init__(self, timeout_seconds: float) -> None:
        self.promise: PromiseWithProgress[InstalledInfoType, StreamChunk] | None = None
        self.chained_promise: PromiseWithProgress[InstallServiceOut, StreamChunk] | None = None
        self.task: asyncio.Task[None] | None = None
        self.timeout_seconds = timeout_seconds
        self._ready = asyncio.Event()
        self._error: BaseException | None = None

    def resolve(
        self,
        promise: PromiseWithProgress[InstalledInfoType, StreamChunk],
        on_success: Callable[[InstalledInfoType], Awaitable[InstallServiceOut]],
        on_error: Callable[[Exception], None],
    ) -> PromiseWithProgress[InstallServiceOut, StreamChunk]:
        """Attach the real install promise once it exists, unblocking any concurrent waiters.

        Chains the one-time post-install bookkeeping (`on_success`/`on_error`) onto the real promise
        exactly once here, rather than letting the caller chain its own copy first and hand this only
        the already-chained result - `wait_ready()` needs the real, unchained promise (the one whose
        `.cancel()` actually reaches `_install_instance()`'s work), while `wait_chained()` needs the
        bookkeeping-wrapped one, and only resolving from the real promise lets both stay available.
        """
        return self.resolve_with_chain(promise, promise.next(on_success, on_error))

    def resolve_with_chain(
        self,
        promise: PromiseWithProgress[InstalledInfoType, StreamChunk],
        chained_promise: PromiseWithProgress[InstallServiceOut, StreamChunk],
    ) -> PromiseWithProgress[InstallServiceOut, StreamChunk]:
        """Like `resolve()`, but for a caller that built its own chained promise instead of via `.next()`.

        `.next()`'s `on_error` is a synchronous, fire-and-forget callback - it can't express "await more
        recovery work, then let *that* decide the final outcome" (see `update_instance()`'s rollback,
        which needs exactly that so a caller isn't told "update failed" before the restore attempt has
        actually concluded). A caller needing that builds its own chained promise and hands it here.
        """
        self.promise = promise
        self.chained_promise = chained_promise

        async def the_func() -> None:
            async for chunk in promise.progress.as_generator():
                self.last_chunk = chunk

        self.task = asyncio.create_task(the_func())
        self._ready.set()
        return self.chained_promise

    def reject(self, error: BaseException) -> None:
        """Unblock any concurrent waiters with a failure - no promise will ever exist for this reservation.

        A raw `CancelledError` is converted to a plain `HTTPException` before being stored: it's only
        meaningful as a real cancellation to whichever task actually owned it (that task sees the
        original error via its own `raise`, untouched by this) - an unrelated waiter (a progress
        reader) reattaching later just needs a clean, ordinary error to turn into a response, not a
        `BaseException` most frameworks won't handle gracefully.
        """
        if isinstance(error, asyncio.CancelledError):
            error = HTTPException(409, "Service instance install was cancelled")
        self._error = error
        self._ready.set()

    async def wait_ready(self) -> PromiseWithProgress[InstalledInfoType, StreamChunk]:
        """Wait until the real install promise is available, then return it.

        Re-raises the original failure if the reservation was rejected instead of resolved, so a
        concurrent reader learns the real outcome instead of hanging forever.
        """
        await self._ready.wait()
        if self._error is not None:
            raise self._error
        if self.promise is None:
            raise RuntimeError("InstallingInstance became ready without a promise or an error - resolve()/reject() were never called")
        return self.promise

    async def wait_chained(self) -> PromiseWithProgress[InstallServiceOut, StreamChunk]:
        """Wait until the reservation resolves, then return the single shared post-install-bookkeeping promise."""
        await self._ready.wait()
        if self._error is not None:
            raise self._error
        if self.chained_promise is None:
            raise RuntimeError("InstallingInstance became ready without a promise or an error - resolve()/reject() were never called")
        return self.chained_promise


@dataclass
class Instance[InstalledInfoType]:
    installed: InstalledInfoType | None
    installing: InstallingInstance[InstalledInfoType] | None
    installing_model_progress: dict[str, InstallingModel]
    config: InstanceConfig


_LOG_CACHE_TTL = 8.0
_FAILED_LOG_CACHE_TTL = 1.0
_DOCKER_LOGS_TAIL_LINES = 1000  # generous for a startup capacity line; bounds memory on a long-running container
_RECONCILE_INTERVAL_SECONDS = 90
_RECONCILE_DEAD_THRESHOLD = 3
# Bounds how long a caller waits for a model install's pre-promise setup work (GPU/image/registry
# checks) to settle - not the install itself, which can legitimately run far longer than this and is
# awaited separately once a promise exists. A generous bound for setup work specifically, so a
# genuinely hung service doesn't leave every caller waiting forever with no way to notice or recover.
_INSTALL_STATUS_WAIT_TIMEOUT_SECONDS = 30
# Bounds resolve_platform_compatibility()'s registry lookup at app startup. A single lookup normally
# takes ~1-1.5s; this is generous headroom for a slow registry, not a normal-case budget - without it,
# a hung connection (rather than an outright error) would block the whole app from ever starting, since
# lifecycle.py awaits every service's check before accepting requests.
_PLATFORM_CHECK_TIMEOUT_SECONDS = 20
_CANCEL_WAIT_TIMEOUT_SECONDS = 30
# Bounds how long a caller waits for install_instance()'s pre-promise setup work (Docker
# image/registry checks) to settle - not the install itself, which can legitimately run far longer
# than this and is awaited separately once a promise exists. A generous bound for setup work
# specifically, so a genuinely hung service doesn't leave every poller waiting forever with no way
# to notice or recover.
_INSTANCE_INSTALL_STATUS_WAIT_TIMEOUT_SECONDS = 30
# update_instance()'s reservation doesn't resolve until _validate_update_options(), _uninstall_instance(),
# AND _install_instance() all complete - unlike install_instance(), its pre-promise phase includes real
# teardown work (stopping/removing containers, unregistering models), whose duration scales with what
# was installed rather than indicating a hang. Budgeted more generously so tearing down a busy instance
# doesn't trip a spurious timeout.
# If _install_instance() itself raises synchronously (before ever producing a promise), the reservation
# also doesn't reject until a same-shaped rollback attempt (_rollback_after_failed_update) finishes -
# so a progress poller can, in that specific case, still time out here even though update_instance()'s
# own caller is legitimately still working through the restore. Rare in practice: this only applies to
# the synchronous-failure path, not the far more common case of _install_instance()'s own promise
# failing later (see the async `on_failure` path, whose rollback isn't bound by this timeout at all).
_INSTANCE_UPDATE_STATUS_WAIT_TIMEOUT_SECONDS = 120
# Bounds cancel_instance_install()'s wait for a triggered DFINFRA-318 rollback to settle. Deliberately
# separate from _INSTANCE_UPDATE_STATUS_WAIT_TIMEOUT_SECONDS above (used for the short pre-promise setup
# wait): a rollback re-runs a full _install_instance(), comparable in duration to the original
# install/update, not to setup - reusing the shorter setup timeout here made this wait time out
# routinely rather than rarely.
_INSTANCE_CANCEL_ROLLBACK_WAIT_TIMEOUT_SECONDS = 300


def default_gpu_cpu_image_name(supported_gpus: Sequence[GpuInfo], get_image: Callable[[bool], DockerImage]) -> str:
    """Shared `get_default_docker_image_name()` body for services whose `_get_image(gpu: bool)` picks a fixed GPU/CPU image pair.

    Selects the GPU image whenever a GPU is supported.
    """
    return get_image(bool(supported_gpus)).name


class Base2Service(Generic[InstalledInfoType, DownloadInfoType], BaseService):  # noqa: UP046
    config: AppSettings
    endpoint_registry: EndpointRegistry
    service_provider: ServiceProvider
    model_downloader: ModelDownloader
    docker_service: DockerService
    services_manager: "ServicesManager"
    models: dict[str, dict[str, Any]]
    models_downloaded: dict[str, DownloadInfoType]
    service_downloaded: bool
    hardware: Hardware
    instances_info: dict[str, Instance[InstalledInfoType]]
    images_download_progress: dict[str, Stream[StreamChunk]]
    models_download_progress: dict[str, Stream[StreamChunk]]
    _log_cache: dict[str, tuple[float, str, float]]
    _installing: set[tuple[str, str]]
    _reconciliation_tasks: dict[str, "asyncio.Task[None]"]
    _crash_poll_state: dict[tuple[str, str], int]
    _failed_models: dict[str, set[str]]
    _warning_tasks: set["asyncio.Task[None]"]
    _platform_unsupported_reason: str | None
    """Cached result of `resolve_platform_compatibility()`, run once at app startup. `None` means
    either the check hasn't found an incompatibility, or `get_default_docker_image_name()` returns
    `None` (no check performed for this service), or the check couldn't complete (fails open)."""

    _persists_model_definitions: bool = True
    """Override to False for services whose ``_generate_instance_config`` never populates ``ModelConfig.definition``
    (e.g. it has no registry snapshot to persist). Otherwise ``load_service`` treats every model as needing a
    backfill save on every startup, forever."""
    _edit_locks: dict[tuple[str, str], asyncio.Lock]
    # GPU utilization of a model being re-installed by an edit, so the reinstall (or a rollback) keeps its current size.
    _edit_kept_gpu_utilization: dict[tuple[str, str], float]

    def __init__(
        self,
        config: AppSettings,
        endpoint_registry: EndpointRegistry,
        service_provider: ServiceProvider,
        model_downloader: ModelDownloader,
        docker_service: DockerService,
        hardware: Hardware,
        services_manager: "ServicesManager",
    ):
        super().__init__()
        self.config = config
        self.endpoint_registry = endpoint_registry
        self.service_provider = service_provider
        self.model_downloader = model_downloader
        self.docker_service = docker_service
        self.hardware = hardware
        self.services_manager = services_manager
        self.models = {}
        self.models_downloaded = {}
        self.service_downloaded = False
        self.instances_info = {"default": Instance(None, None, {}, InstanceConfig())}
        self.models_download_progress = {}
        self.images_download_progress = {}
        self._log_cache = {}
        self._failed_models = {}
        self._warning_tasks = set()
        self._edit_locks = {}
        self._edit_kept_gpu_utilization = {}
        self._platform_unsupported_reason = None
        self._after_init()

    def _after_init(self) -> None:
        """Do some custom initialization."""

    def _init_reconciliation(self) -> None:
        self._reconciliation_tasks = {}
        self._crash_poll_state = {}

    def get_secret_values(self) -> list[str]:
        """Return the values of this service's `password` options, to scrub from Sentry events."""
        names = [field.name for field in self.get_spec().fields if field.type == "password"]
        values: list[str] = []
        for instance in self.instances_info.values():
            option_sets = [instance.config.options] if instance.config.options is not None else []
            if instance.installed is not None:
                option_sets.append(cast("_HasOptions", instance.installed).options)
            values += [str(options.spec[name]) for options in option_sets for name in names if options.spec.get(name)]
        return values

    def _start_reconciliation_task(self, instance: str) -> None:
        """Start (or restart) the background task that detects crashed/stopped model containers and releases their bookkeeping."""
        if instance in self._reconciliation_tasks:
            self._reconciliation_tasks[instance].cancel()

        async def _reconcile_loop() -> None:
            while True:
                await asyncio.sleep(_RECONCILE_INTERVAL_SECONDS)
                info = self.instances_info.get(instance)
                if not info or not info.installed:
                    break
                try:
                    await self._reconcile_instance_models(instance, info.installed)
                except Exception:
                    logger.exception(f"{self.get_id(instance)} reconciliation tick failed")

        self._reconciliation_tasks[instance] = asyncio.create_task(_reconcile_loop())

    def _stop_reconciliation_task(self, instance: str) -> None:
        if instance in self._reconciliation_tasks:
            self._reconciliation_tasks[instance].cancel()
            del self._reconciliation_tasks[instance]

    def _reconcile_container_name(self, instance: str, info: InstalledInfoType, model_info: Any) -> str | None:  # noqa: ANN401
        """Return the container name to inspect for a given model (None => skip this model).

        Per-model-container services (vLLM, llama.cpp, SGLang) return the model's own container. Services that
        never call ``_start_reconciliation_task`` inherit this default, which is not reached in practice.
        """
        raise NotImplementedError

    async def _release_dead_model(self, instance: str, info: InstalledInfoType, model_id: str, model_info: Any) -> None:  # noqa: ANN401
        """Release the bookkeeping of a model whose container was confirmed dead.

        Implementations tear down whatever the service reserved (docker container, VRAM counter/cache),
        drop the model from ``info.models`` and unregister its endpoint. Raising signals a failed teardown:
        the caller keeps the crash-poll state and retries on the next tick instead of releasing prematurely.
        """
        raise NotImplementedError

    async def _reconcile_instance_models(self, instance: str, info: InstalledInfoType) -> None:
        """Detect models whose container is no longer running for good and release their VRAM bookkeeping."""
        for model_id, model_info in cast("_HasModels", info).models.copy().items():
            key = (instance, model_id)
            if key in self._installing:
                self._crash_poll_state.pop(key, None)
                continue

            container_name = self._reconcile_container_name(instance, info, model_info)
            if not container_name:
                self._crash_poll_state.pop(key, None)
                continue

            status = await self.docker_service.get_container_status(container_name)
            is_bad = not status.exists or status.state != "running" or status.health == "unhealthy"

            if not is_bad:
                self._crash_poll_state.pop(key, None)
                continue

            bad_count = self._crash_poll_state.get(key, 0) + 1
            self._crash_poll_state[key] = bad_count
            if bad_count < _RECONCILE_DEAD_THRESHOLD:
                continue

            if key in self._installing or cast("_HasModels", info).models.get(model_id) is not model_info:
                self._crash_poll_state.pop(key, None)
                continue  # uninstalled or reinstalled while we were polling

            if bad_count == _RECONCILE_DEAD_THRESHOLD:
                self._report_dead_container(instance, model_id, status)

            try:
                await self._release_dead_model(instance, info, model_id, model_info)
            except Exception:
                # Keep bookkeeping intact and retry teardown on the next tick rather than releasing VRAM
                # for a container we couldn't confirm is actually gone.
                logger.exception(f"{self.get_id(instance)} failed to tear down dead container for model {model_id}, will retry")
                continue

            self._crash_poll_state.pop(key, None)
            logger.warning(f"{self.get_id(instance)} model {model_id} container is dead, released VRAM bookkeeping")

    def _report_dead_container(self, instance: str, model_id: str, status: ContainerStatus) -> None:
        """Log and report to Sentry a model container found dead for good."""
        error_id = str(uuid.uuid4())
        logger.warning(f"{self.get_id(instance)} model {model_id} container is dead (error_id={error_id})")
        tags = {"service": self.get_type(), "model": model_id, "container_state": status.state if status.exists else "missing"}
        if status.exists:
            tags |= {
                "container_health": status.health,
                "restart_count": str(status.restart_count),
                "exit_code": str(status.exit_code),
                "oom_killed": str(status.oom_killed).lower(),
            }
        failure = UpstreamError(f"{self.get_type()} model container died")
        capture_exception(failure, error_id=error_id, tags=tags, fingerprint=["container-died", self.get_type()])

    async def _get_docker_logs(self, container_name: str) -> str:
        """Return raw docker logs output, with TTL caching per container.

        Tailed to the last `_DOCKER_LOGS_TAIL_LINES` lines: this is only ever used to find a
        single startup-time capacity line (see `_get_max_concurrency_from_logs`), and an
        untailed `docker logs` on a long-running container can read hundreds of MB into memory
        on every restart. A failed or empty read (non-zero exit code, or no output at all —
        e.g. a race right after the container restarts, before the backend has logged anything)
        is cached only briefly (`_FAILED_LOG_CACHE_TTL`) rather than for the full TTL, so a
        container stuck in that state doesn't spawn a `docker logs` subprocess on every poll
        tick, while a real startup log that lands moments later is still picked up quickly.
        """
        cached = self._log_cache.get(container_name)
        if cached and (time.monotonic() - cached[0]) < cached[2]:
            return cached[1]
        result = await Utils.run_command(["docker", "logs", "--tail", str(_DOCKER_LOGS_TAIL_LINES), container_name])
        raw = result.stdout + result.stderr
        if result.exit_code != 0:
            logger.warning("docker logs failed for %r (exit %d): %s", container_name, result.exit_code, result.stderr)
        ttl = _LOG_CACHE_TTL if result.exit_code == 0 and raw else _FAILED_LOG_CACHE_TTL
        self._log_cache[container_name] = (time.monotonic(), raw, ttl)
        return raw

    async def _get_max_concurrency_from_logs(
        self, container_name: str, backend_name: str, parse: Callable[[str], int | None]
    ) -> int | None:
        """Return a backend's reported max concurrency for a running container, or None if unavailable."""
        try:
            raw = await self._get_docker_logs(container_name)
        except Exception:
            logger.exception("Failed to read docker logs for %r; %s capacity will be unknown", container_name, backend_name)
            return None
        return parse(raw)

    async def _resolve_model_capacity(
        self,
        instance: str,
        model_id: str,
        backend_name: str,
        container_name: str,
        restarted: bool,
        parse: Callable[[str], int | None],
    ) -> tuple[int | None, bool]:
        """Determine a model's concurrency capacity, re-parsing logs only if the container was actually (re)started.

        A container that's still running from before this process started never rewrote its
        startup capacity line, so re-reading its logs can't recover it (see `_get_docker_logs`);
        the last persisted value is reused instead, which is safe because an unchanged config
        implies an unchanged real capacity. If there's no persisted value either - e.g. an
        orphan-adopted container (see `_try_adopt_orphaned_container`) that this process never
        started and has no prior config entry for - the startup line may still be sitting in the
        container's tailed log, so it's worth a best-effort read before giving up.
        """
        if restarted:
            self._log_cache.pop(container_name, None)
            capacity = await self._get_max_concurrency_from_logs(container_name, backend_name, parse)
            capacity_known = capacity is not None
            if capacity is None:
                logger.warning(
                    "Could not determine %s concurrency for %r from its startup logs after a real (re)start; "
                    "capacity will be treated as unknown and this instance will be deprioritized for routing.",
                    backend_name,
                    model_id,
                )
            return capacity, capacity_known

        capacity, capacity_known = self._get_persisted_model_capacity(instance, model_id)
        if capacity is not None or capacity_known:
            logger.debug("Container for %r wasn't (re)started; reusing last known %s concurrency %s.", model_id, backend_name, capacity)
            return capacity, capacity_known

        recovered = await self._get_max_concurrency_from_logs(container_name, backend_name, parse)
        if recovered is not None:
            logger.info(
                "Recovered %s concurrency %s for %r from its still-running container's logs (no prior known value).",
                backend_name,
                recovered,
                model_id,
            )
            return recovered, True

        logger.warning(
            "Could not determine %s concurrency for %r: container already running, no prior known value, "
            "and its startup capacity line is no longer in the tailed logs.",
            backend_name,
            model_id,
        )
        return capacity, capacity_known

    def load_default_models(self, instance: str) -> None:
        """Load default models to instance."""

    @abstractmethod
    def get_type(self) -> str:
        """Return the type."""

    @abstractmethod
    def get_description(self) -> str:
        """Return the service description."""

    def check_instance_exists(self, instance: str) -> None:
        """Check is instance exists."""
        if not self.instances_info.get(instance):
            raise HTTPException(404, "Instance doesn't exist.")

    def is_model_installed_in_other_instance(self, instance: str, model: str) -> bool:
        """Check is model installed in other instance."""
        return any(model in getattr(self.instances_info[i].installed, "models", []) for i in self.instances_info if i != instance)

    def get_instance_info(self, instance: str) -> Instance[InstalledInfoType]:
        """Return instance info."""
        instance_info = self.instances_info.get(instance)
        if not instance_info:
            raise HTTPException(404, "Instance doesn't exist.")
        return instance_info

    def _pop_installing_if_current(
        self, instance: str, reservation: InstallingModel | InstallingInstance[InstalledInfoType], *, model_id: str | None = None
    ) -> None:
        """Remove `reservation` if it's still this exact one and the instance still exists.

        Covers both reservation kinds: pass `model_id` to clear a model's slot in
        `installing_model_progress`, or omit it to clear the instance's own single `installing` slot.

        No-ops otherwise instead of raising, since callers use this before reject()/resolve() and
        must not be blocked by either a superseded reservation or an already-uninstalled instance.
        """
        instance_info = self.instances_info.get(instance)
        if instance_info is None:
            return
        if model_id is not None:
            progress = instance_info.installing_model_progress
            if progress.get(model_id) is reservation:
                del progress[model_id]
        elif instance_info.installing is reservation:
            instance_info.installing = None

    @staticmethod
    async def _wait_for_install_setup(
        model_id: str, awaitable: Awaitable[PromiseWithProgress[InstallModelOut, StreamChunk]]
    ) -> PromiseWithProgress[InstallModelOut, StreamChunk]:
        """Await a reservation's pre-promise setup, converting a timeout into a clean 504."""
        try:
            return await asyncio.wait_for(awaitable, timeout=_INSTALL_STATUS_WAIT_TIMEOUT_SECONDS)
        except TimeoutError:
            raise HTTPException(504, f"Model {model_id} is still preparing to install; try again shortly.") from None

    async def get_model_install_progress(self, instance: str, model: str) -> PromiseWithProgress[InstallModelOut, StreamChunk]:
        """Return actually installing models."""
        installing = self.get_instance_info(instance).installing_model_progress.get(model)

        if not installing:
            raise HTTPException(404, "This model is not installing now.")

        return await self._wait_for_install_setup(model, installing.wait_ready())

    async def cancel_model_install(self, instance: str, model_id: str) -> None:
        """Cancel an in-progress model install, stopping the underlying Docker image pull."""
        installing = self.get_instance_info(instance).installing_model_progress.get(model_id)

        if not installing:
            raise HTTPException(404, f"Model {model_id} is not installing now.")

        # Always attempt to cancel the pending setup call first, unconditionally - cancelling an
        # already-finished task is a harmless no-op. Deciding which of "still setting up" vs
        # "real promise exists" applies *before* attempting cancellation would race: the setup
        # work can finish (and install_model() resolve the reservation) in the gap between that
        # check and acting on it, silently leaving the real install running untouched while this
        # reports success. Cancelling first, then waiting for the reservation to settle, means we
        # always act on the real outcome instead of a stale snapshot of it.
        assert installing.pending_task is not None
        installing.pending_task.cancel()

        try:
            promise = await asyncio.wait_for(installing.wait_ready(), timeout=_CANCEL_WAIT_TIMEOUT_SECONDS)
        except TimeoutError:
            # We genuinely don't know whether the cancel above landed or the pending setup is just
            # slow - leaving the reservation untouched lets a later call (retry cancel, or simply
            # waiting) observe the real eventual outcome instead of us guessing and potentially
            # hiding a still-running install by clearing its tracking prematurely.
            raise HTTPException(504, f"Timed out waiting to confirm cancellation for model {model_id}.") from None
        except Exception as e:
            # wait_ready() re-raising here means the pending setup work was cancelled (reject()
            # converts any CancelledError into a clean HTTPException(409) before storing it, so this
            # never surfaces as a raw one from that) or failed on its own - either way, nothing
            # further to cancel. A raw CancelledError reaching this point instead can only mean
            # this call's own task is being cancelled (e.g. its request disconnected); `except
            # Exception` doesn't catch that, so it propagates naturally instead of being treated
            # as if the cancel we asked for had completed.
            #
            # Log the "failed on its own" case specifically - it's otherwise indistinguishable from
            # an ordinary successful cancellation, and a real setup failure that happens to coincide
            # with someone clicking cancel would vanish without a trace.
            if not (isinstance(e, HTTPException) and e.status_code == 409):
                logger.exception(f"{self.get_id(instance)} pending setup for model {model_id} failed independently of the cancel request")
            self._pop_installing_if_current(instance, installing, model_id=model_id)
            return

        # Cancel the real pull work and every chained promise (cancel() propagates down the chain),
        # plus the chunk reader, then await them all so no task is left pending and GC'd mid-run.
        # promise.tasks() also returns the chained promise's task (it is registered as a child).
        promise.cancel()
        assert installing.task is not None
        installing.task.cancel()
        tasks = [*promise.tasks(), installing.task]
        await asyncio.gather(*tasks, return_exceptions=True)

        promise.progress.close()

        self._pop_installing_if_current(instance, installing, model_id=model_id)

    async def get_instance_install_progress(self, instance: str) -> PromiseWithProgress[InstallServiceOut, StreamChunk]:
        """Return actually installing service."""
        installing = self.get_instance_info(instance).installing

        if not installing:
            raise HTTPException(404, "This service is not installing now.")

        try:
            return await asyncio.wait_for(installing.wait_chained(), timeout=installing.timeout_seconds)
        except TimeoutError:
            raise HTTPException(504, f"Service {self.get_id(instance)} is still preparing to install; try again shortly.") from None

    async def cancel_instance_install(self, instance: str) -> None:
        """Cancel an in-progress service instance install or update, stopping the underlying Docker work.

        Unlike `cancel_model_install()`, there's no `pending_task` to cancel up front: the pre-promise
        setup phase of `_install_instance()` is short and already bounded by `Utils.run_command`'s own
        kill-on-cancel, so a cancel request arriving during that window simply waits for it to settle
        (bounded by the reservation's own `timeout_seconds`) instead of interrupting it.

        Cancelling an update may trigger DFINFRA-318's automatic rollback - this method waits for it to
        settle before returning, so a caller never observes "not installing" before the true final
        state (reverted, or fully uninstalled) is known.
        """
        installing = self.get_instance_info(instance).installing

        if not installing:
            raise HTTPException(404, f"Service {self.get_id(instance)} is not installing now.")

        try:
            promise = await asyncio.wait_for(installing.wait_ready(), timeout=installing.timeout_seconds)
        except TimeoutError:
            # Same reasoning as cancel_model_install(): we don't know whether setup is genuinely stuck
            # or just slow, so leave the reservation untouched for a retry or a later read to observe
            # the real eventual outcome instead of guessing.
            raise HTTPException(504, f"Timed out waiting to confirm cancellation for {self.get_id(instance)}.") from None
        except Exception as e:
            # wait_ready() re-raising here means the pre-promise setup failed on its own or was itself
            # rejected (reject() already turns any CancelledError into a clean HTTPException(409)) -
            # either way, there's no real promise left to cancel.
            if not (isinstance(e, HTTPException) and e.status_code == 409):
                logger.exception(f"{self.get_id(instance)} install/update setup failed independently of the cancel request")
            self._pop_installing_if_current(instance, installing)
            return

        # Cancel the real install/update work and every chained promise (cancel() propagates down the
        # chain), plus the chunk reader, then await them all so no task is left pending and GC'd mid-run.
        promise.cancel()
        assert installing.task is not None
        installing.task.cancel()
        tasks = [*promise.tasks(), installing.task]
        await asyncio.gather(*tasks, return_exceptions=True)

        promise.progress.close()

        # `chained_promise` isn't a child of `promise` for an update (see resolve_with_chain()), so
        # cancelling `promise` doesn't cancel the rollback too - it keeps running here. This is the
        # only place that can wait for a cancellation-triggered rollback's outcome. For a plain
        # install, `chained_promise` is already a child of `promise` and already finished above.
        #
        # Shielded: `asyncio.wait_for()` cancels whatever it's timing out on - and since
        # `chained_promise._future` is shared with any other reader (a later cancel retry, or a
        # progress-stream reattachment), an unshielded timeout here would cancel that shared future
        # too, permanently poisoning it for everyone else even though the rollback itself keeps running
        # unaffected in the background. `shield()` lets our own wait time out without touching it.
        assert installing.chained_promise is not None
        try:
            await asyncio.wait_for(
                asyncio.shield(installing.chained_promise.wait()), timeout=_INSTANCE_CANCEL_ROLLBACK_WAIT_TIMEOUT_SECONDS
            )
        except TimeoutError:
            raise HTTPException(504, f"Timed out waiting for {self.get_id(instance)}'s cancellation to fully settle.") from None
        except asyncio.CancelledError:
            # chained_promise's future routinely ends up cancelled here (see DFINFRA-318's the_func) -
            # expected, not a sign this request was cancelled. Only propagate this call's own cancellation.
            if is_own_cancellation():
                raise  # pragma: no cover - coverage.py can't trace a bare raise here; see test above
        except Exception:
            # Already recorded via on_failure()/_rollback_after_failed_update()'s own logging.
            pass

        self._pop_installing_if_current(instance, installing)

    def is_installed(self, instance: str) -> bool:
        """Check whether service is installed."""
        return self.get_instance_info(instance).installed is not None

    def get_downloaded(self) -> bool:
        """Get service downloaded info."""
        return self.service_downloaded

    def _restore_model_definition(self, instance: str, model_id: str, definition: dict[str, Any]) -> None:
        """Re-register a model definition missing from the live registry, from a persisted snapshot. No-op by default."""

    def _get_persisted_model_definition(self, instance: str, model_id: str) -> dict[str, Any] | None:
        """Look up the last-persisted definition snapshot for a model no longer in the live catalog."""
        persisted = self.instances_info.get(instance)
        if not persisted:
            return None
        return next((m.definition for m in persisted.config.models or [] if m.model_id == model_id), None)

    def _get_persisted_model_capacity(self, instance: str, model_id: str) -> tuple[int | None, bool]:
        """Look up the last-persisted capacity for a model, from a prior successful detection."""
        persisted = self.instances_info.get(instance)
        if not persisted:
            return None, False
        match = next((m for m in persisted.config.models or [] if m.model_id == model_id), None)
        return (match.capacity, match.capacity_known) if match else (None, False)

    @staticmethod
    def _parse_gpu_priority(spec: dict[str, Any] | None) -> float | None:
        """Read `gpu_priority` from a raw spec, where it may be a string (e.g. "0.5") since specs are stored as request bodies."""
        raw = (spec or {}).get("gpu_priority")
        if raw is None:
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    def _get_persisted_model_gpu_memory_utilization(self, instance: str, model_id: str, gpu_priority: float | None) -> float | None:
        """Look up the last-persisted auto-computed GPU utilization for a model, from a prior successful install.

        Reusing this avoids spurious docker-compose diffs on reload: the auto-tuned default is sized off
        currently-free VRAM (see `VllmService._get_default_gpu_memory_utilization`), which keeps shrinking
        once the model's own container is already running and holding memory, making every reload look like
        a config change and forcing an unnecessary container restart.

        Only reused while `gpu_priority` matches the one it was computed for - otherwise a changed priority
        (e.g. an edit of a model that failed to load, whose persisted entry survives) would be silently ignored.
        An options edit that left the priority unchanged also lands here, via the value kept across its uninstall.
        """
        if (kept := self._edit_kept_gpu_utilization.get((instance, model_id))) is not None:
            return kept
        persisted = self.instances_info.get(instance)
        if not persisted:
            return None
        match = next((m for m in persisted.config.models or [] if m.model_id == model_id), None)
        if match is None:
            return None
        persisted_priority = self._parse_gpu_priority(match.options.spec)
        if persisted_priority != gpu_priority or (
            persisted_priority is None and (match.options.spec or {}).get("gpu_priority") is not None
        ):
            return None
        return match.gpu_memory_utilization

    def _record_warning_in_background(self, message: str, *, instance: str | None = None, model_id: str | None = None) -> None:
        """Fire-and-forget a warning write from a synchronous callback, e.g. a promise's on_error."""
        task = asyncio.create_task(self.service_provider.add_warning(self.get_type(), message, instance=instance, model_id=model_id))
        self._warning_tasks.add(task)
        task.add_done_callback(self._on_warning_task_done)

    def _on_warning_task_done(self, task: "asyncio.Task[None]") -> None:
        self._warning_tasks.discard(task)
        if not task.cancelled() and (exc := task.exception()) is not None:
            logger.exception("Failed to record warning in background", exc_info=exc)

    async def drain_warning_tasks(self) -> None:
        """Await pending fire-and-forget warning writes so shutdown doesn't drop them."""
        if self._warning_tasks:
            await asyncio.gather(*self._warning_tasks, return_exceptions=True)

    async def load_model(self, instance: str, model: ModelConfig, *, derive_container_name: bool = False) -> None:
        """Load single model.

        derive_container_name re-derives the name instead of pinning the persisted one - set for an
        explicit reinstall's own reload, not for restoring a config as-is (startup, rollback).
        """
        logger.info(f"{self.get_id(instance)} loading model {model.model_id}")
        try:
            if model.model_id not in self.models.get(instance, {}) and model.definition:
                logger.warning(
                    f"{self.get_id(instance)} model {model.model_id} missing from current registry, restoring from persisted definition"
                )
                self._restore_model_definition(instance, model.model_id, model.definition)

            async def on_success(data: InstallModelOut) -> InstallModelOut:
                return data

            # Reserve like install_model() does, so a concurrent POST /install for this model_id
            # joins this load instead of racing it or getting a fabricated "already installing"
            # response - but with no-op bookkeeping callbacks, since this method already has its own
            # separate success/failure handling below. Routing through install_model() directly
            # would run both, double-recording the warning on failure.
            container_name_override = None if derive_container_name else model.container_name
            promise = await self._reserve_and_install(
                instance, model.model_id, model.options, on_success, lambda _e: None, container_name_override=container_name_override
            )
            await promise.wait()
        except Exception as exc:
            logger.exception(f"{self.get_id(instance)} get error while loading model {model.model_id}")
            self._failed_models.setdefault(instance, set()).add(model.model_id)
            try:
                await self.service_provider.add_warning(
                    self.get_type(),
                    f"Model '{model.model_id}' failed to load on instance '{instance}': {exc}",
                    instance=instance,
                    model_id=model.model_id,
                )
            except Exception:
                logger.exception(f"{self.get_id(instance)} failed to record warning for model {model.model_id}")
            return

        if instance_failures := self._failed_models.get(instance):
            instance_failures.discard(model.model_id)
        try:
            await self.service_provider.dismiss_warnings_matching(self.get_type(), instance=instance, model_id=model.model_id)
        except Exception:
            logger.exception(f"{self.get_id(instance)} failed to dismiss warnings for model {model.model_id} after successful load")

    async def load_instance(self, instance: str, instance_data: InstanceConfig) -> None:
        """Load instance of service."""
        if not instance_data.options:
            return

        self.load_default_models(instance)

        if instance_data.custom:
            for custom in instance_data.custom:
                model_id = custom.data.get("id", custom.id)
                try:
                    self._add_custom_model(instance, custom)
                except Exception as exc:
                    # Registering one custom model must not take the whole instance - and, since
                    # load_service gathers all of a service's instances without return_exceptions=True,
                    # every other instance of this service too - down with it. Matches load_model's
                    # per-model warning-and-continue pattern just above.
                    logger.exception(f"{self.get_id(instance)} failed to register custom model {model_id} while loading")
                    try:
                        await self.service_provider.add_warning(
                            self.get_type(),
                            f"Custom model '{model_id}' failed to load on instance '{instance}': {exc}",
                            instance=instance,
                            model_id=model_id,
                        )
                    except Exception:
                        logger.exception(f"{self.get_id(instance)} failed to record warning for custom model {model_id}")

        promise = await self.install_instance(instance, instance_data.options, instance_data, save=False, run_after_install=False)
        await promise.wait()
        logger.info(f"{self.get_id(instance)} service checked")
        tasks = [asyncio.create_task(self.load_model(instance, model)) for model in instance_data.models or []]
        await asyncio.gather(*tasks)
        await self._after_install(instance)

    async def load_service(self, config: ServiceRawConfig) -> None:
        """Load service using the config."""
        cfg = ServiceConfig(**config)
        self.models_downloaded = ({key: self._load_download_info(value) for key, value in cfg.downloaded.items()}) if cfg.downloaded else {}
        self.service_downloaded = cfg.service_downloaded or False

        msg = f"{self.get_type()} service installed."
        logger.info(msg)
        if cfg.instances:
            needs_backfill = self._persists_model_definitions and any(
                m.definition is None for instance in cfg.instances.values() for m in instance.models or []
            )
            tasks = [asyncio.create_task(self.load_instance(name, instance)) for name, instance in cfg.instances.items()]
            await asyncio.gather(*tasks)
            if needs_backfill:
                logger.info(f"{self.get_type()} backfilling model definitions into persisted config")
                await self._save()
            elif unpinned := self._instances_with_unpinned_container_names(cfg.instances):
                logger.info(f"{self.get_type()} backfilling container names into persisted config")
                await self._backfill_container_names(unpinned)

    async def _backfill_container_names(self, instances: list[str]) -> None:
        """Save to pin the container names resolved on load, without letting a failed save fail the load itself.

        The service did load fine, so a failed save must not surface as "Service failed to load" - it gets its own
        per-instance warning instead (a service-level one would be auto-dismissed right after the load succeeds),
        since the names stay unpinned, and a container_name_suffix change would rename them, until the next save.
        """
        try:
            await self._save()
        except Exception as exc:
            logger.exception(f"{self.get_type()} failed to save container names into persisted config")
            for instance in instances:
                try:
                    await self.service_provider.add_warning(
                        self.get_type(),
                        f"Container names of instance '{instance}' could not be saved to services.json: {exc}. Until the next "
                        "successful save, changing container_name_suffix and restarting will rename its containers.",
                        instance=instance,
                    )
                except Exception:
                    logger.exception(f"{self.get_id(instance)} failed to record warning for unsaved container names")

    def _instances_with_unpinned_container_names(self, persisted: dict[str, InstanceConfig]) -> list[str]:
        """Instances for which loading resolved a container name the persisted config doesn't pin yet.

        Non-empty for a services.json written before container names were pinned - without saving right away,
        the name would stay unpinned until some unrelated save, so a container_name_suffix change in the
        meantime would rename the container on the next restart. Compares against the freshly generated
        config rather than just checking for None, so services with no container at all don't re-save on
        every startup.
        """
        unpinned: list[str] = []
        for name, persisted_instance in persisted.items():
            info = self.instances_info.get(name)
            if info is None or info.installed is None:
                continue
            live = self._generate_instance_config(name, info.installed, info.config.custom)
            unpinned_model_ids = {m.model_id for m in persisted_instance.models or [] if m.container_name is None}
            if (persisted_instance.container_name is None and live.container_name is not None) or any(
                m.model_id in unpinned_model_ids and m.container_name is not None for m in live.models or []
            ):
                unpinned.append(name)
        return unpinned

    @abstractmethod
    def _load_download_info(self, data: dict[str, Any]) -> DownloadInfoType:
        pass

    async def _save(self) -> None:
        instances_config = {}
        for instance_name, instance in self.instances_info.items():
            generated = self._generate_instance_config(instance_name, instance.installed, instance.config.custom)
            generated.models = self._preserve_failed_models(instance_name, generated.models or [])
            instance.config.models = generated.models
            instance.config.options = generated.options
            instances_config[instance_name] = generated
        cfg = self.service_config(instances_config)
        await self.service_provider.save_service_config(self.get_type(), cfg.model_dump())

    def _preserve_failed_models(self, instance: str, live_models: list[ModelConfig]) -> list[ModelConfig]:
        """Keep the persisted entry for a model that failed to (re)load.

        Otherwise it would get dropped the moment anything else triggers a save. Explicit uninstalls
        clear the failure marker, so they are never resurrected here.
        """
        failed_ids = self._failed_models.get(instance)
        if not failed_ids:
            return live_models
        live_ids = {m.model_id for m in live_models}
        persisted_by_id = {m.model_id: m for m in self.instances_info[instance].config.models or []}
        preserved = [persisted_by_id[model_id] for model_id in failed_ids if model_id not in live_ids and model_id in persisted_by_id]
        return [*live_models, *preserved]

    @abstractmethod
    def _generate_instance_config(self, instance: str, info: InstalledInfoType | None, custom: list[CustomModel] | None) -> InstanceConfig:
        """Generate instance config."""

    def service_config(self, instances_config: dict[str, InstanceConfig]) -> ServiceConfig:
        """Generate service config."""
        return ServiceConfig(instances=instances_config, downloaded=self.models_downloaded, service_downloaded=self.service_downloaded)

    async def install_instance(
        self,
        instance: str,
        options: InstallServiceIn,
        instance_config: InstanceConfig | None = None,
        save: bool = True,
        run_after_install: bool = True,
    ) -> PromiseWithProgress[InstallServiceOut, StreamChunk]:
        """Install the service.

        `run_after_install=False` lets a caller that still needs to load persisted models onto the
        fresh instance (see `load_instance`) defer `_after_install` until after that happens - running
        it here for a restore would merge a live catalog against an empty `installed.models`, causing
        every not-yet-reloaded model to look uninstalled and get dropped instead of preserved as stale.
        """
        if self.instances_info and self.instances_info.get(instance):
            if self.instances_info[instance].installed:
                raise HTTPException(status_code=400, detail=f"Service {self.get_id(instance)} on {instance} instance already installed")
            if self.instances_info[instance].installing:
                raise HTTPException(status_code=400, detail=f"Service {self.get_id(instance)} on {instance} instance already installing")

        # Reserve the slot synchronously (no `await` before this point) so a concurrent call can
        # never observe "not installing" once this one has started, no matter how long the
        # service's own `_install_instance()` takes to actually produce the real promise.
        reservation: InstallingInstance[InstalledInfoType] = InstallingInstance(_INSTANCE_INSTALL_STATUS_WAIT_TIMEOUT_SECONDS)
        self.instances_info[instance] = Instance(None, reservation, {}, instance_config or InstanceConfig())

        async def func(data: InstalledInfoType) -> InstallServiceOut:
            self.instances_info[instance].installed = data
            if run_after_install:
                await self._after_install(instance)
            if save:
                await self._save()
            self.instances_info[instance].installing = None
            try:
                await self.service_provider.dismiss_warnings_matching_any(self.get_type(), [(None, None), (instance, None)])
            except Exception:
                logger.exception(f"{self.get_id(instance)} failed to dismiss warnings after successful install")
            return InstallServiceOut(status="OK")

        def on_error(e: Exception) -> None:
            self.instances_info[instance].installing = None
            self._record_warning_in_background(f"Service instance '{instance}' failed to install: {e}", instance=instance)

        try:
            promise = await self._install_instance(instance, options, instance_config.container_name if instance_config else None)
        except BaseException as e:
            self.instances_info[instance].installing = None
            reservation.reject(e)
            raise

        return reservation.resolve(promise, func, on_error)

    async def update_instance(self, instance: str, options: InstallServiceIn) -> PromiseWithProgress[InstallServiceOut, StreamChunk]:
        """Update an installed service by re-applying the install flow with new options.

        The underlying container is recreated to pick up the new options; already-installed models are
        preserved (their weights stay on disk) and re-registered against the recreated container.

        If the reinstall step then fails or is cancelled, the previous installation is automatically
        restored from the options/models captured here, just before the uninstall - see
        `_rollback_after_failed_update`. The restore is awaited as part of the same promise the caller
        is already watching, so "update failed" is never reported before the instance's true final state
        - reverted, or genuinely gone - is actually known; a caller isn't left thinking the operation is
        over while a recovery is still quietly happening behind it. Without a rollback at all, a failed
        or cancelled update would otherwise leave the instance permanently uninstalled: the old
        installation is already torn down by the time the new one fails, and nothing else brings it back.
        """
        info = self.get_instance_info(instance)
        if info.installing:
            raise HTTPException(status_code=400, detail=f"Service {self.get_id(instance)} on {instance} instance is currently installing")
        if not info.installed:
            raise HTTPException(status_code=400, detail=f"Service {self.get_id(instance)} on {instance} instance is not installed")

        previous_config = self._generate_instance_config(instance, info.installed, info.config.custom)
        previous_options = previous_config.options
        preserved_models = self._preserve_failed_models(instance, previous_config.models or [])

        # Reserve the slot synchronously (no `await` before this point) so a concurrent call can
        # never observe "not installing" once this one has started, no matter how long validation,
        # the uninstall, the service's own `_install_instance()`, or a subsequent rollback take to run.
        reservation: InstallingInstance[InstalledInfoType] = InstallingInstance(_INSTANCE_UPDATE_STATUS_WAIT_TIMEOUT_SECONDS)
        info.installing = reservation

        async def on_success(data: InstalledInfoType) -> InstallServiceOut:
            await self._finish_successful_reinstall(instance, data, preserved_models, context="update")
            return InstallServiceOut(status="OK")

        async def on_failure(e: BaseException) -> None:
            """Await the rollback (if applicable) before the caller's promise settles with `e`.

            Only attempted if the reinstall itself never actually completed (`installed` is still
            `None`) - if `on_success` already set it before failing later (e.g. `_after_install`/`_save`
            raising), the new instance is genuinely up and running, and rolling back would tear down a
            working install for nothing instead of recovering a reinstall that never completed.
            """
            self._record_warning_in_background(f"Service instance '{instance}' failed to update: {e}", instance=instance)
            if previous_options is not None and self.instances_info[instance].installed is None:
                reverted = await self._rollback_after_failed_update(
                    instance, previous_options, preserved_models, e, previous_container_name=previous_config.container_name
                )
                self._append_rollback_outcome(e, reverted)
            else:
                self.instances_info[instance].installing = None

        async def the_func(_stream: Stream[StreamChunk]) -> InstallServiceOut:
            # Mirrors `PromiseWithProgress.next()`'s own try/except shape (both the wait and the
            # success callback inside one try, so a failure from either routes to on_failure), except
            # on_failure is awaited here rather than fired as a synchronous, side-effect-only callback -
            # that's what lets it run the rollback to completion before this promise (and hence the
            # caller) settles.
            try:
                data = await promise.wait()
                return await on_success(data)
            except BaseException as e:
                await on_failure(e)
                raise

        uninstalled = False
        try:
            await self._validate_update_options(options)
            await self._uninstall_instance(instance, UninstallServiceIn(purge=False))
            uninstalled = True
            self._failed_models.pop(instance, None)
            promise = await self._install_instance(instance, options)
        except BaseException as e:
            # Only attempt a rollback once the previous installation has actually been torn down - a
            # rejection from `_validate_update_options` leaves it untouched, so there's nothing to
            # restore. Awaited inline (this whole method is itself awaited by its caller) so the caller
            # doesn't get a response before the instance's true final state is known.
            if uninstalled and previous_options is not None:
                reverted = await self._rollback_after_failed_update(
                    instance, previous_options, preserved_models, e, previous_container_name=previous_config.container_name
                )
                self._append_rollback_outcome(e, reverted)
            else:
                self.instances_info[instance].installing = None
            reservation.reject(e)
            raise

        chained_promise: PromiseWithProgress[InstallServiceOut, StreamChunk] = PromiseWithProgress(func=the_func)
        chained_promise.progress = promise.progress
        return reservation.resolve_with_chain(promise, chained_promise)

    async def _finish_successful_reinstall(
        self, instance: str, data: InstalledInfoType, preserved_models: list[ModelConfig], *, context: str
    ) -> None:
        """Shared post-reinstall bookkeeping for a reinstall that just succeeded.

        Used by both the original update's own reinstall and a rollback's restore of the previous
        installation - `context` ("update"/"rollback") distinguishes the two in log output, and also
        picks whether preserved models re-derive their container name ("update") or keep their pinned
        one ("rollback"). `installing` is always cleared, even if a step here fails, so neither caller
        can leave the instance stuck permanently rejecting every future install/update as "currently
        installing".
        """
        try:
            self.instances_info[instance].installed = data
            for model in preserved_models:
                await self.load_model(instance, model, derive_container_name=context == "update")
            await self._after_install(instance)
            await self._save()
        finally:
            self.instances_info[instance].installing = None
        try:
            await self.service_provider.dismiss_warnings_matching_any(self.get_type(), [(None, None), (instance, None)])
        except Exception:
            logger.exception(f"{self.get_id(instance)} failed to dismiss warnings after successful {context}")

    async def _rollback_after_failed_update(
        self,
        instance: str,
        previous_options: InstallServiceIn,
        preserved_models: list[ModelConfig],
        original_error: BaseException,
        *,
        previous_container_name: str | None,
    ) -> bool:
        """Best-effort restore of the pre-update installation after `update_instance()`'s reinstall step failed.

        Re-runs `_install_instance` with the options captured before the uninstall and reloads the
        preserved models onto the recreated container - mirroring `update_instance()`'s own success path.
        `previous_container_name` pins the instance container back to its pre-update name, just like the
        preserved models keep theirs - a restore must not pick up a suffix changed since the original install.
        If the restore itself fails too, the instance is genuinely gone: that's surfaced as a distinct,
        more prominent warning so it isn't mistaken for the original, potentially transient, update
        failure.

        Returns True if the previous installation was successfully restored, False if the rollback
        itself also failed - the caller uses this to tell the client which of the two actually happened
        (see `_append_rollback_outcome`).
        """
        logger.warning(f"{self.get_id(instance)} update failed ({original_error}); attempting to restore the previous installation")
        try:
            promise = await self._install_instance(instance, previous_options, previous_container_name)
            data = await promise.wait()
        except asyncio.CancelledError:
            # Never swallow cancellation (e.g. app shutdown) - but still clear `installing` first, so
            # this instance isn't left permanently rejecting every future install/update just because
            # this attempt never got the chance to finish either way.
            self.instances_info[instance].installing = None
            raise
        except Exception as rollback_error:
            logger.exception(f"{self.get_id(instance)} rollback after a failed update also failed; instance left fully uninstalled")
            self.instances_info[instance].installing = None
            self._record_warning_in_background(
                f"Service instance '{instance}' failed to update ({original_error}) and the automatic rollback to its previous "
                f"configuration also failed ({rollback_error}). The service is now fully uninstalled - it needs a fresh install, "
                "not a retry.",
                instance=instance,
            )
            return False

        try:
            await self._finish_successful_reinstall(instance, data, preserved_models, context="rollback")
        except Exception:
            # The rollback's own reinstall succeeded (the instance IS running the previous
            # configuration again) - only the bookkeeping after it failed. Distinguished from the
            # "rollback itself failed" case above since it's a much smaller problem: the caller still
            # gets told the instance was reverted, plus a heads-up that something needs a look.
            logger.exception(f"{self.get_id(instance)} rollback reinstall succeeded but its post-install bookkeeping failed")
            self._record_warning_in_background(
                f"Service instance '{instance}' failed to update ({original_error}) and was reverted to its previous "
                "installation, but finishing the restore ran into a problem afterwards - check the service's state.",
                instance=instance,
            )
            return True

        self._record_warning_in_background(
            f"Service instance '{instance}' failed to update ({original_error}) and was automatically reverted to its previous "
            "configuration.",
            instance=instance,
        )
        return True

    @staticmethod
    def _append_rollback_outcome(e: BaseException, reverted: bool) -> None:
        """Mutate `e`'s message in place so a client displaying just `str(e)` learns the rollback outcome.

        The API layer (and the WebUI, which shows the raised exception's message verbatim in an error
        toast) otherwise only ever sees the original, potentially cryptic reinstall failure - with no
        indication that anything was automatically recovered, or that the instance is now fully gone and
        needs a fresh install rather than a retry. Mutating `.args`/`.detail` works uniformly across
        exception types without assuming a particular constructor signature, and preserves the original
        type (e.g. an `HTTPException`'s `status_code`) for anything downstream that still inspects it.
        """
        suffix = (
            "(automatically reverted to the previous configuration)"
            if reverted
            else "(automatic rollback also failed - the service is now fully uninstalled; a fresh install is required)"
        )
        if isinstance(e, HTTPException):
            e.detail = f"{e.detail} {suffix}"
        elif e.args:
            e.args = (f"{e.args[0]} {suffix}", *e.args[1:])
        else:
            e.args = (suffix,)

    async def _validate_update_options(self, options: InstallServiceIn) -> None:
        """Validate new options before update_instance tears down the existing installation.

        No-op by default. Override for services whose `_install_instance` can reject the new
        options (e.g. an unavailable Docker image tag) — without this, update_instance's
        uninstall-then-reinstall ordering would destroy a working installation before the
        rejection surfaces, since `_install_instance` only validates after being invoked.
        """

    async def _after_install(self, instance: str) -> None:
        """Run right after `instance` becomes installed, on both fresh install and update.

        No-op by default. Override for services that should refresh derived state (e.g. a live
        model catalog) as soon as connection details are known, instead of waiting for a manual
        refresh.
        """

    @abstractmethod
    async def _install_instance(
        self, instance: str, options: InstallServiceIn, container_name_override: str | None = None
    ) -> PromiseWithProgress[InstalledInfoType, StreamChunk]:
        """Install service. container_name_override, if set, must be used verbatim - see InstanceConfig.container_name."""

    async def uninstall_instance(self, instance: str, options: UninstallServiceIn) -> None:
        """Uninstall the service."""
        await self._uninstall_instance(instance, options)
        self._failed_models.pop(instance, None)
        await self.service_provider.dismiss_warnings_for_instance(self.get_type(), instance)
        await self._save()

    @abstractmethod
    async def _uninstall_instance(self, instance: str, options: UninstallServiceIn) -> None:
        """Uninstall instance."""

    async def _resolve_custom_model_size(self, spec: dict[str, Any], instance: str = "") -> str | None:  # noqa: ARG002
        """Return auto-fetched size string for the model, or None if not supported."""
        return None

    def get_custom_model_definition(self, custom_model_id: CustomModelId) -> dict[str, Any] | None:
        """Return the stored spec a custom model was created from, searching all instances by id."""
        for instance in self.instances_info.values():
            for custom in instance.config.custom or []:
                if custom.id == custom_model_id:
                    return custom.data
        return None

    async def _validate_custom_model(self, spec: dict[str, Any], instance: str = "", *, previous: dict[str, Any] | None = None) -> None:
        """Raise `HTTPException(400, ...)` if the custom model spec can't be served; no-op by default.

        Runs before `_resolve_custom_model_size`, so an override may also fill in derived fields
        (e.g. `size`) it already fetched, sparing a redundant lookup - callers must not reorder the
        two.

        `previous` is the stored spec when re-validating an edit of an existing model (`None` when
        adding one), so an override can skip checks the edit can't have affected.
        """

    async def add_custom_model(self, instance: str, options: AddCustomModelIn) -> CustomModelId:
        """Add custom model."""
        spec = dict(options.spec)
        await self._validate_custom_model(spec, instance)
        if not spec.get("size"):
            resolved = await self._resolve_custom_model_size(spec, instance)
            spec["size"] = resolved if resolved else "unknown"
        model = CustomModel(id=str(uuid.uuid4()), data=spec)
        config = self.get_instance_info(instance).config
        self._add_custom_model(instance, model)
        if config.custom is None:
            config.custom = []

        config.custom.append(model)
        await self._save()
        return model.id

    def _add_custom_model(self, instance: str, model: CustomModel) -> None:  # noqa: ARG002
        """Add custom model."""
        raise HTTPException(400, "This service does not support custom models.")

    async def remove_custom_model(self, instance: str, custom_model_id: CustomModelId) -> None:
        """Remove custom model."""
        config = self.get_instance_info(instance).config
        model = next(x for x in config.custom or {} if x.id == custom_model_id)
        self._remove_custom_model(instance, model)
        config.custom = [x for x in config.custom or {} if x.id != custom_model_id]
        model_id = model.data.get("id")
        if model_id:
            self._edit_locks.pop((instance, model_id), None)
        await self._save()

    def _remove_custom_model(self, instance: str, model: CustomModel) -> None:  # noqa: ARG002
        """Remove custom model."""
        raise HTTPException(400, "This service does not support custom models.")

    async def update_custom_model(self, instance: str, custom_model_id: CustomModelId, options: AddCustomModelIn) -> None:
        """Update custom model."""
        config = self.get_instance_info(instance).config
        model = next((x for x in config.custom or [] if x.id == custom_model_id), None)
        if model is None:
            raise HTTPException(404, f"Custom model {custom_model_id} not found.")
        new_data: dict[str, Any] = dict(options.spec)
        if _custom_model_serving_fields_changed(model.data, new_data):
            await self._validate_custom_model(new_data, instance, previous=model.data)
        if not new_data.get("size"):
            resolved = await self._resolve_custom_model_size(new_data, instance)
            new_data["size"] = resolved or model.data.get("size") or "unknown"
        await self._update_custom_model(instance, model, new_data)
        model.data = new_data
        await self._save()

    async def _update_custom_model(self, instance: str, model: CustomModel, new_data: dict[str, Any]) -> None:
        """Update custom model by rebuilding it: remove the old, add the new, restoring the old on failure."""
        self._remove_custom_model(instance, model)
        try:
            self._add_custom_model(instance, CustomModel(id=model.id, data=new_data))
        except Exception:
            self._add_custom_model(instance, model)
            raise

    async def _install_model_and_wait(
        self, instance: str, model_id: str, options: InstallModelIn
    ) -> PromiseWithProgress[InstallModelOut, StreamChunk]:
        """Install the model and wait for the underlying work to actually finish before returning.

        `install_model` returns a `PromiseWithProgress` whose real work runs in a background task -
        awaiting the call itself only awaits kicking it off, not its completion. Edit orchestration needs
        to know whether the (re)install genuinely succeeded before deciding to roll back, so this awaits
        the promise fully and lets any failure propagate.
        """
        promise = await self.install_model(instance, model_id, options)
        await promise.wait()
        return promise

    def _get_edit_lock(self, instance: str, model_id: str) -> asyncio.Lock:
        """Return the lock serializing `edit_model`/`edit_model_install_options` calls for one model.

        Scoped to edit orchestration only - `install_model`/`uninstall_model`/`update_custom_model`
        already guard themselves against concurrent calls to themselves (see design.md), so they don't
        need to acquire this lock.
        """
        key = (instance, model_id)
        if key not in self._edit_locks:
            self._edit_locks[key] = asyncio.Lock()
        return self._edit_locks[key]

    def _validate_edit(self, instance: str, model_id: str) -> None:
        """Reject an edit upfront, before any uninstall happens. Default no-op.

        Override for service-specific preconditions that would otherwise only be caught deep inside
        `update_custom_model`, after `edit_model` has already uninstalled the model for nothing (see
        MCP's OAuth-pending check).
        """

    def _check_prefix_collision(self, instance: str, prefix: str, exclude_model_id: str | None) -> None:
        """Reject `prefix` if another model in `instance` is already registered under it.

        Compares against each model's *effective* prefix - the live install-time prefix when the
        model is installed, falling back to `default_prefix` otherwise - not `default_prefix` alone.
        `edit_model_install_options` can move a model's install-time prefix independently of its
        `default_prefix`, so the two can disagree; the effective prefix is whichever one actually
        reflects what a model is doing right now (serving traffic on it, or reserving it unstalled).

        Not every service's model type carries `default_prefix` - it's only meaningful for services
        (MCP, Custom) whose models are inherently addressed by a prefix, so it's read via `getattr`.
        """
        # Looked up directly rather than via `get_instance_info` (which 404s): this can run for an
        # instance that isn't registered in `instances_info` yet, e.g. while adding its first model.
        info = self.instances_info.get(instance)
        installed_models = cast("_HasModels", info.installed).models if info and info.installed else {}
        for mid, m in self.models.get(instance, {}).items():
            if mid == exclude_model_id:
                continue
            effective_prefix = installed_models[mid].prefix if mid in installed_models else getattr(m, "default_prefix", None)
            if effective_prefix == prefix:
                raise HTTPException(400, f"Prefix '{prefix}' is already in use by model '{mid}'.")

    @contextmanager
    def _keep_gpu_allocation(self, instance: str, model_id: str, installed_models: dict[str, Any]) -> Iterator[Callable[[bool], None]]:
        """Keep a model's current GPU utilization across an uninstall/reinstall edit, clearing it on exit.

        The uninstall drops the model's persisted entry, so without this every reinstall (and rollback) would
        re-derive the size from whatever VRAM is free at that moment. Yields a toggle: `set_active(False)` hides
        the kept value (e.g. for an install whose gpu_priority changed), `set_active(True)` restores it (rollbacks).
        """
        key = (instance, model_id)
        installed = installed_models.get(model_id)
        kept = getattr(installed, "gpu_memory_utilization", None) if installed is not None else None

        def set_active(active: bool) -> None:
            if kept and active:
                self._edit_kept_gpu_utilization[key] = kept
            else:
                self._edit_kept_gpu_utilization.pop(key, None)

        set_active(True)
        try:
            yield set_active
        finally:
            self._edit_kept_gpu_utilization.pop(key, None)

    async def edit_model(  # noqa: C901
        self, instance: str, custom_model_id: CustomModelId, new_definition: AddCustomModelIn
    ) -> PromiseWithProgress[InstallModelOut, StreamChunk] | None:
        """Edit a custom-backed model's definition without requiring a manual uninstall first.

        If the model is currently installed, uninstalls it, applies the new definition, then reinstalls
        it with its previous install-time options (prefix/envs/headers), returning the reinstall promise.
        If it isn't installed, this is a plain definition update (same as calling `update_custom_model`
        directly) and returns None. On failure applying the new definition or reinstalling, restores the
        model to its previous definition and installed state on a best-effort basis.
        """
        config = self.get_instance_info(instance).config
        model = next((x for x in config.custom or [] if x.id == custom_model_id), None)
        if model is None:
            raise HTTPException(404, f"Custom model {custom_model_id} not found.")
        model_id = model.data.get("id")
        if not model_id:
            raise HTTPException(400, "Custom model definition has no id.")
        new_id = new_definition.spec.get("id")
        if new_id is not None and new_id != model_id:
            # `edit_model` uninstalls/reinstalls under this one captured `model_id` throughout - an id
            # change would rename the entry out from under that, so the reinstall step would fail
            # looking for the old id. MCP's own per-kind `_update_custom_model` branches already reject
            # this, but only *after* uninstalling; checking here first avoids that pointless disruption,
            # and closes the same gap for services (e.g. CustomService) whose generic update path never
            # enforced id immutability at all.
            raise HTTPException(400, "Cannot change the model's id while editing. Duplicate it instead to use a new id.")
        self._validate_edit(instance, model_id)

        async with self._get_edit_lock(instance, model_id):
            info = self.get_instance_info(instance)
            installed_models = cast("_HasModels", info.installed).models if info.installed else {}
            was_installed = model_id in installed_models
            install_options: InstallModelIn | None = installed_models[model_id].options if was_installed else None
            old_data = dict(model.data)

            with self._keep_gpu_allocation(instance, model_id, installed_models):
                if was_installed:
                    await self.uninstall_model(instance, model_id, UninstallModelIn(purge=False))

                try:
                    await self.update_custom_model(instance, custom_model_id, new_definition)
                except Exception:
                    logger.exception(f"{self.get_id(instance)} failed to apply new definition for model {model_id} while editing")
                    if was_installed:
                        assert install_options is not None
                        try:
                            await self._install_model_and_wait(instance, model_id, install_options)
                        except Exception:
                            logger.exception(f"{self.get_id(instance)} failed to roll back model {model_id} after a failed edit")
                    raise

                if not was_installed:
                    return None

                assert install_options is not None
                # Read the prefix `update_custom_model` actually computed for the model now sitting in the
                # registry - not `new_definition.spec.get("default_prefix")`. `default_prefix` is optional
                # for MCP `user`/`proxy` models and the WebUI sends it as absent whenever the field is
                # cleared; `_update_custom_model` still falls back to `normalize_name(id)` in that case and
                # writes that as the model's new `default_prefix`, which the raw request body never reflects.
                updated_model = self.models.get(instance, {}).get(model_id)
                new_default_prefix = getattr(updated_model, "default_prefix", None)
                reinstall_options = install_options
                if (
                    install_options.spec is not None
                    and new_default_prefix is not None
                    and install_options.spec.get("prefix") != new_default_prefix
                ):
                    # The install-time `prefix` starts out equal to the definition's `default_prefix` (set by
                    # `_install_model` when "prefix" is absent from the install spec) and there's no user-facing
                    # way to diverge a custom-backed model's `prefix` from that - `edit_model_install_options`
                    # (the only path that edits `prefix` directly) is only reachable for catalog models with no
                    # `CustomModel` definition, mutually exclusive with this one. So `prefix` must always follow
                    # `default_prefix` here, or editing it would silently do nothing to the registered endpoint.
                    # `install_options` itself is left untouched so a rollback below reinstalls with the prefix
                    # that actually matches the restored (old) definition.
                    reinstall_options = install_options.model_copy(update={"spec": {**install_options.spec, "prefix": new_default_prefix}})
                try:
                    return await self._install_model_and_wait(instance, model_id, reinstall_options)
                except Exception:
                    logger.exception(f"{self.get_id(instance)} failed to reinstall model {model_id} while editing")
                    try:
                        await self.update_custom_model(instance, custom_model_id, AddCustomModelIn(spec=old_data))
                        await self._install_model_and_wait(instance, model_id, install_options)
                    except Exception:
                        logger.exception(f"{self.get_id(instance)} failed to roll back model {model_id} after a failed reinstall")
                    raise

    def _carry_over_edited_install_options(self, old_options: InstallModelIn, new_options: InstallModelIn) -> InstallModelIn:  # noqa: ARG002
        """Carry install options the edit form can't express from `old_options` into `new_options` (override hook)."""
        return new_options

    async def edit_model_install_options(
        self, instance: str, model_id: str, new_options: InstallModelIn
    ) -> tuple[bool, PromiseWithProgress[InstallModelOut, StreamChunk]]:
        """Edit a model's install-time options only (no persisted `CustomModel` definition involved).

        For catalog models (no `CustomModel` backing) whose only editable state is install-time options
        like prefix/envs/headers. Uninstalls the model if installed, then reinstalls it with the new
        options, restoring the previous options on a best-effort basis if the reinstall fails. Always
        installs, even if the model wasn't installed to begin with - the returned bool tells the caller
        which case it was, so it can report whether this was actually a reinstall.
        """
        model = self.models.get(instance, {}).get(model_id)
        new_prefix = (new_options.spec or {}).get("prefix") or getattr(model, "default_prefix", None)
        if new_prefix is not None:
            # Checked upfront, before any uninstall, so a doomed edit doesn't disrupt the running
            # model for nothing - mirrors `edit_model`'s id-immutability check above.
            self._check_prefix_collision(instance, new_prefix, exclude_model_id=model_id)

        async with self._get_edit_lock(instance, model_id):
            info = self.get_instance_info(instance)
            installed_models = cast("_HasModels", info.installed).models if info.installed else {}
            was_installed = model_id in installed_models
            old_options: InstallModelIn | None = installed_models[model_id].options if was_installed else None
            if old_options is not None:
                new_options = self._carry_over_edited_install_options(old_options, new_options)

            with self._keep_gpu_allocation(instance, model_id, installed_models) as set_kept_active:
                if was_installed:
                    await self.uninstall_model(instance, model_id, UninstallModelIn(purge=False))

                # A changed gpu_priority must be re-sized from scratch; an unchanged one keeps the current allocation.
                if self._parse_gpu_priority(new_options.spec) != self._parse_gpu_priority(old_options.spec if old_options else None):
                    set_kept_active(False)
                try:
                    promise = await self._install_model_and_wait(instance, model_id, new_options)
                except Exception:
                    logger.exception(f"{self.get_id(instance)} failed to reinstall model {model_id} with new install options")
                    if was_installed:
                        assert old_options is not None
                        set_kept_active(True)
                        try:
                            await self._install_model_and_wait(instance, model_id, old_options)
                        except Exception:
                            logger.exception(
                                f"{self.get_id(instance)} failed to roll back model {model_id} after a failed install-options edit"
                            )
                    raise
                return was_installed, promise

    async def _reserve_and_install(
        self,
        instance: str,
        model_id: str,
        options: InstallModelIn,
        on_success: Callable[[InstallModelOut], Awaitable[InstallModelOut]],
        on_error: Callable[[Exception], None],
        container_name_override: str | None = None,
    ) -> PromiseWithProgress[InstallModelOut, StreamChunk]:
        """Reserve model_id for install, or join an already-in-progress reservation for it.

        Shared by install_model() (the public entry point) and load_model() (which starts an
        install directly at startup, bypassing the public API) so a concurrent call through either
        one sees the same reservation, instead of racing a duplicate install or fabricating a
        result. on_success/on_error carry each caller's own post-install bookkeeping; reservation
        cleanup itself always happens here regardless of what they do.

        container_name_override: only load_model() passes this, to pin the persisted name - see ModelConfig.container_name.
        """
        installing_model_progress = self.get_instance_info(instance).installing_model_progress

        async def wrapped_on_success(data: InstallModelOut) -> InstallModelOut:
            self._pop_installing_if_current(instance, reservation, model_id=model_id)
            return await on_success(data)

        def wrapped_on_error(e: Exception) -> None:
            self._pop_installing_if_current(instance, reservation, model_id=model_id)
            on_error(e)

        existing = installing_model_progress.get(model_id)
        if existing is not None:
            # Already installing (or reserved a moment ago by a concurrent call): join the real
            # install instead of asking the service to start a second one or fabricating a result.
            # The bookkeeping continuation is shared (see resolve()), not re-chained per caller.
            # wait_chained() only waits for the (short) pre-promise setup to settle, not the whole
            # install - the real download is awaited separately, by whoever calls promise.wait()
            # on what this returns - so a bounded wait here doesn't cut a legitimately long install
            # short, only a genuinely hung setup phase.
            return await self._wait_for_install_setup(model_id, existing.wait_chained())

        # Reserve the slot synchronously (no `await` before this point) so a concurrent call can
        # never observe "not reserved yet" once this one has started, no matter how long the
        # service's own `_install_model()` takes to actually produce the real promise.
        reservation = InstallingModel()
        installing_model_progress[model_id] = reservation
        # Run _install_model() as its own task (rather than awaiting it inline) so cancel_model_install(),
        # called from a completely different request, has something real to cancel even before this
        # produces a promise - not just something to wait on.
        reservation.pending_task = asyncio.create_task(self._install_model(instance, model_id, options, container_name_override))
        try:
            promise = await reservation.pending_task
        except BaseException as e:
            self._pop_installing_if_current(instance, reservation, model_id=model_id)
            reservation.reject(e)
            # A CancelledError here can mean two different things: this call's own task was
            # cancelled (e.g. its request disconnected) - which must propagate as-is - or someone
            # else cancelled the pending setup task (cancel_model_install(), from a different
            # request) while this call was merely awaiting it - which isn't this caller's own
            # cancellation and shouldn't surface as a raw CancelledError FastAPI can't handle.
            if isinstance(e, asyncio.CancelledError) and not is_own_cancellation():
                raise HTTPException(409, f"Model {model_id} install was cancelled") from None
            raise

        return reservation.resolve(promise, wrapped_on_success, wrapped_on_error)

    def get_model_spec(self) -> ModelSpecification:
        """Return the install specification shared by all models of this service; none by default."""
        return ModelSpecification(fields=[])

    def _get_model_install_fields(self, instance: str, model_id: str) -> Sequence[ModelField]:  # noqa: ARG002
        """Return the install spec fields of a model; services whose fields depend on the model override this."""
        return self.get_model_spec().fields

    async def install_model(
        self, instance: str, model_id: str, options: InstallModelIn
    ) -> PromiseWithProgress[InstallModelOut, StreamChunk]:
        """Install the model.

        Fields the spec leaves out are filled with their defaults first, so a partial spec sent through
        the API is persisted the same way the WebUI install form would have sent it.
        """
        fields = self._get_model_install_fields(instance, model_id)
        if fields:
            options = options.model_copy(update={"spec": fill_spec_defaults(fields, options.spec)})

        async def on_success(data: InstallModelOut) -> InstallModelOut:
            if instance_failures := self._failed_models.get(instance):
                instance_failures.discard(model_id)
            await self._save()
            try:
                await self.service_provider.dismiss_warnings_matching(self.get_type(), instance=instance, model_id=model_id)
            except Exception:
                logger.exception(f"{self.get_id(instance)} failed to dismiss warnings for model {model_id} after successful install")
            msg = f"{model_id} model installed."
            logger.debug(msg)
            return data

        def on_error(e: Exception) -> None:
            error = str(e)
            self._record_warning_in_background(error, instance=instance, model_id=model_id)

        return await self._reserve_and_install(instance, model_id, options, on_success, on_error)

    @abstractmethod
    async def _install_model(
        self, instance: str, model_id: str, options: InstallModelIn, container_name_override: str | None = None
    ) -> PromiseWithProgress[InstallModelOut, StreamChunk]:
        """Install the model. container_name_override, if set, must be used verbatim - see ModelConfig.container_name."""

    async def uninstall_model(self, instance: str, model_id: str, options: UninstallModelIn) -> None:
        """Uninstall the model."""
        await self._uninstall_model(instance, model_id, options)
        if instance_failures := self._failed_models.get(instance):
            instance_failures.discard(model_id)
        try:
            await self.service_provider.dismiss_warnings_matching(self.get_type(), instance=instance, model_id=model_id)
        except Exception:
            logger.exception(f"{self.get_id(instance)} failed to dismiss warnings for model {model_id} after uninstall")
        await self._save()

    @abstractmethod
    async def _uninstall_model(self, instance: str, model_id: str, options: UninstallModelIn) -> None:
        """Uninstall the model."""

    async def get_docker_logs(self, instance: str, model_id: str | None) -> str:
        """Get docker logs."""
        docker_compose_file_path = self.get_docker_compose_file_path(instance, model_id)
        return await self.docker_service.get_docker_compose_logs(docker_compose_file_path)

    def _get_model_installed_info(self, instance: str, model_id: str) -> bool | InstallModelProgress:
        installing_model_progress = self.get_instance_info(instance).installing_model_progress

        if model_id in installing_model_progress:
            installing = installing_model_progress[model_id]
            if installing.last_chunk and installing.last_chunk["type"] == "progress":
                return InstallModelProgress(stage=installing.last_chunk["stage"], value=installing.last_chunk["value"])
            if installing.last_chunk and installing.last_chunk["type"] == "finish":
                return InstallModelProgress(stage="install", value=1)
            return InstallModelProgress(stage="download", value=0)
        return False

    def _get_service_installed_info(self, instance: str) -> bool | InstallServiceProgress:
        installing = self.get_instance_info(instance).installing

        if not installing:
            return False
        if installing.last_chunk and installing.last_chunk["type"] == "progress":
            return InstallServiceProgress(stage=installing.last_chunk["stage"], value=installing.last_chunk["value"])
        if installing.last_chunk and installing.last_chunk["type"] == "finish":
            return InstallServiceProgress(stage="install", value=1)
        return InstallServiceProgress(stage="download", value=0)

    async def get_docker_compose_file(self, instance: str, model_id: str | None) -> str:
        """Get docker compose file."""
        docker_compose_file_path = self.get_docker_compose_file_path(instance, model_id)
        return await Utils.read_file(docker_compose_file_path)

    async def restart_docker(self, instance: str, model_id: str | None) -> None:
        """Restart the instance/model's docker-backed service, resolving its DockerOptions first.

        Handles the no-compose-file case (e.g. an adopted container) via DockerService.restart_docker_compose.
        """
        options = self.get_docker_options(instance, model_id)
        await self.docker_service.restart_docker_compose(options)

    def get_docker_compose_file_path(self, instance: str, model_id: str | None) -> Path:
        """Get docker compose file path."""
        options = self.get_docker_options(instance, model_id)
        return self.docker_service.get_docker_compose_file_path(options.name)

    def get_docker_options(self, instance: str, model_id: str | None) -> DockerOptions:  # noqa: ARG002
        """Return the resolved DockerOptions for this instance/model."""
        if not self.is_installed(instance):
            raise HTTPException(400, "Instance not installed")
        raise HTTPException(400, "Docker is not bound with this object")

    def _get_working_dir(self) -> Path:
        return self._get_service_dir(self.get_type())

    def _get_service_dir(self, service: str) -> Path:
        """Get service dir."""
        dir = self.config.get_storage_services_dir() / f"./{service}"
        if not dir.is_dir():
            dir.mkdir(parents=True)
        return dir

    async def _clear_working_dir(self) -> None:
        working_dir = self._get_working_dir()
        if working_dir.exists():
            shutil.rmtree(working_dir)

    def get_instance_installed_info(self, instance: str) -> InstalledInfoType:
        """Get instance installed info or return exception."""
        installed = self.get_instance_info(instance).installed
        if installed is None:
            raise HTTPException(status_code=400, detail=f"Service {self.get_id(instance)} not installed")
        return installed

    def get_hugging_face_token(self) -> str:
        """Return Hugging Face Key."""
        return self.config.hugging_face_token.get_secret_value()

    def get_civitai_token(self) -> str:
        """Return Civitai Face Key."""
        return self.config.civitai_token.get_secret_value()

    def _has_gpu_for_spec(self) -> str:
        return "true" if self.docker_service.has_gpu_support else "false"

    async def _download_image_or_set_progress(self, stream: Stream[StreamChunk], image: DockerImage) -> None:
        if image.name not in self.images_download_progress:
            self.images_download_progress[image.name] = stream
            try:
                await self._docker_pull(image, stream)
            finally:
                self.images_download_progress.pop(image.name, None)
        else:
            chunk: StreamChunk
            async for chunk in self.images_download_progress[image.name].as_generator():
                if chunk.get("type") == "progress" and chunk.get("stage") == "download":
                    stream.emit(chunk)
                else:
                    break

    async def _docker_pull(
        self,
        image: DockerImage,
        stream: Stream[StreamChunk],
    ) -> None:
        """Docker pull only if image does not exist."""
        stream.emit(StreamChunkProgress(type="progress", stage="download", value=0, data={}))
        if not await self.docker_service.is_docker_image_pulled(image.name):
            size = await self.docker_service.get_docker_image_size(image.name)
            async for progress in self.docker_service.docker_pull(image.name, size or convert_size_to_bytes(image.size) or 0):
                stream.emit(StreamChunkProgress(type="progress", stage="download", value=progress, data={}))
        stream.emit(StreamChunkProgress(type="progress", stage="download", value=1, data={}))

    async def _stop_docker(self, docker_options: DockerOptions) -> None:
        """Stop docker and log error if it occurs."""
        task: asyncio.Task[None] = asyncio.ensure_future(self.docker_service.stop_docker(docker_options))

        def _log_if_failed(finished: "asyncio.Task[None]") -> None:
            if not finished.cancelled() and (exc := finished.exception()) is not None:
                logger.error("Error during stopping docker compose %s", docker_options.name, exc_info=exc)

        # Logging happens in the done-callback above, not in an except block here: asyncio.shield
        # discards a real teardown failure that lands concurrently with the caller itself being
        # cancelled a second time, which would otherwise silently drop that failure. Shielded so a
        # second cancellation of the caller can't kill this teardown's own subprocess mid-`docker
        # compose down`, which would leave the container half-removed.
        task.add_done_callback(_log_if_failed)
        with suppress(Exception):  # already logged by the done-callback above
            await asyncio.shield(task)

    async def _rollback_failed_install_docker(self, docker_options: DockerOptions, *, adopted: bool) -> None:
        """Undo `install_and_run_docker` after a later step of the same install attempt fails.

        Skips touching the container when `adopted` is True: an adopted container (DFINFRA-281) pre-dates
        this install attempt — it wasn't created by it — so a failed install must leave it exactly as it
        found it instead of tearing it down. Only a container this attempt itself started gets stopped.
        """
        if adopted:
            logger.info(
                "Not stopping %r on install rollback: it was adopted from a pre-existing container, not created by this install attempt",
                docker_options.name,
            )
            return
        await self._stop_docker(docker_options)

    async def _stop_dockers_parallel(self, docker_options_list: list[DockerOptions]) -> None:
        """Stop docker and log error if it occurs."""
        tasks = [asyncio.create_task(self._stop_docker(docker_options)) for docker_options in docker_options_list]
        await asyncio.gather(*tasks)

    async def _verify_docker_image(self, docker_image: str, ignore_warning: bool) -> None:
        warnings = await self.docker_service.get_image_warnings(docker_image)
        if len(warnings) > 0 and not ignore_warning:
            raise HTTPException(400, {"warnings": warnings})

    def get_default_docker_image_name(self) -> str | None:
        """Return the docker image this service would install by default, or None to skip the upfront platform check.

        Used once at app startup by `resolve_platform_compatibility()` to check whether the image is
        published for the host's OS/architecture. Override per service with the same image already
        used (with no I/O) to pick the default install target, e.g. `_get_image(...).name`.
        """
        return None

    async def resolve_platform_compatibility(self) -> None:
        """Resolve, once, whether this service's default docker image supports the host platform.

        Called during app startup, before the server accepts requests. Populates
        `_platform_unsupported_reason`, read synchronously by `get_disabled_reason()` afterwards.
        A registry lookup that fails, or times out, is logged and swallowed - the service is then
        treated as compatible, matching how GPU-detection failures already degrade gracefully
        rather than blocking startup. Skipped entirely when hardware already disables the service,
        since `get_disabled_reason()` would discard the platform result anyway.
        """
        if self.get_hardware_unsupported_reason():
            return
        image = self.get_default_docker_image_name()
        if not image:
            return
        try:
            platforms = await asyncio.wait_for(self.docker_service.get_image_platforms(image), timeout=_PLATFORM_CHECK_TIMEOUT_SECONDS)
        except (DockerImageDoesNotExistError, DockerImageAuthorizationError) as err:
            # Unrelated to platform compatibility (a missing/inaccessible image) - nothing useful
            # to report as a hardware mismatch, so leave the service looking compatible.
            logger.warning("Could not resolve platform compatibility for %s (%s): %s; treating as compatible", self.get_type(), image, err)
            return
        except Exception:
            logger.exception("Could not resolve platform compatibility for %s (%s); treating as compatible", self.get_type(), image)
            return
        if platforms and self.docker_service.host_platform not in platforms:
            # A clean, user-facing summary - deliberately not the raw warning text (image name,
            # digest, full platform list), which reads as an unreadable wall of technical detail
            # in the webui tooltip (DFINFRA-365).
            arch = (
                self.docker_service.host_platform.split("/")[1]
                if "/" in self.docker_service.host_platform
                else self.docker_service.host_platform
            )
            self._platform_unsupported_reason = (
                f"This service is incompatible with your machine ({arch} is not supported). Restart Infra if this has changed."
            )

    def get_disabled_reason(self) -> str | None:
        """Return why this service can't be installed here, combining hardware and platform checks."""
        return self.get_hardware_unsupported_reason() or self._platform_unsupported_reason

    @property
    def _supported_gpus(self) -> list[GpuInfo]:
        """Return GPUs supported by this service. Override to include non-NVIDIA GPUs."""
        return [gpu for gpu in self.hardware.gpus if isinstance(gpu, NvidiaGpuInfo)]

    def is_given_hardware_support_gpu(self, hardware_specification: str | bool | None) -> bool:
        """Return is gpu will be used."""
        if hardware_specification is None:
            return bool(self._supported_gpus)
        if isinstance(hardware_specification, str):
            has_gpu_support = hardware_specification.startswith("GPU")
            if has_gpu_support and not self._supported_gpus:
                raise HTTPException(400, "Given hardware specification is not supported")
            return has_gpu_support
        return hardware_specification

    def get_specified_hardware_parts(self, hardware_specification: str | bool | None) -> Sequence[HardwarePartInfo]:
        """Get specified hardware parts."""
        if hardware_specification is None:
            return self._supported_gpus if self._supported_gpus else [self.hardware.cpu]
        if (hardware_specification is False) or (isinstance(hardware_specification, str) and hardware_specification == "CPU"):
            return [self.hardware.cpu]

        if (hardware_specification is True) or (hardware_specification == "GPUs") or (hardware_specification == "GPU"):
            return self._supported_gpus

        gpus: list[GpuInfo] = []

        for gpu_name in hardware_specification.removeprefix("GPU | ").split(","):
            for gpu in self._supported_gpus:
                if gpu_name == gpu.long_name:
                    gpus.append(gpu)

        return gpus

    def canonicalize_hardware_spec(self, hardware_specification: str | bool | None) -> str:
        """Convert a hardware spec into the descriptive string used by the UI.

        A CLI install may leave `hardware` as a bare bool (or None), which the UI renders literally
        as "true"/"false". This resolves the value to an option produced by `add_hardware_field_to_spec`
        ("CPU", "GPU | <name>" for a single GPU, or "GPUs" for several) so it displays correctly and
        round-trips through the settings form. Descriptive strings are returned as-is.
        """
        if isinstance(hardware_specification, str):
            return hardware_specification

        gpus = [part for part in self.get_specified_hardware_parts(hardware_specification) if isinstance(part, GpuInfo)]
        if not gpus:
            return "CPU"
        if len(gpus) == 1:
            return f"GPU | {gpus[0].long_name}"
        return "GPUs"

    def add_hardware_field_to_spec(
        self,
        fields: list[ServiceField] | None = None,
        cpu_supported: bool = True,
    ) -> list[ServiceField]:
        """Add hardware (CPU/GPU/GPUs) field to specification."""
        fields = fields or []
        options: list[str | OneOfOption] = []
        default: str | None = None
        gpus = self._supported_gpus
        if len(gpus) == 1:
            options.append(OneOfOption(value="GPU", label="Default GPU"))
            default = "GPU"
        elif gpus:
            options.append(OneOfOption(value="GPUs", label="All GPUs"))
            default = "GPUs"
        options.extend([f"GPU | {gpu.long_name}" for gpu in gpus])
        if cpu_supported:
            options.append("CPU")
            if default is None:
                default = "CPU"

        gpus_select_field = ServiceField(type="oneof", name="hardware", description="Choose hardware:", values=options, default=default)
        fields.append(gpus_select_field)

        return fields

    def add_hardware_field_to_model_spec(
        self,
        fields: list[ModelField] | None = None,
        cpu_supported: bool = True,
    ) -> list[ModelField]:
        """Add hardware (CPU/GPU/GPUs) field to specification."""
        fields = fields or []
        options: list[str | OneOfOption] = []
        default: str | None = None
        gpus = self._supported_gpus
        if len(gpus) == 1:
            options.append(OneOfOption(value="GPU", label="Default GPU"))
            default = "GPU"
        elif gpus:
            options.append(OneOfOption(value="GPUs", label="All GPUs"))
            default = "GPUs"
        options.extend([f"GPU | {gpu.long_name}" for gpu in gpus])
        if cpu_supported:
            options.append("CPU")
            if default is None:
                default = "CPU"

        gpus_select_field = ModelField(type="oneof", name="hardware", description="Choose hardware:", values=options, default=default)
        fields.append(gpus_select_field)

        return fields
