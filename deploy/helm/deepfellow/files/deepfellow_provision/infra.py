# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""DeepFellow Infra: register each backend as a service instance with a custom model."""

import os
import time
import urllib.parse

from deepfellow_provision import transport
from deepfellow_provision.config import Backend
from deepfellow_provision.errors import ProvisionError
from deepfellow_provision.log import logger


def _instance_spec(backend: Backend) -> dict[str, str]:
    """Return the options the backend's service instance should have."""
    api_key = os.environ.get(backend["api_key_env"], "") if backend.get("api_key_env") else ""
    return {"api_url": backend["api_url"], "api_key": api_key or "none"}


def _register_service_instance(infra_url: str, auth: dict[str, str], sid: str, service_id: str, backend: Backend) -> None:
    """Register the backend as its own service instance, or reconcile an existing one."""
    spec = _instance_spec(backend)
    status, body = transport.request(
        "POST", f"{infra_url}/admin/services/{sid}", auth, {"spec": spec, "stream": False, "ignore_warnings": True}
    )
    # An existing instance answers 409, or 400 with "already install…" (base2 install_instance).
    # Both mean: reconcile, do not recreate.
    if status == 409 or (status == 400 and transport.body_says(body, "already install")):
        _reconcile_existing_instance(infra_url, auth, sid, service_id, backend, spec)
    elif status not in (200, 201):
        msg = f"service register for {backend['id']} failed: HTTP {status}"
        raise ProvisionError(msg)


def _reconcile_existing_instance(
    infra_url: str, auth: dict[str, str], sid: str, service_id: str, backend: Backend, spec: dict[str, str]
) -> None:
    """Refuse an instance that points at a different endpoint; bring a changed API key up to date."""
    status, body = transport.request("GET", f"{infra_url}/admin/services/{sid}", auth)
    if status != 200:
        # Unverified is not good enough: the next steps grant the project access to this instance.
        msg = f"service instance {service_id!r} exists but reading it failed: HTTP {status}"
        raise ProvisionError(msg)
    # The configured endpoint lives in `installed` (the instance's saved options), not the
    # top-level `spec` (that is the field schema). `installed` is a dict only once installed.
    installed = body.get("installed")
    if not isinstance(installed, dict):
        # Nothing saved to compare against. Say so: otherwise the guard below is skipped in
        # silence and we would grant against an endpoint nobody checked.
        logger.info("service instance %s exists but reports no installed endpoint yet; skipping endpoint comparison", service_id)
        return
    existing_url = installed.get("api_url")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    if existing_url and existing_url != backend["api_url"]:
        msg = (
            f"service instance {service_id!r} already registered with api_url {existing_url!r}, "
            f"but backend {backend['id']} wants {backend['api_url']!r}: refusing to grant against a different endpoint."
        )
        raise ProvisionError(msg)
    # A rotated API key (externalBackends.<name>.apiKey) must reach Infra, which keeps the one it
    # was installed with. Compared, never logged.
    if installed.get("api_key") != spec["api_key"]:  # pyright: ignore[reportUnknownMemberType]
        status, _ = transport.request(
            "PUT", f"{infra_url}/admin/services/{sid}", auth, {"spec": spec, "stream": False, "ignore_warnings": True}
        )
        if status not in (200, 201):
            msg = f"updating the API key of service instance {service_id!r} failed: HTTP {status}"
            raise ProvisionError(msg)
        logger.info("service instance %s: API key updated", service_id)


def _reconcile_custom_model(infra_url: str, auth: dict[str, str], sid: str, model_id: str, context_length: int) -> None:
    """Check an already-present custom model still matches the definition we want."""
    # `/models/<id>` is not a route: Infra's SPA catch-all answers it with 200 text/html, which
    # would pass the status check. The query-parameter form is the real one.
    status, body = transport.request("GET", f"{infra_url}/admin/services/{sid}/models/_?model_id={model_id}", auth)
    if status != 200:
        msg = f"model {model_id} exists but reading it failed: HTTP {status}"
        raise ProvisionError(msg)
    # `type` is a top-level field of the model. `custom_spec` holds the custom definition (null for
    # a model installed from one); `spec` is the field schema and holds no values, so reading a value
    # from it yields None for everything and once turned every healthy re-run into "type=None".
    custom = body.get("custom_spec") or {}
    model_type = body.get("type") or custom.get("type")
    if model_type in (None, "None"):
        msg = (
            f"model {model_id} exists but reports no type (the backend's /v1/models may list this id, "
            "which seeds an untyped model): return an empty /v1/models from the backend and re-run."
        )
        raise ProvisionError(msg)
    if model_type != "llm":
        msg = f"model {model_id} exists with type {model_type!r}, desired 'llm': remove the custom model and re-run."
        raise ProvisionError(msg)
    existing_ctx = custom.get("context_length", body.get("context_length"))
    if existing_ctx is not None and int(existing_ctx) != context_length:
        msg = (
            f"model {model_id} exists with context_length {existing_ctx}, desired {context_length}: "
            "remove the custom model and re-run to change it."
        )
        raise ProvisionError(msg)


def _custom_model_spec(model_id: str, context_length: int) -> dict[str, object]:
    """Describe a chat-completions-only LLM, the only kind a llama.cpp or OpenAI endpoint needs here."""
    return {
        "id": model_id,
        "type": "llm",
        "completions": True,
        "legacy_completions": False,
        "responses": False,
        "messages": False,
        "context_length": context_length,
        "max_context_length": context_length,
    }


def register_backend(infra_url: str, admin_key: str, backend: Backend) -> None:
    """Register (or reconcile) one backend: service instance, custom model, install."""
    auth = {"Authorization": f"Bearer {admin_key}"}
    model_id = backend["id"]
    context_length = int(backend.get("context_length", 8192))

    # Each backend is its own service instance (`<type>|<instance>`) so distinct endpoints never
    # share one service. The instance is the chart's map key (DNS-label safe, which satisfies
    # Infra's stricter instance charset), not model_id, which may contain dots.
    service_id = f"{backend.get('service_type', 'openai')}|{backend.get('instance') or model_id}"
    sid = urllib.parse.quote(service_id, safe="")

    _register_service_instance(infra_url, auth, sid, service_id, backend)

    url = f"{infra_url}/admin/services/{sid}/models/custom"
    status, body = transport.request("POST", url, auth, {"spec": _custom_model_spec(model_id, context_length)})
    already = status == 400 and transport.body_says(body, "exist")
    if status not in (200, 201) and not already:
        msg = f"custom model add for {model_id} failed: HTTP {status}"
        raise ProvisionError(msg)
    if already:
        _reconcile_custom_model(infra_url, auth, sid, model_id, context_length)

    url = f"{infra_url}/admin/services/{sid}/models/_?model_id={model_id}"
    status, _ = transport.request("POST", url, auth, {"stream": False, "ignore_warnings": True})
    if status not in (200, 201, 409):
        msg = f"model install for {model_id} failed: HTTP {status}"
        raise ProvisionError(msg)
    logger.info("backend %s registered (service instance %s)", model_id, service_id)


def wait_model_ready(infra_url: str, api_key: str, model_id: str, attempts: int = 30, delay: int = 5) -> None:
    """Wait until Infra lists the model on /v1/models."""
    # /v1/* takes DF_INFRA_API_KEY. The admin key only works on /admin/* and gets 401 here, which
    # would look like "the model never appeared".
    auth = {"Authorization": f"Bearer {api_key}"}
    last_status: int | None = None
    seen: list[str] = []
    for _ in range(attempts):
        last_status, body = transport.request("GET", f"{infra_url}/v1/models", auth)
        if last_status == 200:
            seen = [m.get("id") for m in body.get("data", [])]
            if model_id in seen:
                return
        time.sleep(delay)
    msg = f"infra model {model_id} not ready (last HTTP {last_status} from {infra_url}/v1/models, models seen: {seen})"
    raise ProvisionError(msg)
