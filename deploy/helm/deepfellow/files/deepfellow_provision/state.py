# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""The state Secret: the project identity and its one-time API key, via the Kubernetes API."""

import base64
import ssl
from pathlib import Path

from deepfellow_provision import transport
from deepfellow_provision.errors import ProvisionError

SA_DIR = Path("/var/run/secrets/kubernetes.io/serviceaccount")
K8S_API = "https://kubernetes.default.svc"

ORG_ID = "organization-id"
PROJECT_ID = "project-id"
PROJECT_KEY = "project-api-key"
KEYS = (ORG_ID, PROJECT_ID, PROJECT_KEY)

State = dict[str, str]


def _tls() -> ssl.SSLContext:
    """Build a TLS context that trusts the in-cluster API server CA."""
    return ssl.create_default_context(cafile=SA_DIR / "ca.crt")


def _auth() -> dict[str, str]:
    """Build the Authorization header from the pod's ServiceAccount token."""
    token = (SA_DIR / "token").read_text().strip()
    return {"Authorization": f"Bearer {token}"}


def _url(namespace: str, name: str) -> str:
    """Return the API URL of the Secret."""
    return f"{K8S_API}/api/v1/namespaces/{namespace}/secrets/{name}"


def read(namespace: str, name: str) -> State | None:
    """Return the decoded state Secret, or None when it does not exist yet."""
    status, body = transport.request("GET", _url(namespace, name), _auth(), ctx=_tls())
    if status == 404:
        return None
    if status != 200:
        msg = f"reading state secret failed: HTTP {status}"
        raise ProvisionError(msg)
    return {k: base64.b64decode(v).decode() for k, v in (body.get("data") or {}).items()}


def write(namespace: str, name: str, data: State) -> None:
    """Merge-patch the state Secret.

    The Role grants only get/update/patch on this one Secret, so this always patches, and a 404
    means nobody created it (the chart does, unless provisioning.stateSecret.name is set).
    """
    encoded = {k: base64.b64encode(v.encode()).decode() for k, v in data.items()}
    headers = {**_auth(), "Content-Type": "application/merge-patch+json"}
    status, _ = transport.request("PATCH", _url(namespace, name), headers, {"data": encoded}, ctx=_tls())
    if status == 404:
        msg = (
            f"state secret {name!r} does not exist: it must be pre-created "
            "(the chart renders it when provisioning.stateSecret.name is empty)."
        )
        raise ProvisionError(msg)
    if status not in (200, 201):
        msg = f"persisting state secret failed: HTTP {status}"
        raise ProvisionError(msg)


def is_complete(state: State | None) -> bool:
    """Whether the state holds the whole project identity and key."""
    return state is not None and all(state.get(k) for k in KEYS)


def assert_whole(state: State | None, name: str) -> None:
    """Refuse to reconcile against a half-written identity: empty or complete, nothing between."""
    if not state:
        return
    present = [k for k in KEYS if state.get(k)]
    if present and len(present) != len(KEYS):
        msg = (
            f"state secret {name!r} is partial ({present}): refusing to reconcile — "
            "manual recovery required (inspect the remote workspace, then complete or clear the secret)."
        )
        raise ProvisionError(msg)
