# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""The DeepFellow Server: admin user, organisation, project, API key and model grants."""

import subprocess
import sys
from pathlib import Path
from typing import Any

from deepfellow_provision import state, transport
from deepfellow_provision.errors import ProvisionError, redact
from deepfellow_provision.log import logger

# The Server image installs its dependencies into a venv that is not on PATH, so the interpreter
# running this package cannot import the server package (it fails on argon2).
SERVER_VENV_PYTHON = Path("/app/.venv/bin/python")
# The Server package is importable only from the image's application directory.
SERVER_APP_DIR = Path("/app")

Workspace = tuple[str, str, str]  # (organization id, project id, project API key)


def _server_python() -> str:
    """Return an interpreter that can import the Server package."""
    return str(SERVER_VENV_PYTHON) if SERVER_VENV_PYTHON.exists() else sys.executable


def create_admin(name: str, email: str, password: str) -> None:
    """Create the admin user. An address that already exists is a successful no-op."""
    proc = subprocess.run(
        [_server_python(), "-m", "server.scripts.create_admin", name, email, password],
        capture_output=True,
        text=True,
        check=False,
        cwd=SERVER_APP_DIR if SERVER_APP_DIR.is_dir() else None,
    )
    if proc.returncode == 0:
        logger.info("admin created")
        return
    output = (proc.stdout + proc.stderr).strip()
    if "exist" in output.lower() or "already" in output.lower():
        logger.info("admin already exists (no-op)")
        return
    # The script gets the password as an argv and the Mongo password from env; a failing run can
    # echo either (an argument dump, a connection-string error), so scrub every credential.
    if password:
        output = output.replace(password, "***")
    msg = f"create_admin failed (rc={proc.returncode}): {redact(output) or '<no output>'}"
    raise ProvisionError(msg)


def login(server_url: str, email: str, password: str) -> str:
    """Log the admin in and return the bearer token."""
    status, body = transport.request("POST", f"{server_url}/auth/login", body={"email": email, "password": password})
    # /auth/login returns access_token, refresh_token, token_type and both expiry times.
    token = body.get("access_token")
    if status != 200 or not token:
        msg = f"login failed: HTTP {status}"
        raise ProvisionError(msg)
    return str(token)


_PAGE = 100
_MAX_PAGES = 50


def _list_all(url: str, headers: dict[str, str]) -> list[dict[str, Any]]:
    """Return every item of a paginated Server list (`data`, `has_more`, `last_id`)."""
    items: list[dict[str, Any]] = []
    after = ""
    for _ in range(_MAX_PAGES):
        query = f"?limit={_PAGE}" + (f"&after={after}" if after else "")
        status, body = transport.request("GET", url + query, headers)
        if status != 200:
            msg = f"listing {url} failed: HTTP {status}"
            raise ProvisionError(msg)
        page: list[dict[str, Any]] = body.get("data") or []
        items.extend(page)
        last = str(body.get("last_id") or (page[-1].get("id") if page else ""))
        if not body.get("has_more") or not last or last == after:
            return items
        after = last
    msg = f"listing {url} did not finish within {_MAX_PAGES} pages"
    raise ProvisionError(msg)


def _find_by_name(items: list[dict[str, Any]], name: str, kind: str) -> str | None:
    """Return the id of the item called `name`, the oldest one if several share it."""
    matches = [i for i in items if i.get("name") == name]
    if not matches:
        return None
    if len(matches) > 1:
        logger.info("%d %ss are named %r; reusing the oldest", len(matches), kind, name)
    return str(min(matches, key=lambda i: int(i.get("created_at") or 0))["id"])


def _create_workspace(server_url: str, jwt: str, org_name: str, project_name: str, key_name: str) -> Workspace:
    """Find or create the organisation and the project, then mint the project's API key.

    The Server accepts duplicate names, so a run that failed after creating the organisation or the
    project would otherwise leave another copy behind on every retry. Both are looked up by name
    first. The key is always new: the Server returns its value only once.
    """
    auth = {"Authorization": f"Bearer {jwt}"}
    org_id = _find_by_name(_list_all(f"{server_url}/admin/organization/", auth), org_name, "organization")
    if org_id is None:
        status, body = transport.request("POST", f"{server_url}/admin/organization/", auth, {"name": org_name})
        if status not in (200, 201):
            msg = f"create organization failed: HTTP {status}"
            raise ProvisionError(msg)
        org_id = str(body["organization"]["id"])
    else:
        logger.info("reusing organization %r", org_name)

    org_auth = {**auth, "OpenAI-Organization": org_id}
    projects_url = f"{server_url}/v1/organization/projects"
    project_id = _find_by_name(_list_all(projects_url, org_auth), project_name, "project")
    if project_id is None:
        status, body = transport.request("POST", projects_url, org_auth, {"name": project_name})
        if status not in (200, 201):
            msg = f"create project failed: HTTP {status}"
            raise ProvisionError(msg)
        project_id = str(body["id"])
    else:
        logger.info("reusing project %r", project_name)

    url = f"{projects_url}/{project_id}/api_keys"
    status, body = transport.request("POST", url, org_auth, {"name": key_name})
    if status not in (200, 201) or not body.get("value"):
        msg = f"create project api key failed: HTTP {status}"
        raise ProvisionError(msg)
    return org_id, project_id, str(body["value"])


def ensure_workspace(
    server_url: str,
    jwt: str,
    names: tuple[str, str, str],
    current: state.State | None,
    namespace: str,
    state_secret: str,
) -> Workspace:
    """Return the organisation, project and key, creating them only when the state has none.

    `names` is (organisation, project, key name).
    """
    if current is not None and state.is_complete(current):
        logger.info("workspace/key already provisioned (from state secret)")
        return current[state.ORG_ID], current[state.PROJECT_ID], current[state.PROJECT_KEY]

    org_id, project_id, project_key = _create_workspace(server_url, jwt, *names)

    # Persist the one-time key before anything else. If this write fails, the key exists remotely
    # but can never be read again: stop for manual recovery rather than let a later run mint a
    # second one.
    try:
        state.write(namespace, state_secret, {state.ORG_ID: org_id, state.PROJECT_ID: project_id, state.PROJECT_KEY: project_key})
    except ProvisionError as e:
        msg = (
            f"workspace/key created remotely but persisting the state secret failed ({e}); "
            "the GET-ONCE key is unrecoverable — manual recovery required "
            "(delete the remote project/key, then re-run)."
        )
        raise ProvisionError(msg) from e
    return org_id, project_id, project_key


def grant_models(server_url: str, jwt: str, org_id: str, project_id: str, model_ids: list[str]) -> None:
    """Grant the project access to `model_ids`."""
    auth = {"Authorization": f"Bearer {jwt}", "OpenAI-Organization": org_id}
    status, _ = transport.request("POST", f"{server_url}/v1/organization/projects/{project_id}", auth, {"models": model_ids})
    if status not in (200, 201):
        msg = f"grant models failed: HTTP {status}"
        raise ProvisionError(msg)
    logger.info("granted %d model(s) to project", len(model_ids))


def verify(server_url: str, project_key: str, model_ids: list[str]) -> None:
    """Check that every granted model is visible through the Server with the project key."""
    status, body = transport.request("GET", f"{server_url}/v1/models", {"Authorization": f"Bearer {project_key}"})
    if status != 200:
        msg = f"verify: server /v1/models HTTP {status}"
        raise ProvisionError(msg)
    listed = {m.get("id") for m in body.get("data", [])}
    missing = [m for m in model_ids if m not in listed]
    if missing:
        msg = f"verify: models not exposed via project key: {missing}"
        raise ProvisionError(msg)
    logger.info("verify: all granted models exposed via the project key")
