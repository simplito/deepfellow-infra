# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""The three actions: reconcile, verify and status."""

import sys

from deepfellow_provision import config, infra, log, server, state, transport
from deepfellow_provision.errors import ProvisionError
from deepfellow_provision.log import logger


def reconcile(c: config.Config) -> None:
    """Bring the suite to the desired state: admin, workspace, backends, grants."""
    # Read the state first and stop on a missing or half-written one, before anything is created:
    # the project key can be saved nowhere else, and every Job retry would mint another one.
    current = state.read(c.namespace, c.state_secret)
    if current is None:
        msg = (
            f"state secret {c.state_secret!r} does not exist: create it before provisioning "
            "(the chart does, unless provisioning.stateSecret.name is set)."
        )
        raise ProvisionError(msg)
    state.assert_whole(current, c.state_secret)

    transport.wait_healthy(c.infra_url, "infra")
    transport.wait_healthy(c.server_url, "server", accept_any=True)
    for b in c.backends:
        if b.get("native"):
            transport.wait_healthy(b["api_url"], f"backend {b['id']}")

    server.create_admin(c.admin_name, c.admin_email, c.admin_password)
    jwt = server.login(c.server_url, c.admin_email, c.admin_password)

    org_id, project_id, project_key = server.ensure_workspace(
        c.server_url, jwt, (c.org_name, c.project_name, c.key_name), current, c.namespace, c.state_secret
    )

    model_ids = [b["id"] for b in c.backends]
    for b in c.backends:
        infra.register_backend(c.infra_url, c.infra_admin_key, b)
    for b in c.backends:
        infra.wait_model_ready(c.infra_url, c.infra_api_key, b["id"])

    if model_ids:
        server.grant_models(c.server_url, jwt, org_id, project_id, model_ids)
        server.verify(c.server_url, project_key, model_ids)
    logger.info("reconcile complete")


def verify(c: config.Config) -> None:
    """Re-check that the provisioned project still sees every backend's model."""
    current = state.read(c.namespace, c.state_secret)
    if not current or not current.get(state.PROJECT_KEY):
        msg = "verify: no provisioned state secret"
        raise ProvisionError(msg)
    server.verify(c.server_url, current[state.PROJECT_KEY], [b["id"] for b in c.backends])


def status(c: config.Config) -> None:
    """Report whether provisioning has run, without changing anything."""
    current = state.read(c.namespace, c.state_secret)
    logger.info("state secret present: %s; backends: %d", state.is_complete(current), len(c.backends))


ACTIONS = {"reconcile": reconcile, "verify": verify, "status": status}


def main(argv: list[str] | None = None) -> None:
    """Run the action named in argv (default: reconcile). A failure exits 1 with its message."""
    args = sys.argv[1:] if argv is None else argv
    name = args[0] if args else "reconcile"
    action = ACTIONS.get(name)
    if action is None:
        msg = f"unknown action {name}: use {' | '.join(ACTIONS)}"
        raise SystemExit(msg)
    log.configure()
    try:
        action(config.load())
    except ProvisionError as e:
        raise SystemExit(str(e)) from None
