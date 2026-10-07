# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""The reconcile order, and how a failure leaves the process."""

from typing import Any
from unittest.mock import patch

import pytest
from deepfellow_provision import cli, config, infra, server, state, transport
from deepfellow_provision.errors import ProvisionError


def _config(backends: list[dict[str, Any]] | None = None) -> config.Config:
    return config.Config(
        namespace="ns",
        state_secret="state",
        infra_url="http://infra",
        infra_admin_key="admin-key",
        infra_api_key="v1-key",
        server_url="http://server",
        admin_name="admin",
        admin_email="a@example.com",
        admin_password="Passw0rd!12",
        org_name="Org",
        project_name="Proj",
        key_name="app",
        backends=tuple(backends if backends is not None else [{"id": "m1", "api_url": "http://b:8080", "native": True}]),
    )


def _trace_reconcile(current: dict[str, str] | None) -> list[str]:
    """Run reconcile with every step stubbed, recording the order they ran in."""
    order: list[str] = []

    def rec(name: str, result: Any = None):
        def _fn(*_a: object, **_k: object):
            order.append(name)
            return result

        return _fn

    with (
        patch.object(state, "read", rec("read_state", current)),
        patch.object(transport, "wait_healthy", rec("wait")),
        patch.object(server, "create_admin", rec("create_admin")),
        patch.object(server, "login", rec("login", "jwt")),
        patch.object(server, "_create_workspace", rec("create_workspace", ("o", "p", "k"))),
        patch.object(state, "write", rec("persist")),
        patch.object(infra, "register_backend", rec("register")),
        patch.object(infra, "wait_model_ready", rec("model_ready")),
        patch.object(server, "grant_models", rec("grant")),
        patch.object(server, "verify", rec("verify")),
    ):
        cli.reconcile(_config())
    return order


def test_reconcile_registers_and_confirms_a_backend_before_granting_it():
    order = _trace_reconcile({})
    assert order.index("register") < order.index("grant")
    assert order.index("model_ready") < order.index("grant"), "never grant a model Infra has not confirmed"
    assert order.index("grant") < order.index("verify")


def test_reconcile_persists_the_key_before_granting():
    order = _trace_reconcile({})
    assert order.index("persist") < order.index("grant"), "the get-once key must be stored before anything else runs"


def test_reconcile_reads_state_before_it_creates_anything():
    order = _trace_reconcile({})
    assert order[0] == "read_state"
    assert order.index("read_state") < order.index("create_admin")


def test_reconcile_does_not_mint_a_second_key_when_state_exists():
    order = _trace_reconcile({"organization-id": "o", "project-id": "p", "project-api-key": "k"})
    assert "create_workspace" not in order, "a provisioned workspace must never be created again"
    assert "persist" not in order
    assert "grant" in order, "reconciliation still converges the rest"


def test_reconcile_with_no_backends_skips_granting():
    with (
        patch.object(state, "read", return_value={}),
        patch.object(transport, "wait_healthy"),
        patch.object(server, "create_admin"),
        patch.object(server, "login", return_value="jwt"),
        patch.object(server, "_create_workspace", return_value=("o", "p", "k")),
        patch.object(state, "write"),
        patch.object(server, "grant_models") as grant,
        patch.object(server, "verify") as verify_fn,
    ):
        cli.reconcile(_config(backends=[]))
    grant.assert_not_called()
    verify_fn.assert_not_called()


def test_reconcile_stops_before_creating_anything_when_the_state_secret_is_missing():
    """Saving the key would fail afterwards, and every Job retry would mint another one."""
    with (
        patch.object(state, "read", return_value=None),
        patch.object(server, "create_admin") as create_admin,
        patch.object(server, "_create_workspace") as create_workspace,
        pytest.raises(ProvisionError) as e,
    ):
        cli.reconcile(_config())
    assert "does not exist" in str(e.value)
    create_admin.assert_not_called()
    create_workspace.assert_not_called()


def test_reconcile_refuses_to_run_on_a_partial_state():
    with patch.object(state, "read", return_value={"organization-id": "o"}), pytest.raises(ProvisionError) as e:
        cli.reconcile(_config())
    assert "partial" in str(e.value)


# ------------------------------------------------------------------------ main


def test_a_failed_step_exits_non_zero_with_its_message():
    with (
        patch.object(config, "load", return_value=_config()),
        patch.object(state, "read", side_effect=ProvisionError("reading state secret failed: HTTP 403")),
        pytest.raises(SystemExit) as e,
    ):
        cli.main(["reconcile"])
    assert str(e.value) == "reading state secret failed: HTTP 403"


def test_an_unknown_action_names_the_valid_ones():
    with pytest.raises(SystemExit) as e:
        cli.main(["provision-everything"])
    assert "reconcile | verify | status" in str(e.value)
