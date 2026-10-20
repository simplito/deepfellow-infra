# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""The Server side: the one-time key, the admin user, login and verification."""

import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from deepfellow_provision import server, state, transport
from deepfellow_provision.errors import ProvisionError

NAMES = ("Org", "Proj", "key")

# ------------------------------------------------------------- ensure_workspace


def test_existing_state_short_circuits_without_creating_anything():
    current = {"organization-id": "o1", "project-id": "p1", "project-api-key": "k1"}
    with patch.object(transport, "request") as req, patch.object(state, "write") as write:
        got = server.ensure_workspace("http://s", "jwt", NAMES, current, "ns", "sec")
    assert got == ("o1", "p1", "k1")
    req.assert_not_called()
    write.assert_not_called()


class FakeServer:
    """The Server's organisation, project and key endpoints, routed by method and path."""

    def __init__(self, orgs: list[dict[str, Any]] | None = None, projects: list[dict[str, Any]] | None = None) -> None:
        self.orgs = list(orgs or [])
        self.projects = list(projects or [])
        self.posts: list[str] = []

    def request(self, method: str, url: str, _headers: object = None, body: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
        path = url.split("?", 1)[0]
        if method == "GET":
            items = self.orgs if path.endswith("/admin/organization/") else self.projects
            return 200, {"data": items, "has_more": False, "last_id": items[-1]["id"] if items else None}
        self.posts.append(path)
        if path.endswith("/admin/organization/"):
            return 200, {"organization": {"id": "org-new", "name": (body or {}).get("name")}}
        if path.endswith("/api_keys"):
            return 200, {"value": "dfproj_secret"}
        return 200, {"id": "proj-new"}


def test_the_one_time_key_is_persisted_before_it_is_returned():
    fake = FakeServer()
    calls: list[str] = []

    def fake_request(*a: Any, **k: Any) -> tuple[int, dict[str, Any]]:
        calls.append("http")
        return fake.request(*a, **k)

    def fake_write(_ns: str, _name: str, data: dict[str, str]) -> None:
        calls.append("persist")
        assert data["project-api-key"] == "dfproj_secret"

    with patch.object(transport, "request", side_effect=fake_request), patch.object(state, "write", fake_write):
        got = server.ensure_workspace("http://s", "jwt", NAMES, None, "ns", "sec")

    assert got == ("org-new", "proj-new", "dfproj_secret")
    assert calls[-1] == "persist", "the key must be stored before ensure_workspace returns"


def test_a_failed_persist_is_reported_as_unrecoverable():
    with (
        patch.object(transport, "request", side_effect=FakeServer().request),
        patch.object(state, "write", side_effect=ProvisionError("boom")),
        pytest.raises(ProvisionError) as e,
    ):
        server.ensure_workspace("http://s", "jwt", NAMES, None, "ns", "sec")
    assert "unrecoverable" in str(e.value)
    assert "manual recovery" in str(e.value)


def test_a_retry_reuses_the_organisation_and_project_a_failed_run_left_behind():
    """The Server accepts duplicate names, so creating them again would pile up copies."""
    fake = FakeServer(
        orgs=[{"id": "org-1", "name": "Org", "created_at": 1}],
        projects=[{"id": "proj-1", "name": "Proj", "created_at": 2}],
    )
    with patch.object(transport, "request", side_effect=fake.request), patch.object(state, "write"):
        got = server.ensure_workspace("http://s", "jwt", NAMES, None, "ns", "sec")
    assert got == ("org-1", "proj-1", "dfproj_secret")
    assert fake.posts == ["http://s/v1/organization/projects/proj-1/api_keys"], "only the key is new"


def test_the_oldest_of_several_same_named_organisations_is_reused():
    fake = FakeServer(
        orgs=[{"id": "org-new", "name": "Org", "created_at": 9}, {"id": "org-old", "name": "Org", "created_at": 3}],
    )
    with patch.object(transport, "request", side_effect=fake.request), patch.object(state, "write"):
        org_id, _, _ = server.ensure_workspace("http://s", "jwt", NAMES, None, "ns", "sec")
    assert org_id == "org-old"


def test_a_list_that_cannot_be_read_stops_the_run():
    with (
        patch.object(transport, "request", return_value=(500, {})),
        patch.object(state, "write"),
        pytest.raises(ProvisionError) as e,
    ):
        server.ensure_workspace("http://s", "jwt", NAMES, None, "ns", "sec")
    assert "HTTP 500" in str(e.value)


def test_every_page_of_a_list_is_read():
    pages = {
        "": {"data": [{"id": "a", "name": "x"}], "has_more": True, "last_id": "a"},
        "a": {"data": [{"id": "b", "name": "Org"}], "has_more": False, "last_id": "b"},
    }
    seen: list[str] = []

    def fake(_m: str, url: str, *_a: object, **_k: object) -> tuple[int, dict[str, Any]]:
        after = url.split("after=", 1)[1] if "after=" in url else ""
        seen.append(after)
        return 200, pages[after]

    with patch.object(transport, "request", side_effect=fake):
        items = server._list_all("http://s/admin/organization/", {})  # pyright: ignore[reportPrivateUsage]
    assert [i["id"] for i in items] == ["a", "b"]
    assert seen == ["", "a"]


# ---------------------------------------------------------------- create_admin


def _proc(rc: int, out: str = "", err: str = "") -> MagicMock:
    m = MagicMock(spec=subprocess.CompletedProcess)
    m.returncode, m.stdout, m.stderr = rc, out, err
    return m


def test_create_admin_succeeds():
    with patch.object(server.subprocess, "run", return_value=_proc(0)):
        server.create_admin("admin", "a@example.com", "pw")


def test_create_admin_runs_where_the_server_package_is_importable():
    """`-m server.scripts.create_admin` resolves from the image's /app, not the Job's cwd."""
    with (
        patch.object(server.subprocess, "run", return_value=_proc(0)) as run,
        patch.object(server, "SERVER_APP_DIR", Path("/")),
    ):
        server.create_admin("admin", "a@example.com", "pw")
    assert run.call_args.kwargs["cwd"] == Path("/")


def test_an_existing_admin_is_a_no_op():
    with patch.object(server.subprocess, "run", return_value=_proc(1, err="User with that email already exists")):
        server.create_admin("admin", "a@example.com", "pw")


def test_a_failure_surfaces_the_output_but_never_the_password():
    secret = "SuperSecret!23"
    with (
        patch.object(server.subprocess, "run", return_value=_proc(2, err=f"boom while using {secret}")),
        pytest.raises(ProvisionError) as e,
    ):
        server.create_admin("admin", "a@example.com", secret)
    message = str(e.value)
    assert "boom while using" in message, "the real error must be visible"
    assert secret not in message, "the password must never be echoed"
    assert "***" in message


def test_a_failure_never_echoes_any_credential_from_the_environment(monkeypatch: pytest.MonkeyPatch):
    """A connection error can print the Mongo URI; every credential the Job holds is scrubbed."""
    monkeypatch.setenv("DF_MONGO_PASSWORD", "mongo-s3cret")
    monkeypatch.setenv("DF_INFRA_ADMIN_API_KEY", "infra-admin-k3y")
    monkeypatch.setenv("PROJECT_NAME", "Default")
    err = "cannot connect to mongodb://deepfellow:mongo-s3cret@mongo:27017 (key infra-admin-k3y), project Default"
    with patch.object(server.subprocess, "run", return_value=_proc(2, err=err)), pytest.raises(ProvisionError) as e:
        server.create_admin("admin", "a@example.com", "pw-0000")
    message = str(e.value)
    assert "mongo-s3cret" not in message
    assert "infra-admin-k3y" not in message
    assert "project Default" in message, "values that are not credentials stay readable"


def test_a_silent_failure_still_says_something():
    with patch.object(server.subprocess, "run", return_value=_proc(3)), pytest.raises(ProvisionError) as e:
        server.create_admin("admin", "a@example.com", "pw")
    assert "<no output>" in str(e.value)


# ----------------------------------------------------------------------- login


def test_login_reads_the_access_token():
    with patch.object(transport, "request", return_value=(200, {"access_token": "jwt-1"})):
        assert server.login("http://s", "a@example.com", "pw") == "jwt-1"


@pytest.mark.parametrize("response", [(401, {}), (200, {"token": "legacy"})])
def test_login_fails_when_no_usable_token_comes_back(response: tuple[int, dict[str, Any]]):
    with patch.object(transport, "request", return_value=response), pytest.raises(ProvisionError) as e:
        server.login("http://s", "a@example.com", "pw")
    assert "login failed" in str(e.value)


# --------------------------------------------------------------------- verify


def test_verify_accepts_a_project_that_exposes_every_model():
    with patch.object(transport, "request", return_value=(200, {"data": [{"id": "a"}, {"id": "b"}]})):
        server.verify("http://s", "dfproj", ["a", "b"])


def test_verify_names_the_models_that_are_missing():
    with (
        patch.object(transport, "request", return_value=(200, {"data": [{"id": "a"}]})),
        pytest.raises(ProvisionError) as e,
    ):
        server.verify("http://s", "dfproj", ["a", "b"])
    assert "b" in str(e.value)
