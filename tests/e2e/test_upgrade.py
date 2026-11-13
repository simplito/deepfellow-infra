# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Upgrade (or restart): installed services, models and inference survive replacing the Infra container.

`DF_E2E_UPGRADE_COMMAND` does the replacing - in CI `instance.sh upgrade <image under test> <id>` on an instance started
from the previous release. Run with the same image, the command checks a restart.

A set is installed and upgraded once, as a whole, but every scenario gets its own test before and after the upgrade
(`test_upgrade[full-before-<scenario>]`, `[full-after-<scenario>]`), so a report shows which service broke and one
broken service does not hide the rest. The tests of a set share one `_SetRun` and rely on pytest keeping the order of
the parameter list: all `before` tests of a set run before its first `after` test triggers the upgrade.
"""

import shlex
import subprocess
import time
from collections.abc import Callable, Iterator
from contextlib import ExitStack

import httpx
import pytest

from tests.e2e.client import E2EClient, E2EError
from tests.e2e.inference import check_inference
from tests.e2e.lifecycle import Ledger, installed, save_logs, skip_reason
from tests.e2e.scenarios import UPGRADE_SETS, Scenario, upgrade_set

STAGES = ("before", "after")


def _retry_until(timeout: float, check: Callable[[], None]) -> None:
    """Run `check` until it passes; a service container may still be starting when Infra already reports healthy.

    Only a connection or server error is retried: a wrong answer fails at once.
    """
    deadline = time.monotonic() + timeout
    while True:
        try:
            check()
        except Exception as e:
            if not _still_starting(e) or time.monotonic() > deadline:
                raise
            time.sleep(5)
        else:
            return


def _still_starting(error: Exception) -> bool:
    if isinstance(error, httpx.TransportError):
        return True
    return isinstance(error, E2EError) and error.status_code is not None and error.status_code >= 500


def _upgrade(client: E2EClient) -> None:
    settings = client.settings
    subprocess.run(shlex.split(settings.upgrade_command), check=True, timeout=settings.install_timeout)
    deadline = time.monotonic() + settings.install_timeout
    while not client.health():
        assert time.monotonic() < deadline, f"{settings.base_url} did not become healthy within {settings.install_timeout:.0f}s"
        time.sleep(2)
    if settings.upgrade_version:
        expected = settings.upgrade_version.removeprefix("v")
        actual = client.info().version
        assert actual == expected, f"after the upgrade the target reports version {actual}, expected {expected}"


def _wait_until_loaded(client: E2EClient, scenarios: list[Scenario]) -> None:
    """Wait until Infra has started the set's services and models again: it answers /health first and loads them in the
    background, each service before its models.

    A GPU vLLM model alone takes minutes to start, so this waits as long as an install may take, not just
    DF_E2E_READY_TIMEOUT. A model that reports not installed right after its service loaded may not have been picked up
    yet, so it is polled for DF_E2E_READY_TIMEOUT before its `after` test calls it lost.
    """
    settings = client.settings
    start = time.monotonic()
    service_ids = {s.service_id for s in scenarios}
    while True:
        waiting = sorted(s.id for s in client.list_services() if s.id in service_ids and s.instance == "default" and s.is_loading)
        if not waiting:
            pending_grace = time.monotonic() - start < settings.ready_timeout
            waiting = [
                f"{s.service_id}/{s.model_id}"
                for s in scenarios
                if client.model_loading(s.service_id, s.model_id)
                or (pending_grace and not client.model_installed(s.service_id, s.model_id))
            ]
        if not waiting:
            return
        elapsed = time.monotonic() - start
        assert elapsed < settings.install_timeout, f"still loading {elapsed:.0f}s after the upgrade: " + ", ".join(waiting)
        time.sleep(5)


class _SetRun:
    """One upgrade set, shared by its tests: installed by its first test, upgraded by its first `after` test, uninstalled
    by `close()`. A failed install or upgrade fails every later test of the set with a pointer to the first one.
    """

    def __init__(self, client: E2EClient, ledger: Ledger, name: str) -> None:
        self.client = client
        self.ledger = ledger
        self.name = name
        self.scenarios: list[Scenario] = []
        self.skipped: dict[str, str] = {}
        self.upgraded = False
        self._stack = ExitStack()
        self._done: set[str] = set()
        self._failed: dict[str, BaseException] = {}

    def install(self) -> None:
        self._once("install", self._install)

    def upgrade(self) -> None:
        self.install()
        self._once("upgrade", self._upgrade)

    def close(self) -> None:
        self._stack.close()

    def _once(self, step: str, action: Callable[[], None]) -> None:
        if step in self._failed:
            error = self._failed[step]
            if isinstance(error, pytest.skip.Exception):
                pytest.skip(str(error))
            pytest.fail(f"{self.name} set: {step} failed in an earlier test: {type(error).__name__}: {error}", pytrace=False)
        if step in self._done:
            return
        try:
            action()
        except BaseException as e:
            e.add_note(f"{self.name} set: {step}")
            self._failed[step] = e
            raise
        self._done.add(step)

    def _install(self) -> None:
        settings = self.client.settings
        # In CI a skip would pass a blocking job that tested nothing, so DF_E2E_REQUIRE_UPGRADE turns it into a failure.
        cannot_run = pytest.fail if settings.require_upgrade else pytest.skip
        if not settings.upgrade_command:
            cannot_run("set DF_E2E_UPGRADE_COMMAND to run the upgrade test")
        # GPU or cloud scenarios the target cannot run are dropped; the rest of the set still goes through the upgrade.
        for scenario in upgrade_set(self.name):
            if reason := skip_reason(self.client, self.ledger, scenario):
                self.skipped[scenario.id] = reason
            else:
                self.scenarios.append(scenario)
        if not self.scenarios:
            cannot_run(f"the target can run none of the {self.name!r} upgrade set")
        self._stack.enter_context(installed(self.client, self.ledger, *self.scenarios))

    def _upgrade(self) -> None:
        self.upgraded = True
        _upgrade(self.client)
        _wait_until_loaded(self.client, self.scenarios)


class _SetRuns:
    """The upgrade sets of a session; only one is installed at a time, since sets share services."""

    def __init__(self, client: E2EClient, ledger: Ledger) -> None:
        self.client = client
        self.ledger = ledger
        self._runs: dict[str, _SetRun] = {}

    def get(self, name: str) -> _SetRun:
        if name not in self._runs:
            self.close()
            self._runs[name] = _SetRun(self.client, self.ledger, name)
        return self._runs[name]

    def close(self) -> None:
        for run in self._runs.values():
            run.close()


@pytest.fixture(scope="module")
def upgrade_runs(e2e_client: E2EClient, e2e_ledger: Ledger) -> Iterator[_SetRuns]:
    runs = _SetRuns(e2e_client, e2e_ledger)
    yield runs
    runs.close()


def _cases() -> list[object]:
    """Per set: every scenario before the upgrade, then every scenario after it - in this order."""
    return [
        pytest.param(set_name, stage, scenario, id=f"{set_name}-{stage}-{scenario.id}")
        for set_name in UPGRADE_SETS
        for stage in STAGES
        for scenario in upgrade_set(set_name)
    ]


def _with_logs(client: E2EClient, scenario: Scenario, check: Callable[[], None]) -> None:
    try:
        check()
    except BaseException as e:
        if client.settings.log_dir:
            for path in save_logs(client, [scenario]):
                e.add_note(f"service log: {path}")
        raise


def _assert_still_installed(client: E2EClient, scenario: Scenario) -> None:
    assert client.get_service(scenario.service_id).is_installed, f"{scenario.service_id} no longer installed after the upgrade"
    assert client.model_installed(scenario.service_id, scenario.model_id), (
        f"{scenario.service_id}/{scenario.model_id} no longer installed after the upgrade"
    )


@pytest.mark.upgrade
@pytest.mark.parametrize(("set_name", "stage", "scenario"), _cases())
def test_upgrade(set_name: str, stage: str, scenario: Scenario, upgrade_runs: _SetRuns) -> None:
    run = upgrade_runs.get(set_name)
    client = run.client
    if stage == "before":
        if run.upgraded:
            pytest.fail("this pre-upgrade check ran after the upgrade: the tests were reordered", pytrace=False)
        run.install()
        if reason := run.skipped.get(scenario.id):
            pytest.skip(reason)
        _with_logs(client, scenario, lambda: check_inference(client, scenario))
    else:
        run.upgrade()
        if reason := run.skipped.get(scenario.id):
            pytest.skip(reason)
        _with_logs(client, scenario, lambda: _assert_still_installed(client, scenario))
        _with_logs(client, scenario, lambda: _retry_until(client.settings.ready_timeout, lambda: check_inference(client, scenario)))
