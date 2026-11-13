# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Session setup for the e2e suite: skip when unconfigured, refuse unsafe targets, sweep leftovers at the end.

The suite is destructive - it installs and uninstalls services - so it only ever runs against a throwaway instance.
See tests/e2e/README.md.
"""

import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.e2e.client import E2EClient
from tests.e2e.config import E2ESettings
from tests.e2e.lifecycle import Ledger

E2E_DIR = Path(__file__).parent


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    if E2ESettings().base_url:
        return
    skip = pytest.mark.skip(reason="set DF_E2E_BASE_URL to run the e2e suite")
    for item in items:
        # Only tests that reach the target instance; checks of the scenario matrix itself run everywhere.
        if item.path.is_relative_to(E2E_DIR) and "e2e_ledger" in getattr(item, "fixturenames", ()):
            item.add_marker(skip)


@pytest.fixture(scope="session")
def e2e_settings() -> E2ESettings:
    settings = E2ESettings()
    missing = [name for name in ("admin_api_key", "api_key") if not getattr(settings, name).get_secret_value()]
    if missing:
        pytest.exit("e2e: missing " + ", ".join(f"DF_E2E_{name.upper()}" for name in missing), returncode=1)
    return settings


@pytest.fixture(scope="session")
def e2e_client(e2e_settings: E2ESettings) -> Iterator[E2EClient]:
    client = E2EClient(e2e_settings)
    yield client
    client.close()


def _wait_until_ready(client: E2EClient, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while not client.health():
        if time.monotonic() > deadline:
            pytest.exit(f"e2e: {client.settings.base_url} did not become healthy within {timeout:.0f}s", returncode=1)
        time.sleep(2)


def _check_version(client: E2EClient, expected: str) -> None:
    expected = expected.removeprefix("v")
    actual = client.info().version
    if actual != expected:
        pytest.exit(f"e2e: target reports version {actual}, expected {expected}", returncode=1)


def _check_clean(client: E2EClient) -> None:
    installed = [f"{s.id}|{s.instance}" for s in client.list_services() if s.is_installed]
    if installed:
        pytest.exit(
            "e2e: refusing to run against an instance with installed services (" + ", ".join(installed) + "). "
            "The suite is destructive - point it at a throwaway instance, or set DF_E2E_ALLOW_DIRTY=1.",
            returncode=1,
        )


@pytest.fixture(scope="session")
def e2e_ledger(e2e_settings: E2ESettings, e2e_client: E2EClient) -> Iterator[Ledger]:
    """Guarded entry point for every scenario; its teardown sweeps whatever a scenario failed to remove."""
    _wait_until_ready(e2e_client, e2e_settings.ready_timeout)
    if e2e_settings.expected_version:
        _check_version(e2e_client, e2e_settings.expected_version)
    if e2e_settings.allow_dirty:
        preexisting = {s.id for s in e2e_client.list_services() if s.is_installed}
    else:
        _check_clean(e2e_client)
        preexisting = set[str]()

    ledger = Ledger(e2e_client, preexisting)
    yield ledger
    if errors := ledger.sweep():
        pytest.fail("e2e: end-of-session sweep could not remove: " + "; ".join(errors))
