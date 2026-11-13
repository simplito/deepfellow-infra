# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Every service the target offers has a scenario or a recorded reason why not (`NOT_COVERED` in scenarios.py).

Runs with the full matrix (nightly), not in the release gate: a new service must not block a release, but it must not
go unnoticed either.
"""

from tests.e2e.client import E2EClient
from tests.e2e.lifecycle import Ledger
from tests.e2e.scenarios import NOT_COVERED, SCENARIOS


def test_every_offered_service_is_covered_or_excluded(e2e_client: E2EClient, e2e_ledger: Ledger) -> None:
    offered = {s.id for s in e2e_client.list_services() if s.instance == "default"}
    covered = {s.service_id for s in SCENARIOS}
    missing = sorted(offered - covered - set(NOT_COVERED))
    assert not missing, "services with no scenario and no NOT_COVERED entry: " + ", ".join(missing)
