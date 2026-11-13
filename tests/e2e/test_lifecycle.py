# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Service lifecycle per scenario: install service -> install model -> inference through `/v1/*` -> uninstall."""

import pytest

from tests.e2e.client import E2EClient
from tests.e2e.inference import check_inference
from tests.e2e.lifecycle import Ledger, installed, skip_reason
from tests.e2e.scenarios import BROKEN_SCENARIOS, SCENARIOS, Scenario

# The smallest GPU scenario model still takes a few hundred MB of VRAM once loaded.
MIN_GPU_FOOTPRINT_GB = 0.2


def _param(scenario: Scenario) -> object:
    marks = [getattr(pytest.mark, mark) for mark in sorted(scenario.marks)]
    if reason := BROKEN_SCENARIOS.get(scenario.id):
        marks.append(pytest.mark.xfail(reason=f"known Infra bug: {reason}", strict=True))
    return pytest.param(scenario, id=scenario.id, marks=marks)


@pytest.mark.parametrize("scenario", [_param(s) for s in SCENARIOS])
def test_lifecycle(scenario: Scenario, e2e_client: E2EClient, e2e_ledger: Ledger) -> None:
    if reason := skip_reason(e2e_client, e2e_ledger, scenario):
        pytest.skip(reason)

    vram_before = e2e_client.gpu_used_vram_gb() if "gpu" in scenario.marks else None
    with installed(e2e_client, e2e_ledger, scenario):
        assert e2e_client.served_model(scenario.served_model), f"{scenario.served_model} is installed but not in /v1/models"
        check_inference(e2e_client, scenario)
        if vram_before is not None:
            vram_after = e2e_client.gpu_used_vram_gb() or 0.0
            # Some backends fall back to the CPU silently when the GPU fails to initialise in the container.
            assert vram_after - vram_before >= MIN_GPU_FOOTPRINT_GB, f"VRAM {vram_before:.2f} -> {vram_after:.2f} GB: not on the GPU?"
    assert e2e_client.served_model(scenario.served_model) is None, f"{scenario.served_model} is still in /v1/models after uninstall"
