# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Rules of the scenario matrix itself; these run without a target instance."""

import json
from collections import Counter
from pathlib import Path

from tests.e2e.scenarios import BROKEN_CHAT_APIS, BROKEN_SCENARIOS, NOT_COVERED, SCENARIOS, UPGRADE_SETS, Mark, upgrade_set

PLACEMENT: frozenset[Mark] = frozenset({"cpu", "gpu", "cloud"})

STATIC = Path(__file__).parents[2] / "static"
# Model catalog Infra ships per service; services missing here (cloud providers) list models from the provider.
CATALOGS = {
    "ollama": "ollama-min.json",
    "llamacpp": "llamacpp-min.json",
    "rerank": "rerank-min.json",
    "sglang": "sglang-min.json",
    "speaches-ai": "speaches-min.json",
    "stable-diffusion-next": "stable-diffusion-min.json",
    "vllm": "vllm-min.json",
}


def test_scenario_ids_are_unique() -> None:
    duplicates = [scenario_id for scenario_id, count in Counter(s.id for s in SCENARIOS).items() if count > 1]
    assert not duplicates


def test_every_scenario_declares_where_it_runs() -> None:
    assert all(s.marks & PLACEMENT for s in SCENARIOS)


def test_gate_is_cpu_only_and_needs_no_credentials() -> None:
    gate = [s for s in SCENARIOS if "gate" in s.marks]
    assert gate
    assert all(s.marks == {"gate", "cpu"} for s in gate)
    assert all(not s.service_secrets for s in gate)


def test_upgrade_sets_name_known_scenarios() -> None:
    known = {s.id for s in SCENARIOS}
    assert all(set(ids) <= known for ids in UPGRADE_SETS.values())


def test_upgrade_sets_install_each_service_with_one_spec_and_no_credentials() -> None:
    for name in UPGRADE_SETS:
        scenarios = upgrade_set(name)
        specs = {s.service_id: s.service_spec for s in scenarios}
        assert all(s.service_spec == specs[s.service_id] for s in scenarios), name
        assert all(not s.service_secrets for s in scenarios), name


def test_gate_upgrade_set_is_the_gate() -> None:
    assert set(UPGRADE_SETS["gate"]) == {s.id for s in SCENARIOS if "gate" in s.marks}


def test_scenario_models_are_in_the_shipped_catalogs() -> None:
    # A catalog refresh that drops a pinned model should fail here, in the MR, rather than in the nightly run.
    missing = [
        f"{s.id}: {s.model_id}"
        for s in SCENARIOS
        if s.service_id in CATALOGS and json.dumps(s.model_id) not in (STATIC / CATALOGS[s.service_id]).read_text()
    ]
    assert not missing, missing


def test_not_covered_services_have_no_scenario() -> None:
    assert not set(NOT_COVERED) & {s.service_id for s in SCENARIOS}


def test_known_bugs_name_known_scenarios_and_apis() -> None:
    assert set(BROKEN_SCENARIOS) <= {s.id for s in SCENARIOS}
    assert {service for service, _ in BROKEN_CHAT_APIS} <= {s.service_id for s in SCENARIOS if s.modality == "chat"}


def test_gate_has_no_known_bugs() -> None:
    # A known-broken gate scenario would turn the release gate into a formality.
    assert not set(UPGRADE_SETS["gate"]) & set(BROKEN_SCENARIOS)
