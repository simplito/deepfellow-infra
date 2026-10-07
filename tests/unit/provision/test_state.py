# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""The state Secret is empty or whole; a half-written one stops the run."""

import pytest
from deepfellow_provision import state
from deepfellow_provision.errors import ProvisionError


def test_state_may_be_absent():
    state.assert_whole(None, "s")


def test_complete_state_is_accepted():
    state.assert_whole({"organization-id": "o", "project-id": "p", "project-api-key": "k"}, "s")


@pytest.mark.parametrize(
    "current",
    [
        {"organization-id": "o"},
        {"organization-id": "o", "project-id": "p"},
        {"project-api-key": "k"},
    ],
)
def test_partial_state_fails_closed(current: dict[str, str]):
    with pytest.raises(ProvisionError) as e:
        state.assert_whole(current, "my-secret")
    assert "partial" in str(e.value)
    assert "manual recovery" in str(e.value)


def test_completeness_needs_all_three_keys():
    assert state.is_complete({"organization-id": "o", "project-id": "p", "project-api-key": "k"})
    assert not state.is_complete({"organization-id": "o", "project-id": "p"})
    assert not state.is_complete(None)
