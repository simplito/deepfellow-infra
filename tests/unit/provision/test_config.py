# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""BACKENDS_JSON is validated once, at the boundary."""

import pytest
from deepfellow_provision import config
from deepfellow_provision.errors import ProvisionError


def test_parse_backends_accepts_a_valid_descriptor():
    out = config.parse_backends('[{"id": "m", "api_url": "http://b:8080"}]')
    assert out == [{"id": "m", "api_url": "http://b:8080"}]


def test_parse_backends_defaults_to_empty():
    assert config.parse_backends("[]") == []


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("not-json", "not valid JSON"),
        ('{"id": "m"}', "must be a JSON array"),
        ("[1]", "must be an object"),
        ('[{"id": "m"}]', "missing required key"),
        ('[{"api_url": "http://b"}]', "missing required key"),
    ],
)
def test_parse_backends_rejects_malformed_input_at_the_boundary(raw: str, expected: str):
    with pytest.raises(ProvisionError) as e:
        config.parse_backends(raw)
    assert expected in str(e.value)


def test_a_missing_required_env_fails_closed(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("DF_TEST_ABSENT", raising=False)
    with pytest.raises(ProvisionError) as e:
        config.env("DF_TEST_ABSENT", required=True)
    assert "DF_TEST_ABSENT" in str(e.value)
