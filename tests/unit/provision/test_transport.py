# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Response parsing, the prose probes for upstream errors, and unreachable endpoints."""

import urllib.request
from unittest.mock import patch

from deepfellow_provision import transport


def test_an_empty_body_is_an_empty_mapping():
    assert transport.parse_body(b"") == {}


def test_a_json_object_is_returned_as_is():
    assert transport.parse_body(b'{"a": 1}') == {"a": 1}


def test_html_from_the_spa_catch_all_does_not_raise():
    """Infra answers an unknown API path with 200 text/html; that must not look like a crash."""
    body = transport.parse_body(b"<!DOCTYPE html><html><body>app</body></html>")
    assert "raw" in body
    assert "DOCTYPE" in body["raw"]


def test_a_json_non_object_is_still_returned_as_data():
    assert transport.parse_body(b"[1, 2]") == {"raw": [1, 2]}


def test_body_probe_matches_upstreams_prose_errors():
    assert transport.body_says({"detail": "Service already installed"}, "already install")
    assert transport.body_says({"detail": "model exists"}, "exist")
    assert not transport.body_says({"detail": "nope"}, "exist")


def test_a_timeout_while_reading_the_response_is_reported_as_unreachable():
    """Not a URLError: before, it escaped the retry loops and crashed the Job."""
    with patch.object(urllib.request, "urlopen", side_effect=TimeoutError("The read operation timed out")):
        status, body = transport.request("GET", "http://infra/health")
    assert status == 0
    assert "timed out" in body["error"]


def test_a_connection_reset_is_reported_as_unreachable():
    with patch.object(urllib.request, "urlopen", side_effect=ConnectionResetError()):
        status, body = transport.request("GET", "http://infra/health")
    assert status == 0
    assert body["error"] == "ConnectionResetError"
