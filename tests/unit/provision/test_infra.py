# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Infra side: model readiness and reconciling an existing custom model."""

from collections.abc import Callable
from typing import Any
from unittest.mock import patch

import pytest
from deepfellow_provision import infra, transport
from deepfellow_provision.errors import ProvisionError

# -------------------------------------------------------------- model readiness


def test_model_readiness_uses_the_non_admin_key():
    seen_headers: list[dict[str, str]] = []

    def fake_request(_m: str, _u: str, headers: dict[str, str] | None = None, *_a: object, **_k: object):
        seen_headers.append(headers or {})
        return 200, {"data": [{"id": "m1"}]}

    with patch.object(transport, "request", side_effect=fake_request):
        infra.wait_model_ready("http://i", "the-v1-key", "m1")

    assert seen_headers[0]["Authorization"] == "Bearer the-v1-key"


def test_model_readiness_reports_the_status_and_what_it_saw():
    with (
        patch.object(transport, "request", return_value=(401, {})),
        patch.object(infra.time, "sleep"),
        pytest.raises(ProvisionError) as e,
    ):
        infra.wait_model_ready("http://i", "wrong-key", "m1", attempts=2, delay=0)
    message = str(e.value)
    assert "401" in message, "an auth failure must not look like a missing model"
    assert "models seen" in message


def test_model_readiness_returns_once_the_model_appears():
    responses = [(200, {"data": []}), (200, {"data": [{"id": "m1"}]})]
    with patch.object(transport, "request", side_effect=responses), patch.object(infra.time, "sleep"):
        infra.wait_model_ready("http://i", "k", "m1", attempts=5, delay=0)


# ------------------------------------------------- custom model reconciliation

Fake = Callable[..., tuple[int, dict[str, Any]]]


def _capture_url(status: int, body: dict[str, Any]) -> tuple[list[str], Fake]:
    urls: list[str] = []

    def fake(_method: str, url: str, *_a: object, **_k: object) -> tuple[int, dict[str, Any]]:
        urls.append(url)
        return status, body

    return urls, fake


def _reconcile(body: dict[str, Any], context_length: int = 2048, sid: str = "openai%7Cb", model: str = "m1") -> list[str]:
    urls, fake = _capture_url(200, body)
    with patch.object(transport, "request", side_effect=fake):
        infra._reconcile_custom_model("http://infra", {}, sid, model, context_length)  # pyright: ignore[reportPrivateUsage]
    return urls


def test_reconcile_asks_for_the_model_on_a_route_that_exists():
    """`/models/<id>` is not a route: it hits the SPA catch-all and returns a page, not a model."""
    urls = _reconcile({"custom_spec": {"type": "llm", "context_length": 2048}})
    assert urls, "the model must actually be looked up"
    assert "models/_?model_id=m1" in urls[0]
    assert not urls[0].endswith("/models/m1"), "that path is served by the SPA, not the API"


def test_reconcile_rejects_an_untyped_model():
    with pytest.raises(ProvisionError) as e:
        _reconcile({"custom_spec": {"type": None, "context_length": 2048}})
    assert "no type" in str(e.value)


def test_reconcile_rejects_a_drifted_context_length():
    with pytest.raises(ProvisionError) as e:
        _reconcile({"custom_spec": {"type": "llm", "context_length": 4096}})
    assert "4096" in str(e.value)


def test_reconcile_accepts_a_matching_model():
    _reconcile({"custom_spec": {"type": "llm", "context_length": 2048}})


# The exact body the live Infra returns for an installed custom model. `type` is top level,
# `custom_spec` is null, and `spec` is the FIELD SCHEMA: reading values out of `spec` yields
# None for everything, which used to fail every healthy re-run with a bogus "type=None".
_REAL_INSTALLED_MODEL: dict[str, Any] = {
    "id": "demo-llm",
    "service": "openai|demo",
    "type": "llm",
    "installed": {"spec": None, "registration_id": "22bae027-b2c2-4dd8-9c80-44679db26014"},
    "downloaded": True,
    "custom": "cb6a1240-9db7-4afb-a0f7-c51b4fa3cc9d",
    "size": "",
    "spec": {
        "fields": [
            {
                "type": "text",
                "name": "alias",
                "description": "Model alias",
                "default": None,
                "placeholder": None,
                "required": False,
                "values": None,
            }
        ]
    },
    "has_docker": False,
    "vram_estimate_gb": None,
    "is_loaded": None,
    "variant": None,
    "command": None,
    "base_image": None,
    "custom_spec": None,
}


def test_a_healthy_installed_model_passes_reconciliation():
    """Regression: the real response has no custom_spec, and `spec` is a schema, not values."""
    _reconcile(_REAL_INSTALLED_MODEL, sid="openai%7Cdemo", model="demo-llm")


def test_the_field_schema_is_never_read_as_values():
    """`spec.fields[0].type` is "text"; mistaking it for the model's type must not happen either."""
    body = {**_REAL_INSTALLED_MODEL, "type": None}
    with pytest.raises(ProvisionError) as e:
        _reconcile(body, sid="openai%7Cdemo", model="demo-llm")
    assert "no type" in str(e.value)


def test_a_context_length_only_in_custom_spec_is_still_compared():
    body = {**_REAL_INSTALLED_MODEL, "custom_spec": {"type": "llm", "context_length": 4096}}
    with pytest.raises(ProvisionError) as e:
        _reconcile(body, sid="openai%7Cdemo", model="demo-llm")
    assert "4096" in str(e.value)


def test_an_absent_context_length_is_not_invented():
    """The real body carries none; that must not be read as a mismatch."""
    _reconcile(_REAL_INSTALLED_MODEL, context_length=999, sid="openai%7Cdemo", model="demo-llm")


def test_reconcile_stops_when_the_model_cannot_be_read():
    """An unreadable model is unverified; the run must not grant access to it."""
    _, fake = _capture_url(500, {})
    with patch.object(transport, "request", side_effect=fake), pytest.raises(ProvisionError) as e:
        infra._reconcile_custom_model("http://infra", {}, "openai%7Cb", "m1", 2048)  # pyright: ignore[reportPrivateUsage]
    assert "HTTP 500" in str(e.value)


def test_reconcile_rejects_a_model_of_another_type():
    with pytest.raises(ProvisionError) as e:
        _reconcile({"custom_spec": {"type": "embedding", "context_length": 2048}})
    assert "'embedding'" in str(e.value)


# -------------------------------------------------------------- existing service instance

_BACKEND: dict[str, Any] = {"id": "m1", "api_url": "http://llm:8080", "api_key_env": "DF_BACKEND_R_API_KEY"}


def _existing_instance(installed: dict[str, Any] | None, get_status: int = 200, put_status: int = 200):
    calls: list[tuple[str, dict[str, Any] | None]] = []

    def fake(method: str, _url: str, _auth: object = None, body: dict[str, Any] | None = None, **_k: object) -> tuple[int, dict[str, Any]]:
        calls.append((method, body))
        if method == "GET":
            return get_status, {"installed": installed}
        return put_status, {}

    return calls, fake


def _reconcile_instance(installed: dict[str, Any] | None, key: str, **kw: int) -> list[tuple[str, dict[str, Any] | None]]:
    calls, fake = _existing_instance(installed, **kw)
    spec = {"api_url": "http://llm:8080", "api_key": key}
    with patch.object(transport, "request", side_effect=fake):
        infra._reconcile_existing_instance("http://infra", {}, "openai%7Cr", "openai|r", _BACKEND, spec)  # pyright: ignore[reportPrivateUsage]
    return calls


def test_an_unreadable_existing_instance_stops_the_run():
    with pytest.raises(ProvisionError) as e:
        _reconcile_instance(None, "k", get_status=503)
    assert "HTTP 503" in str(e.value)


def test_an_unchanged_key_is_left_alone():
    calls = _reconcile_instance({"api_url": "http://llm:8080", "api_key": "same"}, "same")
    assert [m for m, _ in calls] == ["GET"]


def test_a_rotated_key_is_pushed_to_infra():
    calls = _reconcile_instance({"api_url": "http://llm:8080", "api_key": "old"}, "new")
    assert [m for m, _ in calls] == ["GET", "PUT"]
    assert calls[1][1] == {"spec": {"api_url": "http://llm:8080", "api_key": "new"}, "stream": False, "ignore_warnings": True}


def test_a_failed_key_update_stops_the_run():
    with pytest.raises(ProvisionError) as e:
        _reconcile_instance({"api_url": "http://llm:8080", "api_key": "old"}, "new", put_status=500)
    assert "API key" in str(e.value)
    assert "new" not in str(e.value), "a key must never reach the message"
    assert "old" not in str(e.value), "a key must never reach the message"
