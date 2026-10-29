# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Tests for scripts/healthcheck.py."""

from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

import pytest

from scripts.healthcheck import check_health, get_health_url


def _make_response(status: int) -> MagicMock:
    resp = MagicMock()
    resp.status = status
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    return resp


@pytest.mark.parametrize(("status_code", "result"), [(200, 0), (503, 1)])
def test_check_health_200(status_code: int, result: int) -> None:
    with patch("scripts.healthcheck.urlopen", return_value=_make_response(status_code)):
        assert check_health() == result


@pytest.mark.parametrize(
    ("side_effect"),
    [
        HTTPError(
            url="http://localhost:8086/health",
            code=500,
            msg="Internal Server Error",
            hdrs=None,  # pyright: ignore[reportArgumentType]
            fp=None,
        ),
        URLError("Connection refused"),
        RuntimeError("boom"),
    ],
)
def test_check_health_on_http_error(side_effect: Exception) -> None:
    with patch("scripts.healthcheck.urlopen", side_effect=side_effect):
        assert check_health() == 1


def test_get_health_url_defaults_to_http(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("UVICORN_SSL_CERTFILE", raising=False)
    assert get_health_url() == "http://localhost:8086/health"


def test_get_health_url_uses_https_when_tls_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UVICORN_SSL_CERTFILE", "/certs/cert.pem")
    assert get_health_url() == "https://localhost:8086/health"
