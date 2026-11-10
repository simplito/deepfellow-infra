# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Unit tests for server/utils/exceptions.py."""

from server.utils.exceptions import LOADING_RETRY_AFTER_SECONDS, still_loading_error


def test_still_loading_error_is_retryable_503() -> None:
    exc = still_loading_error("Service", "ollama")

    assert exc.status_code == 503
    assert exc.detail == "Service ollama is still loading, try again shortly"
    assert exc.headers == {"Retry-After": str(LOADING_RETRY_AFTER_SECONDS)}
