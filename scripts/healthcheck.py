#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Healthcheck script for checking if the server is responding.

Returns exit code 0 if healthy, 1 if unhealthy.
"""

import os
import sys
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

# ruff: noqa: T201


def get_health_url() -> str:
    """Build the health URL, using https when uvicorn is configured with a TLS certificate."""
    scheme = "https" if os.environ.get("UVICORN_SSL_CERTFILE") else "http"
    return f"{scheme}://localhost:8086/health"


def check_health() -> int:
    """Check if the infra is healthy."""
    url = get_health_url()
    try:
        # Timeout after 5 seconds
        with urlopen(url, timeout=5) as response:
            if response.status == 200:
                print(f"✓ Infra is healthy (status: {response.status})")
                return 0
            print(f"✗ Infra returned status: {response.status}", file=sys.stderr)
            return 1

    except HTTPError as e:
        print(f"✗ HTTP Error: {e.code} - {e.reason}", file=sys.stderr)
        return 1
    except URLError as e:
        print(f"✗ URL Error: {e.reason}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"✗ Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(check_health())
