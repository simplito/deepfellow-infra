#!/usr/bin/env python3

# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Healthcheck script for checking if the server is responding.

Returns exit code 0 if healthy, 1 if unhealthy.
"""

import sys
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

# ruff: noqa: T201


def check_health() -> int:
    """Check if the infra is healthy."""
    url = "http://localhost:8086/docs"
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
