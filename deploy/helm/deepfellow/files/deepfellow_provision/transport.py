# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""JSON over HTTP with the standard library, and waiting for a service to come up."""

import json
import ssl
import time
import urllib.error
import urllib.request
from typing import Any

from deepfellow_provision.errors import ProvisionError
from deepfellow_provision.log import logger

Json = dict[str, Any]


def parse_body(raw: bytes) -> Json:
    """Decode a response body, never raising.

    A 2xx is not a promise of JSON: Infra serves a single-page app, so a URL that matches no API
    route comes back as 200 text/html. Parsing that unguarded turns a wrong URL into an opaque
    JSONDecodeError far from its cause, so a non-JSON body is handed back as data instead.
    """
    if not raw:
        return {}
    try:
        parsed: object = json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw.decode("utf-8", "replace")}
    return parsed if isinstance(parsed, dict) else {"raw": parsed}


def request(
    method: str,
    url: str,
    headers: dict[str, str] | None = None,
    body: object | None = None,
    ctx: ssl.SSLContext | None = None,
    timeout: int = 30,
) -> tuple[int, Json]:
    """Send a JSON request and return (status, parsed body). Status 0 means unreachable."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
            return resp.status, parse_body(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, parse_body(e.read())
    except urllib.error.URLError as e:
        return 0, {"error": str(e.reason)}
    except OSError as e:
        # A timeout or reset while reading the response is not a URLError; report it as
        # unreachable like one, so callers retry instead of crashing the Job.
        return 0, {"error": str(e) or type(e).__name__}


def body_says(body: Json, *needles: str) -> bool:
    """Whether the serialized body mentions every one of `needles`.

    DeepFellow reports "already there" outcomes as prose rather than as a stable code, so these
    string probes are a deliberate, and fragile, protocol assumption. They live here so there is
    one place to revisit when the upstream API grows a structured error field.
    """
    haystack = json.dumps(body).lower()
    return all(n in haystack for n in needles)


def wait_healthy(url: str, label: str, attempts: int = 60, delay: int = 5, *, accept_any: bool = False) -> None:
    """Block until `url`/health answers 200, or fail after `attempts`.

    `accept_any` counts any HTTP response (the connection was accepted) as up, for components
    whose /health shape is not established.
    """
    for _ in range(attempts):
        status, _ = request("GET", f"{url}/health")
        if status == 200 or (accept_any and status != 0):
            logger.info("%s reachable", label)
            return
        time.sleep(delay)
    msg = f"{label} not reachable at {url}"
    raise ProvisionError(msg)
