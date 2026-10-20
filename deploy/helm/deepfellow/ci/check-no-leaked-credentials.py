#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.
"""Fail if a `plaintext` credential from a values file shows up anywhere in the render except a Secret.

Usage: helm template ... -f VALUES | check-no-leaked-credentials.py VALUES

Credentials must reach pods through secretKeyRef. A plaintext value copied into an env literal, a
ConfigMap, an annotation or a Job's arguments would sit unencrypted in the API server and in every
`kubectl get -o yaml`. Secret objects are skipped: that is where the value belongs, base64-encoded.
"""

import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml


def plaintexts(node: Any, path: str = "") -> Iterator[tuple[str, str]]:  # noqa: ANN401 - parsed YAML
    """Yield (values path, value) for every `plaintext` credential in a values tree."""
    if isinstance(node, dict):
        for key, child in node.items():
            here = f"{path}.{key}" if path else str(key)
            if key == "plaintext" and isinstance(child, str) and child:
                yield path, child
            else:
                yield from plaintexts(child, here)
    elif isinstance(node, list):
        for i, child in enumerate(node):
            yield from plaintexts(child, f"{path}[{i}]")


def main() -> int:
    """Check the rendered stream on stdin against the values file in argv[1]."""
    secrets = list(plaintexts(yaml.safe_load(Path(sys.argv[1]).read_text())))
    leaks: list[str] = []
    for doc in yaml.safe_load_all(sys.stdin):
        if not doc or doc.get("kind") == "Secret":
            continue
        text = yaml.safe_dump(doc)
        for path, value in secrets:
            if value in text:
                leaks.append(f"{path} appears in {doc['kind']}/{doc['metadata']['name']}")
    if leaks:
        sys.stderr.write("plaintext credentials outside a Secret:\n  " + "\n  ".join(leaks) + "\n")
        return 1
    sys.stdout.write(f"no leaks: {len(secrets)} plaintext credential(s) appear only in Secrets\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
