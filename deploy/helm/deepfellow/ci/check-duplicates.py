#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.
"""Fail if a rendered manifest stream (stdin) contains two objects with the same kind/namespace/name.

Two templates emitting the same object render fine and then fight on apply; neither `helm lint`
nor kubeconform notices.
"""

import sys

import yaml


def main() -> int:
    """Read manifests from stdin and report duplicates."""
    seen: set[tuple[str, str, str]] = set()
    dupes: list[str] = []
    for doc in yaml.safe_load_all(sys.stdin):
        if not doc:
            continue
        key = (doc["kind"], doc["metadata"].get("namespace", ""), doc["metadata"]["name"])
        if key in seen:
            dupes.append("/".join(part for part in key if part))
        seen.add(key)
    if dupes:
        sys.stderr.write("duplicate objects: " + ", ".join(dupes) + "\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
