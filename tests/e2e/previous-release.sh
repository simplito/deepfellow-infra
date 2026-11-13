#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.
#
# Print the release tag the upgrade test starts from:
#
#   tests/e2e/previous-release.sh <tag>   # the highest vX.Y.Z tag below <tag>
#   tests/e2e/previous-release.sh         # the highest vX.Y.Z tag
#
# Fails when there is none. Reads local tags, so fetch them first (`git fetch --tags`).

set -euo pipefail

below="${1:-}"
tags="$(git tag --list 'v*' | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' || true)"
if [ -n "$below" ]; then
  # Sort <tag> in with the others and drop it and everything after it.
  tags="$(printf '%s\n%s\n' "$tags" "$below" | sort -uV | sed "/^${below//./\\.}\$/,\$d")"
fi
previous="$(printf '%s\n' "$tags" | sed '/^$/d' | sort -V | tail -n1)"
if [ -z "$previous" ]; then
  echo "no release tag${below:+ below $below}" >&2
  exit 1
fi
echo "$previous"
