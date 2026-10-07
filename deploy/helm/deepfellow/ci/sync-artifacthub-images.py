#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.
"""Keep Chart.yaml's `artifacthub.io/images` annotation in sync with the images the chart deploys.

The list is derived from values.yaml (every `{repository, tag, digest}` block) plus the per-backend
defaults in templates/_backends.tpl, so it cannot drift from what a default install pulls.

  ci/sync-artifacthub-images.py          rewrite the annotation in Chart.yaml
  ci/sync-artifacthub-images.py --check  exit 1 if Chart.yaml is out of date (CI)
"""

import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

CHART = Path(__file__).resolve().parent.parent
BEGIN = "  artifacthub.io/images: |\n"


def image_blocks(node: Any) -> Iterator[dict[str, Any]]:  # noqa: ANN401 - parsed YAML is untyped
    """Yield every image block (a mapping with `repository` and `tag`) found under `node`."""
    if isinstance(node, dict):
        if "repository" in node and "tag" in node:
            if node["repository"]:
                yield node
            return
        for child in node.values():
            yield from image_blocks(child)


def backend_defaults() -> Any:  # noqa: ANN401 - parsed YAML is untyped
    """Parse the per-backend defaults block out of templates/_backends.tpl."""
    tpl = (CHART / "templates/_backends.tpl").read_text()
    body = re.search(r'define "deepfellow.modelBackendDefaults" -\}\}\n(.*?)\{\{- end -\}\}', tpl, re.S)
    if body is None:
        sys.exit("deepfellow.modelBackendDefaults not found in templates/_backends.tpl")
    return yaml.safe_load(body.group(1))


def images() -> list[dict[str, str]]:
    """List every distinct image reference a default install can pull, named by repository."""
    values = yaml.safe_load((CHART / "values.yaml").read_text())
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for block in [*image_blocks(values), *image_blocks(backend_defaults())]:
        ref = f"{block['repository']}:{block['tag']}" + (f"@{block['digest']}" if block.get("digest") else "")
        if ref not in seen:
            seen.add(ref)
            out.append({"name": block["repository"].rsplit("/", 1)[-1], "image": ref})
    return out


def render(imgs: list[dict[str, str]]) -> str:
    """Render the annotation as it must appear in Chart.yaml."""
    listing = yaml.safe_dump(imgs, sort_keys=False, width=200)
    return BEGIN + "".join(f"    {line}\n" for line in listing.splitlines())


def main() -> int:
    """Rewrite or check the annotation."""
    chart = (CHART / "Chart.yaml").read_text()
    start = chart.index(BEGIN)
    end = start + len(BEGIN)
    taken = 0
    for line in chart[end:].splitlines(keepends=True):
        if not line.startswith("    "):
            break
        taken += len(line)
    wanted = render(images())
    if chart[start : end + taken] == wanted:
        return 0
    if "--check" in sys.argv:
        sys.stderr.write("Chart.yaml artifacthub.io/images is out of date: run ci/sync-artifacthub-images.py\n")
        return 1
    (CHART / "Chart.yaml").write_text(chart[:start] + wanted + chart[end + taken :])
    sys.stdout.write("Chart.yaml updated\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
