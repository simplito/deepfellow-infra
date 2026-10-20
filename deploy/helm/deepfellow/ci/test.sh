#!/usr/bin/env bash
# Every check the chart's CI runs, in one place, so a local run and CI cannot disagree.
# Needs: helm (+ the helm-unittest plugin), kubeconform, helm-docs, python3 with PyYAML.
set -euo pipefail

CHART="$(cd "$(dirname "$0")/.." && pwd)"
KUBE_VERSION="${KUBE_VERSION:-1.30.0}"
fixtures=("$CHART"/ci/*-values.yaml "$CHART/values-quickstart.yaml")

step() { printf '\n== %s\n' "$*"; }

step "fail-closed: the default values must NOT render"
# helm lint reports a failed template as INFO and exits 0, so assert on helm template instead.
if helm template ci "$CHART" >/dev/null 2>&1; then
  echo "default values rendered: the fail-closed contract is broken" >&2; exit 1
fi

step "helm lint every fixture"
for f in "${fixtures[@]}"; do helm lint "$CHART" -f "$f" --quiet; done

step "render every fixture: no duplicate objects, no leaked credentials, valid against the Kubernetes $KUBE_VERSION schemas"
for f in "${fixtures[@]}"; do
  echo "-- $(basename "$f")"
  out="$(helm template ci "$CHART" -f "$f" --namespace ci --kube-version "$KUBE_VERSION")"
  # Two templates emitting the same object render fine and then fight on apply; nothing else here
  # catches it (lint and kubeconform both accept it).
  printf '%s\n' "$out" | python3 "$CHART/ci/check-duplicates.py"
  # A plaintext credential must end up only inside a Secret, never in an env literal or ConfigMap.
  printf '%s\n' "$out" | python3 "$CHART/ci/check-no-leaked-credentials.py" "$f"
  printf '%s\n' "$out" | kubeconform -strict -summary -kubernetes-version "$KUBE_VERSION" -
done

step "helm unittest"
helm unittest "$CHART"

step "generated files are up to date"
python3 "$CHART/ci/sync-artifacthub-images.py" --check
helm-docs --chart-search-root "$CHART" --template-files README.md.gotmpl --dry-run 2>/dev/null \
  | diff -u "$CHART/README.md" - >/dev/null \
  || { echo "README.md is stale: run helm-docs --chart-search-root $CHART --template-files README.md.gotmpl" >&2; exit 1; }

printf '\nall chart checks passed\n'
