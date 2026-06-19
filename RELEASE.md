# Release Process

## Overview

Releases go through a Merge Request to `main`. The tag is pushed **after** the MR is merged, not before.

## Steps

1. Create a release branch from `main`:
   ```bash
   git checkout main && git pull
   git checkout -b release/0.30.0
   ```
2. Update `[Unreleased]` section in `CHANGELOG.md` — rename it to the new version with today's date, e.g. `## [0.30.0] - 2026-05-22`.
3. Update `version` in `pyproject.toml`.
4. Commit:
   ```bash
   git commit -am "Release 0.30.0"
   git push origin release/0.30.0
   ```
5. Open an MR targeting `main` and get it reviewed and merged via the GitLab UI.
6. After the MR is merged, tag the merge commit on `main`:
   ```bash
   git tag v0.30.0
   git push origin v0.30.0
   ```

CI picks up the tag and automatically pushes to [github.com/simplito/deepfellow-infra](https://github.com/simplito/deepfellow-infra).

## Tag format

Tags must match `v<major>.<minor>.<patch>` (e.g. `v0.30.0`). Only these trigger the GitHub push.

## Prerequisites (one-time setup)

Add a CI/CD variable in GitLab → Settings → CI/CD → Variables:

| Variable | Value |
|---|---|
| `GITHUB_MIRROR_TOKEN` | GitHub Personal Access Token with `repo` scope (classic) or `contents:write` (fine-grained) for `simplito/deepfellow-infra` |
