# Release Process

## Overview

Releases go through a Merge Request to `main`. The tag is pushed **after** the MR is merged, not before.

## Steps

### 1. Create a release branch from `main`

```bash
git checkout main && git pull
git checkout -b release/0.31.0
```

### 2. Update `CHANGELOG.md`

Rename the `[Unreleased]` section to the new version with today's date, e.g. `## [0.31.0] - 2026-05-22`.

### 3. Update the version in `pyproject.toml`

### 4. Commit and push

```bash
git commit -am "Release 0.31.0"
git push origin release/0.31.0
```

### 5. Open a merge request on release branch on Gitlab.

### 6. Wait for pipeline finish.

### 7. Merge the MR to main.

### 8. Wait for the pipeline on `main` to finish

CI picks up the merge commit and automatically pushes to [github.com/simplito/deepfellow-infra](https://github.com/simplito/deepfellow-infra).

### 9. Verify staging

Open [https://dfinfra.test.simplito.com/](https://dfinfra.test.simplito.com/) and confirm the changes look correct.

### 10. Tag the merge commit on `main`

```bash
git checkout main && git pull
git tag v0.31.0
git push origin v0.31.0
```

## Tag format

Tags must match `v<major>.<minor>.<patch>` (e.g. `v0.31.0`). Only these trigger the GitHub push.

## Prerequisites (one-time setup)

Add a CI/CD variable in GitLab → Settings → CI/CD → Variables:

| Variable | Value |
|---|---|
| `GITHUB_MIRROR_TOKEN` | GitHub Personal Access Token with `repo` scope (classic) or `contents:write` (fine-grained) for `simplito/deepfellow-infra` |