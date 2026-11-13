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

### 3. Update the version in `pyproject.toml` and `uv.lock`

`uv.lock` stores the project version too. Change the `version` of the `df-infra` package in it by hand, to the same value as in `pyproject.toml`, and do not run `uv lock`: it also rewrites the `revision` line when the local uv writes a different one than the uv that last wrote the file. Check that `git diff uv.lock` shows that one line only.

### 4. Commit and push

```bash
git commit -am "Release 0.31.0"
git push origin release/0.31.0
```

### 5. Open a merge request on release branch on Gitlab.

### 6. Wait for pipeline finish.

The release branch pipeline runs the e2e suite on the GPU runner: the full matrix and the upgrade from the latest
release, about an hour (see `tests/e2e/README.md`). The MR can't be merged until it passes. If it fails, fix it on
`main`, then recreate the release branch from `main`: no tag exists yet, so no version is lost.

### 7. Merge the MR to main.

Only if nothing was merged to `main` since the release branch was created: the MR must not say the source branch is
behind the target branch. Otherwise those commits would be tagged without the e2e run: recreate the release branch from `main`, so the
pipeline runs again on everything that gets tagged.

### 8. Wait for the pipeline on `main` to finish

It builds the image of the merge commit, which the tag in step 9 publishes as the release, and deploys it to staging.
It runs no e2e: the release branch did (step 6), and step 7 ensures the merge commit holds the same code. The push to
GitHub happens on the tag.

### 9. Tag the merge commit on `main`

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