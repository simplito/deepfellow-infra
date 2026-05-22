## Release [version]
<!-- e.g. Release 0.3.0 -->

## Checklist

- [ ] `[Unreleased]` renamed to `[X.Y.Z] - YYYY-MM-DD` in `CHANGELOG.md`
- [ ] `version` bumped in `pyproject.toml`
- [ ] MR title is `Release X.Y.Z`
- [ ] Targeting `main`

## After merge

Tag the merge commit and push:

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

CI will mirror the tag to GitHub automatically.
