#!/bin/sh
# DFMAIN-78: a branch must add its entry under `## [Unreleased]` in CHANGELOG.md. Catches the case where a release
# lands on main while the branch is open: the rebase merges cleanly, but the entry ends up under the released version.
# DFINFRA-454: GitLab runs no pipeline for that merge onto the newer main, so the check also runs on main itself. There it
# compares CHANGELOG.md with the previous tip of main and fails on a line added outside `## [Unreleased]`.
set -eu

DEFAULT_BRANCH=${CI_DEFAULT_BRANCH:-main}

# Only a pipeline for the default branch itself; locally CI_COMMIT_BRANCH is unset, so a local run is always a branch run.
on_default=0
if [ "${CI_COMMIT_BRANCH:-}" = "$DEFAULT_BRANCH" ]; then on_default=1; fi

# GitLab job logs render ANSI colors too; NO_COLOR turns them off.
if [ -z "${NO_COLOR:-}" ]; then
    OK=$(printf '\033[32mOK\033[0m')
    FAIL=$(printf '\033[31mFAIL\033[0m')
    NOTE=$(printf '\033[1mNote:\033[0m')
else
    OK=OK
    FAIL=FAIL
    NOTE=Note:
fi

test -f CHANGELOG.md || { echo "$FAIL: CHANGELOG.md not found. This job expects a Keep a Changelog file in the repo root."; exit 1; }

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# Intentional opt-out: [skip changelog-check] / [skip-changelog-check] in the title of the branch's open merge request.
# A branch pipeline only gets the title through the API, so this only runs in CI. No merge request yet (first push)
# means nothing to skip. If the title cannot be read the job fails: it cannot tell whether the branch opted out.
# On the default branch no merge request is open any more, so the one that produced the pipeline's commit is asked for
# instead (a direct push has none). The commit can also belong to a merge request aimed at another branch; that one
# does not count. No `state=merged` filter: the merge request may not be marked merged yet when the pipeline starts.
if [ -n "${CI_JOB_TOKEN:-}" ]; then
    if [ "$on_default" = 1 ]; then
        mr_url="$CI_API_V4_URL/projects/$CI_PROJECT_ID/repository/commits/$CI_COMMIT_SHA/merge_requests"
        mr_pick='.[] | select(.target_branch == $default) | .title'
        mr_of="the merge request merged into $DEFAULT_BRANCH"
        set --
    else
        mr_url="$CI_API_V4_URL/projects/$CI_PROJECT_ID/merge_requests"
        mr_pick='.[].title'
        mr_of="this branch's merge request"
        set -- --data-urlencode "source_branch=$CI_COMMIT_BRANCH" --data-urlencode "state=opened"
    fi
    code=$(curl -sS --get -o "$tmp/mrs.json" -w '%{http_code}' --header "JOB-TOKEN: $CI_JOB_TOKEN" "$@" "$mr_url") || code=000
    if [ "$code" != "200" ]; then
        echo "$FAIL: could not read the merge request title (HTTP $code), so the check cannot tell whether $mr_of opted out. Retry this job."
        exit 1
    fi
    titles=$(jq -r --arg default "$DEFAULT_BRANCH" "$mr_pick" "$tmp/mrs.json") || { echo "$FAIL: unexpected answer from the merge request API. Retry this job."; exit 1; }
    if printf '%s\n' "$titles" | grep -qiE '\[skip[ -]changelog[ -]check\]'; then
        echo "SKIPPED: [skip changelog-check] found in the title of $mr_of."
        exit 0
    fi
fi

if [ "$on_default" = 1 ]; then
    # What the last push brought in is everything since the previous tip of the default branch. GitLab sets it to all
    # zeros when it has none (a manual run, a new branch); the parent commit then stands in, which for a merge commit
    # is the previous tip as well.
    BASE=${CI_COMMIT_BEFORE_SHA:-}
    if ! git cat-file -e "$BASE^{commit}" 2>/dev/null; then
        BASE=$(git rev-parse --verify --quiet HEAD^) || { echo "$OK: the first commit has nothing to compare with."; exit 0; }
        echo "$NOTE the previous tip of $DEFAULT_BRANCH is not known, comparing with the parent commit."
    fi
else
    # A CI clone has no origin/main yet. Locally the check stays read-only and uses the origin/main already there.
    if [ -n "${CI:-}" ]; then
        git fetch --quiet origin "$DEFAULT_BRANCH"
    else
        echo "Compared with origin/$DEFAULT_BRANCH @ $(git log -1 --format='%h, %cr' "origin/$DEFAULT_BRANCH")."
        echo "$NOTE CI uses the latest $DEFAULT_BRANCH - run \`git fetch origin $DEFAULT_BRANCH\` first, or the pipeline may fail."
    fi
    BASE=$(git merge-base "origin/$DEFAULT_BRANCH" HEAD)
fi

# Line numbers (in the CHANGELOG.md on disk, the file the next step reads) of everything the branch added.
# Against the working tree, not HEAD: locally the file may hold uncommitted edits (always with jj), which shift lines.
git diff --unified=0 --no-color "$BASE" -- CHANGELOG.md | awk '
  /^@@/ {
    match($0, /\+[0-9]+(,[0-9]+)?/)
    split(substr($0, RSTART + 1, RLENGTH - 1), r, ",")
    for (i = 0; i < (r[2] == "" ? 1 : r[2]); i++) print r[1] + i
  }' > "$tmp/added-lines.txt"

# An added entry (not blank, not a heading) must sit under `## [Unreleased]`.
# On the default branch only that is checked: the branch pipeline already insisted on an entry, so a push without one
# (a direct commit) is not a failure. A push that adds a version section is a release - it renames [Unreleased] - and passes.
awk -v ok="$OK" -v fail="$FAIL" -v added="$tmp/added-lines.txt" -v on_default="$on_default" -v base="$BASE" '
  BEGIN {
    while ((getline n < added) > 0) added_at[n] = 1
    who = on_default ? "this push" : "this branch"
  }
  /^## / {
    section = $0
    if (added_at[NR] && tolower($0) ~ /^## \[unreleased\]/) created++
    if (added_at[NR] && $0 ~ /^## \[[0-9]+\.[0-9]+\.[0-9]+/) released++
  }
  added_at[NR] && $0 !~ /^[[:space:]]*$/ && $0 !~ /^#/ {
    if (tolower(section) ~ /^## \[unreleased\]/) unreleased++
    else { stray++; strays = strays sprintf("  line %d, under \"%s\": %s\n", NR, (section == "" ? "(no section)" : section), $0) }
  }
  END {
    if (on_default && released > 0) { print ok ": this push adds a version section (a release)."; exit 0 }
    # Any line added under a released version fails, even next to a proper [Unreleased] entry.
    if (stray > 0) {
      print fail ": " who " adds line(s) outside \"## [Unreleased]\" in CHANGELOG.md" (on_default ? " (compared with " substr(base, 1, 8) ")" : "") ":"
      printf "%s", strays
      if (on_default) print "A release probably landed on main before this change - move the entry to [Unreleased] in a new merge request."
      else print "A release probably landed on main after the entry was written - move the entry to [Unreleased]."
    }
    else if (unreleased > 0) { printf "%s: %s adds %d line(s) under [Unreleased].\n", ok, who, unreleased; exit 0 }
    # After a release the section is gone; a branch that creates it again is fine, even before any entry is under it.
    else if (created > 0) { print ok ": " who " creates the [Unreleased] section, with no lines under it yet."; exit 0 }
    else if (on_default) { print ok ": " who " adds nothing to CHANGELOG.md outside [Unreleased]."; exit 0 }
    else print fail ": this branch adds no entry under \"## [Unreleased]\" in CHANGELOG.md."
    if (on_default) print "Intentional? Put [skip changelog-check] in the title of the merged merge request, then retry this job."
    else print "Intentional? Put [skip changelog-check] in the merge request title, then retry this job."
    exit 1
  }' CHANGELOG.md
