#!/bin/sh
# DFMAIN-78: a branch must add its entry under `## [Unreleased]` in CHANGELOG.md. Catches the case where a release
# lands on main while the branch is open: the rebase merges cleanly, but the entry ends up under the released version.
# DFINFRA-454: GitLab runs no pipeline for that merge onto the newer main, so the check also runs on main itself. There it
# compares CHANGELOG.md with the previous tip of main and fails on a line added outside `## [Unreleased]`.
# DFMAIN-112: on a branch, `## [Unreleased]` of the branch merged with the latest main must not repeat a `###` heading or use
# one that is not a Keep a Changelog type. Two branches can each add `### Fixed` and git merges them without a conflict.
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
existing=0
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
  }' CHANGELOG.md || existing=1

# Prints the repeated and non-standard `###` headings of `## [Unreleased]` in the CHANGELOG.md given as $1, called $2 in the messages.
# Exit status: 0 when clean, otherwise 1 for a repeated heading plus 2 for a heading that is not a Keep a Changelog type.
check_headings() {
    awk -v ok="$OK" -v fail="$FAIL" -v into="$2" '
      /^## / { section++; unreleased = (tolower($0) ~ /^## \[unreleased\]/); if (unreleased) found = 1; next }
      unreleased && /^### / {
        heading = $0
        sub(/[ \t]+$/, "", heading)
        key = section SUBSEP tolower(heading)
        if (++count[key] == 1) { order[++n] = key; name[key] = heading; lines[key] = NR } else lines[key] = lines[key] ", " NR
        if (heading !~ /^### (Added|Changed|Deprecated|Removed|Fixed|Security)$/) odd = odd sprintf("  \"%s\" at line %d\n", heading, NR)
      }
      END {
        if (!found) { print ok ": CHANGELOG.md merged with " into " has no \"## [Unreleased]\" section, nothing to check."; exit 0 }
        for (i = 1; i <= n; i++) if (count[order[i]] > 1) repeated = repeated sprintf("  \"%s\" at lines %s\n", name[order[i]], lines[order[i]])
        if (repeated == "" && odd == "") { print ok ": \"## [Unreleased]\" of CHANGELOG.md merged with " into " has no repeated or non-standard ### heading."; exit 0 }
        if (repeated != "") {
          print fail ": \"## [Unreleased]\" of CHANGELOG.md merged with " into " repeats a ### heading (line numbers are in the merged file):"
          printf "%s", repeated
          code += 1
        }
        if (odd != "") {
          print fail ": \"## [Unreleased]\" of CHANGELOG.md merged with " into " has a ### heading that is not a Keep a Changelog type:"
          printf "%s", odd
          code += 2
        }
        exit code
      }' "$1"
}

# Not on the default branch: this gates the merge, and a duplicate that already reached it fails every branch until someone
# merges the headings, which is the point. A branch that does not merge cleanly is skipped, GitLab will not merge it anyway.
headings=0
if [ "$on_default" = 0 ]; then
    # Locally the branch may hold uncommitted edits (always with jj); `git stash create` captures them without touching the checkout.
    tip=HEAD
    if [ -z "${CI:-}" ]; then
        stash=$(git stash create 2>/dev/null) || stash=
        if [ -n "$stash" ]; then tip=$stash; fi
    fi

    merge=0
    tree=$(git merge-tree --write-tree --no-messages "origin/$DEFAULT_BRANCH" "$tip" 2>"$tmp/merge-tree.err") || merge=$?
    if [ "$merge" = 1 ]; then
        echo "$NOTE this branch does not merge cleanly with origin/$DEFAULT_BRANCH, so repeated headings are not checked. Rebase it first."
    elif [ "$merge" != 0 ]; then
        echo "$FAIL: could not merge this branch with origin/$DEFAULT_BRANCH to look for repeated headings ($(git --version); --write-tree needs 2.38):"
        sed 's/^/  /' "$tmp/merge-tree.err"
        headings=1
    elif ! git show "$tree:CHANGELOG.md" > "$tmp/merged-changelog.md" 2>/dev/null; then
        echo "$FAIL: CHANGELOG.md is missing from this branch merged with origin/$DEFAULT_BRANCH."
        headings=1
    else
        rc=0
        check_headings "$tmp/merged-changelog.md" "origin/$DEFAULT_BRANCH" || rc=$?
        if [ "$rc" != 0 ]; then
            headings=1
            # When the default branch has the problem itself, nothing in this branch fixes it: say so instead of asking for a rebase.
            main_bad=0
            if git show "origin/$DEFAULT_BRANCH:CHANGELOG.md" > "$tmp/main-changelog.md" 2>/dev/null; then
                check_headings "$tmp/main-changelog.md" "origin/$DEFAULT_BRANCH" > /dev/null || main_bad=1
            fi
            if [ "$main_bad" = 1 ]; then
                echo "$NOTE origin/$DEFAULT_BRANCH already has a repeated or non-standard ### heading, so every branch fails until it is fixed there. Rebasing this branch will not help."
            else
                if [ $((rc & 1)) != 0 ]; then echo "Rebase the branch onto origin/$DEFAULT_BRANCH and merge the repeated blocks into one."; fi
                if [ $((rc & 2)) != 0 ]; then echo "Use one of: Added, Changed, Deprecated, Removed, Fixed, Security."; fi
            fi
            echo "Intentional? Put [skip changelog-check] in the merge request title, then retry this job."
        fi
    fi
fi

if [ "$existing" != 0 ] || [ "$headings" != 0 ]; then exit 1; fi
