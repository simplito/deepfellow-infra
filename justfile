dev:
    uv run uvicorn server.main:app --host "localhost" --port 8086 --reload --log-level debug --log-config server/utils/logging_config.yaml --timeout-graceful-shutdown 5 --ws-ping-interval 5 --ws-ping-timeout 10

dev-trace:
    uv run uvicorn server.main:app --reload --log-level trace --log-config server/utils/logging_config.yaml

test *FLAGS:
    uv run pytest --showlocals --tb=auto -ra --cov server --cov scripts --cov-branch --cov-report=term-missing --no-cov-on-fail {{FLAGS}}

ntest *FLAGS:
    uv run pytest --showlocals --tb=auto -ra --cov server --cov-branch --cov-report=term-missing --no-cov-on-fail -n auto {{FLAGS}}

ruff *FLAGS:
    uv run ruff check server/ scripts/ tests/ deploy/helm/deepfellow/files/ deploy/helm/deepfellow/ci/ {{FLAGS}}

ruff-format *FLAGS:
    uv run ruff format server/ scripts/ tests/ deploy/helm/deepfellow/files/ deploy/helm/deepfellow/ci/ {{FLAGS}}

auth-static:
    uv run python -m server.scripts.check_auth static ./server/ -v

auth-runtime:
    uv run python -m server.scripts.check_auth runtime ./server/main.py -v

pyright:
    uv run pyright

# Every check CI runs on the Kubernetes chart (needs helm + helm-unittest, kubeconform, helm-docs).
helm-test:
    deploy/helm/deepfellow/ci/test.sh

mypy *FLAGS:
    uv run mypy server/ tests/ {{FLAGS}}

pre-commit:
    uv run pre-commit run --all-files

todo:
    git grep "# TODO" -- "*.py"
    git grep "# FIX" -- "*.py"
    git grep "# DONE" -- "*.py"

license-check *FLAGS:
    uv run scripts/check_license_header.py {{FLAGS}}

check: license-check ruff ruff-format pyright auth-static auth-runtime

ui-rebuild:
   (cd $(git rev-parse --show-toplevel)/webui && npm run buildx)

# See webui2/README.md.
# Build the DeepFellow Dashboard into static/, the same way ui-rebuild does for the legacy UI.
ui2-rebuild:
   (cd $(git rev-parse --show-toplevel)/webui2 && npm run buildx)

# `buildx` copies without deleting, so a name that the two UIs do not share stays in static/ for ever.
# The committed files in static/ are left alone.
# This is the whole undo when static/ held no UI before, which is the state of a fresh clone.
#
# The list of names comes from the INSTALLED package, and is never written here. The package states what it emits in its
# own `deepfellowDashboard.emittedFiles` field, derived from the files it carries. A list written here went stale the day
# the Dashboard renamed a file, and nothing reported it.
# Remove every file a Dashboard build wrote into static/.
ui2-clean:
   #!/bin/sh
   set -eu
   root=$(git rev-parse --show-toplevel)
   names=$(just _ui2-emitted-names)
   # The field separator is a NEWLINE alone. With the default, a declared name that holds a space splits into two, and
   # each half is a valid plain name that the guard below cannot refuse. `rm -rf` would then take a name nobody declared.
   # The value is built with printf, because a literal newline inside a just recipe ends the recipe.
   IFS="$(printf '\nx')"; IFS="${IFS%x}"
   # No pathname expansion either. A declared name holding `*` would otherwise expand against this directory, and each
   # match would become a name nobody declared.
   set -f
   # Every name is judged BEFORE anything is removed. A name refused half way through would leave static/ half cleaned.
   for name in $names; do
     case "$name" in ""|*/*|.|..) echo "Refusing the file name \"$name\" from the package manifest." >&2; exit 1;; esac
   done
   for name in $names; do
     rm -rf "$root/static/$name"
   done

# Verify that .gitignore covers every name the installed Dashboard build emits.
# That list lives in two places: this repository's .gitignore, which git reads and cannot compute, and the package's own
# manifest. This recipe is what holds them together, and the CI image job runs it.
ui2-check-ignores:
   #!/bin/sh
   set -eu
   root=$(git rev-parse --show-toplevel)
   names=$(just _ui2-emitted-names)
   # A newline is the only separator, for the reason ui2-clean gives.
   IFS="$(printf '\nx')"; IFS="${IFS%x}"
   # No pathname expansion either. A declared name holding `*` would otherwise expand against this directory, and each
   # match would become a name nobody declared.
   set -f
   missing=0
   for name in $names; do
     # `--no-index` is what makes this ask the RULES. Without it git reports a tracked path as not ignored, whatever
     # .gitignore says, and the check could never fail.
     if ! git -C "$root" check-ignore -q --no-index "static/$name"; then
       echo "static/$name is emitted by the Dashboard build and .gitignore does not cover it." >&2
       missing=1
     fi
   done
   # .dockerignore keeps its own copy of the same names, because this repository re-includes all of static/ and then
   # excludes each UI output by name. Without an entry there, a stray dashboard build in static/ reaches the build
   # context of a LEGACY image. Git cannot answer for that file, so it is read directly.
   # index.html, favicon.ico and assets are excluded by the legacy block above it, so only the Dashboard-only names
   # are asked for here.
   for name in $names; do
     case "$name" in index.html|favicon.ico|assets) continue;; esac
     if ! grep -qxF "static/$name" "$root/.dockerignore"; then
       echo "static/$name is emitted by the Dashboard build and .dockerignore does not list it." >&2
       missing=1
     fi
   done
   if [ "$missing" -ne 0 ]; then
     echo "Add the names above. Without them a UI build's output can be committed, or reach a legacy image's build context." >&2
     exit 1
   fi
   echo "ui2-check-ignores: OK; .gitignore and .dockerignore both cover every name the Dashboard build emits."

# The file names the installed Dashboard package says it emits, one per line.
# Both recipes above read it through here, so neither one repeats the path or the failure handling.
# An assignment from a command substitution IS caught by `set -e`; the same substitution inside a `for` word list is
# NOT, which is how an earlier version reported OK after node threw.
_ui2-emitted-names:
   #!/bin/sh
   set -eu
   root=$(git rev-parse --show-toplevel)
   manifest="$root/webui2/node_modules/@simplito/deepfellow-dashboard-single-instance-dfinfra/package.json"
   if [ ! -f "$manifest" ]; then
     echo "Run 'npm ci' in webui2 first: this reads the file list from the installed package." >&2
     exit 1
   fi
   names=$(node -p "const f=require('$manifest').deepfellowDashboard?.emittedFiles; if(!Array.isArray(f)||f.length===0){throw new Error('This package declares no deepfellowDashboard.emittedFiles. It is older than the field; raise the pinned version.')} f.join('\n')")
   if [ -z "$names" ]; then
     echo "The package declared an empty file list, which cannot be right." >&2
     exit 1
   fi
   printf '%s\n' "$names"

# The legacy build runs FIRST, and it writes to webui/dist. A build that fails therefore leaves static/ as it was.
# The delete runs between that build and the copy, because both UIs write an `assets` directory.
# A delete after the copy would remove the legacy UI's own assets.
# Put the legacy UI back after ui2-rebuild.
ui-restore:
   (cd $(git rev-parse --show-toplevel)/webui && npm run build)
   just ui2-clean
   (cd $(git rev-parse --show-toplevel)/webui && cp -r dist/* ../static/)

# Run the Dashboard against this service, without building anything into static/.
dashboard *FLAGS:
   (cd $(git rev-parse --show-toplevel)/webui2 && npm run dashboard -- {{FLAGS}})

test-webui:
   (cd $(git rev-parse --show-toplevel)/webui && npm test)

ts-check *FLAGS:
   (cd $(git rev-parse --show-toplevel)/webui && npx biome check src {{FLAGS}})

ts-lint *FLAGS:
   (cd $(git rev-parse --show-toplevel)/webui && npx biome lint src {{FLAGS}})

ts-format *FLAGS:
   (cd $(git rev-parse --show-toplevel)/webui && npx biome format src {{FLAGS}})

ts-fix *FLAGS:
   (cd $(git rev-parse --show-toplevel)/webui && npx biome check --write src {{FLAGS}})

env-copy:
    uv run python -m scripts.copy_envs

replace-docker *FLAGS:
    docker stop infra-infra-1; just dev {{FLAGS}}

inspector:
    npx @modelcontextprotocol/inspector node build/index.js

get-ollama-models *FLAGS:
    uv run python -m scripts.get_ollama_models {{FLAGS}}

clear-ollama-cache:
    uv run python -m scripts.get_ollama_models --clear-cache

get-vllm-models *FLAGS:
    uv run python -m scripts.get_huggingface_models --type llm --top-by-downloads 200 --top-by-likes 100 --top-by-trending 50 --output static/vllm-min.json {{FLAGS}}
    uv run python -m scripts.get_huggingface_models --type reranker --top-by-downloads 10 --top-by-trending 10 --output static/vllm-min.json {{FLAGS}}
    uv run python -m scripts.get_huggingface_models --type embedding --top-by-downloads 10 --top-by-trending 10 --output static/vllm-min.json {{FLAGS}}

get-llamacpp-models *FLAGS:
    uv run python -m scripts.get_llamacpp_models --top-by-downloads 200 --top-by-likes 100 --top-by-trending 50 --output static/llamacpp-min.json {{FLAGS}}

get-sglang-models *FLAGS:
    uv run python -m scripts.get_huggingface_models --type llm --top-by-downloads 200 --top-by-likes 100 --top-by-trending 50 --output static/sglang-min.json {{FLAGS}}
    uv run python -m scripts.get_huggingface_models --type reranker --top-by-downloads 50 --top-by-likes 30 --top-by-trending 30 --allow-generative-rerankers --output static/sglang-min.json {{FLAGS}}
    uv run python -m scripts.get_huggingface_models --type embedding --top-by-downloads 10 --top-by-trending 10 --output static/sglang-min.json {{FLAGS}}

# static/ holds nine *-min.json registries; these four are the generated ones. coqui, docker-model-runner,
# rerank, speaches and stable-diffusion are maintained by hand and are deliberately not touched here — a new
# service joins this list only once it has its own get-*-models recipe.
#
# An aggregate in the style of `check`: it takes no flags, so call an individual get-*-models recipe when you
# need to pass any.

# Refresh every model list generated from an upstream source, in one go.
refresh-generated-model-lists: get-ollama-models get-vllm-models get-llamacpp-models get-sglang-models

