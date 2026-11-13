# E2e service-lifecycle suite

End-to-end tests against a **running** DeepFellow Infra instance. Each scenario takes one service and one model through
the full lifecycle:

1. install the service;
2. install the model;
3. make one inference request through the public `/v1/*` API;
4. uninstall the model and the service.

This catches what the other suites cannot: broken service installs, failed model downloads, and inference that no longer
reaches the model.

> **Destructive.** The suite installs and uninstalls services on its target. Only point it at a throwaway instance,
> never at a shared environment such as dev. As a safeguard it refuses to start when the target already has services
> installed (see `DF_E2E_ALLOW_DIRTY` below).

It complements:

- `tests/unit/`: in-process tests.
- `tests/smoke/`: read-only checks, safe against any environment.

## How it works

- **Black box.** Tests talk to the target over HTTP only and import nothing from `server/`.
- **Scenarios are data.** `scenarios.py` lists one entry per (service, model, modality). `test_lifecycle.py` runs each
  one. Inference checks in `inference.py` validate the response shape, not its content:
  - chat: a non-empty reply on every chat API the model advertises in `/v1/models` (`/v1/chat/completions`,
    `/v1/completions`, `/v1/responses`, `/v1/messages`, `/api/chat`), each plain and streamed in its own wire format;
  - embeddings: one numeric vector per text of a two-text batch, different for different texts, of the model's
    dimension when the scenario states it;
  - rerank: one score per document, the relevant document above the unrelated one;
  - image: PNG, JPEG or WebP data;
  - speech: WAV, MP3, Ogg or FLAC data;
  - transcription: a word of what `fixtures/speech-sample.wav` says;
  - decision: one answer per question from `/v1/systemone`;
  - document: doc-chunker through Infra's `/custom/<prefix>/` proxy, the way Server calls it - `fixtures/document-sample.pdf`
    converted with chunks, then both fixtures as background jobs (submit, poll, fetch, delete), after which Infra must
    still answer.
- **Every scenario also checks the registry:** the model is in `/v1/models` while installed and gone after uninstall.
  A `gpu` scenario must raise used VRAM, so a backend that silently falls back to the CPU fails.
- **Teardown always runs.** Uninstalling happens even when an install or the inference fails. A failing teardown is
  attached to the test's own error and does not replace it. At the end of the session a sweep removes anything a
  scenario left behind.
- **Upgrade.** `test_upgrade.py` installs a named set of scenarios together, then runs `DF_E2E_UPGRADE_COMMAND` to
  replace the Infra container on the same storage. It checks that every service and model is still installed and that
  inference still works, then tears down on the upgraded instance. Run with the same image, it checks a restart. The set
  is installed and upgraded once, but each scenario reports on its own, before and after the upgrade
  (`test_upgrade[full-before-vllm-chat-gpu]`, `test_upgrade[full-after-vllm-chat-gpu]`), so one broken service does not
  hide the others.
- **Failures say where.** A failing scenario is noted with its phase (which install, or inference) and, with
  `DF_E2E_LOG_DIR` set, the logs of its service containers are saved before teardown removes them. An install is
  retried once after a connection or gateway error, so a registry outage is not reported as an Infra bug; inference
  is never retried.
- **Images are kept.** Uninstalls do not purge by default, so a host keeps its service images between runs.

## Configuration

Everything is read from `DF_E2E_*` environment variables. pytest also loads `.test.env` and lets it override the shell, so keep `DF_E2E_*` out of that file.

| Variable | Default | Meaning |
|---|---|---|
| `DF_E2E_BASE_URL` | — | Target instance. **The whole suite is skipped while it is unset**, so `pytest tests/` is unaffected. |
| `DF_E2E_ADMIN_API_KEY` | — | Target's `DF_INFRA_ADMIN_API_KEY` (required when the base URL is set). |
| `DF_E2E_API_KEY` | — | Target's server key `DF_INFRA_API_KEY`, used for `/v1/*` and `/info` (required). |
| `DF_E2E_OPENAI_API_KEY` | — | Enables the `cloud` scenarios; they are skipped without it. |
| `DF_E2E_EXPECTED_VERSION` | — | Abort unless `GET /info` reports this version (a leading `v` is ignored). |
| `DF_E2E_ALLOW_DIRTY` | `false` | Run even though the target already has services installed. Scenarios on a service that was installed before the session are skipped, so the suite never uninstalls it. |
| `DF_E2E_REQUIRE_GPU` | `false` | Fail `gpu` scenarios instead of skipping them when the target offers no GPU. CI nightly sets it, so a broken driver on the runner cannot pass as green. |
| `DF_E2E_PURGE` | `false` | Purge on uninstall. This also removes the service images. |
| `DF_E2E_UPGRADE_COMMAND` | — | Command that replaces the target's Infra container, e.g. `tests/e2e/instance.sh upgrade <image> <id>`. The upgrade test is skipped without it. |
| `DF_E2E_UPGRADE_VERSION` | — | Version the target must report after the upgrade command (a leading `v` is ignored). |
| `DF_E2E_REQUIRE_UPGRADE` | `false` | Fail the upgrade test instead of skipping it when it cannot run. CI upgrade jobs set it. |
| `DF_E2E_LOG_DIR` | — | Where to save the service containers' logs when a scenario fails. CI sets it to the artifacts directory. |
| `DF_E2E_INSTALL_TIMEOUT` | `1800` | Seconds allowed for one service or model install. |
| `DF_E2E_INFERENCE_TIMEOUT` | `300` | Seconds allowed for one inference request. |
| `DF_E2E_READY_TIMEOUT` | `120` | Seconds to wait for `/health` before giving up. |

## Markers

| Marker | Scenarios |
|---|---|
| `gate` | A small, fast CPU subset with no external credentials: the default of the on-demand CI jobs and a quick local check. |
| `cpu` | Runs on CPU. |
| `gpu` | Needs a GPU on the target. Installed with the GPU option the target offers (`GPU`, else `GPUs`, else one named GPU); skipped when it offers none. |
| `cloud` | Calls a cloud provider (OpenAI only for now). Skipped without its API key. |
| `upgrade` | The upgrade test: per set in `UPGRADE_SETS` (`gate`, `full`), one test per scenario before and after the upgrade; pick a set with `-k`, and run one set per session. |

## Running locally

`instance.sh` starts a throwaway instance from an Infra image on your local Docker daemon - the one of the current
`docker context`, whose socket it hands to Infra, so rootless Docker works too (untested so far; the runner uses rootful
Docker). `stop` removes everything that instance created except pulled images:

```bash
tests/e2e/instance.sh start gitlab2.simplito.com:5050/df/deepfellow-infra:dev mine > /tmp/e2e.env
set -a; . /tmp/e2e.env; set +a

just test-e2e -m gate                                              # the release gate
just test-e2e -k rerank-cpu                                        # one scenario
uv run pytest tests/e2e -v --no-cov -m "cpu and not gate"          # marker expressions need quoting, so call pytest directly

tests/e2e/instance.sh stop mine
```

Upgrade test, from a release to `:dev` (with the same image twice it tests a restart):

```bash
tests/e2e/instance.sh start gitlab2.simplito.com:5050/df/deepfellow-infra:v0.35.0 mine > /tmp/e2e.env
set -a; . /tmp/e2e.env; set +a
export DF_E2E_UPGRADE_COMMAND="tests/e2e/instance.sh upgrade gitlab2.simplito.com:5050/df/deepfellow-infra:dev mine"

just test-e2e -m upgrade -k gate

tests/e2e/instance.sh stop mine
```

`instance.sh upgrade` stops the Infra container gracefully (Infra stops its services, as in production), saves its log
to `$E2E_LOG_DIR/df-e2e-<id>-before-upgrade.log`, and starts the new image with the same storage, keys and port. It
fails if any service container is still running after the old Infra stopped.

`E2E_STORAGE_ROOT` (default `$TMPDIR` or `/tmp`) is where the instance keeps its storage. `stop` saves the Infra
container's log to `$E2E_LOG_DIR/df-e2e-<id>.log` (default: the storage root) before removing the container.

You can also run the suite against a dev server (`just dev`), as long as that server's storage has no services
installed. Point the keys at the server's `DF_INFRA_ADMIN_API_KEY` and `DF_INFRA_API_KEY`.

## What gets a scenario

The suite covers paths through Infra that only run against real Docker, real downloads and the real proxy - not
models. Add one scenario, with the smallest model of its kind, when a change brings:

- a new service, or a new image for one (image refreshes are the riskiest changes here);
- a new modality or public endpoint;
- a new install or download path (Hugging Face repo vs. single file, civitai, the Ollama registry, gguf);
- state that must survive a restart or upgrade - add it to the `full` upgrade set too;
- a new hardware variant.

Another model through an existing path, logic a unit test can reach, and output quality do not get one.

Where it goes: `gate` only for what breaks most users, is CPU-only, needs no credentials and keeps the gate within
a few minutes; everything else runs nightly.

## Known Infra bugs

`BROKEN_SCENARIOS` and `BROKEN_CHAT_APIS` in `scenarios.py` list bugs the suite has found and Infra still has, each
with what is wrong. A broken scenario is a strict expected failure and is left out of the upgrade sets; a broken chat
API must answer with an error. Both fail once the bug is fixed, so the entry is removed then - the list cannot go
stale. A gate scenario can never be on it.

## Adding a scenario

Append a `Scenario` to `SCENARIOS` in `scenarios.py`:

- `id`: unique; this is the pytest id, used with `-k`.
- `service_id`, `service_spec`: the body of the service install. Use `CPU` or `GPU` for the hardware field.
- `model_id`, `model_spec`: the model to install. Pick the smallest stable catalog entry of its kind.
- `modality`: selects the inference check.
- `marks`: always include one of `cpu`, `gpu` or `cloud`. Only add `gate` to CPU scenarios that need no credentials.
- `service_secrets`: spec fields filled from a secret setting at install time, e.g. `{"api_key": "openai_api_key"}`.
- `api_model`: set this only when the model is served under a different name than `model_id`.

To have the scenario survive an upgrade too, add its id to the `full` set in `UPGRADE_SETS` - once the release the
nightly upgrade starts from supports it. A set installs each service
once, so all its scenarios on one service must share the service spec.

Pin a model that is in the catalog Infra ships (`static/*-min.json`): `test_scenarios.py` fails when a catalog refresh
drops it. A new service needs a scenario or an entry with a reason in `NOT_COVERED`; `test_coverage.py` checks that
against the services the target offers, in the nightly run.

`test_scenarios.py` enforces the matrix rules (unique ids, gate is CPU-only and needs no credentials) in every
`pytest` run.

## CI

`.gitlab-ci.yml` defines six jobs in the `e2e` stage. The stage runs after `build` and before `deploy`.

- **`e2e-release`** and **`e2e-upgrade-release`**: the release gate. Every push to a release branch
  (`release/X.Y.Z`) builds the branch's image on the runner and runs the full matrix on it, then installs the `full`
  set on the latest release and upgrades it to that image. Both check that the image reports version `X.Y.Z`. The
  branch only bumps the version on top of `main` and the tag goes on its merge commit, so this is the code that gets
  tagged, as long as nothing is merged to `main` meanwhile (`RELEASE.md`, step 7). With "Pipelines must succeed", a
  failure blocks the release MR before any tag exists. Together they take about an hour.
- **`e2e-nightly`** and **`e2e-upgrade-nightly`**: the same two runs against the image of `main`'s HEAD, as feedback
  for `main`; they block nothing. They are triggered by a pipeline schedule on `main` that sets `DF_E2E_NIGHTLY=true`.
  The same variable keeps that schedule from rebuilding the image, deploying to dev and running smoke tests. A night
  with no commit on `main` in the last 25 hours ends both jobs green at once (`E2E_SKIP_IF_IDLE`).
- **`e2e-branch`** and **`e2e-upgrade-branch`**: on demand, for any other branch. They are play buttons in the
  branch's pipeline, or come with "Run pipeline" in the UI, and never start on their own (a branch pipeline stays green
  without them). They build the branch's image on the runner without pushing it, then run `-m gate`, or upgrade the
  latest release to it with the `gate` set. Set `E2E_PYTEST_ARGS` when starting the job to run something else, e.g.
  `-k vllm` or `-m "cpu and not gate"` - useful before merging an image or catalog refresh.

Every job that uses the runner's GPU is in the `e2e-gpu` resource group, so only one runs at a time. Each job starts
its own instance with `instance.sh start <image> $CI_JOB_ID` and stops it in `after_script`, so it cleans up even on
failure or cancellation. Each job publishes a JUnit report and the Infra log as artifacts.

The jobs are created only when the CI/CD variable `DF_E2E_ENABLED` is `true`. Until the runner below exists, they
would sit pending forever and stall every release.

### Running the on-demand jobs

1. Open the branch's latest pipeline, e.g. from the MR's **Pipelines** tab.
2. In the `e2e` stage, click the job's **name** (not ▶): `e2e-branch` or `e2e-upgrade-branch`.
3. Add the variable `E2E_PYTEST_ARGS` and click **Run job**. Without it, both run the `gate` subset.

| To run | `e2e-branch` | `e2e-upgrade-branch` |
|---|---|---|
| everything | `-m "not upgrade"` | `-m upgrade -k full` |
| one service | `-k vllm` | - |
| the cloud scenarios | `-m cloud` | - |

The jobs exist only in push pipelines and in those started from **Build → Pipelines → Run pipeline**. The **Run
pipeline** button on the MR's Pipelines tab starts a merge request pipeline, which has none of them; so do
`glab ci run` and other API-triggered pipelines.

### Runner requirements

All e2e jobs, CPU and GPU, run on one dedicated self-hosted runner. It must have:

- the GitLab runner with the **shell executor**, registered with the tag **`deepfellow-e2e-gpu`**. A shell executor uses
  the host's Docker daemon directly, so pulled service images persist between jobs; docker-in-docker would start every
  job with an empty cache;
- **Docker**, with the runner user in the `docker` group, and the `docker` CLI;
- at least **24 GB RAM**, with nothing else using much of it: Stable Diffusion reads its 7 GB model through host
  memory, and the full upgrade set runs every service at once. That set used about 13 GB on a 24 GB host; with only
  6 GB of RAM free, Stable Diffusion was killed while loading;
- an **NVIDIA GPU with at least 16 GB VRAM** (tested on an RTX 5070 Ti 16 GB). vLLM and SGLang models get a fixed
  share of the card (`gpu_memory_utilization` in `scenarios.py`), so on a smaller card the same share may not fit the
  model; the full upgrade set peaked at about 8.3 GB;
- drivers and **`nvidia-container-toolkit`** for that GPU, so that
  `docker run --rm --gpus all gcr.io/distroless/base nvidia-smi` works. Infra detects GPUs this way;
- **`uv`** on `PATH`, to run pytest from the checkout;
- `/var/tmp/df-e2e` writable by the runner user, with room for the models of the full set (tens of GB). Instance
  storage lives there rather than in the checkout, because service containers write it as root;
- `concurrent = 3` for this runner. The GPU jobs queue in the `e2e-gpu` resource group, so the extra slots only
  serve other jobs that may use the runner;
- registry credentials for private service images (e.g. `hub.simplito.com`) in the runner user's
  `~/.docker/config.json`. `instance.sh` mounts that file into the Infra container.

CI/CD variables:

- `DF_E2E_ENABLED=true`: set once the runner is registered.
- `DF_E2E_OPENAI_API_KEY` (masked): enables the cloud scenarios.

Pipeline schedule: on `main`, with the variable `DF_E2E_NIGHTLY=true`.

Maintenance: a job killed hard enough to skip `after_script` can leave containers behind. Their names end with
`_e2e<job id>` and the Infra container is `df-e2e-<job id>`. Periodically remove them and prune old images, for example:

```bash
docker ps -a --format '{{.Names}}' | grep -E '(^df-e2e-|_e2e[0-9]+$)' | xargs -r docker rm -f
docker network ls --format '{{.Name}}' | grep '^df-e2e-' | xargs -r docker network rm
docker image prune -a --filter "until=720h" -f
```

Only run this cleanup while no e2e job is running.
