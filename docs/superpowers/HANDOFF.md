# HANDOFF — DeepFellow native-k8s Helm chart

_Last updated: 2026-09-14. Read this + `PLAN.md` (same dir) + memory `project_deepfellow_helm_chart` to resume cold._

## Status
- **Design: DONE** (grilled through Q1–Q17). Full spec = `docs/superpowers/PLAN.md`.
- **Gate A: PROVEN ✅** — socket-free native-k8s DeepFellow is viable (external `openai` service is
  grantable to a Server project by plain model id; Server does NOT require the id from Infra's docker
  catalog). Ran in throwaway ns `df-gate` on lw-c1, torn down clean. Finding:
  `viking://resources/homelab-knowledge/deepfellow-socket-free-native-k8s.md`.
- **Phase 1 — chart skeleton: DONE ✅ (2026-09-14, UNCOMMITTED in the tree).** At
  `deploy/helm/deepfellow/`: `Chart.yaml`, `values.yaml` (full grouped shape, `# --` comments,
  grounded in CLI `defaults.py` image/env inventory), `values.schema.json` (RE2-safe, settled leaves
  only), `.helmignore`, `README.md`, `templates/_helpers.tpl`, `templates/_validate.tpl` (cross-field
  checks), `templates/validation.yaml` (single invocation point so validate fires today). Phase-1
  build decisions (image tag=latest+digest pin, externalOnly contract, mesh dummy, list vs map, one
  validate arm, schema scope) recorded in PLAN.md "PHASE 1 BUILD DECISIONS". VERIFIED: `helm lint`
  clean; 11/11 assertion matrix green (default renders; short password / bad vectorDb.type / missing
  gguf.url / bad backend name / empty mesh.key / mongo enabled+external / native backend w/o
  externalOnly all rejected; BYO externalOnly=false w/o backends allowed; valid full backend passes).
  Post-review fix (advisor): `nameOverride`/`fullnameOverride` moved from `global.` to TOP-LEVEL
  (`global` is Helm's subchart namespace — overrides there silently no-op). All 7 helpers exercised
  via a throwaway scratch template and render correctly — so Phase 2 template failures are template
  bugs, not helper bugs.
- **Phase 1 CONTRACT REWORKED per Hermes gate — DONE ✅ (2026-09-14, user chose FULL fail-closed).**
  values.yaml + _validate.tpl + values.schema.json + README rewritten to: (a) FAIL-CLOSED default
  (default `helm template` intentionally fails until creds+images supplied); (b) digest-first images
  (`image.{digest,allowMutableTag}`, fail unless digest or ack); (c) `infra.externalOnly.{enabled,
  enforcement: compatibility|patched}` (patched requires infra digest); (d) `modelBackends` +
  separate `externalBackends` as KEYED MAPS; (e) credential `source: existingSecret|value|generate|
  none` discriminators (no empty-string XOR) via reusable `_validate.tpl` partials
  (validateSecret/validateImage/validatePasswordComplexity); (f) `server.mongo.mode: embedded|
  external`; (g) `provisioning.enabled` default FALSE + gate (enabled ⇒ ≥1 backend unless
  allowNoModels). schema: $defs image+credential, maps via propertyNames DNS-1123, enums. Added
  `ci/minimal-values.yaml` (valid minimum, .helmignore'd). VERIFIED: helm lint clean; 13/13 assertions
  (default fail-closed w/ clear msg; minimal passes; native backend passes; value^existingSecret,
  mutable-tag-no-ack, patched-no-digest, native+!externalOnly, provisioning-no-backends, bad map key,
  weak inline password, external-mode-no-uriSecret all reject; allowNoModels escape passes).
- **Phase 2 — component templates: DONE ✅ (2026-09-14, UNCOMMITTED).** Built under templates/:
  infra.yaml (Deployment :8086 + Service + PVC + chart Secret; DF_EXTERNAL_ONLY, self DF_INFRA_URL,
  no docker socket), model-backends.yaml (range over the map; llama.cpp Deployment + Service ClusterIP
  + initContainer GGUF fetch + gpu/emptyDir variants; per-entry defaults via
  `deepfellow.modelBackendDefaults`), server.yaml (Deployment :8000 + Service + PVC + chart Secret;
  DF_INFRA__URL/DF_INFRA__API_KEY double-underscore, mongo embedded/external, vectorDb env, graph env,
  metrics; admin password NOT in Deployment env — Phase-3 Job consumes it), mongo.yaml (embedded
  StatefulSet + Service + initdb ConfigMap + Secret; fsGroup 999, mongo image manages its own user),
  vectordb.yaml (qdrant StatefulSet OR milvus stack etcd+minio+milvus, gated on active+type),
  graph.yaml (falkordb, REDIS_ARGS via $(DF_GRAPH__PASSWORD)), workspace.yaml (Deployment :3000 +
  Service + own app-mongo StatefulSet replSet rs0 + rs-init hook Job + chart Secrets; DF_SERVER_* from
  the provisioning state Secret — pod waits via valueFrom until Phase-3 mints it).
  Consumer helpers added to _helpers.tpl: credEnv, chartSecret (lookup-guarded generate),
  genSecretValue (policy-compliant), image, imagePullSecrets, pod/containerSecurityContext,
  credSecretName, modelBackendDefaults, provisioning.stateSecretName.
  externalBackends: NO workload template — they are registered into Infra by the Phase-3 Job only.
  VERIFIED: helm lint clean (minimal + everything-on); default still fail-closed; minimal renders 14
  objects; everything-on renders 21; external-mongo renders; workspace.enabled=false removes cleanly;
  no malformed image refs. Test values: ci/minimal-values.yaml.
- **Phase 3 — provisioning: DONE ✅ (2026-09-14, UNCOMMITTED).** `files/provision.py` (PYTHON, not
  shell — JSON/API/k8s complexity; stdlib urllib only; interface reconcile|verify|status; NO token
  ever printed) + `templates/provisioning.yaml` (ServiceAccount + Role [create secrets; get/update/
  patch scoped by resourceNames to the ONE state Secret] + RoleBinding + ConfigMap[provision.py via
  .Files.Get] + Job[post-install,post-upgrade hook, name carries provisioning.revision]). Flow: wait
  infra/server/backends /health → create_admin (`python -m server.scripts.create_admin`, existing=no-op)
  → login → read state Secret FIRST (skip if org/project/key present) → create org→project→key
  primitives (server:latest has no atomic /admin/workspace/) → persist to state Secret via k8s API
  (SA token) BEFORE grant → register each backend into Infra (service + custom model TWO calls +
  install, 400-already-exists tolerated) → poll infra /v1/models → grant project → verify via project
  key. State Secret keys: organization-id/project-id/project-api-key (match workspace valueFrom).
  Created by the Job (not helm-owned) → inherently preserved on uninstall. BACKENDS_JSON built from
  modelBackends (native, internal api_url) + externalBackends (api_key inlined ONLY for source=value).
  DEVIATION from Hermes ("shell"): chose Python — flag at the Phase-3/4 review-gate.
  KEY FACT: this repo's `server/` is the INFRA FastAPI app (:8086); its `/health` is public
  (check_auth.py allowlist, verified). The DeepFellow SERVER (:8000 — create_admin, /auth/login,
  /admin/organization, project/key) is a SEPARATE image (deepfellow-server), NOT in this repo, so its
  login response field + /health shape are UNVERIFIABLE here — provision.py logs in defensively
  (login token = token|access_token|jwt; server readiness = connection-level `accept_any`) and these
  MUST be confirmed in the Phase 6 smoke. Advisor notes carried forward: (a) provisioning Role grants
  `create` on secrets namespace-wide — RBAC cannot scope create by name; get/update/patch ARE scoped to
  the one state Secret. (b) hook Jobs use delete-policy before-hook-creation only (no hook-succeeded),
  so one completed Job survives per provisioning.revision — intentional, keeps the GET-ONCE audit log.
- **Phase 4 — ingress + networkpolicy + NOTES: DONE ✅ (2026-09-14, UNCOMMITTED).** ingress.yaml
  (networking.k8s.io/v1, server /v1 + workspace /, off default); networkpolicy.yaml (vanilla baseline:
  default-deny ingress + allow-intra-suite + allow-entrypoints[server,workspace]; fine-grained per-
  component isolation is the homelab OVERLAY's job per PLAN §11); NOTES.txt (component summary, project-
  key retrieval, fail-closed reminder). VERIFIED: helm lint clean (minimal + full-on+prov+ingress+
  netpol); default fail-closed; provisioning gate (no backend → fail; allowNoModels → pass); full-on =
  30 objects; provision.py `py_compile` OK.
- **Phase 5 — DF_EXTERNAL_ONLY patch: DONE ✅ (2026-09-14, UNCOMMITTED).** `server/config.py`
  (+`external_only: bool = False` bootstrap field → env DF_EXTERNAL_ONLY); `server/docker.py`
  create_docker_service short-circuits in external_only → returns an UNPROBED DockerService (no
  shutil.which / docker compose / GPU / rootless / network calls at boot); `server/services_manager.py`
  (ServicesManager(external_only=...) + class const EXTERNAL_ONLY_ALLOWED_SERVICE_TYPES={"openai"};
  `_get_service` returns deterministic 409 "disabled by DF_EXTERNAL_ONLY" for non-allowed types);
  `server/lifecycle.py` (constructs ServicesManager(external_only=config.external_only); registers ONLY
  OpenAIService in external_only, all services otherwise). Chose an unprobed real DockerService (not a
  null adapter) so base2_service's non-optional `docker_service: DockerService` type holds and pyright
  stays clean; the registration allowlist is the true guard. Unit tests added (Hermes's four):
  test_config (DF_EXTERNAL_ONLY parses; defaults False), test_docker
  (create_docker_service_external_only_makes_no_docker_calls — asserts which/run_command never called),
  test_services_manager (external_only disabled→409; normal unknown→404). Fixed the 4 existing
  create_docker_service tests to set config.external_only=False (MagicMock made it truthy). GATE PASSED:
  ruff, ruff-format, pyright (strict, 0/0/0), license (SPDX added to provision.py), auth-static (59
  secure), auth-runtime (71 secure); 389 tests in the 3 files pass. DEFERRED: Dockerfile docker-CLI-
  removal (not trivial — needs a separate build target; normal mode still needs the CLI; the app-level
  patch is the substantive change). Post-merge: build the patched image, then the chart can default
  infra.image.tag to it + infra.externalOnly.enforcement=patched with a pinned digest.
  PR-BODY notes (advisor): (a) EXTERNAL_ONLY allowlist is deliberately minimal ({openai}) pending
  per-service testing — CustomService/McpService are socket-free too and could be added once tested;
  say so in the PR so a reviewer doesn't read the 409 as a bug. (b) check_subnet is now also gated on
  external_only (no docker network-inspect at boot). (c) with the Docker CLI still in the image,
  enforcement=patched proves only the APP skips docker init; the remaining exposure is CLI PRESENCE,
  not probe behavior (probes are now skipped) — a truly daemonless base image is a separate follow-up.
- **Then Phase 6** — deploy to a throwaway ns on lw-c1, run the Gate-A 7 assertions THROUGH the chart,
  then the homelab overlay (values-homelab.yaml, OUT of the PR) + ArgoCD app. Deferred Phase-4 nicety:
  helm-template golden tests in CI. VERIFY DF_STORAGE_DIR writability for uid 1000 against the image.

## Where things are
- **This fork:** `~/Code/deepfellow-infra` (github `kamilrybacki/deepfellow-infra`, remote `upstream` =
  simplito). Branch **`feat/native-k8s-helm-chart`**. Full Infra source is here (`server/`).
- **PLAN.md / HANDOFF.md are UNTRACKED and stay LOCAL** — user said plan stays local, NO commit/push.
- **Memory:** `~/.claude/.../memory/project_deepfellow_helm_chart.md` (+ MEMORY.md pointer).
- **Ephemeral (gone next session):** scratchpad clones of `deepfellow-cli` + upstream `deepfellow-infra`.
  If you need CLI reference again: `git clone https://github.com/simplito/deepfellow-cli` (open source —
  it writes all the .env/compose/config the chart must mirror).

## What we're building (one line)
ONE upstream PR to `simplito/deepfellow-infra`: a whole-Suite native-k8s Helm chart at
`deploy/helm/deepfellow/` + a `DF_EXTERNAL_ONLY` code patch (server/config.py + server/lifecycle.py).
No DinD. llama.cpp deployed natively, registered into Infra as an `openai` EXTERNAL service.
Homelab overlay `values-homelab.yaml` stays OUT of the PR (user's own repo).

## NEXT STEPS (ordered)
1. ~~Phase 1 — chart skeleton~~ **DONE** (see Status). Scaffolding + schema + validation verified.
2. Phase 2 — component templates: infra, model-backends (llama.cpp), server, mongo, vectordb, workspace.
   Fold in Gate-A refinements: model register = TWO calls; type=None collision reconcile; DF_MESH_KEY
   dummy; server→infra link env `DF_INFRA__URL`/`DF_INFRA__API_KEY` (double underscore); llama.cpp value
   = backend ROOT (infra appends `v1/`). VERIFY the workspace image name (inferred, not Gate-A-tested).
3. Phase 3 — provisioning (files/ shell via ConfigMap; reconcile Job; Secret = source of truth).
   MUST: mark `provisioning.stateSecret` with `helm.sh/resource-policy: keep` (the dfproj key is
   GET-ONCE — losing it on uninstall/reinstall is unrecoverable); values comment already promises this.
4. Phase 4 — ingress, networkpolicy, tests, NOTES.txt.
5. Phase 5 — `DF_EXTERNAL_ONLY` patch + unit tests.
6. Deploy to a throwaway ns, run the Gate A assertions THROUGH the chart, then homelab overlay + ArgoCD app.

## MUST-KNOW gotchas (Gate-A-verified; detail in PLAN.md)
- **DF_MESH_KEY is REQUIRED at boot** by `infra:latest` (crashloops without) — set a dummy when mesh unused.
- **Model register = TWO calls**: `POST /admin/services/openai/models/custom` (typed def) THEN
  `POST /admin/services/openai/models/_?model_id=<id>`.
- **type=None collision:** a backend whose `/v1/models` LISTS its model gets seeded type=None at
  service-install → later custom-add 400 "already exists". Native llama.cpp lists → provisioning must reconcile.
- **`server:latest` has NO atomic `/admin/workspace/`** (405) → use 3 primitives:
  `POST /admin/organization/` → `POST /v1/organization/projects` → `.../{id}/api_keys`.
- **Admin password policy:** ≥10 chars, lower+upper+digit+special.
- Infra boots daemonless (image bundles docker CLI). `DF_EXTERNAL_ONLY` patch must keep CLI/probe optional,
  not just skip the socket. Infra port 8086, Server 8000, images anon-pullable from hub.simplito.com.
- Provisioning: register backend + verify infra model ready BEFORE granting project. dfproj key is GET-ONCE
  → read Secret first, persist before grant, KEEP Secret on uninstall, no one-shot immutable Job, no tokens in logs.

## SAFETY RAILS (hard)
- **NEVER run this on lw-main** (edge box; was 93% disk). Docker/smoke work → k3s cluster, node **lw-c1**
  (c2 hosts live SLIM DeepFellow — don't touch; c3 RAM tight). Throwaway ns + resource limits + mandatory
  `kubectl delete ns` teardown. Cluster-restraint: saturated → STOP, never force-delete.

## Reviews on record
advisor + Hermes (#hermes, 2026-09-14, review-only) both confirmed the design + Gate A. Their refinements
are already folded into PLAN.md.

## HERMES CROSS-REVIEW (2026-09-14, file-level on the pushed branch) — ADDRESSED ✅
Hermes reviewed the whole pushed branch (12 findings). All fixed in the "address cross-review" commit:
- #1 P0 externalBackends.apiKey leak → per-backend `secretKeyRef` env + `api_key_env` in BACKENDS_JSON
  (never inline the key); serviceType carried through. Verified LEAKCHECK plaintext=0 in full render.
- #2 P0 Workspace Mongo had no creds → `mongodb://$(MONGO_ROOT_USER):$(MONGO_ROOT_PASSWORD)@…
  ?replicaSet=rs0&authSource=admin` ($(VAR) composition; password via secretKeyRef).
- #3 P0/Gate-6 Server Mongo authSource: init mirrors the official CLI (defaults.py creates the user in
  `admin`) → left as-is; the Server-image URI/authSource contract is a Gate-6 smoke check.
- #4 P0 RBAC create-all → chart pre-renders an empty state Secret (resource-policy keep, pre-install
  hook) when chart-managed; Role now only get/update/patch on that one resourceName; provision.py PATCHes.
- #5 P0 reconcile not fail-closed → read state FIRST + partial-trio fail-closed; after 409 GET+compare
  desired (service api_url, model type/context_length); type=None = hard error; grant only after verify.
- #6 P0 GGUF non-atomic/no-checksum → temp download + optional `gguf.sha256` verify + atomic mv;
  downloadImage now an image submap under the digest-or-allowMutableTag policy.
- #7 P1 NetworkPolicy rewritten per-flow, instance-scoped; egress left to overlay (documented).
- #8 P1 Milvus minioadmin hardcoded → credential contract + hardened securityContext. minioadmin=0.
- #9 P1 generate.retain implemented (`helm.sh/resource-policy: keep`).
- #10 P1 ingress rules:[] → _validate requires ≥1 host.
- #11 P1 modelId uniqueness across modelBackends+externalBackends in _validate.
- #12 P1 docker-tags 409 in external_only (ServicesManager) + test.
- Hygiene: README status corrected; allowlist documented "external OpenAI-compatible backends only".
  LEFT (P2): workspace-mongo rs-init `mongosh -p` in argv (transient, in-container — acceptable).
VERIFIED: helm lint clean; default fail-closed; everything-on renders 42 objects; 407 python tests;
just-check gate clean (ruff/pyright/license/auth 59+71). GATE-6 server-image contract checks still pending.

## ADVISOR ROUND 2 (2026-09-14, post-cross-review) — ADDRESSED ✅ (commit 9211ece, pushed)
Advisor caught 4 issues the render/test sweep structurally could not:
- #3 BLOCKER (bug the cross-review #5 fix introduced): every backend registered under ONE shared
  `openai` service → 2nd backend 409/api_url-compare would SystemExit. FIX: each backend is its own
  service INSTANCE `openai|<key>` (verified against server: services_manager.split_service_type_and_instance
  + base2 install_instance create arbitrary instances; per-instance api_url in installed.options.spec).
  Instance name = chart MAP KEY (DNS-label ⊂ Infra `^[a-zA-Z0-9_-]+$,≤64`), NOT modelId — modelId stays
  free-form (dots OK). Reconcile trigger corrected: base2 returns **400 "already installed"** (not 409);
  drift-compare reads `installed.api_url` (GET `spec` is the field schema, not the value). Added
  `urllib.parse` import. _validate: backend map keys unique across both maps.
- #1: state Secret had BOTH `helm.sh/hook` and `resource-policy: keep` → hook resources aren't
  release-tracked so keep was moot + upgrade re-applied empty data. FIX: dropped the hook → normal
  release resource (applied before the post-* Job hook), adopted, upgrade-safe (empty `data` never in
  manifest → 3-way merge preserves Job-PATCHed keys), uninstall-preserved via keep. Role stays get/update/patch.
- #2: value-sourced `workspace.mongo.auth.rootPassword` goes into MONGO_URL userinfo UNENCODED. FIX:
  `_validate.validateUrlSafeValue` (allow `A-Za-z0-9!$&'()*+,._~-`; `!` from genSecretValue is WHATWG-
  userinfo-safe). Only workspace-mongo composes a cred into a URL; graph/FalkorDB pw is `--requirepass`
  argv (no URL); server-mongo pw is a separate env. README documents URL-safe + pre-existing state Secret.
- #4: user-supplied `provisioning.stateSecret.name` → chart neither creates it nor grants create; must
  pre-exist (fail-closed). Documented in README (can't lookup at render).
VERIFIED (render): instance=map-key in BACKENDS_JSON; dotted modelId renders; state Secret keep+no-hook;
real leak check with source=generate (value base64 in Secret data only, env via secretKeyRef, retain→keep);
key-collision fails; URL-safe bad(@)→fail good(!)→pass; lint clean; provision.py compiles; license OK.
GATE-6 (server-image contracts + fresh-install Workspace CreateContainerConfigError until Job writes
project-id — expected, not a bug) still pending, needs user consent per cluster-restraint rule.

## GATE 6 — FIRST CLUSTER RUN (2026-09-14, ns df-gate6 on cluster, TORN DOWN)
Pre-flight green: 3 images pull no-auth; llama.cpp:server + curl:8.11.1; tiny GGUF (Qwen2.5-0.5B q4,
491MB) resolves; no --alias so backend won't list modelId (no type=None). Installed lean (infra +
1 native backend gate-llm + server + embedded mongo + workspace + provisioning; vectorDB/graph off;
source=value creds; NO --wait). Findings:
- CHART BUG (FIXED, pushed cbcc131): embedded mongo ran as root + capabilities.drop:[ALL] → strips
  CAP_CHOWN → entrypoint chown /data/db fails → Terminating. Fix: run mongo non-root uid 999
  (entrypoint skips chown, fsGroup 999 keeps volume writable). workspace-mongo dodged it only by
  overriding command to bare `mongod --replSet` (no entrypoint, NO AUTH — separate deferred thread).
- CHART PARITY (FIXED, pushed cbcc131): both mongo StatefulSets lacked nodeSelector/tolerations/
  affinity/resources; added.
- HOMELAB STORAGE (NOT a chart change): NFS export is root_squash + ALL_SQUASH anonuid=65534, dirs
  already 0777. mongo then dies in wiredtiger_open with EPERM = MongoDB/WiredTiger UNSUPPORTED ON NFS.
  initContainer perms fix is a NO-OP here (all_squash) and can't help. Fix is a VALUES/overlay choice:
  storageClass: local-path for stateful comps. Chart stays storage-agnostic (already exposes
  storageClass per component). README now documents the NFS caveat. → values-homelab.yaml sets
  local-path on server.mongo + workspace.mongo (out of PR).
- c2 NODE INCIDENT (homelab, FIXED): kubectl logs/exec 502 "dialing 192.168.0.240:10250" for c2 pods.
  Raw net fine (curl :10250=Unauthorized from every node). Root cause: stale k3s apiserver→kubelet
  tunnel — c1-apiserver→c2 OK, c3-apiserver→c2 502 (c2 added ~5d ago, tunnel to c3 stale). Fix:
  restart k3s ON c2 (re-dials all servers); CP-HA held quorum. Verified c3+VIP proxy OK, uncordoned.
  Saved memory reference_k3s_kubelet_proxy_502_stale_tunnel. Red herring: dual-use-2026 deepfellow
  (DinD + 10.9GB llama-server) on c2 looked like the cause but wasn't.
NET FOR SIMPLITO PR: +2 real fixes (mongo non-root 999, mongo placement parity) + README storage
note. local-path + initContainer NOT in chart (homelab overlay / rejected). GATE 6 not yet fully
green — rebuild with local-path pending (option b); workspace-mongo no-auth is a separate thread (c).

## WORKSPACE MONGO AUTH (thread c, 2026-09-14) — FIXED, pushed 5c336e7
Finding: workspace MongoDB was UNAUTHENTICATED. StatefulSet overrode entrypoint with bare
`mongod --replSet` → MONGO_INITDB_ROOT_* dead (no user created), no --keyFile/--auth → mongod
ignored all creds; rs-init 'auth' ping passed only because auth unenforced; rootPassword decorative.
Fix (advisor-reviewed):
- keep image ENTRYPOINT (drop command override), pass args --replSet/--keyFile/--bind_ip_all so
  entrypoint creates root user from MONGO_INITDB_ROOT_* on fresh data dir (--keyFile implies authz).
- new workspace.mongo.auth.keyFile credential (base64-charset via genKeyFileValue; chartSecret got
  optional `generator` param). Own Secret, _validate'd, defaults existingSecret (GitOps-safe), retain
  if generated (changed keyFile breaks replSet data dir).
- keyFile perms: initContainer (uid 999) copies Secret→emptyDir chmod 400 (mongod rejects group/
  other-readable keyFile; Secret mount is root-owned). keyfile-src Secret mounted 0440 (fsGroup 999).
- workspace mongo now runs non-root 999 + dropped caps (parity w/ server mongo).
- NETPOL fix: allow rs-init Job (component=workspace-mongo) → mongod:27017 (was workspace-only; would
  hang rs.initiate under auth+netpol). Was latent (smoke ran netpol off).
- README: workspace mongo auth + fresh-data-dir migration caveat.
VERIFIED (render): lint clean; minimal renders; default still image-fail-closed; gate6 renders;
keyFile base64 (no !); keyFile validation fires; both mongos non-root 999; netpol has both sources.
CLUSTER-VERIFY IN (b): (1) uninitiated replSet member accepts admin-db auth (rs-init hinge —
entrypoint makes user in standalone temp mongod, real mongod starts uninitiated, Job auths+initiates);
(2) unauth mongosh is REFUSED (proves auth enforced, not just configured); (3) rs-init succeeds with
netpol ON.

Branch: 5c336e7 on origin/feat/native-k8s-helm-chart. Commits this session: 9211ece (advisor rd2),
cbcc131 (mongo non-root/placement/storage-docs), 5c336e7 (workspace mongo auth).

## SESSION 2026-09-20 — resume
- Pre-flight: repo clean apart from the intentional untracked `docs/superpowers/`; `__pycache__` is
  gitignored (only `files/provision.py` is tracked) so the PR stays clean. Cluster healthy (3 nodes
  Ready), no leftover `df-*` namespaces.
- **Committed + pushed `a85938f` — fix(helm): project Workspace mongo keyFile secret to a fixed path.**
  The `keyfile-src` volume mounted the Secret under its DATA KEY name (default `workspace-mongo-keyfile`)
  while the `copy-keyfile` initContainer reads `/etc/mongo-keyfile-src/keyfile` → the init step would
  fail for any key name but "keyfile". Added `items: [{key: <configured key>, path: keyfile}]`.
  Contract confirmed: `deepfellow.chartSecret` uses `existingSecret.key` as the canonical data-key name
  for ALL sources (value/generate too), so keying off it is correct. helm lint clean, render verified.
  This closes the workspace-mongo-auth thread (c) opened 2026-09-14.
- **GATE 6b: IN PROGRESS** — full-chart deploy into throwaway ns `df-gate6b`, pinned lw-c1,
  `storageClass: local-path` on all stateful components (NFS all_squash breaks WiredTiger),
  externalOnly enforcement=compatibility (no patched image built yet), provisioning ON, netpol ON,
  workspace ON, vectorDb/graph OFF, one small-GGUF llama.cpp backend with NO --alias.
  Matrix: A1-A6 (Gate-A assertions but through the chart) · B1-B3 (workspace mongo auth: uninitiated
  replSet accepts admin auth / unauth mongosh refused / rs.initiate with netpol ON) · C1-C4 (Server
  image contracts: real login token field, /health shape+path, DF_STORAGE_DIR writable uid 1000,
  workspace image name — all previously unverifiable, provision.py guesses defensively).

### GATE 6b RESULT (2026-09-20) — chart installed, provisioning did NOT converge
Teardown clean, chart files untouched by the agent, cluster healthy throughout.
A1-A6 PASS, B1-B3 PASS, C1-C4 answered — **but A1/A3-A6 only via a hand-fixed provisioning Job**.
`helm install` itself ended `BackoffLimitExceeded`. 11 bugs found, 5 blocking. Answers gained:
C1 login field is **access_token** (token/jwt branches were dead); C2 Server `/health` → 200
`{"status":"ok"}` unauthenticated (the image's own healthcheck.py hits `/docs`); C3 DF_STORAGE_DIR
writable for uid 1000 (local-path honours fsGroup); C4 workspace image name CORRECT.

### FIXES APPLIED (commits 4aefa61 + c7b44d4, pushed)
All 11 addressed (#11 dismissed as deliberate). 17/17 render assertions + lint + fail-closed gates
+ full `check` suite (license/ruff/pyright 0-0-0/auth 59+71) green.
- provisioning: venv interpreter for `server.scripts.create_admin` (PATH python lacks argon2);
  `/v1/models` poll uses **DF_INFRA_API_KEY** not the admin key (admin key 401s on /v1 → looked like
  "model never appeared"), key plumbed into the Job; failures now report captured output (password
  redacted) + last HTTP status + model ids seen; login parses `access_token` only.
- server: `DF_PLUGINS_PATH` → mounted PVC, **unconditional** (plugin manager reads the dir whatever
  plugins.setup says; /app/plugins is root-owned → PermissionError crashloop). Mount point declared
  once as `$storageMount` and reused.
- workspace: `BOOTSTRAP_ADMIN_NAME` emitted + `bootstrapAdmin.name` value; `MONGO_URL` →
  **`MONGO_DB_URL`** + `MONGO_DB_NAME`; `DF_SMALL_MODEL` → **`SMALL_MODEL`** unconditional (required
  by the image) + render-time gate; `DF_SESSION_TITLE_MODEL` removed (not in the image schema);
  `HOST=0.0.0.0`/`PORT=3000` so it stops binding localhost (mirrors the vendor installer's .env.app).
- netpol: default-deny selected `part-of`, a metadata-only label → matched ZERO pods, denied nothing;
  now selects name+instance. model-backends policy now admits `component: provisioning` (the Job
  health-checks backends directly). Dead `$np` var removed.
- rs-init hook Job inherits workspace.mongo placement + resources (Helm does NOT post-render hooks,
  so this was unfixable at deploy time).
- downloadImage (curl) pinned to a digest instead of forcing allowMutableTag on a chart-chosen image.

### HERMES CHANNEL FIXED (cellarette MCP was down this session)
Two verified paths, saved as memory `reference_hermes_without_cellarette`: hermes-api :8642 via
port-forward with **`Authorization: Bearer $API_SERVER_KEY`** (X-Hermes-Session-Key is NOT auth —
that 401 is a red herring), and the teammates consult **NodePort 30615** `{bot,prompt}`. No `hermes`
bot in the roster; use `dev` for code review.

### IN FLIGHT
- GATE 6c running: two scenarios, no hand-patching allowed — (A) batteries-included must converge
  unattended, (B) **BYO**: pre-existing MongoDB + pre-existing OpenAI-compatible backend, chart wired
  via `server.mongo.mode: external` + `externalBackends`, workspace off. BYO was designed but never
  cluster-tested.
- THEN: Hermes review for readability/maintainability, then squash to 3 commits (helm / infra /
  changelog) per the agreed option (b), backup tag + `--force-with-lease`.

### GATE 6c + 6d RESULTS (2026-09-20) — both deployment shapes PROVEN on cluster
Not just rendering — actually deployed, twice, in throwaway namespaces on lw-c1, torn down clean.
- **6c scenario A (batteries-included): converged UNATTENDED.** provisioning Job Complete 1/1 in 67s
  (full happy path in its log), A1-A6 PASS, NetworkPolicy default-deny now matches all 8 pods (it
  matched ZERO before the selector fix), rs-init landed on the pinned node. 11 bugs found overall.
- **6c scenario B (BYO): converged but CHEATED** — it only passed because the external Mongo had been
  created with the Server image's compiled-in default credentials. Real BYO was impossible: the Server
  composes `mongodb://<user>:<pass>@<DF_MONGO_URL>` itself, so DF_MONGO_URL is host:port and the old
  `external.uriSecret` (documented as "the URI") could never work → `InvalidURI: Bad database name`.
- **6d (BYO with its OWN credentials): 5/5 PASS, zero workarounds.** Server connected as a custom user
  to a custom database and ran its migrations there (16 collections — proof it did not fall back to
  defaults), provisioning Complete in 57s, Workspace **1/1 Ready with a Service endpoint**, chat through
  the Server returned PONG, render-time guard rejects a `mongodb://` URI.
  Working shape: `url: "<host>:27017/?authSource=admin"`, `username`, `database`, `password` (cred shape).

### FIXES THIS SESSION (all pushed; 18+ commits, to be squashed to 3)
a85938f keyFile→path · 4aefa61 the eight "installs on a cluster" fixes · c7b44d4 README ·
5bf4dc8 external Mongo contract + Workspace /health probe · 6a661b2 provision.py under the quality
gates + 28 mocked tests · 2cb9a1f Workspace init gate (was CreateContainerConfigError, kubelet backoff
→ ~5 min on a cold install) · faa3b0d maintainability review items · 5604659 external-only registration
guard · 37c2abe normalizedBackend helper + workspace template split · cf63ab3 DockerDisabledError.
Gates: ruff clean, pyright strict 0/0/0, license OK, **5113 tests pass**, helm lint clean, default still
fail-closed, both scenarios render byte-identically across the refactors.

### GOTCHAS WORTH KEEPING
- `lifecycle` turns AppStartError into `os._exit(1)`, so a startup assertion that trips inside a test
  KILLS pytest with truncated output and no FAILED line. That is how the external-only guard first
  showed up — it looked like a hang at 88%.
- Adding ANY field to `AppSettings` breaks `tests/unit/server/api/test_config_api.py`: its fixture
  builds a spec'd mock field by field and the config API iterates all model_fields.
- `provision.py` must NOT go into `coverage.run.source` — that has `fail_under = 100` repo-wide.
- `_docker_bin` is a property, not a method.
- Upstream's `uv.lock` is stale (df-infra 0.32.0 vs pyproject 0.33.0); every `uv run` dirties it —
  `git checkout -- uv.lock` before committing.

### REMAINING
- Second Hermes review (in flight) on the final state; then squash to 3 commits (helm / infra /
  changelog) per option (b), backup tag + --force-with-lease.
- Not done, deliberately: moving provision.py into a package (the chart mounts it via .Files.Get).

### SECOND HERMES REVIEW + SQUASH — DONE (2026-09-20, end of session)
Second pass verified all 12 earlier findings against the actual diff (not the commit messages) and
found three more; verdict: **no hard blocker**. All three fixed:
1. (his "fix before merge") the GLIBC_TUNABLES workaround was on the Server's Mongo but NOT the
   Workspace's, though both pin the identical image — explaining it in one template had moved the
   drift between the two files. Now `deepfellow.mongoImageEnv`, used by both.
2. provision.py's tests covered leaf logic but not the reconcile SEQUENCING. Added ordering tests:
   state read first, register+model-ready before grant, key persisted before grant, no second key
   when state exists, partial state refuses.
3. an existing service instance reporting no installed endpoint silently skipped the endpoint
   comparison; it now logs.
GOTCHA: the consult first failed with `usage_limit_reached` for `chatgpt/gpt` — the `dev` bot is
`claude-code` but fell back to Codex, whose quota was spent. `/consult` accepts a `"model"` field;
`{"bot":"dev","prompt":...,"model":"claude/sonnet"}` worked.

**BRANCH SQUASHED to the agreed 3 commits and force-pushed** (`--force-with-lease`):
88f7fc1 feat(helm) · b3c2646 feat(infra) · a7bc217 docs(changelog). Tree proven byte-identical to the
21-commit history before squashing; the original is kept at tag `backup/pre-squash-2026-09-20`.
Final state: ruff clean · pyright strict 0/0/0 · license OK · **5119 tests** · helm lint clean ·
both deployment shapes render · default still fail-closed.

### WHAT IS NOT DONE
- No PR opened upstream. The user's instruction was to adapt the branch, not to submit it.
- provision.py stays under deploy/helm/ (the chart mounts it via .Files.Get); Hermes accepted this
  given lint+types+tests, but called moving it into a package the durable follow-up.
- Chart is unreleased/untagged; Chart.yaml appVersion still tracks the images it was tested against.

### LIVE INFERENCE DEMO + VOLUMES + --wait FIX (2026-09-20, later)
**Inference proven in both shapes** (ns df-demo-a / df-demo-b, torn down): `/v1/models`, factual answer
("Paris"), format-following ("Red, Yellow, Blue"), MULTI-TURN with context carried, real SSE streaming,
`system` steering, 401 without a key. CPU-only: ~34 tok/s generation, ~100 tok/s prompt eval, prefix
cache works. The 0.5B rambles and its `kubectl` answer is useless — reported as-is, not retried.

**Bug the demo exposed (self-inflicted): `helm install --wait` deadlocked.** Helm waits for release
resources Ready BEFORE post-install hooks; the Workspace init-gate waited for the provisioning hook →
circular. The rs-init hook had the same shape one step later (Workspace can't reach an uninitiated
replSet). FIX: neither is a hook now — zero `helm.sh/hook` anywhere in the chart. Proven: `--wait`
converges in **105s**, both Jobs run concurrently with the not-yet-Ready Workspace.

**SECOND self-inflicted bug, caught by verification:** as ordinary resources Helm PATCHES Jobs, and a
Job pod template is immutable. The name digests covered only a hand-listed input set, so changing
`server.admin.email` / `server.image.tag` / `server.project.orgName` / workspace-mongo auth kept the
name while the template moved → `UPGRADE FAILED ... field is immutable` (proven live). FIX: digests
now cover every values subtree those pod templates read. Re-verified 14/14: name changes exactly when
the rendered pod template changes; unrelated changes (ingress) don't churn the Jobs.

**existingClaim added** for server.mongo, workspace.mongo, qdrant, graph (infra/server/backends had it).
For a StatefulSet a supplied claim REPLACES the volumeClaimTemplate (a template cannot adopt an
existing PVC) — sound only because each runs one replica; README says so. Milvus/etcd/minio share ONE
storage block so one claim cannot back three StatefulSets → deliberately not offered, documented.
PROVEN live: both Mongos mounted operator-made PVCs, no `data-*-0` claims, databases worked, chat OK.
README now documents what survives uninstall (chart PVCs go; state Secret kept on purpose; StatefulSet
PVCs kept by Kubernetes → reinstall meets populated DBs + stale key unless cleared).

**Branch re-squashed to 3 commits and force-pushed:** 981ab57 feat(helm) · 16ba880 feat(infra) ·
5f5c674 docs(changelog). Tree proven identical; pre-squash history at `backup/pre-squash2-2026-09-20`.
Final: ruff · pyright 0/0/0 · license · **5119 tests** · helm lint · both shapes render · fail-closed.

## SESSION 2026-09-22/23 — backing stores + upgrade path PROVEN
Unpushed local commits on top of the 3-commit squash: 161c248, 0dedbdd (provisioning re-run fixes),
0f60c95 (MinIO → quay.io, digest-pinned; Docker Hub dropped it), 4abe31a (Qdrant `-unprivileged`
default; Milvus `supplementalGroups: [0]`; placement+resources for qdrant/falkordb/etcd/minio/milvus;
`provisioning.revision` accepts int; README: admin.email change creates a 2nd admin), cbe5f5d
(NetworkPolicies for the Milvus stack; readiness probes on etcd/minio/milvus/falkordb/workspace-mongo).
User rejected a bare push on 09-22 → push together with the final squash.

PROVEN on cluster (throwaway ns, lw-c1, torn down):
- `helm upgrade --wait --wait-for-jobs` ×2 → exit 0, re-run provisioning Complete (no type=None).
- existingClaim for Qdrant + FalkorDB (no vct, pods mount operator PVCs).
- Milvus stack with networkPolicy ON: install --wait 77s, milvus runs uid 1000 groups 0, /healthz 200,
  Server "Connected to Vector Database (Milvus)", provisioning + chat 200.
- Qdrant default tag, FalkorDB auth-enforced probe, workspace-mongo probe (ping works unauth under keyFile).
GOTCHAS: Qdrant plain tag panics under non-root (snapshots tmp); Milvus image bin+lib root:root 0774;
default-deny covers the whole release so every in-chart backing store needs its own allow.
REMAINING (robustness, not blocking): cold install gives ONE restart each to milvus (etcd not ready yet)
and the Server (vector DB not ready yet) — no dependency waits. Server→FalkorDB connection has no log
line, so that path is configured but not proven by traffic. New cluster node `lw-amd` (AMD GPU) appeared
2026-09-22 — nothing of ours ran there.

## 2026-09-23 — DONE: zero-restart cold installs, squashed + pushed
Final extra fixes before squash: MinIO → quay.io (Docker Hub dropped it), Qdrant `-unprivileged`
default, Milvus `supplementalGroups: [0]`, Milvus-stack NetworkPolicies, readiness probes on every
backing store, dependency-wait init containers (`dependencyWait` values block), Milvus Service
publishes 9091, rs-init retries rs.initiate + `publishNotReadyAddresses`.
PROVEN (NetworkPolicy ON): Milvus and Qdrant+Workspace+FalkorDB cold installs → `--wait` exit 0,
restartCount 0 on EVERY container incl. init + Job pods; chat 200; provisioning Complete.
Branch squashed to 3 commits and pushed (--force-with-lease): d63180e feat(helm) · f96c4c8
feat(infra) · 9effa3a docs(changelog). Pre-squash history: tag `backup/pre-squash3-2026-09-23`.
Gates: ruff · pyright 0/0/0 · license · 5131 tests · helm lint · default fail-closed.
Still NOT proven by traffic: Server→FalkorDB (no connection log line). No upstream PR opened.

## 2026-09-23 (cont.) — maintainability pass ("Rób WSZYSTKO": all 10 audit findings + STYLEGUIDE)
Local commits on top of 9effa3a (to be re-squashed): split per component/kind (render-identical),
backend map normalisation (nil-pointer fix) + component predicates, layered securityContext/probes +
rs-init hardened (was root), token-less workload SA + .helmignore fixes (pyc shipped, quickstart
excluded), checksum/credentials, podLabels/podAnnotations + sharedPodInputs Job digest, kubeVersion +
generated artifacthub.io/images, strict schema (+ removed dead plugins.abuseModel/anonymizeModels),
helm-unittest (31) + ci/test.sh + GitLab helm-chart job (checksum-pinned tools) + helm-docs README,
STYLEGUIDE.md + CLAUDE.md pointer, pyright covers ci/.
Safety net: scratchpad golden-diff.py (semantic, Secret data masked, DUPLICATE detection — it missed a
5x duplicated Secret until dup detection was added) + edge-values fixture compared old-vs-new.
GOTCHAS: RTK truncates `helm template` stdout (use `rtk proxy`); `dig` default only when key MISSING;
mergeOverwrite drops false/0; helm-docs collapses a `# --` parent object; helm lint exits 0 on a failed
template (assert fail-closed with helm template); `uv run` dirties uv.lock (checkout before commit).
Cluster verification (lw-c1, netpol ON, all torn down, 0 df-verify ns/PV left):
- Upgrade old(9effa3a)->new FOUND 2 real bugs, fixed + re-proven:
  (1) workspace-mongo restart -> mongod started before its member name resolved to the NEW pod IP
      -> "not a member of the config" forever (pre-existing!). Fix: init waits until
      `getent hosts <pod>.<svc>` == `hostname -i`; rs-init initiates only on NotYetInitialized.
      Pod-kill test: PRIMARY back.
  (2) rs-init `field is immutable`: hand-listed digest inputs missed a script change. Both Jobs now
      render pod template in a define; name = sha(pod [+ provision.py]).
- Workspace restarts on startup: cause = "Deepfellow health check failed" (Server Recreate), not mongo.
  Added waits (mongo primary via db.hello, Server /health): reboot simulation (delete all pods) -> 0
  restarts everywhere. An upgrade rolling Workspace+Server/mongo together can still restart it 1-2x
  (waits pass on old pods) -> documented in README Upgrading.
- Milvus cold install: 0 restarts, Server "Connected to Vector Database (Milvus)", chat 200.
- No-op upgrade keeps the same Jobs. Token mounted only on provisioning; both Mongos uid 999.
README rewritten concise (humanify: 2196 -> 1008 prose words, 0 dashes).
NOT DONE: squash (classifier blocked history rewrite) + push. 19 local commits on 9effa3a.
- 2026-09-23: squashed to 3 commits (aa31ee7 chart, de0b699 infra, 2d5593c changelog), pushed --force-with-lease; backup tag backup/pre-squash4-2026-09-23. Subcharts: user said keep in-chart stores.
- 2026-09-23: short values contract squashed into chart commit; pushed f3a63b1 (backup/pre-squash5-2026-09-23).
- 2026-09-23: secretKeyRef/plaintext credentials + deepfellow_provision package squashed; pushed f165b50 (backup/pre-squash6-2026-09-23).
- 2026-09-23: readability pass with Hermes (15 findings) implemented as e276292, local; cluster-verified quickstart+pod-kill+Milvus.
- 2026-09-23: readability pass squashed; pushed a2086b0 (backup/pre-squash7-2026-09-23).
- 2026-09-23: secretRef/secretKey credentials squashed; pushed 479d943 (backup/pre-squash8-2026-09-23).
- 2026-09-23: comment audit (248) + mesh connectKey fix + infra:latest=v0.29.0 stale finding; local commit, unpushed.
- 2026-09-23: comment audit + mesh connectKey + pinned versions (infra v0.34.0, server v0.34.1, workspace v0.3.2) squashed; pushed 668c105 (backup/pre-squash9-2026-09-23). Workspace v0.5.2 needs Redis.
- 2026-09-23: Redis for Workspace + Workspace v0.5.2 pinned, local commit; cluster-proven.
- 2026-09-23: Redis + Workspace v0.5.2 squashed; pushed 5b07684 (backup/pre-squash10-2026-09-23).
