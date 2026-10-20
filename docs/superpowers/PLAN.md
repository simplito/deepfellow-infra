# DeepFellow Suite — native-k8s Helm chart — DESIGN (grill output)

## Settled decisions
Q1(c) build standalone now, contribute upstream later · Q2 NO DinD · Q3(a) declarative granular
(no heavy suite install) · Q4 reusable, DeepFellow defaults · Q5 native plugins only (Server) ·
Q6 vanilla k8s defaults · Q7(a) self-contained + per-component BYO · Q8(a) in-chart model backend ·
Q9(c) WHOLE Suite (infra+server+workspace) · Q10(a) chart + upstream external-only patch (parallel) ·
Q11(a) native llama.cpp registered as `openai` external service · Q12 mesh values · Q13(a) workspace ON ·
Q14(a) qdrant, vectorDb.active=false default · Q15 Mongo in-chart + BYO, MinIO milvus-only ·
Q16 structured plugins → DF_PLUGINS_SETUP JSON, empty default, don't pull image · Q17 anon pull works.

## Architecture (request path)
consumer → **Server :8000** (dfproj key, plugins: abuse/anonymize) → **Infra :8086** (OpenAI proxy)
→ **native llama.cpp** Deployment(s) (registered in infra as `openai` EXTERNAL service — no docker socket).
Workspace :3000 (panel) → Server. Only Infra is socket-coupled; bypassed via external registration.

## Manifest inventory
1. **infra** Deployment(:8086) + Service + Secret(DF_INFRA_ADMIN_API_KEY req) + PVC(DF_STORAGE_DIR:
   config.json + endpoint state; NO weights). Boots socket-free (external-only). docker CLI stays in
   image until upstream patch lands.
2. **model-backends** (Q11a) N× llama.cpp Deployment (ghcr.io/ggml-org/llama.cpp:server-*) + Service.
   GGUF via initContainer download → PVC/emptyDir. Command: --parallel/--threads/--ctx-size from values
   (WE own full flags — DeepFellow's managed path can't set --threads). CPU default (homelab), GPU via values.
3. **server** Deployment(:8000) + Service + PVC(/app/storage,/app/plugins) + ConfigMap(boot env) +
   Secret(mongo creds, plugins). Probe = exec /app/scripts/healthcheck.py. init-mongo via ConfigMap mount.
4. **mongo** (server DB) StatefulSet(mongo:8.2.7)+PVC+initScript ConfigMap+Secret — OR external via
   mongo.external.uri (Q15). Service.
5. **qdrant** StatefulSet(qdrant:v1.15)+PVC — rendered only when vectorDb.active && type=qdrant
   (default active=false → NOT rendered). Service.
6. **milvus stack** etcd(v3.5.18)+minio(RELEASE.2024-12-18)+milvus(v2.6.2) — only when type=milvus&&active.
   MinIO exists ONLY here (Q15).
7. **falkordb** graph — opt-in, off default.
8. **workspace** (Q13a ON) Deployment(:3000, deepfellow-workspace) + Service + Secret(BETTER_AUTH_SECRET,
   BOOTSTRAP_ADMIN_*) + its OWN **app-mongo** StatefulSet (mongo, replSet rs0 → needs rs.initiate init job).
   env: DF_SERVER_URL + DF_SERVER_API_KEY + DF_SERVER_ORGANIZATION_ID + DF_SERVER_PROJECT_ID (from prov Secret).
9. **provisioning** (core artifact):
   - Job-A: server image, cmd `python -m server.scripts.create_admin <name> <email> <pw>` (Mongo env). Idempotent.
   - Job-B: login (POST /auth/login) → POST /admin/workspace/ (atomic org"Workspace"/project"Default"/key"app")
     → write dfproj key + org_id + project_id into k8s Secret (GET-ONCE) → grant models
     (POST /v1/organization/projects/{pid} {models:[...]}). Guard: skip if Secret exists.
   - Job-C: register native backend(s) into infra: POST /admin/services/openai {spec:{api_url,api_key},stream:false}
     then POST /admin/services/openai/models/_?model_id=<id> (custom spec type=llm, completions=true, context_length).
     SKIP all infra ollama/heavy-model steps (suite-install steps 5-8) entirely.
   - Ordering: helm hooks + readiness waits (infra ready → Job-C; mongo+server ready → Job-A → Job-B → workspace).
10. **Secrets**: infra-admin-key, mongo creds, workspace secrets, dfproj project-key(minted Job-B),
    HF token(optional), imagePullSecrets(optional — anon pull works).
11. **NetworkPolicy** vanilla: OFF default (portability). Homelab overlay adds default-deny + morsik/whisper allows.
12. **Ingress** vanilla networking.k8s.io/v1, OFF default. Targets: Server(/v1) and/or Workspace.

## values.yaml top-level shape
global{imageRegistry,imagePullSecrets} · infra{image,adminApiKey,storage,resources,service,mesh{connectUrl,key},externalOnly:true}
modelBackends:[{name,image,gguf{url},alias,parallel,threads,ctxSize,resources,gpu{enabled}}]
server{image,storage,mongo{enabled|external{uri,...}},vectorDb{active:false,type:qdrant,embedding,qdrant,milvus},
       graph{enabled:false},plugins{abuseModel,anonymizeModels[]},otel,admin{name,email,password|existingSecret},
       project{orgName,projectName,keyName},resources}
workspace{enabled:true,image,mongo,auth{bootstrapAdmin,betterAuthSecret},sessionTitleModel,smallModel,resources}
provisioning{image} · ingress{enabled:false,className,server{host,path:/v1},workspace{host}} · networkPolicy{enabled:false}

## Upstream patch (Q10a, parallel PR to deepfellow-infra)
Add DF_EXTERNAL_ONLY boot flag: skip create_docker_service (or make DockerNotInstalledError non-fatal +
skip has_gpu/is_rootless probes), hide managed services from API/UI, drop docker CLI from image.
Files: server/config.py (add setting), server/lifecycle.py (gate create_docker_service + service registration).
Chart works WITHOUT the patch today (docker CLI present, external services only); patch = clean contribution.

## Deliverables layout — ONE PR to simplito/deepfellow-infra (chart + patch together)
Fork deepfellow-infra, single branch/PR:
  deploy/helm/deepfellow/   (the chart — whole Suite, vanilla)
  server/config.py          (+ DF_EXTERNAL_ONLY setting)
  server/lifecycle.py       (gate create_docker_service + managed-service registration)
  Dockerfile                (docker CLI optional when external-only, if trivial)
Because the patch ships in the SAME PR: `infra.externalOnly: true` REALLY sets DF_EXTERNAL_ONLY (the
patched image honors it) — advisor fix #1 dissolves. Default chart image tag must be a build INCLUDING
the patch. Pre-merge: build patched image → ghcr/homelab registry; document. Post-merge: upstream tag.
Heads-up (reviewer optics): a WHOLE-SUITE chart living in the infra repo is slightly off (infra repo
deploying server+workspace). User accepts; flag once to reviewers, offer to scope-split if they push back.

## Homelab side — NOT in the PR (stays in user's repos)
values-homelab.yaml (nodeSelector c2, local-path, netpol, Caddy/Traefik, small CPU GGUFs, FALOCHRON
plugins abuseModel: qwen3-0.6b, mesh via ESO) + argocd-apps/apps/deepfellow.yaml (auto-sync OFF initially).

## GATE A RESULT — PROVEN ✅ (2026-09-14, throwaway ns df-gate on lw-c1, torn down)
Socket-free native-k8s DeepFellow is VIABLE. External `openai` service (api_url → plain OpenAI-compat
endpoint, NO docker socket) IS grantable to a Server project by plain model id — Server does NOT require
the id to come from Infra's docker-managed catalog. A1-A6 all PASS (A6: ungranted project → 403 "Model not
allowed"). Infra booted WITHOUT a daemon (image bundles docker CLI; `docker compose version` ok, daemon
probes non-fatal). Server booted + mongo auth (init dfuser on admin/readWrite) ok.
Finding: viking://resources/homelab-knowledge/deepfellow-socket-free-native-k8s.md

### BUILD refinements forced by Gate A (fold into templates)
1. **Model register = TWO calls:** POST /admin/services/openai/models/custom (typed def {id,type:llm,...})
   THEN POST /admin/services/openai/models/_?model_id=<id> (install → proxy). NOT one call.
2. **type=None collision:** a backend whose /v1/models LISTS the model gets it seeded type=None at
   service-install → later custom-add 400 "already exists". For NATIVE llama.cpp (which DOES list its
   model): register custom model BEFORE/independently, or reconcile the collision. Job-C must handle this.
3. **DF_MESH_KEY is REQUIRED at boot by infra:latest** (image drift vs source) — infra crashloops
   "Missing config value for DF_MESH_KEY" without it. Chart MUST set it (dummy when mesh unused). mesh{} value.
4. **server:latest has NO atomic /admin/workspace/** (405) — provisioning must use 3 primitives:
   POST /admin/organization/ → POST /v1/organization/projects → POST /v1/organization/projects/{id}/api_keys.
   (matches setup.sh path, not suite-install's atomic path). Update Job-B.
5. **Admin password policy: >=10 chars, lower+upper+digit+special.** Chart admin password gen/validate must meet it.

## GATE — (HISTORICAL, now PASSED) verify BEFORE writing any template
The spine (external `openai` service → Server model catalog → project grant → /v1 through Server) is
INFERRED from CLI wiring, NOT verified end-to-end. If the Server's project-grant only accepts model
ObjectIds from a catalog that populates ONLY from infra's MANAGED services, external registration can't
be granted → Q11(a) collapses onto the docker-managed path (the whole thing we're avoiding). MUST prove:
  1. docker run infra+server+mongo (anon images) OR `deepfellow server install` (CLI cloned).
  2. stub OpenAI responder (~20 lines py: /v1/models + /v1/chat/completions canned) — NO llama.cpp needed.
  3. POST /admin/services/openai {spec:{api_url:stub,api_key:x}} + register custom model.
  4. create_admin → login → POST /admin/workspace/ → capture dfproj key.
  5. **CRITICAL**: does GET /v1/models on SERVER(:8000, Bearer dfproj) list the stub model?
     does POST /v1/organization/projects/{pid} {models:[id]} ACCEPT its id?
  6. one /v1/chat/completions through the Server.
  GREEN → build chart. RED → renegotiate scope (patch must bridge infra externals into Server catalog).

## Design fixes (advisor)
- infra.externalOnly gates CHART behavior (skip Job-C managed paths + document convention), NOT a
  DF_EXTERNAL_ONLY env — released image ignores that until the patch merges. Don't lie in the values contract.
- Workspace Deployment: initContainer that WAITS for the dfproj Secret (Job-B mints it via post-install hook;
  otherwise Workspace CreateContainerConfigError on first install). Ordering decision, state it.

## Hermes review (2026-09-14) — ADOPTED refinements
Verdict: no-DinD design plausible; smoke gate is the right HARD pre-template gate; OpenViking had no prior
DeepFellow notes. Deltas folded in:
- **Gate A smoke-test FIX (important):** a bare unknown model from stub /v1/models resolves type=None →
  `RemoteService._install_model()` REJECTS it. So DON'T register a raw id. Use EITHER a known catalog id
  (gpt-4o-mini) OR an explicit CUSTOM openai model: {id, type:llm, only real endpoints — completions=true,
  NOT /v1/responses or /v1/messages}. My earlier stub plan would've given a false/inconclusive result.
- **Gate A = "external model authorization compatibility. No chart work before PASS."** 7 assertions:
  (1) infra registers remote openai w/o socket (2) infra /v1/models has it (3) Server project-grant ACCEPTS
  the exact external id (4) Server /v1/models via PROJECT key exposes it (5) completion reaches stub (6) a
  2nd project WITHOUT grant is denied (7) if plugins on, prove request traverses them (not infra-direct).
- **Gate A extra failure modes:** id identity must match across display/infra-registered/llama.cpp/grant;
  base-URL join (infra appends `v1/` → value = backend ROOT `http://llamacpp:8080`, not .../v1); readiness
  race (poll backend /health + infra model-test BEFORE grant); llama.cpp ClusterIP-only + netpol infra→llama
  only; test SSE streaming not just buffered; 600s proxy timeout as a value; failure isolation (no restart loops).
- **DF_EXTERNAL_ONLY contract (patch must genuinely remove docker init, not just skip socket):** no
  create_docker_service; no docker CLI/socket/compose/GPU-probe/network-inspect at boot; register ONLY
  external service types (min: openai); docker-backed install/model endpoints return deterministic 409
  "disabled by DF_EXTERNAL_ONLY"; normal mode unchanged. Prefer explicit external-only ALLOWLIST over a null
  docker adapter (hides deps). Unit tests: boot docker-absent · external_only registers openai · docker install
  rejected · normal mode still inits docker lifecycle.
- **Chart positioning:** deploy/helm/deepfellow/ (NOT named "infra"); position as "the Kubernetes distribution
  of the suite"; no scope creep (Server+Infra+llama.cpp+needed persistence+provisioning only); pin component
  images independently + state compat in Chart.yaml/values.
- **Provisioning ORDER FIX:** register backend + verify infra model ready BEFORE granting project model
  (grant must never reference an unavailable model). New order: wait-healthy → create/reconcile admin → login →
  reconcile workspace/key Secret → register external openai service + custom model → PROVE infra model ready →
  grant project → verify through Server w/ project key.
- **Idempotency (dfproj get-once):** FIRST action = read the k8s Secret; if it has org_id+project_id+key, skip
  /admin/workspace/. Persist the one-time response to Secret BEFORE any grant. If workspace exists remotely but
  Secret write failed → FAIL CLOSED, operator recovery (unavoidable crash window). All later ops = reconcile
  (admin: existing-email no-op; service/model: desired-state compare; grant: additive set). PRESERVE credential
  Secret on uninstall (helm keep policy / documented). AVOID one-shot immutable Job name (upgrades won't rerun)
  → reconciliation Job whose name changes only on deliberate provisioning revision, OR post-install/post-upgrade
  hooks with SAFE retries + no cred leakage. Secret state = source of truth, hooks = scheduling only.
- **Readability additions:** script interface `reconcile`/`verify`/`status` (shallow templates, deep script);
  `set -euo pipefail`, NEVER `set -x`; no tokens in Job stdout/Events/annotations/helm output; dedicated SA +
  Role scoped to ONE secret name in ONE ns; netpol provisioning→Server/Infra only; pod securityContext
  (non-root, RO rootfs where possible, drop caps, no privileged); helm template golden tests + schema in CI;
  chart-level assertion: fail render if native llama.cpp enabled while infra.externalOnly false (and vice versa).

## ⚑ PHASE 1 CONTRACT REWORKED 2026-09-14 (user chose FULL fail-closed after Hermes gate)
The decisions below were SUPERSEDED where they conflict with the Hermes review-gate (see that
section lower). Net contract now: fail-closed default; digest-first images (`allowMutableTag`);
`infra.externalOnly.{enabled,enforcement}`; `modelBackends` + `externalBackends` as KEYED MAPS;
credential `source: existingSecret|value|generate|none` discriminators (no empty-string XOR);
`server.mongo.mode: embedded|external`; `provisioning.enabled` default false + backend gate.
Superseded specifically: #2 (externalOnly bool→enabled+enforcement), #4 (list→map),
#6 (schema now $defs image/credential + propertyNames maps), and the auto-generate-default
(now fail-closed: default source=existingSecret). Verified: helm lint clean; 13/13 assertions
(default FAILS fail-closed; ci/minimal-values passes; native backend passes; all negatives reject).

## PHASE 1 BUILD DECISIONS (resolved 2026-09-14, advisor-reviewed) — see rework note above
Resolves the line-71 vs line-117 contradiction + locks the values contract:
1. **infra.image.tag default = `latest`** (NOT a semver pin). Reason: hub.simplito.com/deepfellow
   images publish only `:latest` (CLI `defaults.py` DF_INFRA_IMAGE/DF_SERVER_IMAGE both `:latest`);
   Gate A ran `:latest`. `infra.image.digest` value exposed (empty default) so a deploy CAN pin a
   digest for reproducibility; README documents digest-pinning as the production/pre-merge step.
   Phase 5 flips the default tag to the patched build with NO values-contract change.
2. **`infra.externalOnly: true` (default)** = chart-side semantics (provisioning skips docker-managed
   paths) AND sets env `DF_EXTERNAL_ONLY`, honored ONLY by images carrying the Phase-5 patch. Honest
   pre- and post-merge; documented in values comment + README. Not a lie: unpatched image ignores env.
3. **`infra.mesh.key` non-empty default** (`"unused-change-to-join-a-mesh"`). Gate A #3: infra:latest
   crashloops on missing DF_MESH_KEY. Empty default = broken chart, so it ships non-empty.
4. **modelBackends = LIST** (PLAN line 52). Helm REPLACES lists (not merges), so values-homelab.yaml
   must restate every backend entry to change one. Accepted for upstream vanilla ergonomics; noted in
   values comment. (A keyed map would fix overlays but lose ordering.)
5. **_validate.tpl ships ONE arm** of Hermes's cross-check: fail render when native llama.cpp backends
   exist AND infra.externalOnly=false. The vice-versa arm (externalOnly=true + no backend) is DROPPED —
   that is a legitimate BYO-endpoint config (the exact shape Gate A ran); hard-failing it is wrong.
6. **values.schema.json = settled invariants only** (RE2-safe, gojsonschema has NO lookahead): types,
   vectorDb.type enum, modelBackends[].name DNS-1123 pattern + gguf.url required, password minLength:10.
   Character-class password complexity + all XOR checks (adminApiKey^existingSecret, mongo.enabled^
   external.uri) live in _validate.tpl via combined `regexMatch` (also RE2), NOT in schema.
7. Component image defaults (from CLI defaults.py, authoritative): infra/server
   `hub.simplito.com/deepfellow/deepfellow-{infra,server}:latest`; mongo
   `mongo:8.2.7@sha256:7abfba0d07c9330373f8173981ea4d09cd8a82cdf0e86ccaf7008848d1d24f62`;
   qdrant `qdrant/qdrant:v1.15`; milvus stack etcd `quay.io/coreos/etcd:v3.5.18` +
   minio `minio/minio:RELEASE.2024-12-18T13-15-44Z` + milvus `milvusdb/milvus:v2.6.2`;
   falkordb `falkordb/falkordb:v4.12.4`; llama.cpp `ghcr.io/ggml-org/llama.cpp:server`.
   workspace `hub.simplito.com/deepfellow/deepfellow-workspace:latest` (INFERRED — not in CLI,
   not Gate-A-tested; VERIFY before Phase-2 workspace template).
8. Server→infra link env is **double-underscore**: `DF_INFRA__URL`, `DF_INFRA__API_KEY` (Gate A finding,
   empirically verified — differs from compose's single-underscore DF_MONGO_URL etc.).
9. Phase-1 validation fires via a dedicated `templates/validation.yaml` (`include` of the partial,
   renders empty when valid) so `helm template --set ...` actually exercises _validate today.

## HERMES PHASE-1 REVIEW-GATE (2026-09-14, #hermes, review-only)
Verdict: **conditional approval for Phase 2 — 3 contract changes first.** Hermes could NOT line-review
(fork branch unpushed; Phase 1 + PLAN/HANDOFF are local-only) → design-level review from the summary +
OpenViking Gate A record. THREE mandatory changes before Phase 2:
1. **modelBackends = MAP, not list** + a SEPARATE `externalBackends` map for BYO endpoints. Native vs
   BYO becomes explicit (`modelBackends.<name>.enabled`, `externalBackends.<name>.{serviceType,apiUrl,
   modelId}`); overlays add/disable one entry without restating all; Phase-2 templates never infer
   intent from missing fields. (REVERSES PLAN line 52 / Phase-1 decision #4 "list".)
2. **Explicit source/mode discriminators, not empty-string XOR.** e.g.
   `auth.infraAdminApiKey.source: existingSecret|value|generate` (+ nested existingSecret{name,key}/
   value); `server.mongo.mode: embedded|external` (+ external.uriSecret{name,key}). _validate.tpl then
   checks per-source required/forbidden fields. Password complexity via separate regexMatch checks OK
   (as built). validation.yaml must include the validator unconditionally (as built).
3. **Digest-first image contract + honest externalOnly split.**
   - `image.{repository,tag,digest,allowMutableTag:false}`; digest = deploy identity; render
     `repo@digest` when set else `repo:tag`; FAIL validation unless digest set OR allowMutableTag=true.
   - `infra.externalOnly.{enabled, enforcement: compatibility|patched}`. compatibility = legacy image,
     README says env not enforced upstream; patched = Phase-5 image, requires an immutable digest.
Additional adopted points:
- **Q5 secrets: FAIL-CLOSED by default** (default source=existingSecret), auto-generate only as explicit
  opt-in (`source: generate` + `generatedSecret.retain` + `lookup` on LIVE helm ops; REJECT/warn under
  GitOps — randAlphaNum drifts, repo-server has no cluster lookup). dfproj key stricter: persist org_id+
  project_id+key to a dedicated Secret immediately, read-first on retry, fail-closed if remote-created
  but persist-failed, never log the key.
- **Q6: add provisioning gate** — `provisioning.enabled=true` ⇒ ≥1 enabled native OR external backend
  (unless provisioning.models explicitly empty), else the suite provisions a project with no usable
  model (opaque failure). `provisioning.enabled` default should be explicit, ideally **false**.
- 10 missing values-contract categories to add before templates: image/pull, secret interface, backend
  interface (incl. GET /v1/models collision note), persistence, scheduling/GPU, security/network,
  provisioning (incl. deterministic `provisioning.revision`), health/rollout, observability,
  compatibility declaration (tested Server/Infra digest pair + DF_MESH_KEY-mandatory legacy note).

### ⚠️ REVIEWER CONFLICT (advisor vs Hermes) — USER DECISION
advisor (Phase-1 round 1) required **default `helm template` to PASS** → drove the auto-generate-when-
empty default. Hermes requires **fail-closed default** (source=existingSecret; digest-or-allowMutableTag
gate) → default render FAILS until the operator supplies secrets/digest. Both legitimate: dev-friendly
default-renders vs GitOps fail-closed/explicit-acknowledgment. This is the chart's contract PHILOSOPHY —
user's call before the Phase-1 values are reworked.

## Open facts for BUILD phase (not blocking design)
- df_abuse_model value shape (scalar vs list) — verify via pulled server image or trial.
- workspace app-mongo replSet rs0 init in k8s — small init job (rs.initiate).
- CPU-only llama.cpp ~11s/call (SLIM doc) — inherent; document; concurrency via --parallel.
