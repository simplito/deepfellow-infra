# Chart style guide

How this chart is put together, and the rules that keep it that way. Most of them exist because
breaking them broke a real install; where that is the case the reason is given, so a rule can be
dropped once its reason no longer holds.

Run every check with `ci/test.sh` (or `just helm-test` from the repo root) before pushing. CI runs
the same script.

## Layout

```
templates/
  <component>/<kind>.yaml     one Kubernetes object kind per file
  _<topic>.tpl                helpers, grouped by topic (no resources)
  validation.yaml             runs _validate.tpl; renders nothing
  ingress.yaml, serviceaccount.yaml, networkpolicy-default-deny.yaml   release-wide objects
tests/*_test.yaml             helm-unittest suites
ci/*-values.yaml              render fixtures (also used by the tests)
ci/test.sh                    every check CI runs
```

- **One kind per file, one directory per component.** `server/deployment.yaml`,
  `server/service.yaml`, `server/networkpolicy.yaml`. A component's NetworkPolicy lives with the
  component, not in a shared file, so adding or removing a component touches one directory.
- **A file with a `range` over entries emits `---` at the start of each iteration**
  (`model-backend/*`, `provisioning/external-backend-secrets.yaml`). Without it the second entry
  merges into the first document.
- **A file emits only its own object.** Variables a file needs are computed in that file; a shared
  preamble must not also emit (the `with $sec` block that renders a Secret belongs in `secret.yaml`
  only). Splitting a file once emitted the same Secret from five files; `ci/check-duplicates.py`
  now fails on that.

| Helper file | Holds |
| --- | --- |
| `_names.tpl` | names, labels, selector labels |
| `_components.tpl` | whether a component renders (`deepfellow.<component>.inChart`) |
| `_images.tpl` | image references, pull secrets |
| `_endpoints.tpl` | the port every component listens on, and `host:port` addresses |
| `_pod.tpl` | pod metadata, ServiceAccount, security contexts, placement, waits (`deepfellow.retry`), volumes |
| `_probes.tpl` | every workload's default probes |
| `_credentials.tpl` | the credential contract and chart-managed Secrets |
| `_mongo.tpl` | what both MongoDBs share |
| `_backends.tpl` | defaults and normalisation of `modelBackends` / `externalBackends` entries |
| `_provisioning.tpl` | provisioning names and the backend registration plan |
| `_validate.tpl` | render-time validation, one `deepfellow.validate.<component>` block each |

## Conditions

- **Guard every file of a component on the same predicate.** Compound conditions live once in
  `_components.tpl` (`{{- if include "deepfellow.qdrant.inChart" . }}`); a single boolean such as
  `.Values.workspace.enabled` may be used directly. Anything that depends on a component (a wait,
  an env var, a NetworkPolicy peer) uses the same predicate, so they cannot disagree about whether
  it exists.

## Values

- **Keep `values.yaml` short.** Leave out empty strings and optional leaves: show them as
  commented examples (`# storageClass: fast-ssd`). Keep a parent map (`image: {}`) when templates
  read keys under it. A template must treat every optional leaf as possibly missing (`default`,
  `with`). The schema still lists every key, so a commented one is valid once uncommented.
- **Every key has a `# --` comment.** The README's values table is generated from them by
  helm-docs; CI fails if `README.md` is stale. Edit `README.md.gotmpl`, never `README.md`. A parent
  object whose children are documented gets a plain `#` comment, or it appears twice.
- **The schema is strict** (`additionalProperties: false` everywhere). A new key needs a schema
  entry, or `helm lint` fails on the default values. Every workload block lists `resources`,
  `nodeSelector`, `tolerations` and `affinity` in `values.yaml`, because users look for them there.
  The other keys every workload block accepts (security contexts, probes, pod labels and
  annotations) are documented once, in the README and the schema, not repeated per component.
- **The schema owns shapes, enums and single-field formats; `_validate.tpl` owns the rest**: rules
  that compare fields and the fail-closed contract. Do not check the same thing in both. Messages in
  `_validate.tpl` name the full values path they are about.
- **Map entries do not inherit defaults.** `modelBackends.<name>` and `externalBackends.<name>` get
  nothing from `values.yaml`; they go through `deepfellow.normalizedBackend` /
  `deepfellow.normalizedExternalBackend`, which merge the defaults in `_backends.tpl`. Read an entry
  only after normalising it. A credential in a map entry once crashed the render with a nil pointer
  because nothing filled in its shape.
- **`dig` with a default only applies when the key is missing**, not when it is `""`. For a value
  that may be empty, use `default <fallback> <value>`.
- **No dead keys.** A key that no template reads is removed, not documented.

## Credentials

- **One vocabulary, in values and in templates.** A credential takes one source, `secretRef`
  (the name of an existing Secret), `plaintext` (the raw value) or `generate: true`, plus
  `secretKey`, the key inside whichever Secret holds it.
  Templates never read values directly: they take `$cred := include "deepfellow.credentials" . | fromYaml`
  and use `$cred.<values path>`, the same credential with every field filled in and
  `source: secretRef | plaintext | generate | none`. Pass it to `deepfellow.credEnv` (env from the
  right Secret), `deepfellow.chartSecret` (the component's own Secret) and `deepfellow.podMetadata`.
- **A new credential is one line in `deepfellow.credentialKeys`** (its values path and default
  secretKey) and a block in `values.yaml` showing that `secretKey`.
  Never change an existing key: the chart's Secret keeps generated values under it. Credentials in
  map entries (`hfToken`, `apiKey`) are normalized by the backend normalizers instead.
- **Nothing sensitive is ever required in values.** Anything a leak would hurt (keys, passwords,
  tokens, the mesh key) is a registered credential and accepts a `secretRef`. A plain string
  field in `values.yaml` must be safe to commit, and no credential reaches a pod except through a
  Secret reference (`ci/check-no-leaked-credentials.py` checks the render).
- **Hash credentials as the user wrote them, never rendered Secrets.** `generate` re-randomises
  wherever `lookup` is empty (`helm template`, GitOps); a pod template that hashed the rendered
  Secret would roll on every sync. `checksum/credentials` goes through `deepfellow.podMetadata`.

## Pods

Every workload's pod spec is built from the same helpers, in this order:

```yaml
  template:
    metadata:
      {{- include "deepfellow.podMetadata" (dict "context" . "values" $x "component" "server" "credentials" (list ...)) | nindent 6 }}
    spec:
      {{- include "deepfellow.imagePullSecrets" . | nindent 6 }}
      serviceAccountName: {{ include "deepfellow.serviceAccountName" . }}
      automountServiceAccountToken: false
      securityContext:
        {{- include "deepfellow.podSecurityContext" (dict "context" . "values" $x) | nindent 8 }}
      {{- include "deepfellow.podPlacement" $x | nindent 6 }}
      containers:
        - securityContext:
            {{- include "deepfellow.containerSecurityContext" (dict "context" . "values" $x) | nindent 12 }}
          {{- include "deepfellow.probes" (dict "defaults" (include "deepfellow.probeDefaults.<c>" . | fromYaml) "values" $x) | nindent 10 }}
```

`$x` is the component's values block. A component that genuinely differs passes a `base` layer
(MongoDB's uid 999, Milvus' supplementary group 0) instead of hard-coding a context.

- **Overrides replace top-level keys; they do not deep-merge.** `mergeOverwrite` skips `false` and
  `0`, so it can never express `runAsNonRoot: false`. The layering helpers use `set`.
- **Probe defaults live in `_probes.tpl`**, one `deepfellow.probeDefaults.<component>` each. A
  handler in an override replaces the default handler; the helper removes the old one.
- **Only the provisioning Job mounts a ServiceAccount token.** It is the only pod that calls the
  Kubernetes API, through a Role scoped to its one Secret.
- **A pod that depends on another in-chart component waits for it** with `deepfellow.waitFor` (an
  init container polling a health URL). The Service it polls must publish that port; the Server
  once waited on a Milvus port its Service did not expose, and the install hung.
- **Every wait is `deepfellow.retry`**: a `check` command and a `what` for the log, with the budget
  from `dependencyWait`. Write a new wait as a check, not as another loop.
- **Ports and addresses come from `_endpoints.tpl`** (`include "deepfellow.port" "qdrant.http"`,
  `deepfellow.address`), never as literals, so a container, its Service, its NetworkPolicy and the
  components that call it cannot disagree. Env values must stay strings: `| quote` a port.
- **A backing store has a readiness probe**, or dependants start against it too early.

## Jobs

- **Jobs are ordinary release resources, not hooks.** As post-install hooks they deadlocked
  `helm install --wait`: Helm waits for Ready before running hooks, and the Workspace cannot become
  Ready until the Jobs have run.
- **A Job's name is a digest of its rendered pod template**, which is immutable. The pod template
  is a `define` in the Job's own file, rendered once for the name and once as the spec, so no
  input can be forgotten. A hand-kept list of inputs was tried and failed on the cluster: a
  changed script kept the old name and the upgrade died with `field is immutable`. Inputs outside
  the pod (the provisioning script in its ConfigMap) are added to the digest explicitly. A Job
  re-runs whenever its pod changes, so what it does must be idempotent.

## Provisioning package

`files/deepfellow_provision/` is a Python package the Job runs as `python -m deepfellow_provision`.
The ConfigMap carries one key per module and is mounted as the package directory, so a new module
needs no template change. It runs in the Server image with only the standard library.

| Module | Holds |
| --- | --- |
| `cli.py` | the reconcile / verify / status actions and the exit contract |
| `config.py` | the environment the chart sets, read once into a frozen `Config` |
| `state.py` | the state Secret, through the Kubernetes API |
| `server.py` | admin, login, organisation / project / key, grants, verification |
| `infra.py` | backend registration and model readiness |
| `transport.py` | JSON over HTTP and health waits |
| `errors.py`, `log.py` | `ProvisionError` and the `[provision]` log lines |

- **Steps raise `ProvisionError`; only `cli.main` turns it into an exit.** The message is the whole
  diagnosis an operator gets, and it never contains a credential.
- **Modules call each other through the module** (`transport.request(...)`, `state.write(...)`),
  so a test patches one attribute and every caller sees it.
- The tests live in `tests/unit/provision/` and run with the repo's pytest, ruff and pyright.

## NetworkPolicy

- **Default-deny covers every pod of the release; each flow gets its own allow**, in the target
  component's directory. Select pods by the selector labels (`app.kubernetes.io/instance` +
  `app.kubernetes.io/component`), never by `part-of`: that label is on the objects, not the pods, and
  a policy selecting on it once matched nothing.

## Images

- **Fail closed.** An image deploys on a digest, or on a tag the user explicitly accepted with
  `allowMutableTag: true`. Images the chart chooses itself (the dependency-wait curl, the llama.cpp
  download helper) are digest-pinned in the chart.
- **`Chart.yaml`'s `artifacthub.io/images` is generated** by `ci/sync-artifacthub-images.py`; CI
  fails if it drifts from `values.yaml`.

## Testing a change

1. `ci/test.sh` must pass.
2. A refactor must not change what renders. Render every `ci/*-values.yaml` fixture before and
   after and compare the objects (not the text), including the number of documents.
3. A bug fix adds a test to `tests/` that fails without the fix.
4. A new branch in a template (a new value, a new `if`) is exercised by at least one `ci/` fixture;
   `ci/edge-values.yaml` collects the unusual ones.
5. Changes to pods, Jobs, NetworkPolicies or probes are proven on a cluster before release:
   install each fixture with NetworkPolicy on, and expect every pod Ready with zero restarts.
