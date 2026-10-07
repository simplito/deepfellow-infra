# deepfellow

Helm chart for the DeepFellow Suite: Infra, Server and Workspace, with their databases and
model backends. It needs no Docker socket. Model backends run as llama.cpp Deployments and are
registered into Infra as external `openai` services. Only the components you enable are rendered.

![Version: 0.1.0](https://img.shields.io/badge/Version-0.1.0-informational?style=flat-square) ![Type: application](https://img.shields.io/badge/Type-application-informational?style=flat-square) ![AppVersion: v0.34.1](https://img.shields.io/badge/AppVersion-v0.34.1-informational?style=flat-square)

## Quickstart

```bash
helm install deepfellow ./deepfellow -f ./deepfellow/values-quickstart.yaml
```

This runs Infra, the Server, the Workspace with its MongoDB and Redis, the Server's MongoDB and
one ~400MB model. The chart generates
every credential and accepts the model backend's image tag. When the provisioning Job finishes,
read the project
key it created:

```bash
kubectl logs -l app.kubernetes.io/component=provisioning --tail=-1
kubectl get secret deepfellow-deepfellow-provisioning-state \
  -o jsonpath='{.data.project-api-key}' | base64 -d
```

MongoDB advises against NFS, and it failed on NFS in testing. If your default StorageClass is
NFS, add:

```bash
--set server.mongo.embedded.storage.storageClass=local-path \
--set workspace.mongo.storage.storageClass=local-path
```

The quickstart file is for trying the chart. For anything else, start from `values.yaml`.

## Defaults fail closed

With plain `values.yaml` the chart refuses to render. You have to choose, for every credential,
where it comes from, and for every image without a digest, `allowMutableTag: true`. A valid
minimum is in [`ci/minimal-values.yaml`](./ci/minimal-values.yaml).

Each credential says where its value comes from, with one of `secretRef`, `plaintext` or
`generate`. `secretKey` is the key inside the Secret that holds it:

```yaml
infra:
  auth:
    adminApiKey:
      secretRef: df-infra             # an existing Secret you manage (ESO, sops, by hand)
      secretKey: admin-key            # the key inside it; default infra-admin-api-key
    apiKey:
      plaintext: s3cr3t               # the raw value, typed into your values file
server:
  admin:
    password:
      generate: true                  # generated on install, kept on uninstall
```

`values.yaml` shows every credential's default `secretKey`. With `plaintext` or `generate` the
chart stores the value in its own Secret, under that same `secretKey`.

You never have to put a sensitive value into a values file: every credential, including the mesh
key, takes a `secretRef`; `plaintext` exists for trials and tests. Values that are not credentials
(names, hosts, model ids) are never secret, and credentials reach pods only through Secret
references, never as a literal in a pod spec or ConfigMap.

`generate` reads the value back with `lookup` on later upgrades, so it needs a live `helm`
command. `helm template` and GitOps renderers such as Argo CD cannot read it back and produce a
new value on every render, which would rotate every password on each sync. Use `secretRef` there.

Values files merge, so an overlay that switches a credential to another source has to unset
the old one, e.g. `plaintext: null` next to the new `secretRef`. Otherwise the chart stops with
"set only one of secretRef, plaintext or generate".

Images:

```yaml
image:
  repository: ...
  tag: latest
  digest: sha256:...       # pin it; used instead of the tag
  # or
  allowMutableTag: true    # accept the tag as it is
  pullPolicy: IfNotPresent # optional
```

Infra, the Server and the Workspace default to pinned releases (tag and digest): Infra v0.34.0,
Server v0.34.1, Workspace v0.5.2. Avoid `latest`: it moves and has lagged far behind the release
tags (Infra's `latest` was v0.29.0 in September 2026).

## How requests flow

```
client -> Server (:8000, project key)
            -> Infra (:8086)
                 -> llama.cpp Deployment   (modelBackends)
                 -> or your own endpoint   (externalBackends)
Workspace (:3000) -> Server
```

Infra gets no Docker socket: the chart always sets `DF_EXTERNAL_ONLY=true`, so it serves only the
backends registered from outside. Upstream images up to v0.34 ignore that variable but boot
without Docker anyway; an image built with the patch in this repository honours it.

## Model backends

```yaml
modelBackends:            # llama.cpp Deployments run by this chart
  llama:
    modelId: my-llm       # defaults to the key
    gguf: { url: "https://.../model.gguf" }
    image:
      allowMutableTag: true

externalBackends:         # endpoints you run elsewhere
  remote:
    apiUrl: http://host:8080   # the root URL; Infra appends v1/
    modelId: my-model
```

Both are maps, so an overlay can add or disable one backend without repeating the others. Each
key becomes its own Infra service instance (`openai|<key>`), so keys must be unique across both
maps. An external endpoint must not list its `modelId` in `GET /v1/models`: Infra would then seed
the model with `type=None` and the custom-model install fails.

## Requirements

- Kubernetes >=1.25.0-0, Helm 3.8+
- A StorageClass that honours `fsGroup`. MongoDB advises against NFS, and on NFS it failed in
  testing with `Operation not permitted`; give `server.mongo` and `workspace.mongo` block or node-local storage
  and pin them with `nodeSelector` if that storage is node-local.
- GGUF weights reachable over HTTP. Each backend downloads its file in an init container.

The Workspace exits at startup without `workspace.helperModel` (a model id you serve, used for
session titles, suggestions and message verification) and the `workspace.auth.bootstrapAdmin`
name, email and password, so the chart refuses to render without them.

## Validation

Every `helm install`, `template` and `lint` runs two checks:

- `values.schema.json` checks types, enums and formats. It rejects unknown keys, so a typo such as
  `workspace.helpModel` is an error, not a silently ignored setting.
- `templates/_validate.tpl` checks rules that span several fields, with messages that name the
  values path. For example: a `plaintext` `provisioning.admin.password` needs 10+ characters with lower,
  upper, digit and special; a `plaintext` `workspace.mongo.rootPassword` must be URL-safe because
  it goes into a connection string unencoded; `server.mongo.external.address` must be `host:port`, not
  a `mongodb://` URI; provisioning with no enabled backend fails unless `allowNoModels: true`.

## Upgrading

```bash
helm upgrade deepfellow ./deepfellow -f my-values.yaml --wait --wait-for-jobs
```

Use `--wait-for-jobs`, on install too when the Workspace is disabled. `--wait` alone covers the
provisioning Job only on an install with the Workspace, which waits for the Job's output. Without
the Workspace, or on upgrade, helm can report success before the Job has finished or failed.

Both Jobs (provisioning and the Workspace replica-set init) are named after a digest of their pod
template. A change to what they run creates a new Job. Anything else leaves them alone. Both are
safe to re-run.

If an upgrade replaces the Workspace together with the Server or the Workspace's MongoDB, the
Workspace may restart once or twice: its dependency checks can pass against the old pods just
before they are replaced. It recovers by itself and `--wait` still succeeds.

Changing `provisioning.admin.email` later creates a second admin rather than renaming the first.

Infra (v0.33 and later, so the default v0.34.0) reads `DF_INFRA_API_KEY`, the mesh settings, the
Hugging Face token, `DF_NAME` and `DF_INFRA_URL` from the environment only on its first start, then
keeps them in `config.json` on its PVC. Changing those values in the chart later does not reach
Infra: change them through Infra's `/admin/config` API as well. (Infra v0.29 read them on every
start.)

## Storage

Every stateful component can use a PVC you created, through `existingClaim`, and any
StorageClass through `storageClass`: `infra.storage`, `server.storage`,
`modelBackends.<name>.storage`, `server.mongo.embedded.storage`, `workspace.mongo.storage`,
`workspace.redis.storage`, `server.vectorDb.qdrant.embedded.storage`,
`server.vectorDb.milvus.storage`, `.milvus.etcd.storage`, `.milvus.minio.storage` and
`server.falkordb.storage`. The chart does not create, delete or check that claim. On StatefulSets it
replaces the claim template with a plain volume, which is fine because each runs one replica.
A StatefulSet's claim template cannot change after install, so switching an installed
StatefulSet to or from `existingClaim` needs `kubectl delete statefulset <name> --cascade=orphan`
before `helm upgrade` (the pod keeps running and the upgrade recreates the StatefulSet).

The Workspace's Redis (Workspace v0.5 and later need one) keeps its data on a PVC as an
append-only file, so it survives a restart. `workspace.redis.storage.enabled: false` runs it in
memory only; what the Workspace keeps there is then lost when the Redis pod restarts.

`helm uninstall` removes the PVCs of Infra, the Server, the Workspace's Redis and the model
backends. It keeps:

- the provisioning state Secret, on purpose. The Server returns the project API key once, and it
  cannot be read again.
- the PVCs of the StatefulSets (both MongoDBs, Qdrant, FalkorDB, Milvus). Kubernetes never deletes
  them.

A reinstall into the same namespace therefore finds old data and an old project key. For a clean
start delete the namespace, or:

```bash
kubectl -n <ns> delete secret <release>-deepfellow-provisioning-state
kubectl -n <ns> delete pvc -l app.kubernetes.io/instance=<release>
```

To bring your own state Secret instead, set `provisioning.stateSecret.name`. That Secret must
exist before the install; the chart does not create it.

## External MongoDB

```yaml
server:
  mongo:
    mode: external
    database: deepfellow
    username: dfuser
    password:
      secretRef: my-mongo
      secretKey: password
    external:
      address: "mongo.data.svc:27017/?authSource=admin"
```

`database`, `username` and `password` apply in both modes; everything under `embedded` is ignored
here. An external Qdrant works the same way: `server.vectorDb.qdrant.mode: external` and
`external.url`.

The Server builds `mongodb://<username>:<password>@<url>` itself, so `url` is host and port plus
options. The user must already exist with read and write access to `database`.

## Pods

Pods run as non-root with a `RuntimeDefault` seccomp profile and all capabilities dropped. Only
the provisioning Job mounts a ServiceAccount token; its Role can touch nothing but the state
Secret. The MongoDB pods run as the image's uid 999, and Milvus gets supplementary group 0 because
its binaries are `root:root 0774`.

Each workload block (`infra`, `server`, `workspace`, `provisioning`, `server.mongo.embedded`,
`workspace.mongo`, `server.vectorDb.qdrant.embedded`, `server.vectorDb.milvus` with its `etcd` and `minio`
blocks, `server.falkordb`, and each `modelBackends.<name>`) also accepts:

| Key | Behaviour |
| --- | --- |
| `podSecurityContext`, `containerSecurityContext` | Applied over the chart-wide values of the same name. Top-level keys replace, so `runAsNonRoot: false` works. |
| `readinessProbe`, `livenessProbe`, `startupProbe` | Applied over the default probe. A new handler (`httpGet`, `exec`, ...) replaces the default one; `enabled: false` removes the probe. |
| `podLabels`, `podAnnotations` | Added to the chart-wide ones. Selector labels cannot be overridden. |
| `resources`, `nodeSelector`, `tolerations`, `affinity` | The usual. |

Changing a `plaintext` credential restarts the pods that read it. A rotated Secret behind a
`secretRef` reaches the pods on their next restart.

## Values

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| containerSecurityContext | object | `{"allowPrivilegeEscalation":false,"capabilities":{"drop":["ALL"]}}` | Container securityContext for every container, init containers included. A component's own `containerSecurityContext` replaces individual top-level keys. |
| dependencyWait.attempts | int | `100` | Checks before giving up. |
| dependencyWait.delaySeconds | int | `3` | Seconds between checks. |
| dependencyWait.image | object | `{"digest":"sha256:c1fe1679c34d9784c1b0d1e5f62ac0a79fca01fb6377cdd33e90473c6f9f9a69","repository":"curlimages/curl","tag":"8.11.1"}` | Image for the HTTP health checks. The chart chose it, so it is pinned. |
| externalBackends | object | `{}` | Bring-your-own OpenAI-compatible endpoints, keyed by name. |
| global.imagePullSecrets | list | `[]` | Image pull secrets for all pods. The default images pull anonymously. |
| infra.affinity | object | `{}` | Affinity for Infra. |
| infra.auth.adminApiKey | object | `{"secretKey":"infra-admin-api-key"}` | Admin API key (`DF_INFRA_ADMIN_API_KEY`). Required. |
| infra.auth.serverApiKey | object | `{"secretKey":"infra-api-key"}` | API key the Server uses to call Infra (`DF_INFRA_API_KEY`). Required. Infra reads it (like the mesh settings and the Hugging Face token) from env only on its first start; see README, Upgrading. |
| infra.huggingFaceToken | object | `{"secretKey":"hf-token"}` | Hugging Face token for gated downloads (`DF_HUGGING_FACE_TOKEN`). Optional. |
| infra.image | object | `{"digest":"sha256:a44b95741d0ee042b05df13c6bdc47c3f73c1152471396a34df1d56de23970de","repository":"hub.simplito.com/deepfellow/deepfellow-infra","tag":"v0.34.0"}` | Infra image, pinned to a release. The digest decides what runs; the tag only names the release. Upgrade by changing both, so they keep matching. |
| infra.mesh.allowChildrenFrom | list | `[]` | With `networkPolicy.enabled`, only this release's Server and provisioning Job reach Infra. List NetworkPolicy peers (`namespaceSelector`, `podSelector`, `ipBlock`) that child Infras connect from. Requires `childKey`. Children outside the cluster also need a reachable Service (`infra.service.type`). |
| infra.mesh.childKey | object | `{"secretKey":"mesh-child-key"}` | Key child Infras must present to connect to THIS one (`DF_MESH_KEY`). Optional. Unset, Infra gets a fixed placeholder (Infra v0.29 would not start without a value), so anything that can reach Infra's port could connect as a child Infra: set a key, or keep `networkPolicy.enabled`, before exposing Infra. |
| infra.mesh.parent.enabled | bool | `false` | Connect to a parent Infra. |
| infra.mesh.parent.key | object | `{"secretKey":"mesh-parent-key"}` | Key this Infra presents to the parent (`DF_CONNECT_TO_MESH_KEY`). Required when enabled. |
| infra.nodeSelector | object | `{}` | Node selector for Infra. |
| infra.resources | object | `{}` | Container resources for Infra. |
| infra.service.port | int | `8086` | Port of the Infra Service. The container always listens on 8086. |
| infra.service.type | string | `"ClusterIP"` | Infra Service type. |
| infra.storage.mountPath | string | `"/app/storage"` | Mount path (`DF_STORAGE_DIR`). |
| infra.storage.size | string | `"1Gi"` | PVC for Infra's state (config and endpoint registry, no model weights). |
| infra.tolerations | list | `[]` | Tolerations for Infra. |
| ingress.annotations | object | `{}` | Annotations for the Ingress (e.g. cert-manager). |
| ingress.enabled | bool | `false` | Create the Ingress. |
| ingress.server.path | string | `"/v1"` | Path prefix routed to the Server. |
| ingress.tls | list | `[]` | TLS blocks (`hosts` + `secretName`), passed through as-is. |
| ingress.workspace | object | `{}` |  |
| modelBackends | object | `{}` | llama.cpp backends, keyed by name. The key names the Deployment and is the default model id. |
| networkPolicy.enabled | bool | `false` | Create the NetworkPolicies. |
| podAnnotations | object | `{}` | Annotations for every pod, Jobs included. A component's own `podAnnotations` adds to these. |
| podLabels | object | `{}` | Labels for every pod. A component's own `podLabels` adds to these. Selector labels win. |
| podSecurityContext | object | `{"fsGroup":1000,"runAsGroup":1000,"runAsNonRoot":true,"runAsUser":1000,"seccompProfile":{"type":"RuntimeDefault"}}` | Pod securityContext for every workload. A component's own `podSecurityContext` replaces individual top-level keys. MongoDB keeps uid/gid 999 and Milvus adds group 0. |
| provisioning.admin.email | string | `"admin@deepfellow.local"` | Admin email (the login). |
| provisioning.admin.name | string | `"admin"` | Admin display name. |
| provisioning.admin.password | object | `{"secretKey":"admin-password"}` | Admin password. Required. A `plaintext` value needs 10+ characters with a lowercase letter, an uppercase letter, a digit and a special character. |
| provisioning.affinity | object | `{}` | Affinity for the provisioning Job. |
| provisioning.allowNoModels | bool | `false` | Allow provisioning with no enabled backend (the project would have no model). |
| provisioning.enabled | bool | `false` | Run the provisioning Job. |
| provisioning.image | object | `{}` |  |
| provisioning.nodeSelector | object | `{}` | Node selector for the provisioning Job. |
| provisioning.project.apiKeyName | string | `"app"` | Name of the project API key, minted into the state Secret. |
| provisioning.project.name | string | `"Default"` | Project name, in that organisation. |
| provisioning.project.organization | string | `"Workspace"` | Organisation name. |
| provisioning.resources | object | `{}` | Container resources for the provisioning Job. |
| provisioning.revision | string | `"1"` | Increment to force the provisioning Job to run again. Not needed after a change to what it runs or reads (backends, credentials, the package): that already starts a new Job. |
| provisioning.serviceAccount.create | bool | `true` | Create the Job's ServiceAccount, with a Role limited to the state Secret. |
| provisioning.stateSecret | object | `{}` |  |
| provisioning.tolerations | list | `[]` | Tolerations for the provisioning Job. |
| server.affinity | object | `{}` | Affinity for the Server. |
| server.falkordb.affinity | object | `{}` | Affinity for FalkorDB. |
| server.falkordb.enabled | bool | `false` | Deploy the FalkorDB graph store. |
| server.falkordb.image | object | `{"digest":"sha256:22fbac7f4f985a54f3d84368c75df6d8f31cf47bbc590690484f5eb357deb8d1","repository":"falkordb/falkordb","tag":"v4.12.4"}` | FalkorDB image. |
| server.falkordb.nodeSelector | object | `{}` | Node selector for FalkorDB. |
| server.falkordb.password | object | `{"secretKey":"falkordb-password"}` | FalkorDB password. Required when enabled. |
| server.falkordb.resources | object | `{}` | Container resources for FalkorDB. |
| server.falkordb.storage.size | string | `"5Gi"` | PVC size for FalkorDB. |
| server.falkordb.tolerations | list | `[]` | Tolerations for FalkorDB. |
| server.image | object | `{"digest":"sha256:0d8795192a37ed65deadb2ed24460d18791028a61102c42cea068386ed5849fc","repository":"hub.simplito.com/deepfellow/deepfellow-server","tag":"v0.34.1"}` | Server image, pinned to a release. The digest decides what runs; the tag only names the release. Upgrade by changing both, so they keep matching. |
| server.logLevel | string | `"INFO"` | Server log level (`DF_LOG_LEVEL`). |
| server.metrics.basicAuth.enabled | bool | `false` | Protect the metrics endpoint with basic auth. |
| server.metrics.basicAuth.password | object | `{"secretKey":"metrics-password"}` | Metrics password. Required when enabled. |
| server.mongo.database | string | `"deepfellow"` | Database the Server uses (`DF_MONGO_DB`). |
| server.mongo.embedded.affinity | object | `{}` | Affinity for the Server's MongoDB. |
| server.mongo.embedded.image | object | `{"digest":"sha256:7abfba0d07c9330373f8173981ea4d09cd8a82cdf0e86ccaf7008848d1d24f62","repository":"mongo","tag":"8.2.7"}` | MongoDB image. |
| server.mongo.embedded.nodeSelector | object | `{}` | Node selector for the Server's MongoDB. Pin it when its storage is node-local. |
| server.mongo.embedded.resources | object | `{}` | Container resources for the Server's MongoDB. |
| server.mongo.embedded.rootPassword | object | `{"secretKey":"mongo-root-password"}` | Root password. Required. |
| server.mongo.embedded.rootUsername | string | `"root"` | Root user (`MONGO_INITDB_ROOT_USERNAME`). |
| server.mongo.embedded.storage.size | string | `"8Gi"` | PVC size. Avoid NFS: MongoDB advises against it, and it failed on NFS in testing. |
| server.mongo.embedded.tolerations | list | `[]` | Tolerations for the Server's MongoDB. |
| server.mongo.external | object | `{}` |  |
| server.mongo.mode | string | `"embedded"` | `embedded`: the chart runs MongoDB, configured under `embedded`. `external`: the Server uses your MongoDB at `external.address`. |
| server.mongo.password | object | `{"secretKey":"mongo-password"}` | That user's password. Required. |
| server.mongo.username | string | `"deepfellow"` | User the Server connects as (`DF_MONGO_USER`). Embedded: the chart creates it. External: it must exist, with read/write on `database`. |
| server.nodeSelector | object | `{}` | Node selector for the Server. |
| server.otel.enabled | bool | `false` | Export OpenTelemetry. |
| server.resources | object | `{}` | Container resources for the Server. |
| server.service.port | int | `8000` | Port of the Server Service. The container always listens on 8000. |
| server.service.type | string | `"ClusterIP"` | Service type for the Server. |
| server.storage.size | string | `"5Gi"` | PVC for uploaded files and the plugins directory. |
| server.tolerations | list | `[]` | Tolerations for the Server. |
| server.vectorDb.embedding.dimensions | int | `1024` | Length of the embedding vectors; must match the model. |
| server.vectorDb.embedding.model | string | `"mxbai-embed-large"` | Embedding model id. |
| server.vectorDb.embedding.service | string | `"openai"` | Infra service that serves the embedding model (`DF_VECTOR_DATABASE__EMBEDDING__ENDPOINT`). A service name, not a URL. |
| server.vectorDb.enabled | bool | `false` | Enable the vector store for RAG (`DF_VECTOR_DATABASE__PROVIDER__ACTIVE`). |
| server.vectorDb.milvus.affinity | object | `{}` | Affinity for the Milvus stack. |
| server.vectorDb.milvus.etcd.image | object | `{"digest":"sha256:d0a641d5fbcc89678c931a61b7de7b8a1cf097149f135c9c73bc81d076a1494b","repository":"quay.io/coreos/etcd","tag":"v3.5.18"}` | etcd image for the Milvus stack. |
| server.vectorDb.milvus.etcd.storage.size | string | `"4Gi"` | PVC size for etcd (Milvus metadata). etcd caps its database at 2GiB by default. |
| server.vectorDb.milvus.image | object | `{"digest":"sha256:215400dc74c03393e4c28d808c1efbbbd7f1f46e94ba73b4e583df3ccd81105e","repository":"milvusdb/milvus","tag":"v2.6.2"}` | Milvus image. |
| server.vectorDb.milvus.minio.image | object | `{"digest":"sha256:1dce27c494a16bae114774f1cec295493f3613142713130c2d22dd5696be6ad3","repository":"quay.io/minio/minio","tag":"RELEASE.2024-12-18T13-15-44Z"}` | MinIO image. Docker Hub no longer serves `minio/minio`; Quay has the same release. |
| server.vectorDb.milvus.minio.rootPassword | object | `{"secretKey":"minio-root-password"}` | MinIO root password. Required with Milvus. |
| server.vectorDb.milvus.minio.rootUser | object | `{"secretKey":"minio-root-user"}` | MinIO root user; Milvus logs in with it. Required with Milvus. |
| server.vectorDb.milvus.minio.storage.size | string | `"20Gi"` | PVC size for MinIO, which holds the vectors and index files. |
| server.vectorDb.milvus.nodeSelector | object | `{}` | Node selector for the Milvus stack. |
| server.vectorDb.milvus.resources | object | `{}` | Container resources for each of milvus, etcd and MinIO. |
| server.vectorDb.milvus.storage.size | string | `"20Gi"` | PVC size for Milvus itself (its local write-ahead log and cache). |
| server.vectorDb.milvus.tolerations | list | `[]` | Tolerations for the Milvus stack. |
| server.vectorDb.qdrant.embedded.affinity | object | `{}` | Affinity for Qdrant. |
| server.vectorDb.qdrant.embedded.image | object | `{"digest":"sha256:48c12634a17d8d54f4e3fb95c2b081668039e6e4517ebba57be1882109199ae7","repository":"qdrant/qdrant","tag":"v1.15-unprivileged"}` | Qdrant image. Use an `-unprivileged` tag: the plain one expects root and panics creating `./snapshots/tmp` under the chart's uid. |
| server.vectorDb.qdrant.embedded.nodeSelector | object | `{}` | Node selector for Qdrant. |
| server.vectorDb.qdrant.embedded.resources | object | `{}` | Container resources for Qdrant. |
| server.vectorDb.qdrant.embedded.storage.size | string | `"10Gi"` | PVC size for Qdrant. |
| server.vectorDb.qdrant.embedded.tolerations | list | `[]` | Tolerations for Qdrant. |
| server.vectorDb.qdrant.external | object | `{}` |  |
| server.vectorDb.qdrant.mode | string | `"embedded"` | `embedded`: the chart runs Qdrant, configured under `embedded`. `external`: the Server uses your Qdrant at `external.url`. |
| server.vectorDb.type | string | `"qdrant"` | `qdrant` or `milvus`. |
| serviceAccount.annotations | object | `{}` | Annotations for the ServiceAccount, e.g. a cloud workload-identity binding. |
| serviceAccount.create | bool | `true` | Create it. With `create: false` pods run as `name`, or as the namespace's `default`. |
| workspace.affinity | object | `{}` | Affinity for the Workspace. |
| workspace.auth.bootstrapAdmin.email | string | `"admin@deepfellow.local"` | First admin's email. |
| workspace.auth.bootstrapAdmin.name | string | `"admin"` | First admin's name. The image needs name, email and password together. |
| workspace.auth.bootstrapAdmin.password | object | `{"secretKey":"workspace-admin-password"}` | First admin's password. Required. |
| workspace.auth.sessionSecret | object | `{"secretKey":"workspace-session-secret"}` | Signing secret for sessions (`BETTER_AUTH_SECRET`). Required. |
| workspace.enabled | bool | `true` | Deploy the Workspace. |
| workspace.image | object | `{"digest":"sha256:5b12c43416e77a6fdd38574287fa43be1952d3472c9e68b1debb5c2febc3373c","repository":"hub.simplito.com/deepfellow/deepfellow-workspace","tag":"v0.5.2"}` | Workspace image, pinned to a release. The digest decides what runs; the tag only names the release. Upgrade by changing both, so they keep matching. |
| workspace.mongo.affinity | object | `{}` | Affinity for the Workspace's MongoDB. |
| workspace.mongo.database | string | `"workspace_db"` | Database name (`MONGO_DB_NAME`). |
| workspace.mongo.image | object | `{"digest":"sha256:7abfba0d07c9330373f8173981ea4d09cd8a82cdf0e86ccaf7008848d1d24f62","repository":"mongo","tag":"8.2.7"}` | Image for the Workspace's own MongoDB (a single-node replica set). |
| workspace.mongo.keyFile | object | `{"secretKey":"workspace-mongo-keyfile"}` | Replica-set keyFile (internal authentication between members): 6 to 1024 characters from the base64 alphabet (a plain secret, not base64-encoded data). Required. A changed keyFile takes effect when the pod restarts. |
| workspace.mongo.nodeSelector | object | `{}` | Node selector for the Workspace's MongoDB. Pin it when its storage is node-local. |
| workspace.mongo.replSetName | string | `"rs0"` | Replica-set name. The Workspace needs transactions, hence a replica set. |
| workspace.mongo.resources | object | `{}` | Container resources for the Workspace's MongoDB. |
| workspace.mongo.rootPassword | object | `{"secretKey":"workspace-mongo-root-password"}` | Root password. Required. It goes into a connection string unencoded, so it must be URL-safe whatever its source; the chart can check only a `plaintext` value. |
| workspace.mongo.rootUsername | string | `"root"` | Root user; the Workspace connects as it. |
| workspace.mongo.storage.size | string | `"4Gi"` | PVC size. Avoid NFS: MongoDB advises against it, and it failed on NFS in testing. |
| workspace.mongo.tolerations | list | `[]` | Tolerations for the Workspace's MongoDB. |
| workspace.nodeSelector | object | `{}` | Node selector for the Workspace. |
| workspace.redis.affinity | object | `{}` | Affinity for the Workspace's Redis. |
| workspace.redis.image | object | `{"digest":"sha256:72cedd9603038893af961e90ac5e1a1a0d8377d5e338dbdad8fe284ea25de18f","repository":"redis","tag":"8.10.2-alpine"}` | Redis image for the Workspace. |
| workspace.redis.nodeSelector | object | `{}` | Node selector for the Workspace's Redis. |
| workspace.redis.password | object | `{"secretKey":"workspace-redis-password"}` | Redis password. Required. It goes into `REDIS_URL` unencoded, so it must be URL-safe whatever its source; the chart can check only a `plaintext` value. |
| workspace.redis.resources | object | `{}` | Container resources for the Workspace's Redis. |
| workspace.redis.storage.enabled | bool | `true` | Keep Redis data on a PVC (append-only file, fsynced every second). Off: it lives in memory only and is lost when the Redis pod restarts. |
| workspace.redis.storage.size | string | `"1Gi"` | PVC size. |
| workspace.redis.tolerations | list | `[]` | Tolerations for the Workspace's Redis. |
| workspace.resources | object | `{}` | Container resources for the Workspace. |
| workspace.service.port | int | `3000` | Port of the Workspace Service. The container always listens on 3000. |
| workspace.service.type | string | `"ClusterIP"` | Service type for the Workspace. |
| workspace.tolerations | list | `[]` | Tolerations for the Workspace. |

## Developing the chart

See [STYLEGUIDE.md](./STYLEGUIDE.md). CI runs `ci/test.sh`; run it locally before pushing.
