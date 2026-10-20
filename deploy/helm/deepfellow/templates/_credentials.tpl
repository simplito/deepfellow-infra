{{/*
Credentials.

In values a credential says where its value comes from, with one of:
  secretRef: <Secret name>   take it from an existing Secret you manage (nothing sensitive in values)
  plaintext: <string>        the RAW value, written into values; the chart copies it into its own Secret
  generate: true             generated on install, read back with lookup, kept on uninstall
and optionally
  secretKey: <key>           the key inside the Secret that holds it: yours for secretRef, the chart's
                             otherwise. Defaults to the key registered in deepfellow.credentialKeys.
No source means "not set": fine for optional credentials, a render error for required ones.

Templates read the same names, normalized so every field is present:
  source: secretRef | plaintext | generate | none
  path: <values path>          for error messages
  secretRef: <Secret name>     empty unless source=secretRef
  secretKey: <key>             always set
  plaintext: <string>          empty unless source=plaintext
Get them all with `$cred := include "deepfellow.credentials" . | fromYaml`, then pass
`$cred.<values path>` to credEnv, chartSecret, the validators and podMetadata.
*/}}

{{/*
Name of the chart's own Secret for a component's plaintext and generated credentials.
  {{ include "deepfellow.credSecretName" (dict "context" . "component" "infra") }}
*/}}
{{- define "deepfellow.credSecretName" -}}
{{- printf "%s-creds" (include "deepfellow.componentName" (dict "context" .context "component" .component)) -}}
{{- end -}}

{{/*
Random password that passes the Server's create_admin policy (10+ characters with a lowercase and
an uppercase letter, a digit and a special character). randAlphaNum alone has no special character,
so one of each class is appended. The default generator of chartSecret.
*/}}
{{- define "deepfellow.genSecretValue" -}}
{{- printf "%sAa1!" (randAlphaNum 28) -}}
{{- end -}}

{{/*
Random MongoDB replica-set keyFile: base64 characters only (randAlphaNum is a subset), well within
Mongo's 6-1024 length limit. No password suffix: "!" is not a valid keyFile character.
*/}}
{{- define "deepfellow.genKeyFileValue" -}}
{{- randAlphaNum 96 -}}
{{- end -}}

{{/*
One env-var list entry that reads a normalized credential from its Secret: the user's Secret for
secretRef, the component's chart Secret for plaintext and generate. Emits nothing for an unset
credential.

`required` repeats the check in _validate.tpl on purpose. Helm renders templates in
subdirectories before templates/validation.yaml, so without it an unset credential would first
surface here, as an env var with no source, instead of as the readable message below.
  {{- include "deepfellow.credEnv" (dict "name" "DF_X" "cred" $cred.x.y "secretName" $secretName "required" true) | nindent 12 }}
*/}}
{{- define "deepfellow.credEnv" -}}
{{- $c := .cred -}}
{{- if and .required (eq $c.source "none") -}}
{{- fail (printf "%s is required: set one of secretRef, plaintext or generate." (default .name $c.path)) -}}
{{- end -}}
{{- if ne $c.source "none" -}}
- name: {{ .name }}
  valueFrom:
    secretKeyRef:
      name: {{ ternary $c.secretRef .secretName (eq $c.source "secretRef") }}
      key: {{ $c.secretKey }}
{{- end -}}
{{- end -}}

{{/*
The component's chart Secret, holding every plaintext and generated credential in `creds` (a list
of normalized credentials) under its registered key. Renders nothing when none of them is
plaintext or generated.

A generated value is read back from the live Secret with lookup, so upgrades keep it. When there
is none yet (a fresh install, or an upgrade that adds a credential) it is minted with `generator`
(default deepfellow.genSecretValue). A render without cluster access sees nothing through lookup:
an offline upgrade (helm template --is-upgrade) fails here, but an offline install cannot be told
apart from a real one (a fresh namespace has no Secrets yet either), so `helm template` and GitOps
renderers get a NEW value on every render. Under Argo CD that rotates every password on each sync,
which is why GitOps needs secretRef. A Secret holding a generated value is kept on uninstall, since
losing it would orphan the data it protects.
  {{ include "deepfellow.chartSecret" (dict "context" . "secretName" $s "creds" (list $cred.x.a $cred.x.b)) }}
*/}}
{{- define "deepfellow.chartSecret" -}}
{{- $ctx := .context -}}
{{- $secretName := .secretName -}}
{{- $gen := .generator | default "deepfellow.genSecretValue" -}}
{{- $data := dict -}}
{{- $keep := false -}}
{{- range $c := .creds -}}
{{- $key := $c.secretKey -}}
{{- if eq $c.source "plaintext" -}}
{{- $data = merge $data (dict $key ($c.plaintext | b64enc)) -}}
{{- else if eq $c.source "generate" -}}
{{- $keep = true -}}
{{- $existing := lookup "v1" "Secret" $ctx.Release.Namespace $secretName -}}
{{- $val := "" -}}
{{- if and $existing (hasKey ($existing.data | default dict) $key) -}}
{{- $val = index $existing.data $key | b64dec -}}
{{- else if or $ctx.Release.IsInstall (include "deepfellow.clusterReachable" $ctx) -}}
{{- $val = include $gen $ctx -}}
{{- else -}}
{{- fail (printf "%s: generated value %q cannot be read back: this render has no cluster access (helm template, GitOps). Use secretRef, or run helm install/upgrade against the cluster." $c.path $key) -}}
{{- end -}}
{{- $data = merge $data (dict $key ($val | b64enc)) -}}
{{- end -}}
{{- end -}}
{{- if gt (len $data) 0 -}}
apiVersion: v1
kind: Secret
metadata:
  name: {{ $secretName }}
  labels:
    {{- include "deepfellow.labels" $ctx | nindent 4 }}
{{- if $keep }}
  annotations:
    helm.sh/resource-policy: keep
{{- end }}
type: Opaque
data:
{{- range $k, $v := $data }}
  {{ $k }}: {{ $v }}
{{- end }}
{{- end -}}
{{- end -}}

{{/*
"true" when this render can see the cluster. Online, lookup of the release namespace's Secrets
always finds at least Helm's own release record; offline (helm template, GitOps renderers) every
lookup is empty. Listing Secrets needs no permission beyond what Helm itself uses.
*/}}
{{- define "deepfellow.clusterReachable" -}}
{{- if (lookup "v1" "Secret" .Release.Namespace "").items -}}true{{- end -}}
{{- end -}}

{{/*
The registry: every chart credential's full values path, and the Secret key it lives under. That
key is the default `secretKey`: where the value sits in the user's Secret (secretRef) or in the
chart's own. Never change one here: the chart Secret keeps generated values under it, and an
upgrade would mint new ones (the same goes for a user changing secretKey on a generated value).
A new credential needs one line here.
Credentials inside modelBackends / externalBackends entries are normalized by their backend
normalizer in _backends.tpl instead, since their paths contain the user's map keys.
*/}}
{{- define "deepfellow.credentialKeys" -}}
infra.auth.adminApiKey: infra-admin-api-key
infra.auth.serverApiKey: infra-api-key
infra.huggingFaceToken: hf-token
infra.mesh.childKey: mesh-child-key
infra.mesh.parent.key: mesh-parent-key
server.mongo.password: mongo-password
server.mongo.embedded.rootPassword: mongo-root-password
server.vectorDb.milvus.minio.rootUser: minio-root-user
server.vectorDb.milvus.minio.rootPassword: minio-root-password
server.falkordb.password: falkordb-password
server.metrics.basicAuth.password: metrics-password
workspace.auth.sessionSecret: workspace-session-secret
workspace.auth.bootstrapAdmin.password: workspace-admin-password
workspace.mongo.rootPassword: workspace-mongo-root-password
workspace.mongo.keyFile: workspace-mongo-keyfile
workspace.redis.password: workspace-redis-password
provisioning.admin.password: admin-password
{{- end -}}

{{/*
One credential from values, normalized. Fails when more than one source is set.
  {{- $c := include "deepfellow.normalizeCredential" (dict "cred" $raw "key" "hf-token" "path" "x.y") | fromYaml }}
*/}}
{{- define "deepfellow.normalizeCredential" -}}
{{- $c := .cred | default dict -}}
{{- $set := list -}}
{{- if $c.secretRef -}}{{- $set = append $set "secretRef" -}}{{- end -}}
{{- if $c.plaintext -}}{{- $set = append $set "plaintext" -}}{{- end -}}
{{- if $c.generate -}}{{- $set = append $set "generate" -}}{{- end -}}
{{- if gt (len $set) 1 -}}
{{- fail (printf "%s: set only one of secretRef, plaintext or generate (got %s). To switch in a values overlay, set the old one to null." .path (join ", " $set)) -}}
{{- end -}}
source: {{ default "none" (first $set) }}
path: {{ .path | quote }}
secretRef: {{ $c.secretRef | default "" | quote }}
secretKey: {{ $c.secretKey | default .key | quote }}
plaintext: {{ $c.plaintext | default "" | quote }}
{{- end -}}

{{/*
Every registered credential, normalized, nested like values:
  {{- $cred := include "deepfellow.credentials" . | fromYaml }}
  ... "cred" $cred.infra.auth.adminApiKey ...
It walks each registered dotted path through .Values (a missing level reads as unset) and builds
the same nesting in the result.
*/}}
{{- define "deepfellow.credentials" -}}
{{- $root := . -}}
{{- $out := dict -}}
{{- range $path, $key := include "deepfellow.credentialKeys" . | fromYaml -}}
{{- $parts := splitList "." $path -}}
{{- $raw := $root.Values -}}
{{- range $p := $parts -}}{{- $raw = index (default (dict) $raw) $p -}}{{- end -}}
{{- $norm := include "deepfellow.normalizeCredential" (dict "cred" $raw "key" $key "path" $path) | fromYaml -}}
{{- $cur := $out -}}
{{- range $p := initial $parts -}}
{{- if not (hasKey $cur $p) -}}{{- $_ := set $cur $p (dict) -}}{{- end -}}
{{- $cur = index $cur $p -}}
{{- end -}}
{{- $_ := set $cur (last $parts) $norm -}}
{{- end -}}
{{- toYaml $out -}}
{{- end -}}
