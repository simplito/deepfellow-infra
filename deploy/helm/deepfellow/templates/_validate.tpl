{{/*
Render-time validation, run from templates/validation.yaml by every helm install, template and
lint. values.schema.json owns shapes, enums and single-field formats; this file owns what a schema
cannot say, or cannot say clearly: rules that compare fields, the fail-closed image and credential
contract, and password policy. Every message names the values path it is about. Do not re-check
here what the schema already rejects.

deepfellow.validate runs one block per component, in the order a reader meets them in values.yaml.
*/}}

{{- define "deepfellow.validate" -}}
{{- include "deepfellow.validate.infra" . -}}
{{- include "deepfellow.validate.backends" . -}}
{{- include "deepfellow.validate.server" . -}}
{{- include "deepfellow.validate.workspace" . -}}
{{- include "deepfellow.validate.provisioning" . -}}
{{- include "deepfellow.validate.ingress" . -}}
{{- include "deepfellow.validate.names" . -}}
{{- end -}}

{{/* ------------------------------------------------------------------ building blocks */}}

{{/* An image deploys on a digest, or on a tag the user accepted with allowMutableTag. */}}
{{- define "deepfellow.validateImage" -}}
{{- $img := .image -}}
{{- if not $img.digest -}}
{{- if not $img.allowMutableTag -}}
{{- fail (printf "%s: set an image digest, or acknowledge the mutable tag with %s.allowMutableTag=true (fail-closed)." .path .path) -}}
{{- end -}}
{{- if not $img.tag -}}
{{- fail (printf "%s.tag is required when no digest is set (otherwise the image ref is malformed)." .path) -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/* A required credential must be set. Conflicting sources are rejected by normalizeCredential. */}}
{{- define "deepfellow.validateSecret" -}}
{{- if and .required (eq .cred.source "none") -}}
{{- fail (printf "%s is required: set one of secretRef, plaintext or generate." .path) -}}
{{- end -}}
{{- end -}}

{{/*
A credential that must match something outside the chart (an existing account, a token issued
elsewhere, another Infra's key) cannot be generated: a random value would only fail to log in.
*/}}
{{- define "deepfellow.validateNotGenerated" -}}
{{- if eq .cred.source "generate" -}}
{{- fail (printf "%s cannot use generate: it has to match %s. Use secretRef or plaintext." .cred.path .what) -}}
{{- end -}}
{{- end -}}

{{/* The Server's password policy, checkable only for a plaintext value. */}}
{{- define "deepfellow.validatePasswordComplexity" -}}
{{- $c := .cred -}}
{{- if eq $c.source "plaintext" -}}
{{- $p := $c.plaintext -}}
{{- $ok := and (ge (len $p) 10) (regexMatch "[a-z]" $p) (regexMatch "[A-Z]" $p) (regexMatch "[0-9]" $p) (regexMatch "[^a-zA-Z0-9]" $p) -}}
{{- if not $ok -}}
{{- fail (printf "%s.plaintext must have 10+ characters including a lowercase letter, an uppercase letter, a digit and a special character." .path) -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/*
A plaintext password that the chart puts into a connection string (mongodb://user:PASS@host)
unencoded must be URL-safe: unreserved characters plus the URL-safe sub-delims, nothing like
%@:/?# that would break the URL. A generated value is URL-safe by construction; a referenced
Secret cannot be read at render time, so its content is the user's responsibility.
*/}}
{{- define "deepfellow.validateUrlSafeValue" -}}
{{- $c := .cred -}}
{{- if eq $c.source "plaintext" -}}
{{- if not (regexMatch "^[A-Za-z0-9!$&'()*+,._~-]+$" $c.plaintext) -}}
{{- fail (printf "%s.plaintext must be URL-safe (it is placed in a connection-string userinfo unencoded): allowed characters are letters, digits, and !$&'()*+,._~- . Put a password with other characters in a Secret and use secretRef." .path) -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/* Whether a keyed backend entry is enabled (missing `enabled` => true). */}}
{{- define "deepfellow.backendEnabled" -}}
{{- if hasKey .b "enabled" -}}{{- if .b.enabled -}}true{{- end -}}{{- else -}}true{{- end -}}
{{- end -}}

{{/* --------------------------------------------------------------------------- blocks */}}

{{- define "deepfellow.validate.infra" -}}
{{- $cred := include "deepfellow.credentials" . | fromYaml -}}
{{- $i := .Values.infra -}}
{{- include "deepfellow.validateImage" (dict "image" $i.image "path" "infra.image") -}}
{{- include "deepfellow.validateImage" (dict "image" .Values.dependencyWait.image "path" "dependencyWait.image") -}}
{{- include "deepfellow.validateSecret" (dict "cred" $cred.infra.auth.adminApiKey "path" "infra.auth.adminApiKey" "required" true) -}}
{{- include "deepfellow.validateSecret" (dict "cred" $cred.infra.auth.serverApiKey "path" "infra.auth.serverApiKey" "required" true) -}}
{{- include "deepfellow.validateNotGenerated" (dict "cred" $cred.infra.huggingFaceToken "what" "a token issued by Hugging Face") -}}
{{- include "deepfellow.validateNotGenerated" (dict "cred" $cred.infra.mesh.parent.key "what" "the key the parent Infra expects") -}}
{{- if and $i.mesh.allowChildrenFrom (eq $cred.infra.mesh.childKey.source "none") -}}
{{- fail "infra.mesh.allowChildrenFrom opens Infra to child Infras: set infra.mesh.childKey, or any of those peers could join with the placeholder key." -}}
{{- end -}}
{{- if $i.mesh.parent.enabled -}}
{{- include "deepfellow.validateSecret" (dict "cred" $cred.infra.mesh.parent.key "path" "infra.mesh.parent.key" "required" true) -}}
{{- if not $i.mesh.parent.url -}}
{{- fail "infra.mesh.parent.url is required when infra.mesh.parent.enabled=true." -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/*
modelBackends and externalBackends together: each map key is an Infra service instance name and
each modelId a model Infra serves, so both must be unique across the two maps.
*/}}
{{- define "deepfellow.validate.backends" -}}
{{- $modelIds := dict -}}
{{- $instances := dict -}}
{{- range $name, $bRaw := .Values.modelBackends -}}
{{- if include "deepfellow.backendEnabled" (dict "b" $bRaw) -}}
{{- $b := include "deepfellow.normalizedBackend" (dict "context" $ "raw" $bRaw "name" $name) | fromYaml -}}
{{- if not $b.gguf.url -}}
{{- fail (printf "modelBackends.%s.gguf.url is required (the GGUF weights to download)." $name) -}}
{{- end -}}
{{- include "deepfellow.validateImage" (dict "image" $b.image "path" (printf "modelBackends.%s.image" $name)) -}}
{{- include "deepfellow.validateImage" (dict "image" $b.downloadImage "path" (printf "modelBackends.%s.downloadImage" $name)) -}}
{{- include "deepfellow.validateNotGenerated" (dict "cred" $b.gguf.hfToken "what" "a token issued by Hugging Face") -}}
{{- if hasKey $modelIds $b.modelId -}}
{{- fail (printf "duplicate modelId %q (modelBackends.%s and %s): model ids must be unique across modelBackends and externalBackends." $b.modelId $name (index $modelIds $b.modelId)) -}}
{{- end -}}
{{- $_ := set $modelIds $b.modelId $name -}}
{{- $_ := set $instances $name true -}}
{{- end -}}
{{- end -}}
{{- range $name, $bRaw := .Values.externalBackends -}}
{{- if include "deepfellow.backendEnabled" (dict "b" $bRaw) -}}
{{- $b := include "deepfellow.normalizedExternalBackend" (dict "raw" $bRaw "name" $name) | fromYaml -}}
{{- if not $b.apiUrl -}}
{{- fail (printf "externalBackends.%s.apiUrl is required (the endpoint ROOT URL; Infra appends v1/)." $name) -}}
{{- end -}}
{{- if not $b.modelId -}}
{{- fail (printf "externalBackends.%s.modelId is required." $name) -}}
{{- end -}}
{{- include "deepfellow.validateNotGenerated" (dict "cred" $b.apiKey "what" "the key the endpoint expects") -}}
{{- if hasKey $modelIds $b.modelId -}}
{{- fail (printf "duplicate modelId %q (externalBackends.%s and %s): model ids must be unique across modelBackends and externalBackends." $b.modelId $name (index $modelIds $b.modelId)) -}}
{{- end -}}
{{- $_ := set $modelIds $b.modelId $name -}}
{{- if hasKey $instances $name -}}
{{- fail (printf "duplicate backend key %q (externalBackends.%s and modelBackends.%s): the map key is the Infra service instance name and must be unique across modelBackends and externalBackends." $name $name $name) -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{- define "deepfellow.validate.server" -}}
{{- $cred := include "deepfellow.credentials" . | fromYaml -}}
{{- $s := .Values.server -}}
{{- include "deepfellow.validateImage" (dict "image" $s.image "path" "server.image") -}}
{{- if include "deepfellow.mongo.inChart" . -}}
{{- include "deepfellow.validateImage" (dict "image" $s.mongo.embedded.image "path" "server.mongo.embedded.image") -}}
{{- include "deepfellow.validateSecret" (dict "cred" $cred.server.mongo.embedded.rootPassword "path" "server.mongo.embedded.rootPassword" "required" true) -}}
{{- else -}}
{{- include "deepfellow.validateNotGenerated" (dict "cred" $cred.server.mongo.password "what" "the user's password in your MongoDB") -}}
{{- if empty $s.mongo.external.address -}}
{{- fail "server.mongo.external.address is required when server.mongo.mode=external." -}}
{{- end -}}
{{- if hasPrefix "mongodb" $s.mongo.external.address -}}
{{- fail "server.mongo.external.address must not be a connection URI: give host:port (optionally with /?options). The Server prefixes mongodb://<user>:<password>@ itself, so a full URI fails at startup with InvalidURI." -}}
{{- end -}}
{{- end -}}
{{- include "deepfellow.validateSecret" (dict "cred" $cred.server.mongo.password "path" "server.mongo.password" "required" true) -}}
{{- if include "deepfellow.qdrant.inChart" . -}}
{{- include "deepfellow.validateImage" (dict "image" $s.vectorDb.qdrant.embedded.image "path" "server.vectorDb.qdrant.embedded.image") -}}
{{- else if and $s.vectorDb.enabled (eq $s.vectorDb.type "qdrant") (empty $s.vectorDb.qdrant.external.url) -}}
{{- fail "server.vectorDb.qdrant.external.url is required when server.vectorDb.qdrant.mode=external." -}}
{{- end -}}
{{- if include "deepfellow.milvus.inChart" . -}}
{{- include "deepfellow.validateImage" (dict "image" $s.vectorDb.milvus.etcd.image "path" "server.vectorDb.milvus.etcd.image") -}}
{{- include "deepfellow.validateImage" (dict "image" $s.vectorDb.milvus.minio.image "path" "server.vectorDb.milvus.minio.image") -}}
{{- include "deepfellow.validateImage" (dict "image" $s.vectorDb.milvus.image "path" "server.vectorDb.milvus.image") -}}
{{- include "deepfellow.validateSecret" (dict "cred" $cred.server.vectorDb.milvus.minio.rootUser "path" "server.vectorDb.milvus.minio.rootUser" "required" true) -}}
{{- include "deepfellow.validateSecret" (dict "cred" $cred.server.vectorDb.milvus.minio.rootPassword "path" "server.vectorDb.milvus.minio.rootPassword" "required" true) -}}
{{- end -}}
{{- if $s.falkordb.enabled -}}
{{- include "deepfellow.validateImage" (dict "image" $s.falkordb.image "path" "server.falkordb.image") -}}
{{- include "deepfellow.validateSecret" (dict "cred" $cred.server.falkordb.password "path" "server.falkordb.password" "required" true) -}}
{{- end -}}
{{- if and $s.otel.enabled (not $s.otel.endpoint) -}}
{{- fail "server.otel.endpoint is required when server.otel.enabled=true." -}}
{{- end -}}
{{- if $s.metrics.basicAuth.enabled -}}
{{- if not $s.metrics.basicAuth.username -}}
{{- fail "server.metrics.basicAuth.username is required when server.metrics.basicAuth.enabled=true (DF_METRICS_USERNAME would be empty)." -}}
{{- end -}}
{{- include "deepfellow.validateSecret" (dict "cred" $cred.server.metrics.basicAuth.password "path" "server.metrics.basicAuth.password" "required" true) -}}
{{- end -}}
{{- end -}}

{{- define "deepfellow.validate.workspace" -}}
{{- if .Values.workspace.enabled -}}
{{- $cred := include "deepfellow.credentials" . | fromYaml -}}
{{- $w := .Values.workspace -}}
{{- include "deepfellow.validateImage" (dict "image" $w.image "path" "workspace.image") -}}
{{- include "deepfellow.validateImage" (dict "image" $w.mongo.image "path" "workspace.mongo.image") -}}
{{- if and (not .Values.provisioning.enabled) (not .Values.provisioning.stateSecret.name) -}}
{{- fail "workspace.enabled=true needs the provisioning state Secret (DF_SERVER_* dfproj identity): enable provisioning.enabled, or set provisioning.stateSecret.name to a Secret you manage. Otherwise the Workspace pod never starts." -}}
{{- end -}}
{{- if empty $w.helperModel -}}
{{- fail "workspace.helperModel is required when workspace.enabled=true: the Workspace image validates SMALL_MODEL as a non-empty string and exits at startup if it is missing. Set it to a model id this release serves." -}}
{{- end -}}
{{- if or (empty $w.auth.bootstrapAdmin.name) (empty $w.auth.bootstrapAdmin.email) -}}
{{- fail "workspace.auth.bootstrapAdmin: name and email must both be set. The Workspace validates BOOTSTRAP_ADMIN_NAME, BOOTSTRAP_ADMIN_EMAIL and BOOTSTRAP_ADMIN_PASSWORD as a group and exits at startup unless all three are supplied." -}}
{{- end -}}
{{- include "deepfellow.validateSecret" (dict "cred" $cred.workspace.auth.sessionSecret "path" "workspace.auth.sessionSecret" "required" true) -}}
{{- include "deepfellow.validateSecret" (dict "cred" $cred.workspace.auth.bootstrapAdmin.password "path" "workspace.auth.bootstrapAdmin.password" "required" true) -}}
{{- include "deepfellow.validateSecret" (dict "cred" $cred.workspace.mongo.rootPassword "path" "workspace.mongo.rootPassword" "required" true) -}}
{{- include "deepfellow.validateUrlSafeValue" (dict "cred" $cred.workspace.mongo.rootPassword "path" "workspace.mongo.rootPassword") -}}
{{- include "deepfellow.validateSecret" (dict "cred" $cred.workspace.mongo.keyFile "path" "workspace.mongo.keyFile" "required" true) -}}
{{- include "deepfellow.validateImage" (dict "image" $w.redis.image "path" "workspace.redis.image") -}}
{{- include "deepfellow.validateSecret" (dict "cred" $cred.workspace.redis.password "path" "workspace.redis.password" "required" true) -}}
{{- include "deepfellow.validateUrlSafeValue" (dict "cred" $cred.workspace.redis.password "path" "workspace.redis.password") -}}
{{- end -}}
{{- end -}}

{{/* Provisioning: the admin it creates needs a password, and a project with no model to grant is
     almost certainly a mistake. */}}
{{- define "deepfellow.validate.provisioning" -}}
{{- if .Values.provisioning.enabled -}}
{{- $cred := include "deepfellow.credentials" . | fromYaml -}}
{{- include "deepfellow.validateSecret" (dict "cred" $cred.provisioning.admin.password "path" "provisioning.admin.password" "required" true) -}}
{{- include "deepfellow.validatePasswordComplexity" (dict "cred" $cred.provisioning.admin.password "path" "provisioning.admin.password") -}}
{{- if .Values.provisioning.image.repository -}}
{{- include "deepfellow.validateImage" (dict "image" .Values.provisioning.image "path" "provisioning.image") -}}
{{- end -}}
{{- $plan := include "deepfellow.provisioning.plan" . | fromJson -}}
{{- if and (empty $plan.backends) (not .Values.provisioning.allowNoModels) -}}
{{- fail "provisioning.enabled=true but no native modelBackends or externalBackends are enabled: the suite would provision a project with no usable model. Enable a backend or set provisioning.allowNoModels=true." -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{- define "deepfellow.validate.ingress" -}}
{{- if .Values.ingress.enabled -}}
{{- $hasWorkspace := and .Values.workspace.enabled .Values.ingress.workspace.host -}}
{{- if not (or .Values.ingress.server.host $hasWorkspace) -}}
{{- fail "ingress.enabled=true but no host is set: set ingress.server.host and/or ingress.workspace.host (an Ingress with no rules is invalid)." -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/*
Names Kubernetes caps below the 63 characters _names.tpl truncates to. A Job's name becomes the
job-name label of its pods (63 max, API >= 1.27 rejects longer ones). A StatefulSet's pods get the
label controller-revision-hash=<name>-<10-char hash>, so its name can be 52 at most. The chart's
Job names end in an 8-character digest; "00000000" stands in for it. StatefulSets are checked
whether or not enabled: the longest ones bind either way.
*/}}
{{- define "deepfellow.validate.names" -}}
{{- $capped := list -}}
{{- range $c := list "mongo" "qdrant" "etcd" "minio" "milvus" "falkordb" -}}
{{- $capped = append $capped (dict "kind" "StatefulSet" "max" 52 "name" (include "deepfellow.componentName" (dict "context" $ "component" $c))) -}}
{{- end -}}
{{- if .Values.workspace.enabled -}}
{{- $appMongo := include "deepfellow.componentName" (dict "context" . "component" "workspace-mongo") -}}
{{- $capped = append $capped (dict "kind" "StatefulSet" "max" 52 "name" $appMongo) -}}
{{- $capped = append $capped (dict "kind" "Job" "max" 63 "name" (printf "%s-rs-init-00000000" $appMongo)) -}}
{{- end -}}
{{- if .Values.provisioning.enabled -}}
{{- $prov := include "deepfellow.componentName" (dict "context" . "component" "provisioning") -}}
{{- $capped = append $capped (dict "kind" "Job" "max" 63 "name" (printf "%s-%s-00000000" $prov (toString .Values.provisioning.revision))) -}}
{{- end -}}
{{- range $n := $capped -}}
{{- if gt (len $n.name) (int $n.max) -}}
{{- fail (printf "release name %q is too long: it makes the %s %q %d characters long, and Kubernetes allows %d. Use a shorter release name or set fullnameOverride." $.Release.Name $n.kind $n.name (len $n.name) (int $n.max)) -}}
{{- end -}}
{{- end -}}
{{- end -}}
