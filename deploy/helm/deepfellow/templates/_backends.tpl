{{/* Model backends (native and external): defaults and normalisation. */}}

{{/*
Map entries (modelBackends.<name>, externalBackends.<name>) get nothing from values.yaml, so every
reader goes through a normalizer here: validation, the templates and the provisioning plan then
see the same backend, and a difference between them cannot make the Job register something other
than what was deployed. Each normalizer fills in the defaults, the entry's credential in the
normalized form (_credentials.tpl), and anything derived from the map key.

They deep-merge with mergeOverwrite, which (unlike the security-context layering in _pod.tpl)
cannot override a default with false or 0. That is safe while every default that is true or
non-zero is one nobody sets to false/0: `enabled` is read from the raw entry (backendEnabled), and
parallel, threads, contextLength and ports must be positive. Keep it that way when adding a default.

  {{- $b := include "deepfellow.normalizedBackend" (dict "context" $ "raw" $bRaw "name" $name) | fromYaml }}
*/}}
{{- define "deepfellow.normalizedBackend" -}}
{{- $b := mergeOverwrite (include "deepfellow.modelBackendDefaults" .context | fromYaml) (deepCopy .raw) -}}
{{- $_ := set $b "modelId" (default .name $b.modelId) -}}
{{- $_ := set $b.gguf "hfToken" (include "deepfellow.normalizeCredential" (dict "cred" $b.gguf.hfToken "key" "hf-token" "path" (printf "modelBackends.%s.gguf.hfToken" .name)) | fromYaml) -}}
{{ toYaml $b }}
{{- end -}}

{{/*
Defaults for one modelBackends entry: the only place they are written down. modelId stays empty
here; normalizedBackend fills it with the map key.
*/}}
{{- define "deepfellow.modelBackendDefaults" -}}
enabled: true
type: llamaCpp
modelId: ""
image:
  repository: ghcr.io/ggml-org/llama.cpp
  tag: server
  digest: ""
  allowMutableTag: false
  pullPolicy: IfNotPresent
gguf:
  url: ""
  sha256: ""
  hfToken: {}
downloadImage:
  repository: curlimages/curl
  tag: "8.11.1"
  digest: "sha256:c1fe1679c34d9784c1b0d1e5f62ac0a79fca01fb6377cdd33e90473c6f9f9a69"
  allowMutableTag: false
  pullPolicy: IfNotPresent
parallel: 1
threads: 4
contextLength: 8192
extraArgs: []
gpu:
  enabled: false
  count: 1
storage:
  size: 20Gi
  medium: ""
  storageClass: ""
  existingClaim: ""
service:
  port: 8080
{{- end -}}

{{/*
One externalBackends entry, normalized (see normalizedBackend above).
  {{- $b := include "deepfellow.normalizedExternalBackend" (dict "raw" $bRaw "name" $name) | fromYaml }}
*/}}
{{- define "deepfellow.normalizedExternalBackend" -}}
{{- $b := mergeOverwrite (include "deepfellow.externalBackendDefaults" . | fromYaml) (deepCopy .raw) -}}
{{- $_ := set $b "apiKey" (include "deepfellow.normalizeCredential" (dict "cred" $b.apiKey "key" "api-key" "path" (printf "externalBackends.%s.apiKey" .name)) | fromYaml) -}}
{{ toYaml $b }}
{{- end -}}

{{- define "deepfellow.externalBackendDefaults" -}}
enabled: true
apiType: openai
apiUrl: ""
modelId: ""
contextLength: 8192
apiKey: {}
{{- end -}}
