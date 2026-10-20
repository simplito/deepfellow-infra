{{/* Image references and pull secrets. */}}

{{/*
Resolve an image ref from an image sub-map ({repository, tag, digest}), honoring
global.imageRegistry. Digest wins over tag when present. Call as:
  {{ include "deepfellow.image" (dict "context" . "image" .Values.infra.image) }}
*/}}
{{- define "deepfellow.image" -}}
{{- $registry := .context.Values.global.imageRegistry -}}
{{- $repo := .image.repository -}}
{{- if $registry -}}
{{- $repo = printf "%s/%s" $registry $repo -}}
{{- end -}}
{{- if .image.digest -}}
{{- printf "%s@%s" $repo .image.digest -}}
{{- else -}}
{{- printf "%s:%s" $repo (.image.tag | toString) -}}
{{- end -}}
{{- end -}}

{{/*
Render global.imagePullSecrets as a `imagePullSecrets:` list value (may be empty).
*/}}
{{- define "deepfellow.imagePullSecrets" -}}
{{- with .Values.global.imagePullSecrets }}
imagePullSecrets:
{{- range . }}
  {{- /* The schema allows both `- regcred` and `- name: regcred`. */}}
  {{- if kindIs "string" . }}
  - name: {{ . }}
  {{- else }}
  - name: {{ .name }}
  {{- end }}
{{- end }}
{{- end -}}
{{- end -}}
