{{/* Names and labels. */}}

{{/*
Chart name / fullname helpers.
*/}}
{{- define "deepfellow.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "deepfellow.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{- define "deepfellow.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Common labels applied to every resource.
*/}}
{{- define "deepfellow.labels" -}}
helm.sh/chart: {{ include "deepfellow.chart" . }}
app.kubernetes.io/name: {{ include "deepfellow.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: deepfellow
{{- end -}}

{{/*
Selector labels for a given component. Call as:
  {{ include "deepfellow.selectorLabels" (dict "context" . "component" "infra") }}
*/}}
{{- define "deepfellow.selectorLabels" -}}
app.kubernetes.io/name: {{ include "deepfellow.name" .context }}
app.kubernetes.io/instance: {{ .context.Release.Name }}
app.kubernetes.io/component: {{ .component }}
{{- end -}}

{{/*
Component resource name: <fullname>-<component>. Call as:
  {{ include "deepfellow.componentName" (dict "context" . "component" "server") }}
*/}}
{{- define "deepfellow.componentName" -}}
{{- printf "%s-%s" (include "deepfellow.fullname" .context) .component | trunc 63 | trimSuffix "-" -}}
{{- end -}}
