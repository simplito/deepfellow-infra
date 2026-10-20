{{/*
Which in-chart components render. Every file of a component guards on the SAME predicate, so a
condition changed here changes the Service, the workload, its NetworkPolicy and anything that
waits on it together. Each returns "true" or "" — use as `{{- if include "..." . }}`.
*/}}

{{/* The Server's MongoDB runs in-chart (vs server.mongo.mode=external). */}}
{{- define "deepfellow.mongo.inChart" -}}
{{- if eq .Values.server.mongo.mode "embedded" -}}true{{- end -}}
{{- end -}}

{{/* Qdrant runs in-chart: vector DB enabled, type qdrant, qdrant.mode=embedded. */}}
{{- define "deepfellow.qdrant.inChart" -}}
{{- $v := .Values.server.vectorDb -}}
{{- if and $v.enabled (eq $v.type "qdrant") (eq $v.qdrant.mode "embedded") -}}true{{- end -}}
{{- end -}}

{{/* The Milvus stack (milvus + etcd + MinIO) runs in-chart: vector DB enabled, type milvus. */}}
{{- define "deepfellow.milvus.inChart" -}}
{{- $v := .Values.server.vectorDb -}}
{{- if and $v.enabled (eq $v.type "milvus") -}}true{{- end -}}
{{- end -}}
