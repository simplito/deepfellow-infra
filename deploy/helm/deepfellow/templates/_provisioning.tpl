{{/* Provisioning Job: names and the backend registration plan. */}}

{{/*
Name of the Secret holding the GET-ONCE dfproj provisioning state (org_id/project_id/key).
Keys: organization-id, project-id, project-api-key. Written by the provisioning Job.
*/}}
{{- define "deepfellow.provisioning.stateSecretName" -}}
{{- default (printf "%s-provisioning-state" (include "deepfellow.fullname" .)) .Values.provisioning.stateSecret.name -}}
{{- end -}}

{{/*
Provisioning ServiceAccount name.
*/}}
{{- define "deepfellow.provisioning.serviceAccountName" -}}
{{- if .Values.provisioning.serviceAccount.create -}}
{{- default (include "deepfellow.componentName" (dict "context" . "component" "provisioning")) .Values.provisioning.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.provisioning.serviceAccount.name -}}
{{- end -}}
{{- end -}}

{{/*
Provisioning plan: every enabled backend the Job registers into Infra, plus the env entries that
carry external backends' API keys. Returned as JSON so the Job can fromJson it.
*/}}
{{- define "deepfellow.provisioning.plan" -}}
{{- $backends := list }}
{{- $extEnvs := list }}
{{- range $name, $bRaw := .Values.modelBackends }}
{{- if include "deepfellow.backendEnabled" (dict "b" $bRaw) }}
{{- $b := include "deepfellow.normalizedBackend" (dict "context" $ "raw" $bRaw "name" $name) | fromYaml }}
{{- $svc := include "deepfellow.componentName" (dict "context" $ "component" (printf "mb-%s" $name)) }}
{{- $backends = append $backends (dict "id" $b.modelId "instance" $name "api_url" (printf "http://%s:%v" $svc $b.service.port) "service_type" "openai" "context_length" $b.contextLength "native" true) }}
{{- end }}
{{- end }}
{{- range $name, $bRaw := .Values.externalBackends }}
{{- if include "deepfellow.backendEnabled" (dict "b" $bRaw) }}
{{- $b := include "deepfellow.normalizedExternalBackend" (dict "raw" $bRaw "name" $name) | fromYaml }}
{{- $entry := dict "id" $b.modelId "instance" $name "api_url" $b.apiUrl "service_type" $b.apiType "context_length" $b.contextLength "native" false }}
{{- $ak := $b.apiKey }}
{{- if ne $ak.source "none" }}
{{- $envName := printf "DF_BACKEND_%s_API_KEY" ($name | upper | replace "-" "_") }}
{{- $extSecret := include "deepfellow.credSecretName" (dict "context" $ "component" (printf "ext-%s" $name)) }}
{{- $entry = merge $entry (dict "api_key_env" $envName) }}
{{- $extEnvs = append $extEnvs (dict "name" $envName "cred" $ak "secretName" $extSecret) }}
{{- end }}
{{- $backends = append $backends $entry }}
{{- end }}
{{- end }}
{{- toJson (dict "backends" $backends "extEnvs" $extEnvs) }}
{{- end }}
