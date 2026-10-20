{{/*
Health probes. Every workload's DEFAULT probes live in this file, so tuning them for the images the
chart ships is a one-file change; a component overrides them from values.

  {{- include "deepfellow.probes" (dict "defaults" (include "deepfellow.probeDefaults.x" . | fromYaml) "values" $x) | nindent 10 }}

`values` is the component's block; its `readinessProbe` / `livenessProbe` / `startupProbe` apply on
top of the default: top-level keys replace, a handler (httpGet/exec/tcpSocket/grpc) replaces the
default handler instead of sitting next to it, and `enabled: false` drops the probe.
*/}}
{{- define "deepfellow.probes" -}}
{{- $values := .values | default dict -}}
{{- range $kind := list "startupProbe" "livenessProbe" "readinessProbe" }}
{{- $probe := deepCopy (dig $kind dict $.defaults) -}}
{{- $o := dig $kind dict $values -}}
{{- if and (hasKey $o "enabled") (not $o.enabled) -}}
{{- $probe = dict -}}
{{- else -}}
{{- if or (hasKey $o "httpGet") (hasKey $o "exec") (hasKey $o "tcpSocket") (hasKey $o "grpc") -}}
{{- $probe = omit $probe "httpGet" "exec" "tcpSocket" "grpc" -}}
{{- end -}}
{{- range $k, $v := omit $o "enabled" -}}
{{- $_ := set $probe $k $v -}}
{{- end -}}
{{- end -}}
{{- with $probe }}
{{ $kind }}:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- end }}
{{- end -}}

{{- define "deepfellow.probeDefaults.etcd" -}}
readinessProbe:
  httpGet:
    path: /health
    port: client
  initialDelaySeconds: 5
  periodSeconds: 10
{{- end -}}

{{- define "deepfellow.probeDefaults.falkordb" -}}
readinessProbe:
  exec:
    command: ["sh", "-c", "redis-cli -a \"$DF_GRAPH__PASSWORD\" --no-auth-warning ping | grep -q PONG"]
  initialDelaySeconds: 5
  periodSeconds: 10
{{- end -}}

{{/*
Infra and the Server ship the same /app/scripts/healthcheck.py (it calls the app's own /health),
so they share one probe definition; tune it here for both.
*/}}
{{- define "deepfellow.probeDefaults.healthcheckScript" -}}
livenessProbe:
  exec:
    command: ["/app/scripts/healthcheck.py"]
  initialDelaySeconds: 40
  periodSeconds: 30
  timeoutSeconds: 5
  failureThreshold: 3
readinessProbe:
  exec:
    command: ["/app/scripts/healthcheck.py"]
  initialDelaySeconds: 10
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 3
{{- end -}}

{{- define "deepfellow.probeDefaults.infra" -}}
{{- include "deepfellow.probeDefaults.healthcheckScript" . -}}
{{- end -}}

{{- define "deepfellow.probeDefaults.milvus" -}}
readinessProbe:
  httpGet:
    path: /healthz
    port: metrics
  initialDelaySeconds: 20
  periodSeconds: 10
  failureThreshold: 30
{{- end -}}

{{- define "deepfellow.probeDefaults.minio" -}}
readinessProbe:
  httpGet:
    path: /minio/health/live
    port: api
  initialDelaySeconds: 5
  periodSeconds: 10
{{- end -}}

{{- define "deepfellow.probeDefaults.modelBackend" -}}
readinessProbe:
  httpGet:
    path: /health
    port: http
  initialDelaySeconds: 15
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 30
livenessProbe:
  httpGet:
    path: /health
    port: http
  initialDelaySeconds: 60
  periodSeconds: 30
  timeoutSeconds: 5
  failureThreshold: 3
{{- end -}}

{{- define "deepfellow.probeDefaults.mongo" -}}
livenessProbe:
  exec:
    command: ["mongosh", "--eval", "db.runCommand('ping')"]
  initialDelaySeconds: 20
  periodSeconds: 15
  timeoutSeconds: 10
  failureThreshold: 5
readinessProbe:
  exec:
    command: ["mongosh", "--eval", "db.runCommand('ping')"]
  initialDelaySeconds: 5
  periodSeconds: 10
  timeoutSeconds: 10
  failureThreshold: 5
{{- end -}}

{{- define "deepfellow.probeDefaults.qdrant" -}}
readinessProbe:
  httpGet:
    path: /readyz
    port: http
  initialDelaySeconds: 5
  periodSeconds: 10
{{- end -}}

{{- define "deepfellow.probeDefaults.server" -}}
{{- include "deepfellow.probeDefaults.healthcheckScript" . -}}
{{- end -}}

{{- define "deepfellow.probeDefaults.workspace" -}}
readinessProbe:
  httpGet:
    {{/* `/` is 404 on this image even though it sets WEB_DIST_PATH itself; /health
         is the endpoint that actually answers 200. */}}
    path: /health
    port: http
  initialDelaySeconds: 10
  periodSeconds: 10
{{- end -}}

{{- define "deepfellow.probeDefaults.workspaceMongo" -}}
readinessProbe:
  exec:
    command: ["mongosh", "--quiet", "--eval", "db.runCommand('ping')"]
  initialDelaySeconds: 5
  periodSeconds: 10
  timeoutSeconds: 10
  failureThreshold: 5
{{- end -}}

{{- define "deepfellow.probeDefaults.workspaceRedis" -}}
readinessProbe:
  exec:
    command: ["sh", "-c", "redis-cli -a \"$REDIS_PASSWORD\" --no-auth-warning ping | grep -q PONG"]
  initialDelaySeconds: 3
  periodSeconds: 10
{{- end -}}
