{{/* MongoDB pieces shared by the Server's mongo and the Workspace's replica set. */}}

{{/*
Turns off glibc's shadow-stack (SHSTK/CET) hwcap for mongod, a workaround for mongod crashing on
hosts where newer glibc enables it. The chart has always set it; no crash without it was
reproduced here, so treat it as a precaution. Both MongoDB StatefulSets run the same image, so it
lives here rather than in one of them.
*/}}
{{- define "deepfellow.mongoImageEnv" -}}
- name: GLIBC_TUNABLES
  value: "glibc.cpu.hwcaps=-SHSTK"
{{- end -}}

{{/*
Pod security context for the MongoDB workloads. They run as the official image's own mongodb
user (uid/gid 999) instead of the chart's uid: the image's data directories belong to it, and
fsGroup 999 makes the volume writable for it. (The image's entrypoint only chowns when started as
root, which the chart never does.) Declared once so the exception cannot drift between the two Mongos; it is the
`base` layer of deepfellow.podSecurityContext, so the chart-wide defaults still apply around it.
*/}}
{{- define "deepfellow.mongoPodSecurityContext" -}}
runAsUser: 999
runAsGroup: 999
fsGroup: 999
{{- end -}}

{{/*
The four env vars the Server (and the provisioning Job) need to reach MongoDB, for either
mongo.mode. Declared once because both templates must stay byte-identical: a drift here means
the Job talks to a different database than the Server.

NOTE on DF_MONGO_URL: the Server composes `mongodb://<user>:<password>@<DF_MONGO_URL>` itself,
so this value is a host:port (optionally with `/?options`), NOT a connection URI. Passing a full
URI yields `InvalidURI: Bad database name`.
  {{ include "deepfellow.mongoEnv" (dict "context" . "mongoSecret" $m "serverSecret" $sec) }}
*/}}
{{- define "deepfellow.mongoEnv" -}}
{{- $cred := include "deepfellow.credentials" .context | fromYaml -}}
{{- $ctx := .context -}}
{{- $m := $ctx.Values.server.mongo -}}
{{- $embedded := eq $m.mode "embedded" -}}
{{- $url := ternary (include "deepfellow.address" (dict "context" $ctx "port" "mongo.mongo")) $m.external.address $embedded -}}
{{/* The password sits in the in-chart Mongo's Secret, or in the Server's when there is no in-chart Mongo. */}}
{{- $secret := ternary .mongoSecret .serverSecret $embedded -}}
- name: DF_MONGO_URL
  value: {{ $url | quote }}
- name: DF_MONGO_USER
  value: {{ $m.username | quote }}
{{ include "deepfellow.credEnv" (dict "name" "DF_MONGO_PASSWORD" "cred" $cred.server.mongo.password "secretName" $secret "required" true) }}
- name: DF_MONGO_DB
  value: {{ $m.database | quote }}
{{- end -}}
