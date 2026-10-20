{{/*
Where every in-chart component listens. Containers, Services, NetworkPolicies, health waits and
the env vars that point one component at another all read ports from here, so a port is written
down once. (The Service ports of Infra, the Server and the Workspace are values,
`<component>.service.port`; the containers behind them listen on the ports below.)
*/}}
{{- define "deepfellow.ports" -}}
infra: {http: 8086}
server: {http: 8000}
workspace: {http: 3000}
mongo: {mongo: 27017}
workspace-mongo: {mongo: 27017}
workspace-redis: {redis: 6379}
qdrant: {http: 6333, grpc: 6334}
milvus: {grpc: 19530, metrics: 9091}
etcd: {client: 2379}
minio: {api: 9000, console: 9001}
falkordb: {redis: 6379}
{{- end -}}

{{/*
One port, named "<component>.<port name>":
  containerPort: {{ include "deepfellow.port" "qdrant.http" }}
*/}}
{{- define "deepfellow.port" -}}
{{- $parts := splitList "." . -}}
{{- $component := index (include "deepfellow.ports" . | fromYaml) (first $parts) -}}
{{- if not (hasKey $component (last $parts)) -}}
{{- fail (printf "deepfellow.port: no port %q" .) -}}
{{- end -}}
{{- index $component (last $parts) -}}
{{- end -}}

{{/*
"<host>:<port>" of an in-chart component's Service:
  {{ include "deepfellow.address" (dict "context" . "port" "etcd.client") }}  ->  rel-deepfellow-etcd:2379
*/}}
{{- define "deepfellow.address" -}}
{{- $component := first (splitList "." .port) -}}
{{- printf "%s:%s" (include "deepfellow.componentName" (dict "context" .context "component" $component)) (include "deepfellow.port" .port) -}}
{{- end -}}
