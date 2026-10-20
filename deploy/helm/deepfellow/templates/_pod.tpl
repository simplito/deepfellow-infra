{{/* What every workload's pod is built from: metadata, ServiceAccount, security contexts, placement,
     dependency waits and data volumes. STYLEGUIDE.md shows the order they are used in. */}}

{{/*
Pod / container security contexts, built from three layers (later wins):
  1. the chart-wide `podSecurityContext` / `containerSecurityContext` in values.yaml
  2. `base`: a component's built-in exception (Mongo's uid 999, Milvus' group 0)
  3. `values`: the component's own `podSecurityContext` / `containerSecurityContext`
Top-level keys are REPLACED, not deep-merged: mergeOverwrite skips false/0, so it could never
express `runAsNonRoot: false`, and replacing `capabilities` whole is easier to reason about.
  {{- include "deepfellow.podSecurityContext" (dict "context" $ "values" $x) | nindent 8 }}
  {{- include "deepfellow.containerSecurityContext" (dict "context" $ "values" $x) | nindent 12 }}
*/}}
{{- define "deepfellow.podSecurityContext" -}}
{{- include "deepfellow.layered" (list .context.Values.podSecurityContext .base (dig "podSecurityContext" dict (.values | default dict))) -}}
{{- end -}}

{{- define "deepfellow.containerSecurityContext" -}}
{{- include "deepfellow.layered" (list .context.Values.containerSecurityContext .base (dig "containerSecurityContext" dict (.values | default dict))) -}}
{{- end -}}

{{/* Shallow, key-replacing merge of a list of dicts (nil entries skipped), as YAML. */}}
{{- define "deepfellow.layered" -}}
{{- $out := dict -}}
{{- range $layer := . -}}
{{- range $k, $v := ($layer | default dict) -}}
{{- $_ := set $out $k $v -}}
{{- end -}}
{{- end -}}
{{- toYaml $out -}}
{{- end -}}

{{/*
Pod placement from a component's values block (nodeSelector / tolerations / affinity). Every
workload in the chart must honour these; the backing stores used to render none of them, so they
ignored the placement every other pod followed.
  {{- include "deepfellow.podPlacement" $x | nindent 6 }}
*/}}
{{- define "deepfellow.podPlacement" -}}
{{- with .nodeSelector }}
nodeSelector:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- with .tolerations }}
tolerations:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- with .affinity }}
affinity:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- end -}}

{{/*
The body of every wait in the chart: run `check` until it succeeds, at most dependencyWait.attempts
times, dependencyWait.delaySeconds apart, then give up and exit 1 so the pod fails visibly. `what`
names what is being waited for in the log; it may use shell variables.
  {{- include "deepfellow.retry" (dict "context" $ "check" `[ -s /state/key ]` "what" "the state file") | nindent 14 }}
*/}}
{{- define "deepfellow.retry" -}}
{{- $w := .context.Values.dependencyWait -}}
i=0
until {{ .check }}; do
  i=$((i+1))
  if [ "$i" -ge {{ int $w.attempts }} ]; then echo "gave up waiting for {{ .what }} after $i attempts" >&2; exit 1; fi
  echo "waiting for {{ .what }}"
  sleep {{ int $w.delaySeconds }}
done
echo "{{ .what }} is ready"
{{- end -}}

{{/*
An init container that blocks until every URL in `urls` answers 2xx.
  {{- include "deepfellow.waitFor" (dict "context" $ "values" $x "name" "wait-for-x" "urls" (list "http://a/health") "resources" $x.resources) | nindent 8 }}
*/}}
{{- define "deepfellow.waitFor" -}}
{{- $w := .context.Values.dependencyWait -}}
- name: {{ .name }}
  image: {{ include "deepfellow.image" (dict "context" .context "image" $w.image) }}
  imagePullPolicy: {{ $w.image.pullPolicy | default "IfNotPresent" }}
  securityContext:
    {{- include "deepfellow.containerSecurityContext" (dict "context" .context "values" .values) | nindent 4 }}
  command:
    - sh
    - -c
    - |
      set -eu
      for url in {{ join " " .urls }}; do
        {{- include "deepfellow.retry" (dict "context" .context "check" `curl -fsS -o /dev/null --max-time 5 "$url"` "what" "$url") | nindent 8 }}
      done
  {{- with .resources }}
  resources:
    {{- toYaml . | nindent 4 }}
  {{- end }}
{{- end -}}

{{/*
Storage for a single-replica StatefulSet, either a claim the operator already made or one the
chart creates. A volumeClaimTemplate cannot point at an existing PVC, so supplying one swaps the
template for an ordinary pod volume — safe here because every StatefulSet in this chart runs one
replica, and a template would only ever produce one claim anyway.

  pod spec:  {{- include "deepfellow.existingDataVolume" (dict "claim" $x.storage.existingClaim) | nindent 6 }}
  sts spec:  {{- include "deepfellow.dataVolumeClaimTemplate" (dict "storage" $x.storage) | nindent 2 }}
*/}}
{{- define "deepfellow.existingDataVolume" -}}
{{- with .claim }}
- name: data
  persistentVolumeClaim:
    claimName: {{ . }}
{{- end }}
{{- end -}}

{{- define "deepfellow.dataVolumeClaimTemplate" -}}
{{- if not .storage.existingClaim }}
volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: [ReadWriteOnce]
      {{- with .storage.storageClass }}
      storageClassName: {{ . }}
      {{- end }}
      resources:
        requests:
          storage: {{ .storage.size }}
{{- end }}
{{- end -}}

{{/*
ServiceAccount of every workload except the provisioning Job (see _provisioning.tpl).
*/}}
{{- define "deepfellow.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- default (include "deepfellow.fullname" .) .Values.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.serviceAccount.name -}}
{{- end -}}
{{- end -}}

{{/*
A workload's pod-template metadata.
  labels:      podLabels (chart-wide, then the component's), then `extraLabels`, then the
               component's selector labels, which always win so a podLabel can never detach a pod
               from its Service.
  annotations: podAnnotations (chart-wide, then the component's), plus checksum/credentials when
               `credentials` is given.
checksum/credentials hashes the normalized credentials the pod reads, so changing a plaintext
value rolls the pod: its env comes from the chart Secret through secretKeyRef, which a running
pod never re-reads. It hashes what the user wrote, not the rendered Secret, because a generated
value re-renders at random wherever lookup is empty (helm template, GitOps) and would roll pods on
every sync. A rotated referenced Secret is invisible at render time; it needs a rollout restart.
On a Job the annotation is part of the pod template the Job's name is digested from, so a changed
credential re-runs the Job.
  {{- include "deepfellow.podMetadata" (dict "context" $ "values" $x "component" "server" "credentials" (list $cred.infra.auth.serverApiKey)) | nindent 6 }}
*/}}
{{- define "deepfellow.podMetadata" -}}
{{- $values := .values | default dict -}}
{{- $selector := include "deepfellow.selectorLabels" (dict "context" .context "component" .component) | fromYaml -}}
{{- $labels := include "deepfellow.layered" (list .context.Values.podLabels (dig "podLabels" dict $values) .extraLabels $selector) | fromYaml -}}
{{- $annotations := include "deepfellow.layered" (list .context.Values.podAnnotations (dig "podAnnotations" dict $values)) | fromYaml -}}
{{- if hasKey . "credentials" -}}
{{- $_ := set $annotations "checksum/credentials" (toJson .credentials | sha256sum) -}}
{{- end -}}
labels:
  {{- toYaml $labels | nindent 2 }}
{{- with $annotations }}
annotations:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- end -}}

