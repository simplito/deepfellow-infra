/*
DeepFellow Software Framework.
Copyright © 2025 Simplito sp. z o.o.

This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
This software is Licensed under the DeepFellow Free License.

See the License for the specific language governing permissions and
limitations under the License.
*/
// Admin Services and Models types
export const MODEL_TYPES = {
  tts: "Text to speech (TTS)",
  stt: "Speech to text (STT)",
  llm: "Large Language Model (LLM)",
  embedding: "Embedding Model (Embeddings)",
  lora: "LORA (lora)",
  reranker: "Reranker (reranker)",
  txt2img: "Text to image (txt2img)",
  decision: "Decision model (System One)",
  mcp: "MCP Server",
} as const;

export type ModelType = keyof typeof MODEL_TYPES;

export interface InstallProgress {
  stage: "install" | "download";
  value: number; // 0.0 to 1.0
}

/** A newer image tag that needs a newer CUDA than the host's NVIDIA driver supports. */
export interface NewerNeedsCuda {
  tag: string;
  required_cuda: string;
  host_cuda: string;
}

export interface SpecField {
  name: string;
  description: string;
  type:
    | "text"
    | "password"
    | "number"
    | "bool"
    | "oneof"
    | "list"
    | "map"
    | "textarea"
    | "docker-tags"
    | "gpu-priority";
  required: boolean;
  default?: string | number | boolean;
  placeholder?: string;
  values?: (string | { label: string; value: string })[]; // For oneof type
  display?: string; // Conditional display: "fieldName=value"
  docker_image?: string; // For docker-tags type: base image name without tag
  depends_on?: string; // For docker-tags type: field whose value is forwarded as ?hardware=
  required_keys?: string[]; // For map type: keys that must be present with a non-empty value
}

export interface ServiceSpec {
  fields: SpecField[];
}

export interface Service {
  id: string;
  type: string;
  instance: string;
  // Backend returns: bool | progress | installed options object
  installed: boolean | Record<string, unknown> | InstallProgress | null;
  downloaded?: boolean;
  spec: ServiceSpec;
  size: string | Record<string, string>;
  description?: string;
  custom_model_spec?: ServiceSpec;
  has_docker?: boolean;
  is_cloud?: boolean;
  disabled_reason?: string | null;
  catalog_refresh_unavailable_reason?: string | null;
}

export interface InfraSettings {
  cloud_enabled: boolean;
}

export interface ServiceListResponse {
  list: Service[];
  load_error?: string | null;
}

export interface ServiceModel {
  id: string;
  service?: string;
  type: ModelType;
  // Backend returns: bool | progress | installed info object
  installed:
    | boolean
    | {
        spec?: Record<string, unknown>;
        stage?: string;
        value?: number;
        registration_id?: string; // For test endpoint
      }
    | null;
  downloaded?: boolean;
  spec: ServiceSpec;
  size: string;
  custom?: string; // Custom model ID if custom
  default_prefix?: string | null; // The definition's declared default - may not match what's actually installed
  effective_prefix?: string | null; // What this model is actually reachable on; use this for collision checks
  has_docker?: boolean;
  vram_estimate_gb?: number | null;
  is_loaded?: boolean | null;
  variant?: string | null; // For user-defined MCP servers
  command?: string | null; // For user-defined MCP servers
  base_image?: string | null; // For user-defined MCP servers
  custom_spec?: Record<string, unknown> | null;
  description?: string | null;
  repository_url?: string | null;
  // False when this catalog entry's type/capabilities couldn't be resolved from either the
  // live provider listing or the known model metadata - installing it is rejected.
  capabilities_resolved?: boolean;
  // True when this entry was previously seen (and possibly installed) but is no longer present
  // in the most recent live provider listing.
  stale?: boolean;
  // null/undefined means the capability is unknown/not yet resolved for this backend; true/false means resolved.
  supports_vision?: boolean | null;
  supports_audio?: boolean | null;
}

export interface GpuCardStats {
  name: string;
  total_vram_gb: number;
  used_vram_gb: number;
}

export interface GpuStats {
  total_vram_gb: number;
  used_vram_gb: number;
  gpus: GpuCardStats[] | null;
}

export interface SystemStats {
  cpu_percent: number;
  cpu_model: string;
  ram_total_gb: number;
  ram_used_gb: number;
}

export interface ServiceModelsResponse {
  list: ServiceModel[];
}

export interface FinetuneAdapter {
  id: string;
  object: string;
  created: number; // Unix timestamp (seconds)
  owned_by: string;
  base_model: string;
  formats: string[];
  lora?: Record<string, unknown> | null;
  gguf?: Record<string, unknown> | null;
}

export interface FinetuneAdapterListResponse {
  object: string;
  data: FinetuneAdapter[];
}

export interface TestResult {
  result?: boolean; // true if test passed
  error?: boolean; // true if test failed
  output?:
    | string
    | {
        content_type: string; // e.g., "audio/wav", "image/png"
        data: string; // base64 encoded data
      };
  details?: Record<string, unknown>; // Additional details
}

export interface InstallRequest {
  spec: Record<string, unknown>;
}

export interface UninstallRequest {
  purge: boolean;
}

export class InstallationWarningsError extends Error {
  warnings: string[];

  constructor(warnings: string[], message?: string) {
    super(message || "Installation warnings detected");
    this.name = "InstallationWarningsError";
    this.warnings = warnings;
  }
}

// Config types
export interface ConfigEntry {
  key: string;
  value: string;
  is_secret: boolean;
  field_name: string;
  is_editable: boolean;
}

export interface ConfigOut {
  entries: ConfigEntry[];
}

export interface ConfigRevealOut {
  key: string;
  value: string;
}

// Mesh types
export interface MeshInfoModel {
  name: string;
  type: string;
}

export interface MeshInfoInfra {
  name: string;
  url: string;
  models: MeshInfoModel[];
}

export interface MeshInfo {
  connections: MeshInfoInfra[];
}

export interface ShowMeshInfoOut {
  info: MeshInfo;
}

export interface MeshTopologyModel {
  name: string;
  type: string;
}

export interface MeshTopologyNode {
  url: string;
  name: string;
  models: MeshTopologyModel[];
  children: MeshTopologyNode[];
  you_are_here: boolean;
}

// MCP OAuth types
export type McpOAuthStatusValue =
  | "disabled"
  | "not_started"
  | "pending"
  | "authorized"
  | "expired"
  | "error";

export interface McpOAuthStatus {
  enabled: boolean;
  status: McpOAuthStatusValue;
  has_client_id: boolean;
  has_client_secret: boolean;
  expires_at?: number | null;
  last_error?: string | null;
}

export interface McpOAuthStartOut {
  authorize_url: string;
}

// MCP JSON config conversion types (backend-owned; see `/admin/mcp/convert-config`)
export type ConvertedMcpKind = "user" | "proxy" | "custom";

export interface ConvertedMcpOAuth {
  client_id?: string | null;
  client_secret?: string | null;
  scope?: string | null;
}

export interface ConvertedUserMcpConfig {
  kind: "user";
  name: string;
  command: string;
  envs: Record<string, string>;
  variant?:
    | "node-headless"
    | "node-headed"
    | "python-headless"
    | "python-headed"
    | null;
}

export interface ConvertedProxyMcpConfig {
  kind: "proxy";
  name: string;
  server_url: string;
  transport: "streamable_http" | "sse";
  headers: Record<string, string>;
  oauth?: ConvertedMcpOAuth | null;
}

export interface ConvertedCustomMcpConfig {
  kind: "custom";
  name: string;
  image: string;
  command: string;
  volumes: string[];
  envs: Record<string, string>;
}

export type ConvertedMcpConfig =
  | ConvertedUserMcpConfig
  | ConvertedProxyMcpConfig
  | ConvertedCustomMcpConfig;

// Warnings
export interface ServiceWarning {
  id: string;
  created_at: string;
  service_id: string;
  instance?: string | null;
  model_id?: string | null;
  message: string;
}

export interface ListWarningsOut {
  list: ServiceWarning[];
  load_error?: string | null;
}
