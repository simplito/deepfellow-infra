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
  mcp: "MCP Server",
} as const;

export type ModelType = keyof typeof MODEL_TYPES;

export interface InstallProgress {
  stage: "install" | "download";
  value: number; // 0.0 to 1.0
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
    | "textarea";
  required: boolean;
  default?: string | number | boolean;
  placeholder?: string;
  values?: (string | { label: string; value: string })[]; // For oneof type
  display?: string; // Conditional display: "fieldName=value"
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
}

export interface InfraSettings {
  cloud_enabled: boolean;
}

export interface ServiceListResponse {
  list: Service[];
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
  has_docker?: boolean;
  vram_estimate_gb?: number | null;
  is_loaded?: boolean | null;
  variant?: string | null; // For user-defined MCP servers
  command?: string | null; // For user-defined MCP servers
  base_image?: string | null; // For user-defined MCP servers
  custom_spec?: Record<string, unknown> | null;
  description?: string | null;
  repository_url?: string | null;
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
