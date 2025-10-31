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
  embedding: "Embedding (embedding)",
  lora: "LORA (lora)",
  txt2img: "Text to image (txt2img)",
} as const;

export type ModelType = keyof typeof MODEL_TYPES;

export interface SpecField {
  name: string;
  description: string;
  type: "text" | "password" | "number" | "bool";
  required: boolean;
  default?: string | number | boolean;
  placeholder?: string;
}

export interface ServiceSpec {
  fields: SpecField[];
}

export interface Service {
  id: string;
  installed: Record<string, any> | null;
  spec: ServiceSpec;
  size: string | Record<string, string>;
}

export interface ServiceListResponse {
  list: Service[];
}

export interface ServiceModel {
  id: string;
  type: ModelType;
  installed: {
    spec?: Record<string, any>;
  } | null;
  spec: ServiceSpec;
  size: string;
}

export interface ServiceModelsResponse {
  list: ServiceModel[];
}

export interface InstallRequest {
  spec: Record<string, any>;
}

export interface UninstallRequest {
  purge: boolean;
}
