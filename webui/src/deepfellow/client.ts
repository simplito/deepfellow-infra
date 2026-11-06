import { type ProgressEvent, readSSEStream } from "@/utils/sse-stream";
/*
DeepFellow Software Framework.
Copyright © 2025 Simplito sp. z o.o.

This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
This software is Licensed under the DeepFellow Free License.

See the License for the specific language governing permissions and
limitations under the License.
*/
import type {
  ConfigOut,
  ConfigRevealOut,
  ConvertedMcpConfig,
  FinetuneAdapterListResponse,
  GpuStats,
  InfraSettings,
  ListWarningsOut,
  McpOAuthStartOut,
  McpOAuthStatus,
  MeshTopologyNode,
  Service,
  ServiceListResponse,
  ServiceModel,
  ServiceModelsResponse,
  ShowMeshInfoOut,
  SystemStats,
  TestResult,
} from "./types";
import { InstallationWarningsError } from "./types";
import type { NewerNeedsCuda } from "./types";

// Simple admin API key storage
export class AdminApiKeyStorage {
  private static readonly KEY = "admin_api_key";

  static get(): string | null {
    return localStorage.getItem(this.KEY);
  }

  static set(key: string): void {
    localStorage.setItem(this.KEY, key);
  }

  static clear(): void {
    localStorage.removeItem(this.KEY);
  }

  static exists(): boolean {
    return !!this.get();
  }
}

// API Client
export class DeepFellowClient {
  private baseURL: string;

  constructor(baseURL = "/") {
    this.baseURL = baseURL.replace(/\/$/, ""); // Remove trailing slash
  }

  private async makeRequest<T>(
    endpoint: string,
    options: RequestInit = {},
  ): Promise<T> {
    const url = `${this.baseURL}${endpoint}`;

    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      ...(options.headers as Record<string, string>),
    };

    // Use admin API key for authorization
    const adminApiKey = AdminApiKeyStorage.get();
    if (adminApiKey) {
      headers.Authorization = `Bearer ${adminApiKey}`;
    }

    const response = await fetch(url, {
      ...options,
      headers,
    });

    const content = await response.text();

    if (!response.ok) {
      try {
        const errorData = JSON.parse(content);
        const errorMessage = formatApiErrorMessage(errorData, content);
        throw new Error(`HTTP ${response.status}: ${errorMessage}`);
      } catch (e) {
        if (e instanceof SyntaxError) {
          throw new Error(
            `HTTP ${response.status}: ${content || response.statusText}`,
          );
        }
        throw e;
      }
    }

    // Parse successful response
    try {
      return JSON.parse(content);
    } catch (e) {
      console.error("JSON parse error", content, e);
      throw new Error(`Invalid response, cannot parse JSON: ${content}`);
    }
  }

  // Admin Services methods
  async listAdminServices(): Promise<ServiceListResponse> {
    return this.makeRequest<ServiceListResponse>("/admin/services");
  }

  async getAdminService(serviceId: string): Promise<Service> {
    return this.makeRequest<Service>(`/admin/services/${serviceId}`);
  }

  async installAdminService(
    serviceId: string,
    spec: Record<string, unknown>,
  ): Promise<void> {
    return this.makeRequest<void>(`/admin/services/${serviceId}`, {
      method: "POST",
      body: JSON.stringify({ spec }),
    });
  }

  async uninstallAdminService(serviceId: string, purge = false): Promise<void> {
    return this.makeRequest<void>(`/admin/services/${serviceId}`, {
      method: "DELETE",
      body: JSON.stringify({ purge }),
    });
  }

  async installAdminServiceStreaming(
    serviceId: string,
    spec: Record<string, unknown>,
    onProgress: (event: ProgressEvent) => void,
    ignoreWarnings = false,
    signal?: AbortSignal,
  ): Promise<void> {
    return this.writeAdminServiceStreaming(
      "POST",
      serviceId,
      spec,
      onProgress,
      ignoreWarnings,
      signal,
    );
  }

  async updateAdminServiceStreaming(
    serviceId: string,
    spec: Record<string, unknown>,
    onProgress: (event: ProgressEvent) => void,
    ignoreWarnings = false,
    signal?: AbortSignal,
  ): Promise<void> {
    return this.writeAdminServiceStreaming(
      "PUT",
      serviceId,
      spec,
      onProgress,
      ignoreWarnings,
      signal,
    );
  }

  private async writeAdminServiceStreaming(
    method: "POST" | "PUT",
    serviceId: string,
    spec: Record<string, unknown>,
    onProgress: (event: ProgressEvent) => void,
    ignoreWarnings = false,
    signal?: AbortSignal,
  ): Promise<void> {
    const url = `${this.baseURL}/admin/services/${serviceId}`;
    const adminApiKey = AdminApiKeyStorage.get();

    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };

    if (adminApiKey) {
      headers.Authorization = `Bearer ${adminApiKey}`;
    }

    const body: Record<string, unknown> = { stream: true, spec };
    if (ignoreWarnings) {
      body.ignore_warnings = true;
    }

    const response = await fetch(url, {
      method,
      headers,
      body: JSON.stringify(body),
      signal,
    });

    if (!response.ok) {
      const content = await response.text();
      try {
        const errorData = JSON.parse(content);

        // Check if this is a 400 with warnings
        if (
          response.status === 400 &&
          errorData.warnings &&
          Array.isArray(errorData.warnings)
        ) {
          throw new InstallationWarningsError(
            errorData.warnings,
            errorData.detail || errorData.message,
          );
        }

        const errorMessage = formatApiErrorMessage(errorData, content);
        throw new Error(`HTTP ${response.status}: ${errorMessage}`);
      } catch (e) {
        if (e instanceof InstallationWarningsError) {
          throw e;
        }
        if (e instanceof SyntaxError) {
          throw new Error(
            `HTTP ${response.status}: ${content || response.statusText}`,
          );
        }
        throw e;
      }
    }

    const contentType = (response.headers.get("content-type") || "")
      .split(";")[0]
      .trim();
    if (contentType === "text/event-stream") {
      await readSSEStream(response, onProgress);
    } else {
      // Non-streaming response
      await response.json();
    }
  }

  async cancelAdminServiceInstall(
    serviceId: string,
  ): Promise<{ status: string }> {
    return this.makeRequest<{ status: string }>(
      `/admin/services/${serviceId}/cancel`,
      {
        method: "POST",
      },
    );
  }

  async getServiceProgress(
    serviceId: string,
    onProgress: (event: ProgressEvent) => void,
    signal?: AbortSignal,
  ): Promise<void> {
    const url = `${this.baseURL}/admin/services/${serviceId}/progress`;
    const adminApiKey = AdminApiKeyStorage.get();

    const headers: Record<string, string> = {};
    if (adminApiKey) {
      headers.Authorization = `Bearer ${adminApiKey}`;
    }

    const response = await fetch(url, { headers, signal });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    await readSSEStream(response, onProgress);
  }

  // Admin Service Models methods
  async listAdminServiceModels(
    serviceId: string,
  ): Promise<ServiceModelsResponse> {
    return this.makeRequest<ServiceModelsResponse>(
      `/admin/services/${serviceId}/models`,
    );
  }

  async getAdminServiceModel(
    serviceId: string,
    modelId: string,
  ): Promise<ServiceModel> {
    return this.makeRequest<ServiceModel>(
      `/admin/services/${serviceId}/models/_?model_id=${encodeURIComponent(modelId)}`,
    );
  }

  async listFinetuneAdapters(
    serviceId: string,
  ): Promise<FinetuneAdapterListResponse> {
    return this.makeRequest<FinetuneAdapterListResponse>(
      `/admin/services/${serviceId}/adapters`,
    );
  }

  async installAdminServiceModel(
    serviceId: string,
    modelId: string,
    spec: Record<string, unknown>,
  ): Promise<void> {
    return this.makeRequest<void>(
      `/admin/services/${serviceId}/models/_?model_id=${encodeURIComponent(modelId)}`,
      {
        method: "POST",
        body: JSON.stringify({ spec }),
      },
    );
  }

  async uninstallAdminServiceModel(
    serviceId: string,
    modelId: string,
    purge = false,
  ): Promise<void> {
    return this.makeRequest<void>(
      `/admin/services/${serviceId}/models/_?model_id=${encodeURIComponent(modelId)}`,
      {
        method: "DELETE",
        body: JSON.stringify({ purge }),
      },
    );
  }

  async cancelAdminServiceModelInstall(
    serviceId: string,
    modelId: string,
  ): Promise<{ status: string }> {
    return this.makeRequest<{ status: string }>(
      `/admin/services/${serviceId}/models/cancel?model_id=${encodeURIComponent(modelId)}`,
      {
        method: "POST",
      },
    );
  }

  async installAdminServiceModelStreaming(
    serviceId: string,
    modelId: string,
    spec: Record<string, unknown>,
    onProgress: (event: ProgressEvent) => void,
    ignoreWarnings = false,
    signal?: AbortSignal,
  ): Promise<void> {
    const url = `${this.baseURL}/admin/services/${serviceId}/models/_?model_id=${encodeURIComponent(modelId)}`;
    const adminApiKey = AdminApiKeyStorage.get();

    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };

    if (adminApiKey) {
      headers.Authorization = `Bearer ${adminApiKey}`;
    }

    const body: Record<string, unknown> = { stream: true, spec };
    if (ignoreWarnings) {
      body.ignore_warnings = true;
    }

    const response = await fetch(url, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
      signal,
    });

    if (!response.ok) {
      const content = await response.text();
      try {
        const errorData = JSON.parse(content);

        // Check if this is a 400 with warnings
        if (
          response.status === 400 &&
          errorData.warnings &&
          Array.isArray(errorData.warnings)
        ) {
          throw new InstallationWarningsError(
            errorData.warnings,
            errorData.detail || errorData.message,
          );
        }

        const errorMessage = formatApiErrorMessage(errorData, content);
        throw new Error(`HTTP ${response.status}: ${errorMessage}`);
      } catch (e) {
        if (e instanceof InstallationWarningsError) {
          throw e;
        }
        if (e instanceof SyntaxError) {
          throw new Error(
            `HTTP ${response.status}: ${content || response.statusText}`,
          );
        }
        throw e;
      }
    }

    const contentType = (response.headers.get("content-type") || "")
      .split(";")[0]
      .trim();
    if (contentType === "text/event-stream") {
      await readSSEStream(response, onProgress);
    } else {
      // Non-streaming response
      await response.json();
    }
  }

  async getModelProgress(
    serviceId: string,
    modelId: string,
    onProgress: (event: ProgressEvent) => void,
    signal?: AbortSignal,
  ): Promise<void> {
    const url = `${this.baseURL}/admin/services/${serviceId}/models/progress?model_id=${encodeURIComponent(modelId)}`;
    const adminApiKey = AdminApiKeyStorage.get();

    const headers: Record<string, string> = {};
    if (adminApiKey) {
      headers.Authorization = `Bearer ${adminApiKey}`;
    }

    const response = await fetch(url, { headers, signal });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    await readSSEStream(response, onProgress);
  }

  async testModel(
    registrationId: string,
    signal?: AbortSignal,
  ): Promise<TestResult> {
    return this.makeRequest<TestResult>(
      `/admin/services/model/test/${registrationId}`,
      {
        signal,
      },
    );
  }

  async addCustomModel(
    serviceId: string,
    spec: Record<string, unknown>,
  ): Promise<{ custom_model_id: string }> {
    return this.makeRequest<{ custom_model_id: string }>(
      `/admin/services/${serviceId}/models/custom`,
      {
        method: "POST",
        body: JSON.stringify({ spec }),
      },
    );
  }

  async removeCustomModel(
    serviceId: string,
    customModelId: string,
  ): Promise<{ status: string }> {
    return this.makeRequest<{ status: string }>(
      `/admin/services/${serviceId}/models/custom/${customModelId}`,
      {
        method: "DELETE",
      },
    );
  }

  async updateCustomModel(
    serviceId: string,
    customModelId: string,
    spec: Record<string, unknown>,
  ): Promise<{ status: string }> {
    return this.makeRequest<{ status: string }>(
      `/admin/services/${serviceId}/models/custom/${customModelId}`,
      {
        method: "PUT",
        body: JSON.stringify({ spec }),
      },
    );
  }

  async editCustomModel(
    serviceId: string,
    customModelId: string,
    spec: Record<string, unknown>,
  ): Promise<{ status: string; reinstalled: boolean }> {
    return this.makeRequest<{ status: string; reinstalled: boolean }>(
      `/admin/services/${serviceId}/models/custom/${customModelId}/edit`,
      {
        method: "POST",
        body: JSON.stringify({ spec }),
      },
    );
  }

  async editModelInstallOptions(
    serviceId: string,
    modelId: string,
    spec: Record<string, unknown>,
  ): Promise<{ status: string; reinstalled: boolean }> {
    return this.makeRequest<{ status: string; reinstalled: boolean }>(
      `/admin/services/${serviceId}/models/_/edit?model_id=${encodeURIComponent(modelId)}`,
      {
        method: "POST",
        body: JSON.stringify({ spec }),
      },
    );
  }

  async getDuplicateSpec(
    serviceId: string,
    modelId: string,
  ): Promise<{ spec: Record<string, unknown> }> {
    return this.makeRequest<{ spec: Record<string, unknown> }>(
      `/admin/services/${serviceId}/models/_/duplicate-spec?model_id=${encodeURIComponent(modelId)}`,
    );
  }

  async startMcpOAuth(
    serviceId: string,
    modelId: string,
  ): Promise<McpOAuthStartOut> {
    return this.makeRequest<McpOAuthStartOut>(
      `/admin/services/${serviceId}/models/${encodeURIComponent(modelId)}/oauth/start`,
      { method: "POST" },
    );
  }

  async getMcpOAuthStatus(
    serviceId: string,
    modelId: string,
  ): Promise<McpOAuthStatus> {
    return this.makeRequest<McpOAuthStatus>(
      `/admin/services/${serviceId}/models/${encodeURIComponent(modelId)}/oauth/status`,
    );
  }

  async convertMcpConfig(
    config: Record<string, unknown>,
  ): Promise<ConvertedMcpConfig> {
    return this.makeRequest<ConvertedMcpConfig>("/admin/mcp/convert-config", {
      method: "POST",
      body: JSON.stringify({ config }),
    });
  }

  async syncModels(serviceId: string): Promise<{ status: string }> {
    return this.makeRequest<{ status: string }>(
      `/admin/services/${serviceId}/models/sync`,
      {
        method: "POST",
      },
    );
  }

  /**
   * Trigger a catalog refresh. The backend may answer immediately with the result (a cache hit,
   * or a fast provider like Ollama) or as a background job streamed via SSE (vLLM/llama.cpp,
   * which hit HuggingFace dozens-to-hundreds of times) - this resolves with the final counts
   * either way, forwarding any progress events to `onProgress` when the response does stream.
   */
  async refreshCatalog(
    serviceId: string,
    force?: boolean,
    onProgress?: (event: ProgressEvent) => void,
  ): Promise<{ added: number; total: number }> {
    const params = force ? "?force=true" : "";
    const url = `${this.baseURL}/admin/services/${serviceId}/catalog/refresh${params}`;
    const adminApiKey = AdminApiKeyStorage.get();
    const headers: Record<string, string> = {};
    if (adminApiKey) {
      headers.Authorization = `Bearer ${adminApiKey}`;
    }

    const response = await fetch(url, { method: "POST", headers });

    if (!response.ok) {
      const content = await response.text();
      try {
        const errorData = JSON.parse(content);
        const errorMessage = formatApiErrorMessage(errorData, content);
        throw new Error(`HTTP ${response.status}: ${errorMessage}`);
      } catch (e) {
        if (e instanceof SyntaxError) {
          throw new Error(
            `HTTP ${response.status}: ${content || response.statusText}`,
          );
        }
        throw e;
      }
    }

    const contentType = (response.headers.get("content-type") || "")
      .split(";")[0]
      .trim();

    if (contentType !== "text/event-stream") {
      return JSON.parse(await response.text());
    }

    let result: { added: number; total: number } | null = null;
    let errorMessage: string | null = null;
    await readSSEStream(response, (event) => {
      onProgress?.(event);
      if (event.type !== "finish") {
        return;
      }
      if (
        event.status === "ok" &&
        event.details &&
        typeof event.details === "object" &&
        typeof event.details.added === "number" &&
        typeof event.details.total === "number"
      ) {
        result = { added: event.details.added, total: event.details.total };
      } else {
        errorMessage =
          typeof event.details === "string"
            ? event.details
            : "Catalog refresh failed";
      }
    });

    if (errorMessage) {
      throw new Error(errorMessage);
    }
    if (!result) {
      throw new Error("Catalog refresh stream ended without a result");
    }
    return result;
  }

  async getDockerTags(
    serviceId: string,
    hardware?: string,
    fieldName?: string,
    modelId?: string,
    current?: string,
  ): Promise<{
    image: string;
    tags: string[];
    default: string | null;
    newest: string | null;
    newest_needs_cuda: NewerNeedsCuda | null;
    registry_unavailable: boolean;
  }> {
    const params = new URLSearchParams();
    if (hardware) params.set("hardware", hardware);
    if (fieldName) params.set("field_name", fieldName);
    if (modelId) params.set("model_id", modelId);
    if (current) params.set("current", current);
    const query = params.toString() ? `?${params.toString()}` : "";
    return this.makeRequest<{
      image: string;
      tags: string[];
      default: string | null;
      newest: string | null;
      newest_needs_cuda: NewerNeedsCuda | null;
      registry_unavailable: boolean;
    }>(`/admin/services/${encodeURIComponent(serviceId)}/docker-tags${query}`);
  }

  async getDockerLogs(
    serviceId: string,
    modelId?: string,
  ): Promise<{ logs: string }> {
    const url = modelId
      ? `/admin/services/${serviceId}/docker/logs?model_id=${encodeURIComponent(modelId)}`
      : `/admin/services/${serviceId}/docker/logs`;
    return this.makeRequest<{ logs: string }>(url);
  }

  async getDockerCompose(
    serviceId: string,
    modelId?: string,
  ): Promise<{ compose_file: string }> {
    const url = modelId
      ? `/admin/services/${serviceId}/docker/compose?model_id=${encodeURIComponent(modelId)}`
      : `/admin/services/${serviceId}/docker/compose`;
    return this.makeRequest<{ compose_file: string }>(url);
  }

  async restartDocker(
    serviceId: string,
    modelId?: string,
  ): Promise<{ status: string }> {
    const url = modelId
      ? `/admin/services/${serviceId}/docker/restart?model_id=${encodeURIComponent(modelId)}`
      : `/admin/services/${serviceId}/docker/restart`;
    return this.makeRequest<{ status: string }>(url, {
      method: "POST",
    });
  }

  async getGpuStats(): Promise<GpuStats | null> {
    try {
      return await this.makeRequest<GpuStats | null>(
        "/admin/settings/hardware/gpu-stats",
      );
    } catch {
      return null;
    }
  }

  async getSystemStats(): Promise<SystemStats> {
    return this.makeRequest<SystemStats>(
      "/admin/settings/hardware/system-stats",
    );
  }

  async getMeshInfo(): Promise<ShowMeshInfoOut> {
    return this.makeRequest<ShowMeshInfoOut>("/admin/mesh/info");
  }

  async getMeshTopology(): Promise<MeshTopologyNode[]> {
    return this.makeRequest<MeshTopologyNode[]>("/admin/mesh/topology");
  }

  async listAdminWarnings(): Promise<ListWarningsOut> {
    return this.makeRequest<ListWarningsOut>("/admin/warnings");
  }

  async dismissAdminWarning(warningId: string): Promise<void> {
    return this.makeRequest<void>(
      `/admin/warnings/${encodeURIComponent(warningId)}`,
      { method: "DELETE" },
    );
  }

  async dismissAllAdminWarnings(): Promise<void> {
    return this.makeRequest<void>("/admin/warnings", { method: "DELETE" });
  }

  async getSettings(): Promise<InfraSettings> {
    return this.makeRequest<InfraSettings>("/admin/settings");
  }

  async updateSettings(
    settings: Partial<InfraSettings>,
  ): Promise<InfraSettings> {
    return this.makeRequest<InfraSettings>("/admin/settings", {
      method: "PUT",
      body: JSON.stringify(settings),
    });
  }

  async getConfig(): Promise<ConfigOut> {
    return this.makeRequest<ConfigOut>("/admin/config");
  }

  async revealConfigEntry(key: string): Promise<ConfigRevealOut> {
    return this.makeRequest<ConfigRevealOut>(
      `/admin/config/${encodeURIComponent(key)}/reveal`,
    );
  }

  async updateDynamicConfig(
    updates: Record<string, unknown>,
  ): Promise<ConfigOut> {
    return this.makeRequest<ConfigOut>("/admin/config", {
      method: "PUT",
      body: JSON.stringify(updates),
    });
  }
}

type ApiErrorPayload = {
  error?: unknown;
  detail?: unknown;
  message?: unknown;
};

export function formatApiErrorMessage(
  errorData: unknown,
  fallback: string,
): string {
  if (!errorData || typeof errorData !== "object") return fallback;

  const data = errorData as ApiErrorPayload;
  const detail = data.detail;
  const message = data.message;

  // The server sends every error as `{"error": {"message": ...}}` (OpenAI style).
  if (data.error && typeof data.error === "object") {
    const errorMessage = (data.error as { message?: unknown }).message;
    if (typeof errorMessage === "string" && errorMessage.trim()) {
      return errorMessage;
    }
  }

  const detailString = formatDetail(detail);
  if (detailString) return detailString;

  if (typeof message === "string" && message.trim()) {
    return message;
  }

  return fallback;
}

function formatDetail(detail: unknown): string {
  if (typeof detail === "string") {
    return detail;
  }

  if (Array.isArray(detail)) {
    const parts = detail
      .map((item) => {
        if (typeof item === "string") return item;
        if (item && typeof item === "object") {
          const obj = item as Record<string, unknown>;
          const msg = obj.msg ?? obj.message;
          if (typeof msg === "string" && msg.trim()) {
            const field = obj.field;
            if (typeof field === "string" && field.trim()) {
              return `${field}: ${msg}`;
            }
            return msg;
          }
          try {
            return JSON.stringify(obj);
          } catch {
            return String(obj);
          }
        }
        return String(item);
      })
      .filter((p) => typeof p === "string" && p.trim());

    return parts.join("; ");
  }

  if (detail && typeof detail === "object") {
    try {
      return JSON.stringify(detail);
    } catch {
      return String(detail);
    }
  }

  return "";
}

// Export a default instance
export const apiClient = new DeepFellowClient();
