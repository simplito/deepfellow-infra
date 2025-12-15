/*
DeepFellow Software Framework.
Copyright © 2025 Simplito sp. z o.o.

This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
This software is Licensed under the DeepFellow Free License.

See the License for the specific language governing permissions and
limitations under the License.
*/
import type {
  ServiceListResponse,
  Service,
  ServiceModelsResponse,
  ServiceModel,
  TestResult,
  ShowMeshInfoOut,
} from "./types";
import { InstallationWarningsError } from "./types";
import { readSSEStream, type ProgressEvent } from "@/utils/sse-stream";

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
        const errorMessage =
          errorData?.detail?.[0]?.msg ||
          errorData?.detail ||
          errorData?.message ||
          content;
        throw new Error(`HTTP ${response.status}: ${errorMessage}`);
      } catch (e) {
        if (e instanceof SyntaxError) {
          throw new Error(`HTTP ${response.status}: ${content || response.statusText}`);
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

  async installAdminService(serviceId: string, spec: Record<string, any>): Promise<void> {
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
    spec: Record<string, any>,
    onProgress: (event: ProgressEvent) => void,
    ignoreWarnings = false
  ): Promise<void> {
    const url = `${this.baseURL}/admin/services/${serviceId}`;
    const adminApiKey = AdminApiKeyStorage.get();

    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };

    if (adminApiKey) {
      headers.Authorization = `Bearer ${adminApiKey}`;
    }

    const body: Record<string, any> = { stream: true, spec };
    if (ignoreWarnings) {
      body.ignore_warnings = true;
    }

    const response = await fetch(url, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const content = await response.text();
      try {
        const errorData = JSON.parse(content);
        
        // Check if this is a 400 with warnings
        if (response.status === 400 && errorData.warnings && Array.isArray(errorData.warnings)) {
          throw new InstallationWarningsError(errorData.warnings, errorData.detail || errorData.message);
        }
        
        const errorMessage =
          errorData?.detail?.[0]?.msg ||
          errorData?.detail ||
          errorData?.message ||
          content;
        throw new Error(`HTTP ${response.status}: ${errorMessage}`);
      } catch (e) {
        if (e instanceof InstallationWarningsError) {
          throw e;
        }
        if (e instanceof SyntaxError) {
          throw new Error(`HTTP ${response.status}: ${content || response.statusText}`);
        }
        throw e;
      }
    }

    const contentType = (response.headers.get("content-type") || "").split(";")[0].trim();
    if (contentType === "text/event-stream") {
      await readSSEStream(response, onProgress);
    } else {
      // Non-streaming response
      await response.json();
    }
  }

  async getServiceProgress(
    serviceId: string,
    onProgress: (event: ProgressEvent) => void
  ): Promise<void> {
    const url = `${this.baseURL}/admin/services/${serviceId}/progress`;
    const adminApiKey = AdminApiKeyStorage.get();

    const headers: Record<string, string> = {};
    if (adminApiKey) {
      headers.Authorization = `Bearer ${adminApiKey}`;
    }

    const response = await fetch(url, { headers });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    await readSSEStream(response, onProgress);
  }

  // Admin Service Models methods
  async listAdminServiceModels(serviceId: string): Promise<ServiceModelsResponse> {
    return this.makeRequest<ServiceModelsResponse>(`/admin/services/${serviceId}/models`);
  }

  async getAdminServiceModel(serviceId: string, modelId: string): Promise<ServiceModel> {
    return this.makeRequest<ServiceModel>(
      `/admin/services/${serviceId}/models/_?model_id=${encodeURIComponent(modelId)}`
    );
  }

  async installAdminServiceModel(serviceId: string, modelId: string, spec: Record<string, any>): Promise<void> {
    return this.makeRequest<void>(
      `/admin/services/${serviceId}/models/_?model_id=${encodeURIComponent(modelId)}`,
      {
        method: "POST",
        body: JSON.stringify({ spec }),
      }
    );
  }

  async uninstallAdminServiceModel(serviceId: string, modelId: string, purge = false): Promise<void> {
    return this.makeRequest<void>(
      `/admin/services/${serviceId}/models/_?model_id=${encodeURIComponent(modelId)}`,
      {
        method: "DELETE",
        body: JSON.stringify({ purge }),
      }
    );
  }

  async installAdminServiceModelStreaming(
    serviceId: string,
    modelId: string,
    spec: Record<string, any>,
    onProgress: (event: ProgressEvent) => void,
    ignoreWarnings = false
  ): Promise<void> {
    const url = `${this.baseURL}/admin/services/${serviceId}/models/_?model_id=${encodeURIComponent(modelId)}`;
    const adminApiKey = AdminApiKeyStorage.get();

    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };

    if (adminApiKey) {
      headers.Authorization = `Bearer ${adminApiKey}`;
    }

    const body: Record<string, any> = { stream: true, spec };
    if (ignoreWarnings) {
      body.ignore_warnings = true;
    }

    const response = await fetch(url, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const content = await response.text();
      try {
        const errorData = JSON.parse(content);
        
        // Check if this is a 400 with warnings
        if (response.status === 400 && errorData.warnings && Array.isArray(errorData.warnings)) {
          throw new InstallationWarningsError(errorData.warnings, errorData.detail || errorData.message);
        }
        
        const errorMessage =
          errorData?.detail?.[0]?.msg ||
          errorData?.detail ||
          errorData?.message ||
          content;
        throw new Error(`HTTP ${response.status}: ${errorMessage}`);
      } catch (e) {
        if (e instanceof InstallationWarningsError) {
          throw e;
        }
        if (e instanceof SyntaxError) {
          throw new Error(`HTTP ${response.status}: ${content || response.statusText}`);
        }
        throw e;
      }
    }

    const contentType = (response.headers.get("content-type") || "").split(";")[0].trim();
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
    onProgress: (event: ProgressEvent) => void
  ): Promise<void> {
    const url = `${this.baseURL}/admin/services/${serviceId}/models/progress?model_id=${encodeURIComponent(modelId)}`;
    const adminApiKey = AdminApiKeyStorage.get();

    const headers: Record<string, string> = {};
    if (adminApiKey) {
      headers.Authorization = `Bearer ${adminApiKey}`;
    }

    const response = await fetch(url, { headers });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    await readSSEStream(response, onProgress);
  }

  async testModel(registrationId: string, signal?: AbortSignal): Promise<TestResult> {
    return this.makeRequest<TestResult>(`/admin/services/model/test/${registrationId}`, {
      signal,
    });
  }

  async addCustomModel(serviceId: string, spec: Record<string, any>): Promise<{ custom_model_id: string }> {
    return this.makeRequest<{ custom_model_id: string }>(
      `/admin/services/${serviceId}/models/custom`,
      {
        method: "POST",
        body: JSON.stringify({ spec }),
      }
    );
  }

  async removeCustomModel(serviceId: string, customModelId: string): Promise<{ status: string }> {
    return this.makeRequest<{ status: string }>(
      `/admin/services/${serviceId}/models/custom/${customModelId}`,
      {
        method: "DELETE",
      }
    );
  }

  async getDockerLogs(serviceId: string, modelId?: string): Promise<{ logs: string }> {
    const url = modelId
      ? `/admin/services/${serviceId}/docker/logs?model_id=${encodeURIComponent(modelId)}`
      : `/admin/services/${serviceId}/docker/logs`;
    return this.makeRequest<{ logs: string }>(url);
  }

  async getDockerCompose(serviceId: string, modelId?: string): Promise<{ compose_file: string }> {
    const url = modelId
      ? `/admin/services/${serviceId}/docker/compose?model_id=${encodeURIComponent(modelId)}`
      : `/admin/services/${serviceId}/docker/compose`;
    return this.makeRequest<{ compose_file: string }>(url);
  }

  async restartDocker(serviceId: string, modelId?: string): Promise<{ status: string }> {
    const url = modelId
      ? `/admin/services/${serviceId}/docker/restart?model_id=${encodeURIComponent(modelId)}`
      : `/admin/services/${serviceId}/docker/restart`;
    return this.makeRequest<{ status: string }>(url, {
      method: "POST",
    });
  }

  async getMeshInfo(): Promise<ShowMeshInfoOut> {
    return this.makeRequest<ShowMeshInfoOut>("/admin/mesh/info");
  }
}

// Export a default instance
export const apiClient = new DeepFellowClient();
