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
} from "./types";

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
}

// Export a default instance
export const apiClient = new DeepFellowClient();
