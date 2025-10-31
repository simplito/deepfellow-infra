/*
DeepFellow Software Framework.
Copyright © 2025 Simplito sp. z o.o.

This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
This software is Licensed under the DeepFellow Free License.

See the License for the specific language governing permissions and
limitations under the License.
*/
import { useState, useEffect } from "react";
import { useRouter } from "@tanstack/react-router";
import { useQuery, useMutation } from "@tanstack/react-query";
import { AdminApiKeyStorage } from "../lib/auth";
import { apiClient } from "../deepfellow/client";
import { toast } from "sonner";

export const useAuth = () => {
  const [apiKey, setApiKey] = useState<string | null>(AdminApiKeyStorage.get());
  const router = useRouter();

  useEffect(() => {
    const key = AdminApiKeyStorage.get();
    setApiKey(key);
  }, []);

  const loginMutation = useMutation({
    mutationFn: async (key: string) => {
      AdminApiKeyStorage.set(key);
      return apiClient.listAdminServices();
    },
    onSuccess: (_, key) => {
      setApiKey(key);
      router.navigate({ to: "/dashboard" });
    },
    onError: (error) => {
      AdminApiKeyStorage.clear();
      setApiKey(null);

      const errorMessage = error instanceof Error ? error.message : "Invalid API key";
      toast.error(`Authentication failed: ${errorMessage}`);
    },
  });

  const logout = () => {
    AdminApiKeyStorage.clear();
    setApiKey(null);
    router.navigate({ to: "/sign-in" });
  };

  return {
    apiKey,
    isAuthenticated: !!apiKey,
    isLoading: loginMutation.isPending,
    login: loginMutation.mutate,
    logout,
  };
};

export const useRequireAuth = () => {
  const { apiKey, isLoading } = useAuth();
  const router = useRouter();

  const { isLoading: isValidating, error } = useQuery({
    queryKey: ["validate-auth", apiKey],
    queryFn: async () => {
      if (!apiKey) {
        throw new Error("No API key");
      }
      return apiClient.listAdminServices();
    },
    enabled: !isLoading && !!apiKey,
    retry: false,
    refetchOnWindowFocus: false,
  });

  useEffect(() => {
    if (isLoading) return;

    if (!apiKey) {
      router.navigate({ to: "/sign-in" });
      return;
    }

    if (error) {
      AdminApiKeyStorage.clear();
      toast.error("Session expired or invalid. Please sign in again.");
      router.navigate({ to: "/sign-in" });
    }
  }, [apiKey, isLoading, error, router]);

  return { apiKey, isLoading: isLoading || isValidating };
};
