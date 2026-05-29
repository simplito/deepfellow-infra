/*
DeepFellow Software Framework.
Copyright © 2025 Simplito sp. z o.o.

This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
This software is Licensed under the DeepFellow Free License.

See the License for the specific language governing permissions and
limitations under the License.
*/
import { useState, useMemo, useEffect, useRef, useDeferredValue, startTransition, memo, useCallback, type RefObject } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "@/deepfellow/client";
import type { GpuStats, GpuCardStats } from "@/deepfellow/types";
import { Tooltip, TooltipTrigger, TooltipContent, TooltipProvider } from "@/components/ui/tooltip";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
} from "@/components/ui/dropdown-menu";
import { MoreVertical, Info } from "lucide-react";
import { DynamicFormModal } from "./DynamicFormModal";
import { ConfirmModal } from "./ConfirmModal";
import { UninstallWithPurgeModal } from "./UninstallWithPurgeModal";
import { ProgressBadge } from "./ProgressBadge";
import { TestResultModal } from "./TestResultModal";
import { ContentModal } from "./ContentModal";
import { WarningsModal } from "./WarningsModal";
import { useModal } from "@/hooks/use-modal";
import type { ServiceModel } from "@/deepfellow/types";
import { InstallationWarningsError } from "@/deepfellow/types";
import { MODEL_TYPES } from "@/deepfellow/types";
import type { ProgressEvent } from "@/utils/sse-stream";
import { toast } from "sonner";
import { getStageLabel } from "@/utils/sse-stream";
import {
  clearModelInstallProgress,
  getSnapshot,
  setModelInstallProgress,
  useModelInstallProgress,
} from "@/state/install-progress-store";
import { startProgressSimulation, getStepPerTick, COMPLETION_SMOOTH_MS, COMPLETION_SMOOTH_MIN_MS } from "@/utils/progress-simulation";
import type { SimulationHandle } from "@/utils/progress-simulation";

interface ServiceModelsProps {
  serviceId: string;
}

export function ServiceModels({ serviceId }: ServiceModelsProps) {
  const modal = useModal();
  const [filterText, setFilterText] = useState("");
  const deferredFilterText = useDeferredValue(filterText);
  const [filterType, setFilterType] = useState<string>("__all");
  const [filterInstalled, setFilterInstalled] = useState<string>("__all");
  const [filterDownloaded, setFilterDownloaded] = useState<string>("__all");
  const [filterCustom, setFilterCustom] = useState<string>("__all");
  const [showEntrySkeleton, setShowEntrySkeleton] = useState(true);
  const [installingModelId, setInstallingModelId] = useState<string | null>(null);
  const pendingInstallationRef = useRef<{ modelId: string; spec: Record<string, unknown>; size?: string } | null>(null);
  const lastAddedCustomModelIdRef = useRef<string | null>(null);
  const hasWarningsRef = useRef(false);
  const simulationStopFnsRef = useRef<Record<string, SimulationHandle>>({});
  const hasRealProgressByModelRef = useRef<Record<string, boolean>>({});
  const toastIdsRef = useRef<Record<string, string | number>>({});
  const restartDockerToastIdRef = useRef<string | number | null>(null);
  const testAbortControllerRef = useRef<AbortController | null>(null);
  const queryClient = useQueryClient();

  const serviceInfoQuery = useQuery({
    queryKey: ["admin", "services", serviceId],
    queryFn: () => apiClient.getAdminService(serviceId),
    refetchInterval: 10_000,
    refetchIntervalInBackground: false,
  });

  const serviceInfo = serviceInfoQuery.data;

  const isOllamaExternal = serviceInfo?.type === "ollama-external";

  const isCpuOnly = (() => {
    const installed = serviceInfo?.installed;
    if (!installed || typeof installed !== "object") return false;
    const spec = installed as Record<string, unknown>;
    if ("stage" in spec) return false;
    return spec.hardware === "CPU" || spec.hardware === false;
  })();

  const modelsQuery = useQuery({
    queryKey: ["admin", "services", serviceId, "models"],
    queryFn: () => apiClient.listAdminServiceModels(serviceId),
    refetchInterval: isOllamaExternal ? 60_000 : 10_000,
    refetchIntervalInBackground: false,
  });

  const modelsData = modelsQuery.data;

  const gpuStatsQuery = useQuery<GpuStats | null>({
    queryKey: ["admin", "settings", "hardware", "gpu-stats"],
    queryFn: () => apiClient.getGpuStats(),
    retry: false,
    refetchInterval: 10_000,
  });


  // Always show skeleton on entry to this page, even if React Query has cached data.
  // This avoids a "blank/lag" feel on subsequent navigations.
  useEffect(() => {
    void serviceId;
    setShowEntrySkeleton(true);
  }, [serviceId]);

  const isEntryBusy =
    modelsQuery.isLoading ||
    modelsQuery.isFetching ||
    serviceInfoQuery.isLoading ||
    serviceInfoQuery.isFetching;

  useEffect(() => {
    if (!showEntrySkeleton) return;
    if (isEntryBusy) return;
    const t = window.setTimeout(() => setShowEntrySkeleton(false), 120);
    return () => window.clearTimeout(t);
  }, [showEntrySkeleton, isEntryBusy]);

  // Clean up any running simulations on unmount
  useEffect(() => {
    return () => {
      for (const sim of Object.values(simulationStopFnsRef.current)) sim.stop();
    };
  }, []);

  // Progress polling for existing installations
  useEffect(() => {
    if (!modelsData?.list) return;

    // Reconcile: clear progress for models that are no longer in-progress.
    const inProgressKeys = new Set<string>();
    for (const model of modelsData.list) {
      const inst = model.installed;
      if (!inst || typeof inst !== "object") continue;
      const s = (inst as { stage?: unknown }).stage;
      const v = (inst as { value?: unknown }).value;
      if ((s === "install" || s === "download") && typeof v === "number" && Number.isFinite(v)) {
        inProgressKeys.add(`${serviceId}::${model.id}`);
      }
    }
    const prefix = `${serviceId}::`;
    for (const key of Object.keys(getSnapshot().models)) {
      if (key.startsWith(prefix) && !inProgressKeys.has(key)) {
        clearModelInstallProgress(serviceId, key.slice(prefix.length));
      }
    }

    const cleanups: Array<() => void> = [];

    for (const model of modelsData.list) {
      const installed = model.installed;
      if (!installed || typeof installed !== "object") continue;

      const installedStage = (installed as { stage?: unknown }).stage;
      const installedValue = (installed as { value?: unknown }).value;
      const installedIsProgress =
        (installedStage === "install" || installedStage === "download") &&
        typeof installedValue === "number" &&
        Number.isFinite(installedValue);

      if (!installedIsProgress) continue;

      const modelId = model.id;
      const simKey = `${serviceId}::${modelId}`;
      // Skip if already tracked by an active install mutation.
      if (simulationStopFnsRef.current[simKey]) continue;

      let currentStage: "install" | "download" = installedStage as "install" | "download";
      const abortController = new AbortController();
      let isCancelled = false;

      // Use existing store value or backend value as starting point.
      const existingProgress = getSnapshot().models[simKey];
      const startValue = existingProgress?.value ?? (installedValue as number);
      if (!existingProgress) {
        setModelInstallProgress(serviceId, modelId, { stage: currentStage, value: installedValue as number });
      }

      const installStartTime = Date.now();
      const sim = startProgressSimulation({
        stepPerTick: getStepPerTick(model.size, serviceId === "vllm" ? 0.2 : 1.6),
        initialValue: startValue,
        onTick: (value) => {
          if (isCancelled) return;
          setModelInstallProgress(serviceId, modelId, { stage: currentStage, value });
        },
      });
      simulationStopFnsRef.current[simKey] = sim;

      apiClient
        .getModelProgress(
          serviceId,
          modelId,
          (event: ProgressEvent) => {
            if (isCancelled) return;

            const stage = event.stage;
            const value = event.value;

            if (event.type === "progress" && stage && value !== undefined) {
              currentStage = stage;
              hasRealProgressByModelRef.current[modelId] = true;
              setModelInstallProgress(serviceId, modelId, { stage, value });
              return;
            }

            if (event.type === "finish") {
              if (event.status === "ok") {
                sim.smoothComplete(
                  Math.max(COMPLETION_SMOOTH_MIN_MS, Math.min(Date.now() - installStartTime, COMPLETION_SMOOTH_MS)),
                  () => {
                    delete simulationStopFnsRef.current[simKey];
                    queryClient.invalidateQueries({ queryKey: ["admin", "services", serviceId, "models"] });
                  },
                  getSnapshot().models[simKey]?.value
                );
              } else {
                sim.stop();
                delete simulationStopFnsRef.current[simKey];
                clearModelInstallProgress(serviceId, modelId);
                toast.error(`Installation failed for ${modelId}: ${event.details || "Unknown error"}`);
              }
            }
          },
          abortController.signal
        )
        .catch((error) => {
          if (isCancelled) return;
          if (error instanceof Error && error.name === "AbortError") return;
          if (!isCancelled) {
            console.error(`Error polling progress for ${modelId}:`, error);
          }
        });

      cleanups.push(() => {
        isCancelled = true;
        abortController.abort();
        sim.stop();
        delete simulationStopFnsRef.current[simKey];
        delete hasRealProgressByModelRef.current[modelId];
        // Progress is NOT cleared here — reconciliation at the top handles cleanup.
      });
    }

    return () => {
      for (const cleanup of cleanups) cleanup();
    };
  }, [modelsData, serviceId, queryClient]);

  const installMutation = useMutation({
    mutationFn: async ({ modelId, spec, size, ignoreWarnings = false }: { modelId: string; spec: Record<string, unknown>; size?: string; ignoreWarnings?: boolean }) => {
      try {
        return await new Promise<void>((resolve, reject) => {
          let currentStage: "install" | "download" = "download";
          const simKey = `${serviceId}::${modelId}`;
          const installStartTime = Date.now();
          const sim = startProgressSimulation({
            stepPerTick: getStepPerTick(size ?? ""),
            onTick: (value) => {
              setModelInstallProgress(serviceId, modelId, { stage: currentStage, value });
              const toastId = toastIdsRef.current[modelId];
              if (toastId && !hasRealProgressByModelRef.current[modelId]) {
                toast.loading(`${getStageLabel(currentStage)} ${modelId}: ${(value * 100).toFixed(1)}%`, { id: toastId });
              }
            },
          });
          simulationStopFnsRef.current[simKey] = sim;

          apiClient.installAdminServiceModelStreaming(
            serviceId,
            modelId,
            spec,
            (event: ProgressEvent) => {
              const stage = event.stage;
              const value = event.value;

              if (event.type === "progress" && stage && value !== undefined) {
                currentStage = stage;
                hasRealProgressByModelRef.current[modelId] = true;
                setModelInstallProgress(serviceId, modelId, { stage, value });
                const toastId = toastIdsRef.current[modelId];
                if (toastId) {
                  const percentage = (value * 100).toFixed(1);
                  const stageLabel = getStageLabel(stage);
                  toast.loading(`${stageLabel} ${modelId}: ${percentage}%`, { id: toastId });
                }
              } else if (event.type === "finish") {
                if (event.status === "ok") {
                  sim.smoothComplete(
                    Math.max(COMPLETION_SMOOTH_MIN_MS, Math.min(Date.now() - installStartTime, COMPLETION_SMOOTH_MS)),
                    () => {
                      delete simulationStopFnsRef.current[simKey];
                      // Do NOT clear progress here — reconciliation clears it once the
                      // backend confirms the model is no longer in-progress. Clearing early
                      // would cause the bar to jump back to the stale backend value.
                      const toastId = toastIdsRef.current[modelId];
                      if (toastId) {
                        toast.success(`Model ${modelId} installed successfully`, { id: toastId });
                        delete toastIdsRef.current[modelId];
                      }
                      resolve();
                    },
                    getSnapshot().models[simKey]?.value
                  );
                } else {
                  sim.stop();
                  delete simulationStopFnsRef.current[simKey];
                  reject(new Error(event.details || "Installation failed"));
                }
              }
            },
            ignoreWarnings
          ).catch(reject);
        });
      }
      catch (e) {
        clearModelInstallProgress(serviceId, modelId);
        const toastId = toastIdsRef.current[modelId];
        if (toastId) {
          toast.error(`Failed to install model ${modelId}: ${(e instanceof Error ? e.message : "") || "Installation failed"}`, { id: toastId });
          delete toastIdsRef.current[modelId];
        }
      }
    },
    onMutate: ({ modelId }) => {
      setInstallingModelId(modelId);
    },
    onSuccess: (_data, variables) => {
      const simKey = `${serviceId}::${variables.modelId}`;
      delete simulationStopFnsRef.current[simKey];
      delete hasRealProgressByModelRef.current[variables.modelId];
      queryClient.invalidateQueries({ queryKey: ["admin", "services", serviceId, "models"] });
      pendingInstallationRef.current = null;
    },
    onError: (error, variables) => {
      const simKey = `${serviceId}::${variables.modelId}`;
      const simStop = simulationStopFnsRef.current[simKey];
      if (simStop) {
        simStop.stop();
        delete simulationStopFnsRef.current[simKey];
      }
      delete hasRealProgressByModelRef.current[variables.modelId];

      if (error instanceof InstallationWarningsError) {
        // Show warnings modal instead of error toast
        hasWarningsRef.current = true;

        const toastId = toastIdsRef.current[variables.modelId];
        if (toastId) {
          toast.loading(`Warnings for ${variables.modelId}: awaiting confirmation...`, { id: toastId });
        }

        modal.open(WarningsModal, {
          warnings: error.warnings,
          onContinue: handleWarningsContinue,
          isLoading: false,
        });
        return;
      }

      hasWarningsRef.current = false;
      const modelId = variables.modelId;
      // Finish toast is handled in the SSE callback. If we failed before streaming starts,
      // fall back to a plain error toast.
      if (!toastIdsRef.current[modelId]) {
        toast.error(`Failed to install model: ${error.message}`);
      }
      clearModelInstallProgress(serviceId, modelId);
    },
    onSettled: () => {
      // Only reset if not showing warnings modal
      if (!hasWarningsRef.current) {
        setInstallingModelId(null);
      }
    },
  });

  const uninstallMutation = useMutation({
    mutationFn: (modelId: string) => apiClient.uninstallAdminServiceModel(serviceId, modelId, false),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["admin", "services", serviceId, "models"] });
      modal.close();
      toast.success("Model uninstalled successfully");
    },
    onError: (error) => {
      toast.error(`Failed to uninstall model: ${error.message}`);
    },
  });

  const purgeMutation = useMutation({
    mutationFn: (modelId: string) => apiClient.uninstallAdminServiceModel(serviceId, modelId, true),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["admin", "services", serviceId, "models"] });
      modal.close();
      toast.success("Model purged successfully");
    },
    onError: (error) => {
      toast.error(`Failed to purge model: ${error.message}`);
    },
  });

  const testMutation = useMutation({
    mutationFn: async (registrationId: string) => {
      // Create a new AbortController for this test run
      const abortController = new AbortController();
      testAbortControllerRef.current = abortController;
      return apiClient.testModel(registrationId, abortController.signal);
    },
    onMutate: () => {
      // Open modal immediately with loading state
      modal.open(TestResultModal, { 
        result: {},
        isLoading: true,
        onCancel: () => {
          testAbortControllerRef.current?.abort();
          testMutation.reset();
          testAbortControllerRef.current = null;
          modal.close();
        }
      });
    },
    onSuccess: (result) => {
      // Clear the abort controller ref
      testAbortControllerRef.current = null;
      queryClient.invalidateQueries({ queryKey: ["admin", "services", serviceId, "models"] });
      // Update modal with actual result
      modal.open(TestResultModal, {
        result,
        isLoading: false,
        onCancel: () => {
          testMutation.reset();
          modal.close();
        }
      });
    },
    onError: (error) => {
      // Clear the abort controller ref
      testAbortControllerRef.current = null;
      // If the error is from user cancellation, just close the modal
      if (error instanceof Error && error.name === 'AbortError') {
        testMutation.reset();
        modal.close();
        return;
      }
      queryClient.invalidateQueries({ queryKey: ["admin", "services", serviceId, "models"] });
      // Show error state in modal
      modal.open(TestResultModal, { 
        result: {
          error: true,
          details: { message: error.message }
        },
        isLoading: false,
        onCancel: () => {
          testMutation.reset();
          modal.close();
        }
      });
    },
  });

  const addCustomModelMutation = useMutation({
    mutationFn: async (spec: Record<string, unknown>) => {
      return apiClient.addCustomModel(serviceId, spec);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["admin", "services", serviceId, "models"] });
      modal.close();
      toast.success("Custom model added successfully");


      const addedId = lastAddedCustomModelIdRef.current;
      if (addedId) {
        setFilterText(addedId);
        setTimeout(() => {
          const input = document.getElementById("search-models") as HTMLInputElement | null;
          input?.focus();
        }, 0);
      }
    },
    onError: (error) => {
      toast.error(`Failed to add custom model: ${error.message}`);
    },
  });

  const removeCustomModelMutation = useMutation({
    mutationFn: async ({ customModelId }: { customModelId: string }) => {
      return apiClient.removeCustomModel(serviceId, customModelId);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["admin", "services", serviceId, "models"] });
      modal.close();  
      toast.success("Custom model removed successfully");
    },
    onError: (error) => {
      // Check if this is a validation error about the model being in use
      const errorMessage = error.message || "";
      if (errorMessage.includes("Cannot remove custom model") || errorMessage.includes("it is in use")) {
        toast.error("Cannot remove custom model: it is currently installed. Please uninstall it first.");
      } else {
        toast.error(`Failed to remove custom model: ${errorMessage}`);
      }
    },
  });

  const syncModelsMutation = useMutation({
    mutationFn: async () => {
      return apiClient.syncModels(serviceId);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["admin", "services", serviceId, "models"] });
      toast.success("Models synced successfully");
    },
    onError: (error) => {
      toast.error(`Failed to sync models: ${error.message}`);
    },
  });

  const dockerRestartMutation = useMutation({
    mutationFn: (modelId: string) => apiClient.restartDocker(serviceId, modelId),
    onMutate: () => {
      const toastId = toast.loading("Restarting Docker...");
      restartDockerToastIdRef.current = toastId;
    },
    onSuccess: () => {
      if (restartDockerToastIdRef.current) {
        toast.success("Docker restarted successfully", { id: restartDockerToastIdRef.current });
        restartDockerToastIdRef.current = null;
      } else {
        toast.success("Docker restarted successfully");
      }
    },
    onError: (error) => {
      if (restartDockerToastIdRef.current) {
        toast.error(`Failed to restart Docker: ${error.message}`, { id: restartDockerToastIdRef.current });
        restartDockerToastIdRef.current = null;
      } else {
        toast.error(`Failed to restart Docker: ${error.message}`);
      }
    },
  });

  const handleShowDockerLogs = useCallback(async (modelId: string) => {
    // Open modal immediately with loading state
    modal.open(ContentModal, {
      title: "Docker Logs",
      content: "",
      wide: true,
      pre: true,
      isLoading: true,
      onCancel: () => {
        modal.close();
      },
    });

    try {
      const data = await apiClient.getDockerLogs(serviceId, modelId);
      // Update modal with actual content
      modal.open(ContentModal, {
        title: "Docker Logs",
        content: data.logs,
        wide: true,
        pre: true,
        isLoading: false,
      });
    } catch (error) {
      modal.close();
      toast.error(`Failed to fetch Docker logs: ${error instanceof Error ? error.message : "Unknown error"}`);
    }
  }, [modal, serviceId]);

  const handleShowDockerCompose = useCallback(async (modelId: string) => {
    // Open modal immediately with loading state
    modal.open(ContentModal, {
      title: "Docker Compose File",
      content: "",
      wide: true,
      pre: true,
      isLoading: true,
      onCancel: () => {
        modal.close();
      },
    });

    try {
      const data = await apiClient.getDockerCompose(serviceId, modelId);
      // Update modal with actual content
      modal.open(ContentModal, {
        title: "Docker Compose File",
        content: data.compose_file,
        wide: true,
        pre: true,
        isLoading: false,
      });
    } catch (error) {
      modal.close();
      toast.error(`Failed to fetch Docker compose file: ${error instanceof Error ? error.message : "Unknown error"}`);
    }
  }, [modal, serviceId]);

  const handleRestartDocker = useCallback((modelId: string) => {
    modal.open(ConfirmModal, {
      title: "Restart Docker",
      description: `Are you sure you want to restart Docker for model ${modelId}?`,
      confirmText: "Restart",
      cancelText: "Cancel",
      onConfirm: () => {
        modal.close();
        dockerRestartMutation.mutate(modelId);
      },
      isLoading: dockerRestartMutation.isPending,
      variant: "warning",
    });
  }, [modal, dockerRestartMutation.isPending, dockerRestartMutation.mutate]);

  const handleInstallClick = useCallback(async (model: ServiceModel) => {
    modal.open(DynamicFormModal, {
      title: `Install ${model.id}`,
      fields: [],
      isLoading: true,
      isSubmitting: false,
      onSubmit: () => {},
    });

    try {
      const modelDetail = await apiClient.getAdminServiceModel(serviceId, model.id);
      modal.open(DynamicFormModal, {
        title: `Install ${modelDetail.id}`,
        fields: modelDetail.spec.fields,
        onSubmit: (spec: Record<string, unknown>) => {
          const cleanedSpec = Object.fromEntries(
            Object.entries(spec).filter(([_, value]) => value !== null && value !== undefined)
          ) as Record<string, unknown>;
          pendingInstallationRef.current = { modelId: modelDetail.id, spec: cleanedSpec, size: modelDetail.size };
          modal.close();
          const toastId: string | number = toast.loading(`Starting installation for ${modelDetail.id}...`);
          toastIdsRef.current[modelDetail.id] = toastId;
          installMutation.mutate({ modelId: modelDetail.id, spec: cleanedSpec, size: modelDetail.size });
        },
        // Keep modal interactive; don't disable because another install is running.
        isSubmitting: false,
      });
    } catch {
      modal.close();
      toast.error("Failed to load model details");
    }
  }, [modal, serviceId, installMutation.mutate]);

  const handleWarningsContinue = useCallback(() => {
    if (pendingInstallationRef.current) {
      hasWarningsRef.current = false;
      installMutation.mutate({
        modelId: pendingInstallationRef.current.modelId,
        spec: pendingInstallationRef.current.spec,
        size: pendingInstallationRef.current.size,
        ignoreWarnings: true
      });
    }
  }, [installMutation.mutate]);

  const handleUninstallClick = useCallback((modelId: string) => {
    modal.open(UninstallWithPurgeModal, {
      title: "Uninstall Model",
      description: `Are you sure you want to uninstall ${modelId}? This action cannot be undone.`,
      confirmText: "Uninstall",
      cancelText: "Cancel",
      purgeLabel: "Purge",
      purgeDescription: "Also remove downloaded model files and local data. This cannot be undone.",
      onConfirm: (purge) => {
        modal.close();
        if (purge) {
          purgeMutation.mutate(modelId);
        } else {
          uninstallMutation.mutate(modelId);
        }
      },
      isLoading: uninstallMutation.isPending || purgeMutation.isPending,
      variant: "destructive",
    });
  }, [
    modal,
    uninstallMutation.isPending,
    purgeMutation.isPending,
    uninstallMutation.mutate,
    purgeMutation.mutate,
  ]);

  const handlePurgeClick = useCallback((modelId: string) => {
    modal.open(ConfirmModal, {
      title: "Purge Model",
      description: "This will remove all downloaded files for the model. This action cannot be undone.",
      confirmText: "Purge",
      cancelText: "Cancel",
      onConfirm: () => {
        modal.close();
        purgeMutation.mutate(modelId);
      },
      isLoading: purgeMutation.isPending,
      variant: "destructive",
    });
  }, [modal, purgeMutation.isPending, purgeMutation.mutate]);

  const handleTestClick = useCallback((model: ServiceModel) => {
    const installedInfo =
      model.installed && typeof model.installed === "object" ? model.installed : null;

    if (!installedInfo?.registration_id) {
      toast.error("Model registration ID not found. Cannot test model.");
      return;
    }
    testMutation.mutate(installedInfo.registration_id);
  }, [testMutation.mutate]);

  const handleAddCustomModel = () => {
    if (!serviceInfo?.custom_model_spec) return;
    const fields = serviceInfo.custom_model_spec.fields;

    modal.open(DynamicFormModal, {
      title: "Add custom model",
      fields,
      deferRender: true,
      submitLabel: "Add",
      submittingLabel: "Adding...",
      onSubmit: (spec: Record<string, unknown>) => {
        const cleanedSpec = Object.fromEntries(
          Object.entries(spec).filter(([_, value]) => value !== null && value !== undefined)
        ) as Record<string, unknown>;

        const maybeId = cleanedSpec.id;
        lastAddedCustomModelIdRef.current =
          typeof maybeId === "string" && maybeId.trim() ? maybeId.trim() : null;
        addCustomModelMutation.mutate(cleanedSpec);
      },
      isSubmitting: addCustomModelMutation.isPending,
    });
  };

  const handleRemoveCustomModelClick = useCallback((model: ServiceModel) => {
    const customModelId = model.custom;
    if (!customModelId) {
      toast.error("Model is not a custom model.");
      return;
    }
    if (model.installed) {
      toast.error("Cannot remove custom model: it is currently installed. Please uninstall it first.");
      return;
    }
    modal.open(ConfirmModal, {
      title: "Remove Custom Model",
      description: `Are you sure you want to remove the custom model ${model.id}? Only uninstalled custom models can be removed. This action cannot be undone.`,
      confirmText: "Remove",
      cancelText: "Cancel",
      onConfirm: () => removeCustomModelMutation.mutate({ customModelId }),
      isLoading: removeCustomModelMutation.isPending,
      variant: "destructive",
    });
  }, [modal, removeCustomModelMutation.isPending, removeCustomModelMutation.mutate]);

  const sortedModels = useMemo(() => {
    if (!modelsData?.list) return [];

    return [...modelsData.list].sort((a, b) => {
      if (a.installed !== b.installed) {
        return a.installed ? -1 : 1;
      }
      if (a.type !== b.type) {
        return a.type.localeCompare(b.type);
      }
      return a.id.localeCompare(b.id);
    });
  }, [modelsData]);

  const filteredModels = useMemo(() => {
    if (!sortedModels) return [];

    const normalizedFilterText = deferredFilterText.toLowerCase();

    return sortedModels.filter((model) => {
      const matchesText = !normalizedFilterText || model.id.toLowerCase().includes(normalizedFilterText);
      const matchesType = filterType === "__all" || model.type === filterType;
      const matchesInstalled =
        filterInstalled === "__all" ||
        (filterInstalled === "installed" && model.installed) ||
        (filterInstalled === "notinstalled" && !model.installed);

      const matchesDownloaded =
        filterDownloaded === "__all" ||
        (filterDownloaded === "downloaded" && !!model.downloaded) ||
        (filterDownloaded === "notdownloaded" && !model.downloaded);
      const matchesCustom =
        filterCustom === "__all" ||
        (filterCustom === "onlycustom" && !!model.custom) ||
        (filterCustom === "onlynotcustom" && !model.custom);

      return matchesText && matchesType && matchesInstalled && matchesDownloaded && matchesCustom;
    });
  }, [sortedModels, deferredFilterText, filterType, filterInstalled, filterDownloaded, filterCustom]);

  if (showEntrySkeleton) {
    return <ServiceModelsSkeleton />;
  }

  return (
    <div className="w-full mx-auto p-6">
      <div className="mb-6">


        <div className="flex items-center justify-between mb-6">
          <h1 className="text-3xl font-bold">Models for {serviceId}</h1>
          <div className="flex gap-2">
            {isOllamaExternal && (
              <Button
                variant="outline"
                onClick={() => syncModelsMutation.mutate()}
                disabled={syncModelsMutation.isPending}
              >
                {syncModelsMutation.isPending ? "Syncing…" : "↺ Sync"}
              </Button>
            )}
            {serviceInfo?.custom_model_spec && (
              <Button onClick={handleAddCustomModel}>
                Add custom model
              </Button>
            )}
          </div>
        </div>

        <div className="flex flex-col md:flex-row gap-4">
          <div className="md:flex-1">
            <Label htmlFor="search-models" className="sr-only">Search models</Label>
            <Input
              id="search-models"
              placeholder="Search models..."
              value={filterText}
              onChange={(e) => setFilterText(e.target.value)}
            />
          </div>
          <div className="flex gap-4">
            <div className="space-y-1">
              <Select value={filterType} onValueChange={(v) => startTransition(() => setFilterType(v))}>
                <SelectTrigger id="filter-type" className="w-[200px]">
                  <SelectValue placeholder="Type" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="__all">All types</SelectItem>
                  {Object.entries(MODEL_TYPES).map(([key, label]) => (
                    <SelectItem key={key} value={key}>
                      {label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-1">
              <Select value={filterInstalled} onValueChange={(v) => startTransition(() => setFilterInstalled(v))}>
                <SelectTrigger id="filter-installed" className="w-[200px]">
                  <SelectValue placeholder="Installation status" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="__all">All statuses</SelectItem>
                  <SelectItem value="installed">Installed</SelectItem>
                  <SelectItem value="notinstalled">Not installed</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-1">
              <Select value={filterDownloaded} onValueChange={(v) => startTransition(() => setFilterDownloaded(v))}>
                <SelectTrigger id="filter-downloaded" className="w-[200px]">
                  <SelectValue placeholder="Downloaded" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="__all">All</SelectItem>
                  <SelectItem value="downloaded">Only downloaded</SelectItem>
                  <SelectItem value="notdownloaded">Only not downloaded</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {serviceInfo?.custom_model_spec && (
              <div className="space-y-1">
                  <Select value={filterCustom} onValueChange={(v) => startTransition(() => setFilterCustom(v))}>
                  <SelectTrigger id="filter-custom" className="w-[200px]">
                    <SelectValue placeholder="Custom" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="__all">All models</SelectItem>
                    <SelectItem value="onlycustom">Only custom</SelectItem>
                    <SelectItem value="onlynotcustom">Only not custom</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            )}
          </div>
        </div>
      </div>

      {gpuStatsQuery.data != null && !gpuStatsQuery.isError && (
        <GpuStatsPanel stats={gpuStatsQuery.data} />
      )}

      <ModelsTable
        models={filteredModels}
        serviceId={serviceId}
        isCpuOnly={isCpuOnly}
        installingModelId={installingModelId}
        isInstallingAny={installMutation.isPending}
        isPurgePending={purgeMutation.isPending}
        isRemoveCustomPending={removeCustomModelMutation.isPending}
        isTestPending={testMutation.isPending}
        hasRealProgressByModelRef={hasRealProgressByModelRef}
        onInstallClick={handleInstallClick}
        onRemoveCustomModelClick={handleRemoveCustomModelClick}
        onPurgeClick={handlePurgeClick}
        onTestClick={handleTestClick}
        onShowDockerLogs={handleShowDockerLogs}
        onShowDockerCompose={handleShowDockerCompose}
        onRestartDocker={handleRestartDocker}
        onUninstallClick={handleUninstallClick}
      />
    </div>
  );
}

const EMPTY_SPEC: Record<string, unknown> = {};

type ModelsTableProps = {
  models: ServiceModel[];
  serviceId: string;
  isCpuOnly: boolean;
  installingModelId: string | null;
  isInstallingAny: boolean;
  isRemoveCustomPending: boolean;
  isPurgePending: boolean;
  isTestPending: boolean;
  hasRealProgressByModelRef: RefObject<Record<string, boolean>>;
  onInstallClick: (model: ServiceModel) => void | Promise<void>;
  onRemoveCustomModelClick: (model: ServiceModel) => void;
  onPurgeClick: (modelId: string) => void;
  onTestClick: (model: ServiceModel) => void;
  onShowDockerLogs: (modelId: string) => void | Promise<void>;
  onShowDockerCompose: (modelId: string) => void | Promise<void>;
  onRestartDocker: (modelId: string) => void;
  onUninstallClick: (modelId: string) => void;
};

const ModelsTable = memo(function ModelsTable({
  models,
  serviceId,
  isCpuOnly,
  installingModelId,
  isInstallingAny,
  isRemoveCustomPending,
  isPurgePending,
  isTestPending,
  hasRealProgressByModelRef,
  onInstallClick,
  onRemoveCustomModelClick,
  onPurgeClick,
  onTestClick,
  onShowDockerLogs,
  onShowDockerCompose,
  onRestartDocker,
  onUninstallClick,
}: ModelsTableProps) {
  return (
    <div className="border rounded-lg">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead className="w-[40%]">Model ID</TableHead>
            <TableHead>Type</TableHead>
            <TableHead className="min-w-[150px]">Status</TableHead>
            <TableHead>Size</TableHead>
            <TableHead>
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <span className="flex items-center gap-1 cursor-default">
                      {isCpuOnly ? "RAM (est.)" : "VRAM (est.)"} <Info className="h-3 w-3 text-muted-foreground" />
                    </span>
                  </TooltipTrigger>
                  <TooltipContent>
                    {isCpuOnly
                      ? "Estimate based on the model architecture. Values are in GiB (binary gigabytes)."
                      : "Estimate based on the model architecture. Values are in GiB (binary gigabytes). Does not account for the CUDA context overhead on the GPU."}
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </TableHead>
            <TableHead>Configuration</TableHead>
            <TableHead className="text-right min-w-[165px]">Actions</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {models.length === 0 ? (
            <TableRow>
              <TableCell colSpan={7} className="text-center text-muted-foreground">
                No models found
              </TableCell>
            </TableRow>
          ) : (
            models.map((model) => (
              <ModelRow
                key={model.id}
                model={model}
                serviceId={serviceId}
                isCpuOnly={isCpuOnly}
                installingModelId={installingModelId}
                isInstallingAny={isInstallingAny}
                isRemoveCustomPending={isRemoveCustomPending}
                isPurgePending={isPurgePending}
                isTestPending={isTestPending}
                hasRealProgressByModelRef={hasRealProgressByModelRef}
                onInstallClick={onInstallClick}
                onRemoveCustomModelClick={onRemoveCustomModelClick}
                onPurgeClick={onPurgeClick}
                onTestClick={onTestClick}
                onShowDockerLogs={onShowDockerLogs}
                onShowDockerCompose={onShowDockerCompose}
                onRestartDocker={onRestartDocker}
                onUninstallClick={onUninstallClick}
              />
            ))
          )}
        </TableBody>
      </Table>
    </div>
  );
});

type ModelRowProps = {
  model: ServiceModel;
  serviceId: string;
  isCpuOnly: boolean;
  installingModelId: string | null;
  isInstallingAny: boolean;
  isRemoveCustomPending: boolean;
  isPurgePending: boolean;
  isTestPending: boolean;
  hasRealProgressByModelRef: RefObject<Record<string, boolean>>;
  onInstallClick: (model: ServiceModel) => void | Promise<void>;
  onRemoveCustomModelClick: (model: ServiceModel) => void;
  onPurgeClick: (modelId: string) => void;
  onTestClick: (model: ServiceModel) => void;
  onShowDockerLogs: (modelId: string) => void | Promise<void>;
  onShowDockerCompose: (modelId: string) => void | Promise<void>;
  onRestartDocker: (modelId: string) => void;
  onUninstallClick: (modelId: string) => void;
};

const ModelRow = memo(function ModelRow({
  model,
  serviceId,
  isCpuOnly,
  installingModelId,
  isInstallingAny,
  isRemoveCustomPending,
  isPurgePending,
  isTestPending,
  hasRealProgressByModelRef,
  onInstallClick,
  onRemoveCustomModelClick,
  onPurgeClick,
  onTestClick,
  onShowDockerLogs,
  onShowDockerCompose,
  onRestartDocker,
  onUninstallClick,
}: ModelRowProps) {
  const isInstalled = !!model.installed;
  const isDownloaded = !!model.downloaded;
  const installedInfo = model.installed && typeof model.installed === "object" ? model.installed : null;
  const installedSpec = installedInfo?.spec ?? EMPTY_SPEC;

  const currentProgress = useModelInstallProgress(serviceId, model.id);
  const isInProgress = !!currentProgress;
  const hasProgressStage = !!installedInfo?.stage && installedInfo?.value !== undefined;
  const isInstallingCurrent = isInstallingAny && installingModelId === model.id;

  const installedSpecEntries = useMemo(() => {
    if (!isInstalled || hasProgressStage) return [] as Array<{ key: string; displayValue: string }>;
    const entries = Object.entries(installedSpec);
    if (entries.length === 0) return [] as Array<{ key: string; displayValue: string }>;

    const fieldTypeByName = new Map<string, string>();
    for (const f of model.spec.fields) fieldTypeByName.set(f.name, f.type);

    return entries.map(([key, value]) => {
      const type = fieldTypeByName.get(key);
      return {
        key,
        displayValue: type === "password" ? "•••••" : ((value && typeof(value) === "object") ? JSON.stringify(value) : String(value ?? "")),
      };
    });
  }, [installedSpec, isInstalled, hasProgressStage, model.spec.fields]);

  return (
    <TableRow>
      <TableCell className="font-semibold">
        <div className="flex items-center gap-2">
          <div className="truncate max-w-md" title={model.id}>
            {model.id}
          </div>
          {model.custom && <Badge variant="secondary">Custom</Badge>}
        </div>
      </TableCell>
      <TableCell className="text-sm">{MODEL_TYPES[model.type] || model.type}</TableCell>
      <TableCell>
        {isInProgress || hasProgressStage ? (
          <ProgressBadge
            stage={currentProgress?.stage || (installedInfo?.stage as "install" | "download")}
            value={currentProgress?.value ?? installedInfo?.value ?? 0}
            variant="default"
            simulated={!hasRealProgressByModelRef.current?.[model.id]}
          />
        ) : (
          <div className="flex flex-wrap items-center gap-2">
            <Badge variant={isInstalled ? "default" : "secondary"}>{isInstalled ? "Installed" : "Not installed"}</Badge>
            {!isInstalled && isDownloaded && <Badge variant="outline">Downloaded</Badge>}
            {isInstalled && model.is_loaded === true && <Badge variant="outline">{isCpuOnly ? "In RAM" : "In VRAM"}</Badge>}
          </div>
        )}
      </TableCell>
      <TableCell className="font-mono text-sm">{model.size || "N/A"}</TableCell>
      <TableCell className="font-mono text-sm">
        {isInstalled
          ? (model.vram_estimate_gb != null ? `${model.vram_estimate_gb.toFixed(1)}GB` : "—")
          : null}
      </TableCell>
      <TableCell>
        {isInstalled && installedSpecEntries.length > 0 && !hasProgressStage ? (
          <div className="space-y-1">
            {installedSpecEntries.map(({ key, displayValue }) => (
              <div key={key} className="text-xs truncate max-w-xs">
                <span className="font-medium">{key}:</span> {displayValue}
              </div>
            ))}
          </div>
        ) : (
          <span className="text-sm text-muted-foreground">—</span>
        )}
      </TableCell>
      <TableCell className="text-right" style={({height: "49px"})}>
        {isInProgress || hasProgressStage ? null : !isInstalled ? (
          <div className="flex justify-end gap-2">
            <Button onClick={() => onInstallClick(model)} size="sm" disabled={isInstallingCurrent}>
              {isInstallingCurrent ? "Installing..." : "Install"}
            </Button>
            {model.custom && (
              <Button
                onClick={() => onRemoveCustomModelClick(model)}
                variant="destructive"
                size="sm"
                disabled={isRemoveCustomPending}
              >
                {isRemoveCustomPending ? "Removing..." : "Remove custom model"}
              </Button>
            )}
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline" size="sm" disabled={!isDownloaded}>
                  <MoreVertical className="h-4 w-4" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <DropdownMenuItem
                  onClick={() => onPurgeClick(model.id)}
                  disabled={!isDownloaded || isPurgePending}
                  variant="destructive"
                >
                  Purge
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        ) : (
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" size="sm">
                <MoreVertical className="h-4 w-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem onClick={() => onTestClick(model)} disabled={isTestPending}>
                {isTestPending ? "Testing..." : "Test"}
              </DropdownMenuItem>
              {model.has_docker && (
                <>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem onClick={() => onShowDockerLogs(model.id)}>Docker Logs</DropdownMenuItem>
                  <DropdownMenuItem onClick={() => onShowDockerCompose(model.id)}>Docker Compose</DropdownMenuItem>
                  <DropdownMenuItem onClick={() => onRestartDocker(model.id)}>Restart Docker</DropdownMenuItem>
                  <DropdownMenuSeparator />
                </>
              )}
              <DropdownMenuItem onClick={() => onUninstallClick(model.id)} variant="destructive">
                Uninstall
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        )}
      </TableCell>
    </TableRow>
  );
});

function GpuStatsPanel({ stats }: { stats: GpuStats }) {
  if (stats.gpus && stats.gpus.length > 1) {
    return (
      <div className="mb-4 rounded-lg border px-4 py-2 text-sm">
        <span className="font-medium">GPU VRAM:</span>
        <div className="mt-1 flex flex-col gap-0.5">
          {stats.gpus.map((gpu: GpuCardStats, i: number) => (
            <span key={i} className="text-muted-foreground">
              <span className="font-medium text-foreground truncate max-w-[200px] inline-block align-bottom">{gpu.name}</span>
              {" — "}{gpu.used_vram_gb.toFixed(1)} / {gpu.total_vram_gb.toFixed(1)} GB
            </span>
          ))}
        </div>
      </div>
    );
  }
  return (
    <div className="mb-4 flex items-center gap-2 rounded-lg border px-4 py-2 text-sm">
      <span className="font-medium">GPU VRAM:</span>
      <span>{stats.used_vram_gb.toFixed(1)} GB used / {stats.total_vram_gb.toFixed(1)} GB total</span>
    </div>
  );
}

function ServiceModelsSkeleton() {
  return (
    <div className="w-full mx-auto p-6">
      <div className="mb-6">
        <div className="flex items-center justify-between mb-6">
          <Skeleton className="h-9 w-64" />
          <Skeleton className="h-10 w-40" />
        </div>
        <div className="flex flex-col md:flex-row gap-4">
          <div className="md:flex-1">
            <Skeleton className="h-10 w-full" />
          </div>
          <div className="flex gap-4">
            <Skeleton className="h-10 w-[200px]" />
            <Skeleton className="h-10 w-[200px]" />
            <Skeleton className="h-10 w-[200px]" />
          </div>
        </div>
      </div>
      <div className="border rounded-lg">
        <div className="p-4 space-y-3">
          <Skeleton className="h-8 w-full" />
          <Skeleton className="h-8 w-full" />
          <Skeleton className="h-8 w-full" />
          <Skeleton className="h-8 w-full" />
          <Skeleton className="h-8 w-full" />
        </div>
      </div>
    </div>
  );
}
