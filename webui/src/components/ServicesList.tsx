/*
DeepFellow Software Framework.
Copyright © 2025 Simplito sp. z o.o.

This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
This software is Licensed under the DeepFellow Free License.

See the License for the specific language governing permissions and
limitations under the License.
*/
import { useState, useMemo, useEffect, useRef } from "react";
import type { MouseEvent } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Link, useNavigate } from "@tanstack/react-router";
import { apiClient } from "@/deepfellow/client";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
} from "@/components/ui/dropdown-menu";
import { MoreVertical } from "lucide-react";
import { DynamicFormModal } from "./DynamicFormModal";
import { ConfirmModal } from "./ConfirmModal";
import { UninstallWithPurgeModal } from "./UninstallWithPurgeModal";
import { ProgressBadge } from "./ProgressBadge";
import { ContentModal } from "./ContentModal";
import { WarningsModal } from "./WarningsModal";
import { MeshInfoModal } from "./MeshInfoModal";
import { ServiceSettingsModal } from "./ServiceSettingsModal";
import { useModal } from "@/hooks/use-modal";
import type { Service, InstallProgress } from "@/deepfellow/types";
import { InstallationWarningsError } from "@/deepfellow/types";
import type { ProgressEvent } from "@/utils/sse-stream";
import {
  clearServiceInstallProgress,
  getSnapshot,
  setServiceInstallProgress,
  useInstallProgressSnapshot,
} from "@/state/install-progress-store";
import { startProgressSimulation, getStepPerTick, COMPLETION_SMOOTH_MS, COMPLETION_SMOOTH_MIN_MS } from "@/utils/progress-simulation";
import type { SimulationHandle } from "@/utils/progress-simulation";
import { toast } from "sonner";

export function ServicesList() {
  const modal = useModal();
  const navigate = useNavigate();
  const [searchQuery, setSearchQuery] = useState("");
  const [installingServiceId, setInstallingServiceId] = useState<string | null>(null);
  const pendingInstallationRef = useRef<{ serviceId: string; spec: Record<string, unknown>; size?: string | Record<string, string>; update?: boolean } | null>(null);
  const hasWarningsRef = useRef(false);
  const simulationStopFnsRef = useRef<Record<string, SimulationHandle>>({});
  const hasRealProgressRef = useRef<Record<string, boolean>>({});
  const restartDockerToastIdRef = useRef<string | number | null>(null);
  const uninstallToastIdRef = useRef<string | number | null>(null);
  const queryClient = useQueryClient();
  const installProgress = useInstallProgressSnapshot().services;

  const { data: servicesData, isLoading } = useQuery({
    queryKey: ["admin", "services"],
    queryFn: () => apiClient.listAdminServices(),
  });

  const { data: settingsData } = useQuery({
    queryKey: ["admin", "settings"],
    queryFn: () => apiClient.getSettings(),
  });

  const cloudEnabled = settingsData?.cloud_enabled ?? true;

  const cloudToggleMutation = useMutation({
    mutationFn: (enabled: boolean) => apiClient.updateSettings({ cloud_enabled: enabled }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["admin", "settings"] });
    },
    onError: (error) => {
      toast.error(`Failed to update cloud setting: ${error.message}`);
    },
  });
  const [anotherInstances, setAnotherInstances] = useState([] as Service[]);
  const servicesList = useMemo(() => {
    const newList = [...(servicesData ? servicesData.list : []), ...anotherInstances];
    newList.sort((a, b) => {
      if (!cloudEnabled) {
        const aCloud = a.is_cloud ? 1 : 0;
        const bCloud = b.is_cloud ? 1 : 0;
        if (aCloud !== bCloud) return aCloud - bCloud;
      }
      return a.type.localeCompare(b.type) || a.id.localeCompare(b.id);
    });
    return newList;
  }, [servicesData, anotherInstances, cloudEnabled])

  // Clean up any running simulations on unmount
  useEffect(() => {
    return () => {
      for (const sim of Object.values(simulationStopFnsRef.current)) sim.stop();
    };
  }, []);

  // Progress polling for existing installations
  useEffect(() => {
    if (!servicesList.length) return;

    // Reconcile: clear progress for services that are no longer in-progress so stale
    // badges don't linger after installation completes between effect runs.
    const inProgressIds = new Set<string>();
    for (const service of servicesList) {
      const inst = service.installed;
      if (!inst || typeof inst !== "object") continue;
      const s = (inst as { stage?: unknown }).stage;
      const v = (inst as { value?: unknown }).value;
      if ((s === "install" || s === "download") && typeof v === "number" && Number.isFinite(v)) {
        inProgressIds.add(service.id);
      }
    }
    for (const id of Object.keys(getSnapshot().services)) {
      if (!inProgressIds.has(id)) clearServiceInstallProgress(id);
    }

    const cleanups: Array<() => void> = [];

    for (const service of servicesList) {
      const installed = service.installed;
      if (!installed || typeof installed !== "object") continue;

      const installedStage = (installed as { stage?: unknown }).stage;
      const installedValue = (installed as { value?: unknown }).value;
      const installedIsProgress =
        (installedStage === "install" || installedStage === "download") &&
        typeof installedValue === "number" &&
        Number.isFinite(installedValue);

      if (!installedIsProgress) continue;

      const serviceId = service.id;
      // Skip if already tracked by an active install mutation — prevents double-tracking
      // and orphaned simulation timers when the 10 s refetch fires mid-install.
      if (simulationStopFnsRef.current[serviceId]) continue;

      let currentStage: "install" | "download" = installedStage as "install" | "download";
      const abortController = new AbortController();
      let isCancelled = false;

      // Use the existing store value (preserved across effect re-runs) or the backend
      // value as the starting point — avoids showing 0 % when the page reloads mid-install.
      const existingProgress = getSnapshot().services[serviceId];
      const startValue = existingProgress?.value ?? (installedValue as number);
      if (!existingProgress) {
        setServiceInstallProgress(serviceId, { stage: currentStage, value: installedValue as number });
      }

      const installStartTime = Date.now();
      const sim = startProgressSimulation({
        stepPerTick: getStepPerTick(service.size),
        initialValue: startValue,
        onTick: (value) => {
          if (isCancelled) return;
          setServiceInstallProgress(serviceId, { stage: currentStage, value });
        },
      });
      simulationStopFnsRef.current[serviceId] = sim;

      apiClient
        .getServiceProgress(
          serviceId,
          (event: ProgressEvent) => {
            if (isCancelled) return;

            const stage = event.stage;
            const value = event.value;

            if (event.type === "progress" && stage && value !== undefined) {
              currentStage = stage;
              hasRealProgressRef.current[serviceId] = true;
              setServiceInstallProgress(serviceId, { stage, value });
              return;
            }

            if (event.type === "finish") {
              if (event.status === "ok") {
                sim.smoothComplete(
                  Math.max(COMPLETION_SMOOTH_MIN_MS, Math.min(Date.now() - installStartTime, COMPLETION_SMOOTH_MS)),
                  () => {
                    delete simulationStopFnsRef.current[serviceId];
                    queryClient.invalidateQueries({ queryKey: ["admin", "services"] });
                  },
                  getSnapshot().services[serviceId]?.value
                );
              } else {
                sim.stop();
                delete simulationStopFnsRef.current[serviceId];
                clearServiceInstallProgress(serviceId);
                toast.error(`Installation failed for ${serviceId}: ${event.details || "Unknown error"}`);
              }
            }
          },
          abortController.signal
        )
        .catch((error) => {
          if (isCancelled) return;
          if (error instanceof Error && error.name === "AbortError") return;
          if (!isCancelled) {
            console.error(`Error polling progress for ${serviceId}:`, error);
          }
        });

      cleanups.push(() => {
        isCancelled = true;
        abortController.abort();
        sim.stop();
        delete simulationStopFnsRef.current[serviceId];
        delete hasRealProgressRef.current[serviceId];
        // Progress is NOT cleared here — the reconciliation step at the top of the next
        // effect run handles cleanup for services that are no longer in-progress.
      });
    }

    return () => {
      for (const cleanup of cleanups) cleanup();
    };
  }, [servicesList, queryClient]);

  const installMutation = useMutation({
    mutationFn: ({ serviceId, spec, size, ignoreWarnings = false, update = false }: { serviceId: string; spec: Record<string, unknown>; size?: string | Record<string, string>; ignoreWarnings?: boolean; update?: boolean }) => {
      return new Promise<void>((resolve, reject) => {
        let currentStage: "install" | "download" = "download";
        const installStartTime = Date.now();
        const sim = startProgressSimulation({
          stepPerTick: getStepPerTick(size ?? "", 10),
          onTick: (value) => {
            setServiceInstallProgress(serviceId, { stage: currentStage, value });
          },
        });
        simulationStopFnsRef.current[serviceId] = sim;

        const stream = update ? apiClient.updateAdminServiceStreaming : apiClient.installAdminServiceStreaming;
        stream.call(
          apiClient,
          serviceId,
          spec,
          (event: ProgressEvent) => {
            const stage = event.stage;
            const value = event.value;

            if (event.type === "progress" && stage && value !== undefined) {
              currentStage = stage;
              hasRealProgressRef.current[serviceId] = true;
              setServiceInstallProgress(serviceId, { stage, value });
            } else if (event.type === "finish") {
              if (event.status === "ok") {
                sim.smoothComplete(
                  Math.max(COMPLETION_SMOOTH_MIN_MS, Math.min(Date.now() - installStartTime, COMPLETION_SMOOTH_MS)),
                  () => {
                    delete simulationStopFnsRef.current[serviceId];
                    // Do NOT clear progress here — if the backend still reports the service
                    // as in-progress on the next refetch, clearing now would cause the bar
                    // to jump back to the stale backend value. Reconciliation clears it once
                    // the backend confirms the service is no longer in-progress.
                    resolve();
                  },
                  getSnapshot().services[serviceId]?.value
                );
              } else {
                sim.stop();
                delete simulationStopFnsRef.current[serviceId];
                clearServiceInstallProgress(serviceId);
                reject(new Error(event.details || "Installation failed"));
              }
            }
          },
          ignoreWarnings
        ).catch(reject);
        modal.close();
      });
    },
    onMutate: ({ serviceId }) => {
      setInstallingServiceId(serviceId);
    },
    onSuccess: (_data, variables) => {
      delete simulationStopFnsRef.current[variables.serviceId];
      delete hasRealProgressRef.current[variables.serviceId];
      queryClient.invalidateQueries({ queryKey: ["admin", "services"] });
      pendingInstallationRef.current = null;
      toast.success(variables.update ? "Service updated successfully" : "Service installed successfully");
    },
    onError: (error, variables) => {
      const simStop = simulationStopFnsRef.current[variables.serviceId];
      if (simStop) {
        simStop.stop();
        delete simulationStopFnsRef.current[variables.serviceId];
      }
      delete hasRealProgressRef.current[variables.serviceId];

      if (error instanceof InstallationWarningsError) {
        // Show warnings modal instead of error toast
        hasWarningsRef.current = true;
        modal.open(WarningsModal, {
          warnings: error.warnings,
          onContinue: handleWarningsContinue,
          isLoading: false,
        });
        return;
      }

      hasWarningsRef.current = false;
      clearServiceInstallProgress(variables.serviceId);
      toast.error(`Failed to ${variables.update ? "update" : "install"} service: ${error.message}`);
    },
    onSettled: () => {
      // Only reset if not showing warnings modal
      if (!hasWarningsRef.current) {
        setInstallingServiceId(null);
        setAnotherInstances([]);
      }
    },
  });

  const uninstallMutation = useMutation({
    mutationFn: (serviceId: string) => apiClient.uninstallAdminService(serviceId, false),
    onMutate: () => {
      const toastId = toast.loading("Uninstalling...");
      uninstallToastIdRef.current = toastId;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["admin", "services"] });
      if (uninstallToastIdRef.current) {
        toast.success("Service uninstalled successfully", { id: uninstallToastIdRef.current });
        uninstallToastIdRef.current = null;
      } else {
        toast.success("Service uninstalled successfully");
      }
    },
    onError: (error) => {
      if (uninstallToastIdRef.current) {
        toast.error(`Failed to uninstall service: ${error.message}`, { id: uninstallToastIdRef.current });
        uninstallToastIdRef.current = null;
      } else {
        toast.error(`Failed to uninstall service: ${error.message}`);
      }
    },
  });

  const purgeMutation = useMutation({
    mutationFn: (serviceId: string) => apiClient.uninstallAdminService(serviceId, true),
    onMutate: () => {
      toast.loading("Purging...", { id: "purge-service" });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["admin", "services"] });
      toast.success("Service purged successfully", { id: "purge-service" });
    },
    onError: (error) => {
      toast.error(`Failed to purge service: ${error.message}`, { id: "purge-service" });
    },
  });

  const dockerRestartMutation = useMutation({
    mutationFn: (serviceId: string) => apiClient.restartDocker(serviceId),
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

  const handleShowDockerLogs = async (serviceId: string) => {
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
      const data = await apiClient.getDockerLogs(serviceId);
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
  };

  const handleShowDockerCompose = async (serviceId: string) => {
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
      const data = await apiClient.getDockerCompose(serviceId);
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
  };

  const handleRestartDocker = (serviceId: string) => {
    modal.open(ConfirmModal, {
      title: "Restart Docker",
      description: `Are you sure you want to restart Docker for ${serviceId}?`,
      confirmText: "Restart",
      cancelText: "Cancel",
      onConfirm: () => {
        modal.close();
        dockerRestartMutation.mutate(serviceId);
      },
      isLoading: dockerRestartMutation.isPending,
      variant: "warning",
    });
  };

  const handleShowMeshInfo = () => {
    modal.open(MeshInfoModal, {});
  };

  const handleInstallClick = async (service: Service, installAnotherInstance: boolean) => {
    modal.open(DynamicFormModal, {
      title: `Install ${service.id}`,
      fields: [],
      isLoading: true,
      isSubmitting: false,
      onSubmit: () => {},
    });

    try {
      const serviceDetail = await apiClient.getAdminService(service.id);
      const orgInstance = serviceDetail.id.split("|")[1] || "default";
      let instanceDefaulValue = "default";
      if (installAnotherInstance) {
        let i = 0;
        while (true) {
          instanceDefaulValue = "new" + (i === 0 ? "" : "-" + i);
          const id = serviceDetail.type + "|" + instanceDefaulValue;
          if (servicesList.find(x => x.id === id)) {
            i++;
          }
          else {
            break;
          }
        }
      }
      modal.open(DynamicFormModal, {
        title: `Install ${serviceDetail.id}`,
        fields: installAnotherInstance ? [{
          name: "instance",
          description: "Instance ID",
          type: "text",
          required: true,
          default: instanceDefaulValue,
          placeholder: "default",
        }, ...serviceDetail.spec.fields] : serviceDetail.spec.fields,
        onSubmit: (spec: Record<string, unknown>) => {
          const instance = installAnotherInstance ? spec.instance as string : orgInstance;
          const serviceId = installAnotherInstance ? serviceDetail.type + (instance && instance !== "default" ? "|" + instance : "") : service.id;
          if (installAnotherInstance && servicesList.find(x => x.id === serviceId)) {
            toast.error("Given Instance ID already in use");
            return;
          }
          pendingInstallationRef.current = { serviceId: serviceId, spec, size: serviceDetail.size };
          if (installAnotherInstance) {
            setAnotherInstances([...anotherInstances, {...serviceDetail, id: serviceId, instance: instance, installed: false}]);
          }
          installMutation.mutate({ serviceId: serviceId, spec, size: serviceDetail.size });
        },
        // Keep modal interactive; don't disable because another install is running.
        isSubmitting: false,
      });
    } catch {
      modal.close();
      toast.error("Failed to load service details");
    }
  };

  const handleEditClick = (service: Service) => {
    const installed = service.installed;
    const currentValues =
      installed && typeof installed === "object" && !("stage" in (installed as Record<string, unknown>))
        ? (installed as Record<string, unknown>)
        : {};
    modal.open(DynamicFormModal, {
      title: `Edit ${service.id}`,
      fields: service.spec.fields,
      initialData: currentValues,
      submitLabel: "Save",
      submittingLabel: "Saving...",
      onSubmit: (spec: Record<string, unknown>) => {
        pendingInstallationRef.current = { serviceId: service.id, spec, size: service.size, update: true };
        installMutation.mutate({ serviceId: service.id, spec, size: service.size, update: true });
      },
      isSubmitting: false,
    });
  };

  const handleWarningsContinue = () => {
    if (pendingInstallationRef.current) {
      hasWarningsRef.current = false;
      installMutation.mutate({
        serviceId: pendingInstallationRef.current.serviceId,
        spec: pendingInstallationRef.current.spec,
        size: pendingInstallationRef.current.size,
        update: pendingInstallationRef.current.update,
        ignoreWarnings: true
      });
    }
  };

  const handleUninstallClick = async (serviceId: string) => {
    // Check if service has custom models before uninstalling
    let customModelsCount = 0;
    try {
      const modelsData = await apiClient.listAdminServiceModels(serviceId);
      customModelsCount = modelsData.list?.filter((model) => model.custom).length || 0;
    } catch (error) {
      // If we can't fetch models, proceed anyway but log the error
      console.warn("Failed to fetch models for service:", error);
    }

    const description = customModelsCount > 0
      ? `Are you sure you want to uninstall ${serviceId}? This will also remove ${customModelsCount} custom model${customModelsCount > 1 ? "s" : ""} associated with this service. This action cannot be undone.`
      : `Are you sure you want to uninstall ${serviceId}? This action cannot be undone.`;

    modal.open(UninstallWithPurgeModal, {
      title: "Uninstall Service",
      description,
      confirmText: "Uninstall",
      cancelText: "Cancel",
      purgeLabel: "Purge",
      purgeDescription: "Also remove downloaded service files and local data to free disk space. This cannot be undone.",
      onConfirm: (purge) => {
        modal.close();
        if (purge) {
          purgeMutation.mutate(serviceId);
        } else {
          uninstallMutation.mutate(serviceId);
        }
      },
      isLoading: uninstallMutation.isPending || purgeMutation.isPending,
      variant: "destructive",
    });
  };

  const handlePurgeClick = (serviceId: string) => {
    modal.open(ConfirmModal, {
      title: "Purge Service",
      description: "This will remove all downloaded files for the service. This action cannot be undone.",
      confirmText: "Purge",
      cancelText: "Cancel",
      onConfirm: () => {
        modal.close();
        purgeMutation.mutate(serviceId);
      },
      isLoading: purgeMutation.isPending,
      variant: "destructive",
    });
  };

  const handleRowNavigate = (e: MouseEvent, serviceId: string) => {
    // Don't hijack clicks on interactive controls inside the row.
    const target = e.target as HTMLElement | null;
    if (!target) return;

    const interactive = target.closest(
      "a,button,[role='button'],[role='menuitem'],input,select,textarea,label,[data-prevent-row-click]"
    );
    if (interactive) return;

    navigate({
      to: "/dashboard/services/$serviceId",
      params: { serviceId },
    });
  };

  const services = servicesList;

  const filteredServices = useMemo(() => {
    if (!searchQuery.trim()) return services;

    const query = searchQuery.toLowerCase();
    return services.filter((service) => {
      if (service.id.toLowerCase().includes(query)) return true;

      const installed = service.installed;
      const installedIsObject = !!installed && typeof installed === "object";
      const installedLooksLikeProgress =
        installedIsObject && "stage" in installed && "value" in installed;

      if (installedIsObject && !installedLooksLikeProgress) {
        const installedValues = Object.entries(installed).some(([key, value]) => {
          const field = service.spec.fields.find((f) => f.name === key);
          if (field?.type === "password") return false;

          return (
            key.toLowerCase().includes(query) ||
            String(value ?? "").toLowerCase().includes(query)
          );
        });
        if (installedValues) return true;
      }

      // Search in size
      const sizeStr = typeof service.size === "string"
        ? service.size
        : Object.entries(service.size).map(([k, v]) => `${k}:${v}`).join(" ");
      if (sizeStr.toLowerCase().includes(query)) return true;

      return false;
    });
  }, [services, searchQuery]);

  if (isLoading) {
    return <ServicesListSkeleton />;
  }

  return (
    <div className="px-4 lg:px-6">
      <div className="mb-6">
        <div className="flex items-center justify-between mb-4">
          <h1 className="text-3xl font-bold">Services</h1>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <Switch
                id="cloud-enabled"
                checked={cloudEnabled}
                onCheckedChange={(checked) => cloudToggleMutation.mutate(checked)}
                disabled={cloudToggleMutation.isPending}
              />
              <Label htmlFor="cloud-enabled">Cloud services</Label>
            </div>
            <Button
              onClick={handleShowMeshInfo}
              variant="outline"
            >
              Show mesh info
            </Button>
          </div>
        </div>
        <Input
          placeholder="Search services..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="max-w-sm"
        />
      </div>

      <div className="border rounded-lg">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Service ID</TableHead>
              <TableHead className="min-w-[150px]">Status</TableHead>
              <TableHead>Resources</TableHead>
              <TableHead className="text-right min-w-[165px]">Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {filteredServices.length === 0 ? (
              <TableRow>
                <TableCell colSpan={4} className="text-center text-muted-foreground">
                  {searchQuery ? "No services found matching your search" : "No services available"}
                </TableCell>
              </TableRow>
            ) : (
              filteredServices.map((service) => {
                const isInstalled = !!service.installed;
                const isDownloaded = !!service.downloaded;
                const sizeInfo = typeof service.size === "string"
                  ? service.size
                  : Object.entries(service.size).map(([key, val]) => `${key.toUpperCase()}: ${val}`).join(", ");

                const isInstallingCurrent = installMutation.isPending && installingServiceId === service.id;
                const currentProgress = installProgress[service.id];
                const isInProgress = !!currentProgress;
                // Check if installed is InstallProgress type (has stage and value)
                const installedIsProgress = service.installed && typeof service.installed === "object" && "stage" in service.installed && "value" in service.installed;
                const isCloudDisabled = service.is_cloud && !cloudEnabled;

                return (
                  <TableRow
                    key={service.id}
                    onClick={(e) => {
                      if (!isInstalled || installedIsProgress || isInProgress || isCloudDisabled) return;
                      handleRowNavigate(e, service.id);
                    }}
                    className={[
                      isInstalled && !installedIsProgress && !isInProgress && !isCloudDisabled
                        ? "cursor-pointer hover:bg-muted/50"
                        : undefined,
                      isCloudDisabled ? "opacity-50" : undefined,
                    ].filter(Boolean).join(" ") || undefined}
                  >
                    <TableCell className="font-semibold">
                      <div>
                        {isInstalled && !installedIsProgress ? (
                          <Link
                            to="/dashboard/services/$serviceId"
                            params={{ serviceId: service.id }}
                            className="font-semibold text-primary hover:underline"
                          >
                            {service.id}
                          </Link>
                        ) : (
                          <div>{service.id}</div>
                        )}
                        {service.description && service.description.length > 0 && (
                          <div className="text-sm text-muted-foreground mt-1">{service.description}</div>
                        )}
                      </div>
                    </TableCell>
                    <TableCell>
                      {isInProgress || installedIsProgress ? (
                        <ProgressBadge
                          stage={currentProgress?.stage || (service.installed as InstallProgress).stage}
                          value={currentProgress?.value ?? (service.installed as InstallProgress).value}
                          variant="default"
                          simulated={!hasRealProgressRef.current[service.id]}
                        />
                      ) : (
                        <div className="flex flex-wrap items-center gap-2">
                          <Badge variant={isInstalled ? "default" : "secondary"}>
                            {isInstalled ? "Installed" : "Not installed"}
                          </Badge>
                          {!isInstalled && isDownloaded && (
                            <Badge variant="outline">Downloaded</Badge>
                          )}
                        </div>
                      )}
                    </TableCell>
                    <TableCell className="font-mono text-sm">{sizeInfo || "N/A"}</TableCell>
                    <TableCell className="text-right">
                      <div className="flex justify-end gap-2" data-prevent-row-click>
                        {isInProgress || installedIsProgress ? null : !isInstalled ? (
                          <>
                            <TooltipProvider>
                              <Tooltip>
                                <TooltipTrigger asChild>
                                  <span>
                                    <Button
                                      onClick={() => handleInstallClick(service, false)}
                                      size="sm"
                                      disabled={isInstallingCurrent || isCloudDisabled}
                                    >
                                      {isInstallingCurrent ? "Installing..." : "Install"}
                                    </Button>
                                  </span>
                                </TooltipTrigger>
                                {isCloudDisabled && (
                                  <TooltipContent>
                                    Cloud services are disabled. Enable cloud to install this service.
                                  </TooltipContent>
                                )}
                              </Tooltip>
                            </TooltipProvider>
                            <DropdownMenu>
                              <DropdownMenuTrigger asChild>
                                <Button
                                  variant="outline"
                                  size="sm"
                                  disabled={!isDownloaded}
                                >
                                  <MoreVertical className="h-4 w-4" />
                                </Button>
                              </DropdownMenuTrigger>
                              <DropdownMenuContent align="end">
                                <DropdownMenuItem
                                  onClick={() => handlePurgeClick(service.id)}
                                  disabled={!isDownloaded || purgeMutation.isPending}
                                  variant="destructive"
                                >
                                  Purge
                                </DropdownMenuItem>
                              </DropdownMenuContent>
                            </DropdownMenu>
                          </>
                        ) : (
                          <>
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() =>
                                modal.open(ServiceSettingsModal, { service, onEdit: handleEditClick })
                              }
                            >
                              Settings
                            </Button>
                            <DropdownMenu>
                            <DropdownMenuTrigger asChild>
                              <Button variant="outline" size="sm">
                                <MoreVertical className="h-4 w-4" />
                              </Button>
                            </DropdownMenuTrigger>
                            <DropdownMenuContent align="end">
                              <DropdownMenuItem
                                onClick={() => handleInstallClick(service, true)}
                              >
                                Install another instance
                              </DropdownMenuItem>
                              {service.has_docker && (
                                <>
                                  <DropdownMenuItem
                                    onClick={() => handleShowDockerLogs(service.id)}
                                  >
                                    Docker Logs
                                  </DropdownMenuItem>
                                  <DropdownMenuItem
                                    onClick={() => handleShowDockerCompose(service.id)}
                                  >
                                    Docker Compose
                                  </DropdownMenuItem>
                                  <DropdownMenuItem
                                    onClick={() => handleRestartDocker(service.id)}
                                  >
                                    Restart Docker
                                  </DropdownMenuItem>
                                  <DropdownMenuSeparator />
                                </>
                              )}
                              <DropdownMenuItem
                                onClick={() => handleUninstallClick(service.id)}
                                variant="destructive"
                              >
                                Uninstall
                              </DropdownMenuItem>
                            </DropdownMenuContent>
                          </DropdownMenu>
                          </>
                        )}
                      </div>
                    </TableCell>
                  </TableRow>
                );
              })
            )}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}

function ServicesListSkeleton() {
  return (
    <div className="px-4 lg:px-6">
      <div className="mb-6">
        <div className="flex items-center justify-between mb-4">
          <Skeleton className="h-9 w-40" />
          <Skeleton className="h-10 w-36" />
        </div>
        <Skeleton className="h-10 w-[320px] max-w-full" />
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
