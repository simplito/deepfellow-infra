import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Skeleton } from "@/components/ui/skeleton";
import { Switch } from "@/components/ui/switch";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { apiClient } from "@/deepfellow/client";
import type { InstallProgress, Service } from "@/deepfellow/types";
import { InstallationWarningsError } from "@/deepfellow/types";
import { useModal } from "@/hooks/use-modal";
import { findImageVersionField, planImageUpdate } from "@/lib/imageUpdate";
import {
  clearServiceInstallProgress,
  getSnapshot,
  setServiceInstallProgress,
  useInstallProgressSnapshot,
} from "@/state/install-progress-store";
import { renderMarkdownLinks } from "@/utils/markdown-links";
import {
  COMPLETION_SMOOTH_MIN_MS,
  COMPLETION_SMOOTH_MS,
  getStepPerTick,
  startProgressSimulation,
} from "@/utils/progress-simulation";
import type { SimulationHandle } from "@/utils/progress-simulation";
import type { ProgressEvent } from "@/utils/sse-stream";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, useNavigate } from "@tanstack/react-router";
import { AlertCircle, MoreVertical } from "lucide-react";
/*
DeepFellow Software Framework.
Copyright © 2025 Simplito sp. z o.o.

This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
This software is Licensed under the DeepFellow Free License.

See the License for the specific language governing permissions and
limitations under the License.
*/
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { MouseEvent } from "react";
import { toast } from "sonner";
import { ConfirmModal } from "./ConfirmModal";
import { ContentModal } from "./ContentModal";
import { DynamicFormModal } from "./DynamicFormModal";
import { MeshInfoModal } from "./MeshInfoModal";
import { ProgressBadge } from "./ProgressBadge";
import { ServiceSettingsModal } from "./ServiceSettingsModal";
import { UninstallWithPurgeModal } from "./UninstallWithPurgeModal";
import { WarningsModal } from "./WarningsModal";

export function ServicesList() {
  const modal = useModal();
  const navigate = useNavigate();
  const [searchQuery, setSearchQuery] = useState("");
  const [installingServiceId, setInstallingServiceId] = useState<string | null>(
    null,
  );
  // Services whose cancel request is in flight - blocks navigation and hides row controls for as
  // long as a possible DFINFRA-318 rollback may still be running.
  const [cancellingServiceIds, setCancellingServiceIds] = useState<Set<string>>(
    new Set<string>(),
  );
  const pendingInstallationRef = useRef<{
    serviceId: string;
    spec: Record<string, unknown>;
    size?: string | Record<string, string>;
    update?: boolean;
  } | null>(null);
  const hasWarningsRef = useRef(false);
  const simulationStopFnsRef = useRef<Record<string, SimulationHandle>>({});
  const hasRealProgressRef = useRef<Record<string, boolean>>({});
  // Per-service, set in installMutation's onMutate from that call's own variables - unlike
  // pendingInstallationRef (a single slot, only meant for the "retry after warnings" flow), this
  // can't be clobbered by a second install/update starting for a different service while an earlier
  // one's cancel is still resolving. Still local-only, so it doesn't survive a page reload.
  const isUpdateByServiceRef = useRef<Record<string, boolean>>({});
  const restartDockerToastIdRef = useRef<string | number | null>(null);
  const uninstallToastIdRef = useRef<string | number | null>(null);
  const installAbortControllersRef = useRef<Record<string, AbortController>>(
    {},
  );
  const cancelledInstallsRef = useRef<Set<string>>(new Set());
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
  const loadError = servicesData?.load_error;

  const cloudToggleMutation = useMutation({
    mutationFn: (enabled: boolean) =>
      apiClient.updateSettings({ cloud_enabled: enabled }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["admin", "settings"] });
    },
    onError: (error) => {
      toast.error(`Failed to update cloud setting: ${error.message}`);
    },
  });
  const [anotherInstances, setAnotherInstances] = useState([] as Service[]);
  const servicesList = useMemo(() => {
    const newList = [
      ...(servicesData ? servicesData.list : []),
      ...anotherInstances,
    ];
    const isDisabled = (service: Service) =>
      !!service.disabled_reason || (!!service.is_cloud && !cloudEnabled);
    newList.sort((a, b) => {
      const aDisabled = isDisabled(a) ? 1 : 0;
      const bDisabled = isDisabled(b) ? 1 : 0;
      if (aDisabled !== bDisabled) return aDisabled - bDisabled;
      return a.type.localeCompare(b.type) || a.id.localeCompare(b.id);
    });
    return newList;
  }, [servicesData, anotherInstances, cloudEnabled]);

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
      if (
        (s === "install" || s === "download") &&
        typeof v === "number" &&
        Number.isFinite(v)
      ) {
        inProgressIds.add(service.id);
      }
    }
    for (const id of Object.keys(getSnapshot().services)) {
      // Skip services with a pending cancel - the query has no periodic refresh, so stale data here
      // could otherwise wipe tracking for a service still mid-rollback, flashing it as "Installed".
      if (!inProgressIds.has(id) && !cancellingServiceIds.has(id)) {
        clearServiceInstallProgress(id);
      }
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

      let currentStage: "install" | "download" = installedStage as
        | "install"
        | "download";
      const abortController = new AbortController();
      let isCancelled = false;

      // Use the existing store value (preserved across effect re-runs) or the backend
      // value as the starting point — avoids showing 0 % when the page reloads mid-install.
      const existingProgress = getSnapshot().services[serviceId];
      const startValue = existingProgress?.value ?? (installedValue as number);
      if (!existingProgress) {
        setServiceInstallProgress(serviceId, {
          stage: currentStage,
          value: installedValue as number,
        });
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
                  Math.max(
                    COMPLETION_SMOOTH_MIN_MS,
                    Math.min(
                      Date.now() - installStartTime,
                      COMPLETION_SMOOTH_MS,
                    ),
                  ),
                  () => {
                    delete simulationStopFnsRef.current[serviceId];
                    queryClient.invalidateQueries({
                      queryKey: ["admin", "services"],
                    });
                  },
                  getSnapshot().services[serviceId]?.value,
                );
              } else {
                sim.stop();
                delete simulationStopFnsRef.current[serviceId];
                clearServiceInstallProgress(serviceId);
                toast.error(
                  renderMarkdownLinks(
                    `Installation failed for ${serviceId}: ${event.details || "Unknown error"}`,
                  ),
                );
              }
            }
          },
          abortController.signal,
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
  }, [servicesList, queryClient, cancellingServiceIds]);

  const installMutation = useMutation({
    mutationFn: ({
      serviceId,
      spec,
      size,
      ignoreWarnings = false,
      update = false,
    }: {
      serviceId: string;
      spec: Record<string, unknown>;
      size?: string | Record<string, string>;
      ignoreWarnings?: boolean;
      update?: boolean;
    }) => {
      const abortController = new AbortController();
      installAbortControllersRef.current[serviceId] = abortController;

      return new Promise<void>((resolve, reject) => {
        let currentStage: "install" | "download" = "download";
        const installStartTime = Date.now();
        const sim = startProgressSimulation({
          stepPerTick: getStepPerTick(size ?? "", 10),
          onTick: (value) => {
            setServiceInstallProgress(serviceId, {
              stage: currentStage,
              value,
            });
          },
        });
        simulationStopFnsRef.current[serviceId] = sim;

        const stream = update
          ? apiClient.updateAdminServiceStreaming
          : apiClient.installAdminServiceStreaming;
        stream
          .call(
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
                    Math.max(
                      COMPLETION_SMOOTH_MIN_MS,
                      Math.min(
                        Date.now() - installStartTime,
                        COMPLETION_SMOOTH_MS,
                      ),
                    ),
                    () => {
                      delete simulationStopFnsRef.current[serviceId];
                      // Do NOT clear progress here — if the backend still reports the service
                      // as in-progress on the next refetch, clearing now would cause the bar
                      // to jump back to the stale backend value. Reconciliation clears it once
                      // the backend confirms the service is no longer in-progress.
                      resolve();
                    },
                    getSnapshot().services[serviceId]?.value,
                  );
                } else {
                  sim.stop();
                  delete simulationStopFnsRef.current[serviceId];
                  clearServiceInstallProgress(serviceId);
                  // Failure events always carry a plain string (see `ProgressEvent`).
                  reject(
                    new Error(
                      typeof event.details === "string"
                        ? event.details
                        : "Installation failed",
                    ),
                  );
                }
              }
            },
            ignoreWarnings,
            abortController.signal,
          )
          .catch(reject);
        modal.close();
      });
    },
    onMutate: ({ serviceId, update = false }) => {
      setInstallingServiceId(serviceId);
      isUpdateByServiceRef.current[serviceId] = update;
    },
    onSuccess: (_data, variables) => {
      delete simulationStopFnsRef.current[variables.serviceId];
      delete hasRealProgressRef.current[variables.serviceId];
      queryClient.invalidateQueries({ queryKey: ["admin", "services"] });
      pendingInstallationRef.current = null;
      toast.success(
        variables.update
          ? "Service updated successfully"
          : "Service installed successfully",
      );
    },
    onError: (error, variables) => {
      // Intentional cancel: the UI was already reset by handleCancelInstall — swallow the AbortError.
      if (cancelledInstallsRef.current.has(variables.serviceId)) {
        cancelledInstallsRef.current.delete(variables.serviceId);
        return;
      }

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
      toast.error(
        renderMarkdownLinks(
          `Failed to ${variables.update ? "update" : "install"} service: ${error.message}`,
        ),
      );
    },
    onSettled: (_data, _error, variables) => {
      cancelledInstallsRef.current.delete(variables.serviceId);
      delete installAbortControllersRef.current[variables.serviceId];
      // Only reset if not showing warnings modal
      if (!hasWarningsRef.current) {
        setInstallingServiceId(null);
        setAnotherInstances([]);
      }
    },
  });

  // Cancel an in-progress install/update: stop the backend Docker work, abort the local SSE read,
  // and reset the UI. Not offered for cloud services (see render below) - they have nothing to cancel.
  const handleCancelInstall = useCallback(
    async (serviceId: string) => {
      // Flag the cancel so the install mutation's error path swallows the AbortError it triggers.
      cancelledInstallsRef.current.add(serviceId);

      // Abort the local SSE fetch (terminates the pending install promise immediately).
      installAbortControllersRef.current[serviceId]?.abort();
      delete installAbortControllersRef.current[serviceId];

      // Freeze the simulated progress - nothing real to report from here. Only an update can trigger
      // DFINFRA-318's rollback, so label it "Rolling back"; a fresh install stays "Installing".
      simulationStopFnsRef.current[serviceId]?.stop();
      delete simulationStopFnsRef.current[serviceId];
      const isUpdateCancel = isUpdateByServiceRef.current[serviceId] === true;
      setServiceInstallProgress(serviceId, {
        stage: isUpdateCancel ? "rollback" : "install",
        value: getSnapshot().services[serviceId]?.value ?? 0,
      });
      setCancellingServiceIds((prev) => new Set(prev).add(serviceId));

      // True immediately: the original work does stop right away. A triggered rollback takes longer
      // to settle and gets its own toast below, once that outcome is known.
      toast.success(`Installation cancelled for ${serviceId}`, {
        duration: 8000,
      });

      let cancelRequestSucceeded = true;
      let isAlreadyGone = false;
      try {
        // Stops the backend work. For an update, this doesn't resolve until a triggered rollback
        // settles - installProgress stays set for that whole wait, so the row stays non-clickable.
        await apiClient.cancelAdminServiceInstall(serviceId);
      } catch (error) {
        cancelRequestSucceeded = false;
        // A 404 just means the install already finished/cleared on the backend — safe to ignore.
        isAlreadyGone =
          error instanceof Error && error.message.includes("HTTP 404");
        if (!isAlreadyGone) {
          toast.error(
            `Failed to cancel installation for ${serviceId} — please try again`,
          );
        }
        console.error(`Failed to cancel install for ${serviceId}:`, error);
      }

      // The cancel response doesn't say whether the rollback reverted or also failed - fetch the
      // service's now-settled state instead of assuming.
      if (cancelRequestSucceeded && isUpdateCancel) {
        try {
          const fresh = await apiClient.getAdminService(serviceId);
          const reverted =
            !!fresh.installed &&
            !(
              typeof fresh.installed === "object" && "stage" in fresh.installed
            );
          if (reverted) {
            toast.success(
              `Rollback finished for ${serviceId} - reverted to its previous configuration`,
            );
          } else {
            toast.error(
              `Rollback for ${serviceId} could not restore the previous configuration - the service is now fully uninstalled`,
            );
          }
        } catch (error) {
          console.error(
            `Failed to check rollback outcome for ${serviceId}:`,
            error,
          );
        }
      }

      // installProgress (the row's source of truth for "still busy") only clears once the backend is
      // known to be done: the call succeeded, or 404'd because the reservation was already gone. Any
      // other failure (e.g. a 504 timeout) leaves `installing` set on the backend, so this stays too.
      if (cancelRequestSucceeded || isAlreadyGone) {
        delete hasRealProgressRef.current[serviceId];
        clearServiceInstallProgress(serviceId);
        setInstallingServiceId(null);
      }

      // Refresh before releasing cancellingServiceIds below: the reconciliation effect decides whether
      // it's safe to treat this service as "not in progress" from the services-list query, so it must
      // see the real (still-busy, on failure) state before that guard comes off.
      await queryClient.invalidateQueries({ queryKey: ["admin", "services"] });

      // cancellingServiceIds only hides the Cancel button/controls while this request is in flight -
      // clear it regardless of outcome, so a failure can be retried instead of requiring a page reload.
      setCancellingServiceIds((prev) => {
        const next = new Set(prev);
        next.delete(serviceId);
        return next;
      });
    },
    [queryClient],
  );

  const uninstallMutation = useMutation({
    mutationFn: (serviceId: string) =>
      apiClient.uninstallAdminService(serviceId, false),
    onMutate: () => {
      const toastId = toast.loading("Uninstalling...");
      uninstallToastIdRef.current = toastId;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["admin", "services"] });
      if (uninstallToastIdRef.current) {
        toast.success("Service uninstalled successfully", {
          id: uninstallToastIdRef.current,
        });
        uninstallToastIdRef.current = null;
      } else {
        toast.success("Service uninstalled successfully");
      }
    },
    onError: (error) => {
      if (uninstallToastIdRef.current) {
        toast.error(`Failed to uninstall service: ${error.message}`, {
          id: uninstallToastIdRef.current,
        });
        uninstallToastIdRef.current = null;
      } else {
        toast.error(`Failed to uninstall service: ${error.message}`);
      }
    },
  });

  const purgeMutation = useMutation({
    mutationFn: (serviceId: string) =>
      apiClient.uninstallAdminService(serviceId, true),
    onMutate: () => {
      toast.loading("Purging...", { id: "purge-service" });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["admin", "services"] });
      toast.success("Service purged successfully", { id: "purge-service" });
    },
    onError: (error) => {
      toast.error(`Failed to purge service: ${error.message}`, {
        id: "purge-service",
      });
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
        toast.success("Docker restarted successfully", {
          id: restartDockerToastIdRef.current,
        });
        restartDockerToastIdRef.current = null;
      } else {
        toast.success("Docker restarted successfully");
      }
    },
    onError: (error) => {
      if (restartDockerToastIdRef.current) {
        toast.error(`Failed to restart Docker: ${error.message}`, {
          id: restartDockerToastIdRef.current,
        });
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
      toast.error(
        `Failed to fetch Docker logs: ${error instanceof Error ? error.message : "Unknown error"}`,
      );
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
      toast.error(
        `Failed to fetch Docker compose file: ${error instanceof Error ? error.message : "Unknown error"}`,
      );
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

  const handleInstallClick = async (
    service: Service,
    installAnotherInstance: boolean,
  ) => {
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
          instanceDefaulValue = `new${i === 0 ? "" : `-${i}`}`;
          const id = `${serviceDetail.type}|${instanceDefaulValue}`;
          if (servicesList.find((x) => x.id === id)) {
            i++;
          } else {
            break;
          }
        }
      }
      modal.open(DynamicFormModal, {
        title: `Install ${serviceDetail.id}`,
        fields: installAnotherInstance
          ? [
              {
                name: "instance",
                description: "Instance ID",
                type: "text",
                required: true,
                default: instanceDefaulValue,
                placeholder: "default",
              },
              ...serviceDetail.spec.fields,
            ]
          : serviceDetail.spec.fields,
        serviceId: serviceDetail.id,
        onSubmit: (spec: Record<string, unknown>) => {
          const instance = installAnotherInstance
            ? (spec.instance as string)
            : orgInstance;
          const serviceId = installAnotherInstance
            ? serviceDetail.type +
              (instance && instance !== "default" ? `|${instance}` : "")
            : service.id;
          if (
            installAnotherInstance &&
            servicesList.find((x) => x.id === serviceId)
          ) {
            toast.error("Given Instance ID already in use");
            return;
          }
          pendingInstallationRef.current = {
            serviceId: serviceId,
            spec,
            size: serviceDetail.size,
          };
          if (installAnotherInstance) {
            setAnotherInstances([
              ...anotherInstances,
              {
                ...serviceDetail,
                id: serviceId,
                instance: instance,
                installed: false,
              },
            ]);
          }
          installMutation.mutate({
            serviceId: serviceId,
            spec,
            size: serviceDetail.size,
          });
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
      installed &&
      typeof installed === "object" &&
      !("stage" in (installed as Record<string, unknown>))
        ? (installed as Record<string, unknown>)
        : {};
    modal.open(DynamicFormModal, {
      title: `Edit ${service.id}`,
      fields: service.spec.fields,
      initialData: currentValues,
      serviceId: service.id,
      submitLabel: "Save",
      submittingLabel: "Saving...",
      onSubmit: (spec: Record<string, unknown>) => {
        pendingInstallationRef.current = {
          serviceId: service.id,
          spec,
          size: service.size,
          update: true,
        };
        installMutation.mutate({
          serviceId: service.id,
          spec,
          size: service.size,
          update: true,
        });
      },
      isSubmitting: false,
    });
  };

  const handleUpdateClick = async (service: Service) => {
    const installed = service.installed;
    const currentValues =
      installed &&
      typeof installed === "object" &&
      !("stage" in (installed as Record<string, unknown>))
        ? (installed as Record<string, unknown>)
        : null;
    const field = findImageVersionField(service.spec.fields);
    if (!field || !currentValues) return;

    const toastId = toast.loading(`Checking versions for ${service.id}...`);
    try {
      const hardware = field.depends_on
        ? currentValues[field.depends_on]
        : undefined;
      const installedTag = currentValues[field.name];
      const result = await apiClient.getDockerTags(
        service.id,
        typeof hardware === "string" ? hardware : undefined,
        field.name,
        undefined,
        typeof installedTag === "string" ? installedTag : undefined,
      );
      toast.dismiss(toastId);
      if (result.registry_unavailable) {
        toast.error(
          "Cannot list versions: DeepFellow Infra cannot reach the Docker registry.",
        );
        return;
      }
      const plan = planImageUpdate(
        typeof installedTag === "string" ? installedTag : null,
        result.default,
        result.newest,
        result.newest_needs_cuda,
      );
      if (plan.kind === "no-release") {
        toast.error("No release version found. Choose one in Edit instead.");
        return;
      }
      if (plan.kind === "up-to-date") {
        if (plan.warning) {
          toast.warning(
            `${service.id} is on the newest version this host supports (${plan.version}).`,
            { description: plan.warning, duration: 15000 },
          );
        } else {
          toast.info(
            `${service.id} is already on the newest version (${plan.version}).`,
          );
        }
        return;
      }
      modal.open(ConfirmModal, {
        title: `Update ${service.id}`,
        description: `Update ${service.id} from ${plan.from ?? "the bundled version"} to ${plan.to}? DeepFellow Infra recreates the container with the new image, keeps its other settings, and reloads the installed models. If the update fails, it restores the previous settings.${plan.warning ? ` Note: ${plan.warning}` : ""}`,
        confirmText: "Update",
        cancelText: "Cancel",
        onConfirm: () => {
          modal.close();
          const spec = { ...currentValues, [field.name]: plan.to };
          pendingInstallationRef.current = {
            serviceId: service.id,
            spec,
            size: service.size,
            update: true,
          };
          installMutation.mutate({
            serviceId: service.id,
            spec,
            size: service.size,
            update: true,
          });
        },
        variant: "warning",
      });
    } catch (error) {
      toast.dismiss(toastId);
      toast.error(
        `Failed to check versions: ${error instanceof Error ? error.message : "Unknown error"}`,
      );
    }
  };

  const canUpdateImage = (service: Service) =>
    !!findImageVersionField(service.spec?.fields) &&
    typeof service.installed === "object" &&
    service.installed !== null &&
    !("stage" in service.installed);

  const handleWarningsContinue = () => {
    if (pendingInstallationRef.current) {
      hasWarningsRef.current = false;
      installMutation.mutate({
        serviceId: pendingInstallationRef.current.serviceId,
        spec: pendingInstallationRef.current.spec,
        size: pendingInstallationRef.current.size,
        update: pendingInstallationRef.current.update,
        ignoreWarnings: true,
      });
    }
  };

  const handleUninstallClick = async (serviceId: string) => {
    // Check if service has custom models before uninstalling
    let customModelsCount = 0;
    try {
      const modelsData = await apiClient.listAdminServiceModels(serviceId);
      customModelsCount =
        modelsData.list?.filter((model) => model.custom).length || 0;
    } catch (error) {
      // If we can't fetch models, proceed anyway but log the error
      console.warn("Failed to fetch models for service:", error);
    }

    const description =
      customModelsCount > 0
        ? `Are you sure you want to uninstall ${serviceId}? This will also remove ${customModelsCount} custom model${customModelsCount > 1 ? "s" : ""} associated with this service. This action cannot be undone.`
        : `Are you sure you want to uninstall ${serviceId}? This action cannot be undone.`;

    modal.open(UninstallWithPurgeModal, {
      title: "Uninstall Service",
      description,
      confirmText: "Uninstall",
      cancelText: "Cancel",
      purgeLabel: "Purge",
      purgeDescription:
        "Also remove downloaded service files and local data to free disk space. This cannot be undone.",
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
      description:
        "This will remove all downloaded files for the service. This action cannot be undone.",
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
      "a,button,[role='button'],[role='menuitem'],input,select,textarea,label,[data-prevent-row-click]",
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
        const installedValues = Object.entries(installed).some(
          ([key, value]) => {
            const field = service.spec.fields.find((f) => f.name === key);
            if (field?.type === "password") return false;

            return (
              key.toLowerCase().includes(query) ||
              String(value ?? "")
                .toLowerCase()
                .includes(query)
            );
          },
        );
        if (installedValues) return true;
      }

      // Search in size
      const sizeStr =
        typeof service.size === "string"
          ? service.size
          : Object.entries(service.size)
              .map(([k, v]) => `${k}:${v}`)
              .join(" ");
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
                onCheckedChange={(checked) =>
                  cloudToggleMutation.mutate(checked)
                }
                disabled={cloudToggleMutation.isPending || !!loadError}
              />
              <Label htmlFor="cloud-enabled">Cloud services</Label>
            </div>
            <Button onClick={handleShowMeshInfo} variant="outline">
              Show mesh info
            </Button>
          </div>
        </div>
        {loadError && (
          <Alert variant="destructive" className="mb-4">
            <AlertCircle />
            <AlertTitle>Installed services were not loaded</AlertTitle>
            <AlertDescription>
              <p>
                The services configuration could not be read. See{" "}
                <Link to="/dashboard/warnings" className="underline">
                  Warnings
                </Link>{" "}
                for details.
              </p>
            </AlertDescription>
          </Alert>
        )}
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
              <TableHead className="text-right min-w-[165px]">
                Actions
              </TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {filteredServices.length === 0 ? (
              <TableRow>
                <TableCell
                  colSpan={4}
                  className="text-center text-muted-foreground"
                >
                  {searchQuery
                    ? "No services found matching your search"
                    : "No services available"}
                </TableCell>
              </TableRow>
            ) : (
              filteredServices.map((service) => {
                const isInstalled = !!service.installed;
                const isDownloaded = !!service.downloaded;
                const sizeInfo =
                  typeof service.size === "string"
                    ? service.size
                    : Object.entries(service.size)
                        .map(([key, val]) => `${key.toUpperCase()}: ${val}`)
                        .join(", ");

                const isInstallingCurrent =
                  installMutation.isPending &&
                  installingServiceId === service.id;
                const currentProgress = installProgress[service.id];
                const isInProgress = !!currentProgress;
                const isCancelling = cancellingServiceIds.has(service.id);
                // Check if installed is InstallProgress type (has stage and value)
                const installedIsProgress =
                  service.installed &&
                  typeof service.installed === "object" &&
                  "stage" in service.installed &&
                  "value" in service.installed;
                const isCloudDisabled = service.is_cloud && !cloudEnabled;
                const isHardwareDisabled = !!service.disabled_reason;
                const isDisabled = isCloudDisabled || isHardwareDisabled;

                return (
                  <TableRow
                    key={service.id}
                    onClick={(e) => {
                      if (
                        !isInstalled ||
                        installedIsProgress ||
                        isInProgress ||
                        isCancelling ||
                        isDisabled
                      )
                        return;
                      handleRowNavigate(e, service.id);
                    }}
                    className={
                      [
                        isInstalled &&
                        !installedIsProgress &&
                        !isInProgress &&
                        !isCancelling &&
                        !isDisabled
                          ? "cursor-pointer hover:bg-muted/50"
                          : undefined,
                        isDisabled ? "opacity-50" : undefined,
                      ]
                        .filter(Boolean)
                        .join(" ") || undefined
                    }
                  >
                    <TableCell className="font-semibold">
                      <div>
                        {isInstalled &&
                        !installedIsProgress &&
                        !isInProgress &&
                        !isCancelling ? (
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
                        {service.description &&
                          service.description.length > 0 && (
                            <div className="text-sm text-muted-foreground mt-1">
                              {service.description}
                            </div>
                          )}
                      </div>
                    </TableCell>
                    <TableCell>
                      {isInProgress || installedIsProgress ? (
                        <ProgressBadge
                          stage={
                            currentProgress?.stage ||
                            (service.installed as InstallProgress).stage
                          }
                          value={
                            currentProgress?.value ??
                            (service.installed as InstallProgress).value
                          }
                          variant="default"
                          simulated={!hasRealProgressRef.current[service.id]}
                        />
                      ) : (
                        <div className="flex flex-wrap items-center gap-2">
                          <Badge
                            variant={isInstalled ? "default" : "secondary"}
                          >
                            {isInstalled ? "Installed" : "Not installed"}
                          </Badge>
                          {!isInstalled && isDownloaded && (
                            <Badge variant="outline">Downloaded</Badge>
                          )}
                        </div>
                      )}
                    </TableCell>
                    <TableCell className="font-mono text-sm">
                      {sizeInfo || "N/A"}
                    </TableCell>
                    <TableCell className="text-right">
                      <div
                        className="flex justify-end gap-2"
                        data-prevent-row-click
                      >
                        {isCancelling ? null : isInProgress ||
                          installedIsProgress ? (
                          service.is_cloud ? null : (
                            <Button
                              onClick={() => handleCancelInstall(service.id)}
                              variant="destructive"
                              size="sm"
                            >
                              Cancel
                            </Button>
                          )
                        ) : !isInstalled ? (
                          <>
                            <TooltipProvider>
                              <Tooltip>
                                <TooltipTrigger asChild>
                                  <span>
                                    <Button
                                      onClick={() =>
                                        handleInstallClick(service, false)
                                      }
                                      size="sm"
                                      disabled={
                                        isInstallingCurrent || isDisabled
                                      }
                                    >
                                      {isInstallingCurrent
                                        ? "Installing..."
                                        : "Install"}
                                    </Button>
                                  </span>
                                </TooltipTrigger>
                                {isCloudDisabled ? (
                                  <TooltipContent>
                                    Cloud services are disabled. Enable cloud to
                                    install this service.
                                  </TooltipContent>
                                ) : (
                                  isHardwareDisabled && (
                                    <TooltipContent className="whitespace-pre-line">
                                      {service.disabled_reason}
                                    </TooltipContent>
                                  )
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
                                  disabled={
                                    !isDownloaded || purgeMutation.isPending
                                  }
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
                                modal.open(ServiceSettingsModal, {
                                  service,
                                  onEdit: handleEditClick,
                                })
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
                                  onClick={() =>
                                    handleInstallClick(service, true)
                                  }
                                >
                                  Install another instance
                                </DropdownMenuItem>
                                {canUpdateImage(service) && (
                                  <DropdownMenuItem
                                    onClick={() => handleUpdateClick(service)}
                                  >
                                    Update
                                  </DropdownMenuItem>
                                )}
                                {service.has_docker && (
                                  <>
                                    <DropdownMenuItem
                                      onClick={() =>
                                        handleShowDockerLogs(service.id)
                                      }
                                    >
                                      Docker Logs
                                    </DropdownMenuItem>
                                    <DropdownMenuItem
                                      onClick={() =>
                                        handleShowDockerCompose(service.id)
                                      }
                                    >
                                      Docker Compose
                                    </DropdownMenuItem>
                                    <DropdownMenuItem
                                      onClick={() =>
                                        handleRestartDocker(service.id)
                                      }
                                    >
                                      Restart Docker
                                    </DropdownMenuItem>
                                    <DropdownMenuSeparator />
                                  </>
                                )}
                                <DropdownMenuItem
                                  onClick={() =>
                                    handleUninstallClick(service.id)
                                  }
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
