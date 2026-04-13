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
import { useModal } from "@/hooks/use-modal";
import type { Service, InstallProgress } from "@/deepfellow/types";
import { InstallationWarningsError } from "@/deepfellow/types";
import type { ProgressEvent } from "@/utils/sse-stream";
import {
  clearServiceInstallProgress,
  setServiceInstallProgress,
  useInstallProgressSnapshot,
} from "@/state/install-progress-store";
import { toast } from "sonner";

export function ServicesList() {
  const modal = useModal();
  const navigate = useNavigate();
  const [searchQuery, setSearchQuery] = useState("");
  const [installingServiceId, setInstallingServiceId] = useState<string | null>(null);
  const pendingInstallationRef = useRef<{ serviceId: string; spec: Record<string, unknown> } | null>(null);
  const hasWarningsRef = useRef(false);
  const restartDockerToastIdRef = useRef<string | number | null>(null);
  const uninstallToastIdRef = useRef<string | number | null>(null);
  const queryClient = useQueryClient();
  const installProgress = useInstallProgressSnapshot().services;

  const { data: servicesData, isLoading } = useQuery({
    queryKey: ["admin", "services"],
    queryFn: () => apiClient.listAdminServices(),
  });
  const [anotherInstances, setAnotherInstances] = useState([] as Service[]);
  const servicesList = useMemo(() => {
    const newList = [...(servicesData ? servicesData.list : []), ...anotherInstances];
    newList.sort((a, b) => a.type.localeCompare(b.type) || a.id.localeCompare(b.id));
    return newList;
  }, [servicesData, anotherInstances])

  // Progress polling for existing installations
  useEffect(() => {
    if (!servicesList.length) return;

    const cleanups: Array<() => void> = [];

    for (const service of servicesList) {
      const installed = service.installed;
      if (!installed || typeof installed !== "object") continue;

      const stage = (installed as { stage?: unknown }).stage;
      const value = (installed as { value?: unknown }).value;
      const installedIsProgress =
        (stage === "install" || stage === "download") &&
        typeof value === "number" &&
        Number.isFinite(value);

      if (!installedIsProgress) continue;

      const serviceId = service.id;
      const abortController = new AbortController();
      let isCancelled = false;

      apiClient
        .getServiceProgress(
          serviceId,
          (event: ProgressEvent) => {
            if (isCancelled) return;

            const stage = event.stage;
            const value = event.value;

            if (event.type === "progress" && stage && value !== undefined) {
              setServiceInstallProgress(serviceId, { stage, value });
              return;
            }

            if (event.type === "finish") {
              if (event.status === "ok") {
                queryClient.invalidateQueries({ queryKey: ["admin", "services"] });
                clearServiceInstallProgress(serviceId);
              } else {
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
      });
    }

    return () => {
      for (const cleanup of cleanups) cleanup();
    };
  }, [servicesList, queryClient]);

  const installMutation = useMutation({
    mutationFn: ({ serviceId, spec, ignoreWarnings = false }: { serviceId: string; spec: Record<string, unknown>; ignoreWarnings?: boolean }) => {
      return new Promise<void>((resolve, reject) => {
        apiClient.installAdminServiceStreaming(
          serviceId,
          spec,
          (event: ProgressEvent) => {
            const stage = event.stage;
            const value = event.value;

            if (event.type === "progress" && stage && value !== undefined) {
              setServiceInstallProgress(serviceId, { stage, value });
            } else if (event.type === "finish") {
              if (event.status === "ok") {
                clearServiceInstallProgress(serviceId);
                resolve();
              } else {
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
      queryClient.invalidateQueries({ queryKey: ["admin", "services"] });
      pendingInstallationRef.current = null;
      clearServiceInstallProgress(variables.serviceId);
      toast.success("Service installed successfully");
    },
    onError: (error, variables) => {
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
      toast.error(`Failed to install service: ${error.message}`);
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

  const handleShowMeshInfo = async () => {
    try {
      const data = await apiClient.getMeshInfo();
      modal.open(MeshInfoModal, {
        meshInfo: data.info,
      });
    } catch (error) {
      toast.error(`Failed to fetch mesh info: ${error instanceof Error ? error.message : "Unknown error"}`);
    }
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
          pendingInstallationRef.current = { serviceId: serviceId, spec };
          if (installAnotherInstance) {
            setAnotherInstances([...anotherInstances, {...serviceDetail, id: serviceId, instance: instance, installed: false}]);
          }
          installMutation.mutate({ serviceId: serviceId, spec });
        },
        // Keep modal interactive; don't disable because another install is running.
        isSubmitting: false,
      });
    } catch {
      modal.close();
      toast.error("Failed to load service details");
    }
  };

  const handleWarningsContinue = () => {
    if (pendingInstallationRef.current) {
      hasWarningsRef.current = false;
      installMutation.mutate({ 
        serviceId: pendingInstallationRef.current.serviceId, 
        spec: pendingInstallationRef.current.spec,
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
          <Button
            onClick={handleShowMeshInfo}
            variant="outline"
          >
            Show mesh info
          </Button>
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
              <TableHead>Configuration</TableHead>
              <TableHead className="text-right min-w-[165px]">Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {filteredServices.length === 0 ? (
              <TableRow>
                <TableCell colSpan={5} className="text-center text-muted-foreground">
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

                return (
                  <TableRow
                    key={service.id}
                    onClick={(e) => {
                      if (!isInstalled || installedIsProgress || isInProgress) return;
                      handleRowNavigate(e, service.id);
                    }}
                    className={
                      isInstalled && !installedIsProgress && !isInProgress
                        ? "cursor-pointer hover:bg-muted/50"
                        : undefined
                    }
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
                    <TableCell>
                      {isInstalled &&
                      service.installed &&
                      typeof service.installed === "object" &&
                      !installedIsProgress ? (
                        <div className="space-y-1">
                          {Object.entries(service.installed).map(([key, value]) => {
                            const field = service.spec.fields.find((f) => f.name === key);
                            const displayValue = field?.type === "password" ? "•••••" : ((value && typeof(value) === "object") ? JSON.stringify(value) : String(value ?? ""));
                            return (
                              <div key={key} className="text-xs truncate max-w-xs">
                                <span className="font-medium">{key}:</span> {displayValue}
                              </div>
                            );
                          })}
                        </div>
                      ) : (
                        <span className="text-sm text-muted-foreground">—</span>
                      )}
                    </TableCell>
                    <TableCell className="text-right">
                      <div className="flex justify-end gap-2" data-prevent-row-click>
                        {isInProgress || installedIsProgress ? null : !isInstalled ? (
                          <>
                            <Button
                              onClick={() => handleInstallClick(service, false)}
                              size="sm"
                              disabled={isInstallingCurrent}
                            >
                              {isInstallingCurrent ? "Installing..." : "Install"}
                            </Button>
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
