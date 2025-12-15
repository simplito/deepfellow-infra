/*
DeepFellow Software Framework.
Copyright © 2025 Simplito sp. z o.o.

This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
This software is Licensed under the DeepFellow Free License.

See the License for the specific language governing permissions and
limitations under the License.
*/
import { useState, useMemo, useEffect, useRef } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import { apiClient } from "@/deepfellow/client";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
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
import { ProgressBadge } from "./ProgressBadge";
import { ContentModal } from "./ContentModal";
import { WarningsModal } from "./WarningsModal";
import { MeshInfoModal } from "./MeshInfoModal";
import { useModal } from "@/hooks/use-modal";
import type { Service, InstallProgress } from "@/deepfellow/types";
import { InstallationWarningsError } from "@/deepfellow/types";
import type { ProgressEvent } from "@/utils/sse-stream";
import { toast } from "sonner";

export function ServicesList() {
  const modal = useModal();
  const [searchQuery, setSearchQuery] = useState("");
  const [installingServiceId, setInstallingServiceId] = useState<string | null>(null);
  const [installProgress, setInstallProgress] = useState<Record<string, { stage: "install" | "download", value: number }>>({});
  const pendingInstallationRef = useRef<{ serviceId: string; spec: Record<string, any> } | null>(null);
  const hasWarningsRef = useRef(false);
  const restartDockerToastIdRef = useRef<string | number | null>(null);
  const uninstallToastIdRef = useRef<string | number | null>(null);
  const queryClient = useQueryClient();

  const { data: servicesData, isLoading } = useQuery({
    queryKey: ["admin", "services"],
    queryFn: () => apiClient.listAdminServices(),
  });

  // Progress polling for existing installations
  useEffect(() => {
    if (!servicesData?.list) return;

    const progressPromises: Array<() => void> = [];

    servicesData.list.forEach((service) => {
      // Check if service has installation in progress (has stage property)
      const installed = service.installed;
      if (installed && typeof installed === "object" && "stage" in installed && "value" in installed) {
        const serviceId = service.id;

        // Start progress polling
        const abortController = new AbortController();
        let isCancelled = false;

        const startProgress = async () => {
          try {
            await apiClient.getServiceProgress(serviceId, (event: ProgressEvent) => {
              if (isCancelled) return;

              if (event.type === "progress" && event.stage && event.value !== undefined) {
                setInstallProgress((prev) => ({
                  ...prev,
                  [serviceId]: { stage: event.stage!, value: event.value! },
                }));
              } else if (event.type === "finish") {
                if (event.status === "ok") {
                  queryClient.invalidateQueries({ queryKey: ["admin", "services"] });
                  setInstallProgress((prev) => {
                    const newProgress = { ...prev };
                    delete newProgress[serviceId];
                    return newProgress;
                  });
                } else {
                  setInstallProgress((prev) => {
                    const newProgress = { ...prev };
                    delete newProgress[serviceId];
                    return newProgress;
                  });
                  toast.error(`Installation failed for ${serviceId}: ${event.details || "Unknown error"}`);
                }
              }
            });
          } catch (error) {
            if (!isCancelled) {
              console.error(`Error polling progress for ${serviceId}:`, error);
            }
          }
        };

        startProgress();

        progressPromises.push(() => {
          isCancelled = true;
          abortController.abort();
        });
      }
    });

    return () => {
      progressPromises.forEach((cleanup) => cleanup());
    };
  }, [servicesData, queryClient]);

  const installMutation = useMutation({
    mutationFn: ({ serviceId, spec, ignoreWarnings = false }: { serviceId: string; spec: Record<string, any>; ignoreWarnings?: boolean }) => {
      return new Promise<void>((resolve, reject) => {
        apiClient.installAdminServiceStreaming(
          serviceId,
          spec,
          (event: ProgressEvent) => {
            if (event.type === "progress" && event.stage && event.value !== undefined) {
              setInstallProgress((prev) => ({
                ...prev,
                [serviceId]: { stage: event.stage!, value: event.value! },
              }));
            } else if (event.type === "finish") {
              if (event.status === "ok") {
                resolve();
              } else {
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
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["admin", "services"] });
      pendingInstallationRef.current = null;
      setInstallProgress((prev) => {
        const newProgress = { ...prev };
        delete newProgress[installingServiceId || ""];
        return newProgress;
      });
      toast.success("Service installed successfully");
    },
    onError: (error) => {
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
      setInstallProgress((prev) => {
        const newProgress = { ...prev };
        delete newProgress[installingServiceId || ""];
        return newProgress;
      });
      toast.error(`Failed to install service: ${error.message}`);
    },
    onSettled: () => {
      // Only reset if not showing warnings modal
      if (!hasWarningsRef.current) {
        setInstallingServiceId(null);
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

  const handleInstallClick = async (service: Service) => {
    try {
      const serviceDetail = await apiClient.getAdminService(service.id);
      modal.open(DynamicFormModal, {
        title: `Install ${serviceDetail.id}`,
        fields: serviceDetail.spec.fields,
        onSubmit: (spec: Record<string, any>) => {
          pendingInstallationRef.current = { serviceId: serviceDetail.id, spec };
          installMutation.mutate({ serviceId: serviceDetail.id, spec });
        },
        isSubmitting: installMutation.isPending,
      });
    } catch (error) {
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

    modal.open(ConfirmModal, {
      title: "Uninstall Service",
      description,
      confirmText: "Uninstall",
      cancelText: "Cancel",
      onConfirm: () => {
        modal.close();
        uninstallMutation.mutate(serviceId);
      },
      isLoading: uninstallMutation.isPending,
      variant: "destructive",
    });
  };

  const services = servicesData?.list || [];

  const filteredServices = useMemo(() => {
    if (!searchQuery.trim()) return services;

    const query = searchQuery.toLowerCase();
    return services.filter((service) => {
      if (service.id.toLowerCase().includes(query)) return true;

      if (service.installed) {
        const installedValues = Object.entries(service.installed).some(([key, value]) => {
          const field = service.spec.fields.find((f) => f.name === key);
          if (field?.type === "password") return false; 
          
          return key.toLowerCase().includes(query) ||
                 String(value ?? "").toLowerCase().includes(query);
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
    return <div className="text-center py-8">Loading services...</div>;
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
              <TableHead>Status</TableHead>
              <TableHead>Resources</TableHead>
              <TableHead>Configuration</TableHead>
              <TableHead className="text-right">Actions</TableHead>
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
                const sizeInfo = typeof service.size === "string"
                  ? service.size
                  : Object.entries(service.size).map(([key, val]) => `${key.toUpperCase()}: ${val}`).join(", ");

                const isInstallingCurrent = installMutation.isPending && installingServiceId === service.id;
                const currentProgress = installProgress[service.id];
                const isInProgress = !!currentProgress;
                // Check if installed is InstallProgress type (has stage and value)
                const installedIsProgress = service.installed && typeof service.installed === "object" && "stage" in service.installed && "value" in service.installed;

                return (
                  <TableRow key={service.id}>
                    <TableCell className="font-medium">
                      <div>
                        {isInstalled && !installedIsProgress ? (
                          <Link
                            to="/dashboard/services/$serviceId"
                            params={{ serviceId: service.id }}
                            className="text-primary hover:underline"
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
                        <Badge variant={isInstalled ? "default" : "secondary"}>
                          {isInstalled ? "Installed" : "Not installed"}
                        </Badge>
                      )}
                    </TableCell>
                    <TableCell className="font-mono text-sm">{sizeInfo || "N/A"}</TableCell>
                    <TableCell>
                      {isInstalled && service.installed && !installedIsProgress ? (
                        <div className="space-y-1">
                          {Object.entries(service.installed).map(([key, value]) => {
                            const field = service.spec.fields.find((f) => f.name === key);
                            const displayValue = field?.type === "password" ? "•••••" : String(value ?? "");
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
                      <div className="flex justify-end gap-2">
                        {isInProgress || installedIsProgress ? null : !isInstalled ? (
                          <Button
                            onClick={() => handleInstallClick(service)}
                            size="sm"
                            disabled={isInstallingCurrent}
                          >
                            {isInstallingCurrent ? "Installing..." : "Install"}
                          </Button>
                        ) : (
                          <DropdownMenu>
                            <DropdownMenuTrigger asChild>
                              <Button variant="outline" size="sm">
                                <MoreVertical className="h-4 w-4" />
                              </Button>
                            </DropdownMenuTrigger>
                            <DropdownMenuContent align="end">
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
                                </>
                              )}
                              <DropdownMenuSeparator />
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
