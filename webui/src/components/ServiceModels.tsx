/*
DeepFellow Software Framework.
Copyright © 2025 Simplito sp. z o.o.

This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
This software is Licensed under the DeepFellow Free License.

See the License for the specific language governing permissions and
limitations under the License.
*/
import { useState, useMemo, useEffect, useRef, useDeferredValue } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "@/deepfellow/client";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
} from "@/components/ui/dropdown-menu";
import { MoreVertical } from "lucide-react";
import { Breadcrumb, BreadcrumbItem, BreadcrumbLink, BreadcrumbList, BreadcrumbPage, BreadcrumbSeparator } from "@/components/ui/breadcrumb";
import { Link } from "@tanstack/react-router";
import { DynamicFormModal } from "./DynamicFormModal";
import { ConfirmModal } from "./ConfirmModal";
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

interface ServiceModelsProps {
  serviceId: string;
}

export function ServiceModels({ serviceId }: ServiceModelsProps) {
  const modal = useModal();
  const [filterText, setFilterText] = useState("");
  const deferredFilterText = useDeferredValue(filterText);
  const [filterType, setFilterType] = useState<string>("__all");
  const [filterInstalled, setFilterInstalled] = useState<string>("__all");
  const [filterCustom, setFilterCustom] = useState<string>("__all");
  const [installingModelId, setInstallingModelId] = useState<string | null>(null);
  const [installProgress, setInstallProgress] = useState<Record<string, { stage: "install" | "download", value: number }>>({});
  const pendingInstallationRef = useRef<{ modelId: string; spec: Record<string, any> } | null>(null);
  const hasWarningsRef = useRef(false);
  const toastIdsRef = useRef<Record<string, string | number>>({});
  const restartDockerToastIdRef = useRef<string | number | null>(null);
  const testAbortControllerRef = useRef<AbortController | null>(null);
  const queryClient = useQueryClient();

  const { data: modelsData, isLoading } = useQuery({
    queryKey: ["admin", "services", serviceId, "models"],
    queryFn: () => apiClient.listAdminServiceModels(serviceId),
  });

  const { data: serviceInfo } = useQuery({
    queryKey: ["admin", "services", serviceId],
    queryFn: () => apiClient.getAdminService(serviceId),
  });

  // Update toast messages as installation progress changes
  useEffect(() => {
    Object.entries(installProgress).forEach(([modelId, progress]) => {
      const toastId = toastIdsRef.current[modelId];
      if (toastId) {
        const percentage = (progress.value * 100).toFixed(1);
        const stageLabel = getStageLabel(progress.stage);
        toast.loading(`${stageLabel} ${modelId}: ${percentage}%`, { id: toastId });
      }
    });
  }, [installProgress]);

  // Progress polling for existing installations
  useEffect(() => {
    if (!modelsData?.list) return;

    const progressPromises: Array<() => void> = [];

    modelsData.list.forEach((model) => {
      // Check if model has installation in progress (has stage property)
      const installed = model.installed;
      if (installed && installed.stage && installed.value !== undefined) {
        const modelId = model.id;

        // Start progress polling
        let isCancelled = false;

        const startProgress = async () => {
          try {
            await apiClient.getModelProgress(serviceId, modelId, (event: ProgressEvent) => {
              if (isCancelled) return;

              if (event.type === "progress" && event.stage && event.value !== undefined) {
                setInstallProgress((prev) => ({
                  ...prev,
                  [modelId]: { stage: event.stage!, value: event.value! },
                }));
              } else if (event.type === "finish") {
                if (event.status === "ok") {
                  queryClient.invalidateQueries({ queryKey: ["admin", "services", serviceId, "models"] });
                  setInstallProgress((prev) => {
                    const newProgress = { ...prev };
                    delete newProgress[modelId];
                    return newProgress;
                  });
                } else {
                  setInstallProgress((prev) => {
                    const newProgress = { ...prev };
                    delete newProgress[modelId];
                    return newProgress;
                  });
                  toast.error(`Installation failed for ${modelId}: ${event.details || "Unknown error"}`);
                }
              }
            });
          } catch (error) {
            if (!isCancelled) {
              console.error(`Error polling progress for ${modelId}:`, error);
            }
          }
        };

        startProgress();

        progressPromises.push(() => {
          isCancelled = true;
        });
      }
    });

    return () => {
      progressPromises.forEach((cleanup) => cleanup());
    };
  }, [modelsData, serviceId, queryClient]);

  const installMutation = useMutation({
    mutationFn: ({ modelId, spec, ignoreWarnings = false }: { modelId: string; spec: Record<string, any>; ignoreWarnings?: boolean }) => {
      return new Promise<void>((resolve, reject) => {
        apiClient.installAdminServiceModelStreaming(
          serviceId,
          modelId,
          spec,
          (event: ProgressEvent) => {
            if (event.type === "progress" && event.stage && event.value !== undefined) {
              setInstallProgress((prev) => ({
                ...prev,
                [modelId]: { stage: event.stage!, value: event.value! },
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
      });
    },
    onMutate: ({ modelId }) => {
      setInstallingModelId(modelId);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["admin", "services", serviceId, "models"] });
      pendingInstallationRef.current = null;
      const modelId = installingModelId || "";
      const toastId = toastIdsRef.current[modelId];
      if (toastId) {
        toast.success(`Model ${modelId} installed successfully`, { id: toastId });
        delete toastIdsRef.current[modelId];
      } else {
        toast.success("Model installed successfully");
      }
      setInstallProgress((prev) => {
        const newProgress = { ...prev };
        delete newProgress[modelId];
        return newProgress;
      });
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
      const modelId = installingModelId || "";
      const toastId = toastIdsRef.current[modelId];
      if (toastId) {
        toast.error(`Failed to install model ${modelId}: ${error.message}`, { id: toastId });
        delete toastIdsRef.current[modelId];
      } else {
        toast.error(`Failed to install model: ${error.message}`);
      }
      setInstallProgress((prev) => {
        const newProgress = { ...prev };
        delete newProgress[modelId];
        return newProgress;
      });
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
    mutationFn: async (spec: Record<string, any>) => {
      return apiClient.addCustomModel(serviceId, spec);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["admin", "services", serviceId, "models"] });
      modal.close();
      toast.success("Custom model added successfully");
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

  const handleShowDockerLogs = async (modelId: string) => {
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
  };

  const handleShowDockerCompose = async (modelId: string) => {
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
  };

  const handleRestartDocker = (modelId: string) => {
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
  };

  const handleInstallClick = async (model: ServiceModel) => {
    try {
      const modelDetail = await apiClient.getAdminServiceModel(serviceId, model.id);
      modal.open(DynamicFormModal, {
        title: `Install ${modelDetail.id}`,
        fields: modelDetail.spec.fields,
        onSubmit: (spec: Record<string, any>) => {
          const cleanedSpec = Object.fromEntries(
            Object.entries(spec).filter(([_, value]) => value !== null && value !== undefined)
          );
          pendingInstallationRef.current = { modelId: modelDetail.id, spec: cleanedSpec };
          modal.close();
          const toastId: string | number = toast.loading(`Starting installation for ${modelDetail.id}...`);
          toastIdsRef.current[modelDetail.id] = toastId;
          installMutation.mutate({ modelId: modelDetail.id, spec: cleanedSpec });
        },
        isSubmitting: installMutation.isPending,
      });
    } catch (error) {
      toast.error("Failed to load model details");
    }
  };

  const handleWarningsContinue = () => {
    if (pendingInstallationRef.current) {
      hasWarningsRef.current = false;
      installMutation.mutate({ 
        modelId: pendingInstallationRef.current.modelId, 
        spec: pendingInstallationRef.current.spec,
        ignoreWarnings: true 
      });
    }
  };

  const handleUninstallClick = (modelId: string) => {
    modal.open(ConfirmModal, {
      title: "Uninstall Model",
      description: `Are you sure you want to uninstall ${modelId}? This action cannot be undone.`,
      confirmText: "Uninstall",
      cancelText: "Cancel",
      onConfirm: () => uninstallMutation.mutate(modelId),
      isLoading: uninstallMutation.isPending,
      variant: "destructive",
    });
  };

  const handleTestClick = async (model: ServiceModel) => {
    if (!model.installed?.registration_id) {
      toast.error("Model registration ID not found. Cannot test model.");
      return;
    }
    testMutation.mutate(model.installed.registration_id);
  };

  const handleAddCustomModel = () => {
    if (!serviceInfo?.custom_model_spec) return;
    modal.open(DynamicFormModal, {
      title: "Add custom model",
      fields: serviceInfo.custom_model_spec.fields,
      onSubmit: (spec: Record<string, any>) => {
        const cleanedSpec = Object.fromEntries(
          Object.entries(spec).filter(([_, value]) => value !== null && value !== undefined)
        );
        addCustomModelMutation.mutate(cleanedSpec);
      },
      isSubmitting: addCustomModelMutation.isPending,
    });
  };

  const handleRemoveCustomModelClick = (model: ServiceModel) => {
    if (!model.custom) {
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
      onConfirm: () => removeCustomModelMutation.mutate({ customModelId: model.custom! }),
      isLoading: removeCustomModelMutation.isPending,
      variant: "destructive",
    });
  };

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
      const matchesCustom =
        filterCustom === "__all" ||
        (filterCustom === "onlycustom" && !!model.custom) ||
        (filterCustom === "onlynotcustom" && !model.custom);

      return matchesText && matchesType && matchesInstalled && matchesCustom;
    });
  }, [sortedModels, deferredFilterText, filterType, filterInstalled, filterCustom]);

  if (isLoading) {
    return <div className="text-center py-8">Loading models...</div>;
  }

  return (
    <div className="container mx-auto p-6">
      <div className="mb-6">
        <Breadcrumb className="mb-4">
          <BreadcrumbList>
            <BreadcrumbItem>
              <BreadcrumbLink asChild>
                <Link to="/dashboard">Services</Link>
              </BreadcrumbLink>
            </BreadcrumbItem>
            <BreadcrumbSeparator />
            <BreadcrumbItem>
              <BreadcrumbPage>{serviceId}</BreadcrumbPage>
            </BreadcrumbItem>
          </BreadcrumbList>
        </Breadcrumb>

        <div className="flex items-center justify-between mb-6">
          <h1 className="text-3xl font-bold">Models for {serviceId}</h1>
          {serviceInfo?.custom_model_spec && (
            <Button onClick={handleAddCustomModel}>
              Add custom model
            </Button>
          )}
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
              <Select value={filterType} onValueChange={setFilterType}>
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
              <Select value={filterInstalled} onValueChange={setFilterInstalled}>
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

            {serviceInfo?.custom_model_spec && (
              <div className="space-y-1">
                <Select value={filterCustom} onValueChange={setFilterCustom}>
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

      <div className="border rounded-lg">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-[40%]">Model ID</TableHead>
              <TableHead>Type</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>Size</TableHead>
              <TableHead>Configuration</TableHead>
              <TableHead className="text-right">Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {filteredModels.length === 0 ? (
              <TableRow>
                <TableCell colSpan={6} className="text-center text-muted-foreground">
                  No models found
                </TableCell>
              </TableRow>
            ) : (
              filteredModels.map((model) => {
                const isInstalled = !!model.installed;
                const installedSpec = model.installed?.spec || {};
                const isInstallingCurrent = installMutation.isPending && installingModelId === model.id;
                const currentProgress = installProgress[model.id];
                const isInProgress = !!currentProgress;
                const hasProgressStage = model.installed?.stage && model.installed?.value !== undefined;

                return (
                  <TableRow key={model.id}>
                    <TableCell className="font-medium">
                      <div className="flex items-center gap-2">
                        <div className="truncate max-w-md" title={model.id}>
                          {model.id}
                        </div>
                        {model.custom && (
                          <Badge variant="secondary">Custom</Badge>
                        )}
                      </div>
                    </TableCell>
                    <TableCell className="text-sm">
                      {MODEL_TYPES[model.type] || model.type}
                    </TableCell>
                    <TableCell>
                      {isInProgress || hasProgressStage ? (
                        <ProgressBadge
                          stage={currentProgress?.stage || (model.installed?.stage as "install" | "download")}
                          value={currentProgress?.value ?? model.installed?.value ?? 0}
                          variant="default"
                        />
                      ) : (
                        <Badge variant={isInstalled ? "default" : "secondary"}>
                          {isInstalled ? "Installed" : "Not installed"}
                        </Badge>
                      )}
                    </TableCell>
                    <TableCell className="font-mono text-sm">{model.size || "N/A"}</TableCell>
                    <TableCell>
                      {isInstalled && Object.keys(installedSpec).length > 0 && !hasProgressStage ? (
                        <div className="space-y-1">
                          {Object.entries(installedSpec).map(([key, value]) => {
                            const field = model.spec.fields.find((f) => f.name === key);
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
                      {isInProgress || hasProgressStage ? null : !isInstalled && model.custom ? (
                        <Button
                          onClick={() => handleRemoveCustomModelClick(model)}
                          variant="destructive"
                          size="sm"
                          disabled={removeCustomModelMutation.isPending}
                        >
                          {removeCustomModelMutation.isPending ? "Removing..." : "Remove custom model"}
                        </Button>
                      ) : !isInstalled ? (
                        <Button
                          onClick={() => handleInstallClick(model)}
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
                            <DropdownMenuItem
                              onClick={() => handleTestClick(model)}
                              disabled={testMutation.isPending}
                            >
                              {testMutation.isPending ? "Testing..." : "Test"}
                            </DropdownMenuItem>
                            {(model.has_docker) && (
                              <>
                                <DropdownMenuSeparator />
                                <DropdownMenuItem
                                  onClick={() => handleShowDockerLogs(model.id)}
                                >
                                  Docker Logs
                                </DropdownMenuItem>
                                <DropdownMenuItem
                                  onClick={() => handleShowDockerCompose(model.id)}
                                >
                                  Docker Compose
                                </DropdownMenuItem>
                                <DropdownMenuItem
                                  onClick={() => handleRestartDocker(model.id)}
                                >
                                  Restart Docker
                                </DropdownMenuItem>
                              </>
                            )}
                            <DropdownMenuSeparator />
                            <DropdownMenuItem
                              onClick={() => handleUninstallClick(model.id)}
                              variant="destructive"
                            >
                              Uninstall
                            </DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      )}
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
