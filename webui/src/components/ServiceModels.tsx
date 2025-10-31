/*
DeepFellow Software Framework.
Copyright © 2025 Simplito sp. z o.o.

This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
This software is Licensed under the DeepFellow Free License.

See the License for the specific language governing permissions and
limitations under the License.
*/
import { useState, useMemo } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "@/deepfellow/client";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Breadcrumb, BreadcrumbItem, BreadcrumbLink, BreadcrumbList, BreadcrumbPage, BreadcrumbSeparator } from "@/components/ui/breadcrumb";
import { Link } from "@tanstack/react-router";
import { DynamicFormModal } from "./DynamicFormModal";
import { ConfirmModal } from "./ConfirmModal";
import type { ServiceModel } from "@/deepfellow/types";
import { MODEL_TYPES } from "@/deepfellow/types";
import { toast } from "sonner";

interface ServiceModelsProps {
  serviceId: string;
}

export function ServiceModels({ serviceId }: ServiceModelsProps) {
  const [filterText, setFilterText] = useState("");
  const [filterType, setFilterType] = useState<string>("__all");
  const [filterInstalled, setFilterInstalled] = useState<string>("__all");
  const [installModalOpen, setInstallModalOpen] = useState(false);
  const [selectedModel, setSelectedModel] = useState<ServiceModel | null>(null);
  const [uninstallModalOpen, setUninstallModalOpen] = useState(false);
  const [modelToUninstall, setModelToUninstall] = useState<string | null>(null);
  const [installingModelId, setInstallingModelId] = useState<string | null>(null);
  const queryClient = useQueryClient();

  const { data: modelsData, isLoading } = useQuery({
    queryKey: ["admin", "services", serviceId, "models"],
    queryFn: () => apiClient.listAdminServiceModels(serviceId),
  });

  const installMutation = useMutation({
    mutationFn: ({ modelId, spec }: { modelId: string; spec: Record<string, any> }) =>
      apiClient.installAdminServiceModel(serviceId, modelId, spec),
    onMutate: ({ modelId }) => {
      setInstallingModelId(modelId);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["admin", "services", serviceId, "models"] });
      setInstallModalOpen(false);
      setSelectedModel(null);
      toast.success("Model installed successfully");
    },
    onError: (error) => {
      toast.error(`Failed to install model: ${error.message}`);
    },
    onSettled: () => {
      setInstallingModelId(null);
    },
  });

  const uninstallMutation = useMutation({
    mutationFn: (modelId: string) => apiClient.uninstallAdminServiceModel(serviceId, modelId, false),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["admin", "services", serviceId, "models"] });
      setUninstallModalOpen(false);
      setModelToUninstall(null);
      toast.success("Model uninstalled successfully");
    },
    onError: (error) => {
      toast.error(`Failed to uninstall model: ${error.message}`);
    },
  });

  const handleInstallClick = async (model: ServiceModel) => {
    try {
      const modelDetail = await apiClient.getAdminServiceModel(serviceId, model.id);
      setSelectedModel(modelDetail);
      setInstallModalOpen(true);
    } catch (error) {
      toast.error("Failed to load model details");
    }
  };

  const handleInstallSubmit = (spec: Record<string, any>) => {
    if (selectedModel) {
      const cleanedSpec = Object.fromEntries(
        Object.entries(spec).filter(([_, value]) => value !== null && value !== undefined)
      );
      installMutation.mutate({ modelId: selectedModel.id, spec: cleanedSpec });
    }
  };

  const handleUninstallClick = (modelId: string) => {
    setModelToUninstall(modelId);
    setUninstallModalOpen(true);
  };

  const handleUninstallConfirm = () => {
    if (modelToUninstall) {
      uninstallMutation.mutate(modelToUninstall);
    }
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

  const isModelVisible = (model: ServiceModel) => {
    const matchesText = !filterText || model.id.toLowerCase().includes(filterText.toLowerCase());
    const matchesType = filterType === "__all" || model.type === filterType;
    const matchesInstalled =
      filterInstalled === "__all" ||
      (filterInstalled === "installed" && model.installed) ||
      (filterInstalled === "notinstalled" && !model.installed);

    return matchesText && matchesType && matchesInstalled;
  };

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

        <h1 className="text-3xl font-bold mb-6">Models for {serviceId}</h1>

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
            {sortedModels.length === 0 ? (
              <TableRow>
                <TableCell colSpan={6} className="text-center text-muted-foreground">
                  No models found
                </TableCell>
              </TableRow>
            ) : (
              sortedModels.map((model) => {
                const isInstalled = !!model.installed;
                const installedSpec = model.installed?.spec || {};
                const visible = isModelVisible(model);
                const isInstallingCurrent = installMutation.isPending && installingModelId === model.id;

                return (
                  <TableRow key={model.id} style={{ display: visible ? undefined : "none" }}>
                    <TableCell className="font-medium">
                      <div className="truncate max-w-md" title={model.id}>
                        {model.id}
                      </div>
                    </TableCell>
                    <TableCell className="text-sm">
                      {MODEL_TYPES[model.type] || model.type}
                    </TableCell>
                    <TableCell>
                      <Badge variant={isInstalled ? "default" : "secondary"}>
                        {isInstalled ? "Installed" : "Not installed"}
                      </Badge>
                    </TableCell>
                    <TableCell className="font-mono text-sm">{model.size || "N/A"}</TableCell>
                    <TableCell>
                      {isInstalled && Object.keys(installedSpec).length > 0 ? (
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
                      {!isInstalled ? (
                        <Button
                          onClick={() => handleInstallClick(model)}
                          size="sm"
                          disabled={isInstallingCurrent}
                        >
                          {isInstallingCurrent ? "Installing..." : "Install"}
                        </Button>
                      ) : (
                        <Button
                          onClick={() => handleUninstallClick(model.id)}
                          variant="destructive"
                          size="sm"
                        >
                          Uninstall
                        </Button>
                      )}
                    </TableCell>
                  </TableRow>
                );
              })
            )}
          </TableBody>
        </Table>
      </div>

      {selectedModel && (
        <DynamicFormModal
          open={installModalOpen}
          onOpenChange={setInstallModalOpen}
          title={`Install ${selectedModel.id}`}
          fields={selectedModel.spec.fields}
          onSubmit={handleInstallSubmit}
          isSubmitting={installMutation.isPending}
        />
      )}

      <ConfirmModal
        open={uninstallModalOpen}
        onOpenChange={setUninstallModalOpen}
        title="Uninstall Model"
        description={`Are you sure you want to uninstall ${modelToUninstall}? This action cannot be undone.`}
        confirmText="Uninstall"
        cancelText="Cancel"
        onConfirm={handleUninstallConfirm}
        isLoading={uninstallMutation.isPending}
        variant="destructive"
      />
    </div>
  );
}
