import { useState, useMemo } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "@tanstack/react-router";
import { apiClient } from "@/deepfellow/client";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { DynamicFormModal } from "./DynamicFormModal";
import { ConfirmModal } from "./ConfirmModal";
import type { Service } from "@/deepfellow/types";
import { toast } from "sonner";

export function ServicesList() {
  const [installModalOpen, setInstallModalOpen] = useState(false);
  const [selectedService, setSelectedService] = useState<Service | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [uninstallModalOpen, setUninstallModalOpen] = useState(false);
  const [serviceToUninstall, setServiceToUninstall] = useState<string | null>(null);
  const [installingServiceId, setInstallingServiceId] = useState<string | null>(null);
  const queryClient = useQueryClient();
  const navigate = useNavigate();

  const { data: servicesData, isLoading } = useQuery({
    queryKey: ["admin", "services"],
    queryFn: () => apiClient.listAdminServices(),
  });

  const installMutation = useMutation({
    mutationFn: ({ serviceId, spec }: { serviceId: string; spec: Record<string, any> }) =>
      apiClient.installAdminService(serviceId, spec),
    onMutate: ({ serviceId }) => {
      setInstallingServiceId(serviceId);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["admin", "services"] });
      setInstallModalOpen(false);
      setSelectedService(null);
      toast.success("Service installed successfully");
    },
    onError: (error) => {
      toast.error(`Failed to install service: ${error.message}`);
    },
    onSettled: () => {
      setInstallingServiceId(null);
    },
  });

  const uninstallMutation = useMutation({
    mutationFn: (serviceId: string) => apiClient.uninstallAdminService(serviceId, false),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["admin", "services"] });
      setUninstallModalOpen(false);
      setServiceToUninstall(null);
      toast.success("Service uninstalled successfully");
    },
    onError: (error) => {
      toast.error(`Failed to uninstall service: ${error.message}`);
    },
  });

  const handleInstallClick = async (service: Service) => {
    try {
      const serviceDetail = await apiClient.getAdminService(service.id);
      setSelectedService(serviceDetail);
      setInstallModalOpen(true);
    } catch (error) {
      toast.error("Failed to load service details");
    }
  };

  const handleInstallSubmit = (spec: Record<string, any>) => {
    if (selectedService) {
      installMutation.mutate({ serviceId: selectedService.id, spec });
    }
  };

  const handleUninstallClick = (serviceId: string) => {
    setServiceToUninstall(serviceId);
    setUninstallModalOpen(true);
  };

  const handleUninstallConfirm = () => {
    if (serviceToUninstall) {
      uninstallMutation.mutate(serviceToUninstall);
    }
  };

  const handleOpenModels = (serviceId: string) => {
    navigate({ to: "/dashboard/services/$serviceId", params: { serviceId } });
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
      <div className="mb-4">
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
              <TableHead>Size</TableHead>
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

                return (
                  <TableRow key={service.id}>
                    <TableCell className="font-medium">{service.id}</TableCell>
                    <TableCell>
                      <Badge variant={isInstalled ? "default" : "secondary"}>
                        {isInstalled ? "Installed" : "Not installed"}
                      </Badge>
                    </TableCell>
                    <TableCell className="font-mono text-sm">{sizeInfo || "N/A"}</TableCell>
                    <TableCell>
                      {isInstalled && service.installed ? (
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
                        {!isInstalled ? (
                          <Button
                            onClick={() => handleInstallClick(service)}
                            size="sm"
                            disabled={isInstallingCurrent}
                          >
                            {isInstallingCurrent ? "Installing..." : "Install"}
                          </Button>
                        ) : (
                          <>
                            <Button onClick={() => handleOpenModels(service.id)} size="sm" variant="outline">
                              Models
                            </Button>
                            <Button
                              onClick={() => handleUninstallClick(service.id)}
                              variant="destructive"
                              size="sm"
                            >
                              Uninstall
                            </Button>
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

      {selectedService && (
        <DynamicFormModal
          open={installModalOpen}
          onOpenChange={setInstallModalOpen}
          title={`Install ${selectedService.id}`}
          fields={selectedService.spec.fields}
          onSubmit={handleInstallSubmit}
          isSubmitting={installMutation.isPending}
        />
      )}

      <ConfirmModal
        open={uninstallModalOpen}
        onOpenChange={setUninstallModalOpen}
        title="Uninstall Service"
        description={`Are you sure you want to uninstall ${serviceToUninstall}? This action cannot be undone.`}
        confirmText="Uninstall"
        cancelText="Cancel"
        onConfirm={handleUninstallConfirm}
        isLoading={uninstallMutation.isPending}
        variant="destructive"
      />
    </div>
  );
}
