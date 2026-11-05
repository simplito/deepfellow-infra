/*
DeepFellow Software Framework.
Copyright © 2025 Simplito sp. z o.o.

This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
This software is Licensed under the DeepFellow Free License.

See the License for the specific language governing permissions and
limitations under the License.
*/
import { ConfirmModal } from "@/components/ConfirmModal";
import { SiteHeader } from "@/components/dashboard/site-header";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { apiClient } from "@/deepfellow/client";
import type { ServiceWarning } from "@/deepfellow/types";
import { useRequireAuth } from "@/hooks/use-auth";
import { useModal } from "@/hooks/use-modal";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { createFileRoute } from "@tanstack/react-router";
import { AlertCircle, Trash2, X } from "lucide-react";
import { toast } from "sonner";

export const Route = createFileRoute("/dashboard/warnings")({
  component: WarningsPage,
});

function formatTimestamp(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

function WarningRow({
  warning,
  onDismiss,
  isDismissing,
}: {
  warning: ServiceWarning;
  onDismiss: (id: string) => void;
  isDismissing: boolean;
}) {
  return (
    <TableRow>
      <TableCell className="pl-6 whitespace-nowrap text-sm text-muted-foreground">
        {formatTimestamp(warning.created_at)}
      </TableCell>
      <TableCell className="font-mono text-sm">
        {warning.service_id}
        {warning.instance ? `|${warning.instance}` : ""}
        {warning.model_id ? ` · ${warning.model_id}` : ""}
      </TableCell>
      <TableCell className="text-sm whitespace-normal break-words">
        {warning.message}
      </TableCell>
      <TableCell className="pr-6 text-right">
        <Button
          variant="ghost"
          size="icon"
          className="size-7"
          onClick={() => onDismiss(warning.id)}
          disabled={isDismissing}
          title="Dismiss"
        >
          <X className="size-4" />
        </Button>
      </TableCell>
    </TableRow>
  );
}

function WarningsPage() {
  useRequireAuth();

  const modal = useModal();
  const queryClient = useQueryClient();

  const { data, isLoading, error } = useQuery({
    queryKey: ["admin", "warnings"],
    queryFn: () => apiClient.listAdminWarnings(),
    refetchInterval: 30000,
  });

  const dismissMutation = useMutation({
    mutationFn: (warningId: string) => apiClient.dismissAdminWarning(warningId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["admin", "warnings"] });
    },
    onError: (e) => {
      toast.error(
        `Failed to dismiss warning: ${e instanceof Error ? e.message : String(e)}`,
      );
    },
  });

  const dismissAllMutation = useMutation({
    mutationFn: () => apiClient.dismissAllAdminWarnings(),
    onSuccess: () => {
      modal.close();
      queryClient.invalidateQueries({ queryKey: ["admin", "warnings"] });
    },
    onError: (e) => {
      toast.error(
        `Failed to dismiss warnings: ${e instanceof Error ? e.message : String(e)}`,
      );
    },
  });

  const warnings = data?.list ?? [];

  const handleDismissAll = () => {
    modal.open(ConfirmModal, {
      title: "Dismiss all warnings",
      description: `This will dismiss all ${warnings.length} warning${warnings.length === 1 ? "" : "s"}. This cannot be undone.`,
      confirmText: "Dismiss all",
      onConfirm: () => dismissAllMutation.mutate(),
      isLoading: dismissAllMutation.isPending,
      variant: "destructive",
    });
  };

  return (
    <>
      <SiteHeader breadcrumbs={[{ label: "Warnings" }]} />
      <div className="flex flex-1 flex-col">
        <div className="@container/main flex flex-1 flex-col gap-2">
          <div className="py-4 md:py-6 px-4 lg:px-6 max-w-[1800px] w-full mx-auto">
            <Card>
              <CardHeader className="flex flex-row items-start justify-between gap-4">
                <div>
                  <CardTitle>Warnings</CardTitle>
                  <CardDescription>
                    Services and models that failed to load, or that are
                    configured but no longer recognized.
                  </CardDescription>
                </div>
                {warnings.length > 0 && (
                  <Button
                    variant="ghost"
                    size="sm"
                    className="text-muted-foreground hover:text-destructive hover:bg-destructive/10"
                    onClick={handleDismissAll}
                    disabled={dismissAllMutation.isPending}
                  >
                    <Trash2 className="size-4" />
                    Dismiss all
                  </Button>
                )}
              </CardHeader>
              <CardContent className="p-0">
                {isLoading && (
                  <div className="space-y-2 p-6">
                    {Array.from({ length: 4 }).map((_, i) => (
                      // biome-ignore lint/suspicious/noArrayIndexKey: static skeleton list
                      <Skeleton key={i} className="h-10 w-full" />
                    ))}
                  </div>
                )}
                {error && (
                  <p className="text-destructive text-sm p-6">
                    Failed to load warnings: {error.message}
                  </p>
                )}
                {data?.load_error && (
                  <div className="p-6">
                    <Alert variant="destructive">
                      <AlertCircle />
                      <AlertTitle>
                        Installed services were not loaded
                      </AlertTitle>
                      <AlertDescription>{data.load_error}</AlertDescription>
                    </Alert>
                  </div>
                )}
                {data && !data.load_error && warnings.length === 0 && (
                  <p className="text-muted-foreground text-sm p-6">
                    No warnings.
                  </p>
                )}
                {data && warnings.length > 0 && (
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead className="pl-6">When</TableHead>
                        <TableHead>Where</TableHead>
                        <TableHead>What</TableHead>
                        <TableHead className="w-[60px] pr-6 text-right">
                          Actions
                        </TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {warnings.map((warning) => (
                        <WarningRow
                          key={warning.id}
                          warning={warning}
                          onDismiss={(id) => dismissMutation.mutate(id)}
                          isDismissing={
                            dismissMutation.isPending &&
                            dismissMutation.variables === warning.id
                          }
                        />
                      ))}
                    </TableBody>
                  </Table>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </>
  );
}
