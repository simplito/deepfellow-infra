/*
DeepFellow Software Framework.
Copyright © 2025 Simplito sp. z o.o.

This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
This software is Licensed under the DeepFellow Free License.

See the License for the specific language governing permissions and
limitations under the License.
*/
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import type { Service } from "@/deepfellow/types";
import { ServiceSettingsList } from "./ServiceSettingsList";

interface ServiceSettingsModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  service: Service;
  onEdit?: (service: Service) => void;
}

export function ServiceSettingsModal({
  open,
  onOpenChange,
  service,
  onEdit,
}: ServiceSettingsModalProps) {
  const installed = service.installed;
  const installedOptions =
    !!installed &&
    typeof installed === "object" &&
    !("stage" in (installed as Record<string, unknown>))
      ? (installed as Record<string, unknown>)
      : null;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[600px]">
        <DialogHeader>
          <DialogTitle>Settings — {service.id}</DialogTitle>
          <DialogDescription>
            The configuration this service is running with.
          </DialogDescription>
        </DialogHeader>
        <div className="max-h-[70vh] overflow-auto px-1">
          {installedOptions ? (
            <ServiceSettingsList
              fields={service.spec.fields}
              installed={installedOptions}
            />
          ) : (
            <p className="text-sm text-muted-foreground">
              This service is not installed yet.
            </p>
          )}
        </div>
        {installedOptions && onEdit && (
          <DialogFooter className="flex-col gap-2 sm:flex-row sm:items-center sm:justify-end">
            <p className="text-xs text-muted-foreground sm:mr-auto">
              Saving changes restarts the service.
            </p>
            <Button onClick={() => onEdit(service)}>Edit</Button>
          </DialogFooter>
        )}
      </DialogContent>
    </Dialog>
  );
}
