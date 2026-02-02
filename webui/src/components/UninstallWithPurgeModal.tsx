/*
DeepFellow Software Framework.
Copyright © 2025 Simplito sp. z o.o.

This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
This software is Licensed under the DeepFellow Free License.

See the License for the specific language governing permissions and
limitations under the License.
*/
import { useId, useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { cn } from "@/lib/utils";

interface UninstallWithPurgeModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: string;
  description: string;
  confirmText?: string;
  cancelText?: string;
  onConfirm: (purge: boolean) => void;
  isLoading?: boolean;
  variant?: "default" | "destructive" | "warning";
  purgeLabel?: string;
  purgeDescription?: string;
  defaultPurgeChecked?: boolean;
}

export function UninstallWithPurgeModal({
  open,
  onOpenChange,
  title,
  description,
  confirmText = "Uninstall",
  cancelText = "Cancel",
  onConfirm,
  isLoading = false,
  variant = "destructive",
  purgeLabel = "Purge",
  purgeDescription = "Also remove downloaded files and local data. This cannot be undone.",
  defaultPurgeChecked = false,
}: UninstallWithPurgeModalProps) {
  const checkboxId = useId();
  const [purgeChecked, setPurgeChecked] = useState(defaultPurgeChecked);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        onInteractOutside={(e) => {
          if (isLoading) e.preventDefault();
        }}
        onEscapeKeyDown={(e) => {
          if (isLoading) e.preventDefault();
        }}
      >
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
          <DialogDescription>{description}</DialogDescription>
        </DialogHeader>

        <div className="rounded-md border bg-muted/30 p-3">
          <label htmlFor={checkboxId} className="flex items-start gap-3 cursor-pointer">
            <Checkbox
              id={checkboxId}
              checked={purgeChecked}
              disabled={isLoading}
              onCheckedChange={(checked) => setPurgeChecked(checked === true)}
            />
            <div className="grid gap-1">
              <Label htmlFor={checkboxId} className="leading-none cursor-pointer select-none">
                {purgeLabel}
              </Label>
              <p className="text-sm text-muted-foreground">{purgeDescription}</p>
            </div>
          </label>
        </div>

        <DialogFooter>
          <Button
            type="button"
            variant="outline"
            onClick={() => onOpenChange(false)}
            disabled={isLoading}
          >
            {cancelText}
          </Button>
          <Button
            type="button"
            onClick={() => onConfirm(purgeChecked)}
            disabled={isLoading}
            variant={variant === "destructive" ? "destructive" : "default"}
            className={cn(
              variant === "warning" &&
                "bg-yellow-500 hover:bg-yellow-600 text-white focus-visible:ring-yellow-500/20 dark:bg-yellow-600 dark:hover:bg-yellow-700"
            )}
          >
            {isLoading ? "Processing..." : confirmText}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
