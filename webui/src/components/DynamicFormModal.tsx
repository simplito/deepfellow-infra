import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Skeleton } from "@/components/ui/skeleton";
import type { SpecField } from "@/deepfellow/types";
/*
DeepFellow Software Framework.
Copyright © 2025 Simplito sp. z o.o.

This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
This software is Licensed under the DeepFellow Free License.

See the License for the specific language governing permissions and
limitations under the License.
*/
import { useEffect, useMemo, useState } from "react";
import {
  DynamicFormFields,
  initFormData,
  validateFields,
} from "./DynamicFormFields";

interface DynamicFormModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: string;
  fields: SpecField[];
  initialData?: Record<string, unknown>;
  onSubmit: (data: Record<string, unknown>) => void;
  isSubmitting?: boolean;
  isLoading?: boolean;
  deferRender?: boolean;
  submitLabel?: string;
  submittingLabel?: string;
}

export function DynamicFormModal({
  open,
  onOpenChange,
  title,
  fields,
  initialData: initialDataProp,
  onSubmit,
  isSubmitting = false,
  isLoading = false,
  deferRender = false,
  submitLabel = "Install",
  submittingLabel = "Installing...",
}: DynamicFormModalProps) {
  const [renderFields, setRenderFields] = useState<SpecField[]>(
    deferRender ? [] : fields,
  );

  useEffect(() => {
    if (!open) return;
    if (!deferRender) {
      setRenderFields(fields);
      return;
    }
    setRenderFields([]);
    const timeoutId = window.setTimeout(() => setRenderFields(fields), 0);
    return () => window.clearTimeout(timeoutId);
  }, [open, deferRender, fields]);

  const effectiveIsLoading =
    isLoading ||
    (deferRender && open && renderFields.length === 0 && fields.length > 0);

  const initialData = useMemo(() => {
    const defaults = initFormData(renderFields);
    if (!initialDataProp) return defaults;
    const merged = { ...defaults };
    for (const field of renderFields) {
      if (field.name in initialDataProp) {
        merged[field.name] = initialDataProp[field.name];
      }
    }
    return merged;
  }, [renderFields, initialDataProp]);
  const [formData, setFormData] =
    useState<Record<string, unknown>>(initialData);
  const [errors, setErrors] = useState<Record<string, string>>({});

  useEffect(() => {
    if (open) {
      setFormData(initialData);
      setErrors({});
    }
  }, [open, initialData]);

  const focusField = (name: string) => {
    const container = document.querySelector(
      `[data-field-name="${CSS.escape(name)}"]`,
    ) as HTMLElement | null;
    if (!container) return;
    const focusable = container.querySelector<HTMLElement>(
      "input,button,textarea,select,[role='combobox'],[role='button']",
    );
    focusable?.focus();
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (effectiveIsLoading || isSubmitting) return;
    const nextErrors = validateFields(renderFields, formData);
    setErrors(nextErrors);
    const firstInvalid = Object.keys(nextErrors)[0];
    if (firstInvalid) {
      focusField(firstInvalid);
      return;
    }
    onSubmit(formData);
  };

  const handleChange = (name: string, value: unknown) => {
    setFormData((prev) => ({ ...prev, [name]: value }));
    setErrors((prev) => {
      if (!prev[name]) return prev;
      const next = { ...prev };
      delete next[name];
      return next;
    });
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        className="sm:max-w-[600px]"
        onInteractOutside={(e) => {
          if (isSubmitting) e.preventDefault();
        }}
        onEscapeKeyDown={(e) => {
          if (isSubmitting) e.preventDefault();
        }}
      >
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
          <DialogDescription>
            {effectiveIsLoading
              ? "Loading form..."
              : "Fill in the required information below."}
          </DialogDescription>
        </DialogHeader>
        <form onSubmit={handleSubmit}>
          <div className="grid gap-4 px-2 py-4 max-h-[70vh] overflow-auto">
            {effectiveIsLoading ? (
              <div className="grid gap-4">
                <div className="grid gap-2">
                  <Skeleton className="h-4 w-40" />
                  <Skeleton className="h-10 w-full" />
                </div>
                <div className="grid gap-2">
                  <Skeleton className="h-4 w-48" />
                  <Skeleton className="h-10 w-full" />
                </div>
                <div className="grid gap-2">
                  <Skeleton className="h-4 w-36" />
                  <Skeleton className="h-10 w-full" />
                </div>
              </div>
            ) : (
              <DynamicFormFields
                fields={renderFields}
                formData={formData}
                errors={errors}
                onChange={handleChange}
              />
            )}
          </div>
          <DialogFooter>
            <Button
              type="button"
              variant="outline"
              onClick={() => onOpenChange(false)}
              disabled={isSubmitting}
            >
              Cancel
            </Button>
            <Button type="submit" disabled={isSubmitting || effectiveIsLoading}>
              {isSubmitting ? submittingLabel : submitLabel}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
