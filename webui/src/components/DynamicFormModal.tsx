/*
DeepFellow Software Framework.
Copyright © 2025 Simplito sp. z o.o.

This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
This software is Licensed under the DeepFellow Free License.

See the License for the specific language governing permissions and
limitations under the License.
*/
import { useState, useMemo, useEffect } from "react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { ListInput } from "./ListInput";
import { MapInput } from "./MapInput";
import type { SpecField } from "@/deepfellow/types";

interface DynamicFormModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: string;
  fields: SpecField[];
  onSubmit: (data: Record<string, any>) => void;
  isSubmitting?: boolean;
}

export function DynamicFormModal({
  open,
  onOpenChange,
  title,
  fields,
  onSubmit,
  isSubmitting = false,
}: DynamicFormModalProps) {
  const initializeFormData = useMemo(() => {
    const initial: Record<string, any> = {};
    for (const field of fields) {
      if (field.type === "bool") {
        initial[field.name] = field.default === true ? true : false;
      } else if (field.type === "list") {
        initial[field.name] = [];
      } else if (field.type === "map") {
        initial[field.name] = {};
      } else if (field.default !== undefined && field.default !== null) {
        initial[field.name] = field.default;
      }
    }
    return initial;
  }, [fields]);

  const [formData, setFormData] = useState<Record<string, any>>(initializeFormData);

  // Reset form data when modal opens or fields change
  useEffect(() => {
    if (open) {
      setFormData(initializeFormData);
    }
  }, [open, initializeFormData]);

  // Conditional field visibility logic
  const isFieldVisible = (field: SpecField): boolean => {
    if (!field.display) return true;
    const [name, value] = field.display.split("=");
    if (name && value) {
      return formData[name] === value;
    }
    return true;
  };

  // Filter fields based on visibility
  const visibleFields = useMemo(() => {
    return fields.filter(isFieldVisible);
  }, [fields, formData]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(formData);
  };

  const handleInputChange = (name: string, value: any) => {
    setFormData((prev) => ({ ...prev, [name]: value }));
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
            Fill in the required information below.
          </DialogDescription>
        </DialogHeader>
        <form onSubmit={handleSubmit}>
          <div className="grid gap-4 py-4">
            {visibleFields.map((field) => (
              <div key={field.name} className="grid gap-2">
                <Label htmlFor={field.name}>
                  {field.description}
                  {!field.required && (
                    <span className="text-muted-foreground text-sm ml-1">
                      (optional)
                    </span>
                  )}
                </Label>
                {field.type === "bool" ? (
                  <div className="flex items-center space-x-2">
                    <Checkbox
                      id={field.name}
                      checked={formData[field.name] || false}
                      onCheckedChange={(checked) =>
                        handleInputChange(field.name, checked === true)
                      }
                    />
                  </div>
                ) : field.type === "oneof" ? (
                  <Select
                    value={formData[field.name] || ""}
                    onValueChange={(value) => handleInputChange(field.name, value)}
                    required={field.required}
                  >
                    <SelectTrigger id={field.name} className="w-full">
                      <SelectValue placeholder="Select an option" />
                    </SelectTrigger>
                    <SelectContent>
                      {field.values?.map((val) => (
                        <SelectItem key={val} value={val}>
                          {val}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                ) : field.type === "list" ? (
                  <ListInput
                    value={formData[field.name] || []}
                    onChange={(value) => handleInputChange(field.name, value)}
                    placeholder={field.placeholder}
                  />
                ) : field.type === "map" ? (
                  <MapInput
                    value={formData[field.name] || {}}
                    onChange={(value) => handleInputChange(field.name, value)}
                    placeholder={field.placeholder}
                  />
                ) : field.type === "textarea" ? (
                  <Textarea
                    id={field.name}
                    placeholder={field.placeholder || ""}
                    required={field.required}
                    value={formData[field.name] || ""}
                    onChange={(e) => handleInputChange(field.name, e.target.value)}
                  />
                ) : (
                  <Input
                    id={field.name}
                    type={field.type}
                    placeholder={field.placeholder || ""}
                    required={field.required}
                    value={formData[field.name] || undefined}
                    onChange={(e) => {
                      const value =
                        field.type === "number"
                          ? e.target.valueAsNumber
                          : e.target.value;
                      handleInputChange(field.name, value);
                    }}
                  />
                )}
              </div>
            ))}
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
            <Button type="submit" disabled={isSubmitting}>
              {isSubmitting ? "Installing..." : "Install"}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
