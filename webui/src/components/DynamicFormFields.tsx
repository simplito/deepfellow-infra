/*
DeepFellow Software Framework.
Copyright © 2025 Simplito sp. z o.o.

This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
This software is Licensed under the DeepFellow Free License.

See the License for the specific language governing permissions and
limitations under the License.
*/
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import type { SpecField } from "@/deepfellow/types";
import { useMemo } from "react";
import { ListInput } from "./ListInput";
import { MapInput } from "./MapInput";

export interface DynamicFormFieldsProps {
  fields: SpecField[];
  formData: Record<string, unknown>;
  errors: Record<string, string>;
  onChange: (name: string, value: unknown) => void;
}

export function DynamicFormFields({
  fields,
  formData,
  errors,
  onChange,
}: DynamicFormFieldsProps) {
  const visibleFields = useMemo(
    () =>
      fields.filter((field) => {
        if (!field.display) return true;
        const [name, value] = field.display.split("=");
        return name && value ? formData[name] === value : true;
      }),
    [fields, formData],
  );

  return (
    <>
      {visibleFields.map((field) => (
        <div
          key={field.name}
          className="grid gap-2"
          data-field-name={field.name}
        >
          <Label htmlFor={field.name}>
            {field.description}
            {field.required && (
              <span className="text-destructive text-sm ml-1">*</span>
            )}
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
                checked={formData[field.name] === true}
                onCheckedChange={(checked) =>
                  onChange(field.name, checked === true)
                }
              />
            </div>
          ) : field.type === "oneof" ? (
            <Select
              value={(formData[field.name] as string | undefined) || (field.default as string | undefined) || "__none__"}
              onValueChange={(value) =>
                onChange(field.name, value === "__none__" ? "" : value)
              }
            >
              <SelectTrigger id={field.name} className="w-full">
                <SelectValue placeholder="Select an option" />
              </SelectTrigger>
              <SelectContent>
                {!field.values?.length && (
                  <SelectItem value="__none__">
                    <span className="text-muted-foreground">None</span>
                  </SelectItem>
                )}
                {field.values
                  ?.filter(
                    (val) => (typeof val === "string" ? val : val.value) !== "",
                  )
                  .map((val) => (
                    <SelectItem
                      key={typeof val === "string" ? val : val.value}
                      value={typeof val === "string" ? val : val.value}
                    >
                      {typeof val === "string" ? val : val.label}
                    </SelectItem>
                  ))}
              </SelectContent>
            </Select>
          ) : field.type === "list" ? (
            <ListInput
              value={(formData[field.name] as string[] | undefined) ?? []}
              onChange={(value) => onChange(field.name, value)}
              placeholder={field.placeholder}
            />
          ) : field.type === "map" ? (
            <MapInput
              value={
                (formData[field.name] as Record<string, string> | undefined) ??
                {}
              }
              onChange={(value) => onChange(field.name, value)}
              placeholder={field.placeholder}
            />
          ) : field.type === "textarea" ? (
            <Textarea
              id={field.name}
              placeholder={field.placeholder || ""}
              required={field.required}
              value={(formData[field.name] as string | undefined) ?? ""}
              onChange={(e) => onChange(field.name, e.target.value)}
              aria-invalid={!!errors[field.name]}
            />
          ) : (
            <Input
              id={field.name}
              type={field.type}
              placeholder={field.placeholder || ""}
              required={field.required}
              value={
                (formData[field.name] as string | number | undefined) ?? ""
              }
              onChange={(e) => {
                const value =
                  field.type === "number"
                    ? e.target.valueAsNumber
                    : e.target.value;
                onChange(field.name, value);
              }}
              aria-invalid={!!errors[field.name]}
            />
          )}
          {errors[field.name] && (
            <div className="text-sm text-destructive">{errors[field.name]}</div>
          )}
        </div>
      ))}
    </>
  );
}

export function initFormData(fields: SpecField[]): Record<string, unknown> {
  const initial: Record<string, unknown> = {};
  for (const field of fields) {
    if (field.type === "bool") {
      initial[field.name] = field.default === true || field.default === "true";
    } else if (field.type === "list") {
      if (typeof field.default === "string" && field.default.startsWith("[")) {
        try {
          initial[field.name] = JSON.parse(field.default) ?? [];
        } catch {
          initial[field.name] = [];
        }
      } else {
        initial[field.name] = Array.isArray(field.default) ? field.default : [];
      }
    } else if (field.type === "map") {
      if (typeof field.default === "string" && field.default.startsWith("{")) {
        try {
          initial[field.name] = JSON.parse(field.default) ?? {};
        } catch {
          initial[field.name] = {};
        }
      } else {
        initial[field.name] = field.default ?? {};
      }
    } else if (field.type === "oneof") {
      const firstVal = field.values?.[0];
      const firstValue = firstVal
        ? typeof firstVal === "string"
          ? firstVal
          : firstVal.value
        : undefined;
      initial[field.name] =
        field.default !== undefined && field.default !== null
          ? field.default
          : (firstValue ?? "");
    } else if (field.default !== undefined && field.default !== null) {
      initial[field.name] = field.default;
    }
  }
  return initial;
}

export function validateFields(
  fields: SpecField[],
  formData: Record<string, unknown>,
): Record<string, string> {
  const errors: Record<string, string> = {};
  for (const field of fields) {
    if (!field.required) continue;
    const value = formData[field.name];
    switch (field.type) {
      case "oneof":
        if (typeof value !== "string" || !value.trim())
          errors[field.name] = "This field is required.";
        break;
      case "list": {
        const list = Array.isArray(value) ? value : [];
        if (list.filter((v) => typeof v === "string" && v.trim()).length === 0)
          errors[field.name] = "Please add at least one item.";
        break;
      }
      case "map": {
        const map =
          value && typeof value === "object"
            ? (value as Record<string, unknown>)
            : {};
        if (Object.keys(map).filter((k) => k.trim()).length === 0)
          errors[field.name] = "Please add at least one pair.";
        break;
      }
      case "number":
        if (typeof value !== "number" || Number.isNaN(value))
          errors[field.name] = "This field is required.";
        break;
      case "bool":
        break;
      default:
        if (
          value === null ||
          value === undefined ||
          String(value).trim() === ""
        )
          errors[field.name] = "This field is required.";
    }
  }
  return errors;
}
