/*
DeepFellow Software Framework.
Copyright © 2025 Simplito sp. z o.o.

This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
This software is Licensed under the DeepFellow Free License.

See the License for the specific language governing permissions and
limitations under the License.
*/
import type { SpecField } from "@/deepfellow/types";

export interface SettingEntry {
  field?: SpecField;
  key: string;
  label: string;
  isSecret: boolean;
  raw: unknown;
  display: string;
}

export const SECRET_MASK = "••••••••";

function labelForOneOf(field: SpecField, value: string): string {
  const match = field.values?.find((v) =>
    typeof v === "string" ? v === value : v.value === value,
  );
  if (!match) return value;
  return typeof match === "string" ? match : match.label;
}

export function formatSettingValue(value: unknown, field?: SpecField): string {
  if (value === null || value === undefined) return "—";

  switch (field?.type) {
    case "oneof":
      return typeof value === "string"
        ? labelForOneOf(field, value)
        : String(value);
    case "bool":
      return value === true || value === "true" ? "Yes" : "No";
    case "list": {
      const list = Array.isArray(value) ? value : [];
      return list.length === 0 ? "—" : list.map((v) => String(v)).join(", ");
    }
    case "map": {
      const map =
        value && typeof value === "object"
          ? (value as Record<string, unknown>)
          : {};
      const pairs = Object.entries(map);
      return pairs.length === 0
        ? "—"
        : pairs.map(([k, v]) => `${k}: ${String(v)}`).join(", ");
    }
    default:
      if (typeof value === "object") return JSON.stringify(value);
      return String(value);
  }
}

// Ordered to match the spec; keys not present in the spec are appended afterwards.
export function buildSettingEntries(
  fields: SpecField[],
  installed: Record<string, unknown>,
): SettingEntry[] {
  const entries: SettingEntry[] = [];
  const seen = new Set<string>();

  for (const field of fields) {
    if (!(field.name in installed)) continue;
    seen.add(field.name);
    const raw = installed[field.name];
    entries.push({
      field,
      key: field.name,
      label: field.description || field.name,
      isSecret: field.type === "password",
      raw,
      display: formatSettingValue(raw, field),
    });
  }

  for (const [key, raw] of Object.entries(installed)) {
    if (seen.has(key)) continue;
    entries.push({
      key,
      label: key,
      isSecret: false,
      raw,
      display: formatSettingValue(raw),
    });
  }

  return entries;
}
