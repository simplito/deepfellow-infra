/*
DeepFellow Software Framework.
Copyright © 2025 Simplito sp. z o.o.

This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
This software is Licensed under the DeepFellow Free License.

See the License for the specific language governing permissions and
limitations under the License.
*/
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Plus, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";

interface MapInputProps {
  value: Record<string, string>;
  onChange: (value: Record<string, string>) => void;
  placeholder?: string;
}

export function MapInput({ value, onChange, placeholder }: MapInputProps) {
  type Row = { id: string; keyText: string; valueText: string };

  const [rows, setRows] = useState<Row[]>(() => mapToRows<Row>(value));

  const lastEmittedRef = useRef<string>(serializeMap(value));

  // If parent value changes externally (e.g. modal reset), resync local rows.
  useEffect(() => {
    const incoming = serializeMap(value);
    if (incoming === lastEmittedRef.current) return;

    setRows(mapToRows<Row>(value));
  }, [value]);

  const emitChange = (nextRows: Row[]) => {
    const nextMap: Record<string, string> = {};
    for (const row of nextRows) {
      const key = row.keyText.trim();
      if (!key) continue;
      nextMap[key] = row.valueText;
    }
    lastEmittedRef.current = serializeMap(nextMap);
    onChange(nextMap);
  };

  const handleAdd = () => {
    setRows((prev) => {
      const next = [...prev, { id: newRowId(), keyText: "", valueText: "" }];
      // Do not emit yet; empty key should not appear in the resulting map.
      return next;
    });
  };

  const handleKeyChange = (rowId: string, newKey: string) => {
    setRows((prev) => {
      const next = prev.map((r) => (r.id === rowId ? { ...r, keyText: newKey } : r));
      emitChange(next);
      return next;
    });
  };

  const handleValueChange = (rowId: string, newValue: string) => {
    setRows((prev) => {
      const next = prev.map((r) => (r.id === rowId ? { ...r, valueText: newValue } : r));
      emitChange(next);
      return next;
    });
  };

  const handleRemove = (rowId: string) => {
    setRows((prev) => {
      const filtered = prev.filter((r) => r.id !== rowId);
      const next = filtered.length ? filtered : [{ id: newRowId(), keyText: "", valueText: "" }];
      emitChange(next);
      return next;
    });
  };

  return (
    <div className="space-y-2">
      {rows.map((row) => (
        <div key={row.id} className="flex gap-2">
          <Input
            value={row.keyText}
            onChange={(e) => handleKeyChange(row.id, e.target.value)}
            placeholder="Key"
            className="flex-1"
          />
          <Input
            value={row.valueText}
            onChange={(e) => handleValueChange(row.id, e.target.value)}
            placeholder={placeholder || "Value"}
            className="flex-1"
          />
          <Button
            type="button"
            variant="ghost"
            size="icon"
            onClick={() => handleRemove(row.id)}
            disabled={rows.length === 1 && row.keyText.trim() === "" && row.valueText.trim() === ""}
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
      ))}
      <Button type="button" variant="outline" onClick={handleAdd} size="sm">
        <Plus className="h-4 w-4 mr-2" />
        Add Pair
      </Button>
    </div>
  );
}

function newRowId(): string {
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function serializeMap(map: Record<string, string>): string {
  const sorted = Object.entries(map)
    .map(([k, v]) => [k, v] as const)
    .sort(([a], [b]) => a.localeCompare(b));
  return JSON.stringify(sorted);
}

function mapToRows<RowType extends { id: string; keyText: string; valueText: string }>(
  value: Record<string, string>
): RowType[] {
  const entries = Object.entries(value);
  if (entries.length === 0) {
    return [{ id: newRowId(), keyText: "", valueText: "" } as RowType];
  }
  return entries.map(([k, v]) => ({ id: newRowId(), keyText: k, valueText: v } as RowType));
}
