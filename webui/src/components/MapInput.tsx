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
import { useState, useEffect } from "react";

interface MapInputProps {
  value: Record<string, string>;
  onChange: (value: Record<string, string>) => void;
  placeholder?: string;
}

export function MapInput({ value, onChange, placeholder }: MapInputProps) {
  // Convert map to array of entries for easier manipulation
  const entries = Object.entries(value);
  
  // Track entries with unique IDs to handle key changes properly
  // Use the key itself as part of the ID to maintain stability
  const [entryKeys, setEntryKeys] = useState<string[]>(() => 
    entries.length > 0 ? entries.map(([k]) => k) : []
  );

  // Sync entryKeys when value prop changes from outside
  useEffect(() => {
    const currentKeys = Object.keys(value);
    if (currentKeys.length === 0 && entryKeys.length === 0) {
      // Both empty, ensure we have at least one empty entry
      setEntryKeys([""]);
    } else if (currentKeys.length !== entryKeys.length || 
               !currentKeys.every(k => entryKeys.includes(k))) {
      // Value changed from outside, sync the keys
      setEntryKeys(currentKeys.length > 0 ? currentKeys : [""]);
    }
  }, [value]);

  const handleAdd = () => {
    const newKey = `__new_${Date.now()}`;
    setEntryKeys([...entryKeys, newKey]);
    onChange({ ...value, "": "" });
  };

  const handleKeyChange = (oldKey: string, newKey: string, entryIndex: number) => {
    const newMap = { ...value };
    const currentValue = value[oldKey] || "";
    
    // Remove old key
    delete newMap[oldKey];
    
    // Add new key if not empty
    if (newKey.trim()) {
      newMap[newKey.trim()] = currentValue;
    }
    
    // Update entryKeys
    const newEntryKeys = [...entryKeys];
    if (newKey.trim()) {
      newEntryKeys[entryIndex] = newKey.trim();
    } else {
      // Keep empty key for now
      newEntryKeys[entryIndex] = "";
    }
    setEntryKeys(newEntryKeys);
    
    // Filter out entries with empty keys
    const filtered: Record<string, string> = {};
    for (const [k, v] of Object.entries(newMap)) {
      if (k.trim()) {
        filtered[k] = v;
      }
    }
    onChange(filtered);
  };

  const handleValueChange = (key: string, newValue: string) => {
    if (key.trim()) {
      onChange({ ...value, [key]: newValue });
    }
  };

  const handleRemove = (key: string, index: number) => {
    const newMap = { ...value };
    delete newMap[key];
    onChange(newMap);
    
    // Remove the corresponding key from entryKeys
    const newEntryKeys = entryKeys.filter((_, i) => i !== index);
    setEntryKeys(newEntryKeys.length > 0 ? newEntryKeys : [""]);
  };

  // Always show at least one empty pair if the map is empty
  const displayEntries = entryKeys.length === 0 
    ? [["", ""] as [string, string]] 
    : entryKeys.map((k) => [k, value[k] || ""] as [string, string]);

  return (
    <div className="space-y-2">
      {displayEntries.map(([key, val], index) => (
        <div key={`${key}-${index}`} className="flex gap-2">
          <Input
            value={key}
            onChange={(e) => handleKeyChange(key, e.target.value, index)}
            placeholder="Key"
            className="flex-1"
          />
          <Input
            value={val}
            onChange={(e) => handleValueChange(key, e.target.value)}
            placeholder={placeholder || "Value"}
            className="flex-1"
          />
          <Button
            type="button"
            variant="ghost"
            size="icon"
            onClick={() => handleRemove(key, index)}
            disabled={displayEntries.length === 1 && key === "" && val === ""}
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


