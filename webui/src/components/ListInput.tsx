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

interface ListInputProps {
  value: string[];
  onChange: (value: string[]) => void;
  placeholder?: string;
}

export function ListInput({ value, onChange, placeholder }: ListInputProps) {
  const handleAdd = () => {
    onChange([...value, ""]);
  };


  const handleRemove = (index: number) => {
    const newList = value.filter((_, i) => i !== index);
    onChange(newList);
  };

  // Always show at least one empty input if the list is empty
  const displayValue = value.length === 0 ? [""] : value;

  return (
    <div className="space-y-2">
      {displayValue.map((item, index) => (
        <div key={index} className="flex gap-2">
          <Input
            value={item}
            onChange={(e) => {
              const newList = [...displayValue];
              newList[index] = e.target.value;
              // Filter out empty strings, but keep at least one empty input
              const filtered = newList.filter((item) => item.trim() !== "");
              onChange(filtered.length === 0 ? [] : filtered);
            }}
            placeholder={placeholder}
            className="flex-1"
          />
          <Button
            type="button"
            variant="ghost"
            size="icon"
            onClick={() => handleRemove(index)}
            disabled={displayValue.length === 1 && displayValue[0] === ""}
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
      ))}
      <Button type="button" variant="outline" onClick={handleAdd} size="sm">
        <Plus className="h-4 w-4 mr-2" />
        Add Item
      </Button>
    </div>
  );
}


