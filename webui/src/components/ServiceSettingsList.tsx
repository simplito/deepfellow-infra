/*
DeepFellow Software Framework.
Copyright © 2025 Simplito sp. z o.o.

This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
This software is Licensed under the DeepFellow Free License.

See the License for the specific language governing permissions and
limitations under the License.
*/
import { Button } from "@/components/ui/button";
import type { SpecField } from "@/deepfellow/types";
import {
  SECRET_MASK,
  type SettingEntry,
  buildSettingEntries,
} from "@/utils/service-settings";
import { Eye, EyeOff } from "lucide-react";
import { useState } from "react";

function SecretValue({ value }: { value: string }) {
  const [revealed, setRevealed] = useState(false);
  return (
    <div className="flex items-center gap-2">
      <span className="font-mono break-all">
        {revealed ? value : SECRET_MASK}
      </span>
      <Button
        type="button"
        variant="ghost"
        size="icon"
        className="h-6 w-6 shrink-0 text-muted-foreground"
        aria-label={revealed ? "Hide value" : "Show value"}
        aria-pressed={revealed}
        onClick={() => setRevealed((v) => !v)}
      >
        {revealed ? (
          <EyeOff className="h-4 w-4" />
        ) : (
          <Eye className="h-4 w-4" />
        )}
      </Button>
    </div>
  );
}

function SettingRow({ entry }: { entry: SettingEntry }) {
  return (
    <div className="grid grid-cols-1 gap-1 py-3 sm:grid-cols-[minmax(0,16rem)_1fr] sm:gap-4">
      <div className="text-sm font-medium text-muted-foreground break-words">
        {entry.label}
      </div>
      <div className="text-sm break-words">
        {entry.isSecret ? (
          <SecretValue value={entry.display} />
        ) : (
          <span className="break-all">{entry.display}</span>
        )}
      </div>
    </div>
  );
}

export function ServiceSettingsList({
  fields,
  installed,
}: {
  fields: SpecField[];
  installed: Record<string, unknown>;
}) {
  const entries = buildSettingEntries(fields, installed);
  if (entries.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">
        This service has no configurable settings.
      </p>
    );
  }
  return (
    <div className="divide-y">
      {entries.map((entry) => (
        <SettingRow key={entry.key} entry={entry} />
      ))}
    </div>
  );
}
