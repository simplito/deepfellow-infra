/*
DeepFellow Software Framework.
Copyright © 2026 Simplito sp. z o.o.

This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
This software is Licensed under the DeepFellow Free License.

See the License for the specific language governing permissions and
limitations under the License.
*/
import type { MeshTopologyModel } from "@/deepfellow/types";
import { ChevronDown, ChevronRight } from "lucide-react";
import { useState } from "react";

const TYPE_ORDER = ["llm", "embedding", "tts", "stt", "txt2img", "mcp"];

const TYPE_LABELS: Record<string, string> = {
  llm: "LLM",
  embedding: "Embedding",
  tts: "TTS",
  stt: "STT",
  txt2img: "Image",
  mcp: "MCP Servers",
};

function normalizeType(type: string): string {
  if (type.startsWith("llm")) return "llm";
  if (type === "image") return "txt2img";
  return type;
}

function groupModels(models: MeshTopologyModel[]): [string, MeshTopologyModel[]][] {
  const groups = new Map<string, MeshTopologyModel[]>();
  for (const m of models) {
    const key = normalizeType(m.type);
    const existing = groups.get(key);
    if (existing) {
      existing.push(m);
    } else {
      groups.set(key, [m]);
    }
  }
  const ordered: [string, MeshTopologyModel[]][] = [];
  for (const key of TYPE_ORDER) {
    const group = groups.get(key);
    if (group) {
      ordered.push([key, group]);
      groups.delete(key);
    }
  }
  for (const [key, group] of groups) {
    ordered.push([key, group]);
  }
  return ordered;
}

export function MeshModelGroups({
  models,
  className,
}: {
  models: MeshTopologyModel[];
  className?: string;
}) {
  const groups = groupModels(models);
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());

  const toggle = (key: string) => {
    setCollapsed((prev) => {
      const next = new Set(prev);
      if (next.has(key)) {
        next.delete(key);
      } else {
        next.add(key);
      }
      return next;
    });
  };

  if (groups.length === 0) return null;

  return (
    <div className={className}>
      {groups.map(([key, items]) => {
        const isCollapsed = collapsed.has(key);
        const label = TYPE_LABELS[key] ?? (key.charAt(0).toUpperCase() + key.slice(1));
        return (
          <div key={key}>
            <button
              type="button"
              onClick={() => toggle(key)}
              className="flex items-center gap-1 w-full text-left py-0.5 text-xs text-muted-foreground hover:text-foreground transition-colors"
            >
              {isCollapsed ? (
                <ChevronRight className="size-3 shrink-0" />
              ) : (
                <ChevronDown className="size-3 shrink-0" />
              )}
              <span className="font-medium">{label}</span>
              <span className="ml-1 opacity-60">{items.length}</span>
            </button>
            {!isCollapsed && (
              <div className="space-y-0.5 mb-1.5">
                {items.map((m) => (
                  <div
                    key={m.name}
                    className="flex items-center text-xs px-2 py-1 rounded bg-muted/50 ml-3.5"
                  >
                    <span className="truncate font-mono">{m.name}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
