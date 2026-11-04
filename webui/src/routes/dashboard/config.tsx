/*
DeepFellow Software Framework.
Copyright © 2025 Simplito sp. z o.o.

This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
This software is Licensed under the DeepFellow Free License.

See the License for the specific language governing permissions and
limitations under the License.
*/
import { MeshModelGroups } from "@/components/MeshModelGroups";
import { SiteHeader } from "@/components/dashboard/site-header";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { Switch } from "@/components/ui/switch";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { apiClient } from "@/deepfellow/client";
import type { ConfigEntry } from "@/deepfellow/types";
import { useRequireAuth } from "@/hooks/use-auth";
import { isBooleanValue, sortConfigEntries } from "@/utils/config-helpers";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { createFileRoute } from "@tanstack/react-router";
import {
  Check,
  Clipboard,
  ClipboardCheck,
  Eye,
  EyeOff,
  Pencil,
  Server,
  X,
} from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";

export const Route = createFileRoute("/dashboard/config")({
  component: ConfigPage,
});

function fmt(gb: number): string {
  return gb.toFixed(1);
}

function StatBar({
  label,
  hint,
  used,
  total,
  unit,
}: {
  label: string;
  hint?: string;
  used: number;
  total: number;
  unit: string;
}) {
  const pct = total > 0 ? Math.round((used / total) * 100) : 0;
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-sm">
        <div className="min-w-0 mr-2">
          <span className="text-muted-foreground">{label}</span>
          {hint && (
            <span className="ml-1.5 text-xs text-muted-foreground/60 truncate max-w-[260px] inline-block align-bottom">
              {hint}
            </span>
          )}
        </div>
        <span className="font-mono tabular-nums text-xs shrink-0 ml-2">
          {fmt(used)} / {fmt(total)} {unit}
        </span>
      </div>
      <Progress value={pct} className="h-1.5" />
    </div>
  );
}

function CpuBar({ percent, model }: { percent: number; model?: string }) {
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-sm">
        <div className="min-w-0 mr-2">
          <span className="text-muted-foreground">CPU</span>
          {model && (
            <span className="ml-1.5 text-xs text-muted-foreground/60 truncate max-w-[260px] inline-block align-bottom">
              {model}
            </span>
          )}
        </div>
        <span className="font-mono tabular-nums text-xs shrink-0">
          {percent.toFixed(1)} %
        </span>
      </div>
      <Progress value={percent} className="h-1.5" />
    </div>
  );
}

function MeshNodeRow({
  node,
}: { node: import("@/deepfellow/types").MeshTopologyNode }) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-center gap-1.5">
        <Server className="size-3 text-muted-foreground shrink-0" />
        <span className="text-sm font-medium truncate">{node.name}</span>
        {node.you_are_here && (
          <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-primary/15 text-primary font-medium shrink-0">
            here
          </span>
        )}
      </div>
      <p className="text-xs text-muted-foreground/70 truncate pl-4.5">
        {node.url}
      </p>
      <p className="text-xs text-muted-foreground/50 truncate pl-4.5">
        {node.url.replace(/^https?/, (p) => (p === "https" ? "wss" : "ws"))}
      </p>
      {node.models.length > 0 && (
        <MeshModelGroups models={node.models} className="pl-4.5" />
      )}
    </div>
  );
}

function InfraPanel() {
  const { data: topology } = useQuery({
    queryKey: ["meshTopology"],
    queryFn: () => apiClient.getMeshTopology(),
    refetchInterval: 5000,
  });

  const { data: gpuStats } = useQuery({
    queryKey: ["gpuStats"],
    queryFn: () => apiClient.getGpuStats(),
    refetchInterval: 5000,
  });

  const { data: systemStats } = useQuery({
    queryKey: ["systemStats"],
    queryFn: () => apiClient.getSystemStats(),
    refetchInterval: 5000,
  });

  const currentNode = topology?.find((n) => n.you_are_here);

  return (
    <div className="flex flex-col gap-4">
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center gap-2">
            <Server className="size-4 text-primary" />
            <CardTitle className="text-base">
              {currentNode?.name ?? "—"}
            </CardTitle>
          </div>
          <CardDescription className="text-xs break-all">
            {currentNode?.url ?? "—"}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          {systemStats ? (
            <>
              <CpuBar
                percent={systemStats.cpu_percent}
                model={systemStats.cpu_model}
              />
              <StatBar
                label="RAM"
                hint={`${fmt(systemStats.ram_total_gb)} GB total`}
                used={systemStats.ram_used_gb}
                total={systemStats.ram_total_gb}
                unit="GB"
              />
            </>
          ) : (
            <div className="space-y-3">
              <Skeleton className="h-8 w-full" />
              <Skeleton className="h-8 w-full" />
            </div>
          )}
          {gpuStats?.gpus?.map((gpu) => (
            <StatBar
              key={gpu.name}
              label="VRAM"
              hint={gpu.name}
              used={gpu.used_vram_gb}
              total={gpu.total_vram_gb}
              unit="GB"
            />
          ))}
        </CardContent>
      </Card>

      {topology && topology.length > 0 && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm text-muted-foreground font-medium uppercase tracking-wider">
              Mesh ({topology.length} {topology.length === 1 ? "node" : "nodes"}
              )
            </CardTitle>
          </CardHeader>
          <CardContent className="pt-0 space-y-4">
            {topology.map((node) => (
              <MeshNodeRow key={node.url} node={node} />
            ))}
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function ConfigPage() {
  useRequireAuth();

  const [revealed, setRevealed] = useState<Record<string, string>>({});
  const [copied, setCopied] = useState<string | null>(null);
  const [editing, setEditing] = useState<Record<string, string>>({});

  const queryClient = useQueryClient();

  const { data, isLoading, error } = useQuery({
    queryKey: ["admin-config"],
    queryFn: () => apiClient.getConfig(),
  });

  const updateMutation = useMutation({
    mutationFn: (updates: Record<string, unknown>) =>
      apiClient.updateDynamicConfig(updates),
  });

  const startEdit = (entry: ConfigEntry) => {
    setEditing((prev) => ({
      ...prev,
      // Never pre-fill a secret's edit field with the masked placeholder.
      [entry.field_name]: entry.is_secret ? "" : entry.value,
    }));
  };

  const cancelEdit = (entry: ConfigEntry) => {
    setEditing((prev) => {
      const next = { ...prev };
      delete next[entry.field_name];
      return next;
    });
  };

  const confirmEdit = (entry: ConfigEntry) => {
    const rawValue = editing[entry.field_name];
    if (entry.is_secret && rawValue === "") {
      // Blank secret edit means "no change" — don't send a request.
      cancelEdit(entry);
      return;
    }

    const value: unknown = isBooleanValue(entry.value)
      ? rawValue === "True"
      : rawValue;

    updateMutation.mutate(
      { [entry.field_name]: value },
      {
        onSuccess: () => {
          queryClient.invalidateQueries({ queryKey: ["admin-config"] });
          toast.success(`Updated ${entry.key}.`);
          cancelEdit(entry);
          // The revealed value is now stale (pre-edit) — stop showing it in place of the mask.
          setRevealed((prev) => {
            if (!(entry.key in prev)) return prev;
            const next = { ...prev };
            delete next[entry.key];
            return next;
          });
        },
        onError: (e) => {
          toast.error(
            `Failed to update ${entry.key}: ${e instanceof Error ? e.message : String(e)}`,
          );
        },
      },
    );
  };

  const fetchSecret = async (key: string): Promise<string> => {
    const result = await apiClient.revealConfigEntry(key);
    return result.value;
  };

  const handleReveal = async (entry: ConfigEntry) => {
    if (entry.key in revealed) {
      setRevealed((prev) => {
        const next = { ...prev };
        delete next[entry.key];
        return next;
      });
      return;
    }
    try {
      const value = await fetchSecret(entry.key);
      setRevealed((prev) => ({ ...prev, [entry.key]: value }));
    } catch (e) {
      toast.error(
        `Failed to reveal: ${e instanceof Error ? e.message : String(e)}`,
      );
    }
  };

  const handleCopy = async (entry: ConfigEntry) => {
    try {
      let value: string;
      if (entry.key in revealed) {
        value = revealed[entry.key];
      } else if (entry.is_secret) {
        value = await fetchSecret(entry.key);
      } else {
        value = entry.value;
      }

      if (!value) {
        toast.info("Value is empty");
        return;
      }

      await navigator.clipboard.writeText(value);
      setCopied(entry.key);
      setTimeout(
        () => setCopied((prev) => (prev === entry.key ? null : prev)),
        2000,
      );
    } catch (e) {
      toast.error(
        `Failed to copy: ${e instanceof Error ? e.message : String(e)}`,
      );
    }
  };

  return (
    <>
      <SiteHeader breadcrumbs={[{ label: "Configuration" }]} />
      <div className="flex flex-1 flex-col">
        <div className="@container/main flex flex-1 flex-col gap-2">
          <div className="py-4 md:py-6 px-4 lg:px-6 max-w-[1400px] w-full mx-auto">
            <div className="grid grid-cols-1 gap-4 lg:grid-cols-[480px_1fr] lg:gap-6 lg:items-start">
              <InfraPanel />

              <Card>
                <CardHeader>
                  <CardTitle>Environment Configuration</CardTitle>
                  <CardDescription>
                    Read-only view of the active server environment variables.
                  </CardDescription>
                </CardHeader>
                <CardContent className="p-0">
                  {isLoading && (
                    <div className="space-y-2 p-6">
                      {Array.from({ length: 10 }).map((_, i) => (
                        // biome-ignore lint/suspicious/noArrayIndexKey: static skeleton list
                        <Skeleton key={i} className="h-10 w-full" />
                      ))}
                    </div>
                  )}
                  {error && (
                    <p className="text-destructive text-sm p-6">
                      Failed to load configuration.
                    </p>
                  )}
                  {data && (
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead className="w-[45%] pl-6">Key</TableHead>
                          <TableHead>Value</TableHead>
                          <TableHead className="w-[90px] pr-6 text-right">
                            Actions
                          </TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {sortConfigEntries(data.entries).map((entry) => (
                          <TableRow key={entry.key}>
                            <TableCell className="pl-6 font-mono text-sm">
                              {entry.key}
                            </TableCell>
                            <TableCell className="font-mono text-sm">
                              {entry.field_name in editing ? (
                                isBooleanValue(entry.value) ? (
                                  <Switch
                                    checked={
                                      editing[entry.field_name] === "True"
                                    }
                                    onCheckedChange={(checked) =>
                                      setEditing((prev) => ({
                                        ...prev,
                                        [entry.field_name]: checked
                                          ? "True"
                                          : "False",
                                      }))
                                    }
                                  />
                                ) : (
                                  <Input
                                    autoFocus
                                    className="h-8 max-w-xs"
                                    type={entry.is_secret ? "password" : "text"}
                                    // Stop browsers from suggesting/autofilling a saved login
                                    // password into this field just because type="password".
                                    autoComplete={
                                      entry.is_secret ? "new-password" : "off"
                                    }
                                    placeholder={
                                      entry.is_secret
                                        ? "Enter new value"
                                        : undefined
                                    }
                                    value={editing[entry.field_name]}
                                    onChange={(e) =>
                                      setEditing((prev) => ({
                                        ...prev,
                                        [entry.field_name]: e.target.value,
                                      }))
                                    }
                                    onKeyDown={(e) => {
                                      if (e.key === "Enter") confirmEdit(entry);
                                      if (e.key === "Escape") cancelEdit(entry);
                                    }}
                                  />
                                )
                              ) : entry.is_secret ? (
                                entry.key in revealed ? (
                                  revealed[entry.key]
                                ) : (
                                  <span className="text-muted-foreground tracking-widest">
                                    ••••••••
                                  </span>
                                )
                              ) : entry.value ? (
                                entry.value
                              ) : (
                                <span className="text-muted-foreground italic">
                                  -
                                </span>
                              )}
                            </TableCell>
                            <TableCell className="pr-6 text-right">
                              <div className="flex justify-end gap-1">
                                {entry.field_name in editing ? (
                                  <>
                                    <Button
                                      variant="ghost"
                                      size="icon"
                                      className="size-7"
                                      onClick={() => confirmEdit(entry)}
                                      disabled={updateMutation.isPending}
                                      title="Save"
                                    >
                                      <Check className="size-4" />
                                    </Button>
                                    <Button
                                      variant="ghost"
                                      size="icon"
                                      className="size-7"
                                      onClick={() => cancelEdit(entry)}
                                      disabled={updateMutation.isPending}
                                      title="Cancel"
                                    >
                                      <X className="size-4" />
                                    </Button>
                                  </>
                                ) : (
                                  <>
                                    {entry.is_editable && (
                                      <Button
                                        variant="ghost"
                                        size="icon"
                                        className="size-7"
                                        onClick={() => startEdit(entry)}
                                        title="Edit value"
                                      >
                                        <Pencil className="size-4" />
                                      </Button>
                                    )}
                                    {entry.is_secret && (
                                      <Button
                                        variant="ghost"
                                        size="icon"
                                        className="size-7"
                                        onClick={() => handleReveal(entry)}
                                        title={
                                          entry.key in revealed
                                            ? "Hide value"
                                            : "Reveal value"
                                        }
                                      >
                                        {entry.key in revealed ? (
                                          <EyeOff className="size-4" />
                                        ) : (
                                          <Eye className="size-4" />
                                        )}
                                      </Button>
                                    )}
                                    <Button
                                      variant="ghost"
                                      size="icon"
                                      className="size-7"
                                      onClick={() => handleCopy(entry)}
                                      title="Copy to clipboard"
                                    >
                                      {copied === entry.key ? (
                                        <ClipboardCheck className="size-4" />
                                      ) : (
                                        <Clipboard className="size-4" />
                                      )}
                                    </Button>
                                  </>
                                )}
                              </div>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  )}
                </CardContent>
              </Card>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
