/*
DeepFellow Software Framework.
Copyright © 2026 Simplito sp. z o.o.

This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
This software is Licensed under the DeepFellow Free License.

See the License for the specific language governing permissions and
limitations under the License.
*/

import { ListInput } from "@/components/ListInput";
import { MapInput } from "@/components/MapInput";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
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
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import type { SpecField } from "@/deepfellow/types";
import {
  type McpVariant,
  type NodeVersion,
  type ProxyMcpServerSpec,
  type PythonVersion,
  detectRuntime,
  useDockerForm,
  useStdioForm,
  useUrlForm,
} from "@/hooks/use-mcp-server-form";
import { useState } from "react";
import { DynamicFormFields } from "./DynamicFormFields";

export type {
  McpVariant,
  NodeVersion,
  PythonVersion,
} from "@/hooks/use-mcp-server-form";
export type { ProxyMcpServerSpec } from "@/hooks/use-mcp-server-form";

const MCP_VARIANTS = [
  { value: "node-headless", label: "Node.js — headless (default)" },
  { value: "node-headed", label: "Node.js — headed (+ Chromium)" },
  { value: "python-headless", label: "Python — headless (default)" },
  { value: "python-headed", label: "Python — headed (+ Chromium)" },
] as const;

const PYTHON_VERSIONS: { value: PythonVersion; label: string }[] = [
  { value: "3.10", label: "Python 3.10" },
  { value: "3.11", label: "Python 3.11" },
  { value: "3.12", label: "Python 3.12" },
  { value: "3.13", label: "Python 3.13 (default)" },
  { value: "3.14", label: "Python 3.14" },
  { value: "latest", label: "Latest" },
];

const NODE_VERSIONS: { value: NodeVersion; label: string }[] = [
  { value: "20", label: "Node 20" },
  { value: "22", label: "Node 22 LTS (default)" },
  { value: "24", label: "Node 24" },
  { value: "latest", label: "Latest" },
];

/** How the server is configured when adding */
type McpServerMode = "auto-import" | "command" | "url" | "docker";

export interface AddMcpServerSpec {
  kind: "user";
  id: string;
  name: string;
  variant: McpVariant;
  command: string;
  base_image?: string;
  python_version?: PythonVersion;
  node_version?: NodeVersion;
  envs?: Record<string, string>;
  default_prefix?: string;
  repository_url?: string;
  description?: string;
}

export type AddMcpServerPayload =
  | AddMcpServerSpec
  | { kind: "docker"; data: Record<string, unknown>; repository_url?: string; description?: string }
  | {
      kind: "proxy";
      id: string;
      name: string;
      server_url: string;
      transport: "streamable_http" | "sse";
      default_prefix?: string;
      headers?: Record<string, string>;
      repository_url?: string;
      description?: string;
    };

interface AddMcpServerModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSubmit: (payload: AddMcpServerPayload) => void;
  isSubmitting: boolean;
  dockerFields: SpecField[];
  initialValues?: Partial<AddMcpServerSpec> | Partial<ProxyMcpServerSpec>;
  title?: string;
  notice?: string;
  apiError?: string | null;
}

function shquote(arg: string): string {
  if (/^[a-zA-Z0-9_./:@,=+\-]+$/.test(arg)) return arg;
  return `'${arg.replace(/'/g, "'\\''")}'`;
}

// docker run flags that consume a following value argument
const DOCKER_VALUE_FLAGS = new Set([
  "-v",
  "--volume",
  "-e",
  "--env",
  "--env-file",
  "--name",
  "-p",
  "--publish",
  "--expose",
  "--network",
  "--hostname",
  "-h",
  "-u",
  "--user",
  "-w",
  "--workdir",
  "--entrypoint",
  "-l",
  "--label",
  "--add-host",
  "--dns",
  "--link",
  "-m",
  "--memory",
  "--memory-swap",
  "--cpus",
  "--runtime",
  "--platform",
  "--ulimit",
  "--security-opt",
  "--cap-add",
  "--cap-drop",
  "--device",
  "--cidfile",
  "--volumes-from",
  "--log-driver",
  "--log-opt",
  "--health-cmd",
  "--health-interval",
  "--health-retries",
  "--health-timeout",
  "--mount",
  "--tmpfs",
  "--shm-size",
  "--ipc",
]);

function parseMountFlag(val: string): string | null {
  const parts: Record<string, string> = {};
  for (const segment of val.split(",")) {
    const eq = segment.indexOf("=");
    if (eq === -1) continue;
    parts[segment.slice(0, eq)] = segment.slice(eq + 1);
  }
  if (parts.type && parts.type !== "bind") return null;
  const src = parts.src ?? parts.source ?? "";
  const dst = parts.dst ?? parts.destination ?? parts.target ?? "";
  if (!src || !dst) return null;
  return parts.ro === "true" || parts.readonly === "true"
    ? `${src}:${dst}:ro`
    : `${src}:${dst}`;
}

function parseDockerRunArgs(args: string[]): {
  image: string;
  cmd: string[];
  volumes: string[];
} {
  const volumes: string[] = [];
  let i = 0;
  while (i < args.length) {
    const arg = args[i];
    if (arg === "run") {
      i++;
      continue;
    }
    if (arg.startsWith("-")) {
      const eqIdx = arg.indexOf("=");
      const flag = eqIdx === -1 ? arg : arg.slice(0, eqIdx);
      const inlineVal = eqIdx === -1 ? null : arg.slice(eqIdx + 1);
      if (flag === "-v" || flag === "--volume") {
        const val = inlineVal ?? args[i + 1];
        if (val) volumes.push(val);
        i += inlineVal !== null ? 1 : 2;
      } else if (flag === "--mount") {
        const val = inlineVal ?? args[i + 1];
        if (val) {
          const mount = parseMountFlag(val);
          if (mount) volumes.push(mount);
        }
        i += inlineVal !== null ? 1 : 2;
      } else {
        i += DOCKER_VALUE_FLAGS.has(flag) && inlineVal === null ? 2 : 1;
      }
      continue;
    }
    return { image: arg, cmd: args.slice(i + 1), volumes };
  }
  throw new Error("Could not find image name in docker run arguments.");
}

type ParsedMcpConfig =
  | {
      kind: "stdio";
      name: string;
      command: string;
      base_image?: string;
      envs: Record<string, string>;
      variant: McpVariant | null;
    }
  | {
      kind: "proxy";
      name: string;
      server_url: string;
      transport: "streamable_http" | "sse";
      headers: Record<string, string>;
    }
  | {
      kind: "docker";
      name: string;
      image: string;
      command: string;
      volumes: string[];
      envs: Record<string, string>;
    };

export function parseMcpJsonConfig(text: string): ParsedMcpConfig {
  const json = JSON.parse(text) as unknown;
  if (!json || typeof json !== "object" || Array.isArray(json))
    throw new Error("Expected a JSON object.");

  const root = json as Record<string, unknown>;
  let entries: [string, unknown][];
  if (root.mcpServers && typeof root.mcpServers === "object") {
    entries = Object.entries(root.mcpServers as Record<string, unknown>);
  } else {
    entries = Object.entries(root);
  }
  if (entries.length === 0) throw new Error("No servers found in config.");

  const [name, config] = entries[0];
  if (!config || typeof config !== "object" || Array.isArray(config))
    throw new Error("Invalid server config.");

  const {
    command: rawCommand,
    args,
    env,
    serverUrl,
    url,
    transport: rawTransport,
    headers: rawHeaders,
  } = config as Record<string, unknown>;

  if (serverUrl || url) {
    const resolvedUrl =
      typeof serverUrl === "string"
        ? serverUrl
        : typeof url === "string"
          ? url
          : "";
    if (!resolvedUrl)
      throw new Error('"serverUrl" must be a non-empty string.');
    const headers: Record<string, string> = {};
    if (
      rawHeaders &&
      typeof rawHeaders === "object" &&
      !Array.isArray(rawHeaders)
    ) {
      for (const [k, v] of Object.entries(
        rawHeaders as Record<string, unknown>,
      )) {
        if (typeof v === "string") headers[k] = v;
      }
    }
    const transport: "streamable_http" | "sse" =
      rawTransport === "sse" || resolvedUrl.endsWith("/sse")
        ? "sse"
        : "streamable_http";
    return { kind: "proxy", name, server_url: resolvedUrl, transport, headers };
  }

  if (typeof rawCommand !== "string" || !rawCommand)
    throw new Error('Missing "command" field.');

  let cmd = rawCommand;
  let argList: string[] = Array.isArray(args) ? args.map(String) : [];

  if (cmd.toLowerCase() === "cmd" && argList[0] === "/c") {
    argList = argList.slice(1);
    cmd = argList.shift() ?? "";
    if (!cmd) throw new Error('Empty command after stripping "cmd /c".');
  }

  const envs: Record<string, string> = {};
  if (env && typeof env === "object" && !Array.isArray(env)) {
    for (const [k, v] of Object.entries(env as Record<string, unknown>)) {
      if (typeof v === "string") envs[k] = v;
    }
  }

  if (cmd === "docker" && argList[0] === "run") {
    const dockerParsed = parseDockerRunArgs(argList);
    return {
      kind: "docker",
      name,
      image: dockerParsed.image,
      command: dockerParsed.cmd.map(shquote).join(" "),
      volumes: dockerParsed.volumes,
      envs,
    };
  }

  const runtime = detectRuntime(cmd);
  return {
    kind: "stdio",
    name,
    command: [cmd, ...argList].map(shquote).join(" "),
    envs,
    variant:
      runtime === "python"
        ? "python-headless"
        : runtime === "node"
          ? "node-headless"
          : null,
  };
}

const JSON_CONFIG_PLACEHOLDER = `{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/data"],
      "env": {}
    }
  }
}`;

function focusField(name: string) {
  const el = document.querySelector(
    `[data-field-name="${CSS.escape(name)}"]`,
  ) as HTMLElement | null;
  el?.querySelector<HTMLElement>(
    "input,button,textarea,select,[role='combobox']",
  )?.focus();
}

export function AddMcpServerModal({
  open,
  onOpenChange,
  onSubmit,
  isSubmitting,
  dockerFields,
  initialValues,
  title = "Add MCP Server",
  notice,
  apiError,
}: AddMcpServerModalProps) {
  const isEditMode = !!initialValues;
  const editKind = initialValues?.kind ?? "user";
  const [serverMode, setServerMode] = useState<McpServerMode>("auto-import");
  const [autoImportWarningOpen, setAutoImportWarningOpen] = useState(false);

  const proxyInitial =
    editKind === "proxy"
      ? (initialValues as Partial<ProxyMcpServerSpec>)
      : undefined;
  const stdioInitial =
    editKind !== "proxy"
      ? (initialValues as Partial<AddMcpServerSpec>)
      : undefined;

  const docker = useDockerForm(dockerFields, open);
  const url = useUrlForm(open, proxyInitial);
  const stdio = useStdioForm(
    stdioInitial,
    open,
    (parsed) => {
      setServerMode("url");
      url.populate(parsed);
    },
    (parsed) => {
      setServerMode("docker");
      docker.populate(parsed);
    },
    parseMcpJsonConfig,
    () => setServerMode("command"),
  );

  const isStdioMode = isEditMode
    ? editKind !== "proxy"
    : serverMode === "command";
  const isUrlMode = isEditMode ? editKind === "proxy" : serverMode === "url";
  const isAutoImportMode = !isEditMode && serverMode === "auto-import";

  const currentTabHasData = () => {
    if (serverMode === "command") return !!stdio.command.trim();
    if (serverMode === "url") return !!url.serverUrl.trim();
    if (serverMode === "docker")
      return Object.values(docker.data).some(
        (v) => typeof v === "string" && v.trim(),
      );
    return false;
  };

  const handleAutoImportTabClick = () => {
    if (currentTabHasData()) {
      setAutoImportWarningOpen(true);
    } else {
      setServerMode("auto-import");
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    if (isAutoImportMode) {
      stdio.convertJson();
      return;
    }

    if (serverMode === "docker") {
      const errs = docker.validate();
      if (Object.keys(errs).length > 0) {
        focusField(Object.keys(errs)[0]);
        return;
      }
      const cleaned = Object.fromEntries(
        Object.entries(docker.data).filter(
          ([, v]) => v !== null && v !== undefined,
        ),
      );
      const volumes = docker.volumes.filter((v) => v.trim());
      onSubmit({
        kind: "docker",
        data: { ...cleaned, ...(volumes.length > 0 ? { volumes } : {}) },
        repository_url: docker.repositoryUrl.trim() || undefined,
        description: docker.description.trim() || undefined,
      });
      return;
    }

    if (isUrlMode) {
      const errs = url.validate();
      if (Object.values(errs).some(Boolean)) return;
      onSubmit({
        kind: "proxy",
        id: url.name.trim(),
        name: url.name.trim(),
        server_url: url.serverUrl.trim(),
        transport: url.transport,
        default_prefix: url.prefix.trim() || undefined,
        headers: Object.keys(url.headers).length > 0 ? url.headers : undefined,
        repository_url: url.repositoryUrl.trim() || undefined,
        description: url.description.trim() || undefined,
      });
      return;
    }

    const payload = stdio.buildPayload();
    if (!payload) return;
    onSubmit(payload);
  };

  const detectedRuntime = detectRuntime(stdio.command);
  const filteredVariants = MCP_VARIANTS.filter(
    (v) => detectedRuntime === null || v.value.startsWith(detectedRuntime),
  );
  const variantLabel =
    detectedRuntime !== null ? "Variant" : "Runtime & variant";
  const variantHint =
    !stdio.variantIsManual && stdio.variant
      ? "Auto-detected from command"
      : null;

  return (
    <>
      <Dialog
        open={open}
        onOpenChange={isSubmitting ? undefined : onOpenChange}
      >
        <DialogContent
          className="max-w-lg"
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
              {isEditMode
                ? "Edit stdio MCP server settings. Reinstall required after saving."
                : "Add a new MCP server — choose the connection type below."}
            </DialogDescription>
          </DialogHeader>

          <form
            id="add-mcp-server-form"
            onSubmit={handleSubmit}
            className="space-y-4 max-h-[70vh] overflow-y-auto pr-1"
          >
            {/* Mode toggle — hidden in edit mode */}
            {!isEditMode && (
              <div className="flex rounded-md border overflow-hidden text-sm">
                <button
                  type="button"
                  className={`flex-1 px-3 py-1.5 transition-colors ${serverMode === "auto-import" ? "bg-primary text-primary-foreground" : "bg-background text-muted-foreground hover:bg-muted"}`}
                  onClick={handleAutoImportTabClick}
                  disabled={isSubmitting}
                >
                  Auto-Import
                </button>
                <button
                  type="button"
                  className={`flex-1 px-3 py-1.5 transition-colors border-l ${serverMode === "command" ? "bg-primary text-primary-foreground" : "bg-background text-muted-foreground hover:bg-muted"}`}
                  onClick={() => setServerMode("command")}
                  disabled={isSubmitting}
                >
                  Command
                </button>
                <button
                  type="button"
                  className={`flex-1 px-3 py-1.5 transition-colors border-l ${serverMode === "docker" ? "bg-primary text-primary-foreground" : "bg-background text-muted-foreground hover:bg-muted"}`}
                  onClick={() => setServerMode("docker")}
                  disabled={isSubmitting}
                >
                  Docker
                </button>
                <button
                  type="button"
                  className={`flex-1 px-3 py-1.5 transition-colors border-l ${serverMode === "url" ? "bg-primary text-primary-foreground" : "bg-background text-muted-foreground hover:bg-muted"}`}
                  onClick={() => setServerMode("url")}
                  disabled={isSubmitting}
                >
                  Remote URL
                </button>
              </div>
            )}

            {/* ── Custom Image fields ──────────────────────────────────────── */}
            {serverMode === "docker" && !isEditMode && (
              <>
                <DynamicFormFields
                  fields={dockerFields}
                  formData={docker.data}
                  errors={docker.errors}
                  onChange={docker.handleChange}
                />
                <div className="space-y-1">
                  <Label>
                    Bind Mounts{" "}
                    <span className="text-muted-foreground text-sm font-normal">
                      (optional)
                    </span>
                  </Label>
                  <ListInput
                    value={docker.volumes}
                    onChange={docker.setVolumes}
                    placeholder="/host/path:/container/path"
                  />
                </div>
                <div className="space-y-1">
                  <Label htmlFor="docker-repository-url">
                    Repository URL{" "}
                    <span className="text-muted-foreground text-sm font-normal">
                      (optional)
                    </span>
                  </Label>
                  <Input
                    id="docker-repository-url"
                    placeholder="https://github.com/owner/repo"
                    value={docker.repositoryUrl}
                    onChange={(e) => docker.setRepositoryUrl(e.target.value)}
                    disabled={isSubmitting}
                  />
                </div>
                <div className="space-y-1">
                  <Label htmlFor="docker-description">
                    Description{" "}
                    <span className="text-muted-foreground text-sm font-normal">
                      (optional)
                    </span>
                  </Label>
                  <Input
                    id="docker-description"
                    placeholder="Short description of what this server does"
                    value={docker.description}
                    onChange={(e) => docker.setDescription(e.target.value)}
                    disabled={isSubmitting}
                  />
                </div>
              </>
            )}

            {/* ── Remote URL (proxy) fields ─────────────────────────────────── */}
            {isUrlMode && (
              <>
                <div className="space-y-1">
                  <Label htmlFor="url-server-url">Server URL</Label>
                  <Input
                    id="url-server-url"
                    placeholder="https://example.com/mcp"
                    value={url.serverUrl}
                    onChange={(e) => {
                      const v = e.target.value;
                      url.setServerUrl(v);
                      try {
                        const path = new URL(v.trim()).pathname;
                        url.setTransport(
                          path.endsWith("/sse") ? "sse" : "streamable_http",
                        );
                      } catch {
                        url.setTransport("streamable_http");
                      }
                    }}
                    disabled={isSubmitting}
                  />
                  {url.errors.server_url && (
                    <p className="text-sm text-destructive">
                      {url.errors.server_url}
                    </p>
                  )}
                </div>
                <div className="space-y-1">
                  <Label htmlFor="url-name">Name</Label>
                  <Input
                    id="url-name"
                    placeholder="my-remote-mcp"
                    value={url.name}
                    onChange={(e) => url.handleNameChange(e.target.value)}
                    disabled={isSubmitting}
                  />
                  {url.errors.name && (
                    <p className="text-sm text-destructive">
                      {url.errors.name}
                    </p>
                  )}
                </div>
                <div className="space-y-1">
                  <Label htmlFor="url-prefix">
                    Endpoint prefix{" "}
                    <span className="text-muted-foreground text-sm font-normal">
                      (optional)
                    </span>
                  </Label>
                  <Input
                    id="url-prefix"
                    placeholder="my-remote-mcp"
                    value={url.prefix}
                    onChange={(e) => url.handlePrefixChange(e.target.value)}
                    disabled={isSubmitting}
                  />
                  {url.errors.prefix && (
                    <p className="text-sm text-destructive">
                      {url.errors.prefix}
                    </p>
                  )}
                </div>
                <div className="space-y-1">
                  <Label>Transport</Label>
                  <div className="flex rounded-md border overflow-hidden text-sm">
                    <button
                      type="button"
                      className={`flex-1 px-3 py-1.5 transition-colors ${url.transport === "streamable_http" ? "bg-primary text-primary-foreground" : "bg-background text-muted-foreground hover:bg-muted"}`}
                      onClick={() => url.setTransport("streamable_http")}
                      disabled={isSubmitting}
                    >
                      Streamable HTTP
                    </button>
                    <button
                      type="button"
                      className={`flex-1 px-3 py-1.5 transition-colors border-l ${url.transport === "sse" ? "bg-primary text-primary-foreground" : "bg-background text-muted-foreground hover:bg-muted"}`}
                      onClick={() => url.setTransport("sse")}
                      disabled={isSubmitting}
                    >
                      SSE
                    </button>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Auto-detected from URL — can be overridden
                  </p>
                </div>
                <div className="space-y-1">
                  <Label>
                    Request headers{" "}
                    <span className="text-muted-foreground text-sm font-normal">
                      (optional)
                    </span>
                  </Label>
                  <MapInput value={url.headers} onChange={url.setHeaders} />
                </div>
                <div className="space-y-1">
                  <Label htmlFor="url-repository-url">
                    Repository URL{" "}
                    <span className="text-muted-foreground text-sm font-normal">
                      (optional)
                    </span>
                  </Label>
                  <Input
                    id="url-repository-url"
                    placeholder="https://github.com/owner/repo"
                    value={url.repositoryUrl}
                    onChange={(e) => url.setRepositoryUrl(e.target.value)}
                    disabled={isSubmitting}
                  />
                </div>
                <div className="space-y-1">
                  <Label htmlFor="url-description">
                    Description{" "}
                    <span className="text-muted-foreground text-sm font-normal">
                      (optional)
                    </span>
                  </Label>
                  <Input
                    id="url-description"
                    placeholder="Short description of what this server does"
                    value={url.description}
                    onChange={(e) => url.setDescription(e.target.value)}
                    disabled={isSubmitting}
                  />
                </div>
              </>
            )}

            {/* ── Auto-Import fields ───────────────────────────────────────── */}
            {isAutoImportMode && (
              <div className="space-y-1">
                <Textarea
                  id="mcp-json"
                  placeholder={JSON_CONFIG_PLACEHOLDER}
                  value={stdio.jsonText}
                  onChange={(e) => stdio.handleJsonChange(e.target.value)}
                  disabled={isSubmitting}
                  rows={Math.min(
                    25,
                    Math.max(9, stdio.jsonText.split("\n").length),
                  )}
                  className="font-mono text-sm"
                />
                {stdio.jsonStatus && !stdio.jsonStatus.ok && (
                  <p className="text-sm text-destructive">
                    {stdio.jsonStatus.error}
                  </p>
                )}
                {stdio.jsonStatus?.ok && (
                  <p className="text-sm text-muted-foreground">
                    Detected:{" "}
                    <span className="font-medium">
                      {stdio.jsonStatus.parsedName}
                    </span>
                  </p>
                )}
                {!stdio.jsonStatus && (
                  <p className="text-xs text-muted-foreground">
                    Standard <code className="text-xs">mcpServers</code> format
                    — fields will be filled automatically.
                  </p>
                )}
              </div>
            )}

            {/* ── stdio (Command) fields ────────────────────────────────────── */}
            {isStdioMode && (
              <>
                <div className="space-y-1">
                  <Label htmlFor="mcp-command">Launch command</Label>
                  <Textarea
                    ref={stdio.commandRef}
                    id="mcp-command"
                    placeholder="npx -y @modelcontextprotocol/server-filesystem /data"
                    value={stdio.command}
                    onChange={(e) => {
                      stdio.handleCommandChange(e.target.value);
                      if (stdio.commandRef.current) {
                        stdio.commandRef.current.style.height = "auto";
                        stdio.commandRef.current.style.height = `${stdio.commandRef.current.scrollHeight}px`;
                      }
                    }}
                    disabled={isSubmitting}
                    rows={1}
                    className="font-mono text-sm resize-none overflow-hidden"
                  />
                  {stdio.errors.command && (
                    <p className="text-sm text-destructive">
                      {stdio.errors.command}
                    </p>
                  )}
                </div>
                {!stdio.baseImage && (stdio.variant || detectedRuntime) && (
                  <div className="space-y-1">
                    <Label htmlFor="mcp-runtime-version">
                      {(stdio.variant || detectedRuntime || "").startsWith(
                        "python",
                      ) || detectedRuntime === "python"
                        ? "Python version"
                        : "Node.js version"}
                    </Label>
                    <Select
                      value={
                        (stdio.variant || "").startsWith("python") ||
                        detectedRuntime === "python"
                          ? stdio.pythonVersion
                          : stdio.nodeVersion
                      }
                      onValueChange={(v) => {
                        if (
                          (stdio.variant || "").startsWith("python") ||
                          detectedRuntime === "python"
                        ) {
                          stdio.setPythonVersion(v as PythonVersion);
                        } else {
                          stdio.setNodeVersion(v as NodeVersion);
                        }
                      }}
                      disabled={isSubmitting}
                    >
                      <SelectTrigger id="mcp-runtime-version">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {((stdio.variant || "").startsWith("python") ||
                        detectedRuntime === "python"
                          ? PYTHON_VERSIONS
                          : NODE_VERSIONS
                        ).map((opt) => (
                          <SelectItem key={opt.value} value={opt.value}>
                            {opt.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                )}

                <div className="space-y-1">
                  <Label htmlFor="mcp-base-image">
                    Base image{" "}
                    <span className="text-muted-foreground text-sm font-normal">
                      (optional)
                    </span>
                  </Label>
                  <Input
                    id="mcp-base-image"
                    placeholder="mcp/filesystem"
                    value={stdio.baseImage ?? ""}
                    onChange={(e) =>
                      stdio.setBaseImage(e.target.value || undefined)
                    }
                    disabled={isSubmitting}
                    className="font-mono text-sm"
                  />
                  <p className="text-xs text-muted-foreground">
                    For <code className="text-xs">docker run</code>-style
                    configs — the runtime bridge layers on top of this image.
                    Leave empty for standard{" "}
                    <code className="text-xs">npx</code> /{" "}
                    <code className="text-xs">uvx</code> commands.
                  </p>
                </div>

                <div className="space-y-1">
                  <Label htmlFor="mcp-model-id">Model ID</Label>
                  <Input
                    id="mcp-model-id"
                    placeholder="my-mcp-server"
                    value={stdio.modelId}
                    onChange={(e) => stdio.handleModelIdChange(e.target.value)}
                    disabled={isSubmitting}
                  />
                  {stdio.errors.name && (
                    <p className="text-sm text-destructive">
                      {stdio.errors.name}
                    </p>
                  )}
                </div>

                <div className="space-y-1">
                  <Label htmlFor="mcp-prefix">
                    Endpoint prefix{" "}
                    <span className="text-muted-foreground text-sm font-normal">
                      (optional)
                    </span>
                  </Label>
                  <Input
                    id="mcp-prefix"
                    placeholder="my-server"
                    value={stdio.prefix}
                    onChange={(e) => stdio.handlePrefixChange(e.target.value)}
                    disabled={isSubmitting}
                  />
                  {stdio.errors.prefix ? (
                    <p className="text-sm text-destructive">
                      {stdio.errors.prefix}
                    </p>
                  ) : (
                    <p className="text-xs text-muted-foreground">
                      Letters, digits, hyphens and underscores only
                    </p>
                  )}
                </div>

                <div className="space-y-1">
                  <Label htmlFor="mcp-variant">{variantLabel}</Label>
                  <Select
                    value={stdio.variant || undefined}
                    onValueChange={stdio.handleVariantChange}
                    disabled={isSubmitting}
                  >
                    <SelectTrigger id="mcp-variant">
                      <SelectValue placeholder="None" />
                    </SelectTrigger>
                    <SelectContent>
                      {filteredVariants.map((v) => (
                        <SelectItem key={v.value} value={v.value}>
                          {detectedRuntime !== null
                            ? v.label.replace(/^.*?—\s*/, "")
                            : v.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  {variantHint && (
                    <p className="text-xs text-muted-foreground">
                      {variantHint}
                    </p>
                  )}
                </div>

                <div className="space-y-1">
                  <Label>
                    Environment variables{" "}
                    <span className="text-muted-foreground text-sm">
                      (optional)
                    </span>
                  </Label>
                  <MapInput value={stdio.envs} onChange={stdio.setEnvs} />
                </div>
                <div className="space-y-1">
                  <Label htmlFor="stdio-repository-url">
                    Repository URL{" "}
                    <span className="text-muted-foreground text-sm font-normal">
                      (optional)
                    </span>
                  </Label>
                  <Input
                    id="stdio-repository-url"
                    placeholder="https://github.com/owner/repo"
                    value={stdio.repositoryUrl}
                    onChange={(e) => stdio.setRepositoryUrl(e.target.value)}
                    disabled={isSubmitting}
                  />
                </div>
                <div className="space-y-1">
                  <Label htmlFor="stdio-description">
                    Description{" "}
                    <span className="text-muted-foreground text-sm font-normal">
                      (optional)
                    </span>
                  </Label>
                  <Input
                    id="stdio-description"
                    placeholder="Short description of what this server does"
                    value={stdio.description}
                    onChange={(e) => stdio.setDescription(e.target.value)}
                    disabled={isSubmitting}
                  />
                </div>
              </>
            )}

            {apiError && <p className="text-sm text-destructive">{apiError}</p>}
            {notice && (
              <p className="text-sm text-muted-foreground">{notice}</p>
            )}
          </form>

          <DialogFooter>
            <Button
              type="button"
              variant="outline"
              onClick={() => onOpenChange(false)}
              disabled={isSubmitting}
            >
              Cancel
            </Button>
            <Button
              type="submit"
              form="add-mcp-server-form"
              disabled={isSubmitting}
            >
              {isSubmitting
                ? "Saving…"
                : isEditMode
                  ? "Save"
                  : isAutoImportMode
                    ? "Convert"
                    : "Add"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <AlertDialog
        open={autoImportWarningOpen}
        onOpenChange={setAutoImportWarningOpen}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Switch to Auto-Import?</AlertDialogTitle>
            <AlertDialogDescription>
              Auto-Import does not update based on changes you've already made
              manually. If you convert a new config here, it will overwrite the
              fields you've filled in.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Stay here</AlertDialogCancel>
            <AlertDialogAction onClick={() => setServerMode("auto-import")}>
              Switch anyway
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}
