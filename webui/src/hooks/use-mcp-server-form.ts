/*
DeepFellow Software Framework.
Copyright © 2025 Simplito sp. z o.o.

This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
This software is Licensed under the DeepFellow Free License.

See the License for the specific language governing permissions and
limitations under the License.
*/

import { initFormData, validateFields } from "@/components/DynamicFormFields";
import type { SpecField } from "@/deepfellow/types";
import { proposePrefix } from "@/utils/prefix";
import type React from "react";
import { useEffect, useMemo, useRef, useState } from "react";

export type McpVariant =
  | "node-headless"
  | "node-headed"
  | "python-headless"
  | "python-headed";

export type PythonVersion =
  | "3.10"
  | "3.11"
  | "3.12"
  | "3.13"
  | "3.14"
  | "latest";
export type NodeVersion = "20" | "22" | "24" | "latest";

export const DEFAULT_PYTHON_VERSION: PythonVersion = "3.13";
export const DEFAULT_NODE_VERSION: NodeVersion = "22";

const PREFIX_PATTERN = /^[a-zA-Z0-9_-]+$/;

const PYTHON_BINS = new Set(["uvx", "python", "python3", "uv", "pipx"]);
const NODE_BINS = new Set([
  "node",
  "npx",
  "npm",
  "yarn",
  "pnpm",
  "bunx",
  "bun",
  "deno",
]);

export function detectRuntime(cmd: string): "python" | "node" | null {
  if (!cmd.trim()) return null;
  const base = (
    cmd.trim().split(/\s+/)[0].split("/").pop() ?? ""
  ).toLowerCase();
  if (PYTHON_BINS.has(base)) return "python";
  if (NODE_BINS.has(base)) return "node";
  return null;
}

export function detectVariant(cmd: string): McpVariant {
  const runtime = detectRuntime(cmd);
  return runtime === "python" ? "python-headless" : "node-headless";
}

function autoResizeTextarea(el: HTMLTextAreaElement) {
  el.style.height = "auto";
  el.style.height = `${el.scrollHeight}px`;
}

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

export type ParsedMcpConfig =
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

export interface DockerFormState {
  data: Record<string, unknown>;
  errors: Record<string, string>;
  volumes: string[];
  setVolumes: (v: string[]) => void;
  repositoryUrl: string;
  setRepositoryUrl: (v: string) => void;
  description: string;
  setDescription: (v: string) => void;
  handleChange: (name: string, value: unknown) => void;
  validate: () => Record<string, string>;
  populate: (parsed: {
    name: string;
    image: string;
    command: string;
    volumes: string[];
  }) => void;
}

export function useDockerForm(
  dockerFields: SpecField[],
  open: boolean,
): DockerFormState {
  const initial = useMemo(() => initFormData(dockerFields), [dockerFields]);
  const [data, setData] = useState<Record<string, unknown>>(initial);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [volumes, setVolumes] = useState<string[]>([]);
  const [repositoryUrl, setRepositoryUrl] = useState("");
  const [description, setDescription] = useState("");
  const [prefixIsManual, setPrefixIsManual] = useState(false);
  const hasPrefixField = useMemo(
    () => dockerFields.some((f) => f.name === "default_prefix"),
    [dockerFields],
  );

  // biome-ignore lint/correctness/useExhaustiveDependencies: open is the intentional reset trigger; dockerFields is stable
  useEffect(() => {
    if (!open) return;
    setData(initFormData(dockerFields));
    setErrors({});
    setVolumes([]);
    setRepositoryUrl("");
    setDescription("");
    setPrefixIsManual(false);
  }, [open]);

  const handleChange = (name: string, value: unknown) => {
    if (name === "default_prefix") {
      setPrefixIsManual(typeof value === "string" && value.trim() !== "");
    }
    setData((prev) => {
      const next = { ...prev, [name]: value };
      if (name === "id" && hasPrefixField && !prefixIsManual) {
        next.default_prefix = proposePrefix(
          typeof value === "string" ? value : "",
        );
      }
      return next;
    });
    setErrors((prev) => {
      if (!prev[name]) return prev;
      const next = { ...prev };
      delete next[name];
      return next;
    });
  };

  const validate = (): Record<string, string> => {
    const errs = validateFields(dockerFields, data);
    setErrors(errs);
    return errs;
  };

  const populate = (parsed: {
    name: string;
    image: string;
    command: string;
    volumes: string[];
  }) => {
    setData((prev) => ({
      ...prev,
      id: parsed.name,
      image: parsed.image,
      ...(parsed.command ? { command: parsed.command } : {}),
      ...(hasPrefixField && !prefixIsManual
        ? { default_prefix: proposePrefix(parsed.name) }
        : {}),
    }));
    if (parsed.volumes.length > 0) setVolumes(parsed.volumes);
    setErrors({});
  };

  return {
    data,
    errors,
    volumes,
    setVolumes,
    repositoryUrl,
    setRepositoryUrl,
    description,
    setDescription,
    handleChange,
    validate,
    populate,
  };
}

export interface UrlFormState {
  serverUrl: string;
  setServerUrl: (v: string) => void;
  name: string;
  setName: (v: string) => void;
  handleNameChange: (v: string) => void;
  prefix: string;
  setPrefix: (v: string) => void;
  handlePrefixChange: (v: string) => void;
  transport: "streamable_http" | "sse";
  setTransport: (v: "streamable_http" | "sse") => void;
  headers: Record<string, string>;
  setHeaders: (v: Record<string, string>) => void;
  repositoryUrl: string;
  setRepositoryUrl: (v: string) => void;
  description: string;
  setDescription: (v: string) => void;
  errors: Record<string, string | undefined>;
  populate: (parsed: {
    name: string;
    server_url: string;
    transport: "streamable_http" | "sse";
    headers: Record<string, string>;
  }) => void;
  validate: () => Record<string, string | undefined>;
}

export interface ProxyMcpServerSpec {
  kind: "proxy";
  id: string;
  name: string;
  server_url: string;
  transport: "streamable_http" | "sse";
  default_prefix?: string;
  headers?: Record<string, string>;
  repository_url?: string;
  description?: string;
}

export function useUrlForm(
  open: boolean,
  initialValues?: Partial<ProxyMcpServerSpec>,
): UrlFormState {
  const [serverUrl, setServerUrl] = useState(initialValues?.server_url ?? "");
  const [name, setName] = useState(initialValues?.name ?? "");
  const [prefix, setPrefix] = useState(initialValues?.default_prefix ?? "");
  const [prefixIsManual, setPrefixIsManual] = useState(
    !!initialValues?.default_prefix,
  );
  const [transport, setTransport] = useState<"streamable_http" | "sse">(
    initialValues?.transport ?? "streamable_http",
  );
  const [headers, setHeaders] = useState<Record<string, string>>(
    initialValues?.headers ?? {},
  );
  const [repositoryUrl, setRepositoryUrl] = useState(
    initialValues?.repository_url ?? "",
  );
  const [description, setDescription] = useState(
    initialValues?.description ?? "",
  );
  const [errors, setErrors] = useState<Record<string, string | undefined>>({});

  // biome-ignore lint/correctness/useExhaustiveDependencies: open is the reset trigger; initialValues is read at reset time via closure
  useEffect(() => {
    if (!open) return;
    setServerUrl(initialValues?.server_url ?? "");
    setName(initialValues?.name ?? "");
    setPrefix(initialValues?.default_prefix ?? "");
    setPrefixIsManual(!!initialValues?.default_prefix);
    setTransport(initialValues?.transport ?? "streamable_http");
    setHeaders(initialValues?.headers ?? {});
    setRepositoryUrl(initialValues?.repository_url ?? "");
    setDescription(initialValues?.description ?? "");
    setErrors({});
  }, [open]);

  const handleNameChange = (v: string) => {
    setName(v);
    if (!prefixIsManual) setPrefix(proposePrefix(v));
  };

  const handlePrefixChange = (v: string) => {
    setPrefix(v);
    setPrefixIsManual(v.trim() !== "");
  };

  const populate = (parsed: {
    name: string;
    server_url: string;
    transport: "streamable_http" | "sse";
    headers: Record<string, string>;
  }) => {
    setName(parsed.name);
    if (!prefixIsManual) setPrefix(proposePrefix(parsed.name));
    setServerUrl(parsed.server_url);
    setTransport(parsed.transport);
    setHeaders(parsed.headers);
    setErrors({});
  };

  const validate = (): Record<string, string | undefined> => {
    const errs: Record<string, string | undefined> = {};
    if (!serverUrl.trim()) errs.server_url = "Server URL is required.";
    else if (!/^https?:\/\/.+/.test(serverUrl.trim()))
      errs.server_url = "Must be a valid http:// or https:// URL.";
    if (!name.trim()) errs.name = "Name is required.";
    if (prefix.trim() && !PREFIX_PATTERN.test(prefix.trim()))
      errs.prefix =
        "Only letters, digits, hyphens and underscores are allowed.";
    setErrors(errs);
    return errs;
  };

  return {
    serverUrl,
    setServerUrl,
    name,
    setName,
    handleNameChange,
    prefix,
    setPrefix,
    handlePrefixChange,
    transport,
    setTransport,
    headers,
    setHeaders,
    repositoryUrl,
    setRepositoryUrl,
    description,
    setDescription,
    errors,
    populate,
    validate,
  };
}

type JsonStatus =
  | { ok: true; parsedName: string }
  | { ok: false; error: string }
  | null;

export interface StdioFormState {
  subMode: "import" | "manual";
  setSubMode: (v: "import" | "manual") => void;
  jsonText: string;
  jsonStatus: JsonStatus;
  command: string;
  baseImage: string | undefined;
  setBaseImage: (v: string | undefined) => void;
  pythonVersion: PythonVersion;
  setPythonVersion: (v: PythonVersion) => void;
  nodeVersion: NodeVersion;
  setNodeVersion: (v: NodeVersion) => void;
  modelId: string;
  setModelId: (v: string) => void;
  handleModelIdChange: (v: string) => void;
  prefix: string;
  setPrefix: (v: string) => void;
  handlePrefixChange: (v: string) => void;
  variant: McpVariant | "";
  variantIsManual: boolean;
  envs: Record<string, string>;
  setEnvs: (v: Record<string, string>) => void;
  repositoryUrl: string;
  setRepositoryUrl: (v: string) => void;
  description: string;
  setDescription: (v: string) => void;
  errors: Record<string, string | undefined>;
  commandRef: React.RefObject<HTMLTextAreaElement | null>;
  handleJsonChange: (text: string) => void;
  convertJson: () => boolean;
  handleCommandChange: (text: string) => void;
  handleVariantChange: (v: string) => void;
  clearErrors: () => void;
  buildPayload: () => {
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
  } | null;
}

export function useStdioForm(
  initialValues: Partial<AddMcpServerSpec> | undefined,
  open: boolean,
  /** Called when JSON paste is detected as a proxy config. Component switches to URL mode. */
  onProxyParsed: (parsed: {
    name: string;
    server_url: string;
    transport: "streamable_http" | "sse";
    headers: Record<string, string>;
  }) => void,
  /** Called when JSON paste is detected as a docker config. Component switches to Custom Image mode. */
  onDockerParsed: (parsed: {
    name: string;
    image: string;
    command: string;
    volumes: string[];
  }) => void,
  parseMcpJson: (text: string) => ParsedMcpConfig,
  /** Called when JSON paste is detected as a stdio config. Component switches to Command mode. */
  onStdioParsed?: () => void,
): StdioFormState {
  const [subMode, setSubMode] = useState<"import" | "manual">(
    initialValues ? "manual" : "import",
  );
  const [jsonText, setJsonText] = useState<string>("");
  const [jsonStatus, setJsonStatus] = useState<JsonStatus>(null);
  const [command, setCommand] = useState(initialValues?.command ?? "");
  const [baseImage, setBaseImage] = useState<string | undefined>(
    initialValues?.base_image,
  );
  const [pythonVersion, setPythonVersion] = useState<PythonVersion>(
    initialValues?.python_version ?? DEFAULT_PYTHON_VERSION,
  );
  const [nodeVersion, setNodeVersion] = useState<NodeVersion>(
    initialValues?.node_version ?? DEFAULT_NODE_VERSION,
  );
  const [modelId, setModelId] = useState(initialValues?.name ?? "");
  const [prefix, setPrefix] = useState(initialValues?.default_prefix ?? "");
  const [prefixIsManual, setPrefixIsManual] = useState(
    !!initialValues?.default_prefix,
  );
  const [variant, setVariant] = useState<McpVariant | "">(
    initialValues?.variant ?? "",
  );
  const [variantIsManual, setVariantIsManual] = useState(
    !!initialValues?.variant,
  );
  const [envs, setEnvs] = useState<Record<string, string>>(
    initialValues?.envs ?? {},
  );
  const [repositoryUrl, setRepositoryUrl] = useState(
    initialValues?.repository_url ?? "",
  );
  const [description, setDescription] = useState(
    initialValues?.description ?? "",
  );
  const [errors, setErrors] = useState<Record<string, string | undefined>>({});
  const commandRef = useRef<HTMLTextAreaElement>(null);

  // biome-ignore lint/correctness/useExhaustiveDependencies: open is the reset trigger; initialValues is read at reset time via closure
  useEffect(() => {
    if (!open) return;
    setSubMode(initialValues ? "manual" : "import");
    setJsonText("");
    setJsonStatus(null);
    setCommand(initialValues?.command ?? "");
    setBaseImage(initialValues?.base_image);
    setPythonVersion(initialValues?.python_version ?? DEFAULT_PYTHON_VERSION);
    setNodeVersion(initialValues?.node_version ?? DEFAULT_NODE_VERSION);
    setModelId(initialValues?.name ?? "");
    setPrefix(initialValues?.default_prefix ?? "");
    setPrefixIsManual(!!initialValues?.default_prefix);
    setVariant(initialValues?.variant ?? "");
    setVariantIsManual(!!initialValues?.variant);
    setEnvs(initialValues?.envs ?? {});
    setRepositoryUrl(initialValues?.repository_url ?? "");
    setDescription(initialValues?.description ?? "");
    setErrors({});
  }, [open]);

  // biome-ignore lint/correctness/useExhaustiveDependencies: command is the intentional trigger
  useEffect(() => {
    if (commandRef.current) autoResizeTextarea(commandRef.current);
  }, [command]);

  const applyAutoVariant = (cmd: string) => {
    if (variantIsManual) return;
    const runtime = detectRuntime(cmd);
    if (runtime === "python") setVariant("python-headless");
    else if (runtime === "node") setVariant("node-headless");
    else setVariant("");
  };

  const tryApplyJson = (text: string): boolean => {
    if (!text.trimStart().startsWith("{")) return false;
    try {
      const parsed = parseMcpJson(text);
      if (parsed.kind === "proxy") {
        onProxyParsed(parsed);
      } else if (parsed.kind === "docker") {
        onDockerParsed(parsed);
      } else {
        setModelId(parsed.name);
        if (!prefixIsManual) setPrefix(proposePrefix(parsed.name));
        setCommand(parsed.command);
        setBaseImage(parsed.base_image);
        setEnvs(parsed.envs);
        if (!variantIsManual) setVariant(parsed.variant ?? "");
        setErrors({});
        onStdioParsed?.();
      }
      setJsonStatus(null);
      return true;
    } catch {
      return false;
    }
  };

  const handleJsonChange = (text: string) => {
    setJsonText(text);
    if (!text.trim()) {
      setJsonStatus(null);
      return;
    }
    if (!text.trimStart().startsWith("{")) {
      setJsonStatus({
        ok: false,
        error: "Invalid JSON — expected mcpServers config.",
      });
      return;
    }
    try {
      const parsed = parseMcpJson(text);
      setJsonStatus({ ok: true, parsedName: parsed.name });
    } catch {
      setJsonStatus({
        ok: false,
        error: "Invalid JSON — expected mcpServers config.",
      });
    }
  };

  const convertJson = (): boolean => {
    if (!tryApplyJson(jsonText)) return false;
    setSubMode("manual");
    return true;
  };

  const handleCommandChange = (text: string) => {
    if (tryApplyJson(text)) return;
    setCommand(text);
    applyAutoVariant(text);
    setErrors((prev) => {
      if (!prev.command) return prev;
      const next = { ...prev };
      next.command = undefined;
      return next;
    });
  };

  const handleVariantChange = (v: string) => {
    setVariant(v as McpVariant);
    setVariantIsManual(true);
  };

  const handleModelIdChange = (v: string) => {
    setModelId(v);
    if (!prefixIsManual) setPrefix(proposePrefix(v));
    clearErrors();
  };

  const handlePrefixChange = (v: string) => {
    setPrefix(v);
    setPrefixIsManual(v.trim() !== "");
    clearErrors();
  };

  const clearErrors = () => setErrors({});

  const buildPayload = () => {
    if (subMode === "import") setSubMode("manual");
    const errs: Record<string, string | undefined> = {};
    if (!modelId.trim()) errs.name = "Model ID is required.";
    if (!command.trim()) errs.command = "Command is required.";
    if (prefix.trim() && !PREFIX_PATTERN.test(prefix.trim()))
      errs.prefix =
        "Only letters, digits, hyphens and underscores are allowed.";
    if (Object.values(errs).some(Boolean)) {
      setErrors(errs);
      return null;
    }
    setErrors({});
    const resolvedVariant: McpVariant =
      variant ||
      detectVariant(command.trim().split(/\s+/)[0]) ||
      "node-headless";
    const isPython = resolvedVariant.startsWith("python");
    return {
      kind: "user" as const,
      id: modelId.trim(),
      name: modelId.trim(),
      variant: resolvedVariant,
      command: command.trim(),
      base_image: baseImage || undefined,
      python_version: isPython ? pythonVersion : undefined,
      node_version: !isPython ? nodeVersion : undefined,
      envs: Object.keys(envs).length > 0 ? envs : undefined,
      default_prefix: prefix.trim() || undefined,
      repository_url: repositoryUrl.trim() || undefined,
      description: description.trim() || undefined,
    };
  };

  return {
    subMode,
    setSubMode,
    jsonText,
    jsonStatus,
    command,
    baseImage,
    setBaseImage,
    pythonVersion,
    setPythonVersion,
    nodeVersion,
    setNodeVersion,
    modelId,
    setModelId,
    handleModelIdChange,
    prefix,
    setPrefix,
    handlePrefixChange,
    variant,
    variantIsManual,
    envs,
    setEnvs,
    repositoryUrl,
    setRepositoryUrl,
    description,
    setDescription,
    errors,
    commandRef,
    handleJsonChange,
    convertJson,
    handleCommandChange,
    handleVariantChange,
    clearErrors,
    buildPayload,
  };
}
